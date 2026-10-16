#!/usr/bin/env python3
"""Record AI-assisted project changes into an Obsidian audit trail.

Typical usage:
    python ai_change_logger.py start --agent data-agent --task "Update daily data"
    python ai_change_logger.py finish --session-id <id> --summary "Done" --tests "pytest -q"
    python ai_change_logger.py run --agent factor-agent --task "Add factor" -- command args...
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import textwrap
import uuid
from typing import Any, Iterable

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
KB_ROOT = PROJECT_ROOT / "my_strategy" / "knowledge_base"
SYSTEM_ROOT = PROJECT_ROOT / "my_strategy" / "knowledge_base_system"
SESSION_JSON_ROOT = SYSTEM_ROOT / "sessions"
PROMPT_ROOT = SYSTEM_ROOT / "prompts"
CHANGE_ROOT = KB_ROOT / "04_AI_Changes"
SESSION_NOTE_ROOT = KB_ROOT / "05_AI_Sessions"
DIFF_ROOT = SYSTEM_ROOT / "diffs"
CHANGELOG = PROJECT_ROOT / "AI_CHANGELOG.md"

REDACTIONS = [
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(token\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(password\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(cookie\s*[=:]\s*)\S+"),
]
MAX_DIFF_CHARS = 20_000
MAX_OUTPUT_CHARS = 50_000
SOURCE_DIRS = ('my_strategy/adapters','my_strategy/core','my_strategy/data_manager',
    'my_strategy/execution','my_strategy/scripts','my_strategy/services','my_strategy/storage',
    'my_strategy/validation','my_strategy/web_dashboard','my_strategy/tests','my_strategy/configs',
    'my_strategy/docs','my_strategy/skills','my_strategy/agent_prompts','vendor','docker','macos','parameter_profiles')
SOURCE_SUFFIXES = {'.py','.pyi','.rs','.c','.h','.cc','.cpp','.hpp','.js','.ts','.tsx','.jsx','.css','.html','.md','.rst','.toml',
    '.yaml','.yml','.json','.ini','.cfg','.sh','.bat','.ps1','.command','.hcl','.lock','.example'}
SOURCE_DENY_DIRS = {'data','artifacts','knowledge_base','knowledge_base_system','kronos_model',
    'logs','outputs','node_modules','__pycache__','.git','.venv','venv','.scratch','.release-work'}
SOURCE_MAX_BYTES = 4 * 1024 * 1024


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def iso_now() -> str:
    return now_local().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in [SESSION_JSON_ROOT, PROMPT_ROOT, CHANGE_ROOT, SESSION_NOTE_ROOT, DIFF_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def redact(text: str) -> str:
    # Process quoted values before the legacy unquoted patterns so a spaced
    # secret cannot leave its trailing words in snapshots or diffs.
    result = re.sub(r"(?i)((?:[\"']?(?:api[_-]?key|token|password|cookie|secret|authorization)[\"']?)\s*[:=]\s*)(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;]+)",
                    lambda match: match.group(1) + '<REDACTED>', text)
    for pattern in REDACTIONS:
        result = pattern.sub(r"\1<REDACTED>", result)
    return result


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n\n... <TRUNCATED {omitted} CHARS> ...\n"


def write_diff_artifact(sid: str, diff_text: str) -> str | None:
    """Store a large redacted diff as a compressed, hash-addressed sidecar."""
    if len(diff_text) <= MAX_DIFF_CHARS:
        return None
    path = DIFF_ROOT / f"{sid}.patch.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(diff_text)
    return path.relative_to(PROJECT_ROOT).as_posix()


def safe_source_path(path: Path) -> bool:
    """Reject linked ancestors as well as leaves, including broken links."""
    try:
        relative = path.relative_to(PROJECT_ROOT)
        cursor = PROJECT_ROOT
        if cursor.is_symlink(): return False
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink(): return False
        return path.resolve().is_relative_to(PROJECT_ROOT.resolve())
    except (OSError, ValueError, RuntimeError):
        return False


def stable_source_bytes(path: Path) -> bytes:
    if not safe_source_path(path): raise ValueError('linked or external source path')
    fd = os.open(path, os.O_RDONLY | getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > SOURCE_MAX_BYTES: raise ValueError('source exceeds snapshot size bound')
        content = handle.read(SOURCE_MAX_BYTES + 1)
        after = os.fstat(handle.fileno())
    current = path.stat(follow_symlinks=False)
    identity = lambda value: (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    if not safe_source_path(path) or len(content) > SOURCE_MAX_BYTES or identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError('source changed during snapshot read')
    return content


def source_snapshot() -> dict[str, Any]:
    """Read only source/templates/docs, never business trees or linked targets."""
    if not safe_source_path(PROJECT_ROOT):
        return {'schema':'source_audit_v1','scope':'allowlisted source, configuration templates and documentation',
                'complete':False,'errors':[{'path':'','error':'linked project root'}],'files':{}}
    root_names = {'Dockerfile','.gitignore','.dockerignore','.env.example','pyproject.toml',
                  'docker-compose.yml','docker-bake.hcl','czsc-source-manifest.json'}
    candidates = {path for path in PROJECT_ROOT.iterdir() if path.is_file() and path.name != 'AI_CHANGELOG.md'
                  and (path.suffix in {'.py','.md','.sh','.bat','.ps1','.command'}
                       or path.name in root_names or path.name.startswith('requirements'))}
    package = PROJECT_ROOT / 'my_strategy'
    if package.is_dir() and safe_source_path(package):
        candidates.update(path for path in package.iterdir() if path.is_file() and path.suffix in {'.py','.md'})
    scan_errors = []
    def unreadable_directory(exc):
        location = Path(exc.filename).relative_to(PROJECT_ROOT).as_posix() if exc.filename else '<unknown>'
        scan_errors.append({'path':location,'error':str(exc)})
    for relative in SOURCE_DIRS:
        base = PROJECT_ROOT / relative
        if not safe_source_path(base):
            scan_errors.append({'path':relative,'error':'linked or external source directory'}); continue
        if not base.is_dir(): continue
        for directory, dirs, files in os.walk(base, followlinks=False, onerror=unreadable_directory):
            if not safe_source_path(Path(directory)):
                scan_errors.append({'path':Path(directory).relative_to(PROJECT_ROOT).as_posix(),'error':'source directory changed to link'});dirs[:]=[];continue
            dirs[:] = [name for name in dirs if name not in SOURCE_DENY_DIRS
                       and not (Path(directory) / name).is_symlink()]
            candidates.update(Path(directory) / name for name in files)
    files, errors = {}, scan_errors
    for path in sorted(candidates):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if not safe_source_path(path) or any(part in SOURCE_DENY_DIRS for part in path.relative_to(PROJECT_ROOT).parts): continue
        if path.name.startswith('.env') and path.name != '.env.example': continue
        if path.name.lower() in {'credentials.json','secrets.json','credentials.yaml','secrets.yaml'}: continue
        if path.suffix not in SOURCE_SUFFIXES and path.name not in {'Dockerfile','.gitignore','.dockerignore'} \
                and not (path.name.startswith('requirements') and path.suffix == '.txt'): continue
        try:
            content = stable_source_bytes(path)
            text = redact(content.decode('utf-8'))
            files[relative] = {'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content),
                'redacted_sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text}
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append({'path': relative, 'error': str(exc)})
    return {'schema':'source_audit_v1','scope':'allowlisted source, configuration templates and documentation',
            'complete':not errors,'errors':errors,'files':files}


def save_source_baseline(sid: str) -> dict[str, Any]:
    snapshot = source_snapshot()
    encoded = gzip.compress(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode(), mtime=0)
    path = DIFF_ROOT / (sid + '.source-before.json.gz')
    path.write_bytes(encoded)
    manifest = {**snapshot, 'files': {key:{k:v for k,v in value.items() if k != 'text'}
                                    for key,value in snapshot['files'].items()}}
    manifest_path = DIFF_ROOT / (sid + '.source-before.manifest.json')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'method':'no_git_source_snapshot','status':'available' if snapshot['complete'] else 'partial',
        'snapshot_path':path.relative_to(PROJECT_ROOT).as_posix(),
        'snapshot_sha256':hashlib.sha256(encoded).hexdigest(),
        'manifest_path':manifest_path.relative_to(PROJECT_ROOT).as_posix(), 'errors':snapshot['errors']}


def source_diff(data: dict[str, Any]) -> tuple[dict[str, Any], list[str], str, str]:
    baseline = data.get('source_baseline') or {}
    try:
        if not baseline.get('snapshot_path'): raise ValueError('session started without a source snapshot')
        path = PROJECT_ROOT / baseline['snapshot_path']
        if path.is_symlink() or not path.resolve().is_relative_to(DIFF_ROOT.resolve()):
            raise ValueError('snapshot path outside audit diff storage')
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != baseline.get('snapshot_sha256'):
            raise ValueError('source snapshot SHA mismatch')
        before = json.loads(gzip.decompress(encoded))
        if before.get('schema') != 'source_audit_v1': raise ValueError('unsupported source snapshot schema')
        after = source_snapshot()
        changes, patches, unknown = [], [], []
        read_errors = before.get('errors',[]) + after['errors']
        for relative in sorted(before['files'].keys() | after['files'].keys()):
            old, new = before['files'].get(relative), after['files'].get(relative)
            if not safe_source_path(PROJECT_ROOT/relative) or any(not error['path'] or relative == error['path'] or relative.startswith(error['path'] + '/') for error in read_errors):
                unknown.append(relative); continue
            if old and new is None and ((PROJECT_ROOT/relative).is_symlink() or (PROJECT_ROOT/relative).exists()):
                unknown.append(relative); continue
            if old and new and old['sha256'] == new['sha256']: continue
            kind = 'added' if old is None else 'deleted' if new is None else 'modified'
            changes.append({'path':relative,'change':kind,'before_sha256':old['sha256'] if old else None,
                            'after_sha256':new['sha256'] if new else None})
            lines = lambda value: [line+'\n' for line in (value or {}).get('text','').splitlines()]
            patch = ''.join(difflib.unified_diff(lines(old), lines(new), fromfile='a/'+relative if old else '/dev/null',
                tofile='b/'+relative if new else '/dev/null'))
            patches.append(patch or '# ' + relative + ': content SHA changed; redacted text is unchanged\n')
        complete = bool(before.get('complete') and after['complete'] and not unknown)
        capture = {'method':'no_git_source_snapshot','status':'available' if complete else 'partial',
            'changes':changes,'errors':read_errors,'unverifiable_paths':unknown,
            'before_manifest_path':baseline.get('manifest_path')}
        stat = '\n'.join(item['change']+' '+item['path'] for item in changes) or ('No verified changes; capture incomplete.' if not complete else 'No source changes within the declared whitelist.')
        if not complete: stat = 'PARTIAL source capture; unknown files may have changed.\n'+stat
        return capture, [item['path'] for item in changes], '\n'.join(patches), stat
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        reason = str(exc)
        return {'method':'no_git_source_snapshot','status':'unavailable','reason':reason}, [], '', 'UNAVAILABLE: '+reason


def external_diff_evidence(sid: str, source: Path) -> dict[str, str]:
    """Attach explicitly supplied evidence; never pretend it was a start snapshot."""
    source = Path(source)
    if source.is_symlink() or not source.is_file(): raise ValueError('external diff must be a regular file')
    if source.stat().st_size > 50 * 1024 * 1024: raise ValueError('external diff exceeds size bound')
    original = source.read_bytes()
    value = redact(original.decode('utf-8'))
    path = DIFF_ROOT / (sid + '.external.patch.gz')
    path.write_bytes(gzip.compress(value.encode(),mtime=0))
    return {'source_path':str(source.resolve()),'source_sha256':hashlib.sha256(original).hexdigest(),
        'redacted_diff_sha256':hashlib.sha256(value.encode()).hexdigest(),
        'artifact_path':path.relative_to(PROJECT_ROOT).as_posix(), 'kind':'supplemental_external_evidence'}


def run_process(
    args: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=check,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(args=args, returncode=127, stdout="")


def git_output(*args: str) -> str:
    result = run_process(["git", *args])
    if result.returncode != 0:
        return ""
    # Keep the first status column: leading whitespace is meaningful in
    # porcelain output (for example " M AGENTS.md").
    return result.stdout.rstrip()


def git_available() -> bool:
    return bool(git_output("rev-parse", "--show-toplevel"))


def git_head() -> str:
    return git_output("rev-parse", "HEAD") or "NO_GIT"


def git_branch() -> str:
    return git_output("branch", "--show-current") or "NO_GIT"


def git_status() -> str:
    return git_output("status", "--short")


def git_diff() -> str:
    unstaged = git_output("diff", "--no-ext-diff")
    staged = git_output("diff", "--cached", "--no-ext-diff")
    parts = []
    if unstaged:
        parts.append("# Unstaged\n" + unstaged)
    if staged:
        parts.append("# Staged\n" + staged)
    return "\n\n".join(parts)


def git_diff_stat() -> str:
    unstaged = git_output("diff", "--stat")
    staged = git_output("diff", "--cached", "--stat")
    parts = []
    if unstaged:
        parts.append("Unstaged:\n" + unstaged)
    if staged:
        parts.append("Staged:\n" + staged)
    return "\n\n".join(parts)


def parse_status_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw.strip('"'))
    return sorted(set(paths))


def task_slug(task: str, limit: int = 48) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", task, flags=re.UNICODE)
    value = re.sub(r"-+", "-", value).strip("-_")
    return (value or "ai-task")[:limit]


def session_id() -> str:
    stamp = now_local().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def session_json_path(sid: str) -> Path:
    return SESSION_JSON_ROOT / f"{sid}.json"


def load_session(sid: str) -> dict[str, Any]:
    path = session_json_path(sid)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {sid}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(data: dict[str, Any]) -> None:
    ensure_dirs()
    path = session_json_path(str(data["session_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_prompt_snapshot(sid: str, task: str, prompt: str | None) -> str | None:
    if not prompt:
        return None
    path = PROMPT_ROOT / f"{sid}.md"
    path.write_text(
        "\n".join(
            [
                "---",
                f"session_id: {yaml_scalar(sid)}",
                f"task: {yaml_scalar(task)}",
                f"created_at: {yaml_scalar(iso_now())}",
                "type: ai-prompt",
                "---",
                "",
                "# AI Prompt",
                "",
                redact(prompt),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def start_session(
    *,
    agent: str,
    task: str,
    prompt: str | None = None,
    target_paths: Iterable[str] = (),
    command: list[str] | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    sid = session_id()
    has_git = git_available()
    baseline_status = git_status() if has_git else ""
    prompt_path = write_prompt_snapshot(sid, task, prompt)
    data: dict[str, Any] = {
        "session_id": sid,
        "task": task,
        "agent": agent,
        "status": "running",
        "started_at": iso_now(),
        "ended_at": None,
        "project_root": str(PROJECT_ROOT),
        "git_available": has_git,
        "git_branch": git_branch(),
        "baseline_commit": git_head(),
        "baseline_status": baseline_status,
        "target_paths": list(target_paths),
        "prompt_path": prompt_path,
        "command": command or [],
        "command_exit_code": None,
        "command_output": "",
        "summary": "",
        "tests": [],
        "risks": [],
        "changed_files": [],
        "final_commit": None,
        "diff_stat": "",
        "diff_sha256": "",
        "note_path": None,
    }
    if not has_git:
        data["source_baseline"] = save_source_baseline(sid)
    save_session(data)
    return data


def markdown_list(items: Iterable[str], empty: str = "- 无") -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {value}" for value in values) if values else empty


def split_multi(values: list[str] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values:
        for part in re.split(r"\s*\|\s*|\s*;\s*", value):
            if part.strip():
                result.append(part.strip())
    return result


def create_change_note(data: dict[str, Any], diff_text: str) -> Path:
    ended = dt.datetime.fromisoformat(str(data["ended_at"]))
    year = ended.strftime("%Y")
    month = ended.strftime("%m")
    target_dir = CHANGE_ROOT / year / month
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = task_slug(str(data["task"]))
    note_path = target_dir / f"{data['session_id']}-{slug}.md"

    changed_files = data.get("changed_files", [])
    links = []
    for file_path in changed_files:
        clean = str(file_path).replace("\\", "/")
        links.append(f"- `[[../../../../{clean}|{clean}]]`")

    diff_text = "\n".join(
        line.rstrip()
        for line in truncate(redact(diff_text), MAX_DIFF_CHARS).splitlines()
    )
    command_output = truncate(
        redact(str(data.get("command_output", ""))),
        MAX_OUTPUT_CHARS,
    )
    command_display = " ".join(
        shlex.quote(str(part)) for part in data.get("command", [])
    )

    content = f"""---
type: ai-change
session_id: {yaml_scalar(str(data['session_id']))}
task: {yaml_scalar(str(data['task']))}
agent: {yaml_scalar(str(data['agent']))}
status: {yaml_scalar(str(data['status']))}
started_at: {yaml_scalar(str(data['started_at']))}
ended_at: {yaml_scalar(str(data['ended_at']))}
baseline_commit: {yaml_scalar(str(data['baseline_commit']))}
final_commit: {yaml_scalar(str(data.get('final_commit') or 'UNCOMMITTED'))}
tags:
  - khquant
  - ai-change
  - agent/{data['agent']}
---

# {data['task']}

## 摘要

{data.get('summary') or '未填写'}

## Agent

- 主责：[[../../02_Agents/Agent_Router|{data['agent']}]]
- 会话：`{data['session_id']}`
- 状态：`{data['status']}`
- 分支：`{data.get('git_branch', '')}`
- 基线提交：`{data.get('baseline_commit', '')}`
- 最终提交：`{data.get('final_commit') or 'UNCOMMITTED'}`

## 修改文件

{chr(10).join(links) if links else '- 变更采集不可用或不完整；无法判断变化数量' if (data.get('change_capture') or {}).get('status') in {'unavailable','partial'} else '- 未检测到白名单范围内的文件变化'}

## 测试与验证

{markdown_list(data.get('tests', []))}

## 风险与限制

{markdown_list(data.get('risks', []))}

## 执行命令

```text
{command_display or '手工/外部调用'}
```

退出码：`{data.get('command_exit_code')}`

## Diff Stat

```text
{data.get('diff_stat') or '无'}
```

## Diff Evidence

- 采集：`{(data.get('change_capture') or {}).get('method', 'git')}` / `{(data.get('change_capture') or {}).get('status', 'available')}`
- 外部补充证据：`{(data.get('external_diff_evidence') or {}).get('artifact_path', '无')}`
- SHA-256: `{data.get('diff_sha256') or '无'}`
- 完整压缩 diff：`{data.get('diff_artifact_path') or '未生成（diff 未超过内联上限）'}`

## 命令输出

```text
{command_output or '无'}
```

## Captured Diff

```diff
{diff_text or '# No diff captured'}
```

## 关联页面

- [[../../00_Home/KHQuant_Home]]
- [[../../01_Project/Architecture]]
- [[../../03_Workflows/AI_Change_Workflow]]
- [[../../09_Decisions/Architecture_Decision_Log]]
"""
    note_path.write_text(content, encoding="utf-8")
    return note_path


def append_changelog(data: dict[str, Any], note_path: Path) -> None:
    if not CHANGELOG.exists():
        CHANGELOG.write_text(
            "# KHQuant AI Change Log\n\n"
            "> 自动记录由 AI、Agent 或包装命令产生的项目修改。\n\n",
            encoding="utf-8",
        )
    rel_note = note_path.relative_to(PROJECT_ROOT).as_posix()
    sid = str(data["session_id"])
    line = (
        f"- {data['ended_at']} | `{data['agent']}` | session `{sid}` | "
        f"`{data['status']}` | [{data['task']}]({rel_note}) | "
        f"{'unknown' if (data.get('change_capture') or {}).get('status') in {'unavailable','partial'} else len(data.get('changed_files', []))} files | "
        f"{data.get('final_commit') or 'UNCOMMITTED'}\n"
    )
    existing = [
        item for item in CHANGELOG.read_text(encoding="utf-8").splitlines()
        if sid not in item
    ]
    with CHANGELOG.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(existing).rstrip() + "\n" + line)


def rebuild_obsidian_indexes() -> bool:
    script = SCRIPT_PATH.with_name("rebuild_obsidian_indexes.py")
    if not script.is_file():
        return False
    return run_process([sys.executable, str(script)]).returncode == 0


def create_session_note(data: dict[str, Any]) -> Path:
    ended = dt.datetime.fromisoformat(str(data["ended_at"]))
    target_dir = SESSION_NOTE_ROOT / ended.strftime("%Y") / ended.strftime("%m")
    target_dir.mkdir(parents=True, exist_ok=True)
    note_path = target_dir / f"{data['session_id']}.md"
    change_note = data.get("note_path", "")
    content = f"""---
type: ai-session
session_id: {yaml_scalar(str(data['session_id']))}
agent: {yaml_scalar(str(data['agent']))}
status: {yaml_scalar(str(data['status']))}
started_at: {yaml_scalar(str(data['started_at']))}
ended_at: {yaml_scalar(str(data['ended_at']))}
tags: [khquant, ai-session]
---

# AI Session {data['session_id']}

- 任务：{data['task']}
- Agent：{data['agent']}
- 状态：{data['status']}
- 变更记录：`{change_note}`
- Prompt 快照：`{data.get('prompt_path') or '无'}`
- 命令退出码：`{data.get('command_exit_code')}`
"""
    note_path.write_text(content, encoding="utf-8")
    return note_path


def finish_session(
    *,
    sid: str,
    summary: str = "",
    tests: Iterable[str] = (),
    risks: Iterable[str] = (),
    exit_code: int | None = None,
    command_output: str | None = None,
    explicit_status: str | None = None,
    external_diff: Path | None = None,
) -> dict[str, Any]:
    data = load_session(sid)
    current_status = git_status() if data.get("git_available") else ""
    diff_text = git_diff() if data.get("git_available") else ""
    changed_files = parse_status_paths(current_status)
    final_commit = git_head() if data.get("git_available") else "NO_GIT"
    if data.get('git_available'):
        capture = {'method':'git','status':'available' if git_available() else 'unavailable'}
        stat = git_diff_stat() if capture['status'] == 'available' else 'UNAVAILABLE: original Git repository is unavailable'
    else:
        capture, changed_files, diff_text, stat = source_diff(data)
    data['change_capture'] = capture
    if external_diff is not None:
        data['external_diff_evidence'] = external_diff_evidence(sid, external_diff)

    if explicit_status:
        status = explicit_status
    elif exit_code is None or exit_code == 0:
        status = "success"
    else:
        status = "failed"

    redacted_diff = redact(diff_text)
    data.update(
        {
            "status": status,
            "ended_at": iso_now(),
            "summary": summary,
            "tests": list(tests),
            "risks": list(risks),
            "command_exit_code": exit_code,
            "command_output": command_output or data.get("command_output", ""),
            "changed_files": changed_files,
            "final_commit": final_commit,
            "diff_stat": stat,
            "diff_sha256": hashlib.sha256(redacted_diff.encode("utf-8", errors="replace")).hexdigest() if capture["status"] != "unavailable" else "",
            "diff_artifact_path": write_diff_artifact(sid, redacted_diff),
        }
    )

    note_path = create_change_note(data, diff_text)
    data["note_path"] = str(note_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    session_note = create_session_note(data)
    data["session_note_path"] = str(
        session_note.relative_to(PROJECT_ROOT)
    ).replace("\\", "/")
    append_changelog(data, note_path)
    data["indexes_rebuilt"] = rebuild_obsidian_indexes()
    save_session(data)
    return data


def command_start(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    data = start_session(
        agent=args.agent,
        task=args.task,
        prompt=prompt,
        target_paths=args.path or [],
    )
    print(data["session_id"])
    return 0


def command_finish(args: argparse.Namespace) -> int:
    data = finish_session(
        sid=args.session_id,
        summary=args.summary or "",
        tests=split_multi(args.tests),
        risks=split_multi(args.risks),
        exit_code=args.exit_code,
        explicit_status=args.status,
        external_diff=Path(args.external_diff) if args.external_diff else None,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("No wrapped command supplied.", file=sys.stderr)
        return 2

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    data = start_session(
        agent=args.agent,
        task=args.task,
        prompt=prompt,
        target_paths=args.path or [],
        command=command,
    )
    sid = str(data["session_id"])
    print(f"[KHQuant] session={sid}", flush=True)

    result = run_process(command, cwd=PROJECT_ROOT)
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")

    summary = args.summary or (
        "包装命令执行完成。" if result.returncode == 0 else "包装命令执行失败。"
    )
    final = finish_session(
        sid=sid,
        summary=summary,
        tests=split_multi(args.tests),
        risks=split_multi(args.risks),
        exit_code=result.returncode,
        command_output=output,
    )
    print(f"[KHQuant] change_note={final['note_path']}")
    return int(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    start = sub.add_parser("start", help="Start an AI modification session.")
    start.add_argument("--agent", required=True)
    start.add_argument("--task", required=True)
    start.add_argument("--prompt")
    start.add_argument("--prompt-file")
    start.add_argument("--path", action="append")
    start.set_defaults(func=command_start)

    finish = sub.add_parser("finish", help="Finish and record a session.")
    finish.add_argument("--session-id", required=True)
    finish.add_argument("--summary")
    finish.add_argument("--tests", action="append")
    finish.add_argument("--risks", action="append")
    finish.add_argument("--exit-code", type=int)
    finish.add_argument("--external-diff", help="Supplemental redacted diff evidence; does not reconstruct a missing baseline.")
    finish.add_argument(
        "--status",
        choices=["success", "partial", "failed", "blocked"],
    )
    finish.set_defaults(func=command_finish)

    run = sub.add_parser("run", help="Wrap a command and record its changes.")
    run.add_argument("--agent", required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--prompt")
    run.add_argument("--prompt-file")
    run.add_argument("--path", action="append")
    run.add_argument("--summary")
    run.add_argument("--tests", action="append")
    run.add_argument("--risks", action="append")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=command_run)

    return parser


def main() -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
