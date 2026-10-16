"""Inventory and explicitly retire obsolete release test/install scratch trees.

Only rebuildable scratch below .release-work is eligible. Release assets,
source checkouts, Git histories, the newest two releases and business paths
are never selected. A saved SHA manifest and a second live-use/tree check
are mandatory for execution. This tool does not manage native/cache objects.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import uuid

SCHEMA = 'release_scratch_retention_v1'
TEMP_NAMES = {'install-env', 'pytest-full', 'pytest-source', 'pytest-final-source',
              'local-test-state', 'source-final-test-state', 'source-native-test-state',
              'staged-current-platform-test-state'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def release_versions(root):
    result = []
    for path in root.iterdir():
        match = re.fullmatch(r'github-(\d+)\.(\d+)\.(\d+)', path.name)
        if match and path.is_dir() and not path.is_symlink():
            result.append((tuple(map(int, match.groups())), path.name))
    return [name for _, name in sorted(result, reverse=True)]


def validate_target(root, relative):
    root = Path(root).resolve(strict=True)
    if root.name != '.release-work':
        raise ValueError('Only .release-work scratch is eligible')
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(x in {'.', '..', '.git'} for x in parts):
        raise ValueError('Unsafe scratch path')
    recent = release_versions(root)[:2]
    allowed = (len(parts) >= 2 and re.fullmatch(r'github-\d+\.\d+\.\d+', parts[0])
               and parts[0] not in recent and parts[1] in TEMP_NAMES)
    # Standalone old pytest base directories identify their exact version.
    match = re.fullmatch(r'v(\d)(\d)(\d)-pytest-full', parts[0])
    if match:
        version = tuple(map(int, match.groups()))
        versions = [tuple(map(int, name[7:].split('.'))) for name in recent]
        allowed = bool(versions and version < min(versions))
    if not allowed:
        raise ValueError('Protected release/source path: ' + relative)
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Scratch target/parent is a symlink: ' + relative)
    if not path.is_dir() or not path.resolve().is_relative_to(root):
        raise ValueError('Scratch directory is missing or escapes root')
    return path


def inventory_tree(path):
    """Never follow directory symlinks; all regular bytes are SHA verified."""
    path = Path(path)
    result = []
    for directory, dirs, files in os.walk(path, followlinks=False):
        if '.git' in dirs or '.git' in files:
            raise ValueError('Git history protected: ' + str(directory))
        # A bare repository has no .git child. Preserve it (including an
        # incomplete repository) without asking Git to repair or fetch it.
        if 'HEAD' in files and 'objects' in dirs and ('refs' in dirs or 'packed-refs' in files):
            raise ValueError('Bare Git history protected: ' + str(directory))
        for name in sorted(dirs + files):
            child = Path(directory) / name
            info = child.lstat()
            record = dict(path=child.relative_to(path).as_posix(), device=info.st_dev,
                          inode=info.st_ino, size=info.st_size, blocks=getattr(info, 'st_blocks', None),
                          links=info.st_nlink, mode=info.st_mode,
                          mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
            if stat.S_ISREG(info.st_mode):
                with child.open('rb') as stream:
                    record['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
                after = child.stat()
                fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink')
                if any(getattr(after, name) != getattr(info, name) for name in fields):
                    raise RuntimeError('Scratch file changed during inventory')
            elif stat.S_ISLNK(info.st_mode):
                record['symlink'] = os.readlink(child)
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError('Unsupported scratch file type')
            result.append(record)
    root_info = path.stat()
    return dict(root_inode=root_info.st_ino, root_device=root_info.st_dev,
                entries=sorted(result, key=lambda x: x['path']))


def active_uses(paths):
    """lsof includes cwd, executable/mapped libraries and open descriptors."""
    if not hasattr(os, 'getuid') or not shutil.which('lsof'):
        raise RuntimeError('Active-use verifier unavailable; refuse cleanup')
    command = subprocess.run(['lsof', '-nP', '-u', str(os.getuid()), '-Fpn'],
                             capture_output=True, text=True, timeout=60)
    if command.returncode not in (0, 1):
        raise RuntimeError('Active-use verifier failed; refuse cleanup')
    # An error/permission warning could conceal a current user's open file.
    if command.stderr.strip():
        raise RuntimeError('Active-use verifier incomplete; refuse cleanup')
    roots = [str(Path(p).resolve()) for p in paths]
    hits, pid = [], None
    for line in command.stdout.splitlines():
        if line.startswith('p'): pid = line[1:]
        elif line.startswith('n'):
            value = line[1:]
            for root in roots:
                if value == root or value.startswith(root + os.sep):
                    hits.append(dict(pid=pid, path=value))
    # Processes may have a not-yet-opened target in their argv.
    processes = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True,
                               text=True, check=True, timeout=30)
    for line in processes.stdout.splitlines():
        for root in roots:
            if root in line:
                pid = line.strip().split(None, 1)[0]
                if int(pid) != os.getpid():
                    hits.append(dict(pid=pid, path=root, kind='command_reference'))
    return hits


def plan(root, relatives, *, check_use=active_uses):
    root = Path(root).resolve(strict=True)
    relatives = sorted(set(relatives))
    paths = [validate_target(root, name) for name in relatives]
    for i, path in enumerate(paths):
        if any(path.is_relative_to(other) for other in paths[:i] + paths[i+1:]):
            raise ValueError('Overlapping scratch targets')
    if check_use(paths):
        raise RuntimeError('Active processes still reference scratch targets')
    trees = {name: inventory_tree(path) for name, path in zip(relatives, paths)}
    entries = [row for tree in trees.values() for row in tree['entries']
               if stat.S_ISREG(row['mode'])]
    occurrences = Counter((row['device'], row['inode']) for row in entries)
    unique = {(row['device'], row['inode']): row for row in entries}
    unknown = sum(row['blocks'] is None for row in unique.values())
    allocated = None if unknown else sum(row['blocks'] * 512 for row in unique.values())
    selected = [row for key, row in unique.items() if occurrences[key] == row['links']]
    fully_unlinked = (None if any(row['blocks'] is None for row in selected)
                      else sum(row['blocks'] * 512 for row in selected))
    payload = dict(schema=SCHEMA, root=str(root), targets=trees,
                   summary=dict(targets=len(paths), regular_paths=len(entries),
                                unique_inodes=len(unique), allocated_bytes=allocated,
                                all_hardlinks_selected_bytes=fully_unlinked,
                                allocation_metrics_available=not unknown, allocation_unknown_inodes=unknown,
                                note='APFS clone/snapshot sharing can reduce actual free-space gain'))
    payload['manifest_sha256'] = digest(payload)
    return payload


def execute(manifest, *, check_use=active_uses, journal_path=None):
    expected = dict(manifest)
    claimed = expected.pop('manifest_sha256')
    if expected.get('schema') != SCHEMA or digest(expected) != claimed:
        raise ValueError('Invalid scratch manifest')
    current = plan(expected['root'], expected['targets'], check_use=check_use)
    if current != manifest:
        raise RuntimeError('Scratch tree changed since dry-run; regenerate plan')
    root = Path(expected['root'])
    removed = []
    journal_path = Path(journal_path or root / ('.scratch-retirement-' + claimed + '.jsonl'))
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    # Prepared records precede renames, so a crash never loses the retired path.
    def record(**value):
        with journal_path.open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    for relative in expected['targets']:
        source = validate_target(root, relative)
        if check_use([source]):
            raise RuntimeError('Scratch became active before retirement')
        if inventory_tree(source) != expected['targets'][relative]:
            raise RuntimeError('Scratch changed before retirement')
        retired = source.with_name('.retired-' + source.name + '-' + uuid.uuid4().hex)
        record(stage='prepared', source=str(source), retired=str(retired), manifest_sha256=claimed)
        source.rename(retired)
        record(stage='retired', source=str(source), retired=str(retired))
        # Revalidate after the atomic rename and before removing any byte.
        try:
            changed = inventory_tree(retired) != expected['targets'][relative]
            in_use = check_use([source, retired])
            if changed or in_use:
                if not source.exists():
                    retired.rename(source)
                    record(stage='restored', source=str(source), reason='changed_or_active')
                raise RuntimeError('Scratch changed or became active during retirement')
            shutil.rmtree(retired)
        except BaseException as error:
            record(stage='failed', source=str(source), retired=str(retired), error=type(error).__name__)
            raise
        record(stage='deleted', source=str(source), retired=str(retired))
        removed.append(relative)
    return dict(schema=SCHEMA, manifest_sha256=claimed, removed=removed,
                allocated_bytes_selected=current['summary']['allocated_bytes'], journal=str(journal_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--target', action='append', default=[])
    parser.add_argument('--targets-json', type=Path)
    parser.add_argument('--execute-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.execute_manifest:
        if args.target or args.root or args.targets_json:
            parser.error('Execution accepts only the previously reviewed manifest')
        value = execute(json.loads(args.execute_manifest.read_text()))
    else:
        if not args.root: parser.error('--root is required for dry-run')
        relatives = args.target + (json.loads(args.targets_json.read_text()) if args.targets_json else [])
        if not relatives: parser.error('Explicit scratch targets are required')
        value = plan(args.root, relatives)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(value.get('summary', {k: v for k, v in value.items() if k != 'removed'})))


if __name__ == '__main__':
    main()
