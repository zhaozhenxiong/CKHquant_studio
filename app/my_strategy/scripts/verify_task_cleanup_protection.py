"""Read-only before/after preservation proof for task-history maintenance.

No service objects, model inference, SQLite writes or cleanup are performed.
Private snapshots contain identities and SHA values, never account payloads,
configuration values, credential environment variables or market rows.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def physical(path):
    path = Path(path)
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError('保护资产在核验期间改变：' + str(path))
    return {'size_bytes': after.st_size, 'sha256': sha}


def readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def capture(*, raw_db, metadata, results_db, runs_root, kronos_root):
    raw_db, metadata, runs_root, kronos_root = map(Path, (raw_db, metadata, runs_root, kronos_root))
    out = {'schema': 'task_cleanup_protection_v1', 'raw': {}, 'watchlist': {}, 'accounts': {},
           'research_objects': {}, 'pointers': {}, 'pins': {}, 'model_files': {}, 'errors': []}
    # A WAL file is part of the physical source, even with read-only connections.
    out['raw']['database'] = physical(raw_db)
    wal = raw_db.with_name(raw_db.name + '-wal')
    out['raw']['wal'] = physical(wal) if wal.exists() and wal.stat().st_size else None
    with closing(readonly(raw_db)) as conn:
        row = conn.execute('SELECT count(*),count(DISTINCT stock),min(date),max(date) FROM stock_daily_normalized').fetchone()
        out['raw']['coverage'] = {'rows': row[0], 'stocks': row[1], 'earliest': row[2], 'latest': row[3]}
        out['raw']['securities'] = conn.execute('SELECT count(*) FROM securities').fetchone()[0]
    with closing(readonly(metadata / 'personal_portfolio.db')) as conn:
        rows = [tuple(row) for row in conn.execute('SELECT * FROM watchlist ORDER BY symbol')]
        out['watchlist'] = {'count': len(rows), 'rows_sha256': stable(rows)}
    research = metadata / 'light_research'
    with closing(readonly(research / 'research.db')) as conn:
        settings = dict(conn.execute('SELECT name,value_json FROM settings'))
        accounts = json.loads(settings.get('enabled_accounts', '{}'))
        out['accounts'] = {'count': len(accounts), 'state_sha256': stable(accounts)}
        for name, value in settings.items():
            if name in {'current_score', 'rollback_score', 'enabled_accounts', 'light_experiment_registry'} or name.startswith('checkpoint:'):
                out['pointers']['research:' + name] = stable(json.loads(value))
        # Preserve all simulations, score models and their lookup/checkpoint
        # objects, including older saved payloads missing a modern pin.
        namespaces = {'simulations', 'result_lookup', 'score_models', 'mlp_score_models', 'simulation_checkpoints', 'score_observation_manifests'}
        rows = conn.execute('''SELECT r.cache_key,r.namespace,r.sha256,o.size_bytes FROM results r JOIN objects o USING(sha256)
            WHERE o.state='active' ''').fetchall()
        pinned = {row[0] for row in conn.execute('SELECT DISTINCT cache_key FROM pins')}
        for row in rows:
            if row['namespace'] not in namespaces and row['cache_key'] not in pinned:
                continue
            path = research / 'objects' / row['sha256'][:2] / (row['sha256'] + '.json.gz')
            actual = physical(path)
            if actual['sha256'] != row['sha256'] or actual['size_bytes'] != row['size_bytes']:
                out['errors'].append('保护对象校验失败：' + row['cache_key'])
            out['research_objects'][row['cache_key']] = {**actual, 'namespace': row['namespace']}
        for row in conn.execute('SELECT cache_key,owner FROM pins ORDER BY cache_key,owner'):
            # Owner can encode an account id; hash rather than emit its text.
            out['pins'][stable(list(row))] = True
    with closing(readonly(results_db)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ['current_model_selection', 'current_model_events', 'czsc_model_aliases', 'czsc_model_release_events', 'current_model_comparisons']:
            if table in tables:
                rows = [tuple(row) for row in conn.execute('SELECT * FROM "' + table + '" ORDER BY rowid')]
                out['pointers']['legacy:' + table] = {'rows': len(rows), 'sha256': stable(rows)}
    for relative in ['model.safetensors', 'config.json', 'tokenizer/model.safetensors', 'tokenizer/config.json']:
        path = kronos_root / relative
        out['model_files']['kronos/' + relative] = physical(path)
    for path in sorted(runs_root.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.pt', '.pth', '.safetensors', '.pkl', '.pickle', '.npz'}:
            if '.task-cleanup' not in path.parts:
                out['model_files']['runs/' + path.relative_to(runs_root).as_posix()] = physical(path)
    out['identity_sha256'] = stable(out)
    return out


def compare(before, after):
    differences = []
    for section in ['raw', 'watchlist', 'accounts']:
        if before[section] != after[section]:
            differences.append({'section': section, 'reason': '保护事实或物理SHA改变'})
    # Newly generated integration-test predictions/scans do not invalidate old
    # protected identities. Every prior object/pointer/pin must still exist.
    for section in ['research_objects', 'pointers', 'pins', 'model_files']:
        for key, value in before[section].items():
            if after[section].get(key) != value:
                differences.append({'section': section, 'identity': key, 'reason': '保护引用丢失或SHA改变'})
    return {'passed': not differences and not before['errors'] and not after['errors'],
            'differences': differences, 'before_errors': before['errors'], 'after_errors': after['errors']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    if not (project / 'AGENTS.md').is_file():
        raise SystemExit('请指定含 AGENTS.md 的应用根目录')
    sys.path.insert(0, str(project))
    from my_strategy.core.paths import METADATA_ROOT, PROCESSED_DATA_ROOT, ARTIFACT_RUNS_ROOT
    from my_strategy.storage.db_paths import raw_db_path
    from my_strategy.services.kronos_assets import asset_root
    snapshot = capture(raw_db=raw_db_path(), metadata=METADATA_ROOT,
                       results_db=PROCESSED_DATA_ROOT / 'czsc/results.db', runs_root=ARTIFACT_RUNS_ROOT,
                       kronos_root=asset_root())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'snapshot': str(args.output.resolve()), 'raw': snapshot['raw']['coverage'],
               'watchlist_count': snapshot['watchlist']['count'], 'enabled_accounts': snapshot['accounts']['count'],
               'protected_objects': len(snapshot['research_objects']), 'model_files': len(snapshot['model_files']),
               'pointers': len(snapshot['pointers']), 'errors': snapshot['errors']}
    if args.compare:
        comparison = compare(json.loads(args.compare.read_text()), snapshot)
        report = args.output.with_suffix('.comparison.json')
        report.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.update(passed=comparison['passed'], comparison_path=str(report.resolve()), differences=comparison['differences'])
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if comparison['passed'] else 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not snapshot['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
