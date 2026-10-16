"""Preview and retire exact Beijing rows after the frozen market scan is accepted.

This deliberately only writes khquant_raw.db. Mixed board snapshots, immutable
runs/audit, models, accounts, and CAS objects are retained. Preview uses SQLite
mode=ro; apply requires the service stopped, complete accepted scans, an unchanged
plan, an offline full preservation hash, and a separate rollback database backup.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from contextlib import closing
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import sys
import time
import uuid

SCHEMA = 'retire_beijing_raw_v1'
METHODS = ('structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2')
TARGETS = {'stock_daily_raw': 'stock', 'stock_daily_normalized': 'stock',
           'data_source_status': 'code', 'stock_basic': 'code',
           'stock_industry_metadata': 'stock', 'stock_industry_exclusions': 'stock',
           'securities': 'stock'}
ACTIVE = {'queued', 'running', 'cancelling'}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False, default=lambda v: {'bytes_b64': base64.b64encode(v).decode()})


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def file_identity(path):
    path = Path(path)
    a = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    b = path.stat()
    if (a.st_size, a.st_mtime_ns, a.st_ctime_ns) != (b.st_size, b.st_mtime_ns, b.st_ctime_ns):
        raise ValueError('File changed during inspection: ' + str(path))
    return {'bytes': b.st_size, 'sha256': sha}


def readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    conn.execute('PRAGMA query_only=ON')
    return conn


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def _configured_paths():
    # Importing my_strategy first applies runtime_env's repository .env contract.
    from my_strategy.core.paths import PROJECT_ROOT, METADATA_ROOT, PROCESSED_DATA_ROOT, ARTIFACT_RUNS_ROOT
    from my_strategy.storage.db_paths import raw_db_path
    from my_strategy.services.kronos_assets import asset_root
    return {'project': PROJECT_ROOT.resolve(), 'raw': raw_db_path().resolve(),
            'metadata': METADATA_ROOT.resolve(), 'results': (PROCESSED_DATA_ROOT / 'czsc/results.db').resolve(),
            'runs': ARTIFACT_RUNS_ROOT.resolve(), 'kronos': asset_root().resolve()}


def paths(project):
    project = Path(project).resolve()
    if not (project / 'AGENTS.md').is_file():
        raise ValueError('Expected application root with AGENTS.md')
    sys.path.insert(0, str(project))
    configured = _configured_paths()
    if configured['project'] != project:
        raise ValueError('Configured project differs from --project-root')
    for key, value in configured.items():
        if value != value.resolve():
            raise ValueError('Storage resolver returned an unresolved path: ' + key)
    return configured


def raw_physical(p):
    out = {'database': file_identity(p['raw'])}
    wal = Path(str(p['raw']) + '-wal')
    if wal.exists():
        out['wal'] = file_identity(wal)
    return out


def validate_raw_scan_identity(expected, observed, accepted_raw):
    """Preserve frozen identities while recording an empty WAL's closed lifetime.

    SQLite may unlink an empty WAL when the last connection closes. Only that
    one-way transition is equivalent here; plan freshness remains byte-exact.
    """
    allowed = {'database', 'wal'}
    for identity in (expected, observed):
        if not isinstance(identity, dict) or 'database' not in identity or set(identity) - allowed:
            raise ValueError('Invalid raw identity keys')
    valid_sha = lambda value: isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
    if not all(valid_sha(value) for value in expected.values()):
        raise ValueError('Invalid frozen raw SHA')
    for value in observed.values():
        if (not isinstance(value, dict) or set(value) != {'bytes', 'sha256'}
                or type(value['bytes']) is not int or value['bytes'] < 0 or not valid_sha(value['sha256'])):
            raise ValueError('Invalid observed raw physical identity')
    actual = {key: value['sha256'] for key, value in observed.items()}
    accepted_db = accepted_raw.get('database', {}) if isinstance(accepted_raw, dict) else {}
    if (actual['database'] != expected['database']
            or not isinstance(accepted_db, dict) or accepted_db.get('sha256') != actual['database']):
        raise ValueError('Current raw input differs from accepted scans')
    empty_sha = hashlib.sha256(b'').hexdigest()
    if 'wal' in observed and ((observed['wal']['bytes'] == 0) != (actual['wal'] == empty_sha)):
        raise ValueError('Observed WAL size and SHA disagree')
    relation = 'exact'
    if actual != expected:
        accepted_wal = accepted_raw.get('wal')
        empty_accepted_wal = ('wal' in accepted_raw and (accepted_wal is None or
                              accepted_wal == {'size_bytes': 0, 'sha256': empty_sha}))
        if (set(expected) != allowed or set(actual) != {'database'}
                or expected['wal'] != empty_sha or not empty_accepted_wal):
            raise ValueError('Current raw input differs from accepted scans')
        relation = 'empty_wal_removed_after_close'
    return {'schema': 'raw_scan_identity_validation_v1', 'relation': relation,
            'frozen_identity': dict(expected),
            'observed_physical': {key: dict(value) for key, value in observed.items()},
            'accepted_database_sha256': accepted_db['sha256']}


def symbols(conn):
    found = []
    for stock, exchange in conn.execute('SELECT stock,exchange FROM securities ORDER BY stock'):
        explicit = bool(re.fullmatch(r'\d{6}\.BJ', stock or ''))
        market = str(exchange).upper() in {'BJ', 'BSE', '北交所', '北京证券交易所'}
        if explicit != market:
            raise ValueError('Ambiguous Beijing identity requires review: ' + str(stock))
        if explicit:
            found.append(stock)
    return found


def raw_schema(conn):
    return digest(list(conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name")))


def validate_symbol_coverage(conn, bj):
    known = set(bj)
    for table, col in TARGETS.items():
        if table not in tables(conn): continue
        # The production columns are canonical. Other explicit BJ encodings or
        # bare BJ codes are ambiguous here and require a separately reviewed map.
        quoted = quote(col)
        candidate = (f"UPPER({quoted}) LIKE '%.BJ' OR UPPER({quoted}) LIKE '%.BSE' "
                     f"OR UPPER({quoted}) LIKE 'BJ%' OR UPPER({quoted}) LIKE 'BSE%' "
                     f"OR {quoted} GLOB '[48][0-9][0-9][0-9][0-9][0-9]' "
                     f"OR {quoted} GLOB '92[0-9][0-9][0-9][0-9]'")
        values = {row[0] for row in conn.execute('SELECT DISTINCT ' + quoted + ' FROM ' + quote(table) + ' WHERE ' + candidate)}
        if values - known:
            raise ValueError('Unmapped/orphan Beijing identifiers in ' + table + ': ' + ','.join(sorted(values - known)))


def tables(conn):
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def row_digest(conn, table, clause='', args=()):
    h = hashlib.sha256(); count = 0
    columns = [r[1] for r in conn.execute('PRAGMA table_info(' + quote(table) + ')')]
    info = list(conn.execute('PRAGMA table_info(' + quote(table) + ')'))
    primary = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
    definition = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
    order = ','.join(map(quote, primary)) if 'WITHOUT ROWID' in definition.upper() else 'rowid'
    for row in conn.execute('SELECT * FROM ' + quote(table) + clause + ' ORDER BY ' + order, args):
        value = encoded(list(row)).encode()
        h.update(len(value).to_bytes(8, 'big')); h.update(value); count += 1
    return {'count': count, 'columns': columns, 'rows_sha256': h.hexdigest()}


def raw_rows(conn, bj, *, protected=False):
    output = {}
    marks = ','.join('?' for _ in bj) or 'NULL'
    for table in tables(conn):
        if table in TARGETS:
            col = quote(TARGETS[table])
            clause = (' WHERE ' + col + ' NOT IN (' + marks + ') OR ' + col + ' IS NULL' if protected
                      else ' WHERE ' + col + ' IN (' + marks + ')')
            output[table] = row_digest(conn, table, clause, bj)
        elif protected:
            output[table] = row_digest(conn, table)
    return output


def protected_assets(p):
    # Same proof fields as verify_task_cleanup_protection, without opening raw:
    # this also works while apply owns an uncommitted raw write transaction.
    from my_strategy.scripts.verify_task_cleanup_protection import physical, stable
    out = {'schema': 'task_cleanup_protection_v1', 'watchlist': {}, 'accounts': {},
           'research_objects': {}, 'pointers': {}, 'pins': {}, 'model_files': {}, 'errors': []}
    with closing(readonly(p['metadata'] / 'personal_portfolio.db')) as conn:
        rows = list(conn.execute('SELECT * FROM watchlist ORDER BY symbol'))
        out['watchlist'] = {'count': len(rows), 'rows_sha256': stable(rows)}
    research = p['metadata'] / 'light_research'
    with closing(readonly(research / 'research.db')) as conn:
        settings = dict(conn.execute('SELECT name,value_json FROM settings'))
        accounts = json.loads(settings.get('enabled_accounts', '{}'))
        out['accounts'] = {'count': len(accounts), 'state_sha256': stable(accounts)}
        for name, value in settings.items():
            if name in {'current_score', 'rollback_score', 'enabled_accounts', 'light_experiment_registry'} or name.startswith('checkpoint:'):
                out['pointers']['research:' + name] = stable(json.loads(value))
        namespaces = {'simulations', 'result_lookup', 'score_models', 'mlp_score_models', 'simulation_checkpoints', 'score_observation_manifests'}
        pinned = {row[0] for row in conn.execute('SELECT DISTINCT cache_key FROM pins')}
        for key, namespace, sha, size in conn.execute("SELECT r.cache_key,r.namespace,r.sha256,o.size_bytes FROM results r JOIN objects o USING(sha256) WHERE o.state='active'"):
            if namespace not in namespaces and key not in pinned: continue
            path = research / 'objects' / sha[:2] / (sha + '.json.gz')
            actual = physical(path)
            if actual['sha256'] != sha or actual['size_bytes'] != size:
                raise ValueError('Protected object hash mismatch: ' + key)
            out['research_objects'][key] = {**actual, 'namespace': namespace}
        for row in conn.execute('SELECT cache_key,owner FROM pins ORDER BY cache_key,owner'):
            out['pins'][stable(list(row))] = True
    with closing(readonly(p['results'])) as conn:
        known = tables(conn)
        for table in ['current_model_selection', 'current_model_events', 'czsc_model_aliases', 'czsc_model_release_events', 'current_model_comparisons']:
            if table in known:
                rows = list(conn.execute('SELECT * FROM ' + quote(table) + ' ORDER BY rowid'))
                out['pointers']['legacy:' + table] = {'rows': len(rows), 'sha256': stable(rows)}
    for relative in ['model.safetensors', 'config.json', 'tokenizer/model.safetensors', 'tokenizer/config.json']:
        out['model_files']['kronos/' + relative] = physical(p['kronos'] / relative)
    for path in sorted(p['runs'].rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.pt', '.pth', '.safetensors', '.pkl', '.pickle', '.npz'} and '.task-cleanup' not in path.parts:
            out['model_files']['runs/' + path.relative_to(p['runs']).as_posix()] = physical(path)
    return out


def project_processes(p):
    import psutil
    problems = []
    root = p['project'].resolve()
    protected = {p['raw'].resolve(), (p['metadata'] / 'training_resource.db').resolve(),
                 (p['metadata'] / 'model_training.db').resolve(),
                 (p['metadata'] / 'light_research/research.db').resolve(), p['results'].resolve()}
    protected |= {Path(str(path) + suffix) for path in tuple(protected) for suffix in ('-wal', '-shm', '-journal')}
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
        if process.pid == os.getpid(): continue
        matching = False
        try:
            info = process.info; name = str(info.get('name') or '').lower()
            command = info.get('cmdline') or []
            python = 'python' in name or 'uvicorn' in name or 'gunicorn' in name
            cwd = Path(info['cwd']).resolve() if info.get('cwd') else None
            matching = python and (cwd is not None and cwd.is_relative_to(root)
                or any(str(root) in arg or 'my_strategy' in arg for arg in command))
            opened = any(Path(item.path).resolve() in protected for item in process.open_files())
            if matching or opened:
                problems.append(f'Project/protected-storage process still alive: pid={process.pid}')
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if locals().get('matching', False):
                problems.append(f'Cannot verify project process: pid={process.pid}')
    return problems


def activity(p, port):
    problems = []
    terminal = {'succeeded', 'completed', 'partial', 'failed', 'blocked', 'cancelled', 'paused'}
    with closing(readonly(p['metadata'] / 'czsc_tasks.db')) as conn:
        states = dict(conn.execute('SELECT status,count(*) FROM czsc_tasks GROUP BY status'))
        for state, count in states.items():
            if state not in terminal:
                problems.append(f'{count} active or unknown persisted tasks: {state}')
    with closing(readonly(p['metadata'] / 'light_research/research.db')) as conn:
        count = conn.execute('SELECT count(*) FROM leases WHERE expires_at>?', (time.time(),)).fetchone()[0]
        if count: problems.append(f'{count} active cache leases')
    # Independent training/runtime registries can use a separate database.
    for name in ('model_training.db', 'training_resource.db'):
        path = p['metadata'] / name
        if not path.exists(): continue
        with closing(readonly(path)) as conn:
            for table in tables(conn):
                columns = {r[1] for r in conn.execute('PRAGMA table_info(' + quote(table) + ')')}
                if 'status' in columns:
                    for state, count in conn.execute('SELECT status,count(*) FROM ' + quote(table) + ' GROUP BY status'):
                        if state not in terminal:
                            problems.append(f'{name}/{table}: {count} active or unknown state {state}')
                if 'lease' in table.lower():
                    rows = conn.execute('SELECT count(*) FROM ' + quote(table)).fetchone()[0]
                    if 'expires_at' in columns:
                        live = conn.execute('SELECT count(*) FROM ' + quote(table) + ' WHERE expires_at>? OR typeof(expires_at) NOT IN (\'real\',\'integer\')', (time.time(),)).fetchone()[0]
                        if live: problems.append(f'{name}/{table}: active or unverified leases')
                    elif rows:
                        problems.append(f'{name}/{table}: lease schema needs review')
    for candidate in {8124, port}:
        sock = socket.socket(); sock.settimeout(.3)
        try:
            if sock.connect_ex(('127.0.0.1', candidate)) == 0:
                problems.append(f'Dashboard service is still listening on {candidate}; stop it after scans finish')
        finally: sock.close()
    problems.extend(project_processes(p))
    return problems


def accepted_scans(p, state_path, evidence):
    """Read actual reports and bind comparison content; never trust a lone flag."""
    state_path, evidence = Path(state_path).resolve(), Path(evidence).resolve()
    state = json.loads(state_path.read_text())
    if state.get('status') != 'completed':
        raise ValueError('Three-method controller is not completed')
    for scope in [state, *state.get('methods', {}).values()]:
        if any(scope.get(key) for key in ('active_job_id', 'submission_intent', 'active_spec')):
            raise ValueError('Controller still holds active work or an unresolved submission')
    comparison_path = evidence / 'comparison.json'
    comparison = json.loads(comparison_path.read_text())
    if (comparison.get('schema') != 'account_free_screening_comparison_v1'
            or comparison.get('trial') is not False or comparison.get('verification', {}).get('passed') is not True
            or comparison['verification'].get('all_population_processed') is not True):
        raise ValueError('Final comparison is absent or not accepted')
    verification_path = evidence / 'protection-after.comparison.json'
    if json.loads(verification_path.read_text()).get('passed') is not True:
        raise ValueError('Scan protection acceptance did not pass')
    references = {str(path): file_identity(path) for path in
                  (state_path, comparison_path, verification_path,
                   evidence / 'all_stock_comparison.csv', evidence / 'screening_comparison.md')}
    reports = {}; first = None
    with closing(readonly(p['results'])) as conn:
        for method in METHODS:
            run_id = state.get('methods', {}).get(method, {}).get('final_run_id')
            if not run_id: raise ValueError('Missing final run: ' + method)
            row = conn.execute('SELECT report_path,summary FROM czsc_runs WHERE run_id=?', (run_id,)).fetchone()
            if not row: raise ValueError('Final report not indexed: ' + method)
            path = (p['runs'] / row[0]).resolve()
            if not path.is_relative_to(p['runs'].resolve()) or not path.is_file():
                raise ValueError('Invalid final report path')
            report = json.loads(path.read_text()); reports[method] = report
            references[str(path)] = file_identity(path)
            if first is None: first = report
            population = report.get('population')
            if (report.get('run_id') != run_id or report.get('analysis_method') != method
                    or report.get('run_context', {}).get('scope') != 'full'
                    or report.get('reference_only') is not True or report.get('account_scope') != 'none'
                    or report.get('coverage', {}).get('complete') is not True
                    or report['coverage'].get('deferred') != 0
                    or not population or len(population) != len(set(population))
                    or any(not re.fullmatch(r'\d{6}\.(SH|SZ)', s) for s in population)
                    or population != first['population'] or population != comparison.get('population')):
                raise ValueError('Final report is incomplete or has a changed population: ' + method)
            counts = Counter(item.get('method_evaluation', {}).get('status') for item in report['rows'])
            expected = {'requested': len(population), 'evaluated': len(population), 'deferred': 0,
                        'success': counts['complete'] + counts['partial'], 'successful': counts['complete'] + counts['partial'],
                        'failed': counts['failed'], 'unavailable': counts['partial']}
            if set(counts) - {'complete', 'partial', 'failed'} or any(report['coverage'].get(k) != v for k, v in expected.items()):
                raise ValueError('Final coverage counts differ from actual row statuses')
            if json.loads(row[1]).get('coverage') != report['coverage']:
                raise ValueError('Indexed coverage differs from final report')
            if comparison['sources'][method]['run_id'] != run_id or comparison['methods'][method]['run_id'] != run_id:
                raise ValueError('Comparison is bound to different runs')
            if comparison['methods'][method]['coverage'] != report['coverage']:
                raise ValueError('Comparison coverage differs')
            for key in ('population', 'raw', 'calendar', 'code', 'prediction'):
                identity = report['frozen_identity'][key]
                if identity != first['frozen_identity'][key]:
                    raise ValueError('Final scans have inconsistent frozen identities')
                if comparison['verification']['shared_frozen_identity_hashes'][method][key] != digest(identity):
                    raise ValueError('Comparison frozen identity digest mismatch')
            for key, value in state.get('parameters', {}).items():
                if report['method_context'].get(key) != value:
                    raise ValueError('Final scan parameters differ from controller')
    with closing(readonly(p['raw'])) as conn:
        expected_population = [row[0] for row in conn.execute("SELECT stock FROM securities WHERE (stock LIKE '6%.SH') OR (stock LIKE '0%.SZ') OR (stock LIKE '3%.SZ') ORDER BY stock")]
    if first['population'] != expected_population:
        raise ValueError('Final scans are not the entire current eligible Shanghai/Shenzhen population')
    stocks = comparison.get('stocks', [])
    if [row.get('symbol') for row in stocks] != first['population']:
        raise ValueError('Comparison stock rows differ from final population')
    for method, report in reports.items():
        if [row.get('symbol') for row in report['rows']] != first['population']:
            raise ValueError('Final report rows differ from frozen population')
        for original, compared in zip(report['rows'], stocks):
            evaluation = original['method_evaluation']; item = compared['methods'][method]
            if (evaluation.get('status') == 'deferred' or item['status'] != evaluation['status']
                    or item['raw_direction'] != evaluation['direction']
                    or item['kronos'] != (evaluation.get('kronos') or {})
                    or item['score'] != (evaluation.get('score') or None)
                    or item['source_identity_sha256'] != (digest(evaluation['source_identity']) if evaluation.get('source_identity') else None)):
                raise ValueError('Comparison content differs from actual final row')
    with (evidence / 'all_stock_comparison.csv').open(newline='', encoding='utf-8-sig') as stream:
        if [row['symbol'] for row in csv.DictReader(stream)] != first['population']:
            raise ValueError('Comparison CSV population mismatch')
    # Bind the original accepted protection proof, not an invented receipt.
    from my_strategy.scripts.verify_task_cleanup_protection import compare
    before_path, after_path = evidence / 'protection-before.json', evidence / 'protection-after.json'
    before, after = json.loads(before_path.read_text()), json.loads(after_path.read_text())
    if not compare(before, after)['passed']:
        raise ValueError('Original scan preservation snapshots do not match')
    for path in (before_path, after_path): references[str(path)] = file_identity(path)
    raw_validation = validate_raw_scan_identity(first['frozen_identity']['raw'], raw_physical(p), after['raw'])
    if protected_assets(p) != {k: v for k, v in after.items() if k not in {'raw', 'identity_sha256'}}:
        raise ValueError('Protected models/accounts changed after scan acceptance')
    return {'final_runs': {m: r['run_id'] for m, r in reports.items()}, 'references': references,
            'raw_identity_validation': raw_validation}


def preview(p, *, state, evidence, port=8124, full_protection=False):
    blockers = activity(p, port)
    try: acceptance = accepted_scans(p, state, evidence)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        blockers.append('Scan acceptance: ' + str(exc)); acceptance = None
    with closing(readonly(p['raw'])) as conn:
        bj = symbols(conn)
        validate_symbol_coverage(conn, bj)
        schema = raw_schema(conn)
        candidates = raw_rows(conn, bj)
        preserved = raw_rows(conn, bj, protected=True) if full_protection and not blockers else None
    if not full_protection: blockers.append('Run offline preview with --full-protection before apply')
    value = {'schema': SCHEMA, 'script_sha256': file_identity(__file__)['sha256'], 'created_at': time.time(), 'project': str(p['project']),
             'raw_path': str(p['raw'].resolve()), 'storage_paths': {k: str(v) for k, v in p.items()}, 'state': str(Path(state).resolve()),
             'scan_evidence': str(Path(evidence).resolve()), 'service_port': port,
             'symbols': bj, 'candidates': candidates, 'raw_schema_sha256': schema, 'raw_physical': raw_physical(p),
             'protected_raw_rows': preserved, 'protected_assets': protected_assets(p),
             'accepted_scans': acceptance, 'blockers': blockers,
             'scope': 'Only exact raw BJ rows in the hardcoded table allowlist; no files, CAS, mixed runs or audit deleted.'}
    value['plan_sha256'] = digest(value)
    return value


def validate_plan(plan):
    if plan.get('script_sha256') != file_identity(__file__)['sha256']:
        raise ValueError('Plan was made by a different cleanup script')
    if plan.get('schema') != SCHEMA or plan.get('plan_sha256') != digest({k: v for k, v in plan.items() if k != 'plan_sha256'}):
        raise ValueError('Invalid or modified plan')
    if plan.get('blockers') or not plan.get('protected_raw_rows') or not plan.get('accepted_scans'):
        raise ValueError('Plan was blocked or lacks full offline preservation proof')
    if not plan['symbols'] or any(not re.fullmatch(r'\d{6}\.BJ', s) for s in plan['symbols']):
        raise ValueError('Plan has no exact Beijing targets')


def audit_output(p, path):
    path = Path(path).resolve()
    protected_roots = (p['raw'].parent, p['metadata'], p['results'].parent, p['runs'].parent, p['kronos'])
    if path.exists() or any(path.is_relative_to(root.resolve()) for root in protected_roots):
        raise ValueError('Evidence/backup output must be a fresh path outside business storage')
    return path


def apply(p, plan, *, backup, journal):
    validate_plan(plan)
    if {k: str(v) for k, v in p.items()} != plan['storage_paths']:
        raise ValueError('Plan belongs to another workspace')
    backup, journal = audit_output(p, backup), audit_output(p, journal)
    if backup.exists() or journal.exists() or backup == p['raw'].resolve() or backup == journal:
        raise ValueError('Backup/journal must be fresh paths distinct from raw input')
    from filelock import FileLock
    with FileLock(str(p['metadata'] / 'light_research/compute.lock'), timeout=0):
        problems = activity(p, plan['service_port'])
        if problems: raise ValueError('; '.join(problems))
        acceptance = accepted_scans(p, plan['state'], plan['scan_evidence'])
        if acceptance != plan['accepted_scans'] or raw_physical(p) != plan['raw_physical']:
            raise ValueError('Accepted input changed after preview')
        if protected_assets(p) != plan['protected_assets']:
            raise ValueError('Protected assets changed after preview')
        backup.parent.mkdir(parents=True, exist_ok=True); journal.parent.mkdir(parents=True, exist_ok=True)
        with closing(readonly(p['raw'])) as source, closing(sqlite3.connect(backup)) as target:
            source.backup(target)
        backup_sha = file_identity(backup)
        record = {'schema': SCHEMA, 'plan_sha256': plan['plan_sha256'], 'status': 'prepared',
                  'backup': str(backup), 'backup_identity': backup_sha, 'deleted': {}}
        def persist():
            temporary = journal.with_name(journal.name + '.' + uuid.uuid4().hex + '.tmp')
            with temporary.open('x') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, journal)
            if hasattr(os, 'O_DIRECTORY'):
                descriptor = os.open(journal.parent, os.O_DIRECTORY)
                try: os.fsync(descriptor)
                finally: os.close(descriptor)
        persist()
        with closing(sqlite3.connect(p['raw'], timeout=30)) as conn:
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute('BEGIN EXCLUSIVE')
            try:
                if activity(p, plan['service_port']): raise ValueError('Activity appeared before transaction')
                bj = symbols(conn)
                validate_symbol_coverage(conn, bj)
                if raw_schema(conn) != plan['raw_schema_sha256']:
                    raise ValueError('Raw schema changed after preview')
                if conn.execute('PRAGMA foreign_key_check').fetchone() is not None:
                    raise ValueError('Existing raw foreign-key inconsistency')
                if bj != plan['symbols'] or raw_rows(conn, bj) != plan['candidates']:
                    raise ValueError('Beijing rows changed after preview')
                if raw_rows(conn, bj, protected=True) != plan['protected_raw_rows']:
                    raise ValueError('Protected raw rows changed after preview')
                marks = ','.join('?' for _ in bj)
                for table, col in TARGETS.items():
                    if table not in plan['candidates']: continue
                    result = conn.execute('DELETE FROM ' + quote(table) + ' WHERE ' + quote(col) + ' IN (' + marks + ')', bj)
                    if result.rowcount != plan['candidates'][table]['count']:
                        raise ValueError('Deletion count mismatch: ' + table)
                    record['deleted'][table] = result.rowcount
                if raw_rows(conn, bj, protected=True) != plan['protected_raw_rows']:
                    raise ValueError('Protected raw row digest changed; rolling back')
                if any(v['count'] for v in raw_rows(conn, bj).values()):
                    raise ValueError('Previewed BJ rows remain')
                if protected_assets(p) != plan['protected_assets']:
                    raise ValueError('Protected account/model changed; rolling back')
                if raw_schema(conn) != plan['raw_schema_sha256'] or conn.execute('PRAGMA foreign_key_check').fetchone() is not None:
                    raise ValueError('Schema/foreign-key verification failed; rolling back')
                if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLite quick_check failed; rolling back')
                conn.commit()
            except BaseException as exc:
                conn.rollback(); record.update(status='rolled_back', error=str(exc)); persist(); raise
        record.update(status='committed', protected_raw_rows_unchanged=True,
                      protected_assets_unchanged=True, raw_after=raw_physical(p))
        persist()
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preview', 'apply'])
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--scan-state', type=Path)
    parser.add_argument('--scan-evidence', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--full-protection', action='store_true')
    parser.add_argument('--service-port', type=int, default=8124)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--backup', type=Path)
    parser.add_argument('--journal', type=Path)
    args = parser.parse_args(argv)
    p = paths(args.project_root); sys.path.insert(0, str(p['project']))
    if args.mode == 'preview':
        if not args.output or not args.scan_state or not args.scan_evidence: parser.error('preview requires --output, --scan-state, --scan-evidence')
        value = preview(p, state=args.scan_state, evidence=args.scan_evidence,
                        port=args.service_port, full_protection=args.full_protection)
        args.output = audit_output(p, args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'plan': str(args.output), 'symbols': len(value['symbols']),
                          'rows': {k: v['count'] for k, v in value['candidates'].items()}, 'blockers': value['blockers']}, ensure_ascii=False))
    else:
        if not args.plan or not args.backup or not args.journal: parser.error('apply requires --plan, --backup, --journal')
        print(json.dumps(apply(p, json.loads(args.plan.read_text()), backup=args.backup, journal=args.journal), ensure_ascii=False))


if __name__ == '__main__':
    main()
