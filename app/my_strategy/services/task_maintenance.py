"""Explicit terminal-task retirement and narrowly scoped, journalled cleanup.

Only ordinary analysis report directories and orphaned job-summary CAS objects
are physically eligible. Raw inputs, model/source/native/board runs, personal
state and audit are never deletion candidates. A manifest is previewed first;
execution rechecks identities, references, active work and every file hash.
"""
from __future__ import annotations

from contextlib import ExitStack, closing, contextmanager, nullcontext
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import uuid

ACTIVE = {'queued', 'running', 'cancelling'}
TERMINAL = {'succeeded', 'partial', 'failed', 'blocked', 'cancelled', 'interrupted'}
SCHEMA = 'task_maintenance_v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_connection(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def quote(name):
    return '"' + name.replace('"', '""') + '"'


class MaintenanceConflict(ValueError):
    pass


class TaskMaintenance:
    def __init__(self, tasks, light, *, metadata_root=None, clock=time.time):
        self.tasks, self.light, self.clock = tasks, light, clock
        self.root = Path(metadata_root or tasks.db_path.parent).resolve()
        self.runs = Path(tasks.results.runs_root).resolve()
        self.results_db = Path(tasks.results.db_path).resolve()
        self.db_path = self.root / 'task_maintenance.db'
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS cleanup_plans(plan_id TEXT PRIMARY KEY,sha256 TEXT NOT NULL,payload TEXT NOT NULL,created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS cleanup_events(event_id INTEGER PRIMARY KEY,plan_id TEXT,action TEXT NOT NULL,identity TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL,created_at REAL NOT NULL);
            ''')
            conn.commit()

        from my_strategy.storage.task_journal import TaskJournal
        self.journal = getattr(light, 'journal', None) or TaskJournal(light.store.root / 'task_journal.db')
        self._prepare_history_index()

    def _prepare_history_index(self):
        # Projection tables/triggers live in mutable task metadata, never raw.
        with closing(sqlite3.connect(self.tasks.db_path, timeout=30)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS task_display(job_id TEXT PRIMARY KEY,status TEXT NOT NULL,
                    kind TEXT,updated_at TEXT,payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS task_display_page ON task_display(status,kind,updated_at DESC,job_id);
                CREATE INDEX IF NOT EXISTS task_display_recent ON task_display(updated_at DESC,job_id);
                CREATE TRIGGER IF NOT EXISTS task_display_insert AFTER INSERT ON czsc_tasks BEGIN
                    INSERT OR REPLACE INTO task_display VALUES(NEW.job_id,NEW.status,json_extract(NEW.payload,'$.kind'),
                    COALESCE(json_extract(NEW.payload,'$.finished_at'),json_extract(NEW.payload,'$.updated_at'),json_extract(NEW.payload,'$.created_at'),''),
                    json_remove(NEW.payload,'$.result','$.fingerprint','$.owner_pid','$.owner_started')); END;
                CREATE TRIGGER IF NOT EXISTS task_display_update AFTER UPDATE ON czsc_tasks BEGIN
                    INSERT OR REPLACE INTO task_display VALUES(NEW.job_id,NEW.status,json_extract(NEW.payload,'$.kind'),
                    COALESCE(json_extract(NEW.payload,'$.finished_at'),json_extract(NEW.payload,'$.updated_at'),json_extract(NEW.payload,'$.created_at'),''),
                    json_remove(NEW.payload,'$.result','$.fingerprint','$.owner_pid','$.owner_started')); END;
                CREATE TRIGGER IF NOT EXISTS task_display_delete AFTER DELETE ON czsc_tasks BEGIN
                    DELETE FROM task_display WHERE job_id=OLD.job_id; END;
                INSERT OR IGNORE INTO task_display SELECT job_id,status,json_extract(payload,'$.kind'),
                    COALESCE(json_extract(payload,'$.finished_at'),json_extract(payload,'$.updated_at'),json_extract(payload,'$.created_at'),''),
                    json_remove(payload,'$.result','$.fingerprint','$.owner_pid','$.owner_started') FROM czsc_tasks;
            """)
            conn.commit()

    def _sync_history(self):
        # A live dashboard must also notice another process's dead owner.
        # Embedded legacy jobs have no PID identity; reconcile only journal adopters.
        now=time.monotonic()
        if hasattr(self.light, 'journal') and now-getattr(self,'_last_reconciled',0)>=5.:
            self.journal.reconcile()
            self._last_reconciled=now
        self.journal.import_legacy(self.light.store)
        # Compatibility for embedded callers that have not adopted durable receipts.
        if not hasattr(self.light, 'journal'):
            with self.light.lock, closing(self.journal.connection()) as conn, conn:
                for value in self.light.jobs.values(): self.journal._put(conn, value)

    @staticmethod
    def _public_task(row):
        item = {k:v for k,v in row.items() if k not in {'result','fingerprint','owner_pid','owner_started'}}
        item['can_delete'] = row.get('status') in TERMINAL
        item['delete_blocked_reason'] = None if item['can_delete'] else '任务运行中或状态未知，不能删除'
        return item

    def _event(self, plan_id, action, identity, status, payload):
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute('INSERT INTO cleanup_events(plan_id,action,identity,status,payload,created_at) VALUES(?,?,?,?,?,?)',
                         (plan_id, action, identity, status, canonical(payload), self.clock()))
            conn.commit()

    @contextmanager
    def _locks(self):
        # Same process ordering as task submission, then the store's writer lock.
        with self.lock, ExitStack() as stack:
            for obj, attr in ((self.tasks, '_lock'), (self.light, 'lock'), (self.light.store, '_lock')):
                lock = getattr(obj, attr, None)
                if lock is not None:
                    stack.enter_context(lock)
            yield

    @contextmanager
    def _compute_guard(self):
        lock = getattr(self.light, 'compute_lock', None)
        if lock is None:
            yield
            return
        from filelock import Timeout
        try:
            with lock.acquire(timeout=0):
                yield
        except Timeout as exc:
            raise MaintenanceConflict('另一个进程正在使用计算资源，请完成后再清理') from exc

    def _decode_job(self, row):
        store = self.light.store
        path = store._path(row['sha256'])
        content = path.read_bytes()
        if len(content) != row['size_bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('任务摘要 SHA/大小不匹配')
        value = json.loads(gzip.decompress(content))
        if store.cache_key({'job_id': value['job_id']}, 'jobs') != row['cache_key']:
            raise ValueError('任务摘要身份不匹配')
        return value

    def _all(self):
        rows = []
        with closing(read_connection(self.tasks.db_path)) as conn:
            for row in conn.execute('SELECT payload FROM czsc_tasks'):
                value = json.loads(row[0])
                rows.append({**value, 'backend': 'legacy'})
        store = self.light.store
        with store._lock:
            persisted = store.connection.execute('''SELECT r.*,o.size_bytes FROM results r JOIN objects o USING(sha256)
                WHERE r.namespace='jobs' AND o.state='active' ''').fetchall()
            merged = {}
            for row in persisted:
                try:
                    value = self._decode_job(row)
                    merged[value['job_id']] = value
                except (OSError, ValueError, KeyError, EOFError, TypeError):
                    continue  # Corrupt payloads are never deletion candidates.
        with closing(self.journal.connection()) as conn:
            for saved, in conn.execute('SELECT payload FROM jobs'):
                value=json.loads(saved); merged[value['job_id']]=value
        with self.light.lock:
            merged.update({key: deepcopy(value) for key, value in self.light.jobs.items()})
        rows.extend({**value, 'backend': 'light'} for value in merged.values())
        return rows

    def list(self, *, status='all', kind='all', backend='all', page=1, page_size=20, _conn=None):
        if status not in {'all', 'active', 'terminal'} | ACTIVE | TERMINAL:
            raise ValueError('任务状态筛选无效')
        if backend not in {'all', 'legacy', 'light'} or page < 1 or not 1 <= page_size <= 100:
            raise ValueError('任务分页或来源无效')
        if _conn is None: self._sync_history()
        with self._history_connection() if _conn is None else nullcontext(_conn) as conn:
            tables = [('legacy','legacy.task_display'),('light','light.jobs')]
            counts = {'total':0,'active':0,'terminal':0}
            for _, table in tables:
                for state,n in conn.execute('SELECT status,COUNT(*) FROM '+table+' GROUP BY status'):
                    counts['total']+=n
                    if state in ACTIVE: counts['active']+=n
                    elif state in TERMINAL: counts['terminal']+=n
            conditions, args = [], []
            if kind != 'all': conditions.append('kind=?'); args.append(kind)
            states = ACTIVE if status == 'active' else TERMINAL if status == 'terminal' else {status} if status != 'all' else None
            if states:
                conditions.append('status IN ('+','.join('?' for _ in states)+')'); args.extend(sorted(states))
            where=' WHERE '+' AND '.join(conditions) if conditions else ''
            selections=[]; select_args=[]; table_by_name=dict(tables)
            for name,table in tables:
                if backend not in ('all',name): continue
                selections.append("SELECT '"+name+"' AS backend,job_id,status,updated_at FROM "+table+where)
                select_args.extend(args)
            combined=' UNION ALL '.join(selections)
            total=conn.execute('SELECT COUNT(*) FROM ('+combined+')',select_args).fetchone()[0]
            rows=conn.execute('SELECT backend,job_id FROM ('+combined+
                ") ORDER BY status IN ('queued','running','cancelling') DESC,updated_at DESC,backend DESC,job_id DESC LIMIT ? OFFSET ?",
                [*select_args,page_size,(page-1)*page_size]).fetchall()
            out=[]
            for name,job_id in rows:
                payload=conn.execute('SELECT payload FROM '+table_by_name[name]+' WHERE job_id=?',(job_id,)).fetchone()[0]
                out.append(self._public_task({**json.loads(payload),'backend':name}))

        return {'items':out,'total':total,'page':page,'page_size':page_size,'counts':counts}

    @contextmanager
    def _history_connection(self):
        with closing(sqlite3.connect(self.db_path,timeout=30)) as conn:
            conn.execute('ATTACH DATABASE ? AS legacy',(str(self.tasks.db_path),))
            conn.execute('ATTACH DATABASE ? AS light',(str(self.journal.path),))
            conn.execute('BEGIN')
            try:
                yield conn
            finally:
                conn.rollback()

    def overview(self, **filters):
        self._sync_history()
        with self._history_connection() as conn:
            history=self.list(**filters,_conn=conn)
            current=self.list(status='active',kind=filters.get('kind','all'),backend=filters.get('backend','all'),page=1,page_size=100,_conn=conn)
        return {**history,'active_items':current['items'],'active_total':current['total']}

    @staticmethod
    def _refs(tasks):
        out = []
        for value in tasks:
            backend, job_id = value.get('backend'), value.get('job_id')
            if backend not in {'legacy', 'light'} or not isinstance(job_id, str) or not re.fullmatch(r'[\w.-]{1,160}', job_id):
                raise ValueError('任务标识无效')
            ref = {'backend': backend, 'job_id': job_id}
            if ref not in out:
                out.append(ref)
        if len(out) > 1000:
            raise ValueError('单次最多处理1000条任务')
        return out

    def _record_hash(self, row):
        return digest({k: v for k, v in row.items() if k != 'result'})

    def delete_records(self, refs, *, plan_id=None, expected=None):
        refs = self._refs(refs)
        outcome = {'deleted': [], 'skipped': [], 'failed': []}
        with self._locks():
            for ref in refs:
                identity = ref['backend'] + ':' + ref['job_id']
                try:
                    current = next((r for r in self._all() if r['backend'] == ref['backend'] and r['job_id'] == ref['job_id']), None)
                    if current is None:
                        outcome['skipped'].append({**ref, 'reason': '记录已不存在'}); continue
                    if current.get('status') not in TERMINAL:
                        outcome['skipped'].append({**ref, 'reason': '任务仍在运行或状态未知'}); continue
                    if expected is not None and expected.get(identity) != self._record_hash(current):
                        outcome['skipped'].append({**ref, 'reason': '任务记录在预览后改变'}); continue
                    saved = {k: v for k, v in current.items() if k != 'result'}
                    self._event(plan_id, 'delete_task', identity, 'prepared', saved)
                    if ref['backend'] == 'legacy':
                        with closing(sqlite3.connect(self.tasks.db_path, timeout=30)) as conn:
                            conn.execute('BEGIN IMMEDIATE')
                            raw = conn.execute('SELECT payload,status FROM czsc_tasks WHERE job_id=?', (ref['job_id'],)).fetchone()
                            if not raw or raw[1] not in TERMINAL or self._record_hash({**json.loads(raw[0]), 'backend': 'legacy'}) != self._record_hash(current):
                                raise MaintenanceConflict('任务状态已变化')
                            conn.execute('DELETE FROM czsc_tasks WHERE job_id=?', (ref['job_id'],)); conn.commit()
                    else:
                        store = self.light.store
                        key = store.cache_key({'job_id': ref['job_id']}, 'jobs')
                        with store._tx():
                            linked = store.connection.execute('SELECT namespace,sha256 FROM results WHERE cache_key=?', (key,)).fetchone()
                            if linked:
                                if linked['namespace'] != 'jobs' or self._cache_protected(key, linked['sha256'], ignore_result=True):
                                    raise MaintenanceConflict('任务摘要仍有固定引用或租约')
                                store.connection.execute('INSERT INTO result_retirements VALUES(?,?,?,?,?)',
                                                         (key, 'jobs', linked['sha256'], 'user_task_history_delete', self.clock()))
                                store.connection.execute('DELETE FROM results WHERE cache_key=?', (key,))
                        self.journal.delete(ref['job_id'])
                        self.light.jobs.pop(ref['job_id'], None)
                        self.light.stops.pop(ref['job_id'], None)
                    self._event(plan_id, 'delete_task', identity, 'deleted', saved)
                    outcome['deleted'].append(ref)
                except MaintenanceConflict as exc:
                    outcome['skipped'].append({**ref, 'reason': str(exc)})
                except Exception as exc:
                    outcome['failed'].append({**ref, 'reason': str(exc)})
                    self._event(plan_id, 'delete_task', identity, 'failed', {'error': str(exc)})
        return outcome

    def _cache_protected(self, key, sha, *, ignore_result=False):
        conn = self.light.store.connection
        shared = conn.execute('SELECT cache_key FROM results WHERE sha256=?', (sha,)).fetchall()
        if any(row[0] != key for row in shared):
            return '共享 SHA 仍被其他结果引用'
        if not ignore_result and shared:
            return '结果索引仍存在'
        if conn.execute('SELECT 1 FROM pins WHERE cache_key=?', (key,)).fetchone():
            return '固定结果引用'
        if conn.execute('SELECT 1 FROM leases WHERE cache_key=? AND expires_at>?', (key, self.clock())).fetchone():
            return '有效租约'
        for row in conn.execute('SELECT value_json FROM settings'):
            if key in row[0] or sha in row[0]:
                return '账户、模型或设置仍引用该对象'
        return None

    def _manifest(self, directory):
        if directory.is_symlink() or directory.parent.resolve() != self.runs:
            raise ValueError('产物目录越界或符号链接')
        permitted = {'metadata.json', 'reports/czsc.json', '.DS_Store', 'reports/.DS_Store'}
        files = []
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ValueError('产物包含符号链接')
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            if relative not in permitted or path.stat().st_nlink != 1:
                raise ValueError('目录包含其他资产或共享硬链接')
            files.append({'relative_path': relative, 'sha256': file_digest(path), 'size_bytes': path.stat().st_size})
        if not any(r['relative_path'] == 'reports/czsc.json' for r in files):
            raise ValueError('分析报告缺失')
        report = json.loads((directory / 'reports/czsc.json').read_text())
        if report.get('run_id') != directory.name:
            raise ValueError('分析报告运行身份不匹配')
        return files

    def _analysis_candidates(self, selected=None):
        candidates, protected = [], []
        with closing(read_connection(self.results_db)) as conn:
            rows = conn.execute("SELECT run_id,kind,report_path FROM czsc_runs WHERE kind='analysis'").fetchall()
        for row in rows:
            rid = row['run_id']
            if selected is not None and rid not in selected:
                continue
            base = {'id': 'analysis:' + rid, 'type': 'analysis_run', 'run_id': rid}
            try:
                if not re.fullmatch(r'[\w.-]+', rid) or rid in {'.', '..'} or row['report_path'] != rid + '/reports/czsc.json':
                    raise ValueError('结果路径不是专属分析报告')
                path = self.runs / rid
                files = self._manifest(path)
                candidates.append({**base, 'path': str(path), 'files': files,
                                   'size_bytes': sum(f['size_bytes'] for f in files), 'reason': '无依赖的独立历史分析报告'})
            except (OSError, ValueError, TypeError) as exc:
                protected.append({**base, 'reason': str(exc)})
        return candidates, protected

    def _database_paths(self):
        paths = {self.results_db, Path(self.tasks.db_path).resolve()}
        for base in (self.root, self.root.parent / 'processed'):
            if base.is_dir():
                paths.update(p.resolve() for pattern in ('*.db', '*.sqlite', '*.sqlite3') for p in base.rglob(pattern))
        return sorted(p for p in paths if p != self.db_path and p.stat().st_size)

    @contextmanager
    def _frozen_indexes(self):
        """Exclude every index writer across processes during graph + retirement."""
        connections = {}
        try:
            for path in self._database_paths():
                conn = sqlite3.connect(path, timeout=30)
                connections[path] = conn
                conn.execute('BEGIN IMMEDIATE')
            yield connections
            for conn in connections.values():
                conn.commit()
        except BaseException:
            for conn in connections.values():
                conn.rollback()
            raise
        finally:
            for conn in connections.values():
                conn.close()

    def _reference_snapshot(self, candidates, ignored_tasks=()):
        """Inspect typed indexes and JSON control files; native numeric blobs are leaves.

        Native/board/model/source manifests and all SQLite TEXT columns are
        scanned. Native OHLC/structural JSON.gz leaves contain no run paths by
        contract and are never garbage-collected here. Any unreadable control
        source blocks all physical analysis retirement.
        """
        ids = {c['run_id'] for c in candidates}
        if not ids:
            return {'protected': {}, 'errors': [], 'sources': 0, 'digest': digest([])}
        terms = {}
        for candidate in candidates:
            for term in [candidate['run_id'], *[f['sha256'] for f in candidate['files']]]:
                terms.setdefault(term, set()).add(candidate['run_id'])
        custom = [t for t in terms if not re.fullmatch(r'[a-f0-9]{64}|[0-9]{8}-[\w.-]+', t)]
        # Extract identities once; hundreds of literal alternatives over numeric
        # history JSON cause avoidable O(content * candidate_count) scans.
        matcher = re.compile(r'(?<![A-Za-z0-9_.-])(?:[0-9]{8}-[\w.-]+|[a-f0-9]{64}' + (('|' + '|'.join(re.escape(t) for t in custom)) if custom else '') + r')')
        protected, errors, stamp, count = {}, [], hashlib.sha256(), 0
        ignored = {(r['backend'], r['job_id']) for r in ignored_tasks}
        def consume(text, source, own=None):
            nonlocal count
            count += 1; stamp.update(source.encode()); stamp.update(hashlib.sha256(text.encode()).digest())
            for found in matcher.finditer(text):
                for target in terms.get(found.group(0), ()):
                    if target != own:
                        protected.setdefault(target, set()).add(source)
        # Catalog every application metadata/processed DB; never open raw quote DBs.
        data_root = self.root.parent
        databases = self._database_paths()
        for path in sorted(databases):
            if path == self.db_path or path.stat().st_size == 0:
                continue
            try:
                with closing(read_connection(path)) as conn:
                    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                    for table in tables:
                        columns = [r[1] for r in conn.execute('PRAGMA table_info(' + quote(table) + ')') if any(t in r[2].upper() for t in ('TEXT', 'CHAR', 'CLOB'))]
                        if not columns:
                            continue
                        for values in conn.execute('SELECT ' + ','.join(quote(c) for c in columns) + ' FROM ' + quote(table)):
                            row = dict(zip(columns, values)); own = None
                            if path == self.journal.path.resolve() and table == 'jobs' and ('light', row.get('job_id')) in ignored:
                                continue
                            if path == self.results_db and table == 'czsc_runs' and row.get('run_id') in ids:
                                own = row['run_id']
                            if path == Path(self.tasks.db_path).resolve() and table in {'czsc_tasks','task_display'} and ('legacy', row.get('job_id')) in ignored:
                                continue
                            text = '\n'.join(str(v) for v in values if v is not None)
                            consume(text, f'{path.name}:{table}', own)
            except (OSError, sqlite3.Error, ValueError) as exc:
                errors.append(f'{path}: {exc}')
        # JSON references in frozen reports, calendar ancestors, model manifests,
        # board checkpoints and metadata. Stream large reports with overlap.
        files = set()
        for base in (self.runs, self.root, data_root / 'processed'):
            if base.is_dir():
                files.update(p for p in base.rglob('*.json') if '.task-cleanup' not in p.parts)
        for path in sorted(files):
            try:
                if path.is_symlink():
                    raise ValueError('引用来源是符号链接')
                own = next((p for p in path.relative_to(self.runs).parts[:1] if p in ids), None) if path.is_relative_to(self.runs) else None
                with path.open('r', encoding='utf-8') as stream:
                    carry = ''
                    while chunk := stream.read(1024 * 1024):
                        text = carry + chunk
                        consume(text, str(path), own)
                        carry = text[-512:]
            except (OSError, ValueError) as exc:
                errors.append(f'{path}: {exc}')
        # Research CAS can carry saved account/source dependencies. Never use get()
        # here, because preview must not refresh its cache timestamps.
        store = self.light.store
        with store._lock:
            for row in store.connection.execute('SELECT r.cache_key,r.namespace,r.identity_json,r.sha256,o.size_bytes FROM results r JOIN objects o USING(sha256)'):
                try:
                    if row['namespace'] == 'jobs':
                        identity = json.loads(row['identity_json'])
                        if ('light', identity.get('job_id')) in ignored:
                            continue
                    path = store._path(row['sha256']); payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != row['sha256']:
                        raise ValueError('CAS引用 SHA 不匹配')
                    consume(gzip.decompress(payload).decode(), 'research:' + row['cache_key'])
                except (OSError, ValueError, EOFError) as exc:
                    errors.append(f'research:{row["cache_key"]}: {exc}')
        return {'protected': {k: sorted(set(v))[:20] for k, v in protected.items()}, 'errors': errors,
                'sources': count, 'digest': stamp.hexdigest()}

    def preview(self, tasks=None, *, include_artifacts=True):
        explicit = bool(tasks)
        refs = self._refs(tasks or [])
        with self._locks():
            rows = self._all()
            if not explicit:
                refs = [{'backend': r['backend'], 'job_id': r['job_id']} for r in rows if r.get('status') in TERMINAL]
            selected = {(r['backend'], r['job_id']) for r in refs}
            chosen = [r for r in rows if (r['backend'], r['job_id']) in selected]
            candidates, protected = [], []
            for row in chosen:
                ref = {'backend': row['backend'], 'job_id': row['job_id']}
                base = {**ref, 'id': row['backend'] + ':' + row['job_id'], 'type': 'task_record', 'size_bytes': 0}
                if row.get('status') not in TERMINAL:
                    protected.append({**base, 'reason': '任务仍在运行或状态未知'}); continue
                candidates.append({**base, 'record_sha256': self._record_hash(row), 'reason': '终态任务记录；业务结果保留'})
            snapshot = {'protected': {}, 'errors': [], 'sources': 0}
            if include_artifacts:
                if any(r.get('status') in ACTIVE for r in rows):
                    protected.append({'type': 'artifacts', 'reason': '存在活跃任务，保留全部产物'})
                else:
                    analysis, blocked = self._analysis_candidates({r.get('run_id') for r in chosen} if explicit else None)
                    protected.extend(blocked)
                    snapshot = self._reference_snapshot(analysis, refs)
                    for item in analysis:
                        reasons = snapshot['protected'].get(item['run_id'])
                        if snapshot['errors'] or reasons:
                            protected.append({**{k: v for k, v in item.items() if k != 'files'}, 'reason': '引用证据不完整' if snapshot['errors'] else '仍被来源、模型、账户或其他结果引用', 'references': reasons or []})
                        else:
                            candidates.append(item)
                    store = self.light.store
                    for row in chosen:
                        if row['backend'] != 'light' or row.get('status') not in TERMINAL:
                            continue
                        key = store.cache_key({'job_id': row['job_id']}, 'jobs')
                        cached = store.connection.execute('SELECT r.sha256,o.size_bytes FROM results r JOIN objects o USING(sha256) WHERE r.cache_key=?', (key,)).fetchone()
                        if cached and not self._cache_protected(key, cached['sha256'], ignore_result=True):
                            path = store._path(cached['sha256'])
                            if path.is_file() and file_digest(path) == cached['sha256']:
                                candidates.append({'id': 'job-object:' + cached['sha256'], 'type': 'cache_job_object', 'cache_key': key,
                                    'sha256': cached['sha256'], 'path': str(path), 'size_bytes': cached['size_bytes'], 'reason': '已删除终态任务的独占摘要对象'})
            plan_id = uuid.uuid4().hex
            plan = {'schema': SCHEMA, 'plan_id': plan_id, 'created_at': self.clock(), 'expires_at': self.clock() + 3600,
                    'candidates': candidates, 'protected': protected, 'candidate_bytes': sum(c['size_bytes'] for c in candidates),
                    'protection_snapshot': snapshot,
                    'protected_classes': ['原始行情', 'Kronos与评分模型', '保存账户', '关注', '活跃任务', '原生/板块/训练/日历来源', '私有审计']}
            sha = digest(plan)
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute('INSERT INTO cleanup_plans VALUES(?,?,?,?)', (plan_id, sha, canonical(plan), self.clock())); conn.commit()
            return {**plan, 'plan_sha256': sha}

    def execute(self, plan_id, plan_sha256):
        with self._locks(), self._compute_guard():
            with closing(read_connection(self.db_path)) as conn:
                row = conn.execute('SELECT sha256,payload FROM cleanup_plans WHERE plan_id=?', (plan_id,)).fetchone()
            if not row:
                raise KeyError(plan_id)
            plan = json.loads(row['payload'])
            if row['sha256'] != plan_sha256 or digest(plan) != plan_sha256:
                raise MaintenanceConflict('清理计划身份不匹配')
            if plan['expires_at'] < self.clock():
                raise MaintenanceConflict('清理计划已过期，请重新预览')
            with closing(read_connection(self.db_path)) as conn:
                finished = conn.execute("SELECT payload FROM cleanup_events WHERE plan_id=? AND action='execute_finished' ORDER BY event_id DESC LIMIT 1", (plan_id,)).fetchone()
                started = conn.execute("SELECT 1 FROM cleanup_events WHERE plan_id=? AND action='execute'", (plan_id,)).fetchone()
            if finished:
                return {**json.loads(finished[0]), 'replayed': True}
            if started:
                recovered = self.recover(plan_id, plan_sha256)
                raise MaintenanceConflict('中断清理已恢复可恢复产物，请重新预览；恢复失败数：' + str(len(recovered['failed'])))
            self._event(plan_id, 'execute', plan_id, 'started', {'candidate_count': len(plan['candidates'])})
            records = [c for c in plan['candidates'] if c['type'] == 'task_record']
            result = self.delete_records([{'backend': c['backend'], 'job_id': c['job_id']} for c in records], plan_id=plan_id,
                                         expected={c['id']: c['record_sha256'] for c in records})
            result.update(plan_id=plan_id, deleted_bytes=0)
            artifacts = [c for c in plan['candidates'] if c['type'] != 'task_record']
            staged = []
            try:
                with self._frozen_indexes() as indexes:
                    active = any(r.get('status') in ACTIVE for r in self._all())
                    analysis = [c for c in artifacts if c['type'] == 'analysis_run']
                    snapshot = self._reference_snapshot(analysis) if not active else {'errors': ['存在活跃任务'], 'protected': {}}
                    for item in analysis:
                        try:
                            if active or snapshot['errors'] or item['run_id'] in snapshot['protected']:
                                raise MaintenanceConflict('存在活跃任务、引用状态已变化或引用证据不完整')
                            quarantine = self._stage_analysis(plan_id, item, indexes[self.results_db])
                            staged.append((item, quarantine))
                        except MaintenanceConflict as exc:
                            result['skipped'].append({'id': item['id'], 'reason': str(exc)})
                            self._event(plan_id, 'delete_artifact', item['id'], 'skipped', {'reason': str(exc)})
                        except Exception as exc:
                            result['failed'].append({'id': item['id'], 'reason': str(exc)})
                            self._event(plan_id, 'delete_artifact', item['id'], 'failed', {'reason': str(exc)})
            except BaseException:
                for item, quarantine in reversed(staged):
                    if quarantine.exists() and not Path(item['path']).exists():
                        os.replace(quarantine, item['path'])
                raise
            for item, quarantine in staged:
                try:
                    self._event(plan_id, 'delete_artifact', item['id'], 'retired', {'quarantine': str(quarantine)})
                    shutil.rmtree(quarantine)
                    self._event(plan_id, 'delete_artifact', item['id'], 'deleted', item)
                    result['deleted'].append({k: v for k, v in item.items() if k != 'files'})
                    result['deleted_bytes'] += item['size_bytes']
                except Exception as exc:
                    result['failed'].append({'id': item['id'], 'reason': str(exc)})
                    self._event(plan_id, 'delete_artifact', item['id'], 'failed', {'reason': str(exc)})
            for item in [c for c in artifacts if c['type'] == 'cache_job_object']:
                try:
                    if any(r.get('status') in ACTIVE for r in self._all()):
                        raise MaintenanceConflict('存在活跃任务，保留产物')
                    self._delete_job_object(plan_id, item)
                    result['deleted'].append(item)
                    result['deleted_bytes'] += item['size_bytes']
                except MaintenanceConflict as exc:
                    result['skipped'].append({'id': item['id'], 'reason': str(exc)})
                    self._event(plan_id, 'delete_artifact', item['id'], 'skipped', {'reason': str(exc)})
                except Exception as exc:
                    result['failed'].append({'id': item['id'], 'reason': str(exc)})
                    self._event(plan_id, 'delete_artifact', item['id'], 'failed', {'reason': str(exc)})
            self._event(plan_id, 'execute_finished', plan_id, 'completed', result)
            return result

    def recover(self, plan_id, plan_sha256):
        """Restore interrupted quarantines; never discard a partially retired run."""
        with self._locks(), self._compute_guard():
            with closing(read_connection(self.db_path)) as conn:
                row = conn.execute('SELECT sha256,payload FROM cleanup_plans WHERE plan_id=?', (plan_id,)).fetchone()
                prepared = conn.execute("SELECT identity,payload FROM cleanup_events WHERE plan_id=? AND action='delete_artifact' AND status='prepared' ORDER BY event_id", (plan_id,)).fetchall()
            if not row or row['sha256'] != plan_sha256 or digest(json.loads(row['payload'])) != plan_sha256:
                raise MaintenanceConflict('清理计划身份不匹配')
            result = {'plan_id': plan_id, 'restored': [], 'failed': []}
            with self._frozen_indexes() as indexes:
                for event in prepared:
                    item = json.loads(event['payload'])
                    if item.get('type') != 'analysis_run':
                        continue
                    quarantine = self._quarantine_path(plan_id, item['run_id'])
                    target = self.runs / item['run_id']
                    if not quarantine.exists():
                        continue
                    try:
                        if str(quarantine) != item.get('quarantine') or target.exists() or quarantine.is_symlink():
                            raise MaintenanceConflict('隔离路径冲突，保留现场')
                        observed = []
                        for path in sorted(quarantine.rglob('*')):
                            if path.is_symlink():
                                raise MaintenanceConflict('隔离产物含符号链接')
                            if path.is_file():
                                observed.append({'relative_path': path.relative_to(quarantine).as_posix(), 'sha256': file_digest(path), 'size_bytes': path.stat().st_size})
                        if observed != item['files']:
                            raise MaintenanceConflict('隔离产物改变，保留现场')
                        conn = indexes[self.results_db]
                        old = conn.execute('SELECT * FROM czsc_runs WHERE run_id=?', (item['run_id'],)).fetchone()
                        if old is not None and list(old) != item['index_row']:
                            raise MaintenanceConflict('结果索引冲突，保留现场')
                        if old is None:
                            conn.execute('INSERT INTO czsc_runs VALUES(?,?,?,?,?,?)', item['index_row'])
                        os.replace(quarantine, target)
                        result['restored'].append(item['run_id'])
                        self._event(plan_id, 'recover', item['id'], 'restored', {'path': str(target)})
                    except Exception as exc:
                        result['failed'].append({'id': item['id'], 'reason': str(exc)})
            return result

    def _quarantine_path(self, plan_id, run_id):
        path = self.runs / '.task-cleanup' / plan_id / run_id
        cursor = self.runs
        for part in path.relative_to(self.runs).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise MaintenanceConflict('隔离路径包含符号链接')
        if not path.resolve().is_relative_to(self.runs):
            raise MaintenanceConflict('隔离路径越界')
        return path

    def _stage_analysis(self, plan_id, item, conn):
        path = self.runs / item['run_id']
        if str(path) != item['path'] or self._manifest(path) != item['files']:
            raise MaintenanceConflict('产物内容在预览后改变')
        quarantine = self._quarantine_path(plan_id, item['run_id'])
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        row = conn.execute('SELECT * FROM czsc_runs WHERE run_id=?', (item['run_id'],)).fetchone()
        if not row or row[1] != 'analysis' or row[-1] != item['run_id'] + '/reports/czsc.json':
            raise MaintenanceConflict('分析索引在预览后改变')
        self._event(plan_id, 'delete_artifact', item['id'], 'prepared', {**item, 'quarantine': str(quarantine), 'index_row': list(row)})
        os.replace(path, quarantine)
        try:
            conn.execute('DELETE FROM czsc_runs WHERE run_id=?', (item['run_id'],))
        except BaseException:
            os.replace(quarantine, path)
            raise
        return quarantine

    def _delete_job_object(self, plan_id, item):
        store = self.light.store
        with store._tx():
            reason = self._cache_protected(item['cache_key'], item['sha256'])
            if reason:
                raise MaintenanceConflict(reason)
            row = store.connection.execute('SELECT size_bytes,state FROM objects WHERE sha256=?', (item['sha256'],)).fetchone()
            path = store._path(item['sha256'])
            if not row or row['state'] != 'active' or row['size_bytes'] != item['size_bytes'] or str(path) != item['path'] or file_digest(path) != item['sha256']:
                raise MaintenanceConflict('摘要对象在预览后改变')
            self._event(plan_id, 'delete_artifact', item['id'], 'prepared', item)
            store.connection.execute("INSERT INTO gc_journal(sha256,reason,state,created_at) VALUES(?,?,'pending',?)", (item['sha256'], 'task_cleanup:' + plan_id, self.clock()))
            store.connection.execute("UPDATE objects SET state='deleting' WHERE sha256=?", (item['sha256'],))
        with store._tx():
            if store.connection.execute('SELECT 1 FROM results WHERE sha256=?', (item['sha256'],)).fetchone():
                raise MaintenanceConflict('共享对象获得新引用')
            path.chmod(path.stat().st_mode | 0o200); path.unlink()
            store.connection.execute('DELETE FROM objects WHERE sha256=?', (item['sha256'],))
            store.connection.execute("UPDATE gc_journal SET state='deleted',completed_at=? WHERE sha256=? AND state='pending'", (self.clock(), item['sha256']))
        self._event(plan_id, 'delete_artifact', item['id'], 'deleted', item)
