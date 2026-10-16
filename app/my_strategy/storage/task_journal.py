"""Mutable, durable receipts; no executable work is restored on startup."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import psutil
from my_strategy.core.tz import local_now

ACTIVE = ('queued', 'running', 'cancelling')
TERMINAL = ('succeeded', 'partial', 'failed', 'blocked', 'cancelled', 'interrupted')


def summary(value):
    out = {k: v for k, v in value.items() if k != 'result'}
    result = value.get('result') or {}
    if result:
        out['result_metadata']={key:result[key] for key in ('model_id','model_ref','manifest_id','manifest_ref','reused','saved','result_ref') if key in result}
    if result.get('result_id'):
        out['result_id'] = result['result_id']
        out.setdefault('result_ref', {'type':'simulation','result_id':result['result_id']})
    return out


class TaskJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner = {'owner_pid': os.getpid(), 'owner_started': psutil.Process().create_time()}
        with closing(self.connection()) as conn, conn:
            conn.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, fingerprint TEXT, status TEXT NOT NULL,
                    kind TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS active_fingerprint ON jobs(fingerprint)
                    WHERE status IN ('queued','running','cancelling') AND fingerprint IS NOT NULL;
                CREATE INDEX IF NOT EXISTS jobs_page ON jobs(status,kind,updated_at DESC,job_id);
                CREATE INDEX IF NOT EXISTS jobs_recent ON jobs(updated_at DESC,job_id);
                CREATE TABLE IF NOT EXISTS imports(cache_key TEXT PRIMARY KEY,sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS job_outputs(job_id TEXT NOT NULL,namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,value_sha256 TEXT NOT NULL,phase TEXT NOT NULL,
                    object_sha256 TEXT,created_at TEXT NOT NULL,
                    PRIMARY KEY(job_id,namespace,cache_key,value_sha256));
            ''')
        self.reconcile()

    def connection(self):
        return sqlite3.connect(self.path, timeout=30)

    def _put(self, conn, job):
        value = summary(job)
        when = str(value.get('finished_at') or value.get('updated_at') or value.get('created_at') or '')
        conn.execute('''INSERT INTO jobs VALUES(?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET
            fingerprint=excluded.fingerprint,status=excluded.status,kind=excluded.kind,
            updated_at=excluded.updated_at,payload=excluded.payload''',
            (value['job_id'],value.get('fingerprint'),value['status'],value['kind'],when,
             json.dumps(value,ensure_ascii=False,allow_nan=False)))

    @staticmethod
    def _owner_live(job):
        try:
            return abs(psutil.Process(job.get('owner_pid', -1)).create_time() - job.get('owner_started', 0)) < .001
        except (psutil.Error, ValueError, TypeError):
            return False

    def _interrupt(self, conn, job):
        job.update(status='interrupted', finished_at=local_now().isoformat(),
            error='任务所有者已退出；保留最后进度与检查点，未自动恢复', interrupted_from=job['status'])
        self._put(conn, job)

    def reconcile(self):
        with closing(self.connection()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            for payload, in conn.execute("SELECT payload FROM jobs WHERE status IN ('queued','running','cancelling')"):
                job = json.loads(payload)
                if not self._owner_live(job):
                    self._interrupt(conn, job)

    def create(self, job):
        with closing(self.connection()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            found = conn.execute("SELECT payload FROM jobs WHERE fingerprint=? AND status IN ('queued','running','cancelling')", (job['fingerprint'],)).fetchone()
            if found:
                previous=json.loads(found[0])
                if self._owner_live(previous):return previous, False
                # An explicit new submit may proceed after another owner died;
                # it creates a distinct receipt, never resumes the old work.
                self._interrupt(conn, previous)
            job = {**deepcopy(job), **self.owner, 'attempt': 1, 'updated_at': local_now().isoformat()}
            self._put(conn, job)
            return job, True

    def update(self, job):
        with closing(self.connection()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT status FROM jobs WHERE job_id=?', (job['job_id'],)).fetchone()
            # A worker cannot overwrite a durable remote cancellation with progress.
            if row and row[0] == 'cancelling':
                if job['status'] in ('queued','running'): job['status']='cancelling'
                elif job['status']=='succeeded': job['status']='cancelled'
                job['cancel_requested']=True
            job['updated_at'] = local_now().isoformat()
            job['heartbeat_at'] = job['updated_at']
            self._put(conn, job)

    def heartbeat(self, job_id):
        now=local_now().isoformat()
        with closing(self.connection()) as conn, conn:
            conn.execute("""UPDATE jobs SET payload=json_set(payload,'$.heartbeat_at',?)
                WHERE job_id=? AND status IN ('queued','running','cancelling')
                AND json_extract(payload,'$.owner_pid')=? AND json_extract(payload,'$.owner_started')=?""",
                (now,job_id,self.owner['owner_pid'],self.owner['owner_started']))

    def output_intent(self, job_id, namespace, key, value_sha256):
        with closing(self.connection()) as conn, conn:
            conn.execute("""INSERT OR IGNORE INTO job_outputs VALUES(?,?,?,?, 'intended',NULL,?)""",
                (job_id,namespace,key,value_sha256,local_now().isoformat()))

    def output_committed(self, job_id, namespace, key, value_sha256, reference):
        with closing(self.connection()) as conn, conn:
            conn.execute("""UPDATE job_outputs SET phase='committed',object_sha256=?
                WHERE job_id=? AND namespace=? AND cache_key=? AND value_sha256=?""",
                (reference.get('sha256'),job_id,namespace,key,value_sha256))

    def recovered_outputs(self, job_id, store):
        """Report verified publications; never infer job success or resume work."""
        from my_strategy.storage.research_store import canonical
        with closing(self.connection()) as conn:
            rows=conn.execute('''SELECT namespace,cache_key,value_sha256,phase,object_sha256
                FROM job_outputs WHERE job_id=? ORDER BY created_at,namespace,cache_key''',(job_id,)).fetchall()
        output=[]
        for namespace,key,expected,phase,object_sha in rows:
            item={'namespace':namespace,'cache_key':key,'value_sha256':expected,'publication_phase':phase}
            try:
                with store._lock:
                    row=store.connection.execute('SELECT namespace,sha256 FROM results WHERE cache_key=?',(key,)).fetchone()
                if row is None:
                    item['availability']='not_committed_or_retired'
                elif row['namespace']!=namespace or (object_sha and object_sha!=row['sha256']):
                    item['availability']='identity_mismatch'
                else:
                    value=store.get_by_key(key,touch=False)
                    if value is None:item['availability']='expired_or_unavailable'
                    elif hashlib.sha256(canonical(value).encode()).hexdigest()!=expected:item['availability']='value_mismatch'
                    else:item.update(availability='verified',sha256=row['sha256'])
            except (OSError,ValueError,KeyError,TypeError,EOFError) as exc:
                item.update(availability='unverifiable',error=str(exc))
            output.append(item)
        return output

    def get(self, job_id):
        with closing(self.connection()) as conn:
            row = conn.execute('SELECT payload FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def cancel(self, job_id):
        with closing(self.connection()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT payload FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if not row: raise KeyError(job_id)
            job = json.loads(row[0])
            if job['status'] in ACTIVE:
                job.update(status='cancelling', cancel_requested=True, updated_at=local_now().isoformat())
                self._put(conn, job)
            return job

    def recent(self, limit=30):
        with closing(self.connection()) as conn:
            return [json.loads(r[0]) for r in conn.execute('SELECT payload FROM jobs ORDER BY updated_at DESC,job_id DESC LIMIT ?', (limit,))]

    def delete(self, job_id):
        with closing(self.connection()) as conn, conn:
            deleted=conn.execute("DELETE FROM jobs WHERE job_id=? AND status NOT IN ('queued','running','cancelling')", (job_id,)).rowcount
            if deleted:conn.execute('DELETE FROM job_outputs WHERE job_id=?',(job_id,))

    def import_legacy(self, store):
        """Decode each old CAS summary once; new receipts never need CAS polling."""
        with closing(self.connection()) as conn, conn:
            conn.execute('ATTACH DATABASE ? AS cache', (str(store.root / 'research.db'),))
            rows = conn.execute('''SELECT r.cache_key,r.sha256 FROM cache.results r
                JOIN cache.objects o USING(sha256) LEFT JOIN imports i USING(cache_key)
                WHERE r.namespace='jobs' AND o.state='active' AND (i.sha256 IS NULL OR i.sha256<>r.sha256)''').fetchall()
            for key, sha in rows:
                try:
                    value = store.get_by_key(key,touch=False)
                    if (isinstance(value, dict) and value.get('status') in TERMINAL and value.get('job_id')
                            and store.cache_key({'job_id':value['job_id']}, 'jobs') == key
                            and not conn.execute('SELECT 1 FROM jobs WHERE job_id=?', (value['job_id'],)).fetchone()):
                        self._put(conn, value)
                except (OSError, ValueError, KeyError, TypeError, EOFError):
                    pass
                conn.execute('INSERT OR REPLACE INTO imports VALUES(?,?)', (key,sha))


class JournalStop:
    """Combine local cancellation with a bounded cross-process receipt check."""
    def __init__(self, journal, job_id, event):
        self.journal,self.job_id,self.event = journal,job_id,event
        self.checked = 0.
    def set(self): self.event.set()
    def is_set(self):
        if self.event.is_set(): return True
        now = time.monotonic()
        if now - self.checked >= .5:
            self.checked = now
            job = self.journal.get(self.job_id)
            if job and (job.get('cancel_requested') or job['status'] == 'cancelling'):
                self.event.set()
        return self.event.is_set()


def bind_publication_intents(store, journal):
    """Bind only this thread's final business publications to durable receipts.

    The intent commits BEFORE the result publication. An exit between the two
    commits is resolved by verifying the result's identity, CAS SHA and expected
    canonical value SHA. No cross-database atomicity is assumed. Inner source/
    prediction-cache writes of another job kind are excluded from this journal.
    """
    import threading
    from my_strategy.storage.research_store import canonical
    context=threading.local()
    original=store.put_many
    kinds={'prediction':{'predictions'},'score_training':{'score_models','mlp_score_models'},
           'tuning':{'tuning'},'simulation':{'simulations'},
           'score_observations':{'score_observation_manifests'},
           'incremental':{'incremental_plan_results'},'daily':{'simulations'}}
    def publish(items, **options):
        job=getattr(context,'job',None)
        if job is None:return original(items,**options)
        items=list(items);intents=[]
        namespaces=kinds.get(job['kind'],set())|{'job_results'}
        for index,item in enumerate(items):
            namespace=item.get('namespace','default')
            if namespace not in namespaces:continue
            key=store.cache_key(item['identity'],namespace)
            value_sha=hashlib.sha256(canonical(item['result']).encode()).hexdigest()
            journal.output_intent(job['job_id'],namespace,key,value_sha)
            intents.append((index,namespace,key,value_sha))
        references=original(items,**options)
        for index,namespace,key,value_sha in intents:
            journal.output_committed(job['job_id'],namespace,key,value_sha,references[index])
        return references
    store.put_many=publish
    return context
