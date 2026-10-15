"""Precisely retire four unreferenced terminal training wrappers, never children.

Dry-run records complete file identities. Apply rechecks task outcomes, active
users, model/result/account references and every file before deleting anything.
The small immutable receipt retains failures and outbound dependency names.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import time

from my_strategy.scripts.migrate_native_storage import atomic_json
from my_strategy.services.czsc_native_cache import file_sha256

RUN_IDS = (
    '20261014-111217-015050-20261012-web-model-training-NO_GIT',
    '20261014-131542-678456-20261012-web-model-training-NO_GIT',
    '20261014-205852-411927-20261012-web-model-training-NO_GIT',
    '20261014-223500-914001-20261012-web-model-training-NO_GIT',
)


def _read_db(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def dependency_check(app, run_ids):
    strategy = app/'my_strategy'; meta = strategy/'data/metadata'
    taskdb = _read_db(meta/'czsc_tasks.db')
    tasks = [json.loads(row[0]) for row in taskdb.execute('SELECT payload FROM czsc_tasks')]
    taskdb.close()
    records = {}
    for run in run_ids:
        own = [t for t in tasks if t.get('run_id') == run]
        if not own or any(t.get('kind') != 'model_training' or t.get('status') not in ('failed','cancelled') for t in own):
            raise ValueError('wrapper has absent, active, succeeded or recovered task: '+run)
        records[run] = [{'job_id': t['job_id'], 'status': t['status'], 'error': t.get('error'),
                        'children': (t.get('progress_detail') or {}).get('children', {})} for t in own]
    references = []
    for task in tasks:
        if task.get('status') in ('running','queued','cancelling'):
            content = json.dumps(task, ensure_ascii=False)
            for run in run_ids:
                if run in content: references.append({'kind':'active_task','id': task['job_id'],'run':run})
    # Current selection, successful model manifests, immutable result index and
    # the new store's pinned results/account continuation pointers are roots.
    files = list(meta.glob('*selection*.json*'))
    runs = strategy/'artifacts/runs'
    files += list(runs.glob('*/models/*/manifest.json'))
    files += list(runs.glob('*/reports/reference_training.json'))
    for file in files:
        content = file.read_text(encoding='utf-8')
        for run in run_ids:
            if run in content: references.append({'kind':'model_selection_or_manifest','path':str(file),'run':run})
    results = _read_db(strategy/'data/processed/czsc/results.db')
    for row in results.execute('SELECT run_id,summary,report_path FROM czsc_runs'):
        for run in run_ids:
            if run in '\n'.join(str(v) for v in row): references.append({'kind':'saved_result','id':row['run_id'],'run':run})
    results.close()
    research = strategy/'data/processed/research/research.db'
    if research.exists():
        db = _read_db(research)
        names = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if {'results','pins'} <= names:
            for row in db.execute('SELECT DISTINCT r.cache_key,r.identity_json,r.sha256 FROM results r JOIN pins p USING(cache_key)'):
                sha = row['sha256']
                if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha): raise ValueError('pinned content identity invalid')
                path = research.parent/'objects'/sha[:2]/(sha+'.json.gz')
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != sha: raise ValueError('pinned result content SHA drift')
                body = gzip.decompress(content).decode('utf-8')
                for run in run_ids:
                    if run in row['identity_json'] or run in body: references.append({'kind':'pinned_result','id':row['cache_key'],'run':run})
        if 'settings' in names:
            for row in db.execute('SELECT name,value_json FROM settings'):
                for run in run_ids:
                    if run in row['value_json']: references.append({'kind':'setting_or_account','id':row['name'],'run':run})
        db.close()
    import psutil
    roots = [str((runs/run).resolve())+'/' for run in run_ids]
    for process in psutil.process_iter(['pid']):
        try:
            for opened in process.open_files():
                if any(opened.path.startswith(root) for root in roots): references.append({'kind':'open_file','pid':process.pid,'path':opened.path})
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    if references: raise ValueError('wrapper is referenced: '+json.dumps(references,ensure_ascii=False))
    return records


def inspect(app):
    app = Path(app).resolve(); runs = app/'my_strategy/artifacts/runs'
    records = dependency_check(app, RUN_IDS)
    candidates = []
    for run in RUN_IDS:
        root = runs/run
        if root.is_symlink() or not root.is_dir() or not root.resolve().is_relative_to(runs.resolve()):
            raise ValueError('exact wrapper target missing or unsafe: '+run)
        files = []
        for path in sorted(root.rglob('*')):
            if path.is_symlink(): raise ValueError('wrapper contains symlink')
            if path.is_file():
                stat = path.stat()
                blocks=getattr(stat,'st_blocks',None)
                files.append({'path':str(path.relative_to(root)), 'size':stat.st_size, 'allocated_bytes':blocks*512 if blocks is not None else None,
                              'sha256':file_sha256(path), 'mtime_ns':stat.st_mtime_ns, 'nlink':stat.st_nlink})
        candidates.append({'run_id':run,'path':str(root),'task_outcomes':records[run],'files':files,
                           'allocated_bytes':sum(x['allocated_bytes'] for x in files) if all(x['allocated_bytes'] is not None for x in files) else None})
    return {'status':'dry_run_verified','application_root':str(app),'candidates':candidates,
            'candidate_allocated_bytes':sum(x['allocated_bytes'] for x in candidates) if all(x['allocated_bytes'] is not None for x in candidates) else None,
            'protected_children_deleted':False, 'references_found':0}


def apply(app, audit_dir):
    audit_dir = Path(audit_dir).resolve()
    prior = json.loads((audit_dir/'failed_wrapper_cleanup_dry_run.json').read_text())
    current = inspect(app)
    if current != prior: raise ValueError('cleanup target or references changed since dry-run')
    atomic_json(audit_dir/'failed_wrapper_cleanup_journal.json', {'status':'deleting','plan':prior})
    free_before = shutil.disk_usage(app).free; deleted = []
    for target in prior['candidates']:
        root = Path(target['path'])
        for record in target['files']:
            path = root/record['path']
            if file_sha256(path) != record['sha256'] or path.stat().st_mtime_ns != record['mtime_ns']:
                raise ValueError('wrapper file changed immediately before deletion')
        # All targets are literal IDs, constrained to the run root and already
        # rechecked. Dependencies listed in children are retained verbatim.
        shutil.rmtree(root)
        deleted.append(target['run_id'])
        atomic_json(audit_dir/'failed_wrapper_cleanup_journal.json', {'status':'deleting','deleted_run_ids':deleted,'plan':prior})
    result = {**prior,'status':'completed','deleted_run_ids':deleted,
              'retired_allocated_bytes':prior['candidate_allocated_bytes'],
              'free_bytes_before':free_before,'free_bytes_after':shutil.disk_usage(app).free,
              'raw_files_deleted':0,'models_deleted':0,'native_sources_deleted':0}
    atomic_json(audit_dir/'failed_wrapper_cleanup_completed.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['dry-run','apply'])
    parser.add_argument('--project-root',required=True)
    parser.add_argument('--audit-dir',required=True)
    args=parser.parse_args(); audit=Path(args.audit_dir)
    if args.action=='dry-run':
        result=inspect(args.project_root);atomic_json(audit/'failed_wrapper_cleanup_dry_run.json',result)
    else:result=apply(args.project_root,audit)
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},ensure_ascii=False))


if __name__=='__main__':main()
