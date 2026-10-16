"""Read-only inventory of native CAS references and physical hard-link groups.

This is a retention-review report, never an executable delete manifest. A live
snapshot cannot prove global absence of references: indexed objects, run aliases,
unknown hard links, unreadable sources and concurrent changes all remain held.
Numeric native payloads and raw quote tables are not scanned for references.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time

SHA = re.compile(r'^[a-f0-9]{64}$')
TOKENS = re.compile(r'(?<![A-Za-z0-9_.-])(?:[a-f0-9]{64}|20\d{6}-[\w.-]+)(?![A-Za-z0-9_.-])')
ACTIVE = {'queued', 'running', 'cancelling'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def quote(name):
    return '"'+name.replace('"','""')+'"'


def ro(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    return connection


def classify(path):
    name = str(path).replace('\\', '/').lower().split('my_strategy/', 1)[-1]
    if any(word in name for word in ('account','simulation','ledger','position')): return 'account'
    if any(word in name for word in ('model','training','score','selection')): return 'model'
    if any(word in name for word in ('board','fluent','sector')): return 'board'
    if any(word in name for word in ('task','job','checkpoint')): return 'task'
    if any(word in name for word in ('native','source','calendar')): return 'source'
    if 'artifacts' in name: return 'run'
    return 'metadata'


def inspect(project_root, *, database=None, alias_roots=(), verify_content=False,
            max_reference_bytes=512*1024**2, max_table_rows=100000, max_cas_bytes=8*1024**2):
    project = Path(project_root).expanduser().resolve()
    strategy = project/'my_strategy'
    if not strategy.is_dir(): raise ValueError('application root must contain my_strategy')
    database = Path(database or strategy/'data/processed/czsc/native_structure.db').expanduser()
    if database.is_symlink() or not database.is_file(): raise ValueError('native index is absent or unsafe')
    database = database.resolve()
    gaps, active, objects, entries = [], [], {}, []
    before = database.stat()
    with closing(ro(database)) as connection:
        connection.execute('BEGIN')
        setting = connection.execute("SELECT value FROM native_structure_settings WHERE name='object_root'").fetchone()
        if not setting: raise ValueError('native object root is missing')
        raw_root = Path(setting[0]).expanduser()
        if raw_root.is_symlink(): raise ValueError('native object root is a symlink')
        object_root = raw_root.resolve()
        if not object_root.is_dir(): raise ValueError('native object root is absent')
        query = '''SELECT e.cache_key,e.identity_json,e.metadata_json,e.metadata_sha256,
            e.producer_json,e.producer_sha256,e.created_at,length(e.features_blob) AS features_blob_bytes,
            length(e.native_blob) AS native_blob_bytes,f.features_path,f.native_path,f.paths_sha256
            FROM native_structure_entries e LEFT JOIN native_structure_files f USING(cache_key) ORDER BY e.cache_key'''
        for row in connection.execute(query):
            try:
                identity, metadata, producer = (json.loads(row[name]) for name in ('identity_json','metadata_json','producer_json'))
                if digest(canonical(identity)) != row['cache_key'] or digest(row['metadata_json']) != row['metadata_sha256'] or digest(row['producer_json']) != row['producer_sha256']:
                    raise ValueError('native index metadata SHA mismatch')
                record = {'cache_key':row['cache_key'],'symbol':identity.get('raw',{}).get('symbol'),
                    'data_end':identity.get('raw',{}).get('data_end'),'created_at':row['created_at'],
                    'generation_id':digest(canonical({k:v for k,v in identity.items() if k not in {'raw','liquidity','calendar'}})),
                    'producer_paths':producer,'object_shas':[]}
                if row['features_path'] is None:
                    gaps.append({'source':row['cache_key'],'reason':'legacy_blob_entry_retained',
                        'bytes':row['features_blob_bytes']+row['native_blob_bytes']})
                    entries.append(record); continue
                paths={kind+'_path':row[kind+'_path'] for kind in ('features','native')}
                if digest(canonical(paths)) != row['paths_sha256']: raise ValueError('native file-index paths SHA mismatch')
                if row['features_blob_bytes'] or row['native_blob_bytes']: raise ValueError('file index contains duplicate legacy BLOB')
                for kind, sha_name in (('features','sha256'),('native','native_sha256')):
                    sha = metadata.get(sha_name)
                    path = Path(paths[kind+'_path'])
                    if not isinstance(sha,str) or not SHA.fullmatch(sha): raise ValueError('invalid native object SHA')
                    expected = object_root/sha[:2]/sha
                    if path != expected or path.is_symlink() or not path.resolve().is_relative_to(object_root):
                        raise ValueError('native object path escaped canonical root')
                    obj=objects.setdefault(sha,{'sha256':sha,'path':str(path),'index_references':[], 'references':[]})
                    obj['index_references'].append({'cache_key':row['cache_key'],'symbol':record['symbol'],
                        'data_end':record['data_end'],'generation_id':record['generation_id'],'kind':kind})
                    record['object_shas'].append(sha)
                entries.append(record)
            except (ValueError,TypeError,KeyError,OSError) as exc:
                gaps.append({'source':row['cache_key'],'reason':str(exc)})
        connection.commit()
    # Discover unindexed CAS filenames without following symlinks or reading
    # their contents. Indexed rows missing from disk remain explicit gaps.
    for shard in sorted(object_root.iterdir()):
        if shard.is_symlink():
            gaps.append({'source':str(shard),'reason':'symlink shard not followed'}); continue
        if not shard.is_dir(): continue
        for path in sorted(shard.iterdir()):
            if path.is_symlink():
                gaps.append({'source':str(path),'reason':'symlink object not followed'}); continue
            if SHA.fullmatch(path.name) and path.parent.name==path.name[:2] and path.is_file():
                objects.setdefault(path.name,{'sha256':path.name,'path':str(path),'index_references':[],'references':[]})
    inode_objects = defaultdict(list)
    for sha,obj in objects.items():
        path=Path(obj['path'])
        try:
            stat=path.stat(follow_symlinks=False)
            obj.update({'device':stat.st_dev,'inode':stat.st_ino,'size_bytes':stat.st_size,
                'allocated_bytes':getattr(stat,'st_blocks',0)*512,'hardlink_count':stat.st_nlink,
                'mtime_ns':stat.st_mtime_ns,'aliases':[str(path)],'content_sha_verified':False})
            inode_objects[(stat.st_dev,stat.st_ino)].append(sha)
            if verify_content:
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                if h.hexdigest()!=sha: raise ValueError('native object content SHA mismatch')
                obj['content_sha_verified']=True
        except (OSError,ValueError) as exc:
            obj['error']=str(exc); gaps.append({'source':str(path),'reason':str(exc)})
    # Include all aliases visible inside explicitly declared roots. Hard links
    # elsewhere are detectable as a count discrepancy, never guessed as freed.
    roots=list(dict.fromkeys([project,*[Path(p).resolve() for p in alias_roots]]))
    reference_files, databases=set(),set()
    exclude={'.git','__pycache__','.venv','node_modules','.scratch','.release-work','knowledge_base_system'}
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            gaps.append({'source':str(root),'reason':'alias root absent or unsafe'});continue
        for current,dirs,files in os.walk(root,followlinks=False):
            for name in list(dirs):
                path=Path(current)/name
                if path.is_symlink():
                    gaps.append({'source':str(path),'reason':'symlink directory not followed'});dirs.remove(name)
                elif name in exclude:
                    dirs.remove(name)
            for name in files:
                path=Path(current)/name
                try:
                    if path.is_symlink():
                        gaps.append({'source':str(path),'reason':'symlink file not followed'});continue
                    stat=path.stat(follow_symlinks=False)
                    for sha in inode_objects.get((stat.st_dev,stat.st_ino),()):
                        if str(path) not in objects[sha]['aliases']:objects[sha]['aliases'].append(str(path))
                    # Raw/warehouse quote tables and numeric leaves are outside
                    # reference scope; they cannot be deletion targets here.
                    if path.is_relative_to(strategy/'data/raw'):continue
                    if path.suffix in {'.json','.jsonl'} and not path.is_relative_to(object_root):reference_files.add(path)
                    if path.suffix in {'.db','.sqlite','.sqlite3'} and path!=database and not path.is_relative_to(strategy/'data/processed/warehouse'):
                        databases.add(path)
                except OSError as exc:gaps.append({'source':str(path),'reason':str(exc)})
    targets=defaultdict(set)
    runs_root=strategy/'artifacts/runs'
    for sha,obj in objects.items():
        targets[sha].add(sha)
        for ref in obj['index_references']:targets[ref['cache_key']].add(sha)
        for alias in obj.get('aliases',()):
            path=Path(alias)
            if path.is_relative_to(runs_root):
                run=path.relative_to(runs_root).parts[0]
                targets[run].add(sha)
                obj['references'].append({'kind':'run_alias','source':alias,'token':run})
    scanned_bytes=0;source_count=0;seen_refs=set();budget_exhausted=False
    def consume(text,source,kind=None):
        nonlocal scanned_bytes,source_count,budget_exhausted
        size=len(text.encode('utf-8'))
        if scanned_bytes+size>max_reference_bytes:
            budget_exhausted=True
            raise ValueError('reference scan byte budget exhausted; retain all unresolved objects')
        scanned_bytes+=size;source_count+=1
        kind=kind or classify(source)
        for token in set(TOKENS.findall(text)):
            for sha in targets.get(token,()):
                marker=(sha,str(source),token)
                if marker in seen_refs:continue
                seen_refs.add(marker)
                objects[sha]['references'].append({'kind':kind,'source':str(source),'token':token})
    for path in sorted(databases,key=lambda p:(not p.is_relative_to(strategy/'data/metadata'),str(p))):
        try:
            with closing(ro(path)) as connection:
                connection.execute('BEGIN')
                tables=[r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                for table in tables:
                    # Accounting/index projections only repeat references.
                    if table in {'object_usage','research_totals','research_accounting_version'}:continue
                    columns=[r[1] for r in connection.execute('PRAGMA table_info('+quote(table)+')') if any(t in r[2].upper() for t in ('TEXT','CHAR','CLOB'))]
                    if not columns:continue
                    if budget_exhausted:
                        gaps.append({'source':str(path)+':'+table,'reason':'reference scan budget exhausted; table not scanned'})
                        if 'status' in columns:
                            id_column='job_id' if 'job_id' in columns else 'run_id' if 'run_id' in columns else None
                            fields='status'+(','+quote(id_column) if id_column else '')
                            for state in connection.execute('SELECT '+fields+' FROM '+quote(table)+" WHERE status IN ('queued','running','cancelling') LIMIT 1000"):
                                active.append({'source':str(path)+':'+table,'status':state[0],'job_id':state[1] if id_column else None})
                        continue
                    for n,row in enumerate(connection.execute('SELECT '+','.join(quote(c) for c in columns)+' FROM '+quote(table)+' LIMIT ?', (max_table_rows+1,))):
                        if n==max_table_rows:
                            gaps.append({'source':str(path)+':'+table,'reason':'reference table row budget exhausted'});break
                        values=dict(zip(columns,row))
                        if values.get('status') in ACTIVE:active.append({'source':str(path)+':'+table,'status':values['status'],
                            'job_id':values.get('job_id') or values.get('run_id')})
                        consume('\n'.join(str(v) for v in row if v is not None),str(path)+':'+table)
                # Saved research objects may contain account/model references;
                # inspect bounded payloads without constructing ResearchStore.
                if {'results','objects'}<=set(tables) and not budget_exhausted:
                    for row in connection.execute("SELECT r.cache_key,r.namespace,r.sha256,o.size_bytes FROM results r JOIN objects o USING(sha256) WHERE o.state='active'"):
                        source=str(path)+':CAS:'+row['cache_key']
                        if row['size_bytes']>max_cas_bytes:
                            gaps.append({'source':source,'reason':'compressed CAS payload exceeds read budget'});continue
                        sha=row['sha256']
                        if not SHA.fullmatch(sha):raise ValueError('invalid research CAS SHA')
                        obj_path=path.parent/'objects'/sha[:2]/(sha+'.json.gz')
                        if obj_path.is_symlink() or not obj_path.resolve().is_relative_to((path.parent/'objects').resolve()):raise ValueError('research CAS path escaped')
                        payload=obj_path.read_bytes()
                        if len(payload)!=row['size_bytes'] or hashlib.sha256(payload).hexdigest()!=sha:raise ValueError('research CAS SHA/size mismatch')
                        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
                            decoded=compressed.read(max_cas_bytes+1)
                        if len(decoded)>max_cas_bytes:
                            gaps.append({'source':source,'reason':'expanded CAS payload exceeds read budget'});continue
                        consume(decoded.decode('utf-8'),source,classify(row['namespace']))
                connection.commit()
        except (OSError,ValueError,sqlite3.Error) as exc:gaps.append({'source':str(path),'reason':str(exc)})
    reference_priority={'account':0,'model':1,'board':2,'task':3,'metadata':4,'source':5,'run':6}
    for path in sorted(reference_files,key=lambda p:(reference_priority.get(classify(p),9),str(p))):
        if budget_exhausted:
            gaps.append({'source':str(path),'reason':'reference scan budget exhausted; source not read'});continue
        try:
            # Stream JSON/JSONL so a full market source manifest is not resident.
            with path.open('r',encoding='utf-8') as stream:
                carry=''
                while chunk:=stream.read(1024*1024):
                    consume(carry+chunk,path);carry=chunk[-512:]
        except (OSError,ValueError) as exc:gaps.append({'source':str(path),'reason':str(exc)})
    after=database.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        gaps.append({'source':str(database),'reason':'native index changed during inventory'})
    by_symbol=defaultdict(list)
    for entry in entries:by_symbol[entry['symbol']].append(entry)
    for generations in by_symbol.values():
        for order,entry in enumerate(sorted(generations,key=lambda e:(e['data_end'] or '',e['created_at'],e['cache_key']),reverse=True),1):
            entry['symbol_retention_order']=order
    for sha,obj in objects.items():
        obj['aliases']=sorted(obj.get('aliases',()))
        obj['unknown_hardlinks']=max(0,obj.get('hardlink_count',0)-len(obj['aliases']))
        reasons=[]
        if obj['index_references']:reasons.append('native_index_reference')
        if obj['references']:reasons.append('run_or_external_reference')
        if active:reasons.append('active_task_barrier')
        if obj.get('error'):reasons.append('object_error')
        if obj['unknown_hardlinks']:reasons.append('unknown_hardlink_paths')
        obj['classification']='protected' if reasons else 'unresolved'
        obj['hold_reasons']=reasons or ['absence_of_references_is_not_retirement_authorization']
        obj['deletion_authorized']=False
        obj['possible_release_after_all_links_retired_bytes']=obj.get('allocated_bytes',0) if not obj['unknown_hardlinks'] else None
    unique={}
    for obj in objects.values():
        if 'inode' in obj:unique[(obj['device'],obj['inode'])]=obj
    report={'schema':'native_storage_inventory_v1','generated_at_epoch':time.time(),'dry_run':True,
        'project_root':str(project),'database':str(database),'object_root':str(object_root),
        'content_hash_mode':'full' if verify_content else 'metadata_and_paths_only',
        'reference_scope':'JSON/JSONL control files, metadata/processed SQLite TEXT columns and bounded research CAS; raw quotes, numeric leaves, audit archives and runtime dependencies excluded',
        'deletion_authorized':False,'ready_for_gc':False,
        'required_before_retirement':['complete frozen scans and compare results','freeze every dependency writer',
            'verify model/account/run/board/rollback references','verify exact object SHA and every hard-link path',
            'review exact explicit retired generation keys','recheck plan immediately before journalled unlink'],
        'summary':{'index_entries':len(entries),'symbols':len(by_symbol),'symbols_with_multiple_entries':sum(len(v)>1 for v in by_symbol.values()),
            'unique_objects':len(objects),'unique_inodes':len(unique),'indexed_objects':sum(bool(o['index_references']) for o in objects.values()),
            'unindexed_objects':sum(not o['index_references'] for o in objects.values()),
            'unique_logical_bytes':sum(o['size_bytes'] for o in unique.values()),'unique_allocated_bytes':sum(o['allocated_bytes'] for o in unique.values()),
            'protected_objects':sum(o['classification']=='protected' for o in objects.values()),'unresolved_objects':sum(o['classification']=='unresolved' for o in objects.values()),
            'unknown_hardlink_objects':sum(o['unknown_hardlinks']>0 for o in objects.values()),'deletable_bytes':0,
            'reference_bytes_scanned':scanned_bytes,'reference_sources_scanned':source_count,'coverage_gaps':len(gaps),'active_tasks':len(active)},
        'coverage_gaps':gaps,'active_tasks':active,'entries':entries,'objects':list(objects.values())}
    return report
