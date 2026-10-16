import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from my_strategy.storage.native_inventory import inspect, canonical, digest


@pytest.fixture
def inventory_fixture(tmp_path):
    app=tmp_path/'app';strategy=app/'my_strategy';objects=strategy/'data/processed/czsc/native_structure.objects';objects.mkdir(parents=True)
    database=objects.with_suffix('.db')
    with sqlite3.connect(database) as conn:
        conn.executescript('''CREATE TABLE native_structure_entries(cache_key TEXT PRIMARY KEY,identity_json TEXT,metadata_json TEXT,metadata_sha256 TEXT,
            producer_json TEXT,producer_sha256 TEXT,features_blob BLOB,native_blob BLOB,created_at TEXT);
            CREATE TABLE native_structure_files(cache_key TEXT PRIMARY KEY,features_path TEXT,native_path TEXT,paths_sha256 TEXT);
            CREATE TABLE native_structure_settings(name TEXT PRIMARY KEY,value TEXT);''')
        conn.execute('INSERT INTO native_structure_settings VALUES(?,?)',('object_root',str(objects)))
    def add(generation=1,linked=True):
        run='20261015-test-source-'+str(generation)
        folder=strategy/'artifacts/runs'/run/'data_snapshots/native/000001_SZ';folder.mkdir(parents=True,exist_ok=True)
        producer={'raw_path':str(folder/'raw.parquet')};paths={};hashes=[]
        for kind,filename in [('features','features.parquet'),('native','native.json.gz')]:
            content=(kind+str(generation)).encode();sha=hashlib.sha256(content).hexdigest();hashes.append(sha)
            target=objects/sha[:2]/sha;target.parent.mkdir(exist_ok=True);target.write_bytes(content)
            if linked:os.link(target,folder/filename)
            producer[kind+'_path']=str(folder/filename);paths[kind+'_path']=str(target)
        identity={'raw':{'symbol':'000001.SZ','data_end':'2026-10-0'+str(generation)},'config':generation}
        metadata={'sha256':hashes[0],'native_sha256':hashes[1]};key=digest(canonical(identity))
        with sqlite3.connect(database) as conn:
            conn.execute('INSERT INTO native_structure_entries VALUES(?,?,?,?,?,?,?,?,?)',
                (key,canonical(identity),canonical(metadata),digest(canonical(metadata)),canonical(producer),digest(canonical(producer)),b'',b'','2026-10-15'))
            conn.execute('INSERT INTO native_structure_files VALUES(?,?,?,?)',(key,paths['features_path'],paths['native_path'],digest(canonical(paths))))
        return key,hashes,run
    return app,database,objects,add


def test_readonly_inventory_deduplicates_inodes_and_keeps_every_index_generation(inventory_fixture):
    app,db,objects,add=inventory_fixture
    add(1);add(2)
    before=db.read_bytes();payloads={p:p.read_bytes() for p in objects.glob('*/*')}
    report=inspect(app,verify_content=True)
    assert report['summary']['index_entries']==2 and report['summary']['symbols_with_multiple_entries']==1
    assert report['summary']['unique_objects']==4 and report['summary']['unique_inodes']==4
    assert report['summary']['unique_logical_bytes']==sum(len(v) for v in payloads.values())
    assert report['summary']['protected_objects']==4 and report['summary']['deletable_bytes']==0
    assert {e['symbol_retention_order'] for e in report['entries']}=={1,2}
    assert all(o['hardlink_count']==2 and len(o['aliases'])==2 and o['content_sha_verified'] for o in report['objects'])
    assert all('run_alias' in {r['kind'] for r in o['references']} for o in report['objects'])
    assert not report['deletion_authorized'] and not report['ready_for_gc']
    assert db.read_bytes()==before and {p:p.read_bytes() for p in payloads}==payloads


def test_model_account_board_and_active_task_references_are_visible(inventory_fixture):
    app,db,objects,add=inventory_fixture;key,hashes,run=add()
    metadata=app/'my_strategy/data/metadata';metadata.mkdir(parents=True)
    for name,token in [('model_selection.json',key),('enabled_accounts.json',hashes[0]),('board_sources.json',run)]:
        (metadata/name).write_text(json.dumps({'source':token}))
    with sqlite3.connect(metadata/'tasks.db') as conn:
        conn.execute('CREATE TABLE jobs(job_id TEXT,status TEXT,payload TEXT)')
        conn.execute('INSERT INTO jobs VALUES(?,?,?)',('j','running',json.dumps({'source':key})))
    report=inspect(app)
    all_kinds={r['kind'] for o in report['objects'] for r in o['references']}
    assert {'model','account','board','task'}<=all_kinds
    assert report['summary']['active_tasks']==1
    assert all('active_task_barrier' in o['hold_reasons'] for o in report['objects'])


def test_unknown_hardlink_outside_scope_is_never_counted_as_releasable(inventory_fixture,tmp_path):
    app,db,objects,add=inventory_fixture;_,hashes,_=add(linked=False)
    outside=tmp_path/'unknown';os.link(objects/hashes[0][:2]/hashes[0],outside)
    report=inspect(app)
    row=next(o for o in report['objects'] if o['sha256']==hashes[0])
    assert row['unknown_hardlinks']==1 and row['possible_release_after_all_links_retired_bytes'] is None
    assert outside.exists() and row['classification']=='protected'
    scoped=inspect(app,alias_roots=[tmp_path])
    row=next(o for o in scoped['objects'] if o['sha256']==hashes[0])
    assert row['unknown_hardlinks']==0 and str(outside) in row['aliases']


def test_unindexed_object_and_coverage_gap_remain_unresolved(inventory_fixture):
    app,db,objects,add=inventory_fixture;add()
    content=b'unindexed';sha=hashlib.sha256(content).hexdigest();path=objects/sha[:2]/sha;path.parent.mkdir(exist_ok=True);path.write_bytes(content)
    metadata=app/'my_strategy/data/metadata';metadata.mkdir(parents=True)
    (metadata/'large.json').write_text(json.dumps({'unknown':'x'*100}))
    report=inspect(app,max_reference_bytes=1)
    assert report['summary']['unindexed_objects']==1 and report['summary']['coverage_gaps']>0
    row=next(o for o in report['objects'] if o['sha256']==sha)
    assert row['classification']=='unresolved' and not row['deletion_authorized']
    assert report['summary']['deletable_bytes']==0 and path.read_bytes()==content


def test_symlinks_are_not_followed_and_index_path_escape_is_a_gap(inventory_fixture,tmp_path):
    app,db,objects,add=inventory_fixture;key,hashes,_=add()
    outside=tmp_path/'outside';outside.mkdir();(outside/'private').write_text('retain')
    shard=objects/'zz';shard.symlink_to(outside,target_is_directory=True)
    report=inspect(app)
    assert any('symlink shard' in g['reason'] for g in report['coverage_gaps'])
    with sqlite3.connect(db) as conn:
        paths={'features_path':str(outside/'private'),'native_path':str(objects/hashes[1][:2]/hashes[1])}
        conn.execute('UPDATE native_structure_files SET features_path=?,paths_sha256=? WHERE cache_key=?',
            (paths['features_path'],digest(canonical(paths)),key))
    report=inspect(app)
    assert any('escaped canonical root' in g['reason'] for g in report['coverage_gaps'])
    assert (outside/'private').read_text()=='retain'


def test_full_hash_flags_corruption_but_never_deletes_it(inventory_fixture):
    app,db,objects,add=inventory_fixture;_,hashes,_=add()
    path=objects/hashes[0][:2]/hashes[0];path.write_bytes(b'corruption')
    report=inspect(app,verify_content=True)
    row=next(o for o in report['objects'] if o['sha256']==hashes[0])
    assert 'SHA mismatch' in row['error'] and row['classification']=='protected'
    assert path.read_bytes()==b'corruption'


def test_raw_quote_database_is_not_opened_as_a_reference_source(inventory_fixture,monkeypatch):
    from my_strategy.storage import native_inventory
    app,db,objects,add=inventory_fixture;add()
    raw=app/'my_strategy/data/raw';raw.mkdir();(raw/'raw.db').write_bytes(b'not SQLite')
    original=native_inventory.ro
    def checked(path):
        assert not Path(path).is_relative_to(raw)
        return original(path)
    monkeypatch.setattr(native_inventory,'ro',checked)
    assert inspect(app)['summary']['protected_objects']==2
