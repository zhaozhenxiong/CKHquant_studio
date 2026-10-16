"""Successful scope proofs are reusable only for the same observed database."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import sqlite3
import threading
import time

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.board_calculation_scope import calculation_membership
from my_strategy.storage import board_fluent as storage
from my_strategy.storage.board_fluent import BoardFluentStore, canonical, digest
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_fluent import bundle


@pytest.fixture(autouse=True)
def empty_scope_proofs():
    with storage._scope_cache_lock:
        storage._scope_cache.clear();storage._scope_cache_bytes=0
    yield
    with storage._scope_cache_lock:
        storage._scope_cache.clear();storage._scope_cache_bytes=0


@pytest.fixture
def published(market,tmp_path):
    value=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db')
    generation=store.save_market(value)
    return store,generation,value


def count_validations(monkeypatch):
    calls=[];original=BoardFluentStore._validate_scope_projection
    def observed(conn,generation):
        calls.append(generation['generation_id'])
        return original(conn,generation)
    monkeypatch.setattr(BoardFluentStore,'_validate_scope_projection',staticmethod(observed))
    return calls


def wide_bundle(market,count=710):
    """Independent stored boards with identical known seven-stock prices."""
    value=bundle(market);original=value['boards'][0];boards=[]
    for index in range(count):
        snapshot=market['snapshot']('industry' if index<320 else 'concept',str(100000+index))
        projection=calculation_membership(snapshot);row=deepcopy(original)
        row.update(board_id=snapshot['board_id'],kind=snapshot['kind'],name=snapshot['name'],
            snapshot_hash=snapshot['snapshot_hash'],source_snapshot_hash=snapshot['snapshot_hash'],
            projection_hash=projection['projection_hash'],calculation_membership=projection)
        boards.append(row)
    value.update(boards=boards,stocks=[],calculation_membership={row['board_id']:row['calculation_membership'] for row in boards},
        cache_key=stable_hash({'fixture':'scope-verification', 'boards':count}))
    return value


def corrupt_board(path,field):
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TRIGGER immutable_board_daily_features_update')
        if field=='payload_sha':conn.execute("UPDATE board_daily_features SET payload_sha=?",('0'*64,))
        elif field=='board_amount':conn.execute('UPDATE board_daily_features SET board_amount=board_amount+1')
        elif field=='derived':conn.execute('UPDATE board_daily_features SET activity_score=999')
        else:
            conn.row_factory=sqlite3.Row
            import json
            row=conn.execute('SELECT board_id,payload_json FROM board_daily_features ORDER BY board_id LIMIT 1').fetchone()
            value=json.loads(row['payload_json'])
            value['calculation_membership']['projection_hash']='0'*64
            conn.execute('UPDATE board_daily_features SET payload_json=?,payload_sha=? WHERE board_id=?',
                (canonical(value),digest(value),row['board_id']))


def test_successful_proof_is_shared_across_readers_and_preserves_all_results(published,monkeypatch):
    store,generation,value=published;calls=count_validations(monkeypatch)
    first=store.current_generation();assert calls==[generation]
    reader=BoardFluentStore(store.path,readonly=True)
    assert reader.current_generation()==first
    assert reader.verify_calculation_scope(generation)==first
    assert reader.list_generations()==[first]
    assert reader.rankings('concept','capital')==store.rankings('concept','capital')
    assert reader.boards('industry')==store.boards('industry')
    assert calls==[generation]
    assert all(isinstance(size,int) for size in storage._scope_cache.values())
    # A caller cannot mutate a cached population through its own response.
    first['calculation_membership'].clear()
    assert reader.current_generation()['calculation_membership']==value['calculation_membership']
    assert calls==[generation]


def test_710_board_status_read_pattern_checks_each_payload_once(market,tmp_path,monkeypatch):
    store=BoardFluentStore(tmp_path/'fluent.db');generation=store.save_market(wide_bundle(market))
    checked=[];original=BoardFluentStore._row
    def observed(row,board=False):
        if board:checked.append(row['board_id'])
        return original(row,board=board)
    monkeypatch.setattr(BoardFluentStore,'_row',staticmethod(observed))
    # These calls model selection, freshness and the generation catalog.
    assert store.current_generation()['generation_id']==generation
    assert store.current_generation()['generation_id']==generation
    assert store.current_generation('full') is None
    assert store.list_generations()[0]['generation_id']==generation
    assert len(checked)==710
    assert len(storage._scope_cache)==1 and storage._scope_cache_bytes<1024


@pytest.mark.parametrize('field',['payload_sha','board_amount','derived','projection'])
def test_every_original_row_integrity_check_runs_again_after_database_edit(published,field,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch)
    store.current_generation();corrupt_board(store.path,field)
    with pytest.raises(ValueError):store.current_generation()
    with pytest.raises(ValueError):store.current_generation()
    assert calls==[generation]*3  # Failed proofs are never cached.


def test_committed_wal_corruption_invalidates_even_when_main_database_is_unchanged(published,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch)
    with sqlite3.connect(store.path) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        store.current_generation();before=storage._scope_file_signature(store.path)
        writer.execute('DROP TRIGGER immutable_board_daily_features_update')
        writer.execute('UPDATE board_daily_features SET board_amount=board_amount+1');writer.commit()
        after=storage._scope_file_signature(store.path)
        assert before[0]==after[0] and before[1]!=after[1]
        with pytest.raises(ValueError,match='numeric columns'):store.current_generation()
    assert calls==[generation,generation]


def test_same_size_restored_mtime_does_not_hide_a_modified_database(published,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch)
    store.current_generation();before=store.path.stat()
    corrupt_board(store.path,'board_amount')
    os.utime(store.path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert store.path.stat().st_size==before.st_size
    assert store.path.stat().st_mtime_ns==before.st_mtime_ns
    with pytest.raises(ValueError):store.current_generation()
    assert calls==[generation,generation]


def replacement_database(store,path):
    with sqlite3.connect(store.path) as source,sqlite3.connect(path) as target:source.backup(target)
    corrupt_board(path,'board_amount')
    return path


def test_atomic_replacement_with_same_size_and_mtime_is_not_a_cache_hit(published,tmp_path,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch)
    staged=replacement_database(store,tmp_path/'replacement.db')
    store.current_generation();before=store.path.stat()
    os.utime(staged,ns=(before.st_atime_ns,before.st_mtime_ns));assert staged.stat().st_size==before.st_size
    os.replace(staged,store.path)
    # A stale WAL may make SQLite reject the replaced file before the scope
    # verifier runs. Both paths leave the new pathname uncertified.
    with pytest.raises((ValueError,sqlite3.DatabaseError)):store.current_generation()
    assert calls in ([generation],[generation,generation])


def test_replacement_while_connection_is_open_cannot_certify_the_new_path(published,tmp_path):
    store,generation,_=published;staged=replacement_database(store,tmp_path/'replacement.db')
    with store._read() as conn:
        frozen=store._generation(conn,generation)
        os.replace(staged,store.path)
        with pytest.raises(ValueError,match='changed before scope'):store._scope_projection(conn,frozen)
    assert not storage._scope_cache


def test_signature_change_during_complete_verification_fails_without_caching(published,monkeypatch):
    store,generation,_=published;original=BoardFluentStore._validate_scope_projection
    def changed(conn,value):
        result=original(conn,value)
        stat=store.path.stat();os.utime(store.path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
        return result
    monkeypatch.setattr(BoardFluentStore,'_validate_scope_projection',staticmethod(changed))
    with pytest.raises(ValueError,match='changed during scope'):store.current_generation()
    assert not storage._scope_cache and storage._scope_cache_bytes==0
    monkeypatch.setattr(BoardFluentStore,'_validate_scope_projection',staticmethod(original))
    assert store.current_generation()['generation_id']==generation


def test_signature_change_at_cache_hit_is_rejected(published,monkeypatch):
    store,generation,_=published;store.current_generation()
    with store._read() as conn:
        frozen=store._generation(conn,generation);original=storage._scope_file_signature;checks=[]
        def changing(path):
            checks.append(path)
            signature=original(path)
            if len(checks)==3:
                main=list(signature[0]);main[-1]+=1
                return (tuple(main),*signature[1:])
            return signature
        monkeypatch.setattr(storage,'_scope_file_signature',changing)
        with pytest.raises(ValueError,match='changed during scope'):store._scope_projection(conn,frozen)


def test_generation_content_is_part_of_the_proof_key(published,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch)
    store.current_generation()
    with store._read() as conn:
        frozen=store._generation(conn,generation);frozen['board_count']+=1
        with pytest.raises(ValueError,match='population drift'):store._scope_projection(conn,frozen)
    assert calls==[generation,generation]


def test_uncommitted_transactions_and_external_connections_always_verify(published,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch);store.current_generation()
    with pytest.raises(ValueError,match='numeric columns'):
        with store._write() as conn:
            conn.execute('DROP TRIGGER immutable_board_daily_features_update')
            conn.execute('UPDATE board_daily_features SET board_amount=board_amount+1')
            store._scope_projection(conn,store._generation(conn,generation))
    with sqlite3.connect(store.path) as conn:
        conn.row_factory=sqlite3.Row
        for _ in range(2):store._scope_projection(conn,store._generation(conn,generation))
    assert calls==[generation]*4


def test_concurrent_readers_share_one_successful_complete_verification(published,monkeypatch):
    store,generation,_=published;original=BoardFluentStore._validate_scope_projection;calls=[]
    def observed(conn,value):
        calls.append(value['generation_id']);time.sleep(.02)
        return original(conn,value)
    monkeypatch.setattr(BoardFluentStore,'_validate_scope_projection',staticmethod(observed))
    barrier=threading.Barrier(8)
    def read(_):
        barrier.wait();return BoardFluentStore(store.path,readonly=True).current_generation()
    # Keep the existing WAL open so simultaneous first-reader creation does not
    # obscure the actual concurrency property under test.
    with sqlite3.connect(store.path) as keeper,ThreadPoolExecutor(max_workers=8) as executor:
        keeper.execute('SELECT COUNT(*) FROM feature_generations').fetchone()
        results=list(executor.map(read,range(8)))
    assert all(value==results[0] for value in results) and calls==[generation]


def test_entry_and_byte_budgets_are_enforced_and_evicted_proofs_revalidate(market,tmp_path,monkeypatch):
    value=bundle(market);stores=[]
    for index in range(4):
        store=BoardFluentStore(tmp_path/f'fluent-{index}.db');store.save_market(value);stores.append(store)
    calls=count_validations(monkeypatch);monkeypatch.setattr(storage,'_SCOPE_CACHE_ENTRIES',2)
    for store in stores:store.current_generation()
    assert len(storage._scope_cache)==2 and storage._scope_cache_bytes<=storage._SCOPE_CACHE_BYTES
    stores[0].current_generation();assert len(calls)==5
    with storage._scope_cache_lock:storage._scope_cache.clear();storage._scope_cache_bytes=0
    monkeypatch.setattr(storage,'_SCOPE_CACHE_BYTES',32)
    stores[0].current_generation();stores[0].current_generation()
    assert len(calls)==7 and not storage._scope_cache and storage._scope_cache_bytes==0


def test_journal_presence_is_part_of_the_proof_identity(published,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch);store.current_generation()
    journal=Path(str(store.path)+'-journal');journal.write_bytes(b'')
    store.current_generation();journal.unlink();store.current_generation()
    assert calls==[generation]*2
    assert len(storage._scope_cache)==2


def test_first_read_can_initialize_an_empty_wal_without_losing_validation(published,monkeypatch):
    store,generation,_=published;calls=count_validations(monkeypatch)
    for suffix in ('-wal','-shm'):Path(str(store.path)+suffix).unlink(missing_ok=True)
    assert storage._scope_file_signature(store.path)[1] is None
    with store._read() as conn:
        frozen=store._generation(conn,generation)
        assert conn.scope_signature[1] is not None and conn.scope_signature[1][2]<=32
        assert store._scope_projection(conn,frozen)==frozen['calculation_membership']
    assert calls==[generation]


def test_committed_wal_created_while_opening_is_never_treated_as_empty_initialization(published,monkeypatch):
    store,_,_=published
    for suffix in ('-wal','-shm'):Path(str(store.path)+suffix).unlink(missing_ok=True)
    original=sqlite3.connect;writers=[]
    def connect(*args,**kwargs):
        connection=original(*args,**kwargs)
        if args and str(args[0]).startswith(store.path.as_uri()) and '?mode=ro' in str(args[0]):
            writer=original(store.path);writers.append(writer)
            writer.execute('UPDATE feature_generations SET created_at=created_at+1');writer.commit()
        return connection
    monkeypatch.setattr(storage.sqlite3,'connect',connect)
    try:
        with pytest.raises(ValueError,match='changed while opening'):store.current_generation()
        assert not storage._scope_cache
        assert storage._scope_file_signature(store.path)[1][2]>32
    finally:
        for writer in writers:writer.close()


def test_committed_write_during_complete_verification_is_rejected(published,monkeypatch):
    store,_,_=published;original=BoardFluentStore._validate_scope_projection
    def changed(conn,generation):
        result=original(conn,generation)
        corrupt_board(store.path,'board_amount')
        return result
    monkeypatch.setattr(BoardFluentStore,'_validate_scope_projection',staticmethod(changed))
    with pytest.raises(ValueError,match='changed during scope'):store.current_generation()
    assert not storage._scope_cache
    monkeypatch.setattr(BoardFluentStore,'_validate_scope_projection',staticmethod(original))
    with pytest.raises(ValueError,match='numeric columns'):store.current_generation()


def test_full_market_verification_still_checks_all_row_and_bundle_hashes(published,monkeypatch):
    store,generation,value=published;store.current_generation();checked=[];original=BoardFluentStore._row
    def observed(row,board=False):
        if board:checked.append(row['board_id'])
        return original(row,board=board)
    monkeypatch.setattr(BoardFluentStore,'_row',staticmethod(observed))
    assert store.verify_generation(generation)['generation_id']==generation
    assert len(checked)==len(value['boards'])
    assert store.load_market(value['cache_key'])==value
