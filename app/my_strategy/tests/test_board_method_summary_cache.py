"""Authenticated saved board projections: equivalent, bounded and invalidated."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from my_strategy.services import board_method_summary as service
from my_strategy.storage import board_knowledge as kb
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.tests.test_board_knowledge import snapshot

DAY = '2026-10-15'
METHOD = 'structure_kronos'
BOARD = 'THS_I_881155'


def report():
    context = dict(analysis_method=METHOD, start='2026-01-01', end=DAY,
        usage_mode='historical', score_model_id=None, lookback=128, horizon=5,
        samples=32, seed=42, device='cpu')
    rows = []
    for number in range(1, 4):
        rows.append({'symbol': f'00000{number}.SZ', 'name': f'synthetic {number}',
            'date': DAY, 'data_end': DAY, 'analysis_method': METHOD,
            'method_context': deepcopy(context), 'reference_price': 12., 'close': 12.,
            'reference_price_date': DAY, 'reference_price_source': 'verified_local_close',
            'method_evaluation': {'schema': 'account_free_methods_v1', 'method': METHOD,
                'as_of': DAY, 'status': 'complete', 'direction': 'BUY', 'missing_reasons': [],
                'model_id': None, 'feature_profile': None, 'score': None,
                'kronos': {'status': 'completed', 'as_of': DAY,
                    'quality': {'prediction_usable': True, 'calendar_verified': True,
                        'input_contract_verified': True, 'source_verified': True,
                        'requested_samples': 32, 'valid_samples': 32,
                        'invalid_samples': 0, 'valid_fraction': 1.},
                    'summary': {'terminal_return_p10': -.05, 'terminal_return_median': .02,
                        'terminal_return_p90': .05, 'terminal_return_std': .02,
                        'up_path_fraction': .6, 'worst_path_low_return': -.1},
                    'result_ref': {'cache_key': 'c'*64}}}})
    population = [row['symbol'] for row in rows]
    return {'run_id': 'synthetic-board-cache', 'status': 'succeeded',
        'strategy_version': 'account_free_methods_v1', 'analysis_mode': 'reference',
        'analysis_method': METHOD, 'method_context': context, 'as_of': DAY,
        'data_end': DAY, 'rows': rows, 'population': population,
        'frozen_identity': {'context': deepcopy(context), 'population': population,
            'model_version': None},
        'coverage': {'requested': 3, 'failed': 0, 'deferred': 0,
            'unavailable': 0, 'evaluated': 3, 'complete': True}}


@pytest.fixture
def saved(tmp_path, monkeypatch):
    clock = [datetime(2026, 10, 14, 6, tzinfo=timezone.utc)]
    monkeypatch.setattr(kb, '_now', lambda: clock[0])
    results = ResultStore(tmp_path/'results.db', tmp_path/'runs')
    original = report(); results.save('scan', original)
    store = kb.BoardKnowledgeStore(tmp_path/'kb.db')
    store.import_snapshot(snapshot(members=['000001.SZ', '000002.SZ', '000003.SZ',
        '000007.SZ', '830001.BJ']))
    reader = service.BoardMethodSummary(results.db_path, results.runs_root, store.path)
    return results, store, reader, original, clock


def read(reader, original, **kwargs):
    return reader.summary(original['run_id'], method=METHOD, as_of=DAY, **kwargs)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def counted(reader, monkeypatch):
    calls = []; original = reader._boards
    def build(value):
        calls.append(1)
        return original(value)
    monkeypatch.setattr(reader, '_boards', build)
    return calls


def test_warm_summary_and_member_filters_are_exact_and_cannot_mutate_cache(saved, monkeypatch):
    results, store, reader, original, _ = saved
    uncached = service.BoardMethodSummary(results.db_path, results.runs_root,
        store.path, cache_bytes=0)
    calls = counted(reader, monkeypatch)
    expected = read(uncached, original)
    first = read(reader, original)
    assert first == expected and digest(first) == digest(read(reader, original))
    assert first['boards'][0]['counts']['expected'] == 4
    assert first['boards'][0]['counts']['not_in_scan'] == 1
    assert first['boards'][0]['membership']['excluded_count'] == 1
    for category in ('all', 'buy', 'wait', 'not_in_scan', 'failed'):
        options = dict(board_id=BOARD, page_size=1, category=category)
        assert read(reader, original, **options) == read(uncached, original, **options)
    missing = read(reader, original, board_id=BOARD, category='not_in_scan')
    assert missing['rows'][0]['symbol'] == '000007.SZ'
    assert missing['rows'][0]['method_view']['direction'] is None
    missing['board']['counts']['expected'] = 999
    missing['rows'][0]['method_view']['status'] = 'fake'
    first['boards'][0]['membership']['snapshot_hash'] = 'fake'
    first['identity']['memberships'].clear()
    assert read(reader, original) == expected
    assert len(calls) == 1
    assert len(reader._cache) <= reader.cache_entries
    assert reader._bytes <= reader.cache_bytes


def test_concurrent_cold_summary_and_members_share_one_verified_projection(saved, monkeypatch):
    _, _, reader, original, _ = saved
    calls = counted(reader, monkeypatch)
    ready = threading.Barrier(8)
    def request(number):
        ready.wait()
        return read(reader, original, **({'board_id': BOARD} if number % 2 else {}))
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(request, range(8)))
    assert len(calls) == 1
    assert all(value['identity'] == values[0]['identity'] for value in values)
    assert values[0] == values[2] and values[1] == values[3]


def test_wal_only_new_snapshot_and_running_manifest_invalidate(saved, monkeypatch):
    _, store, reader, original, clock = saved
    calls = counted(reader, monkeypatch)
    with closing(sqlite3.connect(store.path)) as keeper, keeper:
        keeper.execute('PRAGMA wal_autocheckpoint=0')
        keeper.execute('BEGIN')
        keeper.execute('SELECT COUNT(*) FROM board_refresh_runs').fetchone()
        first = read(reader, original)
        main_before = service._signature(store.path)
        clock[0] = datetime(2026, 10, 14, 6, 10, tzinfo=timezone.utc)
        run = store.begin_refresh(requested_boards=['THS_C_885001'])
        store.import_snapshot(snapshot(board='THS_C_885001', members=['000001.SZ']), run)
        # Unpublished snapshots remain invisible even after invalidation.
        assert read(reader, original) == first
        store.finish_refresh(run)
        changed = read(reader, original)
        assert len(changed['boards']) == 2
        assert changed['identity']['summary_sha256'] != first['identity']['summary_sha256']
        assert service._signature(store.path) == main_before
        assert Path(str(store.path)+'-wal').stat().st_size > 0
    assert len(calls) == 3


@pytest.mark.parametrize('table,operation,sql', [
    ('board_membership', 'delete', 'DELETE FROM board_membership WHERE symbol=\'000001.SZ\''),
    ('source_payloads', 'update', "UPDATE source_payloads SET blob_sha256='"+'0'*64+"'"),
    ('source_payloads', 'delete', 'DELETE FROM source_payloads'),
    ('board_snapshots', 'update', "UPDATE board_snapshots SET payload_json='{}'"),
    ('board_catalog_versions', 'update', "UPDATE board_catalog_versions SET payload_json='{}'"),
])
def test_wal_source_membership_or_catalog_corruption_cannot_hide_behind_warm_cache(saved, table, operation, sql):
    _, store, reader, original, _ = saved
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('PRAGMA wal_autocheckpoint=0')
        connection.execute('SELECT COUNT(*) FROM board_refresh_runs').fetchone()
        read(reader, original)
        main_before = service._signature(store.path)
        connection.execute(f'DROP TRIGGER immutable_{table}_{operation}')
        connection.execute(sql); connection.commit()
        assert service._signature(store.path) == main_before
        with pytest.raises(ValueError, match='SHA|identity|proof|source'):
            read(reader, original)


def test_checkpoint_main_file_replacement_and_missing_database_invalidate(saved, monkeypatch, tmp_path):
    _, store, reader, original, _ = saved
    calls = counted(reader, monkeypatch)
    expected = read(reader, original)
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    assert read(reader, original) == expected
    # Replace with equivalent verified bytes and a different inode.
    other = tmp_path/'replacement.db'
    with closing(sqlite3.connect(store.path)) as source, closing(sqlite3.connect(other)) as destination, destination:
        source.backup(destination)
    os.replace(other, store.path)
    assert read(reader, original) == expected
    store.path.unlink()
    for suffix in ('-wal', '-shm'):
        Path(str(store.path)+suffix).unlink(missing_ok=True)
    assert read(reader, original)['boards'] == []
    assert len(calls) >= 3
    assert not store.path.exists()


def test_journal_change_invalidates_without_changed_main_file(saved, monkeypatch):
    _, store, reader, original, _ = saved
    calls = counted(reader, monkeypatch)
    expected = read(reader, original)
    journal = Path(str(store.path)+'-journal')
    journal.write_bytes(bytes(28))  # A zero header is explicitly non-hot in SQLite.
    assert read(reader, original) == expected
    journal.unlink()
    assert read(reader, original) == expected
    assert len(calls) == 3


def test_changed_database_during_verification_retries_and_never_caches_mixed_result(saved, monkeypatch):
    _, store, reader, original, clock = saved
    build = reader._boards; calls = []
    def change_after_read(report):
        result = build(report); calls.append(1)
        if len(calls) == 1:
            clock[0] = datetime(2026, 10, 14, 6, 10, tzinfo=timezone.utc)
            store.import_snapshot(snapshot(board='THS_C_885001', members=['000001.SZ']))
        return result
    monkeypatch.setattr(reader, '_boards', change_after_read)
    assert len(read(reader, original)['boards']) == 2
    assert len(calls) == 2
    assert len(read(reader, original)['boards']) == 2 and len(calls) == 2


def test_continuously_changing_database_fails_explicitly(saved, monkeypatch):
    _, _, reader, original, _ = saved
    signature = reader._knowledge_signature; count = [0]
    def changing():
        count[0] += 1
        return signature(), count[0]
    monkeypatch.setattr(reader, '_knowledge_signature', changing)
    with pytest.raises(ValueError, match='changed during'):
        read(reader, original)
    assert not any(key[0] == 'boards' for key in reader._cache)


@pytest.mark.parametrize('budget,entries', [(0,3),(1,3),(96*1024*1024,0),(96*1024*1024,1),(96*1024*1024,2)])
def test_report_and_projection_share_one_bounded_lru(saved, budget, entries):
    results, store, _, original, _ = saved
    reader = service.BoardMethodSummary(results.db_path, results.runs_root, store.path,
        cache_bytes=budget, cache_entries=entries)
    assert read(reader, original)['boards']
    assert read(reader, original, board_id=BOARD)['rows']
    assert len(reader._cache) <= entries and reader._bytes <= budget
    assert reader._bytes == sum(value[1] for value in reader._cache.values())


@pytest.mark.parametrize('field', ['method','date','population'])
def test_warm_cache_cannot_bypass_report_binding(saved, field):
    results, _, reader, original, _ = saved
    read(reader, original)
    changed = deepcopy(original)
    if field == 'method': changed['analysis_method'] = 'structure_kronos_ma_l2'
    elif field == 'date': changed['data_end'] = '2026-10-14'
    else: changed['frozen_identity']['population'] = ['000099.SZ']
    results.save('scan', changed)
    with pytest.raises(ValueError): read(reader, original)


def test_warm_cache_cannot_bypass_report_sha_or_missing_file(saved):
    results, _, reader, original, _ = saved
    read(reader, original)
    path = results.runs_root/original['run_id']/'reports/czsc.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='SHA'): read(reader, original)
    path.unlink()
    with pytest.raises(FileNotFoundError): read(reader, original)


def test_missing_knowledge_can_appear_after_empty_projection(saved, tmp_path):
    results, _, _, original, _ = saved
    path = tmp_path/'late'/'kb.db'
    reader = service.BoardMethodSummary(results.db_path, results.runs_root, path)
    assert read(reader, original)['boards'] == []
    assert not path.parent.exists()
    store = kb.BoardKnowledgeStore(path)
    store.import_snapshot(snapshot(members=['000001.SZ']))
    assert read(reader, original)['boards'][0]['counts']['buy'] == 1


def test_qualified_board_retains_six_of_seven_coverage_after_cache_hit(saved):
    results, store, reader, original, clock = saved
    changed = deepcopy(original)
    for number in range(4, 7):
        row = deepcopy(changed['rows'][0]); row['symbol'] = f'00000{number}.SZ'
        changed['rows'].append(row)
    changed['population'] = [row['symbol'] for row in changed['rows']]
    changed['frozen_identity']['population'] = changed['population']
    changed['coverage'].update(requested=6, evaluated=6)
    results.save('scan', changed)
    clock[0] = datetime(2026, 10, 14, 6, 10, tzinfo=timezone.utc)
    store.import_snapshot(snapshot(members=[f'00000{number}.SZ' for number in range(1,8)]))
    first = read(reader, changed)
    assert first['boards'][0]['ranking_eligible'] is True
    assert first['boards'][0]['coverage']['method_complete_ratio'] == 6/7
    assert read(reader, changed) == first
    assert read(reader, changed, board_id=BOARD)['board'] == first['boards'][0]
