"""Identified Fuyao current snapshots, bounded requests and frozen resume."""
from __future__ import annotations

import copy
import json
import sqlite3

import pytest
import requests

from my_strategy.data_manager import board_provider as provider_module
from my_strategy.data_manager.board_provider import (FuyaoBoardProvider, BoardAccessError,
    BoardSourceError, BoardBudgetExhausted, SOURCE, TAXONOMY, CATALOG_PATH, MEMBERS_PATH)
from my_strategy.scripts.refresh_board_knowledge import refresh_knowledge, build_parser, _read_stock_basic
from my_strategy.storage.board_knowledge import BoardKnowledgeStore


class Response:
    def __init__(self, payload=None, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}
    def json(self):
        if isinstance(self.payload, Exception): raise self.payload
        return self.payload


class Session:
    def __init__(self, responses): self.responses, self.calls, self.closed = list(responses), [], False
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return response
    def close(self): self.closed = True


def envelope(items, **data):
    return {'code': 0, 'request_id': 'fixture-request', 'message': 'success',
        'data': {'timestamp': 1791950000000, 'item': items, **data}}


def catalogs(rows=None):
    return envelope(rows or [{'thscode': '881155.TI', 'name': '银行'}, {'thscode': '881156.TI', 'name': '券商'}])


def members(codes=('000001.SZ', '600000.SH')):
    return envelope([{'thscode': code, 'ticker': code.split('.')[0], 'name': '测试股票'} for code in codes])


@pytest.fixture
def make_provider(monkeypatch):
    monkeypatch.setenv('FUYAO_API_KEY', 'test-only-credential')
    def factory(responses, **kwargs):
        return FuyaoBoardProvider(session=Session(responses), min_interval=0, **kwargs)
    return factory


@pytest.fixture
def raw_db(tmp_path):
    path = tmp_path / 'raw.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE stock_basic(code TEXT PRIMARY KEY,name TEXT,updated_at TEXT,source TEXT)')
        connection.executemany('INSERT INTO stock_basic VALUES(?,?,?,?)', [
            ('000001.SZ', '平安银行', '2026-10-12', 'raw_fixture'),
            ('600000.SH', '浦发银行', '2026-10-12T19:00:00+08:00', 'raw_fixture')])
    return path


def test_catalog_is_full_stable_sorted_and_keeps_exact_api_origin(make_provider):
    first = catalogs([{'thscode': '881156.TI', 'name': '券商'}, {'thscode': '881155.TI', 'name': '银行'}])
    second = catalogs()
    provider = make_provider([Response(first), Response(second)])
    boards = provider.catalog(['industry'])
    assert [item['board_id'] for item in boards] == ['THS_I_881155.TI', 'THS_I_881156.TI']
    assert all(item['source'] == SOURCE and item['taxonomy'] == TAXONOMY for item in boards)
    proof = boards[0]['catalog_provenance']
    assert proof['catalog_set_stable'] and proof['board_count'] == 2
    assert provider_module._hash(proof['source_response']) == proof['first_response_sha256']
    assert provider_module._hash(proof['repeated_source_response']) == proof['repeated_response_sha256']
    url, arguments = provider.session.calls[0]
    assert url.endswith(CATALOG_PATH) and arguments['params'] == {'tag': 'industry'}
    assert arguments['headers']['X-api-key'] == 'test-only-credential'
    assert arguments['allow_redirects'] is False
    assert 'Cookie' not in arguments['headers'] and 'hexin-v' not in arguments['headers']


def test_catalog_has_no_historical_old_size_caps(make_provider):
    industry = envelope([{'thscode': f'{881100 + i}.TI', 'name': f'行业{i}'} for i in range(320)])
    concepts = envelope([{'thscode': f'{885100 + i}.TI', 'name': f'概念{i}'} for i in range(390)])
    provider = make_provider([Response(industry), Response(industry), Response(concepts), Response(concepts)])
    assert len(provider.catalog()) == 710
    assert provider.request_count == 4


@pytest.mark.parametrize('second', [
    catalogs([{'thscode': '881155.TI', 'name': '银行'}]),
    catalogs([{'thscode': '881155.TI', 'name': '改名'}, {'thscode': '881156.TI', 'name': '券商'}])])
def test_catalog_identity_or_name_change_rejects_frozen_directory(make_provider, second):
    provider = make_provider([Response(catalogs()), Response(second)])
    with pytest.raises(BoardSourceError, match='catalog changed'): provider.catalog(['industry'])


def test_members_repeat_all_codes_and_raw_proof_carries_more_than_200(make_provider):
    codes = [f'{600000 + i:06d}.SH' for i in range(3800)]
    first, second = members(tuple(reversed(codes))), members(tuple(codes))
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(first), Response(second)])
    board = provider.catalog(['industry'])[0]
    snapshot = provider.snapshot(board)
    assert snapshot['members'] == sorted(codes)
    assert snapshot['expected_count'] == 3800 and snapshot['complete'] is True
    proof = snapshot['provenance']
    assert proof['membership_set_hash'] == proof['repeated_membership_set_hash']
    assert proof['unique_members'] == 3800 and proof['pagination_complete'] is True
    assert 'reported_count' not in proof
    assert proof['first_response_sha256'] == provider_module._hash(proof['source_response'])
    assert proof['catalog_provenance']['board_count'] == 2
    assert snapshot['current_only'] and not snapshot['historical_membership_verified']
    assert snapshot['effective_at'] == snapshot['observed_at'] == snapshot['available_at']
    assert len(provider.session.calls) == 4
    assert 'membership_scope' not in snapshot and 'excluded_source_members' not in snapshot


def test_nq_projection_retains_full_source_proof_and_distinct_source_count(make_provider, tmp_path):
    first = members(('600000.SH', '834683.NQ', '000001.SZ'))
    first['data']['total'] = 3
    second = members(('834683.NQ', '000001.SZ', '600000.SH'))
    second['data']['total'] = 3
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(first), Response(second)])
    snapshot = provider.snapshot(provider.catalog(['industry'])[0])
    assert snapshot['members'] == ['000001.SZ', '600000.SH'] and snapshot['expected_count'] == 2
    assert snapshot['membership_scope'] == 'a_share_projection_v1' and snapshot['source_member_count'] == 3
    assert snapshot['excluded_source_members'] == ['834683.NQ'] and snapshot['complete'] is True
    proof = snapshot['provenance']
    assert proof['reported_count'] == 3 and proof['unique_members'] == 2
    assert len(proof['source_response']['data']['item']) == len(proof['repeated_source_response']['data']['item']) == 3
    assert proof['membership_set_hash'] == proof['repeated_membership_set_hash'] == provider_module._hash(snapshot['members'])
    assert proof['full_source_membership_set_hash'] == proof['repeated_full_source_membership_set_hash'] == provider_module._hash(['000001.SZ', '600000.SH', '834683.NQ'])
    assert snapshot['payload_sha256'] == provider_module._hash(first)
    store = BoardKnowledgeStore(tmp_path / 'projected.db')
    store.import_snapshot(snapshot)
    stored = store.catalog(current=True)[0]
    assert stored['members'] == snapshot['members'] and stored['excluded_source_members'] == ['834683.NQ']


def test_changed_excluded_set_is_rejected_even_when_a_share_projection_matches(make_provider):
    first, second = members(('600000.SH', '834683.NQ')), members(('600000.SH', '834684.NQ'))
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(first), Response(second)])
    with pytest.raises(BoardSourceError, match='set changed'):
        provider.snapshot(provider.catalog(['industry'])[0])


def test_verified_empty_a_share_projection_preserves_nonempty_source(make_provider, tmp_path):
    response = members(('834683.NQ',))
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(response), Response(response)])
    snapshot = provider.snapshot(provider.catalog(['industry'])[0])
    assert snapshot['expected_count'] == 0 and snapshot['members'] == [] and snapshot['source_member_count'] == 1
    assert snapshot['provenance']['member_list_state'] == 'empty_verified'
    store = BoardKnowledgeStore(tmp_path / 'projected-empty.db')
    store.import_snapshot(snapshot)
    assert store.catalog(current=True)[0]['excluded_source_members'] == ['834683.NQ']


@pytest.mark.parametrize('bad', [
    members(('600000.SH', '600000.SH')),
    members(('600000.SH', '600001.SZ')),
    envelope([{'thscode': '600000.SH', 'ticker': '600001'}]),
    envelope([{'thscode': '00700.HK', 'ticker': '00700'}]),
    envelope([{'thscode': '200001.SZ', 'ticker': '200001'}]),
    envelope([{'thscode': '834683.UNKNOWN', 'ticker': '834683'}]),
    envelope([{'thscode': '34683.NQ', 'ticker': '34683'}]),
    envelope([{'thscode': '834683.NQ', 'ticker': '834684'}]),
    members(('834683.NQ', '834683.NQ')),
])
def test_member_duplicates_foreign_codes_and_conflicts_are_not_silently_omitted(make_provider, bad):
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(bad), Response(bad)])
    board = provider.catalog(['industry'])[0]
    with pytest.raises(BoardSourceError): provider.snapshot(board)


def test_membership_set_change_is_rejected_even_with_equal_count(make_provider):
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(members()), Response(members(('600000.SH', '300001.SZ')))])
    with pytest.raises(BoardSourceError, match='set changed'): provider.snapshot(provider.catalog(['industry'])[0])


@pytest.mark.parametrize('response', [Response({}, 401), Response({}, 403), Response({}, 429),
    Response({'code': 2001}), Response({'code': 2003}), Response({'code': 4001})])
def test_access_and_rate_limits_stop_without_retry_or_other_vendor(make_provider, response):
    provider = make_provider([response, Response(catalogs())])
    with pytest.raises(BoardAccessError): provider.catalog(['industry'])
    assert len(provider.session.calls) == 1
    assert provider.request_count == 1


@pytest.mark.parametrize('response', [
    Response(ValueError('not JSON')), Response({'code': 0, 'data': {'item': []}}),
    Response(envelope([], total=5)), Response(envelope([{'thscode': '881155.TI', 'name': '银行'}], has_more=True)),
    Response({'code': 3002, 'message': 'not ready', 'data': None}),
])
def test_invalid_or_incomplete_envelope_is_explicit_failure(make_provider, response):
    provider = make_provider([response, Response(catalogs())])
    with pytest.raises(BoardSourceError): provider.catalog(['industry'])
    assert len(provider.session.calls) == 1


def test_redirect_is_refused_without_forwarding_credentials_to_another_origin(make_provider):
    provider = make_provider([Response({}, 302, {'Location': 'https://foreign.example/collector'}), Response(catalogs())])
    with pytest.raises(BoardSourceError, match='HTTP response invalid: 302'):
        provider.catalog(['industry'])
    assert len(provider.session.calls) == 1
    url, arguments = provider.session.calls[0]
    assert url.startswith('https://fuyao.aicubes.cn/') and arguments['allow_redirects'] is False


def test_empty_members_are_valid_only_after_two_full_success_responses(make_provider, tmp_path):
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(members(())), Response(members(()))])
    snapshot = provider.snapshot(provider.catalog(['industry'])[0])
    assert snapshot['members'] == [] and snapshot['expected_count'] == 0
    assert snapshot['provenance']['member_list_state'] == 'empty_verified'
    store = BoardKnowledgeStore(tmp_path / 'empty.db')
    run = store.begin_refresh(SOURCE, [snapshot['board_id']])
    result = store.import_snapshot(snapshot, run_id=run)
    assert store.finish_refresh(run)['status'] == 'succeeded'
    assert result


def test_budget_is_bounded_and_no_credential_enters_failure_text(make_provider):
    provider = make_provider([Response(catalogs()), Response(catalogs())], request_budget=1)
    with pytest.raises(BoardBudgetExhausted): provider.catalog(['industry'])
    assert len(provider.session.calls) == 1
    provider = make_provider([Response({'code': 2003, 'message': 'test-only-credential'})])
    with pytest.raises(BoardAccessError) as error: provider.catalog(['industry'])
    assert 'test-only-credential' not in str(error.value)


def test_current_collector_stock_basic_read_preserves_source_and_is_idempotent(raw_db, tmp_path):
    first, second = _read_stock_basic(raw_db), _read_stock_basic(raw_db)
    assert first == second
    assert first[0]['source_updated_at'] == '2026-10-12' and 'updated_at' not in first[0]
    assert second[1]['updated_at'] == '2026-10-12T19:00:00+08:00'
    assert first[0]['source'] == 'raw_fixture'
    store = BoardKnowledgeStore(tmp_path / 'idempotent.db')
    assert store.seed_stock_basic(first)['inserted'] == 2
    assert store.seed_stock_basic(second)['inserted'] == 0


def test_refresh_budget_interrupt_resumes_same_frozen_catalog_and_skips_success(make_provider, raw_db, tmp_path):
    db = tmp_path / 'resume.db'
    first = make_provider([Response(catalogs()), Response(catalogs()), Response(members()), Response(members())], request_budget=4)
    progress = []
    partial = refresh_knowledge(db_path=db, raw_db_path=raw_db, kinds=['industry'], provider=first, on_progress=progress.append)
    assert partial['status'] == 'interrupted' and partial['completed_boards'] == 1
    saved = BoardKnowledgeStore(db).get_refresh(partial['run_id'])
    assert saved['status'] == 'running' and len(saved['requested_boards']) == 2
    second = make_provider([Response(members()), Response(members())])
    result = refresh_knowledge(db_path=db, raw_db_path=raw_db, resume=partial['run_id'], provider=second)
    assert result['run_id'] == partial['run_id'] and result['status'] == 'succeeded'
    assert result['completed_boards'] == 2 and second.request_count == 2
    assert second.session.calls[0][1]['params']['thscode'] == '881156.TI'
    assert BoardKnowledgeStore(db).status()['board_count'] == 2
    assert progress[0]['completed'] == 0 and progress[1]['completed'] == 1


def test_source_auth_failure_is_audited_and_finished_retry_creates_a_new_catalog(make_provider, raw_db, tmp_path):
    db = tmp_path / 'source-failure.db'
    first = make_provider([Response({'code': 2003})])
    failed = refresh_knowledge(db_path=db, raw_db_path=raw_db, kinds=['industry'], provider=first)
    assert failed['status'] == 'failed' and failed['failures'][0]['board_id'] is None
    second = make_provider([Response(catalogs()), Response(catalogs()), Response(members()), Response(members()), Response(members()), Response(members())])
    result = refresh_knowledge(db_path=db, raw_db_path=raw_db, kinds=['industry'], provider=second, resume=failed['run_id'])
    assert result['status'] == 'succeeded' and result['run_id'] != failed['run_id']
    assert BoardKnowledgeStore(db).get_refresh(failed['run_id'])['status'] == 'failed'


def test_validated_json_import_needs_no_api_key_and_preserves_current_semantics(make_provider, raw_db, tmp_path, monkeypatch):
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(members()), Response(members())])
    snapshot = provider.snapshot(provider.catalog(['industry'])[0])
    path = tmp_path / 'authorized-export.json'; path.write_text(json.dumps({'snapshots': [snapshot]}))
    monkeypatch.delenv('FUYAO_API_KEY')
    result = refresh_knowledge(db_path=tmp_path / 'import.db', raw_db_path=raw_db, import_path=path)
    assert result['status'] == 'succeeded' and result['request_count'] == 0
    assert result['current_only'] and not result['historical_membership_verified']


def test_invalid_import_and_resume_cannot_mutate_original_success(make_provider, raw_db, tmp_path):
    provider = make_provider([Response(catalogs()), Response(catalogs()), Response(members()), Response(members())])
    snapshot = provider.snapshot(provider.catalog(['industry'])[0])
    db = tmp_path / 'reject.db'; path = tmp_path / 'export.json'; path.write_text(json.dumps([snapshot]))
    result = refresh_knowledge(db_path=db, raw_db_path=raw_db, import_path=path)
    broken = copy.deepcopy(snapshot); broken['provenance']['source_response']['data']['item'].pop()
    path.write_text(json.dumps([broken]))
    bad = refresh_knowledge(db_path=db, raw_db_path=raw_db, import_path=path)
    assert bad['status'] == 'failed'
    assert BoardKnowledgeStore(db).get_refresh(result['run_id'])['status'] == 'succeeded'
    assert BoardKnowledgeStore(db).status()['board_count'] == 1


def test_parser_offers_only_current_industry_concept_and_no_plaintext_key_argument():
    parser = build_parser()
    args = parser.parse_args(['--db', 'boards.db', '--raw-db', 'raw.db', '--board', 'THS_I_881155.TI',
        '--kinds', 'industry', '--resume', 'a' * 32, '--limit-budget', '50', '--min-interval', '1.3'])
    assert args.board == ['THS_I_881155.TI'] and args.limit_budget == 50 and args.min_interval == 1.3
    assert parser.parse_args([]).min_interval == 1.0
    with pytest.raises(SystemExit): parser.parse_args(['--api-key', 'never-a-real-credential'])
    with pytest.raises(SystemExit): parser.parse_args(['--kinds', 'region'])


def test_provider_defaults_to_one_second_start_interval_without_real_wait(monkeypatch):
    monkeypatch.setenv('FUYAO_API_KEY', 'test-only-credential')
    clock, sleeps = [10.0], []
    monkeypatch.setattr(provider_module.time, 'monotonic', lambda: clock[0])
    def sleep(seconds): sleeps.append(seconds); clock[0] += seconds
    monkeypatch.setattr(provider_module.time, 'sleep', sleep)
    session = Session([Response(catalogs()), Response(catalogs())])
    provider = FuyaoBoardProvider(session=session)
    assert provider.min_interval == 1.0
    assert len(provider.catalog(['industry'])) == 2
    assert sleeps == [1.0] and provider.request_count == 2


@pytest.mark.parametrize('interval', [1.0, 1.3])
def test_refresh_threads_explicit_interval_into_owned_provider(tmp_path, raw_db, monkeypatch, interval):
    from my_strategy.scripts import refresh_board_knowledge as refresh_module
    calls = []
    def provider(**kwargs):
        calls.append(kwargs)
        raise BoardAccessError('fixture refused')
    monkeypatch.setattr(refresh_module, 'FuyaoBoardProvider', provider)
    result = refresh_knowledge(db_path=tmp_path / 'kb.db', raw_db_path=raw_db,
        kinds=['industry'], min_interval=interval, limit_budget=10)
    assert result['status'] == 'failed'
    assert calls == [{'request_budget': 10, 'min_interval': interval}]


@pytest.mark.parametrize('interval', [-1, float('nan'), float('inf'), True])
def test_invalid_request_interval_does_not_create_state(tmp_path, interval):
    with pytest.raises(ValueError, match='interval'):
        refresh_knowledge(db_path=tmp_path / 'kb.db', min_interval=interval)
    assert not (tmp_path / 'kb.db').exists()
