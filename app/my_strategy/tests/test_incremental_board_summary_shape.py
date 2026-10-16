"""Real incremental native evaluations must satisfy the board summary contract."""
from copy import deepcopy
import sqlite3

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import board_native_summary as boards
from my_strategy.services.board_calculation_scope import calculation_contract
from my_strategy.services.board_features import build_board_features
from my_strategy.storage.board_fluent import BoardFluentStore
from my_strategy.tests.test_board_features import market, MemoryStore
from my_strategy.tests.test_incremental_source import source


def test_real_source_summary_is_normalized_and_preserves_causal_scores_events_and_manifest(source):
    service, store, raw, dates = source
    day = dates[-1]
    result = service.summary('002436.SZ', day, calendar_dates=dates)
    manifest = service._peek(result['identity']['source_cache_key'])
    original = deepcopy(manifest)
    evaluation = manifest['latest_evaluation']
    for name in ('structure', 'wyckoff'):
        expected = boards._evaluation(evaluation[name], '002436.SZ', name, day)
        assert result[name] == expected
        assert result[name]['buy_score'] == (evaluation[name]['current_state']['buy_score'] if expected['input_eligible'] else None)
        assert result[name]['active_buy_event_ids'] == evaluation[name]['active_buy_event_ids']
    assert result['row_hash'] == stable_hash({k:v for k,v in result.items() if k != 'row_hash'})
    assert service._peek(result['identity']['source_cache_key']) == original


def test_actual_incremental_summaries_aggregate_and_publish_full_without_mock_evidence(source, market, tmp_path):
    service, cache, raw, dates = source
    symbols = market['symbols'][:5]
    day = dates[-1]
    raw_path = tmp_path / 'source-prices.db'
    with sqlite3.connect(raw_path) as con:
        for symbol in symbols:
            frame = raw.copy(); frame['symbol'] = symbol
            frame.rename(columns={'symbol':'stock'}).to_sql('stock_daily_normalized', con, index=False, if_exists='append')
    snapshot = market['snapshot']('industry', '111111', members=symbols, at=dates[0]+'T15:00:00+08:00')
    bundle = build_board_features(MemoryStore([snapshot]), raw_db_path=raw_path, calendar_dates=dates,
        price_date=day, mode='current_reference')
    summaries = {symbol:service.summary(symbol, day, calendar_dates=dates) for symbol in symbols}
    memberships = {row['board_id']:row['calculation_membership'] for row in bundle['boards']}
    refs = [{'entity_id':symbol,'source_identity':value['identity'],'row_hash':value['row_hash']}
        for symbol,value in sorted(summaries.items())]
    identity = {'format':'incremental_native_summary_collection_v1', 'price_date':day,
        'mode':'current_reference', 'training_eligible':False, 'scope':'full',
        'completed_symbols':symbols, 'input_refs_hash':stable_hash(refs), **calculation_contract(memberships)}
    native = {'identity':identity, 'summaries':summaries, 'input_refs':refs, 'complete':True}
    aggregate = boards.aggregate_board_native(snapshot, native)
    for name in ('structure', 'wyckoff'):
        eligible = [r[name] for r in summaries.values() if r[name]['input_eligible']]
        assert aggregate[name+'_eligible_members'] == len(eligible)
        if aggregate[name+'_available']:
            assert aggregate[name+'_buy_score_mean'] == pytest.approx(sum(r['buy_score'] for r in eligible)/len(eligible))
    assert aggregate['wyckoff_available'] is True
    published = BoardFluentStore(tmp_path/'fluent.db')
    generation = published.save_market(bundle)
    published.attach_native(generation, native, [aggregate])
    assert published.current_generation('market')['generation_id'] == generation
    assert published.current_generation('full')['generation_id'] == generation
    assert published.board(snapshot['board_id'])['native_component_complete'] is True
    assert published.board(snapshot['board_id'])['native_structure'] == [aggregate]


@pytest.mark.parametrize('changed', ['date', 'future', 'score', 'active_event'])
def test_summary_rejects_invalid_cached_target_evidence_before_board_aggregation(source, monkeypatch, changed):
    service, store, raw, dates = source
    day = dates[-1]
    result = service.source('002436.SZ', day, calendar_dates=dates)
    manifest = service._peek(result[3]['source_cache_key'])
    value = manifest['latest_evaluation']['wyckoff']
    if changed == 'date': value['date'] = dates[-2]
    elif changed == 'future': value['available_at'] = day+'T15:00:01+08:00'
    elif changed == 'score': value['current_state']['buy_score'] = 2.
    else: value['active_buy_event_ids'] = ['not-a-real-support-fact']
    monkeypatch.setattr(service, '_peek', lambda key: manifest)
    monkeypatch.setattr(service, 'source', lambda *args,**kwargs:result)
    with pytest.raises(ValueError): service.summary('002436.SZ', day, calendar_dates=dates)
