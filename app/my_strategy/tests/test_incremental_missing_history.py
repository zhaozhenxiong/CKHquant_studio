"""Only exact missing closed history can become an unavailable native member."""
import sqlite3

import pytest

from my_strategy.adapters.czsc_adapter import MissingHistoricalBars, load_bars
from my_strategy.core.run_context import stable_hash
from my_strategy.services import incremental_sync as sync
from my_strategy.storage.board_fluent import BoardFluentStore
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_fluent import actual_market
from my_strategy.tests.test_czsc_core import raw_db
from my_strategy.tests.test_incremental_sync import planned


@pytest.mark.parametrize('case', ['absent', 'future_only', 'before_first_close'])
def test_exact_absence_of_closed_observations_has_a_distinct_exception(raw_db, case):
    symbol = '000002.SZ' if case == 'absent' else '000001.SZ'
    cutoff = '2019-12-31' if case == 'future_only' else '2020-01-02T14:59:59+08:00'
    with pytest.raises(MissingHistoricalBars):
        load_bars(symbol, as_of=cutoff, db_path=raw_db)


@pytest.mark.parametrize('case', ['negative_price', 'bad_range', 'nonfinite', 'duplicate', 'schema'])
def test_corrupt_input_is_never_a_missing_history_exception(raw_db, case):
    with sqlite3.connect(raw_db) as con:
        if case == 'negative_price': con.execute('UPDATE stock_daily_normalized SET close=-1 WHERE rowid=1')
        elif case == 'bad_range': con.execute('UPDATE stock_daily_normalized SET high=low-1 WHERE rowid=1')
        elif case == 'nonfinite': con.execute('UPDATE stock_daily_normalized SET amount=? WHERE rowid=1', (float('inf'),))
        elif case == 'duplicate': con.execute('INSERT INTO stock_daily_normalized SELECT * FROM stock_daily_normalized LIMIT 1')
        else: con.execute('ALTER TABLE stock_daily_normalized DROP COLUMN amount')
    with pytest.raises(ValueError) as error:
        load_bars('000001.SZ', db_path=raw_db)
    assert not isinstance(error.value, MissingHistoricalBars)


@pytest.mark.parametrize('error', [ValueError('bad prices'), OSError('unreadable source'), KeyError('missing field')])
def test_generic_inspect_errors_remain_failed_plan_rows(planned, monkeypatch, error):
    options, calls, cache, market = planned
    original = sync.IncrementalSource
    class BrokenSource(original):
        def inspect(self, *args, **kwargs): raise error
    monkeypatch.setattr(sync, 'IncrementalSource', BrokenSource)
    plan = sync.create_plan(**options, scope='full')
    assert plan['counts']['failed'] == plan['requested']
    assert plan['counts']['missing_price'] == 0
    assert all(row['action'] == 'failed' for row in plan['rows'])


def test_missing_member_survives_partial_then_full_in_original_population_with_reason(planned, monkeypatch):
    options, calls, cache, market = planned
    original = sync.IncrementalSource
    missing = sorted(market['symbols'])[-1]
    class MissingSource(original):
        def inspect(self, symbol, *args, **kwargs):
            if symbol == missing: raise MissingHistoricalBars('no exact closed rows')
            return super().inspect(symbol, *args, **kwargs)
        def source(self, symbol, *args, **kwargs):
            assert symbol != missing, 'missing source must never be computed or fabricated'
            return super().source(symbol, *args, **kwargs)
    monkeypatch.setattr(sync, 'IncrementalSource', MissingSource)
    first = sync.create_plan(**options, scope='full', max_symbols=1)
    absent = next(r for r in first['rows'] if r['symbol'] == missing)
    assert absent['observed_end'] is None and absent['reason'] == 'verified_history_unavailable'
    partial = sync.execute_plan(**options, plan_id=first['plan_id'])
    assert partial['status'] == 'partial' and not partial['full_published']
    assert partial['counts']['missing_price'] == 1
    assert sum(partial['counts'].values()) == first['requested']
    later = sync.create_plan(**options, scope='full', max_symbols=100)
    result = sync.execute_plan(**options, plan_id=later['plan_id'])
    assert result['status'] == 'completed' and result['full_published']
    assert result['counts']['missing_price'] == 1 and result['counts']['failed'] == 0
    assert next(r for r in result['rows'] if r['symbol'] == missing)['reason'] == 'verified_history_unavailable'
    fluent = BoardFluentStore(market['fluent'], readonly=True)
    full = fluent.current_generation('full')
    assert full['native_identity']['required_symbols'] == sorted(market['symbols'])
    native = fluent.get_native_component(full['generation_id'])
    unavailable = native['summaries'][missing]
    assert unavailable['status'] == 'unavailable' and unavailable['structure'] is unavailable['wyckoff'] is None
    assert unavailable['reason_codes'] == ['verified_history_unavailable']
    assert unavailable['row_hash'] == stable_hash({k:v for k,v in unavailable.items() if k != 'row_hash'})
    board = fluent.board(market['store'].catalog(current=True)[0]['board_id'])['native_structure'][0]
    assert board['expected_members'] == len(market['symbols'])
    assert board['native_members'] == len(market['symbols']) - 1


def test_full_with_empty_calculation_projection_cannot_publish_fake_full(planned, tmp_path):
    from my_strategy.storage.board_knowledge import BoardKnowledgeStore
    from my_strategy.tests.test_board_knowledge import snapshot
    options, calls, cache, market = planned
    kb = BoardKnowledgeStore(tmp_path/'only-excluded.db')
    kb.import_snapshot(snapshot(members=['830001.BJ']))
    options = {**options, 'kb_db_path':kb.path, 'fluent_db_path':tmp_path/'empty-fluent.db'}
    plan = sync.create_plan(**options, scope='full')
    assert plan['requested'] == plan['publication']['global_population'] == 0
    result = sync.execute_plan(**options, plan_id=plan['plan_id'])
    assert result['full_published'] is False and result['generation_id'] is None
    assert not options['fluent_db_path'].exists() and calls == []
