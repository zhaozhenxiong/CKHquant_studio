from __future__ import annotations

import copy
import hashlib
import sqlite3

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import board_features as features


class MemoryStore:
    def __init__(self,snapshots):self.snapshots=snapshots;self.cache={}
    def catalog(self,kind=None,as_of=None,current=False):
        rows={}
        for row in sorted(self.snapshots,key=lambda value:value['available_at']):
            if kind and row['kind']!=kind:continue
            if not current and pd.Timestamp(row['available_at'])>pd.Timestamp(as_of):continue
            rows[row['board_id']]=copy.deepcopy(row)
        return list(rows.values())
    def get_feature(self,kind,key):return copy.deepcopy(self.cache.get((kind,key)))
    def put_feature(self,kind,key,payload):self.cache[(kind,key)]=copy.deepcopy(payload);return payload


@pytest.fixture
def market(tmp_path):
    dates=pd.bdate_range('2025-01-02',periods=65).strftime('%Y-%m-%d').tolist()
    symbols=['00000'+str(index)+'.SZ' for index in range(1,8)]
    path=tmp_path/'raw.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,'
            'close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER,PRIMARY KEY(stock,date))')
        for index,symbol in enumerate([*symbols,'600010.SH']):
            rate=.2 if index==0 else .01
            amount=3000. if index==7 else 1000.
            for offset,day in enumerate(dates):
                price=10.*(1+rate)**offset
                conn.execute('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                    [symbol,day,price,price*1.001,price*.999,price,100.,amount,'tushare',1])
    stamp=dates[0]+'T15:00:00+08:00'
    def snapshot(kind,code,members=None,at=stamp):
        members=symbols if members is None else members
        row=dict(board_id='THS_'+('I' if kind=='industry' else 'C')+'_'+code,kind=kind,name=code,
            source='tonghuashun_public',source_code=code,taxonomy='THS_PUBLIC',hierarchy_verified=True,level=1,
            observed_at=at,available_at=at,effective_at=at,complete=True,expected_count=len(members),members=members)
        row['snapshot_hash']=stable_hash(row)
        return row
    store=MemoryStore([snapshot('industry','111111'),snapshot('concept','222222'),snapshot('concept','333333')])
    return dict(path=path,dates=dates,symbols=symbols,store=store,snapshot=snapshot)


def build(market,**kwargs):
    return features.build_board_features(market['store'],raw_db_path=market['path'],
        calendar_dates=market['dates'],price_date=market['dates'][-1],**kwargs)


def test_twelve_real_formulas_and_peer_exclusion_without_raw_mutation(market):
    before=hashlib.sha256(market['path'].read_bytes()).hexdigest()
    result=build(market);board=result['boards'][0]
    daily=(.2+6*.01)/7
    for count in (1,5,10,20):assert board['board_return_'+str(count)]==pytest.approx((1+daily)**count-1)
    assert board['board_advance_ratio']==1 and board['board_above_ma20_ratio']==1
    assert board['board_breakout20_ratio']==1 and board['board_amount']==7000.
    assert board['board_market_amount_share']==pytest.approx(.7)
    assert board['board_relative_amount20']==1
    assert board['board_volatility20']==pytest.approx(0,abs=1e-12) and board['board_drawdown20']==0
    stock=next(row for row in result['stocks'] if row['symbol']==market['symbols'][0])
    assert stock['industry_return_20']==pytest.approx(1.01**20-1)
    assert stock['relative_industry_20']==pytest.approx(1.2**20-1.01**20)
    assert stock['industry_amount']==6000 and stock['industry_market_amount_share']==pytest.approx(2/3)
    assert stock['industry_peer_rank20']==1 and stock['peer_excludes_self'] is True
    assert stock['training_eligible'] is True and result['qualification']=='pending'
    assert len(features.require_board_training_features(stock,decision_at=market['dates'][-1]+'T15:00:00+08:00'))==22
    assert hashlib.sha256(market['path'].read_bytes()).hexdigest()==before


def test_concept_many_to_many_creates_one_sample_and_summaries_exclude_self(market):
    result=build(market)
    assert len(result['stocks'])==len(market['symbols'])
    assert len({row['symbol'] for row in result['stocks']})==len(market['symbols'])
    stock=result['stocks'][0]
    assert stock['concept_known_count']==2 and len(stock['concept_ids'])==2
    assert stock['concept_return_20_mean']==pytest.approx(1.01**20-1)
    assert stock['concept_advance_ratio_mean']==1 and stock['concept_positive_ratio20']==1


def test_future_current_members_are_diagnostic_only_not_historical_training(market):
    future='2026-10-14T16:00:00+08:00'
    market['store']=MemoryStore([market['snapshot']('industry','111111',at=future),
        market['snapshot']('concept','222222',at=future)])
    causal=build(market)
    assert causal['boards']==[] and causal['stocks']==[] and causal['training_eligible'] is False
    reference=build(market,mode='current_reference');stock=reference['stocks'][0]
    assert reference['membership_date']=='2026-10-14' and reference['price_date']==market['dates'][-1]
    assert stock['industry_return_20'] is not None and stock['training_eligible'] is False
    with pytest.raises(ValueError,match='diagnostic-only'):features.require_board_training_features(stock,decision_at=market['dates'][-1]+'T15:00:00+08:00')
    forged={**stock,'mode':'causal','training_eligible':True}
    with pytest.raises(ValueError,match='future-visible'):features.require_board_training_features(forged,decision_at=market['dates'][-1]+'T15:00:00+08:00')


def test_first_visible_snapshot_does_not_backfill_rolling_membership_history(market):
    today=market['dates'][-1]+'T15:00:00+08:00'
    market['store']=MemoryStore([market['snapshot']('industry','111111',at=today)])
    row=build(market)['boards'][0]
    assert row['board_return_1'] is not None and row['board_return_20'] is None
    assert row['board_relative_amount20'] is None and row['training_eligible'] is False


def test_causal_daily_membership_change_uses_old_visible_population_before_change(market):
    changed_day=market['dates'][-5]
    market['store'].snapshots.append(market['snapshot']('industry','111111',members=market['symbols'][1:],
        at=changed_day+'T15:00:00+08:00'))
    row=next(row for row in build(market)['boards'] if row['kind']=='industry')
    old=(.2+6*.01)/7
    assert row['board_return_20']==pytest.approx((1+old)**15*1.01**5-1)
    assert row['expected_members']==6


@pytest.mark.parametrize('missing',[1,2])
def test_coverage_threshold_and_current_quote_missing_are_explicit(market,missing):
    with sqlite3.connect(market['path']) as conn:
        for symbol in market['symbols'][1:1+missing]:conn.execute('DELETE FROM stock_daily_normalized WHERE stock=? AND date=?',[symbol,market['dates'][-1]])
    result=build(market);row=result['boards'][0]
    assert row['coverage']==pytest.approx((7-missing)/7) and row['effective_members']==7-missing
    if missing==1:assert row['board_return_1'] is not None
    else:assert row['board_return_1'] is None and row['board_amount'] is None
    stock=next(row for row in result['stocks'] if row['symbol']==market['symbols'][1])
    assert stock['relative_industry_1'] is None and stock['training_eligible'] is False
    assert 'current_price_unavailable' in stock['reason_codes']


def test_small_board_is_unavailable_and_true_zero_return_is_not_missing(market):
    market['store']=MemoryStore([market['snapshot']('industry','111111',members=market['symbols'][:4])])
    row=build(market)['boards'][0]
    assert row['coverage']==1 and row['board_return_1'] is None
    market['store']=MemoryStore([market['snapshot']('industry','111111')])
    with sqlite3.connect(market['path']) as conn:
        conn.execute('UPDATE stock_daily_normalized SET open=10,high=10,low=10,close=10')
    row=build(market)['boards'][0]
    assert row['board_return_20']==0 and row['board_advance_ratio']==0
    assert row['board_volatility20']==0 and row['board_drawdown20']==0


def test_same_cache_returns_original_payload_without_feature_recalculation(market,monkeypatch):
    first=build(market)
    def forbidden(values):pytest.fail('same input was recomputed')
    monkeypatch.setattr(features,'_compound',forbidden)
    assert build(market)==first


def test_price_revision_changes_key_and_rollup_while_unused_old_revision_does_not(market):
    first=build(market)
    with sqlite3.connect(market['path']) as conn:
        conn.execute('UPDATE stock_daily_normalized SET close=close*1.1,open=open*1.1,high=high*1.1,low=low*1.1 WHERE stock=? AND date=?',
            [market['symbols'][0],market['dates'][-3]])
    revised=build(market)
    assert first['cache_key']!=revised['cache_key']
    assert first['boards'][0]['board_volatility20']!=revised['boards'][0]['board_volatility20']
    with sqlite3.connect(market['path']) as conn:
        conn.execute('UPDATE stock_daily_normalized SET close=close*1.1,open=open*1.1,high=high*1.1,low=low*1.1 WHERE stock=? AND date=?',
            [market['symbols'][0],market['dates'][0]])
    assert build(market)==revised


def test_future_quotes_do_not_change_cached_past_feature(market):
    first=build(market)
    future=(pd.Timestamp(market['dates'][-1])+pd.offsets.BDay()).date().isoformat()
    with sqlite3.connect(market['path']) as conn:
        conn.execute('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
            [market['symbols'][0],future,999.,1000.,990.,999.,100.,1000.,'tushare',1])
    market['dates'].append(future)
    result=features.build_board_features(market['store'],raw_db_path=market['path'],calendar_dates=market['dates'],price_date=first['price_date'])
    assert result==first


def test_multiple_industry_indices_do_not_choose_an_arbitrary_primary_industry(market):
    other=market['snapshot']('industry','444444');other['hierarchy_verified']=False;other['level']=None
    other['snapshot_hash']=stable_hash({key:value for key,value in other.items() if key!='snapshot_hash'})
    market['store'].snapshots.append(other)
    result=build(market,mode='current_reference');stock=result['stocks'][0]
    assert len(result['boards'])==4 and stock['industry_match_count']==2
    assert stock['industry_id'] is None and stock['relative_industry_20'] is None
    assert len(stock['industry_matches'])==2 and stock['reason_messages']


def test_wrong_provider_and_invalid_policy_are_rejected(market):
    market['store'].snapshots[0]['source']='eastmoney'
    with pytest.raises(ValueError,match='Tonghuashun'):build(market)
    with pytest.raises(ValueError,match='min_members'):build(market,policy={'min_members':1})


def test_wrong_schema_policy_and_frozen_source_key_are_rejected_at_training_read(market):
    row=build(market)['stocks'][0];decision=market['dates'][-1]+'T15:00:00+08:00'
    for changed in [{'schema_hash':'wrong'},{'policy_hash':'wrong'},{'membership_dependency_hashes':[]},
            {'price_content_sha256':None},{'calendar_hash':None},{'code_hashes':{}}]:
        with pytest.raises(ValueError):features.require_board_training_features({**row,**changed},decision_at=decision)
    with pytest.raises(ValueError):features.require_board_training_features(row,decision_at=decision,expected_cache_key='wrong')
    assert features.require_board_training_features(row,decision_at=decision,expected_cache_key=row['cache_key'])


def test_identified_fuyao_ths_current_source_is_allowed_without_fuyao_prices(market):
    for row in market['store'].snapshots:
        row['source']='fuyao:ths';row['taxonomy']='THS_INDEX';row['source_code']+='.TI';row['board_id']+='.TI'
        row['hierarchy_verified']=False;row['level']=None
        row['snapshot_hash']=stable_hash({key:value for key,value in row.items() if key!='snapshot_hash'})
    result=build(market,mode='current_reference')
    assert len(result['boards'])==3 and result['training_eligible'] is False
    assert result['stocks'][0]['reason_messages']
    assert result['cache_identity']['price_basis']=='stored_unadjusted_no_total_return_factor'


def test_verified_empty_board_and_missing_prices_never_become_zero_features(market):
    market['store']=MemoryStore([market['snapshot']('industry','111111',members=[])])
    result=build(market,mode='current_reference',symbols=[market['symbols'][0]])
    assert all(result['boards'][0][name] is None for name in features.BOARD_FEATURE_COLUMNS)
    assert 'verified_empty_membership' in result['boards'][0]['reason_codes']
    with sqlite3.connect(market['path']) as conn:conn.execute('DELETE FROM stock_daily_normalized')
    result=build(market,mode='current_reference',symbols=[market['symbols'][0]])
    assert result['stocks'][0]['relative_industry_1'] is None
    assert result['stocks'][0]['training_eligible'] is False


def test_market_amount_excludes_b_shares_even_with_valid_quotes(market):
    with sqlite3.connect(market['path']) as conn:
        for day in market['dates']:
            conn.execute('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                ['900901.SH',day,10.,11.,9.,10.,100.,1_000_000.,'tushare',1])
    assert build(market)['boards'][0]['board_market_amount_share']==pytest.approx(.7)


def test_mixed_scope_prices_projection_and_content_identity_ignore_excluded_rows(market):
    excluded=['430001.BJ','920001.BJ','834683.NQ','900901.SH']
    source=market['snapshot']('industry','111111',members=market['symbols'][:5]+excluded)
    original=copy.deepcopy(source);market['store']=MemoryStore([source])
    with sqlite3.connect(market['path']) as conn:
        for symbol in excluded:
            conn.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                [[symbol,day,10.,11.,9.,10.,1e15,1e18,'tushare',1] for day in market['dates']])
    mixed=build(market,mode='current_reference');row=mixed['boards'][0]
    assert market['store'].snapshots[0]==original
    assert row['snapshot_hash']==original['snapshot_hash']
    assert row['expected_members']==row['effective_members']==5 and row['coverage']==1
    assert row['calculation_membership']['source_member_count']==9
    assert row['calculation_membership']['excluded_members']==sorted(excluded)
    assert row['board_amount']==5000 and row['board_market_amount_share']==.5
    assert {r['symbol'] for r in mixed['stocks']}==set(market['symbols'][:5])
    with sqlite3.connect(market['path']) as conn:
        conn.executemany('UPDATE stock_daily_normalized SET amount=amount*10 WHERE stock=?',[(s,) for s in excluded])
    changed=build(market,mode='current_reference',use_cache=False)
    assert changed==mixed
    with sqlite3.connect(market['path']) as conn:
        conn.executemany('DELETE FROM stock_daily_normalized WHERE stock=?',[(s,) for s in excluded])
    assert build(market,mode='current_reference',use_cache=False)==mixed
    with sqlite3.connect(market['path']) as conn:
        conn.execute('DELETE FROM stock_daily_normalized WHERE stock=? AND date=?',
            (market['symbols'][0],market['dates'][-1]))
    short=build(market,mode='current_reference')['boards'][0]
    assert short['expected_members']==5 and short['effective_members']==4 and short['coverage']==.8
    assert short['board_amount'] is None


def test_excluded_only_members_keep_null_and_source_proof(market):
    source=market['snapshot']('industry','111111',members=['430001.BJ','834683.NQ'])
    market['store']=MemoryStore([source]);result=build(market,mode='current_reference')
    row=result['boards'][0]
    assert row['expected_members']==0 and row['effective_members']==0
    assert row['calculation_membership']['source_member_count']==2
    assert 'no_supported_members' in row['reason_codes'] and 'membership_incomplete' not in row['reason_codes']
    assert all(row[name] is None for name in features.BOARD_FEATURE_COLUMNS)
    assert not result['stocks'] and not row['training_eligible']


def test_actual_knowledge_store_persists_bulk_and_readonly_latest_flat_rows(market,tmp_path):
    from my_strategy.storage.board_knowledge import BoardKnowledgeStore, FORMAT
    path=tmp_path/'basic_kb.db';store=BoardKnowledgeStore(path)
    now=(pd.Timestamp.now(tz='Asia/Shanghai')-pd.Timedelta(seconds=2)).isoformat()
    for row in market['store'].snapshots:
        value={key:value for key,value in row.items() if key!='snapshot_hash'}
        value.update(format=FORMAT,observed_at=now,available_at=now,effective_at=now,
            source_url='https://q.10jqka.com.cn/thshy/',
            provenance=dict(pagination_complete=True,reported_count=len(row['members']),unique_members=len(row['members'])))
        store.import_snapshot(value)
    market['store']=store
    result=build(market,mode='current_reference')
    assert result['training_eligible'] is False
    reader=BoardKnowledgeStore(path,readonly=True)
    assert reader.get_feature('daily',result['cache_key'])==result
    assert reader.latest_feature('daily',board_id=result['boards'][0]['board_id'],mode='current_reference')==result['boards'][0]
    assert reader.latest_feature('stock',symbol=market['symbols'][0],mode='current_reference')==result['stocks'][0]
    assert build(market,mode='current_reference')==result
    assert build(market)['training_eligible'] is False


def test_late_quote_is_not_available_at_its_declared_decision_close(market):
    with sqlite3.connect(market['path']) as conn:
        conn.execute('ALTER TABLE stock_daily_normalized ADD COLUMN available_at TEXT')
        conn.execute("UPDATE stock_daily_normalized SET available_at=date||'T15:00:00+08:00'")
        conn.execute('UPDATE stock_daily_normalized SET available_at=? WHERE stock=? AND date=?',
            [market['dates'][-1]+'T15:01:00+08:00',market['symbols'][0],market['dates'][-1]])
    row=build(market)['stocks'][0]
    assert row['relative_industry_1'] is None and row['training_eligible'] is False
    assert 'current_price_unavailable' in row['reason_codes']


def test_invalid_cached_metadata_is_rejected_without_silent_recomputation(market):
    result=build(market)
    market['store'].cache[('daily',result['cache_key'])]['mode']='current_reference'
    with pytest.raises(ValueError,match='cache identity drift'):build(market)


@pytest.mark.parametrize('mode',['causal','current_reference'])
def test_reverse_membership_index_preserves_scan_order_and_exact_results(market,monkeypatch,mode):
    market['store'].snapshots.insert(0,market['snapshot']('industry','444444',members=market['symbols'][2:]))
    market['store'].snapshots.append(market['snapshot']('concept','555555',members=market['symbols'][1:]))
    changed=market['snapshot']('concept','222222',members=market['symbols'][2:],
        at=market['dates'][-5]+'T15:00:00+08:00')
    market['store'].snapshots.append(changed)
    selected=[*reversed(market['symbols']),'000099.SZ']
    indexed=build(market,mode=mode,symbols=selected,use_cache=False)
    def original_scan(snapshots):
        symbols={symbol for snapshot in snapshots for symbol in snapshot['members']}
        return {symbol:[snapshot for snapshot in snapshots if symbol in snapshot['members']] for symbol in symbols}
    monkeypatch.setattr(features,'_member_board_index',original_scan)
    scanned=build(market,mode=mode,symbols=selected,use_cache=False)
    assert indexed==scanned
    assert [row['symbol'] for row in indexed['stocks']]==sorted(set(selected))
    assert len(indexed['stocks'])==len(selected)


def test_a_share_projection_keeps_full_source_count_and_prices_212_member_population(market):
    additional=[f'{index:06d}.SZ' for index in range(1000,1205)]
    with sqlite3.connect(market['path']) as conn:
        observations=conn.execute('SELECT date,open,high,low,close,volume,amount,source,has_trade_price '
            'FROM stock_daily_normalized WHERE stock=?',[market['symbols'][1]]).fetchall()
        conn.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
            [[symbol,*row] for symbol in additional for row in observations])
    members=[*market['symbols'],*additional]
    snapshot=market['snapshot']('concept','555555',members=members)
    snapshot.update(membership_scope='a_share_projection_v1',source_member_count=213,
        excluded_source_members=['834683.NQ'])
    snapshot['snapshot_hash']=stable_hash({key:value for key,value in snapshot.items() if key!='snapshot_hash'})
    market['store']=MemoryStore([snapshot])
    result=build(market,mode='current_reference',symbols=[members[0]])
    row=result['boards'][0]
    assert row['membership_scope']=='a_share_projection_v1' and row['source_member_count']==213
    assert row['excluded_source_members']==['834683.NQ']
    assert row['expected_members']==212 and row['effective_members']==212 and row['coverage']==1
    daily=(.2+211*.01)/212
    for count in (1,5,10,20):assert row['board_return_'+str(count)]==pytest.approx((1+daily)**count-1)
    for name in ('board_advance_ratio','board_above_ma20_ratio','board_breakout20_ratio','board_relative_amount20'):
        assert row[name]==1
    assert row['board_amount']==212000 and row['board_market_amount_share']==pytest.approx(212/215)
    assert row['board_volatility20']==pytest.approx(0,abs=1e-12) and row['board_drawdown20']==0
    assert all(row[name] is not None for name in features.BOARD_FEATURE_COLUMNS)
    with sqlite3.connect(market['path']) as conn:
        conn.executemany('DELETE FROM stock_daily_normalized WHERE stock=? AND date=?',
            [[symbol,market['dates'][-1]] for symbol in members[1:43]])
    projected=build(market,mode='current_reference',symbols=[members[0]])['boards'][0]
    assert projected['metric_counts']['return']==170 and projected['coverage']==pytest.approx(170/212)
    assert projected['board_return_1'] is not None and projected['board_amount'] is not None
    with sqlite3.connect(market['path']) as conn:
        conn.execute('DELETE FROM stock_daily_normalized WHERE stock=? AND date=?',[members[43],market['dates'][-1]])
    insufficient=build(market,mode='current_reference',symbols=[members[0]])['boards'][0]
    assert insufficient['metric_counts']['return']==169 and insufficient['board_return_1'] is None


def test_unprojected_board_metadata_explicitly_defaults_to_full_source(market):
    row=build(market)['boards'][0]
    assert row['membership_scope']=='source_full'
    assert row['source_member_count']==row['expected_members']
    assert row['excluded_source_members']==[]
