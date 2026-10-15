import sqlite3

from my_strategy.services.board_features import build_board_features
from my_strategy.storage.research_store import ResearchStore
from my_strategy.tests.test_board_features import market


def build(market,**kwargs):
    return build_board_features(market['store'],raw_db_path=market['path'],calendar_dates=market['dates'],
        price_date=market['dates'][-1],mode='current_reference',**kwargs)


def test_components_equal_uncached_and_reuse_unaffected_members(market,tmp_path):
    with ResearchStore(tmp_path/'components') as components:
        first_stats={};first=build(market,component_store=components,component_stats=first_stats)
        assert first_stats=={'rebuilt':3}
        assert first==build(market,use_cache=False)
        # A nonmember changes the market denominator but none of these boards'
        # member sums. Whole publication changes; all three local parts reuse.
        with sqlite3.connect(market['path']) as conn:
            conn.execute('UPDATE stock_daily_normalized SET amount=amount*2 WHERE stock=? AND date=?',
                ['600010.SH',market['dates'][-1]])
        stats={};second=build(market,component_store=components,component_stats=stats)
        assert stats=={'reused':3}
        assert second==build(market,use_cache=False)
        assert second['boards'][0]['board_amount']==first['boards'][0]['board_amount']
        assert second['boards'][0]['board_market_amount_share']!=first['boards'][0]['board_market_amount_share']


def test_membership_revision_rebuilds_only_affected_board(market,tmp_path):
    with ResearchStore(tmp_path/'components') as components:
        build(market,component_store=components)
        market['store'].snapshots.append(market['snapshot']('concept','333333',members=market['symbols'][1:],
            at=market['dates'][-1]+'T16:00:00+08:00'))
        stats={};result=build(market,component_store=components,component_stats=stats)
        assert stats=={'reused':2,'rebuilt':1}
        assert result==build(market,use_cache=False)


def test_same_input_reuses_publication_without_creating_components(market,tmp_path):
    with ResearchStore(tmp_path/'components') as components:
        original=build(market,component_store=components)
        before=components.list(limit=1000)
        stats={};assert build(market,component_store=components,component_stats=stats)==original
        assert stats=={} and components.list(limit=1000)==before
