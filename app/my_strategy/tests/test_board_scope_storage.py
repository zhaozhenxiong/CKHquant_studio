"""New calculation populations cannot silently reuse mixed historical indices."""
import copy

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.storage.board_fluent import BoardFluentStore
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_fluent import bundle,native_result


def legacy_bundle(market):
    value=copy.deepcopy(bundle(market))
    for row in [value,*value['boards'],*value['stocks']]:
        for key in ('calculation_scope','calculation_membership','projection_hash','source_snapshot_hash'):
            row.pop(key,None)
    value['cache_key']=stable_hash({'historical_fixture':value['cache_key']})
    return value


def test_historical_generation_requires_explicit_selection(market,tmp_path):
    store=BoardFluentStore(tmp_path/'fluent.db');old=legacy_bundle(market)
    generation=store.save_market(old)
    before=store.load_market(old['cache_key'])
    assert store.current_generation() is None
    assert store.boards('concept')==[] and store.rankings('concept')==[]
    assert store.board(old['boards'][0]['board_id']) is None
    assert store.stock(market['symbols'][0]) is None
    assert store.board(old['boards'][0]['board_id'],generation_id=generation)
    assert store.stock(market['symbols'][0],generation_id=generation)
    assert len(store.rankings('concept',generation_id=generation))==2
    assert store.list_generations()[0]['generation_id']==generation
    assert store.load_market(old['cache_key'])==before==old
    current=bundle(market);new=store.save_market(current)
    assert store.current_generation()['generation_id']==new
    assert store.load_market(old['cache_key'])==old


@pytest.mark.parametrize('field',['eligible_members','excluded_members','projection_hash'])
def test_mutated_projection_cannot_be_saved_as_current(market,tmp_path,field):
    value=bundle(market)
    projection=value['boards'][0]['calculation_membership']
    if field=='projection_hash':projection[field]='0'*64
    else:projection[field]=list(projection[field])+['430001.BJ']
    store=BoardFluentStore(tmp_path/'fluent.db')
    with pytest.raises(ValueError):store.save_market(value)
    assert store.current_generation() is None and store.list_generations()==[]


def test_mixed_native_identity_cannot_attach_to_projected_market(market,tmp_path):
    value=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db')
    generation=store.save_market(value);native,rows=native_result(market,value['boards'])
    native['identity'].pop('calculation_scope',None)
    native['identity'].pop('projection_hashes',None)
    with pytest.raises(ValueError,match='scope or population'):store.attach_native(generation,native,rows)
    assert store.current_generation('full') is None
    assert store.board(value['boards'][0]['board_id'])['native_structure'] is None


def test_different_board_projection_cannot_attach_even_with_same_source(market,tmp_path):
    value=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db')
    generation=store.save_market(value);native,rows=native_result(market,value['boards'])
    rows[0]['projection_hash']='0'*64
    rows[0]['row_hash']=stable_hash({k:v for k,v in rows[0].items() if k!='row_hash'})
    with pytest.raises(ValueError,match='projection mismatch'):store.attach_native(generation,native,rows)
    assert not store.get_generation(generation)['full_ready']
