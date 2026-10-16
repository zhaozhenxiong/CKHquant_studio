"""Independent storage boundary review; all database fixtures are temporary."""
import copy
import pytest
from my_strategy.core.run_context import stable_hash
from my_strategy.services.board_calculation_scope import calculation_contract
from my_strategy.services.board_native_summary import aggregate_board_native
from my_strategy.storage.board_fluent import BoardFluentStore
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_fluent import bundle


def test_new_scope_stock_row_must_not_contain_beijing(market,tmp_path):
    value=bundle(market)
    value['stocks'][0]['symbol']='430001.BJ'
    with pytest.raises(ValueError):
        BoardFluentStore(tmp_path/'fluent.db').save_market(value)


def test_new_scope_stock_projection_must_match_saved_board_projection(market,tmp_path):
    value=bundle(market)
    value['stocks'][0]['calculation_projection_hashes']={value['boards'][0]['board_id']:'0'*64}
    with pytest.raises(ValueError):
        BoardFluentStore(tmp_path/'fluent.db').save_market(value)


@pytest.mark.parametrize('field',['projection_hash','source_snapshot_hash'])
def test_new_scope_board_scalar_proof_must_match_projection(market,tmp_path,field):
    value=bundle(market)
    value['boards'][0][field]='0'*64
    with pytest.raises(ValueError):
        BoardFluentStore(tmp_path/'fluent.db').save_market(value)


def test_publication_population_must_match_calculation_contract(market,tmp_path):
    value=bundle(market)
    publication=calculation_contract(value['calculation_membership'])
    publication['required_symbols']=['430001.BJ']
    with pytest.raises(ValueError):
        BoardFluentStore(tmp_path/'fluent.db').save_market(value,publication_inputs=publication)


def test_no_supported_members_cannot_attach_a_full_native_generation(market,tmp_path):
    snapshot=market['snapshot']('concept','222222',members=['430001.BJ'])
    market['store'].snapshots=[snapshot]
    value=bundle(market)
    assert value['stocks']==[] and value['boards'][0]['expected_members']==0
    store=BoardFluentStore(tmp_path/'fluent.db')
    generation=store.save_market(value)
    identity={'price_date':market['dates'][-1],'mode':'current_reference','training_eligible':False,
              **calculation_contract(value['calculation_membership'])}
    native={'identity':identity,'complete':True,'summaries':{},'input_refs':[]}
    aggregate=aggregate_board_native(snapshot,native)
    assert aggregate['status']=='unavailable' and aggregate['expected_members']==0
    with pytest.raises(ValueError):
        store.attach_native(generation,native,[aggregate])
    assert store.current_generation('full') is None
