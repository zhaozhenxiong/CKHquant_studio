import copy

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.board_calculation_scope import (CALCULATION_SCOPE,
    calculation_membership, verify_calculation_membership, calculation_contract)


def snapshot():
    members=['600001.SH','000001.SZ','000002.SZ','300001.SZ','300002.SZ',
             '430001.BJ','920001.BJ','834683.NQ','900901.SH']
    row=dict(board_id='THS_I_111111',members=members,expected_count=len(members),complete=True)
    return {**row,'snapshot_hash':stable_hash(row)}


def test_projection_preserves_source_and_has_exact_partition():
    source=snapshot();before=copy.deepcopy(source);value=calculation_membership(source)
    assert source==before and value['source_snapshot_hash']==source['snapshot_hash']
    assert value['source_member_count']==9 and value['eligible_count']==5
    assert value['excluded_members']==['430001.BJ','834683.NQ','900901.SH','920001.BJ']
    assert value['usable_membership'] and value['calculation_scope']==CALCULATION_SCOPE
    assert verify_calculation_membership(value,source)==value
    assert calculation_contract({source['board_id']:value})['required_symbols']==value['eligible_members']


@pytest.mark.parametrize('field,value',[
    ('eligible_members',['430001.BJ']),('excluded_members',[]),('projection_hash','0'*64),
    ('source_member_count',1),('calculation_scope','old'),('usable_membership',False)])
def test_saved_projection_tamper_is_rejected(field,value):
    source=snapshot();projection=calculation_membership(source);projection[field]=value
    with pytest.raises(ValueError,match='projection'):
        verify_calculation_membership(projection,source)


def test_incomplete_source_cannot_become_usable_after_exclusions():
    source=snapshot();source['expected_count']=5
    assert not calculation_membership(source)['usable_membership']
    source['expected_count']=9;source['complete']=False
    assert not calculation_membership(source)['usable_membership']
    source['members'].append('000001.SZ')
    with pytest.raises(ValueError,match='duplicate'):calculation_membership(source)


def test_valid_self_hash_cannot_replace_source_binding():
    source=snapshot();projection=calculation_membership(source)
    changed=copy.deepcopy(source);changed['members'][0]='600002.SH'
    with pytest.raises(ValueError,match='projection'):
        verify_calculation_membership(calculation_membership(changed),source)
    assert calculation_membership(changed)!=projection
