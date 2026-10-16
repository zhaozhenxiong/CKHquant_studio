from copy import deepcopy
import pytest

from my_strategy.services.light_account_compatibility import (
    LEGACY_CODE,ACCELERATED_CODE,account_code_compatible,execution_adapter_identity)
from my_strategy.services.light_research import code_identity


def test_only_the_sealed_exact_legacy_to_accelerated_pair_is_compatible():
    assert LEGACY_CODE and ACCELERATED_CODE
    assert len(LEGACY_CODE)==19
    assert len(ACCELERATED_CODE)==20
    before=deepcopy(LEGACY_CODE)
    assert account_code_compatible(LEGACY_CODE,ACCELERATED_CODE)
    assert account_code_compatible(ACCELERATED_CODE,ACCELERATED_CODE)
    assert LEGACY_CODE==before
    assert 'light_prediction_inputs.py' not in LEGACY_CODE
    assert 'light_prediction_inputs.py' in ACCELERATED_CODE
    changed={key for key in set(LEGACY_CODE)|set(ACCELERATED_CODE) if LEGACY_CODE.get(key)!=ACCELERATED_CODE.get(key)}
    assert changed=={'light_research.py','light_daily_scheduler.py','light_account_continuation.py','light_prediction_inputs.py'}


def test_current_execution_closure_cannot_enter_the_historical_exact_bridge():
    current=code_identity()
    helpers={'light_source_preparation.py','method_scan_acceleration.py','method_compute.py','analysis_methods.py'}
    assert set(current)==set(ACCELERATED_CODE)|helpers
    assert len(current)==24
    assert current!=ACCELERATED_CODE
    assert not account_code_compatible(LEGACY_CODE,current)
    assert not account_code_compatible(ACCELERATED_CODE,current)
    assert account_code_compatible(current,current)
    for key in current:
        modified={**current,key:('0' if current[key][0]!='0' else '1')+current[key][1:]}
        assert not account_code_compatible(current,modified),key


@pytest.mark.parametrize('side',['legacy','accelerated'])
def test_every_dependency_mutation_addition_or_removal_rejects_the_bridge(side):
    target=LEGACY_CODE if side=='legacy' else ACCELERATED_CODE
    for key in target:
        modified={**target,key:'unreviewed-sha'}
        left,right=(modified,ACCELERATED_CODE) if side=='legacy' else (LEGACY_CODE,modified)
        assert not account_code_compatible(left,right),key
        removed={k:v for k,v in target.items() if k!=key}
        left,right=(removed,ACCELERATED_CODE) if side=='legacy' else (LEGACY_CODE,removed)
        assert not account_code_compatible(left,right),key
    modified={**target,'unexpected.py':'unreviewed'}
    assert not account_code_compatible(modified,ACCELERATED_CODE) if side=='legacy' else not account_code_compatible(LEGACY_CODE,modified)


def test_execution_identity_separately_binds_pipeline_and_checkpoint_policy():
    identity=execution_adapter_identity()
    assert identity['cpu_preparation_workers']==2
    assert identity['maximum_prepared_prefixes']==8
    assert identity['checkpoint_interval_seconds']==5.
    assert identity['model_inference']=='serial_original_rng'
    assert identity['terminal_cancel_interruption_checkpoints']=='always_complete'
    assert len(identity['compatibility_sha256'])==64
