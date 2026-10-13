"""A new target family cannot reinterpret an old model or saved account."""
import json

import pytest

from my_strategy.web_dashboard import trend_routes as routes
from my_strategy.services.czsc_trend_v2_contracts import STRATEGY_ID as V2


@pytest.fixture
def families(monkeypatch, tmp_path):
    for name,version in [('v1','czsc_trend_consensus'),('v2',V2)]:
        path=tmp_path/name/'reports';path.mkdir(parents=True)
        (path/'trend_consensus.json').write_text(json.dumps({'strategy_version':version}))
    monkeypatch.setattr(routes,'ARTIFACT_RUNS_ROOT',tmp_path)
    monkeypatch.setattr(routes,'CURRENT_PATH',tmp_path/'current.json')
    monkeypatch.setattr(routes,'_saved_consensus',lambda model_run_id=None,family=None:None)
    routes.CURRENT_PATH.write_text(json.dumps({'strategy':V2}))
    return tmp_path


def test_new_default_never_reinterprets_explicit_older_run(families):
    assert routes.trend_family()==V2
    assert routes.trend_family(model_run_id='v1')=='czsc_trend_consensus'
    assert routes.trend_family(model_run_id='v2')==V2
    for run_id,explicit in [('v1',V2),('v2','czsc_trend_consensus'),('v2','czsc_agent_consensus_v4')]:
        with pytest.raises(ValueError,match='不能混入旧'):
            routes.trend_family(model_run_id=run_id,consensus_version=explicit)
    version,config=routes.resolve_analysis_consensus(research=True,model_family='high_return',model_run_id='v2')
    assert version==config['version']==V2
    assert routes.strategy_adapter(model_run_id='v1').__name__.endswith('.trend_strategy')
    assert routes.strategy_adapter(model_run_id='v2').__name__.endswith('.trend_v2_strategy')


def test_new_api_accepts_account_context_and_rejects_other_family(families):
    from my_strategy.web_dashboard.api import AnalysisRequest
    spec=AnalysisRequest(symbol='002436.SZ',research=True,model_family='high_return',
        consensus_version=V2,model_run_id='v2',account_start='2023-01-01',account_initial_cash=100000)
    assert spec.consensus_version==V2 and spec.entry_policy=='risk'
    with pytest.raises(ValueError,match='不能混入旧'):
        AnalysisRequest(symbol='002436.SZ',research=True,model_family='high_return',
            consensus_version=V2,model_run_id='v1')


def test_new_family_rejects_obsolete_training_holding_and_forward_routes(families):
    assert routes.reject_legacy_operation('high_return_backtest',run_id='v2') is None
    for operation in ['holding','forward','high_return_train','high_return_points_build']:
        with pytest.raises(ValueError,match='旧入口'):
            routes.reject_legacy_operation(operation,run_id='v2',consensus_version=V2)
