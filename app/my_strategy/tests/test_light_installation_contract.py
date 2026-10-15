from copy import deepcopy
from pathlib import Path

import pytest

import verify_installation as installation


def test_shipped_primary_navigation_has_five_actual_views():
    html=(Path(installation.__file__).parent/'my_strategy/web_dashboard/static/index.html').read_text()
    assert installation.verify_primary_navigation(html)==list(installation.PRIMARY_PAGES)
    with pytest.raises(RuntimeError,match='navigation'):
        installation.verify_primary_navigation(html.replace('data-route="simulation"','data-route="training"'))
    with pytest.raises(RuntimeError,match='controls'):
        installation.verify_primary_navigation(html.replace('id="light-score-variant"','id="missing"'))
    with pytest.raises(RuntimeError,match='navigation'):
        installation.verify_primary_navigation('<html><body>ordinary fallback</body></html>')


@pytest.fixture
def fresh_api(monkeypatch):
    paths={'/api/light/status':{'version':'agent_mlp_sim_v1','score_model':None,'score_candidates':[],
        'jobs':[],'cancelled_tasks_resumed':False,'defaults':{'daily_mode':'lightweight','automatic_training':False},
        'assets':{'ready':False,'missing':['model.safetensors']}},
        '/api/light/accounts':{'accounts':{}},'/api/light/saved':{'results':[]},
        '/openapi.json':{'paths':{path:{method:{}} for path,method in {
            '/api/light/status':'get','/api/light/predict':'post','/api/light/simulate':'post',
            '/api/light/score/train':'post','/api/light/score/observations':'post','/api/light/daily':'post',
            '/api/boards/incremental/plan':'post','/api/boards/incremental/execute':'post'}.items()}}}
    calls=[]
    def read(base,path,method='GET',body=None):
        assert method=='GET' and body is None
        calls.append(path);return paths[path]
    monkeypatch.setattr(installation,'request',read)
    return paths,calls


def test_new_api_probe_is_read_only_and_missing_models_remain_explicit(fresh_api):
    _,calls=fresh_api
    report=installation.verify_lightweight_api('http://isolated')
    assert report['status']=='passed' and report['work_submitted'] is False
    assert report['kronos_assets']=='absent_explicitly'
    assert len(calls)==4


@pytest.mark.parametrize('change',[
    lambda p:p['/api/light/status'].update(score_model={'private':'model'}),
    lambda p:p['/api/light/status'].update(score_candidates=[{'private':'candidate'}]),
    lambda p:p['/api/light/status'].update(jobs=[{'status':'queued'}]),
    lambda p:p['/api/light/status']['defaults'].update(automatic_training=True),
    lambda p:p['/api/light/status'].update(cancelled_tasks_resumed=True),
    lambda p:p['/api/light/status']['assets'].update(ready=True,missing=[]),
    lambda p:p['/openapi.json']['paths'].pop('/api/boards/incremental/plan'),
    lambda p:p['/api/light/accounts'].update(accounts={'personal':{}}),
])
def test_new_api_probe_rejects_hidden_state_or_missing_entrypoints(fresh_api,change):
    paths,_=fresh_api;change(paths)
    with pytest.raises(RuntimeError):installation.verify_lightweight_api('http://isolated')


def test_probe_preserves_real_mlp_device_split_and_rejects_false_qualification():
    report={'status':'passed','family':'agent_mlp_sim_v1','fixture':'synthetic_installation_only',
        'publication_allowed':False,'market_effectiveness_certified':False,
        'mlp':{'training_device':'mps','calibration_device':'cpu','inference_device':'cpu','save_reload_identical':True},
        'incremental_source':{},'simulation_ledger':{}}
    assert installation.verify_lightweight_probe(report,'mps') is report
    with pytest.raises(RuntimeError,match='device'):
        installation.verify_lightweight_probe(report,'cpu')
    changed=deepcopy(report);changed['publication_allowed']=True
    with pytest.raises(RuntimeError,match='qualification'):
        installation.verify_lightweight_probe(changed,'mps')
    changed=deepcopy(report);changed['mlp']['inference_device']='mps'
    with pytest.raises(RuntimeError,match='device'):
        installation.verify_lightweight_probe(changed,'mps')


@pytest.mark.parametrize('change',[
    lambda r:r.update(preview_id='c'*64),
    lambda r:r.update(reused=False),
    lambda r:r.update(unreported_payload_change=True),
    lambda r:r.update(run_id='permanent-run'),
    lambda r:r.update(saved=True),
])
def test_source_preview_probe_rejects_changed_identity_payload_or_missing_reuse(monkeypatch,change):
    calls=[]
    def request(base,path,method='GET',body=None):
        assert path=='/api/analysis' and method=='POST'
        assert body['research'] is False and body['save'] is False
        calls.append(body)
        result={'preview':True,'saved':False,'preview_id':'b'*64,'reused':len(calls)>1,
            'projection_version':'structure_sources_v1','capabilities':{'source_evidence':True,
                'direction_model':False,'reference_position':False,'simulated_account':False},
            'reference_points':[],'direction_points':[],'request':deepcopy(body)}
        if len(calls)>1:change(result)
        return result
    monkeypatch.setattr(installation,'request',request)
    with pytest.raises(RuntimeError):
        installation.verify_analysis_preview_cache('http://isolated',{'symbol':'000001.SZ','end':'2024-12-30'})
