from copy import deepcopy
from pathlib import Path

import pytest

import verify_installation as installation


def v101_evidence():
    return {'ma_profile': dict(status='passed', feature_profile='ma_trend_v1', input_dimensions=20,
        training_device='mps', calibration_device='cpu', inference_device='cpu', same_input_l2=True,
        distinct_scorer_same_model=True, save_reload_identical=True, causal_prefix_identical=True,
        missing_calendar_rejected=True, ineligible_stays_unscored=True, consumes_kronos=False,
        active_model_registered=False, paper_allocation_eligible=False),
        'board_scope': dict(status='passed', calculation_scope='sh_sz_a_v1', source_member_count=6,
            eligible_count=2, excluded_count=4, bj_aliases_rejected=5, source_preserved=True,
            projection_tamper_rejected=True, empty_scope_unusable=True, historical_requires_explicit=True,
            scoped_publication_verified=True, mixed_denominator_rejected=True),
        'task_maintenance': dict(status='passed', pagination=True, active_delete_rejected=True,
            terminal_record_deleted=True, preview_non_destructive=True, wrong_plan_hash_rejected=True,
            exact_fixture_cleanup=True, production_state_touched=False),
        'spawn': dict(status='passed', start_method='spawn', stocks=2, prediction='synthetic_stub_no_weights',
            adapter_sha256='a'*64, worker_pids=[101, 102], parent_pid=100, real_native_preparation=True,
            rows_and_frozen_identity_identical=True, cancelled_preserves_pending=True, expired_budget_rejected=True,
            children_joined=True, algorithm_bytecode_preserved=True)}


def valid_probe():
    return {'status':'passed','family':'agent_mlp_sim_v1','fixture':'synthetic_installation_only',
        'publication_allowed':False,'market_effectiveness_certified':False,
        'mlp':{'training_device':'mps','calibration_device':'cpu','inference_device':'cpu','save_reload_identical':True},
        'incremental_source':{},'simulation_ledger':{},'v101':v101_evidence()}


def test_shipped_primary_navigation_has_five_actual_views():
    html=(Path(installation.__file__).parent/'my_strategy/web_dashboard/static/index.html').read_text()
    assert installation.verify_primary_navigation(html)==list(installation.PRIMARY_PAGES)
    with pytest.raises(RuntimeError,match='navigation'):
        installation.verify_primary_navigation(html.replace('data-route="simulation"','data-route="training"'))
    with pytest.raises(RuntimeError,match='controls'):
        installation.verify_primary_navigation(html.replace('id="light-score-variant"','id="missing"'))
    with pytest.raises(RuntimeError,match='navigation'):
        installation.verify_primary_navigation('<html><body>ordinary fallback</body></html>')
    for before, after in [('value="structure_kronos_ma_l2"', 'value="missing"'),
                          ('id="task-history-panel"', 'id="missing"')]:
        with pytest.raises(RuntimeError, match='controls'):
            installation.verify_primary_navigation(html.replace(before, after))


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
    report=valid_probe()
    assert installation.verify_lightweight_probe(report,'mps') is report
    with pytest.raises(RuntimeError,match='device'):
        installation.verify_lightweight_probe(report,'cpu')
    changed=deepcopy(report);changed['publication_allowed']=True
    with pytest.raises(RuntimeError,match='qualification'):
        installation.verify_lightweight_probe(changed,'mps')
    changed=deepcopy(report);changed['mlp']['inference_device']='mps'
    with pytest.raises(RuntimeError,match='device'):
        installation.verify_lightweight_probe(changed,'mps')


@pytest.mark.parametrize('section,key,value', [
    ('ma_profile','input_dimensions',34), ('ma_profile','feature_profile','mixed_v1'),
    ('ma_profile','same_input_l2',False), ('ma_profile','active_model_registered',True),
    ('ma_profile','paper_allocation_eligible',True), ('ma_profile','consumes_kronos',True),
    ('ma_profile','training_device','cpu'), ('ma_profile','inference_device','mps'),
    ('ma_profile','ineligible_stays_unscored',False), ('ma_profile','missing_calendar_rejected',False),
    ('board_scope','calculation_scope','mixed'), ('board_scope','source_preserved',False),
    ('board_scope','eligible_count',6), ('board_scope','bj_aliases_rejected',0),
    ('board_scope','projection_tamper_rejected',False), ('board_scope','empty_scope_unusable',False),
    ('board_scope','historical_requires_explicit',False), ('board_scope','scoped_publication_verified',False),
    ('board_scope','mixed_denominator_rejected',False),
    ('task_maintenance','production_state_touched',True), ('task_maintenance','active_delete_rejected',False),
    ('task_maintenance','wrong_plan_hash_rejected',False), ('task_maintenance','exact_fixture_cleanup',False),
    ('spawn','start_method','fork'), ('spawn','worker_pids',[100]), ('spawn','worker_pids',[]),
    ('spawn','worker_pids',[101,101]), ('spawn','worker_pids',[101,102,103]),
    ('spawn','adapter_sha256','invalid'), ('spawn','prediction','real_model'),
    ('spawn','rows_and_frozen_identity_identical',False), ('spawn','cancelled_preserves_pending',False),
    ('spawn','expired_budget_rejected',False), ('spawn','children_joined',False),
])
def test_v101_probe_gate_rejects_missing_or_contradictory_evidence(section,key,value):
    report=valid_probe(); report['v101'][section][key]=value
    with pytest.raises(RuntimeError,match='v1.0.1'):
        installation.verify_lightweight_probe(report,'mps')


@pytest.mark.parametrize('section', ['ma_profile','board_scope','task_maintenance','spawn'])
def test_v101_probe_gate_cannot_reuse_v100_receipt(section):
    report=valid_probe();report['v101'].pop(section)
    with pytest.raises(RuntimeError,match='v1.0.1'):
        installation.verify_lightweight_probe(report,'mps')


@pytest.fixture
def v101_api(monkeypatch):
    from fastapi import FastAPI
    from urllib.error import HTTPError
    from my_strategy.web_dashboard import api, light_api, task_history_api, board_api
    app=FastAPI()
    for router in (api.router,light_api.router,task_history_api.router,board_api.router):
        app.include_router(router)
    values={'/openapi.json':app.openapi(),'/api/task-history':{'total':0,'items':[]},
        '/api/light/status':{'jobs':[]},'/api/boards/generations':{'required_calculation_scope':'sh_sz_a_v1',
            'current_generation_id':None,'generations':[],'needs_rebuild':True}}
    calls=[]
    def request(base,path,method='GET',body=None):
        calls.append((path,method,body))
        if method=='POST':raise HTTPError(base+path,422,'synthetic refusal',{},None)
        return values[path]
    monkeypatch.setattr(installation,'request',request)
    return values,calls


def test_v101_api_probe_checks_shipped_schemas_and_only_sends_invalid_work(v101_api):
    _,calls=v101_api
    report=installation.verify_v101_api('http://isolated')
    assert report['status']=='passed' and report['rejected_requests']==13
    assert report['combined_methods']==list(installation.COMBINED_METHODS)
    assert report['invalid_requests_created_jobs'] is report['work_submitted'] is False
    assert sum(method=='POST' for _,method,_ in calls)==13


@pytest.mark.parametrize('mutation',[
    lambda v:v['/openapi.json']['paths'].pop('/api/task-history/cleanup/recover'),
    lambda v:v['/openapi.json']['components']['schemas']['AnalysisRequest']['properties'].pop('analysis_method'),
    lambda v:v['/openapi.json']['components']['schemas']['TaskSpec']['properties'].pop('analysis_method'),
    lambda v:v['/openapi.json']['components']['schemas']['ScoreSpec']['properties'].pop('feature_profile'),
    lambda v:v['/openapi.json']['components']['schemas']['SimulationSpec']['properties'].pop('feature_profile'),
    lambda v:v['/api/task-history'].update(total=1,items=[{'job_id':'unexpected'}]),
    lambda v:v['/api/light/status'].update(jobs=[{'status':'queued'}]),
    lambda v:v['/api/boards/generations'].update(required_calculation_scope='old'),
    lambda v:v['/api/boards/generations'].update(current_generation_id='mixed-old'),
    lambda v:v['/api/boards/generations'].update(needs_rebuild=False),
])
def test_v101_api_probe_rejects_incomplete_boundaries_or_hidden_work(v101_api,mutation):
    values,_=v101_api;mutation(values)
    with pytest.raises(RuntimeError):installation.verify_v101_api('http://isolated')


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
