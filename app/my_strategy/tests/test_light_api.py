"""HTTP contracts for bounded lightweight jobs without live data or model loads."""
from __future__ import annotations

from types import SimpleNamespace
import threading
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from my_strategy.services import kronos_prediction
from my_strategy.web_dashboard import light_api
from my_strategy.web_dashboard import api as workbench_api
from my_strategy.web_dashboard import board_api


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(kronos_prediction, "asset_status", lambda: {"ready": True, "offline_only": True})
    light = SimpleNamespace(jobs={},
        current_score=Mock(return_value=None), list_score_models=Mock(return_value=[]),
        submit=Mock(side_effect=lambda kind, spec: {"job_id": "fixture", "status": "queued", "kind": kind, "spec": spec}),
        get_job=Mock(side_effect=KeyError("missing")), result=Mock(side_effect=KeyError("missing")),
        cancel=Mock(side_effect=KeyError("missing")), enable_account=Mock(side_effect=KeyError("missing")),
        store=SimpleNamespace(list=Mock(return_value=[]), get_setting=Mock(return_value={})))
    legacy_jobs = []
    app = FastAPI()
    app.state.light = light
    def submit_legacy(kind, spec):
        job = {"job_id": "legacy", "kind": kind, "spec": spec, "status": "queued"}
        legacy_jobs.append(job)
        return job
    app.state.tasks = SimpleNamespace(list=lambda: legacy_jobs, submit=Mock(side_effect=submit_legacy))
    app.state.training = SimpleNamespace(validate_plan=Mock())
    app.state.compute_submit_lock = threading.Lock()
    app.state.board_integrated = True
    app.state.board_jobs = {}
    app.include_router(light_api.router)
    app.include_router(workbench_api.router)
    app.include_router(board_api.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, light=light, legacy_jobs=legacy_jobs, app=app)


def test_predict_http_normalizes_symbol_and_freezes_bounded_defaults(api):
    response = api.client.post("/api/light/predict", json={"symbol": " 002436.sz ", "end": "2026-10-14"})
    assert response.status_code == 202
    kind, spec = api.light.submit.call_args.args
    assert kind == "prediction"
    assert spec["symbol"] == "002436.SZ" and spec["end"] == "2026-10-14"
    assert (spec["lookback"], spec["horizon"], spec["samples"], spec["seed"]) == (128, 5, 8, 42)


@pytest.mark.parametrize("extra", [
    {"samples": 256}, {"lookback": 4096}, {"horizon": 100},
    {"seed": -1}, {"asset_root": "/private/model"},
])
def test_prediction_limits_and_unknown_options_are_rejected_before_queueing(api, extra):
    response = api.client.post("/api/light/predict", json={"symbol": "002436.SZ", **extra})
    assert response.status_code == 422
    api.light.submit.assert_not_called()


@pytest.mark.parametrize("status", ["queued", "running", "cancelling"])
def test_active_legacy_task_blocks_new_compute_but_not_status_read(api, status):
    api.legacy_jobs.append({"status": status})
    assert api.client.post("/api/light/predict", json={"symbol": "002436.SZ"}).status_code == 409
    assert api.client.get("/api/light/status").status_code == 200
    api.light.submit.assert_not_called()


def test_simulation_http_preserves_explicit_company_actions_and_deduplicates_symbols(api):
    actions = [{"action_id": "fixture", "symbol": "002436.SZ", "source_id": "known-source",
                "record_date": "2026-09-01", "ex_date": "2026-09-02", "cash_per_share": .1,
                "tax_per_share": 0, "pay_date": "2026-09-03"}]
    response = api.client.post("/api/light/simulate", json={"symbols": ["002436.sz", "002436.SZ"],
        "start": "2026-01-01", "end": "2026-10-14", "corporate_actions": actions, "save": True})
    assert response.status_code == 202
    _, spec = api.light.submit.call_args.args
    assert spec["symbols"] == ["002436.SZ"]
    assert spec["corporate_actions"] == actions and spec["save"]


def test_score_status_never_converts_historical_learning_to_positive_qualification(api):
    api.light.current_score.return_value = {"row_count": 4570, "feature_columns": list(range(14)),
        "artifact": {"qualification": {"positive_allocation_eligible": False,
                                         "independent_forward_status": "pending"}}}
    api.light.jobs["active"] = {"job_id": "active", "status": "running", "result": {"large": "body"}}
    data = api.client.get("/api/light/status").json()
    assert not data["score_model"]["artifact"]["qualification"]["positive_allocation_eligible"]
    assert data["score_model"]["artifact"]["qualification"]["independent_forward_status"] == "pending"
    assert not data["defaults"]["automatic_training"]
    assert data["defaults"]["daily_mode"] == "lightweight"
    assert "result" not in data["jobs"][0]


@pytest.mark.parametrize("method,path,body", [
    ("get", "/api/light/jobs/missing", None),
    ("get", "/api/light/results/missing", None),
    ("post", "/api/light/jobs/missing/cancel", {}),
    ("post", "/api/light/accounts/missing", {"enabled": True}),
])
def test_unknown_or_expired_identifiers_have_a_reviewable_404(api, method, path, body):
    response = getattr(api.client, method)(path, **({"json": body} if body is not None else {}))
    assert response.status_code == 404
    assert response.json().get("detail")


@pytest.mark.parametrize("path,body", [
    ("/api/tasks", {"kind": "scan", "spec": {"symbols": ["002436.SZ"]}}),
    ("/api/tasks", {"kind": "update", "spec": {"symbols": ["002436.SZ"]}}),
    ("/api/training/jobs", {"plan_id": "fixture"}),
    ("/api/training/validation-jobs", {"plan_id": "fixture", "model_run_id": "fixture"}),
    ("/api/boards/refresh", {}),
    ("/api/boards/features/build", {}),
    ("/api/boards/sync", {}),
    ("/api/analysis", {"symbol": "002436.SZ", "research": False}),
])
def test_active_lightweight_job_blocks_every_legacy_compute_entry(api, path, body):
    api.light.jobs["active"] = {"status": "cancelling"}
    response = api.client.post(path, json=body)
    assert response.status_code == 409
    api.app.state.tasks.submit.assert_not_called()


def test_accepted_legacy_submission_reserves_before_lightweight_submission(api):
    first = api.client.post("/api/tasks", json={"kind": "scan", "spec": {"symbols": ["002436.SZ"]}})
    assert first.status_code == 202
    second = api.client.post("/api/light/predict", json={"symbol": "002436.SZ"})
    assert second.status_code == 409
    assert api.app.state.tasks.submit.call_count == 1
    api.light.submit.assert_not_called()


def test_external_compute_lock_blocks_submission_without_starting_work(api, tmp_path):
    from filelock import FileLock
    api.light.compute_lock = FileLock(tmp_path / "compute.lock")
    with FileLock(tmp_path / "compute.lock"):
        response = api.client.post("/api/tasks", json={"kind": "scan"})
        assert response.status_code == 409
        assert api.client.get("/api/light/status").status_code == 200
    api.app.state.tasks.submit.assert_not_called()


def test_cancellation_remains_available_while_compute_is_active(api):
    api.light.jobs["active"] = {"status": "running"}
    api.light.cancel.side_effect = None
    api.light.cancel.return_value = {"status": "cancelling"}
    response = api.client.post("/api/light/jobs/active/cancel", json={})
    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"


def test_structure_preview_checks_actual_inputs_before_computation_and_reuses_100_reads(api, monkeypatch, tmp_path):
    import pandas as pd
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import czsc_analysis, czsc_trend_support, light_research
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.web_dashboard import trend_strategy
    versions = {"raw": "prefix-1", "calendar": "calendar-1", "code": "code-1"}
    loads, analyses = [], []
    def load(symbol, **kwargs):
        loads.append((symbol, kwargs))
        frame = pd.DataFrame({"date": [pd.Timestamp("2026-10-14")]})
        frame.attrs["data_version"] = versions["raw"]
        return frame
    def analyze(**kwargs):
        analyses.append(kwargs)
        return {"symbol": kwargs["symbol"], "data_end": "2026-10-14", "data_version": versions["raw"],
                "strategy_version": "fixture-structure", "research": {}, "events": []}
    monkeypatch.setattr(czsc_adapter, "load_bars", load)
    monkeypatch.setattr(czsc_analysis, "strategy_config", lambda: {"strategy": "fixed"})
    monkeypatch.setattr(czsc_trend_support, "_calendar", lambda _id: (["2026-10-14"], versions["calendar"]))
    monkeypatch.setattr(light_research, "structure_identity", lambda: {"code": versions["code"]})
    monkeypatch.setattr(trend_strategy, "analyze_trend_consensus", analyze)
    with ResearchStore(tmp_path / "preview") as store:
        api.light.store = store
        request = {"symbol": "002436.SZ", "research": False, "save": False,
                   "end": "2026-10-14", "as_of": "2026-10-14T15:00:00+08:00"}
        first = api.client.post("/api/analysis", json=request)
        assert first.status_code == 200, first.text
        assert not first.json()["reused"]
        for _ in range(100):
            response = api.client.post("/api/analysis", json=request)
            assert response.status_code == 200 and response.json()["reused"]
        assert len(analyses) == 1
        assert len(store.list(namespace="analysis_previews")) == 1
        assert loads[0][1]["as_of"] == request["as_of"]
        assert str(loads[0][1]["end"]) == request["end"]
        for field in ("raw", "calendar", "code"):
            versions[field] += "-changed"
            response = api.client.post("/api/analysis", json=request)
            assert response.status_code == 200 and not response.json()["reused"]
        changed_date = api.client.post("/api/analysis", json={**request, "end": "2026-10-13"})
        assert changed_date.status_code == 200 and not changed_date.json()["reused"]
        assert len(analyses) == 5
        assert not list(tmp_path.glob("**/run_manifest.json"))


def _save_cost_fixture(store, scale=1., *, mutate=None, saved=True):
    from copy import deepcopy
    from my_strategy.tests.test_light_experiment_report import completed_result
    from my_strategy.services.light_experiment_report import build_experiment_report
    value=completed_result(scale=scale)
    value['identity'].update(kind='agent-simulation-v1',version='fixture-v1',calendar='fixed-calendar',
        start=value['start'],end=value['end'],model={'sha':'frozen-kronos'},score_model_type='mlp',
        max_predictions=60,budget_seconds=300,corporate_actions=[],code={'generator':'frozen'},
        execution_mode='historical',qualification_event=None,
        configs={policy:deepcopy(account['lineage']['config']) for policy,account in value['comparisons'].items()})
    if mutate:mutate(value)
    value['result_id']=store.cache_key(value['identity'],'simulations')
    value['experiment_report']=build_experiment_report(value,bootstrap_samples=1)
    reference=store.put(value['identity'],value,namespace='simulations',pin='saved_simulation' if saved else None)
    return value,reference


def _serve_store(api,store):
    api.light.store=store
    api.light.result.side_effect=lambda key:{**store.get_by_key(key),'saved':store.is_pinned(key)}


def test_saved_result_derives_exact_cost_report_without_mutating_cas_or_scanning(api,tmp_path,monkeypatch):
    from my_strategy.storage.research_store import ResearchStore
    with ResearchStore(tmp_path/'derived') as store:
        items={str(scale):_save_cost_fixture(store,scale) for scale in (1.,1.5,2.)}
        _serve_store(api,store)
        # Even a tempting higher-return result from another model must never be considered.
        _save_cost_fixture(store,1.5,mutate=lambda value:value['identity'].update(score='different-model'))
        monkeypatch.setattr(store,'list',lambda **kwargs:pytest.fail('result enumeration forbidden'))
        physical={key:store._path(ref['sha256']).read_bytes() for key,(_,ref) in items.items()}
        before_settings=store.connection.execute('SELECT * FROM settings').fetchall()
        response=api.client.get('/api/light/results/'+items['1.0'][0]['result_id'])
        assert response.status_code==200,response.text
        value=response.json();report=value['experiment_report'];derived=value['experiment_report_derivation']
        assert report['readiness']['cost_stress_complete']
        assert report['status']=='pending'  # one window still cannot certify performance
        assert not report['qualification_granted'] and report['best_model_selected'] is None
        assert derived['verified_results']==3 and derived['candidate_limit']==3
        assert derived['persisted'] is False and derived['issues']=={}
        assert derived['stored_report_id']==items['1.0'][0]['experiment_report']['report_id']
        for scale,(original,reference) in items.items():
            source=derived['source_results'][scale]
            assert source['result_id']==original['result_id']
            assert source['physical_sha256']==reference['sha256']
            assert source['compressed_bytes']==reference['size_bytes']
            assert len(source['identity_sha256'])==64
            assert store._path(reference['sha256']).read_bytes()==physical[scale]
            assert store.get_by_key(original['result_id'])==original
        assert store.connection.execute('SELECT * FROM settings').fetchall()==before_settings
        assert api.client.get('/api/light/results/'+items['1.0'][0]['result_id']).json()==value
        api.light.submit.assert_not_called()


@pytest.mark.parametrize('change',[
    lambda r:r['identity'].update(score='other-model'),
    lambda r:r['identity'].update(sources={'fixture':'revised-price'}),
    lambda r:r['identity'].update(calendar='other-calendar'),
    lambda r:r['identity'].update(code={'generator':'revised'}),
    lambda r:r['identity'].update(budget_seconds=600),
    lambda r:r['identity']['configs']['structure_kronos_score'].update(initial_weight=.19),
    lambda r:r['identity'].update(qualification_event={'id':'different'}),
])
def test_cost_report_does_not_join_different_replay_contracts(api,tmp_path,change):
    from my_strategy.storage.research_store import ResearchStore
    with ResearchStore(tmp_path/'mismatch') as store:
        original,_=_save_cost_fixture(store)
        _save_cost_fixture(store,1.5,mutate=change)
        _serve_store(api,store)
        value=api.client.get('/api/light/results/'+original['result_id']).json()
        assert value['experiment_report']['cost_stress']['1.5']['status']=='pending'
        assert value['experiment_report_derivation']['issues']['1.5']=='matching_saved_result_missing'
        assert set(value['experiment_report_derivation']['source_results'])=={'1.0'}


@pytest.mark.parametrize('case',['unsaved','partial','corrupt','bad_payload_identity','inconsistent_ledger'])
def test_cost_report_missing_partial_or_corrupt_evidence_stays_pending(api,tmp_path,case):
    from my_strategy.storage.research_store import ResearchStore
    with ResearchStore(tmp_path/case) as store:
        original,_=_save_cost_fixture(store)
        def change(value):
            if case=='partial':value['status']='partial'
            if case=='inconsistent_ledger':value['comparisons']['structure_rules']['metrics']['total_fees']=0
        candidate,reference=_save_cost_fixture(store,1.5,mutate=change,saved=case!='unsaved')
        if case=='corrupt':
            path=store._path(reference['sha256'])
            path.chmod(path.stat().st_mode | 0o200)  # intentional corruption of an isolated immutable fixture
            path.write_bytes(b'corrupt')
        if case=='bad_payload_identity':
            # Same indexed identity with a forged payload key is not a valid source.
            bad={**candidate,'result_id':'0'*64}
            store.connection.execute('DELETE FROM results WHERE cache_key=?',(candidate['result_id'],))
            store.put(candidate['identity'],bad,namespace='simulations',pin='saved_simulation')
        _serve_store(api,store)
        response=api.client.get('/api/light/results/'+original['result_id'])
        assert response.status_code==200,response.text
        value=response.json()
        assert value['experiment_report']['cost_stress']['1.5']['status']=='pending'
        assert not value['experiment_report']['readiness']['cost_stress_complete']
        assert not value['experiment_report']['qualification_granted']
        if case=='inconsistent_ledger':
            assert '1.5' in value['experiment_report_derivation']['source_results']
            assert 'structure_rules:ledger_fees' in value['experiment_report']['cost_stress']['1.5']['issues']
        else:assert '1.5' not in value['experiment_report_derivation']['source_results']


def test_partial_or_unsaved_requested_result_keeps_original_report(api,tmp_path):
    from my_strategy.storage.research_store import ResearchStore
    with ResearchStore(tmp_path/'previews') as store:
        preview,_=_save_cost_fixture(store,saved=False)
        _save_cost_fixture(store,1.5)
        _serve_store(api,store)
        value=api.client.get('/api/light/results/'+preview['result_id']).json()
        assert 'experiment_report_derivation' not in value
        assert value['experiment_report']==preview['experiment_report']
        partial,_=_save_cost_fixture(store,2.,mutate=lambda r:r.update(status='partial'))
        value=api.client.get('/api/light/results/'+partial['result_id']).json()
        assert 'experiment_report_derivation' not in value
        assert value['experiment_report']==partial['experiment_report']


def test_requested_stress_view_keeps_its_metrics_and_reads_only_three_exact_objects(api,tmp_path,monkeypatch):
    from my_strategy.storage.research_store import ResearchStore
    with ResearchStore(tmp_path/'stress-view') as store:
        items={str(scale):_save_cost_fixture(store,scale)[0] for scale in (1.,1.5,2.)}
        reads=[];original_get=store.get_by_key
        def get(key):
            reads.append(key)
            return original_get(key)
        monkeypatch.setattr(store,'get_by_key',get)
        _serve_store(api,store)
        value=api.client.get('/api/light/results/'+items['1.5']['result_id']).json()
        assert value['result_id']==items['1.5']['result_id']
        assert value['experiment_report']['metrics']['structure_rules']['total_fees']==150.
        assert value['experiment_report']['cost_stress']['1.0']['metrics']['structure_rules']['total_fees']==100.
        assert value['experiment_report']['cost_stress']['2.0']['metrics']['structure_rules']['total_fees']==200.
        assert reads[0]==items['1.5']['result_id']  # original endpoint load
        assert sorted(reads[1:])==sorted(value['result_id'] for value in items.values())


def _restart_with_jobs(api,store):
    from my_strategy.services.light_research import LightResearch
    api.light.store=store
    api.light.jobs={}
    api.light.lock=threading.RLock()
    api.light.get_job.side_effect=lambda key:LightResearch.get_job(api.light,key)


def test_restart_restores_terminal_summaries_by_actual_time_and_memory_wins(api,tmp_path):
    from my_strategy.storage.research_store import ResearchStore
    root=tmp_path/'restart';now=[1000.]
    rows=[{'job_id':'old','kind':'incremental','status':'succeeded','created_at':'2026-10-15T09:00:00+08:00',
           'finished_at':'2026-10-15T09:02:00+08:00','spec':{'plan_id':'a'*32},'result_id':None},
          {'job_id':'new','kind':'incremental','status':'cancelled','created_at':'2026-10-15T09:03:00+08:00',
           'finished_at':'2026-10-15T01:04:00Z','spec':{'plan_id':'b'*32},'result_id':None},
          {'job_id':'stale-active','kind':'incremental','status':'running','created_at':'2026-10-15T09:05:00+08:00'}]
    with ResearchStore(root,clock=lambda:now[0]) as store:
        for value in rows:
            store.put({'job_id':value['job_id']},value,namespace='jobs');now[0]+=1
    with ResearchStore(root,clock=lambda:now[0]) as store:
        _restart_with_jobs(api,store)
        # Reading the old record recently must not make it the latest operation.
        store.get({'job_id':'old'},namespace='jobs')
        value=api.client.get('/api/light/status').json()
        assert [job['job_id'] for job in value['jobs']]==['new','old']
        assert api.light.jobs=={} and not value['cancelled_tasks_resumed']
        api.light.jobs['old']={'job_id':'old','kind':'incremental','status':'running',
            'created_at':'2026-10-15T09:06:00+08:00','progress':{'completed':1,'total':2},'result':{'large':'excluded'}}
        value=api.client.get('/api/light/status').json()
        assert [(job['job_id'],job['status']) for job in value['jobs']]==[('old','running'),('new','cancelled')]
        assert all('result' not in job for job in value['jobs'])
        api.light.submit.assert_not_called()
        assert not list(root.glob('**/run_manifest.json'))


def test_restart_history_reads_at_most_thirty_summaries_and_skips_expired(api,tmp_path,monkeypatch):
    from datetime import datetime,timedelta,timezone
    from my_strategy.storage.research_store import ResearchStore
    root=tmp_path/'bounded';now=[1000.]
    with ResearchStore(root,clock=lambda:now[0]) as store:
        for index in range(36):
            stamp=(datetime(2026,10,15,tzinfo=timezone.utc)+timedelta(seconds=index)).isoformat()
            value={'job_id':str(index),'kind':'daily','status':'succeeded','finished_at':stamp}
            store.put({'job_id':str(index)},value,namespace='jobs',ttl_seconds=1 if index==35 else 1000)
            now[0]+=1
    now[0]+=5
    with ResearchStore(root,clock=lambda:now[0]) as store:
        _restart_with_jobs(api,store)
        reads=[];original_get=store.get_by_key
        def get(key):
            reads.append(key);return original_get(key)
        monkeypatch.setattr(store,'get_by_key',get)
        response=api.client.get('/api/light/status')
        assert response.status_code==200,response.text
        assert [job['job_id'] for job in response.json()['jobs']]==[str(i) for i in range(34,4,-1)]
        assert len(reads)==30 and api.light.jobs=={}
        api.light.submit.assert_not_called()


@pytest.mark.parametrize('case',['available','expired_plan','missing_plan','missing_result','wrong_plan','corrupt_result'])
def test_restarted_incremental_job_reads_existing_plan_or_marks_result_unavailable(api,tmp_path,case):
    from my_strategy.storage.research_store import ResearchStore
    root=tmp_path/case;now=[1000.];plan_id='a'*32
    summary={'job_id':'saved-incremental','kind':'incremental','status':'succeeded','result_id':None,
        'created_at':'2026-10-15T09:00:00+08:00','finished_at':'2026-10-15T09:01:00+08:00',
        'spec':{'plan_id':plan_id},'progress':{'completed':2,'total':2}}
    result={'plan_id':plan_id,'status':'completed','requested':2,'processed':2,
        'counts':{'reused':1,'appended':0,'rebuilt':1,'missing_price':0,'failed':0,'deferred':0},
        'full_published':False,'cancelled_tasks_resumed':False,'training_started':False}
    with ResearchStore(root,clock=lambda:now[0]) as store:
        store.put({'job_id':summary['job_id']},summary,namespace='jobs')
        if case!='missing_plan':
            plan={'plan_id':plan_id,'status':'planned','expires_at':1001. if case=='expired_plan' else 2000.}
            store.put({'plan_id':plan_id},plan,namespace='incremental_plans')
        if case!='missing_result':
            if case=='wrong_plan':result['plan_id']='b'*32
            reference=store.put({'fixture':case},result,namespace='incremental_plan_results')
            store.set_setting('incremental_plan:'+plan_id,{'status':'completed','result_key':reference['cache_key']})
            if case=='corrupt_result':
                path=store._path(reference['sha256']);path.chmod(path.stat().st_mode | 0o200)
                path.write_bytes(b'corrupt isolated fixture')
    now[0]=1005.
    with ResearchStore(root,clock=lambda:now[0]) as store:
        _restart_with_jobs(api,store)
        response=api.client.get('/api/light/jobs/'+summary['job_id'])
        assert response.status_code==200,response.text
        value=response.json()
        assert value['status']=='succeeded' and value['progress']==summary['progress']
        if case=='available':
            assert value['result']==result and value['result_availability']=='available'
            assert value['result_source']=='incremental_plan'
        else:
            assert 'result' not in value and 'counts' not in value
            assert value['result_unavailable_reason']
            assert value['result_availability']==({'expired_plan':'expired_or_missing','missing_plan':'expired_or_missing',
                'missing_result':'not_recorded','wrong_plan':'not_recorded','corrupt_result':'verification_failed'}[case])
        assert store.get({'job_id':summary['job_id']},namespace='jobs')==summary
        assert api.light.jobs=={}
        api.light.submit.assert_not_called()
