"""Old-policy controls retain causal routes, flat windows, fees and coverage."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_legacy_rule_ma_comparison as legacy
from my_strategy.services import czsc_high_return_research as research
from my_strategy.tests.test_czsc_high_return_research import frozen_source


@pytest.fixture
def market(frozen_source):
    _,dates,records,_,config=frozen_source
    frame=pd.read_parquet(records[0]["path"])
    frame["date"]=pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame["input_eligible"]=True;frame["rule_buy"]=False;frame["rule_sell"]=False
    frame.loc[130,"rule_buy"]=True;frame.loc[145,"rule_buy"]=True;frame.loc[150,"rule_sell"]=True
    frame.loc[165,"rule_buy"]=True;frame.loc[170,"rule_sell"]=True
    return frame,dates,config,records


class Resolver:
    def __init__(self,available):self.available=available;self.catalog=[];self.catalog_errors=[]
    def resolve(self,day):
        return {"date":day,"model_dir":"old-fold" if day>=self.available else None,
            "available_at":self.available,"probability_threshold":.55,"applied_to_entry":False,
            "status":"historical_shadow" if day>=self.available else "rules_no_model"}
    def resolve_dates(self,dates):return [self.resolve(d) for d in dates]


class Predictor:
    def __init__(self,probability=.01):self.probability=probability;self.calls=[]
    def predict(self,frame,directory,as_of=None):
        self.calls.append(frame.date.tolist());return np.full(len(frame),self.probability)


def test_position_is_flat_at_window_start_but_broker_keeps_real_warmup(market):
    frame,dates,config,_=market
    # A pre-window opening point must not become an invented holding at start.
    account,diagnostic=legacy._replay_legacy(frame,dates,config,dates[144],dates[158])
    assert account["ledger"][0]["date"]==dates[146]
    assert account["ledger"][0]["signal_date"]==dates[145]
    assert account["ledger"][1]["date"]==dates[151]
    assert not account["rejections"]
    assert account["ledger"][0]["fee"]>0
    assert diagnostic["decisions"][0]["target_weight"]==0


def test_mature_old_head_filters_only_after_its_own_date_and_reports_no_release(market):
    frame,dates,config,_=market;predictor=Predictor()
    values,routes=legacy._chronological_probabilities(frame,Resolver(dates[160]),predictor)
    assert np.isnan(values[:160]).all()
    assert min(predictor.calls[0])==dates[160]
    account,diagnostic=legacy._replay_legacy(frame,dates,config,dates[140],dates[175],
        probabilities=values,routes=routes,offline=True)
    buys=[row for row in account["ledger"] if row["action"]=="BUY"]
    assert [row["signal_date"] for row in buys]==[dates[145]]
    assert diagnostic["coverage"]["actual_release_applied_sessions"]==0
    assert diagnostic["coverage"]["offline_filter_candidate_sessions"]==1
    assert diagnostic["coverage"]["rules_fallback_candidate_sessions"]==1


def test_missing_model_input_is_recorded_rule_fallback_never_fake_probability(market):
    frame,dates,config,_=market;frame.loc[165,"model_input_eligible"]=False
    values,routes=legacy._chronological_probabilities(frame,Resolver(dates[160]),Predictor())
    assert np.isnan(values[165])
    account,diagnostic=legacy._replay_legacy(frame,dates,config,dates[160],dates[175],
        probabilities=values,routes=routes,offline=True)
    assert account["ledger"][0]["signal_date"]==dates[165]
    assert diagnostic["coverage"]["offline_filter_candidate_sessions"]==0
    assert diagnostic["coverage"]["rules_fallback_candidate_sessions"]==1


@pytest.mark.parametrize("change",["late","bad_sha","old_missing"])
def test_worker_preserves_missing_source_budget_and_failure_distinction(market,monkeypatch,tmp_path,change):
    frame,dates,config,records=market;record=dict(records[0])
    if change=="late":frame=frame.loc[frame.date>=dates[160]].copy()
    elif change=="old_missing":frame=frame.loc[(frame.date<dates[140])|(frame.date>dates[158])].copy()
    path=tmp_path/"source.parquet";frame.to_parquet(path,index=False)
    record.update(path=str(path),sha256=research._file_hash(path))
    if change=="bad_sha":record["sha256"]="wrong"
    monkeypatch.setattr(legacy,"_resolver",lambda *a:Resolver(dates[160]))
    from my_strategy.services import czsc_research_ml
    monkeypatch.setattr(czsc_research_ml,"PredictorSession",lambda **k:Predictor())
    window={"name":"window","start":dates[140],"end":dates[158],"dates":dates[140:159]}
    result=legacy._worker((record,dates,config,[window],"old-model",[],[],str(tmp_path/"out")))
    assert len(result)==2
    for binding in result:
        assert binding["status"]==("cash_only_not_observed_yet" if change=="late" else "cash_retained_failure")
        daily=pd.read_csv(Path(binding["path"])/"daily.csv")
        assert daily.date.tolist()==window["dates"]
        assert daily.equity.eq(100000).all() and daily.cash.eq(100000).all()
        assert pd.read_csv(Path(binding["path"])/"ledger.csv").empty
        if change=="late":assert binding["source_first_date"]==dates[160]


def test_model_prediction_failure_does_not_replace_valid_rule_account(market,monkeypatch,tmp_path):
    frame,dates,config,records=market;record=dict(records[0]);path=tmp_path/"source.parquet"
    frame.to_parquet(path,index=False);record.update(path=str(path),sha256=research._file_hash(path))
    monkeypatch.setattr(legacy,"_resolver",lambda *a:Resolver(dates[140]))
    monkeypatch.setattr(legacy,"_chronological_probabilities",lambda *a:(_ for _ in ()).throw(ValueError("model drift")))
    window={"name":"window","start":dates[140],"end":dates[158],"dates":dates[140:159]}
    result=legacy._worker((record,dates,config,[window],"old-model",[],[],str(tmp_path/"out")))
    assert result[0]["status"]=="complete"
    assert not pd.read_csv(Path(result[0]["path"])/"ledger.csv").empty
    assert result[1]["status"]=="cash_retained_failure"


def test_new_supplement_does_not_enter_frozen_high_return_dependency_glob():
    assert Path(legacy.__file__) not in research._code_files()


def test_control_rejects_non_fixed_ten_day_target(tmp_path):
    manifest=tmp_path/"manifest.json";manifest.write_text("{}")
    resolver=SimpleNamespace(catalog=[{"model_dir":str(tmp_path),"manifest":{"label_contract":{
        "horizon":60,"initial_cash":100000}},"feature_profile":"ma_trend_v1"}])
    with pytest.raises(ValueError,match="fixed-ten-session"):legacy._model_bindings(resolver)


def test_numpy_decision_scalars_are_saved_as_json_values(tmp_path):
    path=tmp_path/"evidence.json";legacy._save(path,{"flag":np.bool_(True),"probability":np.float32(.5)})
    assert json.loads(path.read_text())=={"flag":True,"probability":.5}


@pytest.mark.parametrize("change",[None,"source","cash_config","full_claim","window_dates","request_hash"])
def test_independent_audit_binds_original_anchor_request(tmp_path,monkeypatch,change):
    from my_strategy.services.czsc_high_return_audit import _Audit
    from my_strategy.services.czsc_high_return_evaluation import _window
    dates=["2025-01-02","2025-01-03","2025-01-06"]
    records=[{"symbol":"000001.SZ","path":"/original/one","sha256":"one"},
        {"symbol":"001400.SZ","path":"/original/two","sha256":"two"}]
    snapshot={"run_id":"source","effective_data_end":dates[-1],"snapshot_hash":"sourcehash"}
    calendar={"dates":dates,"verified":True};config={"initial_cash":100000}
    anchor=tmp_path/"anchor"
    for name,value in [("source_records",records),("source",snapshot),("calendar",calendar)]:
        legacy._save(anchor/"data_snapshots"/(name+".json"),value)
    legacy._save(anchor/"config/research.json",config)
    monkeypatch.setattr(legacy,"ARTIFACT_RUNS_ROOT",tmp_path)
    monkeypatch.setattr(research,"_source",lambda *a:(records,dates,snapshot,None))
    monkeypatch.setattr(legacy,"_code_hashes",lambda p:{"frozen.py":"sha"})
    plan=_window({"name":"window","start":dates[0],"end":dates[-1]},dates)
    identity={"anchor_run_id":"anchor","config":config,"code_hashes":{"frozen.py":"sha"},"windows":[plan]}
    report={"anchor_run_id":"anchor","dataset_records":[dict(records[0])],"anchor_mainboard_count":2,
        "requested_mainboard_count":1,"complete_anchor_mainboard_universe":False,"config":config,
        "calendar":calendar,"source_snapshot":snapshot,"evaluation":{"evaluation_identity":legacy.stable_hash(identity),
            "windows":{"window":{"fold":{k:v for k,v in plan.items() if k!="dates"}}}}}
    if change=="source":report["dataset_records"][0]["sha256"]="other-valid-source"
    if change=="cash_config":report["config"]={"initial_cash":1}
    if change=="full_claim":report["complete_anchor_mainboard_universe"]=True
    if change=="window_dates":identity["windows"][0]["dates"]=dates[1:]
    if change=="request_hash":report["evaluation"]["evaluation_identity"]="resigned-other-request"
    state=_Audit();legacy._audit_anchor_bindings(report,identity,state)
    assert bool(state.errors)==(change is not None)


def test_policy_reexecution_rejects_changed_cash_path_with_same_final_equity():
    rows=[{"date":"2025-01-02","cash":100000.,"equity":100000.},
        {"date":"2025-01-03","cash":100010.,"equity":100010.}]
    changed=[{**rows[0],"cash":90000.},rows[1]]
    assert not legacy._same_rows(rows,changed,("date","cash","equity"))


@pytest.mark.parametrize('offline',[False,True])
def test_actual_release_eligibility_does_not_claim_application_in_rule_arm(market,offline):
    frame,dates,config,_=market
    values,routes=legacy._chronological_probabilities(frame,Resolver(dates[140]),Predictor(.8))
    for route in routes:route['applied_to_entry']=bool(route['model_dir'])
    _,diagnostic=legacy._replay_legacy(frame,dates,config,dates[140],dates[175],
        probabilities=values,routes=routes,offline=offline)
    coverage=diagnostic['coverage']
    assert coverage['actual_release_eligible_sessions']==coverage['sessions']
    assert coverage['actual_release_applied_sessions']==(coverage['model_input_and_probability_sessions'] if offline else 0)


def test_unified_cli_runs_exact_supplement_request(monkeypatch,capsys):
    from my_strategy.cli import main
    calls=[]
    def run(anchor,**kwargs):
        calls.append((anchor,kwargs));return {'run_id':'new-controls','publication_allowed':False}
    monkeypatch.setattr(legacy,'run_legacy_comparison',run)
    assert main(['research-legacy-comparison','--anchor-run-id','frozen-full','--stocks','000001.SZ,001400.SZ',
        '--cpu-workers','2','--json'])==0
    anchor,request=calls[0]
    assert anchor=='frozen-full' and request['symbols']==['000001.SZ','001400.SZ']
    assert request['legacy_model_run_id']==legacy.DEFAULT_MA_RUN and request['cpu_workers']==2
    assert json.loads(capsys.readouterr().out)=={'run_id':'new-controls','publication_allowed':False}


@pytest.mark.parametrize('valid',[False,True])
def test_unified_cli_audit_preserves_integrity_exit_status(monkeypatch,capsys,valid):
    from my_strategy.cli import main
    calls=[]
    def audit(run_id,**kwargs):
        calls.append((run_id,kwargs));return {'artifact_integrity_passed':valid,'publication_allowed':False}
    monkeypatch.setattr(legacy,'audit_legacy_comparison',audit)
    assert main(['research-legacy-comparison-audit','--comparison-run-id','saved-controls','--as-of',
        '2026-10-11T00:00:00+08:00','--json'])==int(not valid)
    assert calls==[('saved-controls',{'as_of':'2026-10-11T00:00:00+08:00'})]
    assert json.loads(capsys.readouterr().out)['publication_allowed'] is False


def test_unified_cli_rejects_worker_count_before_launch(monkeypatch,capsys):
    from my_strategy.cli import main
    monkeypatch.setattr(legacy,'run_legacy_comparison',lambda *a,**k:pytest.fail('invalid request launched'))
    assert main(['research-legacy-comparison','--anchor-run-id','frozen-full','--cpu-workers','0'])==1
    assert '1到16' in capsys.readouterr().err


def _subset_anchor_controls_fixture(frozen_source, monkeypatch, tmp_path):
    """Three physical sources; a two-source anchor is independent of its control subset."""
    from my_strategy.tests.test_czsc_high_return_research import Context, write_json
    from my_strategy.storage import czsc_results
    from my_strategy.services import czsc_high_return_audit
    root, dates, source_records, source_report, config = frozen_source
    records = [dict(r) for r in source_records]
    third = dict(records[0], symbol="000002.SZ")
    third["path"] = str(root / "dataset/000002_SZ.parquet")
    records.append(third)
    for record in records:
        frame = pd.read_parquet(source_records[0]["path"] if record is third else record["path"])
        frame["symbol"] = record["symbol"]
        frame["input_eligible"] = True
        frame["rule_buy"] = False
        frame["rule_sell"] = False
        frame.loc[145, "rule_buy"] = True
        frame.loc[150, "rule_sell"] = True
        frame.to_parquet(record["path"], index=False)
        record["sha256"] = research._file_hash(Path(record["path"]))
    source_report = {**source_report, "dataset_records": records}
    write_json(root / "reports/czsc.json", source_report)
    monkeypatch.setattr(legacy, "ARTIFACT_RUNS_ROOT", root.parent)
    monkeypatch.setattr(czsc_high_return_audit, "ARTIFACT_RUNS_ROOT", root.parent)
    old = root.parent / "old-ma/reports/research.json"
    write_json(old, {"fixed_ten_day_fixture": True})
    # Models are outside this regression's source-subset concern; no head is
    # available, so both real Broker arms execute their explicit rule fallback.
    monkeypatch.setattr(legacy, "_resolver", lambda *a: Resolver("2030-01-01"))
    monkeypatch.setattr(legacy, "_release_events", lambda: [])
    contexts = []

    def create(**kwargs):
        context = Context(root.parent / ("controls-" + str(len(contexts))))
        context.run_id = context.root.name
        contexts.append(context)
        return context

    class Store:
        def save(self, kind, report):
            return report

    monkeypatch.setattr(legacy, "create_run_context", create)
    monkeypatch.setattr(czsc_results, "ResultStore", Store)

    def anchor(name, selected):
        frozen, calendar, snapshot, _ = research._source("frozen-source", symbols=selected)
        path = root.parent / name
        for key, value in (("source_records", frozen), ("source", snapshot),
                ("calendar", {"dates": calendar, "verified": True})):
            write_json(path / "data_snapshots" / (key + ".json"), value)
        write_json(path / "config/research.json", config)
        write_json(path / "config/initial_code_hashes.json", {
            str(p): research._file_hash(p) for p in research._code_files()})
        return path

    return root, dates, records, anchor


def test_actual_subset_and_full_anchor_controls_reconstruct_their_own_source_request(
        frozen_source, monkeypatch, tmp_path):
    root, dates, records, anchor = _subset_anchor_controls_fixture(frozen_source, monkeypatch, tmp_path)
    source_before = {str(p): research._file_hash(p) for p in root.rglob("*") if p.is_file()}
    window = {"name": "actual_broker", "start": dates[140], "end": dates[158]}
    for name, symbols in (("subset-anchor", [r["symbol"] for r in records[:2]]),
            ("full-anchor", [r["symbol"] for r in records])):
        path = anchor(name, symbols)
        report = legacy.run_legacy_comparison(path.name, legacy_model_run_id="old-ma",
            symbols=[records[0]["symbol"]], windows=[window], cpu_workers=1)
        assert report["anchor_mainboard_count"] == len(symbols)
        assert report["source_snapshot"]["selected_count"] == len(symbols)
        assert report["source_snapshot"]["full_source_universe"] == (len(symbols) == 3)
        assert report["requested_mainboard_count"] == 1
        assert report["dataset_records"] == [records[0]]
        assert report["requested_initial_cash"] == 100000.
        for bindings in report["evaluation"]["windows"]["actual_broker"]["ledger_bindings"].values():
            assert bindings[0]["status"] == "complete"
            ledger = pd.read_csv(Path(bindings[0]["path"]) / "ledger.csv")
            assert ledger.action.tolist() == ["BUY", "SELL"]
            assert ledger.iloc[0].signal_date == dates[145]
            assert ledger.iloc[0].date == dates[146]
            assert ledger.fee.gt(0).all()
        audit = legacy.audit_legacy_comparison(report["run_id"])
        assert audit["artifact_integrity_passed"] and not audit["errors"], audit["errors"]
        assert not audit["publication_allowed"] and not audit["alias_updated"]
        assert report["legacy_risk_qualified_high_outcome"] == "unavailable_without_actual_entry_stop_identity"
    assert source_before == {str(p): research._file_hash(p) for p in root.rglob("*") if p.is_file()}


def test_subset_anchor_wrong_or_extra_sources_fail_execution_and_independent_audit(
        frozen_source, monkeypatch, tmp_path):
    root, dates, records, anchor = _subset_anchor_controls_fixture(frozen_source, monkeypatch, tmp_path)
    path = anchor("subset-anchor", [r["symbol"] for r in records[:2]])
    window = {"name": "actual_broker", "start": dates[140], "end": dates[158]}
    report = legacy.run_legacy_comparison(path.name, legacy_model_run_id="old-ma",
        symbols=[records[0]["symbol"]], windows=[window], cpu_workers=1)
    original = json.loads((path / "data_snapshots/source_records.json").read_text())
    for fault in ("wrong_sha", "extra_original_source", "foreign_source"):
        changed = [dict(r) for r in original]
        if fault == "wrong_sha":
            changed[0]["sha256"] = "unapproved-source-revision"
        elif fault == "extra_original_source":
            changed.append(records[2])
        else:
            changed[0]["symbol"] = "000999.SZ"
        legacy._save(path / "data_snapshots/source_records.json", changed)
        with pytest.raises(ValueError, match="anchor does not match|frozen source universe"):
            legacy.run_legacy_comparison(path.name, legacy_model_run_id="old-ma",
                symbols=[records[0]["symbol"]], windows=[window], cpu_workers=1)
        audit = legacy.audit_legacy_comparison(report["run_id"])
        assert not audit["artifact_integrity_passed"] and audit["errors"]
        assert any(row["rule"] in {"legacy_original_frozen_source_references",
            "legacy_exact_anchor_input_subset", "legacy_original_anchor_audit"} for row in audit["errors"])
    legacy._save(path / "data_snapshots/source_records.json", original)
    restored = legacy.audit_legacy_comparison(report["run_id"])
    assert restored["artifact_integrity_passed"], restored["errors"]
