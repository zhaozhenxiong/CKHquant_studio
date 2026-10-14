"""Versioned, account-free reference training, source reuse and model registry.

The price target, statistical checks and reference path are independent of the
retired Broker labels. Technical completion does not certify future returns.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, METADATA_ROOT
from my_strategy.core.run_context import stable_hash, current_git_commit
from my_strategy.core.tz import local_now

FAMILY = "reference_position_v1"
DEFAULTS = dict(device="auto", feature_start="2023-01-01", end=None,
    source_native_run_id=None, horizon_sessions=10, epochs=12, batch_size=8192,
    cpu_workers=4, seed=42, max_stock_weight=.20, risk_budget=.01,
    position_step=.025, hysteresis=.01)


def _roots(runs_root=None, metadata_root=None):
    return Path(runs_root or ARTIFACT_RUNS_ROOT), Path(metadata_root or METADATA_ROOT)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w.-]+", value) or value in {".", ".."}:
        raise ValueError("无效的参考模型/计划标识")
    return value


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _save(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parameters(values=None):
    supplied = values or {}
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError("参考架构不接受旧训练参数：" + ", ".join(sorted(unknown)))
    result = {**DEFAULTS, **supplied}
    from my_strategy.core.device import requested_torch_device
    result["device"] = requested_torch_device(result["device"])
    for key in ("feature_start", "end"):
        if result[key] is not None: date.fromisoformat(result[key])
    for key, low, high in (("epochs", 1, 200), ("batch_size", 64, 65536), ("cpu_workers", 1, 16), ("seed", 0, 2**31-1)):
        value = result[key]
        if isinstance(value, bool) or int(value) != value or not low <= value <= high:
            raise ValueError(f"{key} 超出范围")
        result[key] = int(value)
    if result["horizon_sessions"] not in (10, 20): raise ValueError("参考方向只支持 10 或 20 个交易日")
    for key in ("max_stock_weight", "risk_budget", "position_step", "hysteresis"):
        if not math.isfinite(float(result[key])) or not 0 < float(result[key]) <= 1:
            raise ValueError(f"{key} 必须为 (0,1] 内有限数")
        result[key] = float(result[key])
    if result["position_step"] > result["max_stock_weight"]: raise ValueError("仓位档位不能高于每股上限")
    if result["hysteresis"] >= result["position_step"] / 2: raise ValueError("滞回必须小于半个仓位档位")
    return result


def capabilities(**unused):
    from my_strategy.core.device import get_default_device, is_mps_available, is_cuda_available
    fields = []
    labels = {"device": "计算设备", "feature_start": "特征起始日", "end": "冻结来源截止日", "source_native_run_id": "复用结构来源",
        "horizon_sessions": "方向预测交易日", "epochs": "训练轮数", "batch_size": "GPU 批大小", "cpu_workers": "数据准备线程",
        "seed": "随机种子", "max_stock_weight": "每股参考仓位上限", "risk_budget": "参考风险预算", "position_step": "仓位变化档位", "hysteresis": "仓位滞回"}
    for key, value in DEFAULTS.items():
        field = dict(key=key, label=labels[key], group="参考模型", type="number" if isinstance(value,(float,int)) else "text", default=value)
        if key == "device": field.update(type="select", options=["auto", "mps", "cuda", "cpu"])
        if key == "horizon_sessions": field.update(type="select", options=[10,20])
        fields.append(field)
    return dict(model_family="current", strategy_version=FAMILY, devices=["auto", *(["cuda"] if is_cuda_available() else []), *(["mps"] if is_mps_available() else []), "cpu"],
        actual_auto_device=get_default_device(), schema={"fields":fields}, defaults=DEFAULTS.copy(),
        stages=[dict(key=k, label=v, trainable=k=="reference_model", status="available") for k,v in
            (("source_verification","行情、日历与因果结构核验"),("reference_labels","纯价格方向标签"),("reference_model","Torch 三分类、校准与独立验证"),("reference_path","每股目标仓位与变化"))],
        qualification=dict(technical="来源、时间、资产、数值验证", predictive="独立政策段对训练先验的 LogLoss 与 Brier 检查", publication_allowed=False))


def profiles(*, metadata_root=None, **unused):
    _, meta = _roots(metadata_root=metadata_root)
    base = dict(profile_id="reference_builtin", name="参考买卖点与每股仓位", revision=0, parameters=DEFAULTS.copy(), builtin=True)
    saved = _read(meta/"reference_default.json") if (meta/"reference_default.json").exists() else base
    return dict(items=[base] if saved is base else [base,saved], default_profile=saved, default_revision=saved["revision"], default_profile_id=saved["profile_id"])


def save_default(values=None, expected_revision=0, *, parameters=None, metadata_root=None, **unused):
    _, meta = _roots(metadata_root=metadata_root); meta.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(meta/"reference_profile_lock.db") as lock:
        lock.execute("BEGIN IMMEDIATE")
        current = profiles(metadata_root=meta)["default_revision"]
        if current != expected_revision: raise ValueError("默认参数已改变，请刷新后保存")
        values = globals()["parameters"](parameters if parameters is not None else values)
        result = dict(profile_id="reference_personal", name="个人参考默认", revision=current+1, parameters=values, builtin=False)
        _save(meta/"reference_default.json", result)
    return profiles(metadata_root=meta)


def _code_manifest():
    root = Path(__file__).parent
    return {name:_sha(root/name) for name in ("reference_model.py", "reference_training.py", "reference_runtime.py")}


def _source_header(source_id, runs):
    source_id = _id(source_id)
    source_path = runs/source_id/"reports/native_source.json"
    native = _read(source_path)
    if native.get("status") != "completed" or native.get("failures") or native.get("population_mode") != "full_liquidity_union":
        raise ValueError("仅复用完整、无失败的历史合格池原生来源")
    if native.get("reused_model_or_label") is not False: raise ValueError("复用来源包含旧标签或模型")
    records = native["dataset_records"]
    symbols = native["selected_symbols"]
    if not symbols or symbols != sorted(set(symbols)) or symbols != [r["symbol"] for r in records]:
        raise ValueError("来源股票人口不完整")
    snapshot = native["source_snapshot"]
    raw = _read(runs/_id(snapshot["raw_source_run_id"])/"reports/raw_source.json")
    if raw.get("snapshot_hash") != stable_hash({k:v for k,v in raw.items() if k != "snapshot_hash"}) or snapshot["raw_snapshot_hash"] != raw["snapshot_hash"]:
        raise ValueError("原始来源摘要哈希不符")
    if raw.get("status") != "completed" or raw.get("failures") or raw.get("selected_symbols") != symbols:
        raise ValueError("原始来源不完整")
    if native["liquidity"] != raw["liquidity"] or native["liquidity"] != snapshot["liquidity"]:
        raise ValueError("历史流动性人口谱系不同")
    from my_strategy.services.czsc_research_calendar import load_verified_calendar
    calendar, calendar_meta = load_verified_calendar(snapshot["calendar_run_id"],runs_root=runs)
    dates = [d for d in calendar if d <= native["data_end"]]
    if dates != native["calendar_dates"] or stable_hash(dates) != snapshot["calendar_hash"] or raw["dates"] != dates:
        raise ValueError("冻结来源日历不一致")
    # Cache scheduling code may change; all actual numerical generators must match.
    generator_root = Path(__file__).parents[1]
    for name in ("services/czsc_market_features.py", "services/czsc_market_profiles.py", "services/czsc_price_volume_features.py",
                 "services/czsc_price_volume_events.py", "services/czsc_native_evidence.py", "services/czsc_analysis.py", "services/czsc_liquidity.py"):
        if _sha(generator_root/name) != snapshot["code_hashes"].get(name):
            raise ValueError("因果数值生成码改变，不能复用来源："+name)
    return native, calendar, dict(source_native_run_id=source_id, native_report_sha256=_sha(source_path),
        raw_source_run_id=raw["run_id"], raw_snapshot_hash=raw["snapshot_hash"],
        calendar_run_id=snapshot["calendar_run_id"], calendar_hash=stable_hash(dates),
        calendar_report_sha256=calendar_meta["file_sha256"], historical_union_count=len(symbols),
        reused=["verified_raw_quotes","causal_structure_features","historical_liquidity_membership"],
        reused_models=False, reused_labels=False)


def _find_source(runs, end=None):
    candidates = []
    for path in runs.glob("*/reports/native_source.json"):
        try:
            body = _read(path)
            if body.get("status") == "completed" and not body.get("failures") and (end is None or body.get("data_end") == end):
                candidates.append((body["data_end"], path.parent.parent.name))
        except (ValueError,OSError,KeyError): pass
    if not candidates: raise ValueError("未找到完成的因果结构来源；请先准备对应截止日的原始行情与结构")
    return max(candidates)[1]


def preflight(parameters=None, profile_id=None, *, source_native_run_id=None, runs_root=None, metadata_root=None, progress=None, **unused):
    runs, meta = _roots(runs_root,metadata_root)
    errors=[]; warnings=[]; plan={}; values=globals()["parameters"](parameters if parameters is not None else profiles(metadata_root=meta)["default_profile"]["parameters"])
    try:
        from my_strategy.core.device import get_default_device
        actual = get_default_device(values["device"])
        source_id=source_native_run_id or values["source_native_run_id"] or _find_source(runs,values["end"])
        native, calendar, proof = _source_header(source_id,runs)
        if values["end"] and values["end"] != native["data_end"]: raise ValueError("请求截止日与冻结来源不同")
        if values["feature_start"] != native["liquidity"]["feature_start"]: raise ValueError("复用人口起始日必须与冻结来源相同")
        values.update(end=native["data_end"],source_native_run_id=source_id)
        warnings = ["复用冻结来源截至 "+native["data_end"]+"；新行情未自动补入此快照。", "历史研究重建不代表当时真实已发布的模型；独立前瞻收益认证仍待累积。"]
        plan=dict(plan_id=uuid.uuid4().hex, family=FAMILY, parameters=values, data_end=native["data_end"], effective_end=native["data_end"],
            source_reuse=proof, cohort=native["selected_symbols"], historical_union_count=len(native["selected_symbols"]), latest_eligible_count=native["liquidity"].get("latest_count",0),
            calendar_run_id=proof["calendar_run_id"], calendar_hash=proof["calendar_hash"], actual_device=actual,
            code_hashes=_code_manifest(), created_at=local_now().isoformat(), windows=development_windows(native["data_end"]),
            valid=True, errors=[], warnings=warnings, config=values, config_hash=stable_hash(values), publication_allowed=False)
        plan["plan_hash"]=stable_hash(plan); _save(meta/"reference_plans"/(plan["plan_id"]+".json"),plan)
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        errors.append(str(exc))
    return dict(valid=not errors, errors=errors, warnings=warnings, plan=plan)


def validate_plan(plan_id, parameters=None, *, runs_root=None, metadata_root=None, check_code=True, **unused):
    runs,meta=_roots(runs_root,metadata_root)
    plan=_read(meta/"reference_plans"/(_id(plan_id)+".json"))
    if plan.get("family") != FAMILY or plan.get("plan_hash") != stable_hash({k:v for k,v in plan.items() if k!="plan_hash"}):
        raise ValueError("参考计划身份改变")
    if check_code and plan["code_hashes"] != _code_manifest(): raise ValueError("预检后参考代码改变，请重新预检")
    if parameters is not None:
        values=globals()["parameters"](parameters)
        for key in ("end","source_native_run_id"):
            if values[key] is None: values[key]=plan["parameters"][key]
        if values != plan["parameters"]: raise ValueError("提交参数与冻结计划不同")
    _,_,proof=_source_header(plan["parameters"]["source_native_run_id"],runs)
    if proof != plan["source_reuse"]: raise ValueError("预检后来源改变")
    return plan


def development_windows(end):
    # Full quarters only; calibration and policy are distinct halves of prior quarter.
    end=date.fromisoformat(end); quarter=(end.month-1)//3+1
    qstart=date(end.year,(quarter-1)*3+1,1)
    def prior(d):
        d=d-timedelta(days=1); return date(d.year,((d.month-1)//3)*3+1,1)
    starts=[]; cursor=qstart
    for _ in range(4): cursor=prior(cursor); starts.append(cursor)
    windows=[]
    for test_start in reversed(starts):
        month=test_start.month+3; next_q=date(test_start.year+month//13,(month-1)%12+1,1)
        val_start=prior(test_start); train_end=val_start-timedelta(days=1)
        cal_end=val_start+timedelta(days=44); policy_start=cal_end+timedelta(days=1)
        windows.append(dict(name=f"{test_start.year}Q{(test_start.month-1)//3+1}",train_end=train_end.isoformat(),
            calibration_start=val_start.isoformat(),calibration_end=cal_end.isoformat(),policy_start=policy_start.isoformat(),policy_end=(test_start-timedelta(days=1)).isoformat(),
            test_start=test_start.isoformat(),test_end=(next_q-timedelta(days=1)).isoformat()))
    val_start=prior(qstart); cal_end=val_start+timedelta(days=44)
    windows.append(dict(name="production",train_end=(val_start-timedelta(days=1)).isoformat(),calibration_start=val_start.isoformat(),calibration_end=cal_end.isoformat(),
        policy_start=(cal_end+timedelta(days=1)).isoformat(),policy_end=(qstart-timedelta(days=1)).isoformat(),test_start=None,test_end=None))
    return windows


def _verified_record(record,runs):
    from my_strategy.services.czsc_trend_support import resolve_frozen_asset
    paths={}
    for role,path_key,hash_key in (("features","path","sha256"),("raw","raw_path","raw_sha256"),("liquidity","liquidity_path","liquidity_sha256")):
        path=resolve_frozen_asset(record[path_key],runs_root=runs)
        if not path.resolve().is_relative_to(runs.resolve()) or path.is_symlink(): raise ValueError("来源资产越界")
        if _sha(path) != record[hash_key]: raise ValueError(record["symbol"]+" 来源文件哈希不符："+role)
        paths[role]=path
    return paths


def _prepare_one(item):
    record,runs,calendar,config,out=item
    from my_strategy.services.reference_model import make_observations
    paths=_verified_record(record,runs)
    frame=pd.read_parquet(paths["features"])
    if len(frame)!=record["rows"] or frame.symbol.nunique()!=1 or str(frame.symbol.iloc[0])!=record["symbol"]:
        raise ValueError("特征人口/行数错误")
    observations=make_observations(frame,calendar,config)
    observations=observations[observations.date.astype(str).str[:10]>=config["feature_start"]].reset_index(drop=True)
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    observations.to_parquet(out,index=False)
    return dict(symbol=record["symbol"],path=str(out),sha256=_sha(out),rows=len(observations),
        eligible=int(observations.input_eligible.sum()),labels=int(observations.label_available.sum()),source_sha256=record["sha256"])


def _predictive_gate(validation):
    policy=validation.get("policy",validation.get("metrics",{}).get("policy",{}))
    baseline=policy.get("baseline",{})
    if not isinstance(baseline,dict): baseline={}
    loss=policy.get("log_loss"); brier=policy.get("brier")
    baseline_loss=policy.get("baseline_log_loss",baseline.get("log_loss")); baseline_brier=policy.get("baseline_brier",baseline.get("brier"))
    ok=all(isinstance(x,(int,float)) and math.isfinite(x) for x in (loss,brier,baseline_loss,baseline_brier))
    passed=bool(ok and loss<=baseline_loss and brier<=baseline_brier)
    return dict(passed=passed, status="passed" if passed else "pending", basis="policy_only_prior_baseline_logloss_and_brier",
        log_loss=loss,brier=brier,baseline_log_loss=baseline_loss,baseline_brier=baseline_brier)


def execute_training(plan_id, parameters=None, *, runs_root=None, metadata_root=None, progress=None, check_cancel=None, run_callback=None, **unused):
    runs,meta=_roots(runs_root,metadata_root)
    plan=validate_plan(plan_id,parameters,runs_root=runs,metadata_root=meta)
    claim=meta/"reference_plans"/(plan_id+".claimed")
    with claim.open("x") as f: f.write(local_now().isoformat())
    cfg=plan["parameters"]; start=time.monotonic()
    run_id=local_now().strftime("%Y%m%d-%H%M%S-%f")+"-"+cfg["end"].replace("-","")+"-reference-train-"+current_git_commit(short=True)
    root=runs/run_id; root.mkdir(parents=True,exist_ok=False)
    if run_callback: run_callback(run_id)
    report=dict(run_id=run_id,plan_id=plan_id,family=FAMILY,strategy_version=FAMILY,status="running",data_end=cfg["end"],config={**cfg,"position":{k:cfg[k] for k in ("max_stock_weight","risk_budget","position_step","hysteresis")}},
        source_reuse=plan["source_reuse"],historical_union_count=len(plan["cohort"]),started_at=local_now().isoformat(),folds=[],production=None,publication_allowed=False,
        training_started=True,technical_validation={"status":"pending"},qualification={"independent_forward":"pending","profit_certification":False})
    _save(root/"metadata.json",dict(run_id=run_id,task="reference-train",scope="train",source="web",as_of_date=cfg["end"],config_hash=stable_hash(cfg),git_commit=current_git_commit(),created_at=report["started_at"]))
    _save(root/"config/training_plan.json",plan)
    for name in plan["code_hashes"]:
        dest=root/"config/frozen_code"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__).parent/name,dest)
    _save(root/"reports/reference_training.json",report)
    def emit(stage,current=0,total=1,**details):
        if check_cancel: check_cancel()
        event=dict(stage=stage,current=int(current),total=int(total),at=local_now().isoformat(),**details)
        (root/"logs").mkdir(parents=True,exist_ok=True)
        with (root/"logs/training.jsonl").open("a") as stream:
            stream.write(json.dumps(event,ensure_ascii=False,default=str)+"\n")
        if progress: progress(stage,int(current),int(total),**details)
    try:
        native,calendar,_=_source_header(cfg["source_native_run_id"],runs)
        records=native["dataset_records"]; dataset=[]
        payloads=((r,runs,calendar,cfg,root/"dataset"/(r["symbol"].replace(".","_")+".parquet")) for r in records)
        # Bounded batches: avoid retaining thousands of futures or all raw frames.
        with ThreadPoolExecutor(max_workers=cfg["cpu_workers"]) as pool:
            from itertools import islice
            while batch:=list(islice(payloads,cfg["cpu_workers"]*2)):
                for result in pool.map(_prepare_one,batch):
                    dataset.append(result)
                    if len(dataset)%20==0 or len(dataset)==len(records): emit("reference_labels",len(dataset),len(records),symbol=result["symbol"],device="cpu",reason="float64_price_calendar_and_io")
        _save(root/"dataset/manifest.json",dict(family=FAMILY,records=dataset,source_reuse=plan["source_reuse"],code_hash=plan["code_hashes"]["reference_model.py"]))
        emit("dataset_load",0,len(dataset))
        data=pd.concat([pd.read_parquet(row["path"]) for row in dataset],ignore_index=True)
        for column in ("symbol","date"):
            if column in data: data[column]=data[column].astype("category")
        report["dataset"]=dict(rows=len(data),eligible=int(data.input_eligible.sum()),labeled=int(data.label_available.sum()),stocks=len(dataset),manifest_sha256=_sha(root/"dataset/manifest.json"))
        from my_strategy.services.reference_model import fit_fold
        for i,split in enumerate(plan["windows"]):
            emit("reference_model",i,len(plan["windows"]),fold=split["name"],device=plan["actual_device"])
            def model_progress(event):
                event=dict(event);stage=event.pop("stage","model_fit");current=event.pop("current",event.pop("completed",0));total=event.pop("total",1)
                event.setdefault("fold",split["name"]);emit(stage,current,total,**event)
            result=fit_fold(data,split,root/"models"/split["name"],device=cfg["device"],config=cfg,progress=model_progress)
            fold=dict(name=split["name"],split=split,model_path=str(result["model_path"]),validation=result["validation"],predictive_gate=_predictive_gate(result["validation"]),
                research_reconstruction=split["name"]!="production",effective_from=split.get("test_start"))
            if split["name"]=="production": report["production"]=fold
            else: report["folds"].append(fold)
            _save(root/"reports/reference_training.json",report)
        completed=local_now().isoformat(); today=completed[:10]
        production=report["production"]
        production["effective_from"]=next((d for d in calendar if d>today),None)
        production["frozen_at"]=completed
        production["effective_rule"]="first_verified_session_strictly_after_actual_freeze"
        report.update(status="validating",elapsed_seconds=round(time.monotonic()-start,3),actual_device=plan["actual_device"],
            predictive_gate=production["predictive_gate"],qualified=False,technical_validation={"status":"pending"},
            model_selection="inner_train_only",calibration="separate_chronological_positive_temperature",execution="none_reference_only")
        _save(root/"reports/reference_training.json",report)
        evaluate(run_id,runs_root=runs,metadata_root=meta)
        report.update(status="completed",completed_at=local_now().isoformat(),qualified=production["predictive_gate"]["passed"],
            technical_validation={"status":"passed","full_population":True,"artifact_hashes":True,"chronological_purge":True})
        _save(root/"reports/reference_training.json",report)
        # The current pointer is a research artifact selection, never a promise of positive targets.
        event=dict(run_id=run_id,family=FAMILY,effective_from=production["effective_from"],selected_at=completed,predictive_gate=production["predictive_gate"],publication_allowed=False)
        meta.mkdir(parents=True,exist_ok=True)
        with (meta/"reference_selection_events.jsonl").open("a") as stream: stream.write(json.dumps(event,ensure_ascii=False)+"\n")
        _save(meta/"reference_current.json",{**event,"report_sha256":_sha(root/"reports/reference_training.json")})
        emit("completed",len(plan["windows"]),len(plan["windows"]),run_id=run_id,predictive_gate=production["predictive_gate"])
        return dict(run_id=run_id,train_run_id=run_id,model_run_id=run_id,status="completed",report=report,reference_only=True,publication_allowed=False)
    except BaseException as exc:
        report.update(status="cancelled" if "cancel" in type(exc).__name__.lower() else "failed",error=f"{type(exc).__name__}: {exc}",ended_at=local_now().isoformat())
        _save(root/"reports/reference_training.json",report)
        raise


def active_reference_run_id(*,metadata_root=None,**unused):
    _,meta=_roots(metadata_root=metadata_root);path=meta/"reference_current.json"
    return _read(path).get("run_id") if path.exists() else None


def load_reference_run(run_id=None, *,runs_root=None,metadata_root=None,allow_validating=False):
    runs,meta=_roots(runs_root,metadata_root);run_id=run_id or active_reference_run_id(metadata_root=meta)
    if not run_id: raise ValueError("尚未完成参考模型训练")
    root=runs/_id(run_id);report=_read(root/"reports/reference_training.json")
    allowed={"completed","validating"} if allow_validating else {"completed"}
    if report.get("family")!=FAMILY or report.get("status") not in allowed: raise ValueError("参考模型尚未完整验证")
    pointer=meta/"reference_current.json"
    if pointer.exists():
        selected=_read(pointer)
        if selected.get("run_id")==run_id and selected.get("report_sha256")!=_sha(root/"reports/reference_training.json"):
            raise ValueError("当前参考报告哈希改变")
    for fold in [*report["folds"],report["production"]]:
        fold["model_path"]=str(root/"models"/fold["name"])
    return root,report


def model_for_date(report,day,mode="historical"):
    day=str(day)[:10]
    if mode in {"production","forward"}:
        fold=report.get("production")
        if fold and fold.get("effective_from") and day>=fold["effective_from"]: return fold.copy()
        return None
    for fold in report.get("folds",[]):
        split=fold["split"]
        if split["test_start"]<=day<=split["test_end"]: return fold.copy()
    # A production fit can support historical research only AFTER its policy cutoff,
    # explicitly as reconstruction; it never acquires a past live publication date.
    fold=report.get("production")
    if fold and day>fold["split"]["policy_end"]:
        result=fold.copy();result["research_reconstruction"]=True;return result
    return None


def evaluate(run_id, *,runs_root=None,metadata_root=None,progress=None,check_cancel=None,**unused):
    from my_strategy.services.reference_validation import validate_reference_artifacts
    return validate_reference_artifacts(run_id,runs_root=runs_root,metadata_root=metadata_root,
        progress=progress,check_cancel=check_cancel)


def status(*,runs_root=None,metadata_root=None,**unused):
    runs,meta=_roots(runs_root,metadata_root);models=[]
    active=active_reference_run_id(metadata_root=meta)
    for path in sorted(runs.glob("*/reports/reference_training.json"),reverse=True):
        try:
            report=_read(path);models.append(dict(run_id=report["run_id"],plan_id=report.get("plan_id"),validation_available=report["status"]=="completed",strategy_version=FAMILY,family=FAMILY,status=report["status"],data_end=report["data_end"],
                created_at=report["started_at"],completed_at=report.get("completed_at"),active=report["run_id"]==active,qualified=report.get("qualified",False),
                technical_validation=report.get("technical_validation"),predictive_gate=report.get("predictive_gate"),actual_device=report.get("actual_device"),dataset=report.get("dataset")))
        except (ValueError,OSError,KeyError):pass
    events=[];path=meta/"reference_selection_events.jsonl"
    if path.exists():events=[json.loads(line) for line in path.read_text().splitlines() if line]
    return dict(model_family="current",strategy_version=FAMILY,active_model_run_id=active,active_model={"run_id":active} if active else None,models=models,selection_events=events,
        current_selection=events[-1] if events else None,
        certification=dict(status="pending",publication_allowed=False,reason="参考模型不使用模拟成交认证；独立前瞻样本尚待累积"),integrity_errors=[])
