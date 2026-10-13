"""Independent next-open to horizon-close three-class trend observations.

Prices, not trading profits, define this head. Actual account exits and high
return intervals have separate contracts. Old two-class profitable-episode
heads cannot be loaded or renamed as trend forecasts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_support import _save, _file_hash
from my_strategy.services.czsc_trend_math import _fit_preprocess, _transform
from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, TREND_TARGET

FAMILY = STRATEGY_ID
HEAD = "trend_direction"
CLASSES = ("UP", "DOWN", "NEUTRAL")
TARGET = TREND_TARGET
FEATURE_VERSION = "czsc_trend_independent_MA_direction"
DEFAULT_TREND_CONFIG = {"horizon_sessions": 10, "neutral_floor": .01, "atr_multiple": 1.,
    "min_train_per_class": 100, "min_calibration_per_class": 30, "min_policy_per_class": 30}
SCHEMA_HASH = stable_hash({"family": FAMILY, "feature_version": FEATURE_VERSION, "head": HEAD, "columns": MA_TREND_COLUMNS,
    "input": "known_close_twenty_independent_MA_inputs", "output": CLASSES,
    "dependency": "65_verified_raw_sessions_MA60_slope5;no_missing_or_unverified_quotes",
    "target": TARGET, "price_basis": "unadjusted_verified_no_ambiguous_corporate_action"})


def trend_config(config=None):
    incoming = (config or {}).get("trend", {})
    cfg = {**DEFAULT_TREND_CONFIG, **{k: incoming[k] for k in DEFAULT_TREND_CONFIG if k in incoming}}
    if "minimum_return_boundary" in incoming:
        cfg["neutral_floor"] = incoming["minimum_return_boundary"]
    for key in ("horizon_sessions", "min_train_per_class", "min_calibration_per_class", "min_policy_per_class"):
        if not math.isfinite(float(cfg[key])) or int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError("trend counts must be positive integers")
        cfg[key] = int(cfg[key])
    if cfg["horizon_sessions"] not in {10, 20} or not 0 < cfg["neutral_floor"] < 1 or cfg["atr_multiple"] <= 0:
        raise ValueError("invalid preregistered trend horizon/band")
    return cfg


def trend_contract(config=None, *, source_snapshot_hash, pipeline_contract_hash):
    from my_strategy.services.czsc_liquidity import liquidity_policy
    policy = liquidity_policy((config or {}).get('liquidity_universe'))
    body = {"family": FAMILY, "head": HEAD, "target": TARGET, "classes": list(CLASSES),
        "parameters": trend_config(config), "feature_version": FEATURE_VERSION, "feature_schema_hash": SCHEMA_HASH,
        "source_snapshot_hash": source_snapshot_hash, "pipeline_contract_hash": pipeline_contract_hash,
        "horizon": "H includes next verified entry-reference session; terminal=t+H market-calendar close",
        "band": "log1p(max(neutral_floor,atr_multiple*prior14_ATR/signal_close))",
        "maturity": "terminal_verified_close", "population": "historical_daily_liquidity_qualified_MA_observations",
        "liquidity_policy": policy, "liquidity_policy_hash": stable_hash(policy),
        "execution": "analytical_direction_not_profit;actual_entry_guards_separate",
        "missing": "no_missing_open_end_calendar_or_action_is_neutral",
        "publication_allowed": False}
    return {**body, "contract_hash": stable_hash(body)}


def _validate_trend_contract(contract):
    body = {k: v for k, v in contract.items() if k != "contract_hash"}
    if (contract.get("contract_hash") != stable_hash(body)
            or contract.get("family") != FAMILY or contract.get("head") != HEAD
            or contract.get("target") != TARGET or contract.get("classes") != list(CLASSES)
            or contract.get("feature_version") != FEATURE_VERSION
            or contract.get("feature_schema_hash") != SCHEMA_HASH
            or contract.get('population') != 'historical_daily_liquidity_qualified_MA_observations'
            or contract.get('liquidity_policy_hash') != stable_hash(contract.get('liquidity_policy'))
            or not contract.get("source_snapshot_hash") or not contract.get("pipeline_contract_hash")):
        raise ValueError("trend target/input family contract mismatch")


def _calendar(values):
    dates = [pd.Timestamp(x).date().isoformat() for x in values]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("trend observations require an explicit sorted verified calendar")
    return dates


def bind_trend_inputs(frame,market_dates=None,config=None):
    if not set(MA_TREND_COLUMNS) <= set(frame):
        raise ValueError("trend requires the twenty ordered MA columns")
    result = frame.copy()
    finite = np.isfinite(result[list(MA_TREND_COLUMNS)].apply(pd.to_numeric,errors='coerce').to_numpy(float)).all(axis=1)
    original = result.get('model_input_eligible',pd.Series(False,index=result.index)).fillna(False).to_numpy(bool)
    calendar=_calendar(market_dates or result.attrs.get('verified_market_dates',[]))
    dates=pd.to_datetime(result.date).dt.strftime('%Y-%m-%d').tolist();market={d:i for i,d in enumerate(calendar)}
    quality=result.source.isin(VERIFIED_SOURCES)&result.has_trade_price.eq(1)
    for key in ('open','high','low','close'):
        numeric=pd.to_numeric(result[key],errors='coerce');quality &= np.isfinite(numeric)&numeric.gt(0)
    if 'available_at' in result:
        stamps=pd.to_datetime(result.available_at,errors='coerce',utc=True)
        cutoffs=pd.to_datetime(pd.Series(dates,index=result.index)+'T15:00:00+08:00',utc=True)
        quality &= stamps.notna() & stamps.le(cutoffs)
    if 'wyckoff_corporate_action_suspected' in result:
        quality &= ~result.wyckoff_corporate_action_suspected.fillna(False).eq(True)
    raw_ok=quality.to_numpy(bool);dependencies=[]
    for i,day in enumerate(dates):
        pos=market.get(day,-1)
        complete=i>=64 and pos>=64 and dates[i-64:i+1]==calendar[pos-64:pos+1]
        dependencies.append(bool(complete and raw_ok[i-64:i+1].all()))
    result['trend_dependency_complete']=dependencies
    result['model_input_eligible'] = original & finite & np.asarray(dependencies,bool)
    result['trend_input_reason'] = np.where(result['model_input_eligible'],'qualified_twenty_MA_inputs','MA_dependency_or_feature_input_unavailable')
    from my_strategy.services.czsc_liquidity import annotate_liquidity
    result = annotate_liquidity(result, calendar, (config or {}).get('liquidity_universe'), verify_existing=True)
    # Liquidity qualifies new forecasts/entries. It must never remove the same
    # twenty MA inputs from a position's already pinned holding-exit head.
    result['trend_input_eligible'] = result['model_input_eligible'] & result['liquidity_pool_eligible']
    result.loc[result.model_input_eligible & ~result.liquidity_pool_eligible, 'trend_input_reason'] = 'historical_liquidity_ineligible'
    result.attrs.update(feature_version=FEATURE_VERSION, schema_hash=SCHEMA_HASH, feature_columns=list(MA_TREND_COLUMNS), head=HEAD)
    return result


def build_trend_labels(frame, market_dates, config=None, *, source_snapshot_hash, pipeline_contract_hash):
    """Keep unavailable/immature rows and exact nominal calendar horizons."""
    cfg = trend_config(config); calendar = _calendar(market_dates); market = {d: i for i, d in enumerate(calendar)}
    data = bind_trend_inputs(frame,calendar,config); data["date"] = pd.to_datetime(data.date).dt.strftime("%Y-%m-%d")
    if data.empty or data.symbol.nunique() != 1 or data.date.duplicated().any() or list(data.date) != sorted(data.date):
        raise ValueError("trend labels require one unique ascending stock prefix")
    if not set(data.date) <= set(calendar):
        raise ValueError("stock dates are outside verified calendar")
    rows = data.to_dict("records"); index = {r["date"]: i for i, r in enumerate(rows)}
    numeric = data[list(MA_TREND_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    high, low, close = (pd.to_numeric(data[k], errors="coerce") for k in ("high", "low", "close"))
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.shift().rolling(14, min_periods=14).mean().to_numpy(float)
    def quote(row, key):
        try:
            return bool(row["source"] in VERIFIED_SOURCES and row["has_trade_price"] == 1
                        and math.isfinite(float(row[key])) and float(row[key]) > 0)
        except (ValueError, TypeError, KeyError):
            return False
    outcomes = []
    for i, row in enumerate(rows):
        day = row["date"]; m = market[day]; h = cfg["horizon_sessions"]
        entry_day = calendar[m+1] if m+1 < len(calendar) else None
        terminal = calendar[m+h] if m+h < len(calendar) else None
        input_ok = bool(row.get("trend_input_eligible", False) and np.isfinite(numeric[i]).all() and quote(row, "close"))
        band = math.log1p(max(cfg["neutral_floor"], cfg["atr_multiple"]*atr[i]/float(row["close"]))) if input_ok and np.isfinite(atr[i]) else None
        outcome = {"symbol": str(row["symbol"]), "date": day, "available_at": day+"T15:00:00+08:00",
            "input_eligible": input_ok, "label_available": False, "label": np.nan, "label_class": None,
            "label_start": entry_day, "label_end": terminal, "label_available_at": None,
            "horizon_sessions": h, "trend_band_log": band, "future_log_return": np.nan,
            "label_reason": "input_ineligible", "observation_id": stable_hash({"symbol": row["symbol"], "date": day,
                "horizon": h, "band_parameters": cfg, "schema_hash": SCHEMA_HASH,
                "liquidity_policy_hash": row['liquidity_policy_hash']})}
        outcome.update({key: row[key] for key in row if key.startswith('liquidity_')})
        if not row['liquidity_pool_eligible']:
            outcome['label_reason'] = 'historical_liquidity_ineligible'
        if input_ok and band is not None:
            if terminal is None or terminal > rows[-1]["date"]:
                outcome["label_reason"] = "horizon_not_mature"
            elif entry_day not in index or terminal not in index:
                outcome["label_reason"] = "missing_reference_open_or_horizon_close"
            else:
                e, z = index[entry_day], index[terminal]
                complete = z-e+1 == h and all(calendar[m+1+j] == rows[e+j]["date"] for j in range(h))
                action = any(bool(r.get("wyckoff_corporate_action_suspected", False)) for r in rows[e:z+1])
                if not complete:
                    outcome["label_reason"] = "future_calendar_dependency_incomplete"
                elif action:
                    outcome["label_reason"] = "future_price_basis_or_corporate_action_ambiguous"
                elif not quote(rows[e], "open") or not quote(rows[z], "close"):
                    outcome["label_reason"] = "reference_quote_unavailable"
                else:
                    r = math.log(float(rows[z]["close"])/float(rows[e]["open"]))
                    label = 0 if r > band else 1 if r < -band else 2
                    outcome.update(label=float(label), label_class=CLASSES[label], future_log_return=r,
                        label_available=True, label_available_at=terminal+"T15:00:00+08:00", label_reason="available")
        outcomes.append(outcome)
    result = pd.concat([data[list(MA_TREND_COLUMNS)].reset_index(drop=True), pd.DataFrame(outcomes)], axis=1)
    result.attrs.update(label_contract=trend_contract(config, source_snapshot_hash=source_snapshot_hash,
                        pipeline_contract_hash=pipeline_contract_hash), schema_hash=SCHEMA_HASH)
    return result


def chronological_folds(end, market_dates, *, first="2024Q1"):
    """Expanding train; separate chronological calibration and policy quarters halves."""
    calendar = _calendar(market_dates); folds = []
    for period in pd.period_range(first, pd.Timestamp(end).to_period("Q"), freq="Q"):
        validation, training = period-1, period-2
        dates = [d for d in calendar if str(validation.start_time.date()) <= d <= str(validation.end_time.date())]
        if len(dates) < 4:
            continue
        mid = len(dates)//2
        folds.append({"name": "trend_"+str(period), "train_end": str(training.end_time.date()),
            "calibration_start": dates[0], "calibration_end": dates[mid-1], "policy_start": dates[mid],
            "policy_end": dates[-1], "validation_end": dates[-1], "test_start": str(period.start_time.date()),
            "test_end": min(str(period.end_time.date()), str(end))})
    return folds


def trend_time_masks(frame, fold):
    if not fold["train_end"] < fold["calibration_start"] <= fold["calibration_end"] < fold["policy_start"] <= fold["policy_end"]:
        raise ValueError("trend train/calibration/policy overlap")
    dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d"); ends = pd.to_datetime(frame.label_end, errors="coerce").dt.strftime("%Y-%m-%d")
    eligible = frame.input_eligible.fillna(False) & frame.label_available.fillna(False) & frame.label.isin([0., 1., 2.]) & ends.notna()
    if (eligible & (ends <= dates)).any():
        raise ValueError("trend label maturity does not follow observation")
    return tuple(mask.to_numpy(bool) for mask in (
        eligible & (dates <= fold["train_end"]) & (ends <= fold["train_end"]),
        eligible & (dates >= fold["calibration_start"]) & (ends <= fold["calibration_end"]),
        eligible & (dates >= fold["policy_start"]) & (ends <= fold["policy_end"])))


def _softmax(logits):
    values = np.asarray(logits, float); values -= values.max(axis=1, keepdims=True)
    ex = np.exp(np.clip(values, -60, 0)); return ex/ex.sum(axis=1, keepdims=True)


def _concurrency_weights(frame, mask):
    selected = frame.loc[mask,['symbol','date','label_end']]; result = []
    for _, group in selected.groupby("symbol", sort=False):
        days = sorted(set(group.date.tolist()+group.label_end.tolist()))
        pos = {d:i for i,d in enumerate(days)}
        starts=group.date.map(pos).to_numpy(int);ends=group.label_end.map(pos).to_numpy(int)
        changes=np.zeros(len(days)+1,float)
        np.add.at(changes,starts,1.);np.add.at(changes,ends+1,-1.)
        counts=np.cumsum(changes[:-1]);reciprocal=np.zeros_like(counts)
        np.divide(1.,counts,out=reciprocal,where=counts>0)
        prefix=np.r_[0.,np.cumsum(reciprocal)]
        weights=(prefix[ends+1]-prefix[starts])/(ends-starts+1)
        result.extend(zip(group.index,weights))
    weights = pd.Series(dict(result)).reindex(selected.index).to_numpy(float)
    return weights/weights.mean()


def trend_metrics(labels, probabilities):
    from sklearn.metrics import log_loss, f1_score, confusion_matrix
    y = np.asarray(labels, int); p = np.asarray(probabilities, float)
    return {"rows": len(y), "class_counts": {name:int((y==i).sum()) for i,name in enumerate(CLASSES)},
        "log_loss": float(log_loss(y, p, labels=[0,1,2])),
        "brier_multiclass": float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
        "macro_f1": float(f1_score(y, p.argmax(axis=1), labels=[0,1,2], average="macro", zero_division=0)),
        "confusion": confusion_matrix(y, p.argmax(axis=1), labels=[0,1,2]).tolist(),
        "mean_probability": dict(zip(CLASSES, p.mean(axis=0).tolist()))}


def train_trend_head(dataset, output_dir, fold, config, *, contract):
    """Train-only logistic preprocessing; independent vector calibration segment."""
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize
    import joblib
    from threadpoolctl import threadpool_limits
    if contract != dataset.attrs.get("label_contract") or contract.get("contract_hash") != stable_hash({k:v for k,v in contract.items() if k!="contract_hash"}):
        raise ValueError("trend dataset/target identity mismatch")
    _validate_trend_contract(contract)
    data = dataset if dataset.observation_id.is_unique else dataset.drop_duplicates("observation_id").reset_index(drop=True)
    train, calibration, policy = trend_time_masks(data, fold); cfg = trend_config(config)
    labels = pd.to_numeric(data.label).to_numpy(float)
    counts = {name: {c:int((labels[m] == i).sum()) for i,c in enumerate(CLASSES)}
              for name,m in (("train",train),("calibration",calibration),("policy",policy))}
    for name, limit in (("train",cfg["min_train_per_class"]),("calibration",cfg["min_calibration_per_class"]),("policy",cfg["min_policy_per_class"])):
        if min(counts[name].values()) < limit:
            raise ValueError("insufficient mature trend classes "+name+" "+str(counts[name]))
    started = local_now().isoformat(); x = data[list(MA_TREND_COLUMNS)].to_numpy(float)
    pre = _fit_preprocess(x[train]); xx = _transform(x, pre)
    model = LogisticRegression(C=1., max_iter=500, solver="lbfgs", random_state=int(config.get("seed",42)))
    with threadpool_limits(limits=2): model.fit(xx[train], labels[train].astype(int), sample_weight=_concurrency_weights(data,train))
    if list(model.classes_) != [0,1,2]: raise ValueError("trend model classes changed")
    logits = model.decision_function(xx[calibration]); cy = labels[calibration].astype(int); cw = _concurrency_weights(data,calibration)
    def objective(parameters):
        t = math.exp(float(parameters[0])); bias = np.r_[parameters[1:], 0.]
        p = _softmax(logits/t+bias)
        return float(np.sum(cw*-np.log(np.maximum(p[np.arange(len(cy)),cy],1e-12)))/cw.sum()+.001*np.sum(parameters**2))
    fit = minimize(objective, np.zeros(3), method="L-BFGS-B", bounds=[(-2,2),(-5,5),(-5,5)])
    if not fit.success or not np.isfinite(fit.x).all(): raise ValueError("three-class calibration failed")
    cal = {"method":"separate_chronological_temperature_and_class_intercepts", "temperature":math.exp(float(fit.x[0])), "bias":np.r_[fit.x[1:],0.].tolist()}
    def probabilities(mask):return _softmax(model.decision_function(xx[mask])/cal["temperature"]+np.asarray(cal["bias"]))
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):raise FileExistsError("trend head requires a new empty directory")
    directory.mkdir(parents=True,exist_ok=True);path=directory/"model.joblib";joblib.dump(model,path)
    manifest={"family":FAMILY,"feature_version":FEATURE_VERSION,"head":HEAD,"target":TARGET,"classes":list(CLASSES),"columns":list(MA_TREND_COLUMNS),
        "feature_schema_hash":SCHEMA_HASH,"target_contract":contract,"target_contract_hash":stable_hash(contract),
        "pipeline_contract_hash":contract["pipeline_contract_hash"],"fold":fold,"preprocess":pre,"calibration":cal,
        "train_label_end":str(data.loc[train,"label_end"].max()),"calibration_label_end":str(data.loc[calibration,"label_end"].max()),
        "policy_label_end":str(data.loc[policy,"label_end"].max()),"sample_counts":counts,
        "calibration_metrics":trend_metrics(cy,probabilities(calibration)),"policy_metrics":trend_metrics(labels[policy],probabilities(policy)),
        "simulated_available_at":fold["policy_end"],"available_at":fold["policy_end"],
        "training_started_at":started,"actual_training_completed_at":local_now().isoformat(),
        "checkpoint":"model.joblib","checkpoint_sha256":_file_hash(path),"publication_allowed":False,
        "input_population":"all_qualified_daily_MA","overlap_weighting":"inverse_concurrent_label_information_not_transitive_groups",
        "requested_device":"cpu","actual_device":"cpu"}
    manifest["manifest_sha256"]=stable_hash(manifest);_save(directory/"manifest.json",manifest);return manifest


def load_trend_head(path):
    directory=Path(path).resolve();m=json.loads((directory/"manifest.json").read_text());sha=m.pop("manifest_sha256",None)
    if sha!=stable_hash(m) or m.get("family")!=FAMILY or m.get("feature_version")!=FEATURE_VERSION or m.get("head")!=HEAD or m.get("target")!=TARGET or m.get("classes")!=list(CLASSES):
        raise ValueError("trend manifest identity mismatch")
    m["manifest_sha256"]=sha;checkpoint=(directory/m["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(directory) or _file_hash(checkpoint)!=m["checkpoint_sha256"] or m["feature_schema_hash"]!=SCHEMA_HASH:
        raise ValueError("trend checkpoint/schema integrity mismatch")
    if stable_hash(m["target_contract"])!=m["target_contract_hash"]:raise ValueError("trend contract drift")
    _validate_trend_contract(m["target_contract"])
    return m


class TrendSession:
    def __init__(self):self.cache={};self.diagnostics={"model_loads":0,"rows":0,"batches":0,"actual_device":"cpu"}
    def predict(self,frame,path,*,expected_contract,mode="historical",policy=False):
        import joblib
        key=str(Path(path).resolve())
        if key not in self.cache:
            m=load_trend_head(path);self.cache[key]=(m,joblib.load(Path(path)/m["checkpoint"]));self.diagnostics["model_loads"]+=1
        m,model=self.cache[key]
        if expected_contract!=m["target_contract"] or frame.attrs.get("schema_hash")!=SCHEMA_HASH:raise ValueError("trend inference target/input mismatch")
        days=pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        valid=((days>=m["fold"]["policy_start"])&(days<=m["fold"]["policy_end"])) if policy else days>m["simulated_available_at"]
        if not valid.all():raise ValueError("trend head not available at forecast")
        if mode=="production":
            if policy or (pd.to_datetime(days+"T15:00:00+08:00",utc=True)<pd.Timestamp(m["actual_training_completed_at"])).any():raise ValueError("trend head not actually trained at forecast")
        elif mode!="historical":raise ValueError("unknown trend inference mode")
        xx=_transform(frame[list(MA_TREND_COLUMNS)].to_numpy(float),m["preprocess"])
        p=_softmax(model.decision_function(xx)/m["calibration"]["temperature"]+np.asarray(m["calibration"]["bias"]))
        eligible=frame.get("trend_input_eligible",frame.get("model_input_eligible",frame.get("input_eligible",pd.Series(False,index=frame.index)))).fillna(False).to_numpy(bool)
        p[~eligible]=np.nan;self.diagnostics["rows"]+=len(frame);self.diagnostics["batches"]+=1
        return p,m
