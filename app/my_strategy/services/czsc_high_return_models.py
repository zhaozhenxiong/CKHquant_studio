"""Chronological, episode-weighted heads for the independent interval family."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research_ml import _device, _fit_preprocess, _transform, _network, _metrics
from my_strategy.services.czsc_research import _file_hash, _save

VERSION = "czsc_high_return_v1"
FEATURE_VERSION = "causal_signal_interval_inputs_v1"


def _columns_for(head):
    from my_strategy.services.czsc_high_return_features import (L1_MA_ENTRY_COLUMNS,
        L1_MA_EXIT_COLUMNS, ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS)
    choices = {"ma_entry": L1_MA_ENTRY_COLUMNS, "ma_exit": L1_MA_EXIT_COLUMNS,
        "entry": ENTRY_FEATURE_COLUMNS, "exit": EXIT_FEATURE_COLUMNS,
        "return": ENTRY_FEATURE_COLUMNS, "loss": ENTRY_FEATURE_COLUMNS,
        "exit_advantage": EXIT_FEATURE_COLUMNS}
    if head not in choices:
        raise ValueError("unknown independent interval head")
    return tuple(choices[head])


def _group_ids(frame):
    fallback = pd.Series([f"row:{i}" for i in range(len(frame))], index=frame.index)
    for key in ("opportunity_id", "episode_id", "economic_group_id"):
        if key in frame:
            values = frame[key].astype("string")
            fallback = values.where(values.notna() & values.ne(""), fallback)
    return frame.symbol.astype(str) + ":" + fallback.astype(str) if "symbol" in frame else fallback.astype(str)


def _validate_contract(contract, head, columns):
    if not contract or not contract.get("target") or not contract.get("source_snapshot_hash"):
        raise ValueError("a frozen nonempty target and source contract is required")
    saved = contract.get("contract_hash")
    body = {k:v for k,v in contract.items() if k != "contract_hash"}
    if saved != stable_hash(body):
        raise ValueError("target contract identity mismatch")
    if tuple(columns) != _columns_for(head):
        raise ValueError("unapproved causal feature columns")
    if contract.get("model_head") != head or contract.get("model_input_schema_hash") != schema_hash(columns, head):
        raise ValueError("target contract head/input binding mismatch")
    target = contract["target"]
    if head == "ma_entry" and target != "actual_complete_fee_net_roundtrip_positive":
        raise ValueError("MA entry cannot reuse a fixed-horizon target")
    if head in {"ma_exit", "exit", "exit_advantage"} and "exit_next_open_minus_frozen_continue" not in target:
        raise ValueError("holding head requires remaining-value exit target")
    if head == "entry" and "high_return_and_initial_risk" not in target:
        raise ValueError("entry head requires actual episode high-return target")
    if head in {"return", "loss"} and target != "actual_complete_fee_net_roundtrip_" + head:
        raise ValueError("auxiliary head target mismatch")


def schema_hash(columns, head):
    value={"version": FEATURE_VERSION, "columns": list(columns), "head": head,
           "semantics": "entry_close_or_actual_holding_prefix_only_no_future_outcomes_v1"}
    if head not in {'ma_entry','ma_exit'}:
        from my_strategy.services.czsc_high_return_features import ENTRY_SCHEMA_HASH, EXIT_SCHEMA_HASH
        value['operation_schema_hash']=EXIT_SCHEMA_HASH if head in {'exit','exit_advantage'} else ENTRY_SCHEMA_HASH
    return stable_hash(value)


def bind_inputs(frame, columns, head):
    if tuple(columns) != _columns_for(head):
        raise ValueError("unapproved causal feature columns")
    if not set(columns).issubset(frame):
        raise ValueError("high-return input columns missing")
    result = frame.copy()
    result.attrs.update(feature_version=FEATURE_VERSION, schema_hash=schema_hash(columns, head))
    return result


def split_validation(fold, calendar):
    dates = [x for x in calendar if fold["validation_start"] <= x <= fold["validation_end"]]
    if len(dates) < 4:
        raise ValueError("not enough verified validation sessions")
    middle = len(dates) // 2
    return {**fold, "calibration_start": dates[0], "calibration_end": dates[middle-1],
            "policy_start": dates[middle], "policy_end": dates[-1]}


def time_masks(frame, fold, regression=False):
    if not (fold["train_end"] < fold["calibration_start"] <= fold["calibration_end"] <
            fold["policy_start"] <= fold["policy_end"] <= fold["validation_end"]):
        raise ValueError("train/calibration/policy chronology overlaps")
    dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(frame.label_end, errors="coerce").dt.strftime("%Y-%m-%d")
    labels = pd.to_numeric(frame.label, errors="coerce").to_numpy(float)
    available = frame.label_available.fillna(False).to_numpy(dtype=bool, copy=True)
    available &= frame.get("input_eligible", pd.Series(True, index=frame.index)).fillna(False).to_numpy(bool)
    available &= np.isfinite(labels) if regression else np.isin(labels, [0., 1.])
    available &= ends.notna().to_numpy()
    if (available & (ends <= dates).fillna(False).to_numpy()).any():
        raise ValueError("label maturity must follow decision date")
    # All daily states and policy branches of an economic episode stay together.
    group = _group_ids(frame)
    available &= pd.Series(available, index=frame.index).groupby(group).transform("all").to_numpy(bool)
    group_end = ends.groupby(group).transform("max")
    group_start = dates.groupby(group).transform("min")
    train = available & (dates <= fold["train_end"]).to_numpy() & (group_end <= fold["train_end"]).fillna(False).to_numpy()
    calibration = available & (dates >= fold["calibration_start"]).to_numpy() & (group_start >= fold["calibration_start"]).to_numpy()
    calibration &= (group_end <= fold["calibration_end"]).fillna(False).to_numpy()
    policy = available & (dates >= fold["policy_start"]).to_numpy() & (group_start >= fold["policy_start"]).to_numpy()
    policy &= (group_end <= fold["policy_end"]).fillna(False).to_numpy()
    return train, calibration, policy


def _weights(frame, mask):
    group = _group_ids(frame)
    selected = group.loc[mask]
    weights = 1. / selected.map(selected.value_counts()).to_numpy(float)
    return weights / weights.mean()


def _sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -40., 40.)))


def _calibrate(logits, labels, weights):
    """Two-parameter Platt calibration on a separate chronological segment."""
    design = np.column_stack([np.clip(logits, -30, 30), np.ones(len(logits))])
    coefficients = np.array([1., 0.])
    for _ in range(60):
        probability = _sigmoid(design @ coefficients)
        gradient = design.T @ (weights * (probability - labels)) + .001 * coefficients
        curvature = (design.T * (weights * probability * (1-probability))) @ design + np.eye(2)*.001
        step = np.linalg.solve(curvature, gradient)
        # Damping prevents separable rare-positive calibration from exploding.
        coefficients -= step / max(1., float(np.linalg.norm(step)))
        coefficients[0] = np.clip(coefficients[0], .01, 20.)
        coefficients[1] = np.clip(coefficients[1], -20., 20.)
        if np.linalg.norm(step) < 1e-7:
            break
    return {"method": "separate_chronological_platt", "slope": float(coefficients[0]), "intercept": float(coefficients[1])}


def train_head(dataset, columns, output_dir, fold, config, *, head, architecture="linear",
               regression=False, device="mps", contract=None, check_cancel=None, progress=None):
    import torch
    check = check_cancel or (lambda: None)
    started = local_now().isoformat()
    _validate_contract(contract, head, columns)
    if stable_hash(dataset.attrs.get("label_contract")) != stable_hash(contract):
        raise ValueError("dataset target contract differs from requested training target")
    if regression != (head in {"return", "exit_advantage"}):
        raise ValueError("head classification/continuous output contract mismatch")
    frame = bind_inputs(dataset, columns, head)
    train, calibration, policy = time_masks(frame, fold, regression)
    if train.sum() < config.get("min_train_rows", 100) or calibration.sum() < config.get("min_calibration_rows", 30):
        raise ValueError(f"insufficient mature {head} rows train={train.sum()} calibration={calibration.sum()}")
    labels = pd.to_numeric(frame.label).to_numpy(float)
    if not regression and (len(np.unique(labels[train])) < 2 or len(np.unique(labels[calibration])) < 2):
        raise ValueError(f"{head} train/calibration require both classes")
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    preprocess = _fit_preprocess(values[train])
    x = _transform(values, preprocess)
    weights = _weights(frame, train)
    positive_weight = 1.
    if not regression:
        positive_weight = float(weights[labels[train] == 0].sum() / weights[labels[train] == 1].sum())
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("head output must be an isolated empty directory")
    directory.mkdir(parents=True, exist_ok=True)
    check()
    training_batches, losses, export, actual_device = 0, [], None, "cpu"
    if architecture == "tree":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits
        import joblib
        estimator = (HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier)(
            max_iter=100, max_leaf_nodes=15, max_depth=3, min_samples_leaf=30,
            l2_regularization=1., early_stopping=False, random_state=int(config.get("seed", 42)))
        fit_weights = weights * (np.where(labels[train] == 1, positive_weight, 1.) if not regression else 1.)
        with threadpool_limits(limits=4):
            estimator.fit(x[train], labels[train], sample_weight=fit_weights)
        path = directory / "model.joblib"
        joblib.dump(estimator, path)
        with threadpool_limits(limits=4):
            if regression:
                calibration_logits = estimator.predict(x[calibration])
            else:
                probability = np.clip(estimator.predict_proba(x[calibration])[:,1], 1e-8, 1-1e-8)
                calibration_logits = np.log(probability/(1-probability))
        training_batches = 100
    elif architecture in {"linear", "mlp"}:
        selected = _device(device)
        actual_device = str(selected)
        torch.manual_seed(int(config.get("seed", 42)))
        torch.use_deterministic_algorithms(True)
        hidden = () if architecture == "linear" else (32,16)
        model = _network(len(columns), hidden).to(selected)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
        x_tensor = torch.from_numpy(x[train])
        y_tensor = torch.from_numpy(labels[train].astype(np.float32)).reshape(-1,1)
        weight_tensor = torch.from_numpy(weights.astype(np.float32)).reshape(-1,1)
        generator = torch.Generator().manual_seed(int(config.get("seed",42)))
        epochs, batch = int(config.get("epochs",20)), int(config.get("batch_size",2048))
        model.train()
        for epoch in range(epochs):
            total = 0.
            for indices in torch.randperm(len(x_tensor),generator=generator).split(batch):
                check()
                xx,yy,ww = x_tensor[indices].to(selected),y_tensor[indices].to(selected),weight_tensor[indices].to(selected)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xx)
                error = (prediction-yy).square() if regression else torch.nn.functional.binary_cross_entropy_with_logits(
                    prediction,yy,pos_weight=torch.tensor(positive_weight,device=selected),reduction="none")
                loss = (error*ww).mean()
                loss.backward(); optimizer.step()
                total += float(loss.detach().item())*len(indices); training_batches += 1
            losses.append(total/len(x_tensor))
            if progress: progress(epoch+1,epochs)
        model.eval()
        with torch.no_grad():
            calibration_logits = np.concatenate([model(torch.from_numpy(xx).to(selected)).cpu().numpy().ravel()
                for xx in np.array_split(x[calibration],max(1,math.ceil(calibration.sum()/batch)))])
        path = directory / "model.pt"
        torch.save({"state_dict":{key:value.detach().cpu() for key,value in model.state_dict().items()},
                    "columns":list(columns),"hidden_sizes":list(hidden),"family":VERSION,"head":head},path)
        export = [{"weight":layer.weight.detach().cpu().numpy().tolist(), "bias":layer.bias.detach().cpu().numpy().tolist()}
                  for layer in model.modules() if isinstance(layer,torch.nn.Linear)]
        _save(directory/"linear-layers.json",export)
    else:
        raise ValueError("unknown high-return architecture")
    check()
    calibration_value = {"method":"none_continuous_target"} if regression else _calibrate(
        calibration_logits,labels[calibration],_weights(frame,calibration))
    diagnostics = {"mean_squared_error":float(np.mean((calibration_logits-labels[calibration])**2))} if regression else _metrics(
        labels[calibration],_sigmoid(calibration_logits*calibration_value["slope"]+calibration_value["intercept"]))
    manifest = {"family":VERSION,"head":head,"architecture":architecture,"regression":regression,
        "feature_version":FEATURE_VERSION,"feature_schema_hash":schema_hash(columns,head),"columns":list(columns),
        "target_contract":contract or {},"target_contract_hash":stable_hash(contract or {}),"fold":fold,
        "preprocess":preprocess,"calibration":calibration_value,"calibration_metrics":diagnostics,
        "available_at":fold["validation_end"],"training_started_at":started,"training_completed_at":local_now().isoformat(),
        "train_label_end":str(pd.to_datetime(frame.loc[train,"label_end"]).max().date()),
        "calibration_label_end":str(pd.to_datetime(frame.loc[calibration,"label_end"]).max().date()),
        "sample_counts":{"train":int(train.sum()),"calibration":int(calibration.sum()),"policy":int(policy.sum()),"all":len(frame)},
        "episode_weighting":"inverse_rows_per_economic_episode_training_only",
        "positive_weight":positive_weight,"training_batches":training_batches,"training_loss":losses,
        "requested_device":device,"actual_device":actual_device,"actual_mps_training":actual_device=="mps",
        "checkpoint":path.name,"checkpoint_sha256":_file_hash(path),
        "export_sha256":_file_hash(directory/"linear-layers.json") if export is not None else None,
        "publication_allowed":False,"seed":config.get("seed",42)}
    manifest["manifest_sha256"] = stable_hash(manifest)
    _save(directory/"manifest.json",manifest)
    return manifest


def load_head(path):
    root = Path(path).resolve()
    manifest = json.loads((root/"manifest.json").read_text())
    expected = manifest.pop("manifest_sha256",None)
    if expected != stable_hash(manifest) or manifest.get("family") != VERSION:
        raise ValueError("high-return head identity/hash mismatch")
    manifest["manifest_sha256"] = expected
    checkpoint = (root/manifest["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root) or _file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("high-return checkpoint integrity mismatch")
    if manifest["feature_schema_hash"] != schema_hash(manifest["columns"],manifest["head"]):
        raise ValueError("high-return feature semantics mismatch")
    if stable_hash(manifest["target_contract"]) != manifest["target_contract_hash"]:
        raise ValueError("high-return target contract mismatch")
    _validate_contract(manifest["target_contract"], manifest["head"], manifest["columns"])
    if manifest["regression"] != (manifest["head"] in {"return", "exit_advantage"}):
        raise ValueError("saved head classification/continuous output mismatch")
    if manifest["export_sha256"] and _file_hash(root/"linear-layers.json") != manifest["export_sha256"]:
        raise ValueError("high-return inference export integrity mismatch")
    return manifest


class HeadSession:
    """Verified CPU batch and actual-state inference; training hardware is separate."""
    def __init__(self):
        self.cache = {}
        self.diagnostics = {"rows":0,"batches":0,"model_loads":0,"actual_device":"cpu"}

    def _model(self,path):
        key = str(Path(path).resolve())
        if key not in self.cache:
            manifest = load_head(path)
            if manifest["architecture"] == "tree":
                import joblib
                model = joblib.load(Path(path)/manifest["checkpoint"])
            else:
                model = [(np.asarray(x["weight"],np.float32),np.asarray(x["bias"],np.float32))
                         for x in json.loads((Path(path)/"linear-layers.json").read_text())]
            self.cache[key] = manifest,model
            self.diagnostics["model_loads"] += 1
        return self.cache[key]

    def _inference(self,values,manifest,model):
        values[~np.isfinite(values)] = np.nan
        x = _transform(values,manifest["preprocess"])
        if manifest["architecture"] == "tree":
            if manifest["regression"]:
                z = model.predict(x)
            else:
                p = np.clip(model.predict_proba(x)[:,1],1e-8,1-1e-8); z = np.log(p/(1-p))
        else:
            for index,(weight,bias) in enumerate(model):
                x = x @ weight.T + bias
                if index < len(model)-1: x = np.maximum(x,0)
            z = x.ravel()
        return z if manifest["regression"] else _sigmoid(z*manifest["calibration"]["slope"]+manifest["calibration"]["intercept"])

    def predict(self,frame,path,*,as_of=None,expected_head=None,expected_contract=None,validation=False,
                usage_mode="historical_reconstruction"):
        manifest,model = self._model(path)
        if expected_head is not None and manifest["head"] != expected_head:
            raise ValueError("wrong high-return head")
        days = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        if validation:
            fold = manifest["fold"]
            valid = (days > fold["calibration_end"]) & (days >= fold["policy_start"]) & (days <= fold["policy_end"])
        else:
            valid = days > manifest["available_at"]
        if not valid.all() or as_of is not None and (days > str(as_of)[:10]).any():
            raise ValueError("high-return head unavailable at decision date")
        if usage_mode == "production":
            completed = pd.to_datetime(manifest["training_completed_at"], utc=True)
            if validation or (pd.to_datetime(days + "T15:00:00+08:00", utc=True) < completed).any():
                raise ValueError("head had not actually completed training at production decision")
        elif usage_mode != "historical_reconstruction":
            raise ValueError("unknown head usage mode")
        caller_contract = expected_contract or frame.attrs.get("label_contract")
        if caller_contract is None or stable_hash(caller_contract) != manifest["target_contract_hash"]:
            raise ValueError("caller target/execution contract mismatch")
        if frame.attrs.get("schema_hash") != manifest["feature_schema_hash"]:
            raise ValueError("high-return feature schema missing or mismatched")
        values = frame[manifest["columns"]].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float,copy=True)
        result = self._inference(values,manifest,model)
        eligible = frame.get("input_eligible",pd.Series(True,index=frame.index)).fillna(False).to_numpy(bool)
        result = np.where(eligible,result,np.nan)
        self.diagnostics["rows"] += len(result); self.diagnostics["batches"] += 1
        return result

    def predict_one(self,values,path,*,expected_head=None,expected_contract=None,validation=False,
                    usage_mode="historical_reconstruction"):
        # Daily holding heads accept the same ordered causal columns and clocks
        # as batch inference, without constructing a wide DataFrame per fill.
        manifest,model=self._model(path)
        if expected_head is not None and manifest['head']!=expected_head:
            raise ValueError('wrong high-return head')
        stamp=pd.Timestamp(values['date'])
        if pd.isna(stamp): raise ValueError('high-return head unavailable at decision date')
        day=stamp.date().isoformat()
        fold=manifest['fold']
        valid=(fold['calibration_end']<day and fold['policy_start']<=day<=fold['policy_end']) if validation else day>manifest['available_at']
        if not valid: raise ValueError('high-return head unavailable at decision date')
        if usage_mode=='production':
            if validation or pd.Timestamp(day+'T15:00:00+08:00')<pd.Timestamp(manifest['training_completed_at']):
                raise ValueError('head had not actually completed training at production decision')
        elif usage_mode!='historical_reconstruction': raise ValueError('unknown head usage mode')
        if expected_contract is None or stable_hash(expected_contract)!=manifest['target_contract_hash']:
            raise ValueError('caller target/execution contract mismatch')
        if not set(manifest['columns']).issubset(values): raise ValueError('high-return input columns missing')
        def numeric(value):
            try: return float(value)
            except (ValueError,TypeError): return np.nan
        row=np.asarray([[numeric(values[column]) for column in manifest['columns']]],dtype=float)
        output=float(self._inference(row,manifest,model)[0])
        eligible=values.get('input_eligible',True)
        if pd.isna(eligible) or not bool(eligible): output=float('nan')
        self.diagnostics['rows']+=1; self.diagnostics['batches']+=1
        return output
