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
from my_strategy.services.czsc_trend_v2_contracts import (STRATEGY_ID, TRAINING_POPULATION,
    CANDIDATE_POLICY, ATTEMPT_IDENTITY, ECONOMIC_GROUP_IDENTITY, SPLIT_POLICY, ECONOMIC_WEIGHTING,
    ENTRY_HEADS, HOLDING_HEADS, head_dependency_policy)

VERSION = STRATEGY_ID
FEATURE_VERSION = "causal_trend_interval_daily_attempt_inputs_v2"


def _columns_for(head):
    from my_strategy.services.czsc_trend_v2_features import (L1_MA_ENTRY_COLUMNS,
        L1_MA_EXIT_COLUMNS, ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS)
    choices = {"ma_exit": L1_MA_EXIT_COLUMNS,
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


def _held_group_ids(frame):
    if "episode_id" not in frame:
        if frame.empty: return pd.Series(index=frame.index, dtype=str)
        raise ValueError("holding head requires actual held episode identity")
    episodes = frame.episode_id.astype("string")
    if (episodes.isna() | episodes.eq("")).any():
        raise ValueError("holding head requires actual held episode identity")
    return frame.symbol.astype(str) + ":" + episodes if "symbol" in frame else episodes


def _head(frame, requested):
    return requested or (frame.attrs.get("label_contract") or {}).get("model_head")


def _validate_contract(contract, head, columns):
    from my_strategy.services.czsc_trend_v2_labels import label_version_for
    if not contract or not contract.get("target") or not contract.get("source_snapshot_hash"):
        raise ValueError("a frozen nonempty target and source contract is required")
    saved = contract.get("contract_hash")
    body = {k:v for k,v in contract.items() if k != "contract_hash"}
    if saved != stable_hash(body):
        raise ValueError("target contract identity mismatch")
    if contract.get("family") != VERSION:
        raise ValueError("target contract belongs to another strategy family")
    if tuple(columns) != _columns_for(head):
        raise ValueError("unapproved causal feature columns")
    if contract.get("model_head") != head or contract.get("model_input_schema_hash") != schema_hash(columns, head):
        raise ValueError("target contract head/input binding mismatch")
    if (contract.get("feature_schema_hash") != schema_hash(columns, head)
            or contract.get("label_version") != label_version_for(head)):
        raise ValueError("target contract label/feature version mismatch")
    if any(contract.get(key) != value for key, value in head_dependency_policy(head).items()):
        raise ValueError("target contract head dependency policy mismatch")
    target = contract["target"]
    if head in HOLDING_HEADS and (target != "remaining_net_sell_proceeds_exit_next_open_minus_frozen_continue_over_current_market_value"
            or contract.get("training_population") != "actual_held_Broker_prefix_paired_liquidations_v2"
            or contract.get("population") != "actual_holding_remaining_inventory_pair"
            or contract.get("admission") != "actual_held_Broker_state_only"
            or contract.get("stage") != ("E0" if head == "ma_exit" else "E1")):
        raise ValueError("holding head requires actual-held remaining-value target and continuation stage")
    if head == "entry" and target != "actual_complete_fee_net_roundtrip_high_return_and_initial_risk":
        raise ValueError("entry head requires actual episode trend-interval target")
    if head in ENTRY_HEADS:
        population = contract.get("reference_population") or {}
        if (contract.get("stage") != "E1" or contract.get("training_population") != TRAINING_POPULATION
                or contract.get("population") != "actual_flat_pre_gate_daily_attempt_forced_entry_clone"
                or contract.get("admission") != "valid_original_BUY_base_risk_qualified_actual_flat_before_L2_gate"
                or contract.get("candidate_policy") != CANDIDATE_POLICY
                or contract.get("attempt_identity") != ATTEMPT_IDENTITY
                or contract.get("economic_group_identity") != ECONOMIC_GROUP_IDENTITY
                or contract.get("split_policy") != SPLIT_POLICY
                or contract.get("economic_weighting") != ECONOMIC_WEIGHTING
                or population.get("version") != TRAINING_POPULATION
                or population.get("split_policy") != SPLIT_POLICY
                or population.get("economic_weighting") != ECONOMIC_WEIGHTING
                or contract.get("reference_population_hash") != stable_hash(population)
                or not isinstance(population.get("prior_entry_catalog"), list)
                or population.get("prior_entry_catalog_hash") != stable_hash(population["prior_entry_catalog"])):
            raise ValueError("entry head requires strict-prior actual-flat daily-attempt population")
        _validate_reference_population(population, population.get("fold") or {})
    if head in {"return", "loss"} and target != "actual_complete_fee_net_roundtrip_" + head:
        raise ValueError("auxiliary head target mismatch")
    if head in {"return", "loss"} and contract.get("auxiliary_label_availability") != "same_entry_clone_risk_identity_verified_price_basis_complete_liquidation":
        raise ValueError("auxiliary head label availability qualification mismatch")


def _validate_reference_population(population, fold):
    if (not fold.get("name") or stable_hash(population.get("fold")) != stable_hash(fold)
            or population.get("population_end") != fold.get("policy_end")
            or not fold.get("calibration_start")):
        raise ValueError("entry reference population belongs to another chronological fold")
    for bundle in population["prior_entry_catalog"]:
        if (bundle.get("name") == fold["name"] or not bundle.get("available_at")
                or bundle["available_at"] >= fold["calibration_start"]):
            raise ValueError("entry reference population contains a non-prior or self head")


def schema_hash(columns, head):
    value={"family": VERSION, "version": FEATURE_VERSION, "columns": list(columns), "head": head,
           **head_dependency_policy(head),
           "semantics": "actual_flat_pre_gate_daily_attempt_or_actual_holding_prefix_only_no_future_outcomes_v2"}
    if head not in {'ma_entry','ma_exit'}:
        from my_strategy.services.czsc_trend_v2_features import ENTRY_SCHEMA_HASH, EXIT_SCHEMA_HASH
        value['operation_schema_hash']=EXIT_SCHEMA_HASH if head in {'exit','exit_advantage'} else ENTRY_SCHEMA_HASH
    return stable_hash(value)


def bind_inputs(frame, columns, head):
    if tuple(columns) != _columns_for(head):
        raise ValueError("unapproved causal feature columns")
    if not set(columns).issubset(frame):
        raise ValueError("trend-interval input columns missing")
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


def time_masks(frame, fold, regression=False, *, head=None):
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
    # The first observed decision assigns each economic family to one segment.
    # Mature rows within that segment use their own real liquidation cutoff;
    # later or unclosed attempts cannot erase already mature prior attempts.
    selected_head = _head(frame, head)
    group = _held_group_ids(frame) if selected_head in HOLDING_HEADS else _group_ids(frame)
    group_start = dates.groupby(group).transform("min")
    train = available & (group_start <= fold["train_end"]).to_numpy() & (ends <= fold["train_end"]).fillna(False).to_numpy()
    calibration = available & (group_start >= fold["calibration_start"]).to_numpy()
    calibration &= (group_start <= fold["calibration_end"]).to_numpy() & (ends <= fold["calibration_end"]).fillna(False).to_numpy()
    policy = available & (group_start >= fold["policy_start"]).to_numpy()
    policy &= (group_start <= fold["policy_end"]).to_numpy() & (ends <= fold["policy_end"]).fillna(False).to_numpy()
    if selected_head not in HOLDING_HEADS and "original_event_ids" in frame:
        atoms, first_dates = _atomic_first_observations(frame)
        for mask, start, end in ((calibration, fold["calibration_start"], fold["calibration_end"]),
                                 (policy, fold["policy_start"], fold["policy_end"])):
            mask &= np.asarray([bool(ids) and all(start <= first_dates[(symbol, atom)] <= end for atom in ids)
                for symbol, ids in atoms], dtype=bool)
    return train, calibration, policy


def _original_atoms(value):
    if isinstance(value, str):
        value = json.loads(value) if value.startswith("[") else [value]
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return tuple(sorted(set(str(item) for item in value if item is not None and str(item))))
    return ()


def _atomic_first_observations(frame):
    """Read each atom's prefix first pre-gate date, never future components."""
    columns = [name for name in ("date", "symbol", "original_event_ids", "original_event_first_pre_gate_dates") if name in frame]
    rows = frame[columns].to_dict("records")
    atoms = [(str(row.get("symbol", "")), _original_atoms(row.get("original_event_ids"))) for row in rows]
    first_dates = {}
    provided = "original_event_first_pre_gate_dates" in frame
    for index in sorted(range(len(rows)), key=lambda i: str(rows[i]["date"])[:10]):
        row = rows[index]; date = pd.Timestamp(row["date"]).date().isoformat()
        symbol, identifiers = atoms[index]
        mapping = row.get("original_event_first_pre_gate_dates")
        if provided:
            if isinstance(mapping, str):
                mapping = json.loads(mapping)
            if not isinstance(mapping, dict) or any(atom not in mapping for atom in identifiers):
                raise ValueError("original atom first pre-gate prefix dates missing")
        for atom in identifiers:
            value = pd.to_datetime(mapping[atom], errors="coerce") if provided else pd.Timestamp(date)
            if pd.isna(value) or value.date().isoformat() > date:
                raise ValueError("original atom first pre-gate date is not observable")
            first = value.date().isoformat(); key = symbol, atom
            if key in first_dates and provided and first_dates[key] != first:
                raise ValueError("original atom first pre-gate date changed after observation")
            first_dates.setdefault(key, first)
    return atoms, first_dates


def economic_component_ids(frame, mask, *, head=None):
    """Dependency components within this selected segment, not online IDs."""
    columns = [name for name in ("symbol", "opportunity_id", "episode_id", "economic_group_id", "original_event_ids") if name in frame]
    selected = frame[columns].iloc[np.flatnonzero(mask)]
    if _head(frame, head) in HOLDING_HEADS:
        return _held_group_ids(selected)
    group = _group_ids(selected).tolist()
    parent = list(range(len(selected)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    owners = {}
    for index, row in enumerate(selected.to_dict("records")):
        keys = [("group", group[index])]
        keys.extend(("atom", str(row.get("symbol", "")), atom) for atom in _original_atoms(row.get("original_event_ids")))
        for key in keys:
            if key in owners:
                parent[find(index)] = find(owners[key])
            else:
                owners[key] = index
    return pd.Series([find(i) for i in range(len(selected))], index=selected.index, dtype=int)


def _weights(frame, mask, *, head=None):
    components = economic_component_ids(frame, mask, head=head)
    weights = 1. / components.map(components.value_counts()).to_numpy(float)
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
    if head in ENTRY_HEADS:
        _validate_reference_population(contract["reference_population"], fold)
    if stable_hash(dataset.attrs.get("label_contract")) != stable_hash(contract):
        raise ValueError("dataset target contract differs from requested training target")
    if regression != (head in {"return", "exit_advantage"}):
        raise ValueError("head classification/continuous output contract mismatch")
    frame = bind_inputs(dataset, columns, head)
    if head in ENTRY_HEADS:
        required = {"attempt_id", "economic_group_id", "original_opportunity_ids",
            "original_event_ids", "original_event_first_pre_gate_dates"}
        if not required.issubset(frame) or frame[list(required)].isna().any().any():
            raise ValueError("daily-attempt entry identity fields missing")
        keys = ["symbol", "attempt_id"] if "symbol" in frame else ["attempt_id"]
        if frame.duplicated(keys).any() or frame.attempt_id.astype(str).eq("").any() or frame.economic_group_id.astype(str).eq("").any():
            raise ValueError("daily-attempt entry identities must be nonempty and unique")
        if any(not _original_atoms(ids) for ids in frame.original_event_ids):
            raise ValueError("daily-attempt entry original atom identities missing")
    train, calibration, policy = time_masks(frame, fold, regression, head=head)
    if train.sum() < config.get("min_train_rows", 100) or calibration.sum() < config.get("min_calibration_rows", 30):
        raise ValueError(f"insufficient mature {head} rows train={train.sum()} calibration={calibration.sum()}")
    labels = pd.to_numeric(frame.label).to_numpy(float)
    if not regression and (len(np.unique(labels[train])) < 2 or len(np.unique(labels[calibration])) < 2):
        raise ValueError(f"{head} train/calibration require both classes")
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    preprocess = _fit_preprocess(values[train])
    x = _transform(values, preprocess)
    weights = _weights(frame, train, head=head)
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
        raise ValueError("unknown trend-interval architecture")
    check()
    calibration_value = {"method":"none_continuous_target"} if regression else _calibrate(
        calibration_logits,labels[calibration],_weights(frame,calibration,head=head))
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
        "independent_dependency_components":{name:int(economic_component_ids(frame,mask,head=head).nunique())
            for name,mask in (("train",train),("calibration",calibration),("policy",policy))},
        "episode_weighting":head_dependency_policy(head)["economic_weighting"],
        "group_split_policy":head_dependency_policy(head)["split_policy"],
        "economic_group_identity":head_dependency_policy(head)["economic_group_identity"],
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
        raise ValueError("trend-interval head identity/hash mismatch")
    manifest["manifest_sha256"] = expected
    checkpoint = (root/manifest["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root) or _file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("trend-interval checkpoint integrity mismatch")
    if manifest["feature_schema_hash"] != schema_hash(manifest["columns"],manifest["head"]):
        raise ValueError("trend-interval feature semantics mismatch")
    dependency = head_dependency_policy(manifest["head"])
    if (manifest.get("group_split_policy") != dependency["split_policy"]
            or manifest.get("episode_weighting") != dependency["economic_weighting"]
            or manifest.get("economic_group_identity") != dependency["economic_group_identity"]):
        raise ValueError("saved head dependency policy metadata mismatch")
    if stable_hash(manifest["target_contract"]) != manifest["target_contract_hash"]:
        raise ValueError("trend-interval target contract mismatch")
    _validate_contract(manifest["target_contract"], manifest["head"], manifest["columns"])
    if manifest["head"] in ENTRY_HEADS:
        _validate_reference_population(manifest["target_contract"]["reference_population"], manifest["fold"])
    if manifest["regression"] != (manifest["head"] in {"return", "exit_advantage"}):
        raise ValueError("saved head classification/continuous output mismatch")
    if manifest["export_sha256"] and _file_hash(root/"linear-layers.json") != manifest["export_sha256"]:
        raise ValueError("trend-interval inference export integrity mismatch")
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
            raise ValueError("wrong trend-interval head")
        days = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        if validation:
            fold = manifest["fold"]
            valid = (days > fold["calibration_end"]) & (days >= fold["policy_start"]) & (days <= fold["policy_end"])
        else:
            valid = days > manifest["available_at"]
        if not valid.all() or as_of is not None and (days > str(as_of)[:10]).any():
            raise ValueError("trend-interval head unavailable at decision date")
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
            raise ValueError("trend-interval feature schema missing or mismatched")
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
            raise ValueError('wrong trend-interval head')
        stamp=pd.Timestamp(values['date'])
        if pd.isna(stamp): raise ValueError('trend-interval head unavailable at decision date')
        day=stamp.date().isoformat()
        fold=manifest['fold']
        valid=(fold['calibration_end']<day and fold['policy_start']<=day<=fold['policy_end']) if validation else day>manifest['available_at']
        if not valid: raise ValueError('trend-interval head unavailable at decision date')
        if usage_mode=='production':
            if validation or pd.Timestamp(day+'T15:00:00+08:00')<pd.Timestamp(manifest['training_completed_at']):
                raise ValueError('head had not actually completed training at production decision')
        elif usage_mode!='historical_reconstruction': raise ValueError('unknown head usage mode')
        if expected_contract is None or stable_hash(expected_contract)!=manifest['target_contract_hash']:
            raise ValueError('caller target/execution contract mismatch')
        if not set(manifest['columns']).issubset(values): raise ValueError('trend-interval input columns missing')
        def numeric(value):
            try: return float(value)
            except (ValueError,TypeError): return np.nan
        row=np.asarray([[numeric(values[column]) for column in manifest['columns']]],dtype=float)
        output=float(self._inference(row,manifest,model)[0])
        eligible=values.get('input_eligible',True)
        if pd.isna(eligible) or not bool(eligible): output=float('nan')
        self.diagnostics['rows']+=1; self.diagnostics['batches']+=1
        return output
