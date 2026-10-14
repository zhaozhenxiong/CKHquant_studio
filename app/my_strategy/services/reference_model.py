"""Causal, account-free direction model for reference-position research.

Prices and availability checks use CPU float64. Fitting, preprocessing,
calibration and batched neural inference use the explicitly selected Torch
float32 device. Historical folds are research reconstructions, never evidence
of a model having existed before its actual training time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from my_strategy.core.device import select_torch_device, synchronize_torch_device
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES

CLASSES = ("UP", "DOWN", "NEUTRAL")
FAMILY = "reference_position_v1"
WYCKOFF_COLUMNS = (
    "w_spread_atr", "w_body_atr", "w_close_location", "w_gap_atr",
    "w_relative_volume20", "w_relative_amount20", "w_up_down_volume20",
    "w_price_volume_divergence", "w_pullback_push_volume20", "w_volume_contraction5",
    "w_range_width_atr", "w_range_position", "w_support_distance_atr",
    "w_resistance_distance_atr", "w_demand_score", "w_supply_score",
)
FEATURE_COLUMNS = (*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS, *WYCKOFF_COLUMNS)
SCHEMA_HASH = stable_hash({"family": FAMILY, "columns": FEATURE_COLUMNS, "classes": CLASSES,
    "optional_missing": "train_mean_with_missing_indicators", "required": MA_TREND_COLUMNS,
    "target": "next_verified_open_to_t_plus_H_close_ATR_neutral_band",
    "input": "known_close_verified_65_sessions_confirmed_structure_and_causal_wyckoff_v1"})
DEFAULT_CONFIG = {"horizon_sessions": 10, "neutral_floor": .01, "atr_multiple": 1.,
    "epochs": 12, "batch_size": 8192, "seed": 42, "inner_epochs": 5,
    "inner_sample_cap": 200_000, "min_train_per_class": 30, "min_segment_per_class": 5}


def _config(config=None):
    incoming = config or {}
    cfg = {**DEFAULT_CONFIG, **{k: incoming[k] for k in DEFAULT_CONFIG if k in incoming}}
    if cfg["horizon_sessions"] not in (10, 20):
        raise ValueError("reference horizon must be 10 or 20 verified sessions")
    for key in ("epochs", "batch_size", "inner_epochs", "inner_sample_cap", "min_train_per_class", "min_segment_per_class"):
        if int(cfg[key]) != cfg[key] or int(cfg[key]) < 1:
            raise ValueError(f"invalid positive integer {key}")
        cfg[key] = int(cfg[key])
    if not 0 < float(cfg["neutral_floor"]) < 1 or not 0 < float(cfg["atr_multiple"]) < 100:
        raise ValueError("invalid neutral band")
    cfg["seed"] = int(cfg["seed"])
    return cfg


def _dates(values):
    return pd.to_datetime(values, errors="raise").dt.strftime("%Y-%m-%d")


def _calendar(values):
    dates = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise")).strftime("%Y-%m-%d").tolist()
    if not dates or dates != sorted(set(dates)):
        raise ValueError("an explicit ascending unique verified market calendar is required")
    return dates


def _flag(frame, name):
    return frame[name].fillna(False).eq(True).to_numpy(bool) if name in frame else np.zeros(len(frame), bool)


def _observable(frame, field, cutoff, *, missing_ok=False):
    if field not in frame:
        return np.full(len(frame), missing_ok, bool)
    observed = pd.to_datetime(frame[field], errors="coerce", utc=True, format="mixed")
    valid = observed.notna() & observed.le(cutoff)
    if missing_ok:
        valid |= frame[field].isna()
    return valid.to_numpy(bool)


def prepare_inputs(frame, calendar=None):
    """Build the same compact input schema for training and live inference.

    Optional evidence is explicitly missing when its own qualification fails;
    imputation never turns an unqualified MA observation into a scored row.
    A calendar is mandatory unless already bound in frame.attrs.
    """
    required = {"symbol", "date", "source", "has_trade_price", "open", "high", "low", "close", *MA_TREND_COLUMNS}
    absent = required - set(frame)
    if absent:
        raise ValueError(f"reference input columns missing: {sorted(absent)}")
    days = _dates(frame["date"])
    if frame.empty or frame.symbol.nunique() != 1 or days.duplicated().any() or not days.is_monotonic_increasing:
        raise ValueError("reference inputs require one nonempty ascending unique stock")
    cal = _calendar(calendar if calendar is not None else frame.attrs.get("verified_market_dates", []))
    pos = pd.Index(cal).get_indexer(days)
    if (pos < 0).any():
        raise ValueError("stock dates outside verified calendar")
    out = pd.DataFrame(index=range(len(frame)))
    out["symbol"] = frame.symbol.astype(str).to_numpy()
    out["date"] = days.to_numpy()
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32) if col in frame else np.nan
    quotes = frame.source.isin(VERIFIED_SOURCES).to_numpy() & frame.has_trade_price.eq(1).to_numpy()
    for col in ("open", "high", "low", "close"):
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        quotes &= np.isfinite(arr) & (arr > 0)
    quotes &= ~_flag(frame, "wyckoff_corporate_action_suspected")
    cutoff = pd.to_datetime(days + "T15:00:00+08:00", utc=True)
    quotes &= _observable(frame, "available_at", cutoff)
    # A missing market session invalidates every MA window crossing the gap.
    bad_prefix = np.r_[0, np.cumsum(~quotes)]
    ix = np.arange(len(frame)); start = np.maximum(0, ix - 64)
    complete = (ix >= 64) & ((pos - pos[start]) == 64) & ((bad_prefix[ix + 1] - bad_prefix[start]) == 0)
    ma_finite = np.isfinite(out[list(MA_TREND_COLUMNS)].to_numpy()).all(axis=1)
    eligible = _flag(frame, "model_input_eligible") & _flag(frame, "liquidity_pool_eligible") & complete & ma_finite
    eligible &= _observable(frame, "liquidity_observed_at", cutoff)
    structure_ok = _flag(frame, "structure_input_eligible") & _observable(frame, "structure_confirmed_at", cutoff)
    for field in ("zone_confirmed_at", "weekly_available_at", "monthly_available_at"):
        structure_ok &= _observable(frame, field, cutoff, missing_ok=True)
    out.loc[~structure_ok, list(STRUCTURE_COLUMNS)] = np.nan
    wyckoff_ok = _flag(frame, "wyckoff_input_eligible") & _observable(frame, "wyckoff_available_at", cutoff)
    out.loc[~wyckoff_ok, list(WYCKOFF_COLUMNS)] = np.nan
    out["input_eligible"] = eligible
    out["structure_evidence_available"] = structure_ok
    out["wyckoff_evidence_available"] = wyckoff_ok
    out["close"] = pd.to_numeric(frame.close, errors="coerce").to_numpy(np.float64)
    out.attrs.update(schema_hash=SCHEMA_HASH, feature_columns=list(FEATURE_COLUMNS), verified_market_dates=cal)
    return out


def make_observations(frame, calendar, config=None):
    """Vectorized exact-calendar labels; unavailable futures never become neutral."""
    cfg = _config(config); cal = _calendar(calendar); h = cfg["horizon_sessions"]
    frame = frame.reset_index(drop=True)
    out = prepare_inputs(frame, cal)
    n = len(out); pos = pd.Index(cal).get_indexer(out.date)
    starts = pos + 1; ends = pos + h
    entry_days = np.asarray(cal, dtype=object)[np.minimum(starts, len(cal)-1)]
    end_days = np.asarray(cal, dtype=object)[np.minimum(ends, len(cal)-1)]
    entry_days[starts >= len(cal)] = None; end_days[ends >= len(cal)] = None
    row_for_market = np.full(len(cal), -1, dtype=int); row_for_market[pos] = np.arange(n)
    entry = row_for_market[np.minimum(starts, len(cal)-1)]
    terminal = row_for_market[np.minimum(ends, len(cal)-1)]
    entry[starts >= len(cal)] = -1; terminal[ends >= len(cal)] = -1
    close = pd.to_numeric(frame.close, errors="coerce").to_numpy(np.float64)
    opening = pd.to_numeric(frame.open, errors="coerce").to_numpy(np.float64)
    high = pd.to_numeric(frame.high, errors="coerce").to_numpy(np.float64)
    low = pd.to_numeric(frame.low, errors="coerce").to_numpy(np.float64)
    previous = np.r_[np.nan, close[:-1]]
    tr = np.fmax(high-low, np.fmax(np.abs(high-previous), np.abs(low-previous)))
    atr = pd.Series(tr).shift().rolling(14, min_periods=14).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        band = np.log1p(np.maximum(cfg["neutral_floor"], cfg["atr_multiple"] * atr / close))
    quote_ok = frame.source.isin(VERIFIED_SOURCES).to_numpy() & frame.has_trade_price.eq(1).to_numpy()
    for arr in (opening, high, low, close):
        quote_ok &= np.isfinite(arr) & (arr > 0)
    quote_ok &= ~_flag(frame, "wyckoff_corporate_action_suspected")
    quote_ok &= _observable(frame, "available_at", pd.to_datetime(out.date+"T15:00:00+08:00", utc=True))
    bad_prefix = np.r_[0, np.cumsum(~quote_ok)]
    e = np.maximum(entry, 0); z = np.maximum(terminal, 0)
    continuous = (entry >= 0) & (terminal >= 0) & ((terminal-entry+1) == h)
    continuous &= (bad_prefix[z+1] - bad_prefix[e]) == 0
    available = out.input_eligible.to_numpy() & np.isfinite(band) & continuous
    returns = np.full(n, np.nan)
    returns[available] = np.log(close[z[available]] / opening[e[available]])
    labels = np.full(n, np.nan, dtype=np.float32)
    labels[available] = np.where(returns[available] > band[available], 0,
                                np.where(returns[available] < -band[available], 1, 2))
    out["label_start"] = entry_days; out["label_end"] = end_days
    out["label_available"] = available; out["label"] = labels
    out["future_log_return"] = returns; out["trend_band_log"] = band
    out["label_reason"] = np.select([~out.input_eligible.to_numpy(), terminal < 0, ~continuous],
        ["input_unavailable", "horizon_not_mature", "future_quote_calendar_or_price_basis_unavailable"], default="available")
    out.attrs["target_config"] = {k: cfg[k] for k in ("horizon_sessions", "neutral_floor", "atr_multiple")}
    return out


def time_masks(data, split):
    """All four windows are disjoint and labels mature inside their own window."""
    keys = ("train_end", "calibration_start", "calibration_end", "policy_start", "policy_end")
    te, cs, ce, ps, pe = [str(split[k]) for k in keys]
    ts, ze = split.get("test_start"), split.get("test_end")
    no_test = ts is None and ze is None
    if (not te < cs <= ce < ps <= pe
            or (not no_test and (ts is None or ze is None or not pe < str(ts) <= str(ze)))):
        raise ValueError("train/calibration/policy/test windows overlap or are reversed")
    dates = data.date.astype(str).to_numpy(); ends = data.label_end.fillna("").astype(str).to_numpy()
    eligible = data.input_eligible.fillna(False).to_numpy(bool) & data.label_available.fillna(False).to_numpy(bool)
    eligible &= data.label.isin([0, 1, 2]).to_numpy() & (ends != "")
    if np.any(eligible & (ends <= dates)):
        raise ValueError("label maturity must follow the observation")
    return {
        "train": eligible & (dates <= te) & (ends <= te),
        "calibration": eligible & (dates >= cs) & (dates <= ce) & (ends <= ce),
        "policy": eligible & (dates >= ps) & (dates <= pe) & (ends <= pe),
        "test": np.zeros(len(data), dtype=bool) if no_test else eligible & (dates >= str(ts)) & (dates <= str(ze)) & (ends <= str(ze)),
    }


def _torch_device(requested):
    import torch
    return torch, torch.device(select_torch_device(torch, requested))


def _network(torch, architecture, n_features):
    if architecture == "linear":
        return torch.nn.Linear(n_features * 2, 3)
    if architecture == "mlp32x16":
        return torch.nn.Sequential(torch.nn.Linear(n_features*2, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    raise ValueError("unsupported reference architecture")


def _fit_preprocessor(torch, x):
    finite = torch.isfinite(x); clean = torch.where(finite, x, 0.)
    count = finite.sum(dim=0).clamp(min=1)
    mean = clean.sum(dim=0) / count
    centered = torch.where(finite, x - mean, 0.)
    scale = torch.sqrt((centered.square().sum(dim=0) / count).clamp(min=1e-8))
    return mean, scale


def _transform(torch, x, mean, scale):
    finite = torch.isfinite(x)
    values = torch.where(finite, (x-mean)/scale, 0.).clamp(-8., 8.)
    return torch.cat((values, (~finite).to(torch.float32)), dim=1)


def _emit(progress, stage, current, total, **details):
    if progress:
        progress({"stage": stage, "current": current, "total": total, **details})


def _train(torch, device, x_np, y_np, architecture, cfg, epochs, progress, stage):
    torch.manual_seed(cfg["seed"])
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    mean, scale = _fit_preprocessor(torch, x)
    transformed = _transform(torch, x, mean, scale); del x
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    model = _network(torch, architecture, len(FEATURE_COLUMNS)).to(device)
    prior = torch.bincount(y, minlength=3).float().clamp(min=1) / len(y)
    last = model if architecture == "linear" else model[-1]
    with torch.no_grad():
        last.bias.copy_(torch.log(prior))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01 if architecture == "linear" else .003, weight_decay=.01)
    batch = cfg["batch_size"]
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(y), device=device)
        total_loss = torch.zeros((), device=device)
        for start in range(0, len(y), batch):
            ix = order[start:start+batch]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(transformed[ix]), y[ix])
            loss.backward(); optimizer.step()
            total_loss += loss.detach() * len(ix)
        synchronize_torch_device(torch, device)
        _emit(progress, stage, epoch+1, epochs, device=str(device), architecture=architecture,
              loss=float((total_loss / len(y)).item()), samples=len(y))
    model.eval()
    return model, mean, scale


def _logits(torch, model, mean, scale, x_np, device, batch=32768):
    out = []
    with torch.inference_mode():
        for start in range(0, len(x_np), batch):
            x = torch.as_tensor(x_np[start:start+batch], dtype=torch.float32, device=device)
            out.append(model(_transform(torch, x, mean, scale)).detach())
    return torch.cat(out) if out else torch.empty((0, 3), device=device)


def evaluate_probabilities(labels, probabilities, baseline_prior=None, *, device="auto"):
    """Natural-population proper scoring rules and calibration, computed in Torch."""
    torch, dev = _torch_device(device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long, device=dev)
    p = torch.as_tensor(np.asarray(probabilities, dtype=np.float32), device=dev)
    if not len(y):
        return {"status": "no_mature_observations", "count": 0}
    if p.shape != (len(y), 3) or not torch.isfinite(p).all().item() or (p < 0).any().item():
        raise ValueError("invalid reference probabilities")
    if not torch.allclose(p.sum(1), torch.ones(len(p), device=dev), atol=1e-4):
        raise ValueError("reference probabilities do not sum to one")
    p = p.clamp(1e-7, 1.); onehot = torch.nn.functional.one_hot(y, 3).float()
    prediction = p.argmax(1); confusion = torch.bincount(y*3+prediction, minlength=9).reshape(3, 3)
    tp = confusion.diag().float(); denominator = confusion.sum(0)+confusion.sum(1)
    f1 = torch.where(denominator > 0, 2*tp/denominator, 0.)
    confidence = p.max(1).values; bins = (confidence*10).long().clamp(max=9)
    counts = torch.bincount(bins, minlength=10).float()
    confidence_sum = torch.zeros(10, device=dev).scatter_add_(0, bins, confidence)
    accuracy_sum = torch.zeros(10, device=dev).scatter_add_(0, bins, (prediction == y).float())
    ece = (confidence_sum - accuracy_sum).abs().sum()/len(y)
    result = {"status": "evaluated", "count": len(y), "log_loss": float((-p[torch.arange(len(y), device=dev), y].log().mean()).item()),
        "brier": float((p-onehot).square().sum(1).mean().item()), "macro_f1": float(f1.mean().item()),
        "accuracy": float((prediction == y).float().mean().item()), "ece_10": float(ece.item()),
        "class_counts": confusion.sum(1).cpu().tolist(), "confusion": confusion.cpu().tolist(),
        "calibration_bin_counts": counts.cpu().int().tolist(), "device": str(dev)}
    if baseline_prior is not None:
        prior = torch.as_tensor(baseline_prior, dtype=torch.float32, device=dev).clamp(min=1e-7)
        prior = prior/prior.sum()
        result.update(baseline_log_loss=float((-prior[y].log().mean()).item()),
                      baseline_brier=float((prior[None,:]-onehot).square().sum(1).mean().item()),
                      baseline="frozen_train_class_prior")
    return result


def _file_hash(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _save_json(path, value):
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def fit_fold(data, split, output_dir, device="auto", config=None, progress=None):
    """Select using train-internal time only; fit train, calibrate, then evaluate."""
    cfg = _config(config); torch, dev = _torch_device(device); began = time.monotonic()
    masks = time_masks(data, split)
    counts = {key: np.bincount(data.loc[mask, "label"].to_numpy(np.int64), minlength=3).tolist() for key, mask in masks.items()}
    if min(counts["train"]) < cfg["min_train_per_class"]:
        raise ValueError(f"insufficient train class population: {counts['train']}")
    for key in ("calibration", "policy"):
        if min(counts[key]) < cfg["min_segment_per_class"]:
            raise ValueError(f"insufficient {key} class population: {counts[key]}")
    x_all = data[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    y_all = data.label.fillna(-1).to_numpy(np.int64)
    dates = data.date.astype(str).to_numpy(); ends = data.label_end.fillna("").astype(str).to_numpy()
    train_dates = np.unique(dates[masks["train"]]); cut = train_dates[max(0, int(len(train_dates)*.8)-1)]
    inner_train = np.flatnonzero(masks["train"] & (ends <= cut))
    inner_valid = np.flatnonzero(masks["train"] & (dates > cut))
    if len(inner_train) < 30 or len(inner_valid) < 30:
        raise ValueError("insufficient purged train-internal selection window")
    rng = np.random.default_rng(cfg["seed"])
    if len(inner_train) > cfg["inner_sample_cap"]:
        inner_train = np.sort(rng.choice(inner_train, cfg["inner_sample_cap"], replace=False))
    if len(inner_valid) > cfg["inner_sample_cap"]:
        inner_valid = np.sort(rng.choice(inner_valid, cfg["inner_sample_cap"], replace=False))
    candidates = []
    for architecture in ("linear", "mlp32x16"):
        model, mean, scale = _train(torch, dev, x_all[inner_train], y_all[inner_train], architecture,
            cfg, cfg["inner_epochs"], progress, "inner_selection")
        logits = _logits(torch, model, mean, scale, x_all[inner_valid], dev)
        with torch.inference_mode():
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y_all[inner_valid], device=dev)).item()
        candidates.append({"architecture": architecture, "inner_log_loss": float(loss)})
        del model, mean, scale, logits
    architecture = min(candidates, key=lambda x: x["inner_log_loss"])["architecture"]
    train_ix = np.flatnonzero(masks["train"])
    model, mean, scale = _train(torch, dev, x_all[train_ix], y_all[train_ix], architecture,
        cfg, cfg["epochs"], progress, "fit_train")
    logits = _logits(torch, model, mean, scale, x_all[masks["calibration"]], dev)
    cal_y = torch.as_tensor(y_all[masks["calibration"]], device=dev)
    # Fixed positive grid preserves class ordering and cannot add class intercepts.
    temperatures = [0.25, .35, .5, .7, 1., 1.4, 2., 2.8, 4., 5.6, 8.]
    with torch.inference_mode():
        losses = torch.stack([torch.nn.functional.cross_entropy(logits/t, cal_y) for t in temperatures])
        temperature = temperatures[int(losses.argmin().item())]
    del logits, cal_y
    prior = np.asarray(counts["train"], dtype=float); prior /= prior.sum()
    validation = {"population": "historical_daily_liquidity_qualified_price_observations",
        "independent_forward_evidence": False, "class_order": list(CLASSES), "split": dict(split),
        "counts": counts, "candidate_selection": {"source": "purged_train_internal_only", "cutoff": str(cut),
            "inner_train_rows": len(inner_train), "inner_validation_rows": len(inner_valid), "candidates": candidates},
        "calibration_method": "chronological_positive_temperature_no_intercepts", "temperature": temperature}
    for key in ("calibration", "policy", "test"):
        logits = _logits(torch, model, mean, scale, x_all[masks[key]], dev)
        with torch.inference_mode():
            probs = torch.softmax(logits/temperature, dim=1).cpu().numpy()
        validation[key] = evaluate_probabilities(y_all[masks[key]], probs, prior, device=str(dev))
        del logits, probs
    p = validation["policy"]
    validation["policy_predictive_gate"] = p["log_loss"] <= p["baseline_log_loss"] and p["brier"] <= p["baseline_brier"]
    validation["elapsed_seconds"] = round(time.monotonic()-began, 3)
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    weights = destination/"weights.pt"; temporary = destination/"weights.pt.tmp"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "mean": mean.detach().cpu(), "scale": scale.detach().cpu()}, temporary)
    os.replace(temporary, weights)
    manifest = {"family": FAMILY, "schema_hash": SCHEMA_HASH, "feature_columns": list(FEATURE_COLUMNS),
        "classes": list(CLASSES), "architecture": architecture, "config": cfg, "split": dict(split),
        "temperature": temperature, "train_class_prior": prior.tolist(), "preprocessor": "train_mean_std_clip8_plus_missing_mask",
        "weights_file": weights.name, "weights_sha256": _file_hash(weights), "training_device": str(dev),
        "numeric_dtype": "float32", "price_label_dtype": "float64", "validation": validation,
        "actual_trained_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "historical_availability_claim": False}
    manifest["manifest_hash"] = stable_hash(manifest)
    _save_json(destination/"manifest.json", manifest)
    _save_json(destination/"validation.json", validation)
    _emit(progress, "fold_complete", 1, 1, device=str(dev), architecture=architecture, samples=len(train_ix))
    return {"manifest": manifest, "validation": validation, "model_path": str(destination)}


class ModelSession:
    """One device-resident model; reuse across stocks to avoid weight transfers."""
    def __init__(self, model_path, device="auto"):
        self.torch, self.device = _torch_device(device)
        path = Path(model_path); path = path/"manifest.json" if path.is_dir() else path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
        if (manifest.get("manifest_hash") != stable_hash(body) or manifest.get("family") != FAMILY
                or manifest.get("schema_hash") != SCHEMA_HASH or manifest.get("feature_columns") != list(FEATURE_COLUMNS)
                or manifest.get("classes") != list(CLASSES)):
            raise ValueError("reference model schema or manifest hash mismatch")
        weights = path.parent/manifest["weights_file"]
        if weights.parent.resolve() != path.parent.resolve() or _file_hash(weights) != manifest["weights_sha256"]:
            raise ValueError("reference model weights hash/path mismatch")
        state = self.torch.load(weights, map_location="cpu", weights_only=True)
        self.model = _network(self.torch, manifest["architecture"], len(FEATURE_COLUMNS)).to(self.device)
        self.model.load_state_dict(state["state_dict"], strict=True); self.model.eval()
        self.mean = state["mean"].to(self.device, dtype=self.torch.float32)
        self.scale = state["scale"].to(self.device, dtype=self.torch.float32)
        if self.mean.shape != (len(FEATURE_COLUMNS),) or self.scale.shape != self.mean.shape:
            raise ValueError("reference preprocessing shape mismatch")
        if not self.torch.isfinite(self.mean).all().item() or not self.torch.isfinite(self.scale).all().item() or (self.scale <= 0).any().item():
            raise ValueError("invalid reference preprocessing values")
        self.temperature = float(manifest["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid reference calibration temperature")
        self.manifest = manifest

    def predict(self, frame):
        if not set(FEATURE_COLUMNS) <= set(frame) or "input_eligible" not in frame:
            raise ValueError("prepare_inputs must bind the reference input schema and eligibility")
        if frame.attrs.get("schema_hash") not in (None, SCHEMA_HASH):
            raise ValueError("reference inference schema mismatch")
        eligible = frame.input_eligible.fillna(False).to_numpy(bool, copy=True)
        x = frame[list(FEATURE_COLUMNS)].to_numpy(np.float32)
        eligible &= np.isfinite(x[:, :len(MA_TREND_COLUMNS)]).all(1)
        out = np.full((len(frame), 3), np.nan, dtype=np.float32)
        if eligible.any():
            logits = _logits(self.torch, self.model, self.mean, self.scale, x[eligible], self.device)
            with self.torch.inference_mode():
                out[eligible] = self.torch.softmax(logits/self.temperature, 1).cpu().numpy()
        return out


def load_model(model_path, device="auto"):
    return ModelSession(model_path, device)


def predict(model_path, frame, device="auto"):
    return load_model(model_path, device).predict(frame)
