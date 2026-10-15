"""Bounded, offline Kronos forecasting with explicit samples and causal inputs."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.runtime_env import runtime_resource_root
from my_strategy.services.kronos_assets import asset_root as resolve_asset_root, asset_status

SCHEMA = "kronos_forecast_v1"
FEATURES = ("open", "high", "low", "close", "volume", "amount")
_MODEL_LOCK = threading.RLock()
_SINGLETON: dict[str, Any] = {}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _date(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Invalid date")
    if result.tzinfo is not None:
        result = result.tz_convert("Asia/Shanghai").tz_localize(None)
    return result.normalize()


def _source_identity() -> dict:
    root = runtime_resource_root() / "vendor" / "kronos"
    manifest = json.loads((root / "source_manifest.json").read_text())
    observed = []
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Kronos source escapes resource root")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Kronos vendor source identity mismatch: {entry['path']}")
        observed.append({"path": entry["path"], "sha256": actual})
    return {"vendor_sha256": _hash(observed),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device is unavailable")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if requested != "cpu" and requested != "mps" and not requested.startswith("cuda"):
        raise ValueError("device must be auto, cpu, mps or cuda")
    return str(torch.device(requested))


def prepare_input(frame: pd.DataFrame, future_dates, *, symbol: str, as_of: str,
                  lookback: int = 128, horizon: int = 5) -> tuple[pd.DataFrame, list[str], dict]:
    """Clip before any statistics/quality checks, then require the actual target close."""
    if not 16 <= int(lookback) <= 512 or not 1 <= int(horizon) <= 10:
        raise ValueError("lookback must be 16–512 and horizon 1–10")
    if not isinstance(frame, pd.DataFrame) or not set(FEATURES).issubset(frame.columns):
        raise ValueError("Actual OHLC, volume and amount columns are required")
    column = next((key for key in ("date", "timestamps", "dt") if key in frame), None)
    if column is None and not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("A date/timestamps/dt column or DatetimeIndex is required")
    dates = pd.DatetimeIndex([_date(x) for x in (frame[column] if column else frame.index)])
    cutoff = _date(as_of)
    instant = pd.Timestamp(as_of)
    if instant.tzinfo is not None:
        instant = instant.tz_convert("Asia/Shanghai").tz_localize(None)
    if ("T" in str(as_of) or " " in str(as_of)) and instant < cutoff + pd.Timedelta(hours=15):
        raise ValueError("Daily close forecast requires an as_of at or after 15:00 Asia/Shanghai")
    current = frame.loc[dates <= cutoff].copy()
    current.index = dates[dates <= cutoff]
    if current.empty or current.index[-1] != cutoff:
        raise ValueError("No actual daily close on the requested as_of; stale prices are not substituted")
    if current.index.has_duplicates or not current.index.is_monotonic_increasing:
        raise ValueError("Historical dates must be unique and increasing")
    if len(current) < lookback:
        raise ValueError(f"Insufficient history: {len(current)} < {lookback}")
    current = current.tail(lookback)
    if "symbol" in current and set(current["symbol"].astype(str)) != {symbol}:
        raise ValueError("Input contains a different symbol")
    values = current[list(FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise ValueError("Actual prices/volumes must be finite and legal")
    if ((values[:, 1] < values[:, [0, 2, 3]].max(axis=1)) |
            (values[:, 2] > values[:, [0, 1, 3]].min(axis=1))).any():
        raise ValueError("Actual OHLC geometry is invalid")
    future = [_date(value) for value in future_dates]
    if len(future) != horizon or len(set(future)) != horizon or future != sorted(future) or future[0] <= cutoff:
        raise ValueError("future_dates must contain exactly horizon unique ascending future trading dates")
    attrs = frame.attrs
    description = {"symbol": symbol, "as_of": cutoff.date().isoformat(), "lookback": lookback,
                   "horizon": horizon, "history_start": current.index[0].date().isoformat(),
                   "history_end": current.index[-1].date().isoformat(), "rows": len(current),
                   "price_basis": attrs.get("price_basis", "unverified"),
                   "volume_unit": attrs.get("volume_unit", "unverified"),
                   "amount_unit": attrs.get("amount_unit", "unverified"),
                   "last_close": float(values[-1, 3]), "timezone": "Asia/Shanghai",
                   "source": sorted(set(current["source"].astype(str))) if "source" in current else [],
                   "unverified_trade_price_bars": int((current["has_trade_price"] != 1).sum()) if "has_trade_price" in current else None}
    description["data_sha256"] = _hash({"dates": [d.date().isoformat() for d in current.index],
                                        "values": values.tolist(), "contract": description})
    return current, [value.date().isoformat() for value in future], description


def _load_model(root: Path, device: str, identity: dict):
    import torch
    from safetensors.torch import load_file
    key = _hash({"assets": identity["assets"], "source": identity["source"], "device": device, "root": str(root)})
    if _SINGLETON.get("key") == key:
        return _SINGLETON["model"], _SINGLETON["tokenizer"], _SINGLETON["inference"], True
    release_prediction_resources()
    package_name = "_khquant_kronos_vendor"
    source = runtime_resource_root() / "vendor" / "kronos"
    spec = importlib.util.spec_from_file_location(package_name, source / "__init__.py",
                                                 submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec)
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Initializers do not disturb the application's independent CPU random stream.
    with torch.random.fork_rng(devices=[]):
        model = module.Kronos(**json.loads((root / "config.json").read_text()))
        tokenizer = module.KronosTokenizer(**json.loads((root / "tokenizer/config.json").read_text()))
    model.load_state_dict(load_file(str(root / "model.safetensors"), device="cpu"), strict=True)
    tokenizer.load_state_dict(load_file(str(root / "tokenizer/model.safetensors"), device="cpu"), strict=True)
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    tokenizer.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    _SINGLETON.update(key=key, model=model, tokenizer=tokenizer, inference=module.auto_regressive_inference)
    return model, tokenizer, module.auto_regressive_inference, False


@contextmanager
def _seeded(seed: int, device: str):
    import torch
    cpu = torch.get_rng_state()
    mps = torch.mps.get_rng_state() if device == "mps" else None
    cuda = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    try:
        torch.random.default_generator.manual_seed(seed)
        if mps is not None:
            torch.mps.manual_seed(seed)
        if cuda is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(cpu)
        if mps is not None:
            torch.mps.set_rng_state(mps)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def summarize_samples(paths: np.ndarray, future_dates: list[str], last_close: float) -> dict:
    """Filter entire illegal paths using a fixed direction-neutral rule, never repair."""
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != 6 or paths.shape[1] != len(future_dates):
        raise ValueError("Expected sample × horizon × 6 OHLCVA paths")
    finite = np.isfinite(paths).all(axis=(1, 2))
    price_ok = (paths[:, :, :4] > 0).all(axis=(1, 2))
    activity_ok = (paths[:, :, 4:] >= 0).all(axis=(1, 2))
    geometry = ((paths[:, :, 1] >= paths[:, :, [0, 2, 3]].max(axis=2)) &
                (paths[:, :, 2] <= paths[:, :, [0, 1, 3]].min(axis=2))).all(axis=1)
    valid = finite & price_ok & activity_ok & geometry
    count = int(valid.sum())
    usable = count >= 8 and count / len(paths) >= .8
    quality = {"requested_samples": len(paths), "valid_samples": count,
               "invalid_samples": int((~valid).sum()), "valid_fraction": count / len(paths),
               "minimum_valid_samples": 8, "minimum_valid_fraction": .8,
               "prediction_usable": usable, "valid_mask": valid.tolist(),
               "rejected_by_rule": {"nonfinite": int((~finite).sum()), "price": int((~price_ok).sum()),
                                    "negative_activity": int((~activity_ok).sum()), "ohlc_geometry": int((~geometry).sum())}}
    finite_terminal = paths[finite, -1, 3] / last_close - 1
    quality["unfiltered_finite_terminal_return_mean"] = float(finite_terminal.mean()) if len(finite_terminal) else None
    quality["filtered_terminal_return_mean"] = float((paths[valid, -1, 3] / last_close - 1).mean()) if count else None
    if not count:
        return {"quality": quality, "forecast": [], "summary": None}
    data = paths[valid]
    q = np.quantile(data, [.1, .5, .9], axis=0)
    forecast = [{"date": date, **{key: float(q[1, i, j]) for j, key in enumerate(FEATURES)},
                 "close_p10": float(q[0, i, 3]), "close_p90": float(q[2, i, 3]),
                 "close_mean": float(data[:, i, 3].mean())} for i, date in enumerate(future_dates)]
    terminal = data[:, -1, 3] / last_close - 1
    summary = {"terminal_return_p10": float(np.quantile(terminal, .1)),
               "terminal_return_median": float(np.median(terminal)),
               "terminal_return_p90": float(np.quantile(terminal, .9)),
               "up_path_fraction": float((terminal > 0).mean()),
               "terminal_return_std": float(terminal.std()),
               "worst_path_low_return": float(data[:, :, 2].min() / last_close - 1),
               "probability_semantics": "uncalibrated_model_sample_frequency",
               "interval_semantics": "empirical_sample_quantiles_not_calibrated_confidence",
               "return_basis": "as_of_close_to_forecast_close_not_executable_profit"}
    return {"quality": quality, "forecast": forecast, "summary": summary}


def predict(frame: pd.DataFrame, future_dates, *, symbol: str, as_of: str,
            lookback: int = 128, horizon: int = 5, samples: int = 8, seed: int = 42,
            device: str = "auto", temperature: float = .8, top_p: float = .9,
            calendar_identity: str | dict | None = None, calendar_verified: bool = False,
            asset_root: str | Path | None = None, cancel_check=None) -> dict:
    """Forecast only; no automatic model promotion, orders, run directories or downloads."""
    import torch
    started = time.monotonic()
    if not 1 <= samples <= 64 or not isinstance(samples, int):
        raise ValueError("samples must be an integer in 1–64")
    if not 0 < temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("temperature must be (0, 2], top_p (0, 1]")
    if not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    if calendar_verified and not calendar_identity:
        raise ValueError("Verified calendar requires an identity")
    current, dates, input_info = prepare_input(frame, future_dates, symbol=symbol, as_of=as_of,
                                              lookback=lookback, horizon=horizon)
    root = resolve_asset_root(asset_root)
    assets = asset_status(root)
    if not assets["ready"]:
        raise FileNotFoundError("Kronos local assets unavailable: " + json.dumps(assets, ensure_ascii=False))
    actual_device = _device(device)
    params = {"lookback": lookback, "horizon": horizon, "samples": samples, "seed": seed,
              "temperature": temperature, "top_p": top_p, "top_k": 0, "clip": 5.0,
              "device": actual_device, "torch_version": torch.__version__, "dtype": "float32"}
    identity = {"schema": SCHEMA, "assets": assets["files"], "source": _source_identity(),
                "input": input_info, "future_dates": dates, "parameters": params,
                "calendar": {"verified": bool(calendar_verified), "identity": calendar_identity}}
    cache_key = _hash(identity)
    def check_cancel():
        if cancel_check and cancel_check():
            raise InterruptedError("Kronos prediction cancelled")
    check_cancel()
    values = current[list(FEATURES)].to_numpy(dtype=np.float32)
    mean, std = values.mean(axis=0), values.std(axis=0)
    normalized = np.clip((values - mean) / (std + 1e-5), -5, 5)
    def stamps(index):
        ts = pd.DatetimeIndex(index) + pd.Timedelta(hours=15)
        return np.column_stack([ts.minute, ts.hour, ts.weekday, ts.day, ts.month]).astype(np.float32)
    with _MODEL_LOCK:
        model, tokenizer, inference, reused = _load_model(root, actual_device, identity)
        check_cancel()
        with _seeded(seed, actual_device), torch.inference_mode():
            tensor = lambda value: torch.from_numpy(value[None, ...]).to(actual_device)
            raw = inference(tokenizer, model, tensor(normalized), tensor(stamps(current.index)),
                            tensor(stamps(pd.to_datetime(dates))), max_context=512, pred_len=horizon,
                            clip=5, T=temperature, top_k=0, top_p=top_p, sample_count=samples,
                            verbose=False, cancel_check=check_cancel)
            paths = raw[0, :, -horizon:, :] * (std[None, None, :] + 1e-5) + mean[None, None, :]
        allocated = int(torch.mps.current_allocated_memory()) if actual_device == "mps" else None
    result = summarize_samples(paths, dates, input_info["last_close"])
    verified_sources = {"sina_unadjusted_v1", "fuyao_unadjusted_v1"}
    source_ok = bool(input_info["source"]) and set(input_info["source"]).issubset(verified_sources)
    contract_ok = (input_info["volume_unit"] == "shares" and input_info["amount_unit"] == "CNY"
                   and input_info["price_basis"] == "unadjusted"
                   and input_info["unverified_trade_price_bars"] == 0 and source_ok)
    result["quality"].update(calendar_verified=bool(calendar_verified), input_contract_verified=contract_ok,
                             source_verified=source_ok)
    result["quality"]["prediction_usable"] &= bool(calendar_verified) and contract_ok
    result.update(schema=SCHEMA, status="completed", symbol=symbol, as_of=input_info["as_of"],
                  cache_key=cache_key, identity=identity, input=input_info,
                  samples=[[[float(v) if np.isfinite(v) else None for v in bar] for bar in path] for path in paths],
                  qualification={"research_only": True, "independent_forward_status": "pending",
                                 "positive_allocation_eligible": False, "allow_increase": False,
                                 "reason": "Forecast availability does not certify a trading policy"},
                  resource={"actual_device": actual_device, "model_reused": reused,
                            "elapsed_seconds": time.monotonic() - started,
                            "mps_allocated_bytes": allocated, "model_training_started": False})
    return result


def release_prediction_resources() -> None:
    """Drop the sole model pair after a cancelled task or an explicit idle release."""
    with _MODEL_LOCK:
        _SINGLETON.clear()
        gc.collect()
        if "torch" in sys.modules:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
