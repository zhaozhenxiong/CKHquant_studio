"""Current pipeline numerical helpers without legacy model routing."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd


def _device(name: str | None):
    import torch
    from my_strategy.core.device import select_torch_device, synchronize_torch_device
    selected = torch.device(select_torch_device(torch, name))
    # Actual float32 work checks the backend beyond hardware discovery.
    torch.ones(1, dtype=torch.float32, device=selected).add_(1)
    synchronize_torch_device(torch, selected)
    return selected


def _network(feature_count: int, hidden_sizes: Sequence[int]):
    import torch
    if len(hidden_sizes) == 0:
        return torch.nn.Sequential(torch.nn.Linear(feature_count, 1))
    if len(hidden_sizes) != 2 or any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
        raise ValueError("MLP requires two positive hidden sizes")
    return torch.nn.Sequential(torch.nn.Linear(feature_count, hidden_sizes[0]), torch.nn.ReLU(),
                               torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]), torch.nn.ReLU(),
                               torch.nn.Linear(hidden_sizes[1], 1))


def _fit_preprocess(values: np.ndarray) -> dict[str, Any]:
    all_missing = np.isnan(values).all(axis=0)
    medians = np.array([0.0 if absent else np.nanmedian(values[:, index]) for index, absent in enumerate(all_missing)])
    filled = np.where(np.isnan(values), medians, values)
    means, scales = filled.mean(axis=0), filled.std(axis=0)
    scales[scales < 1e-8] = 1.0
    return {"median": medians.tolist(), "mean": means.tolist(), "scale": scales.tolist(),
            "all_missing": all_missing.tolist(), "clip_standard_deviations": 8.0, "fit_scope": "purged_training_only"}


def _transform(values: np.ndarray, preprocess: dict[str, Any]) -> np.ndarray:
    filled = np.where(np.isnan(values), np.array(preprocess["median"]), values)
    return np.clip((filled - np.array(preprocess["mean"])) / np.array(preprocess["scale"]), -8, 8).astype(np.float32)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    p = np.clip(probabilities.astype(float), 1e-7, 1 - 1e-7)
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    ranks = pd.Series(p).rank(method="average").to_numpy()
    auc = float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None
    bins = []
    for lower in np.arange(0.0, 1.0, 0.1):
        mask = (p >= lower) & (p < lower + 0.1 + (1e-8 if lower > 0.89 else 0))
        if mask.any():
            bins.append({"lower": round(float(lower), 1), "rows": int(mask.sum()), "predicted": float(p[mask].mean()), "observed": float(labels[mask].mean())})
    return {"rows": len(labels), "positive_rate": float(labels.mean()), "brier": float(np.mean((p - labels) ** 2)),
            "log_loss": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))), "auc": auc, "calibration_bins": bins}
