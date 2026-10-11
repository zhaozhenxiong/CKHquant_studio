"""Optional immutable three-stock smoke audit; never reads a business database.

Run with KHQUANT_REAL_EPISODE_EVIDENCE=/absolute/path/smoke.json to persist the
inspection. A checkout without the frozen local source snapshot skips this test.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.cost_model import CostModel
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_high_return_features import (
    ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS, L1_MA_ENTRY_COLUMNS, L1_MA_EXIT_COLUMNS,
)
from my_strategy.services.czsc_trade_episodes import build_episode_dataset, training_frame


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.slow
def test_three_immutable_mainboard_inputs_have_real_episodes_and_bound_contracts():
    app_root = Path(__file__).resolve().parents[1]
    config_path = app_root / "configs/czsc_high_return_research.json"
    cfg = json.loads(config_path.read_text())
    default = app_root / "artifacts/runs" / cfg["source_training_run_id"] / "reports/czsc.json"
    source_path = Path(os.environ.get("KHQUANT_REAL_EPISODE_SOURCE", default))
    if not source_path.is_file():
        pytest.skip("immutable local full Wyckoff source not present")
    source_sha = _sha(source_path)
    report = json.loads(source_path.read_text())
    calendar_path = Path(report["calendar"]["path"])
    calendar_value = json.loads(calendar_path.read_text())
    assert _sha(calendar_path) == report["calendar"]["file_sha256"]
    assert calendar_value.get("verified") and calendar_value.get("source")
    dates = calendar_value["dates"]
    assert stable_hash(calendar_value) == report["calendar"]["hash"]
    assert dates == sorted(set(dates))
    complete_config = {**strategy_config(), **cfg}
    cost = CostModel(**{k: float(complete_config["execution"][k]) for k in
        ("commission", "stamp_tax", "min_commission", "slippage")}, version="flat_v1", impact_tiers=())
    records = [record for record in report["dataset_records"] if
        record["symbol"].endswith(".SZ") and record["symbol"].startswith("00")
        or record["symbol"].endswith(".SH") and record["symbol"].startswith("60")][:3]
    assert len(records) == 3
    started = time.monotonic(); summaries = []; totals = Counter()
    for record in records:
        path = Path(record["path"])
        assert _sha(path) == record["sha256"]
        features = pd.read_parquet(path)
        # Existing features remain untouched. Raw execution keeps the already
        # verified pre-start bars needed by the shared seasoning contract.
        first = int(np.flatnonzero((pd.to_datetime(features.date) >= pd.Timestamp("2023-01-01")).to_numpy())[0])
        prefix = features.iloc[:first]
        valid = np.flatnonzero(((prefix.volume > 0) & (prefix.amount > 0) & (prefix.has_trade_price == 1)).to_numpy())
        count = int(complete_config["execution"]["seasoned_bars"]) + 1
        begin = int(valid[-count]) if len(valid) >= count else 0
        features = features.iloc[begin:].copy().reset_index(drop=True)
        assert features.symbol.nunique() == 1 and features.symbol.iloc[0] == record["symbol"]
        assert not features.date.duplicated().any() and features.date.is_monotonic_increasing
        dataset = build_episode_dataset(features, dates, complete_config,
            start="2023-01-01", source_snapshot_hash=source_sha)
        entry, exit, episodes, events = (dataset[k] for k in ("entries", "exits", "episodes", "events"))
        assert len(entry) > 0 and len(exit) > 0 and len(episodes) > 0
        assert set(entry.source_snapshot_hash) == {source_sha}
        assert set(episodes.source_snapshot_hash) == {source_sha}
        assert entry.opportunity_id.notna().all() and entry.economic_group_id.notna().all()
        assert entry.reference_path_hash.str.len().eq(64).all()
        assert entry.candidate_path_hash.str.len().eq(64).all()
        event_ids = set(events.event_id)
        assert all(set(ids) <= event_ids for ids in entry.source_event_ids)
        assert all(set(ids) <= event_ids for ids in episodes.source_event_ids)
        assert set(exit.episode_id) <= set(episodes.episode_id)
        mature = entry.loc[entry.label_available]
        assert (pd.to_datetime(mature.label_end) > pd.to_datetime(mature.date)).all()
        assert np.allclose(mature.net_return, mature.net_profit / mature.entry_cash)
        assert np.array_equal(mature.label.to_numpy(bool),
            ((mature.net_return >= .08) & (mature.r_multiple >= 2)).to_numpy())
        mature_exit = exit.loc[exit.label_available]
        assert all(mature_exit.label_end == mature_exit[["forced_exit_date", "continuation_exit_date"]].max(axis=1))
        assert np.allclose(mature_exit.exit_advantage,
            (mature_exit.forced_net_proceeds - mature_exit.continuation_net_proceeds) /
            (mature_exit.actual_shares * pd.to_numeric(features.set_index("date").close.reindex(mature_exit.date).to_numpy())))
        account_audits = []
        for key, account in dataset["reference_accounts"].items():
            ledger = account["ledger"]
            last_date = None
            for fill in ledger:
                assert fill["date"] in dates
                assert pd.Timestamp(fill["signal_date"]) < pd.Timestamp(fill["date"])
                assert last_date is None or fill["date"] >= last_date
                last_date = fill["date"]
                assert fill["shares"] > 0 and fill["shares"] % 100 == 0
                assert fill["fee"] == pytest.approx(cost.fees("BUY" if fill["action"] == "BUY" else "SELL", fill["shares"] * fill["price"]))
            assert account["metrics"]["fees"] == pytest.approx(sum(fill["fee"] for fill in ledger))
            assert account["daily"][-1]["cash"] == pytest.approx(100000 + sum(fill["cash_flow"] for fill in ledger))
            assert all(row["ledger_entries_seen"] <= len(ledger) and row["shares"] > 0 for row in account["holding_snapshots"])
            assert not any(row["reason"] == "entry_plan_not_active" for row in account["rejections"])
            account_audits.append({"key": key, "ledger_rows": len(ledger), "fees": account["metrics"]["fees"],
                "rejections": len(account["rejections"]), "open_shares": account["metrics"]["open_shares"]})
        contracts = []
        for source in ("ma", "structure", "wyckoff", "union"):
            for head, columns, ma_only in (("entry", ENTRY_FEATURE_COLUMNS, False),
                ("exit", EXIT_FEATURE_COLUMNS, False), ("entry", L1_MA_ENTRY_COLUMNS, True),
                ("exit", L1_MA_EXIT_COLUMNS, True)):
                train = training_frame(dataset, source=source, head=head, ma_only=ma_only)
                assert set(columns) <= set(train) or train.empty
                assert train.attrs["feature_columns"] == list(columns)
                contract = train.attrs["label_contract"]
                assert contract["contract_hash"] == stable_hash({k: v for k, v in contract.items() if k != "contract_hash"})
                assert contract["source_snapshot_hash"] == source_sha
                assert contract["calendar_hash"] == stable_hash(dates)
                assert contract["source"] == source and contract["feature_schema_hash"] == train.attrs["schema_hash"]
                contracts.append({"source": source, "head": head, "ma_only": ma_only,
                    "rows": len(train), "mature": int(train.label_available.sum()) if len(train) else 0,
                    "columns": len(columns), "contract_hash": contract["contract_hash"]})
        summary = {"symbol": record["symbol"], "raw_rows": len(features), "source_sha256": record["sha256"],
            "events": len(events), "entry_candidates": len(entry), "mature_entries": len(mature),
            "entry_reasons": entry.label_reason.value_counts().to_dict(),
            "entry_classes": entry.outcome_class.fillna("unavailable").value_counts().to_dict(),
            "exit_states": len(exit), "mature_exits": len(mature_exit),
            "actual_episodes": len(episodes), "closed": int(episodes.exit_date.notna().sum()),
            "censored": int(episodes.exit_date.isna().sum()),
            "actual_classes": episodes.outcome_class.fillna("unavailable").value_counts().to_dict(),
            "sources": {s: {"candidates": int((entry.source == s).sum()),
                "candidate_classes": entry.loc[entry.source == s].outcome_class.fillna("unavailable").value_counts().to_dict(),
                "actual_classes": episodes.loc[episodes.source == s].outcome_class.fillna("unavailable").value_counts().to_dict()}
                for s in ("ma", "structure", "wyckoff", "union")},
            "accounts": account_audits, "contracts": contracts}
        summaries.append(summary)
        totals.update({k: summary[k] for k in ("raw_rows", "events", "entry_candidates", "mature_entries",
            "exit_states", "mature_exits", "actual_episodes", "closed", "censored")})
        totals.update({"actual_" + k: int(summary["actual_classes"].get(k, 0)) for k in ("high", "loss", "ordinary")})
    assert _sha(source_path) == source_sha
    evidence = {"status": "passed", "scope": "three_mainboard_frozen_stocks_engineering_only",
        "source_report": str(source_path), "source_sha256": source_sha,
        "config_sha256": _sha(config_path), "calendar_sha256": _sha(calendar_path),
        "seconds": time.monotonic() - started, "totals": dict(totals), "stocks": summaries,
        "training_or_publication_performed": False,
        "limitations": ["Unadjusted prices and conservative 5% guards; no total-return certification.",
            "Rules bootstrap v0; this is an engineering smoke, not an OOF profitability claim."]}
    output = os.environ.get("KHQUANT_REAL_EPISODE_EVIDENCE")
    if output:
        Path(output).write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(json.dumps({"totals": evidence["totals"], "seconds": evidence["seconds"]}, ensure_ascii=False))
