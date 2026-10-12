"""Compare completed actual Broker validation reports of both native risk arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_high_return_runtime_v4 import read_run


def _read_report(path: str) -> dict:
    value = json.loads(Path(path).read_text())
    if value.get("status") not in {"completed", "completed_with_failures_retained"}:
        raise ValueError("comparison requires a completed actual Broker report")
    return value


def compare_reports(main: dict, control: dict, *, minimum_round_trips: int = 50) -> list[dict]:
    if main["config"]["cohort_hash"] != control["config"]["cohort_hash"]:
        raise ValueError("different cohorts cannot be called paired risk controls")
    for key in ("windows", "evidence_bars", "slippage_multipliers", "initial_cash_per_account", "calendar_hash", "raw_end"):
        if main["config"][key] != control["config"][key]:
            raise ValueError("risk arms disagree on " + key)
    keyed = {}
    for arm, report in (("main", main), ("risk_control", control)):
        for row in report["aggregate"]:
            key = (row["window"], row["evidence_bars"], row["slippage_multiplier"])
            named = "v3" if row["variant"] == "v3" else arm
            if named in keyed.setdefault(key, {}):
                raise ValueError("duplicated actual account aggregate")
            keyed[key][named] = {k: v for k, v in row.items() if k != "daily"}
    result = []
    for key, arms in sorted(keyed.items()):
        if set(arms) != {"main", "risk_control", "v3"}:
            raise ValueError("paired report is missing a main, risk control or frozen v3 arm")
        if any(row["retained_accounts"] != row["requested_accounts"] for row in arms.values()):
            raise ValueError("a failed or selected account was dropped from the denominator")
        actual = arms["main"]
        result.append({"window": key[0], "evidence_bars": key[1], "slippage_multiplier": key[2],
                       "arms": arms,
                       "main_return_minus_frozen_v3": actual["total_return"] - arms["v3"]["total_return"],
                       "main_return_minus_risk_control": actual["total_return"] - arms["risk_control"]["total_return"],
                       "main_drawdown_minus_frozen_v3": actual["max_drawdown"] - arms["v3"]["max_drawdown"],
                       "minimum_development_round_trips": minimum_round_trips,
                       "main_sample_minimum_met": actual["completed_round_trips"] >= minimum_round_trips,
                       "failed_cash_accounts_retained": {name: value["failed_cash_accounts"] for name, value in arms.items()}})
    return result


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-report", required=True)
    parser.add_argument("--risk-control-report", required=True)
    args = parser.parse_args(argv)
    baseline = _read_report(args.main_report); control = _read_report(args.risk_control_report)
    _, training = read_run(baseline["config"]["model_run_id"])
    config = {"version": "native_v4_actual_broker_paired_risk_comparison_20261012",
              "main_validation_run": baseline["run_id"], "risk_control_validation_run": control["run_id"],
              "main_report_sha256": hashlib.sha256(Path(args.main_report).read_bytes()).hexdigest(),
              "risk_control_report_sha256": hashlib.sha256(Path(args.risk_control_report).read_bytes()).hexdigest(),
              "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "publication_allowed": False, "development_diagnostic_only": True}
    context = create_run_context(task="czsc-native-v4-paired-broker-comparison", as_of_date=baseline["config"]["raw_end"],
                                 config=config, data_version=stable_hash(config), stocks=baseline["config"]["symbols"])
    comparison = compare_reports(baseline, control,
                                 minimum_round_trips=int(training["config"]["development_min_round_trips_per_window"]))
    default_quarters = [row for row in comparison if row["evidence_bars"] == 3 and row["slippage_multiplier"] == 1
                        and row["window"] != "2026YTD"]
    sample_count = sum(row["arms"]["main"]["completed_round_trips"] for row in default_quarters)
    total_minimum = int(training["config"]["development_min_round_trips_total"])
    report = {"run_id": context.run_id, "config": config, "comparisons": comparison,
              "actual_accounts": baseline["actual_account_count"] + control["actual_account_count"],
              "failed_cash_accounts": baseline["failed_cash_accounts"] + control["failed_cash_accounts"],
              "default_age3_normal_development_samples": {
                  "completed_quarter_round_trips_excluding_overlapping_YTD": sample_count,
                  "minimum_total_round_trips": total_minimum,
                  "all_quarter_sample_minima_met": all(row["main_sample_minimum_met"] for row in default_quarters),
                  "total_sample_minimum_met": sample_count >= total_minimum,
                  "windows_without_positive_return_delta_against_v3": [row["window"] for row in default_quarters
                                                                        if row["main_return_minus_frozen_v3"] <= 0]},
              "independent_window_status": "pending", "publication_allowed": False,
              "interpretation": ["returns reconstruct actual Broker equity of fixed equal-cash selected accounts",
                                 "YTD overlaps two quarterly windows and cannot be added to them as independent evidence",
                                 "chart B gates entry; frozen raw union source exits and signal-pinned MA exit retain label continuation",
                                 "architecture and threshold diagnostics are separate from this frozen main replay",
                                 "read historical windows and sample minima do not grant independent production acceptance"]}
    path = context.subdir("reports") / "paired_broker.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 原生 v4 两风险真实账本对照", "", "均为固定等额独立现金账户，含所有冻结股票；历史窗口为开发诊断，独立验收仍 pending。", "",
             "默认 age3、正常滑点；下表为账户总资金的费用后收益。YTD 与季度有重叠。", "",
             "| 窗口 | 主风险 v4 | 旧风险 v4 对照 | 原冻结 v3 | 主风险回撤 | 主风险完整往返 |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison:
        if row["evidence_bars"] != 3 or row["slippage_multiplier"] != 1:
            continue
        a = row["arms"]
        lines.append(f"| {row['window']} | {a['main']['total_return']:.2%} | {a['risk_control']['total_return']:.2%} | {a['v3']['total_return']:.2%} | {a['main']['max_drawdown']:.2%} | {a['main']['completed_round_trips']} |")
    lines.extend(["", f"实际账户 {report['actual_accounts']}，保留现金失败账户 {report['failed_cash_accounts']}。", "",
                  "全部 1/3/5 有效期和正常/双滑点结果、拒单、费用、库存、原始退出与图表方向差别保存在 JSON 和各账户账本中。"])
    markdown = context.subdir("reports") / "paired_broker.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    context.write_metadata({"status": "completed", "config": config, "publication_allowed": False})
    print(json.dumps({"run_id": context.run_id, "report": str(path), "markdown": str(markdown)}), flush=True)
    return report


if __name__ == "__main__":
    main()
