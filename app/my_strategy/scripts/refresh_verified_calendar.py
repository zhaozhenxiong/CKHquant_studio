"""Append a verified 2026 schedule without changing frozen training calendars."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import requests

from my_strategy.adapters.czsc_adapter import latest_market_date
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash


OFFICIAL_URL = "https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml"
HOLIDAYS = (
    ("2026-01-01", "2026-01-03", "1月1日（星期四）至1月3日（星期六）休市"),
    ("2026-02-15", "2026-02-23", "2月15日（星期日）至2月23日（星期一）休市"),
    ("2026-04-04", "2026-04-06", "4月4日（星期六）至4月6日（星期一）休市"),
    ("2026-05-01", "2026-05-05", "5月1日（星期五）至5月5日（星期二）休市"),
    ("2026-06-19", "2026-06-21", "6月19日（星期五）至6月21日（星期日）休市"),
    ("2026-09-25", "2026-09-27", "9月25日（星期五）至9月27日（星期日）休市"),
    ("2026-10-01", "2026-10-07", "10月1日（星期四）至10月7日（星期三）休市"),
)


def refresh_calendar(base_run_id: str, audit_session: str | None = None) -> dict:
    from bs4 import BeautifulSoup
    from akshare.tool.trade_date_hist import hk_js_decode, py_mini_racer

    if Path(base_run_id).name != base_run_id or base_run_id in {".", ".."}:
        raise ValueError("invalid base calendar run")
    root = ARTIFACT_RUNS_ROOT / base_run_id
    report_path = root / "reports/calendar.json"
    base = json.loads(report_path.read_text(encoding="utf-8"))
    if base.get("verified") is not True or base.get("independent_from_market_bars") is not True:
        raise ValueError("base calendar is not independently verified")
    originals = {}
    for item in base["source_artifacts"]:
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"calendar source hash changed: {path.name}")
        originals[path.name] = path
    installed = json.loads(originals["installed_akshare_calendar.json"].read_text())
    installed_dates = sorted({pd.Timestamp(item).date().isoformat() for item in installed})
    decoder = py_mini_racer.MiniRacer()
    try:
        decoder.eval(hk_js_decode)
        source = originals["sina_klc_td_sh.txt"].read_text()
        decoded = decoder.call("d", source.split("=", 1)[1].split(";", 1)[0].replace('"', ""))
    finally:
        decoder.close()
    dates = sorted({pd.Timestamp(item).date().isoformat() for item in decoded
                    if "2015-01-01" <= pd.Timestamp(item).date().isoformat() <= "2026-12-31"})
    if dates != [item for item in installed_dates if "2015-01-01" <= item <= "2026-12-31"]:
        raise ValueError("Sina and installed independent calendar do not match")
    if base["dates"] != dates[:len(base["dates"])]:
        raise ValueError("calendar history is revised; append-only extension rejected")
    response = requests.get(OFFICIAL_URL, timeout=30)
    response.raise_for_status()
    html = response.content
    official_text = BeautifulSoup(html, "html.parser").get_text("", strip=True)
    if not all(fragment in official_text for _, _, fragment in HOLIDAYS):
        raise ValueError("official 2026 holiday notice has changed or could not be verified")
    excluded = set()
    for first, last, _ in HOLIDAYS:
        day = date.fromisoformat(first)
        while day <= date.fromisoformat(last):
            excluded.add(day.isoformat())
            day += timedelta(days=1)
    year_days = pd.date_range("2026-01-01", "2026-12-31")
    official_dates = [day.date().isoformat() for day in year_days
                      if day.weekday() < 5 and day.date().isoformat() not in excluded]
    if official_dates != [day for day in dates if day.startswith("2026-")]:
        raise ValueError("calendar does not exactly match official 2026 weekday/holiday schedule")
    observed_end = latest_market_date()
    if not observed_end or observed_end not in dates:
        raise ValueError("latest raw market date is outside verified schedule")
    context = create_run_context(task="czsc-verified-calendar", as_of_date=observed_end,
        scope="implementation", source="cli", start_date="2015-01-01", end_date=dates[-1],
        config={"base_run_id": base_run_id, "schedule_end": dates[-1], "official_url": OFFICIAL_URL},
        data_version=stable_hash(dates))
    copied = []
    for name in ("sina_klc_td_sh.txt", "installed_akshare_calendar.json"):
        target = context.subdir("data_snapshots") / name
        shutil.copyfile(originals[name], target)
        copied.append({"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    official_path = context.subdir("data_snapshots") / "sse_2026_closed_days.html"
    official_path.write_bytes(html)
    copied.append({"path": str(official_path), "sha256": hashlib.sha256(html).hexdigest()})
    artifact = context.subdir("data_snapshots") / "verified_dates.parquet"
    pd.DataFrame({"date": dates}).to_parquet(artifact, index=False)
    copied.append({"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
    value = {**base, "start": "2015-01-01", "end": dates[-1], "dates": dates,
        "artifact": str(artifact), "sha256": copied[-1]["sha256"], "source_artifacts": copied,
        "observed_market_data_end": observed_end, "executable_verified_range_end": observed_end,
        "schedule_verified_range_end": dates[-1], "future_source_dates_excluded": 0,
        "future_dates_scope": "preannounced_trading_schedule_only_no_future_prices",
        "parent_calendar": {"run_id": base_run_id, "path": str(report_path),
            "file_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(), "hash": stable_hash(base)},
        "verification_details": {"trading_dates": len(dates), "sorted_unique": True,
            "installed_akshare_exact_match": True, "sse_2026_exact_match": True,
            "sse_2026_sessions_to_end": len(official_dates), "parent_prefix_exact_match": True},
        "audit_session": audit_session}
    (context.subdir("reports") / "calendar.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
    context.write_metadata({"status": "completed", "calendar": value, "artifacts": copied})
    return {"run_id": context.run_id, "verified": True, "data_end": observed_end,
            "calendar_end": dates[-1], "next_market_session": next(day for day in dates if day > observed_end),
            "parent_run_id": base_run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--audit-session")
    args = parser.parse_args()
    print(json.dumps(refresh_calendar(args.base_run_id, args.audit_session), ensure_ascii=False))


if __name__ == "__main__":
    main()
