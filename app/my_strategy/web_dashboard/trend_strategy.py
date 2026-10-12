"""Analysis, screening and explicit backtests share the delivered account engine."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import multiprocessing

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, DISPLAY_NAME
from my_strategy.services.czsc_trend_runtime import PRICING_ASSUMPTIONS


def _source_vintage(symbol, dates, *, db_path=None, vintage_end=None):
    """Same source identity for analysis, scanning and explicit account replay.

    Full vintage metadata identifies revisions; runtime cuts the observed prefix
    before every model, native rule and account input.
    """
    from my_strategy.adapters.czsc_adapter import latest_market_date
    from my_strategy.web_dashboard.high_return import _prepare
    end = vintage_end or latest_market_date()
    item = _prepare(symbol, end, dates, db_path)
    context = stable_hash({"raw_series_revision": item["data_version"],
        "calendar_hash": stable_hash(dates), "raw_vintage_end": end, "price_basis": "unadjusted"})
    return item, context, end


def analyze_trend_consensus(*, symbol, start="2026-01-01", end=None, as_of=None, model_run_id=None,
                            model_fold=None, usage_mode="historical", calendar_run_id=None, device="cpu",
                            config=None, db_path=None, consensus_config=None, account_start=None,
                            account_initial_cash=None, **unused):
    from my_strategy.services.czsc_analysis import analyze_stock
    from my_strategy.services.czsc_research import _json
    from my_strategy.services.czsc_trend_runtime import runtime_overlay
    from my_strategy.web_dashboard.high_return import _calendar
    from my_strategy.web_dashboard.chart_evidence_trend import decorate_analysis
    result = analyze_stock(symbol, start=start, end=end, as_of=as_of, research=False, config=config, db_path=db_path)
    dates, calendar_id = _calendar(calendar_run_id)
    # The vintage identifies the full raw series. Only its strict requested
    # prefix can enter native structure, inference or the Broker replay below.
    item, source_context, vintage_end = _source_vintage(symbol, dates, db_path=db_path)
    requested_end = str(as_of or end or result.get("as_of") or vintage_end)[:10]
    available_sessions = [day for day in dates if day <= min(requested_end, vintage_end)]
    if not available_sessions:
        raise ValueError("请求日期之前没有已核验交易日")
    decision_date = available_sessions[-1]
    overlay = runtime_overlay(item["features"], dates, model_run_id=model_run_id, model_fold=model_fold,
        model_mode=usage_mode, as_of=decision_date, device=device or "cpu",
        account_start=account_start, initial_cash=account_initial_cash,
        source_context_hash=source_context, consensus_config=consensus_config)
    result.update(as_of=decision_date, decision_date=decision_date, requested_end=requested_end)
    out = decorate_analysis(result, raw=item["raw"], overlay=overlay, start=start)
    out["research"].update(calendar_run_id=calendar_id, series_data_version=item["data_version"],
        raw_vintage_end=vintage_end, observed_end=result["data_end"], requested_end=requested_end,
        decision_date=decision_date, model_run_id=overlay["model_run_id"])
    return _json(out)


def _stock_worker(payload):
    from my_strategy.web_dashboard.high_return import _worker_init
    from my_strategy.services.czsc_trend_runtime import runtime_overlay
    _worker_init()
    symbol, end, dates, params, vintage_end = payload
    item, source_context, _ = _source_vintage(symbol, dates, vintage_end=vintage_end)
    overlay = runtime_overlay(item["features"], dates, as_of=end, source_context_hash=source_context, **params)
    return symbol, item, overlay


def _batch(symbols, end, dates, params, workers, *, progress=None, check_cancel=None):
    from my_strategy.adapters.czsc_adapter import latest_market_date
    vintage_end = latest_market_date()
    payloads = [(s, end, dates, params, vintage_end) for s in symbols]
    done = 0
    if int(workers or 1) <= 1:
        for payload in payloads:
            if check_cancel:
                check_cancel()
            try:
                yield _stock_worker(payload), None
            except Exception as exc:
                yield None, {"symbol": payload[0], "error": f"{type(exc).__name__}: {exc}"}
            done += 1
            if progress:
                progress("趋势评分与账户回放", done, len(payloads))
        return
    pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"))
    queued = iter(payloads)
    futures = {}
    try:
        def submit_next():
            payload = next(queued, None)
            if payload is not None:
                futures[pool.submit(_stock_worker, payload)] = payload[0]
                return True
            return False
        for _ in range(min(len(payloads), int(workers) * 2)):
            submit_next()
        while futures:
            if check_cancel:
                check_cancel()
            ready, _ = wait(futures, timeout=.5, return_when=FIRST_COMPLETED)
            for future in ready:
                symbol = futures.pop(future)
                try:
                    value = future.result()
                except Exception as exc:
                    yield None, {"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    yield value, None
                done += 1
                if progress:
                    progress("趋势评分与账户回放", done, len(payloads))
                if check_cancel:
                    check_cancel()
                submit_next()
    finally:
        # A cancelled full-market request leaves at most the bounded active
        # workers to finish, rather than thousands of already queued replays.
        pool.shutdown(wait=True, cancel_futures=True)


def scan_trend_consensus(*, symbols=None, start="2026-01-01", end=None, model_run_id=None, model_fold=None,
                         usage_mode="historical", calendar_run_id=None, device="cpu", cpu_workers=4,
                         consensus_config=None, progress=None, check_cancel=None, run_callback=None, **unused):
    from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
    from my_strategy.services.czsc_research import _json, _save
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.web_dashboard.high_return import _calendar
    from my_strategy.services.czsc_trend_runtime import load_model_run
    target = latest_market_date(end)
    if not target:
        raise ValueError("截止日之前没有本地行情")
    dates, calendar_id = _calendar(calendar_run_id)
    root, report = load_model_run(model_run_id)
    selected = list(dict.fromkeys(symbols or [r["symbol"] for r in list_symbols()]))
    params = {"model_run_id": root.name if root else None, "model_fold": model_fold, "model_mode": usage_mode,
              "device": device or "cpu", "consensus_config": consensus_config}
    context = create_run_context(task="czsc-trend-consensus-scan", as_of_date=target, stocks=selected,
        config={"strategy": STRATEGY_ID, "parameters": params, "calendar_run_id": calendar_id},
        scope=("full" if symbols is None else "single" if len(selected) == 1 else "batch"),
        source="web", start_date=start, end_date=target)
    if run_callback:
        run_callback(context.run_id)
    rows, failed = [], []
    for value, failure in _batch(selected, target, dates, params, cpu_workers, progress=progress, check_cancel=check_cancel):
        if failure:
            failed.append(failure)
            continue
        symbol, item, overlay = value
        row = dict(overlay["latest_decision"])
        observed_end = str(overlay["features"].iloc[-1].date)[:10]
        row.update(symbol=symbol, data_end=observed_end, data_version=item["data_version"],
            raw_vintage_end=item["data_end"], observed_end=observed_end, requested_end=target,
            model_family="high_return", display_name=DISPLAY_NAME, publication_allowed=False,
            category="buy" if row.get("action") == "BUY" else "exit" if row.get("action") == "SELL"
                else "excluded" if observed_end != target else "watch")
        rows.append(row)
    out = _json({"run_id": context.run_id, "run_context": context.to_dict(), "strategy_version": STRATEGY_ID,
        "display_name": DISPLAY_NAME, "model_family": "high_return", "model_run_id": root.name if root else None,
        "data_end": target, "data_range": {"start": start, "end": target, "requested_end": end},
        "rows": rows, "failed": failed, "failures": failed, "publication_allowed": False,
        "model_gate": (report or {}).get("model_gate", {"passed": False}),
        "coverage": {"requested": len(selected), "successful": len(rows), "success": len(rows), "failed": len(failed)},
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "account_scope": "one_explicit_research_account_per_stock", "usage_mode": usage_mode})
    out["pricing_assumptions"] = dict(PRICING_ASSUMPTIONS)
    out["high_return_target"] = dict((report or {}).get("config", {}).get("high_return", {}))
    _save(context.subdir("reports") / "trend_scan.json", out)
    return ResultStore().save("scan", out)


def backtest_trend_consensus(*, model_run_id, symbols=None, start="2026-01-01", end=None,
                             initial_cash=100000., device="cpu", cpu_workers=4, calendar_run_id=None,
                             progress=None, check_cancel=None, run_callback=None, consensus_config=None, **unused):
    from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
    from my_strategy.services.czsc_research import _json, _save
    from my_strategy.services.czsc_trend_runtime import load_model_run
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.web_dashboard.high_return import _calendar
    root, report = load_model_run(model_run_id)
    if root is None:
        raise ValueError("账户回测需要完整趋势共识模型")
    target = latest_market_date(end)
    if not target or target < start:
        raise ValueError("回测区间内没有本地核验行情")
    dates, calendar_id = _calendar(calendar_run_id)
    selected = list(dict.fromkeys(symbols or [r["symbol"] for r in list_symbols()]))
    params = {"model_run_id": root.name, "model_mode": "historical", "device": device or "cpu",
              "account_start": start, "initial_cash": initial_cash, "consensus_config": consensus_config}
    context = create_run_context(task="czsc-trend-consensus-backtest", as_of_date=target, stocks=selected,
        config={"strategy": STRATEGY_ID, "parameters": params, "calendar_run_id": calendar_id},
        scope=("full" if symbols is None else "single" if len(selected) == 1 else "batch"),
        source="web", start_date=start, end_date=target)
    if run_callback:
        run_callback(context.run_id)
    summaries, failed, all_trades, all_rejections, account_equity = [], [], [], [], {}
    for value, failure in _batch(selected, target, dates, params, cpu_workers, progress=progress, check_cancel=check_cancel):
        if failure:
            failed.append(failure)
            continue
        symbol, item, overlay = value
        account = overlay["account_actions"]
        directory = context.subdir("accounts", symbol.replace(".", "_"))
        for name in ("ledger", "daily", "rejections", "holding_snapshots", "rows", "intents", "context"):
            _save(directory / (name + ".json"), _json(account.get(name, [])))
        summaries.append({"symbol": symbol, **account["metrics"], "series_context_hash": account["series_context_hash"]})
        all_trades.extend(account["ledger"])
        all_rejections.extend(account["rejections"])
        account_equity[symbol] = {r["date"]: float(r["equity"]) for r in account["daily"]}
    denominator = max(len(selected), 1)
    portfolio_daily = []
    marks = {symbol: float(initial_cash) for symbol in selected}
    peak = float(initial_cash) * denominator
    maximum_drawdown = 0.
    for day in dates:
        if not start <= day <= target:
            continue
        for symbol, values in account_equity.items():
            marks[symbol] = values.get(day, marks[symbol])
        equity = sum(marks.values())
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1)
        portfolio_daily.append({"date": day, "equity": equity, "initial_cash": float(initial_cash) * denominator})
    metrics = {"total_return": sum(r["total_return"] for r in summaries) / denominator,
        "max_drawdown": maximum_drawdown,
        "final_equity": sum(marks.values()),
        "trade_count": len(all_trades), "rejection_count": len(all_rejections), "failed_cash_accounts": len(failed),
        "account_scope": "fixed_equal_initial_cash_per_stock_not_shared_cash"}
    out = _json({"run_id": context.run_id, "run_context": context.to_dict(), "strategy_version": STRATEGY_ID,
        "display_name": DISPLAY_NAME, "model_family": "high_return", "model_run_id": root.name,
        "data_end": target, "data_range": {"start": start, "end": target}, "metrics": metrics,
        "initial_cash": float(initial_cash) * len(selected), "per_stock_initial_cash": float(initial_cash),
        "allocation": "每股固定独立初始资金；失败账户保留现金；合计资金曲线不代表共享现金账户",
        "stocks": summaries, "trades": all_trades, "rejections": all_rejections,
        "daily": portfolio_daily, "failed": failed, "failures": failed,
        "coverage": {"requested": len(selected), "successful": len(summaries), "success": len(summaries), "failed": len(failed)},
        "publication_allowed": False, "model_gate": report.get("model_gate", {"passed": False})})
    out["pricing_assumptions"] = dict(PRICING_ASSUMPTIONS)
    out["high_return_target"] = dict(report.get("config", {}).get("high_return", {}))
    _save(context.subdir("reports") / "trend_backtest.json", out)
    return ResultStore().save("backtest", out)
