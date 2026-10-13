"""Account-free analysis and screening; explicit backtests retain the Broker."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import multiprocessing

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, DISPLAY_NAME
from my_strategy.services.czsc_trend_runtime import PRICING_ASSUMPTIONS


def prepare_observed_data(symbol, end, dates, db_path=None):
    """Prepare causal current inputs from local raw observations and verified sessions."""
    from my_strategy.adapters.czsc_adapter import load_bars
    from my_strategy.services.czsc_market_features import build_features
    from my_strategy.services.czsc_market_profiles import bind_feature_profile, prepare_structure_inputs
    from my_strategy.services.czsc_price_volume_features import prepare_wyckoff_inputs
    from my_strategy.services.czsc_trend_support import _calendar_quality
    raw = load_bars(symbol, end=end, as_of=end, db_path=db_path)
    features = bind_feature_profile(_calendar_quality(build_features(raw), raw, dates), "ma_trend")
    features = prepare_wyckoff_inputs(prepare_structure_inputs(features), raw=raw, config={"market_dates": dates})
    for column in raw:
        if column not in features:
            features[column] = raw[column].to_numpy(copy=True)
    return {"symbol": symbol, "raw": raw, "features": features,
            "data_end": str(raw.date.iloc[-1])[:10], "data_version": raw.attrs["data_version"]}


def _source_vintage(symbol, dates, *, db_path=None, vintage_end=None):
    """Same source identity for analysis, scanning and explicit account replay.

    Full vintage metadata identifies revisions; runtime cuts the observed prefix
    before every model, native rule and account input.
    """
    from my_strategy.adapters.czsc_adapter import latest_market_date
    from my_strategy.web_dashboard.trend_strategy import prepare_observed_data as _prepare
    end = vintage_end or latest_market_date()
    item = _prepare(symbol, end, dates, db_path)
    context = stable_hash({"raw_series_revision": item["data_version"],
        "calendar_hash": stable_hash(dates), "raw_vintage_end": end, "price_basis": item["raw"].attrs.get("price_basis")})
    return item, context, end


def _reference_structure(raw, *, start, decision_date, config=None):
    """Use the existing native chart generator without creating Position state."""
    import pandas as pd
    import hashlib
    from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
    from my_strategy.services.czsc_analysis import CzscSession, strategy_config, _day
    settings = strategy_config(config)
    frame = raw[pd.to_datetime(raw.date).dt.strftime("%Y-%m-%d") <= decision_date].copy()
    if frame.empty:
        raise ValueError("参考日期之前没有本地行情")
    legacy = int(frame["source"].map(legacy_fuyao_source).sum()) if "source" in frame else 0
    frame.attrs.update(price_basis="mixed_legacy_adjustment_unverified" if legacy else "unadjusted",
        legacy_fuyao_bars=legacy,
        unverified_trade_price_bars=int((frame["has_trade_price"] != 1).sum()) if "has_trade_price" in frame else len(frame),
        data_version=hashlib.sha256(frame.to_csv(index=False, float_format="%.10g").encode()).hexdigest())
    lower = pd.Timestamp(start).date().isoformat() if start else None
    if lower and lower > _day(frame.iloc[-1].date):
        raise ValueError("起始日期晚于可用行情截止日")
    session = CzscSession(str(frame.iloc[0].symbol), settings, len(frame), create_position=False)
    for row in frame.itertuples(index=False):
        session.advance_structure(row)
    end = _day(frame.iloc[-1].date)
    return {"symbol": str(frame.iloc[0].symbol), "data_end": end, "as_of": decision_date,
        "analysis_mode": "reference", "reference_only": True, "account_scope": "none",
        "config_hash": stable_hash(settings), "strategy_version": settings["strategy_version"],
        "czsc_version": settings["czsc_version"], "source_sha256": settings["source_sha256"],
        "engine_contract": settings["engine_contract"], "data_version": frame.attrs["data_version"],
        "data_range": {"start": _day(frame.iloc[0].date), "end": end, "bars": len(frame)},
        "price_basis": frame.attrs["price_basis"], "input_quality": {key: frame.attrs[key] for key in
            ("legacy_fuyao_bars", "unverified_trade_price_bars")},
        "frequencies": session.payload(lower), "events": [],
        "warnings": ["周/月线在下一周期首次观测后才确认闭合，尚未闭合的末桶不用于规则。",
            "包含旧fuyao来源，复权口径未核验；保留原bar作为结构参考。" if legacy else "使用未复权价格；除权除息跳空会影响结构。",
            f"未核验真实交易价格的bar：{frame.attrs['unverified_trade_price_bars']}。",
            "as_of截断当前行情库快照；未重建当时的数据修订版本。",
            "B/S为各时点已知收盘价上的独立方向参考，不代表成交、账户操作或已达高收益门槛。"],
        "warmup": {"required": settings["warmup_bars"], "available": len(frame),
            "eligible": len(frame) >= settings["warmup_bars"]}}


def analyze_trend_consensus(*, symbol, start="2026-01-01", end=None, as_of=None, model_run_id=None,
                            model_fold=None, usage_mode="historical", calendar_run_id=None, device="cpu",
                            config=None, db_path=None, consensus_config=None, account_start=None,
                            account_initial_cash=None, analysis_mode="reference", structure_only=False, **unused):
    if usage_mode not in {"historical", "forward"}:
        raise ValueError("旧生产用途已退役；当前模型仅用于历史或当前时点研究")
    if analysis_mode != "reference":
        raise ValueError("分析入口只提供参考判断；资金模拟请使用显式回测入口")
    from my_strategy.services.czsc_trend_support import _json
    from my_strategy.services.czsc_trend_runtime import reference_overlay
    from my_strategy.services.czsc_trend_support import _calendar
    from my_strategy.web_dashboard.chart_evidence_trend import decorate_analysis
    from my_strategy.adapters.czsc_adapter import date_cutoff
    import pandas as pd
    dates, calendar_id = _calendar(calendar_run_id)
    # The vintage identifies the full raw series. Only its strict requested
    # prefix can enter native structure, inference or the Broker replay below.
    item, source_context, vintage_end = _source_vintage(symbol, dates, db_path=db_path)
    requested_end = str(as_of or end or vintage_end)[:10]
    cutoff = min(date_cutoff(value) for value in (as_of, end, vintage_end) if value is not None)
    available_sessions = [day for day in dates if pd.Timestamp(day) + pd.Timedelta(hours=15) <= cutoff]
    if not available_sessions:
        raise ValueError("请求日期之前没有已核验交易日")
    decision_date = available_sessions[-1]
    result = _reference_structure(item["raw"], start=start, decision_date=decision_date, config=config)
    overlay = reference_overlay(item["features"], dates, model_run_id=model_run_id, model_fold=model_fold,
        model_mode="production" if usage_mode == "forward" else "historical", as_of=decision_date, device=device or "cpu",
        source_context_hash=source_context, consensus_config=consensus_config, use_models=not structure_only)
    result.update(as_of=decision_date, decision_date=decision_date, requested_end=requested_end)
    out = decorate_analysis(result, raw=item["raw"], overlay=overlay, start=start)
    out["research"].update(calendar_run_id=calendar_id, series_data_version=item["data_version"],
        raw_vintage_end=vintage_end, observed_end=result["data_end"], requested_end=requested_end,
        decision_date=decision_date, model_run_id=overlay["model_run_id"])
    out["research"]["usage_mode"] = usage_mode
    out["usage_mode"] = usage_mode
    out["analysis_mode"] = "reference"
    out["structure_only"] = bool(structure_only)
    return _json(out)


def _stock_worker(payload):
    from threadpoolctl import threadpool_limits
    from my_strategy.services.czsc_trend_runtime import runtime_overlay, reference_overlay
    threadpool_limits(limits=1)
    symbol, end, dates, params, vintage_end = payload
    item, source_context, _ = _source_vintage(symbol, dates, vintage_end=vintage_end)
    values = dict(params)
    purpose = values.pop("analysis_mode", "reference")
    adapter = reference_overlay if purpose == "reference" else runtime_overlay
    if purpose not in {"reference", "backtest"}:
        raise ValueError("未知分析用途")
    if purpose == "reference":
        observed = [str(day)[:10] for day in item["features"].date if str(day)[:10] <= end]
        last_observed = max(observed) if observed else None
        if last_observed != end:
            raise ValueError(f"行情缺少扫描日 {end}，标的最后日期为 {last_observed}")
    overlay = adapter(item["features"], dates, as_of=end, source_context_hash=source_context, **values)
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
                progress("多源参考分析" if params.get("analysis_mode") == "reference" else "趋势评分与账户回放", done, len(payloads))
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
                    progress("多源参考分析" if params.get("analysis_mode") == "reference" else "趋势评分与账户回放", done, len(payloads))
                if check_cancel:
                    check_cancel()
                submit_next()
    finally:
        # A cancelled full-market request leaves at most the bounded active
        # workers to finish, rather than thousands of already queued replays.
        pool.shutdown(wait=True, cancel_futures=True)


def scan_trend_consensus(*, symbols=None, start="2026-01-01", end=None, model_run_id=None, model_fold=None,
                         usage_mode="historical", calendar_run_id=None, device="cpu", cpu_workers=4,
                         consensus_config=None, progress=None, check_cancel=None, run_callback=None,
                         analysis_mode="reference", structure_only=False, **unused):
    if usage_mode not in {"historical", "forward"}:
        raise ValueError("旧生产用途已退役；当前模型仅用于历史或当前时点研究")
    if analysis_mode != "reference":
        raise ValueError("扫描入口只提供参考判断；资金模拟请使用显式回测入口")
    from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
    from my_strategy.services.czsc_trend_support import _json, _save
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.services.czsc_trend_support import _calendar
    from my_strategy.services.czsc_trend_runtime import load_reference_model_run
    from my_strategy.services.czsc_market_history import MarketHistoryBuilder, load_classification
    from my_strategy.storage.czsc_market_history import MarketHistoryStore
    target = latest_market_date(end)
    if not target:
        raise ValueError("截止日之前没有本地行情")
    dates, calendar_id = _calendar(calendar_run_id)
    root, report = load_reference_model_run(model_run_id, as_of=target) if not structure_only else (None, None)
    catalog = list_symbols()
    selected = list(dict.fromkeys(symbols or [r["symbol"] for r in catalog]))
    params = {"model_run_id": root.name if root else None, "model_fold": model_fold, "model_mode": "production" if usage_mode == "forward" else "historical",
              "device": device or "cpu", "consensus_config": consensus_config, "analysis_mode": "reference",
              "model_selection_frozen": True, "use_models": not structure_only}
    context = create_run_context(task="trend-consensus-scan", as_of_date=target, stocks=selected,
        config={"strategy": STRATEGY_ID, "parameters": params, "calendar_run_id": calendar_id},
        scope=("full" if symbols is None else "single" if len(selected) == 1 else "batch"),
        source=unused.get("source", "web"), start_date=start, end_date=target)
    if run_callback:
        run_callback(context.run_id)
    result_store = ResultStore()
    history_store = MarketHistoryStore(result_store.db_path)
    classifications, classification = load_classification(selected)
    history = MarketHistoryBuilder(selected, dates, target, run_id=context.run_id,
        classifications=classifications, classification=classification,
        names={row["symbol"]: row.get("name", "") for row in catalog},
        contract={"strategy_version": STRATEGY_ID, "model_run_id": params["model_run_id"],
            "usage_mode": usage_mode, "model_fold": model_fold, "calendar_run_id": calendar_id,
            "config_hash": stable_hash(params), "analysis_mode": "reference", "account_scope": "none",
            "source_policy": "causal_reference_overlay_without_account"})
    history_store.begin(context.run_id, {"dates": history.dates, "symbols": selected,
        "classification": classification, "contract": history.contract})
    rows, failed = [], []
    effective_consensus = None
    for value, failure in _batch(selected, target, dates, params, cpu_workers, progress=progress, check_cancel=check_cancel):
        if failure:
            failed.append(failure)
            history_store.save_stock(context.run_id, history.add_stock(failure["symbol"], failure=failure))
            continue
        symbol, item, overlay = value
        if effective_consensus is None:
            effective_consensus = dict(overlay.get("trend_consensus", {}).get("config", {}))
        facts = history.add_stock(symbol, overlay=overlay)
        for fact in facts:
            fact.update(data_version=item["data_version"], raw_vintage_end=item["data_end"],
                model_run_id=params["model_run_id"], calendar_run_id=calendar_id)
        history_store.save_stock(context.run_id, facts)
        row = dict(overlay["latest_decision"])
        observed_end = str(overlay["features"].iloc[-1].date)[:10]
        row.update(symbol=symbol, data_end=observed_end, data_version=item["data_version"],
            raw_vintage_end=item["data_end"], observed_end=observed_end, requested_end=target,
            model_family="current", display_name=DISPLAY_NAME, publication_allowed=False,
            category="buy" if row.get("action") == "BUY" else "exit" if row.get("action") == "SELL"
                else "excluded" if observed_end != target else "watch")
        rows.append(row)
    if selected and len(failed) == len(selected):
        raise ValueError(f"全部 {len(selected)} 个标的失败：{failed[:3]}")
    out = _json({"run_id": context.run_id, "run_context": context.to_dict(), "strategy_version": STRATEGY_ID,
        "display_name": DISPLAY_NAME, "model_family": "current", "model_run_id": root.name if root else None,
        "data_end": target, "data_range": {"start": start, "end": target, "requested_end": end},
        "rows": rows, "failed": failed, "failures": failed, "publication_allowed": False,
        "model_gate": (report or {}).get("model_gate", {"passed": False}),
        "coverage": {"requested": len(selected), "successful": len(rows), "success": len(rows), "failed": len(failed)},
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "account_scope": "none", "analysis_mode": "reference", "reference_only": True, "usage_mode": usage_mode,
        "models_enabled": not structure_only, "structure_only": bool(structure_only)})
    from my_strategy.services.czsc_trend_runtime import REFERENCE_PRICING_ASSUMPTIONS
    out["pricing_assumptions"] = dict(REFERENCE_PRICING_ASSUMPTIONS)
    out["consensus_config"] = effective_consensus or consensus_config
    out["calendar_run_id"] = calendar_id
    out["market_history"] = history.finish()
    history_store.finish(context.run_id, out["market_history"])
    _save(context.subdir("reports") / "trend_scan.json", out)
    return result_store.save("scan", out)


def backtest_trend_consensus(*, model_run_id=None, model_fold=None, usage_mode="historical", symbols=None, start="2026-01-01", end=None,
                             initial_cash=100000., device="cpu", cpu_workers=4, calendar_run_id=None,
                             progress=None, check_cancel=None, run_callback=None, consensus_config=None, **unused):
    if usage_mode != 'historical' or model_fold is not None:
        raise ValueError('账户回测只支持历史按日期匹配，不能静默忽略固定检查点或当前时点研究')
    from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
    from my_strategy.services.czsc_trend_support import _json, _save
    from my_strategy.services.czsc_trend_runtime import load_model_run
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.services.czsc_trend_support import _calendar
    target = latest_market_date(end)
    root, report = load_model_run(model_run_id, as_of=target)
    if not target or target < start:
        raise ValueError("回测区间内没有本地核验行情")
    dates, calendar_id = _calendar(calendar_run_id)
    selected = list(dict.fromkeys(symbols or [r["symbol"] for r in list_symbols()]))
    params = {"model_run_id": root.name if root else None, "model_mode": "historical", "device": device or "cpu",
              "account_start": start, "initial_cash": initial_cash, "consensus_config": consensus_config,
              "analysis_mode": "backtest"}
    context = create_run_context(task="trend-consensus-backtest", as_of_date=target, stocks=selected,
        config={"strategy": STRATEGY_ID, "parameters": params, "calendar_run_id": calendar_id},
        scope=("full" if symbols is None else "single" if len(selected) == 1 else "batch"),
        source=unused.get("source", "web"), start_date=start, end_date=target)
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
    if selected and len(failed) == len(selected):
        raise ValueError(f"全部 {len(selected)} 个标的失败：{failed[:3]}")
    out = _json({"run_id": context.run_id, "run_context": context.to_dict(), "strategy_version": STRATEGY_ID,
        "display_name": DISPLAY_NAME, "model_family": "current", "model_run_id": root.name if root else None,
        "data_end": target, "data_range": {"start": start, "end": target}, "metrics": metrics,
        "initial_cash": float(initial_cash) * len(selected), "per_stock_initial_cash": float(initial_cash),
        "allocation": "每股固定独立初始资金；失败账户保留现金；合计资金曲线不代表共享现金账户",
        "stocks": summaries, "trades": all_trades, "rejections": all_rejections,
        "daily": portfolio_daily, "failed": failed, "failures": failed,
        "coverage": {"requested": len(selected), "successful": len(summaries), "success": len(summaries), "failed": len(failed)},
        "publication_allowed": False, "model_gate": (report or {}).get("model_gate", {"passed": False})})
    out["pricing_assumptions"] = dict(PRICING_ASSUMPTIONS)
    out["high_return_target"] = dict((report or {}).get("config", {}).get("high_return", {}))
    _save(context.subdir("reports") / "trend_backtest.json", out)
    return ResultStore().save("backtest", out)
