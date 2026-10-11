"""Explicit shadow adapters for the independent high-return research family.

The existing strategy and model routing stay frozen. These adapters prepare the
same causal inputs, then attach independent signal/target evidence to the UI.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import pandas as pd

from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
from my_strategy.core.run_context import create_run_context, stable_hash


def _calendar(calendar_run_id=None):
    from my_strategy.services.czsc_research import _calendar as verified_calendar, research_status
    identity = calendar_run_id or research_status().get("calendar_run_id")
    if not identity:
        raise ValueError("三路买卖点研究需要已核验交易日历")
    return verified_calendar(identity)[0], identity


def _worker_init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _prepare(symbol, end, dates, db_path=None):
    from my_strategy.services.czsc_research_runtime import _prepare_stock
    item = _prepare_stock(symbol, end, dates, db_path=db_path,
                          feature_profile="ma_trend_v1", expert_family="wyckoff")
    features, raw = item["features"].copy(), item["raw"]
    feature_dates = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d").tolist()
    raw_dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d").tolist()
    if (feature_dates != raw_dates or len(set(raw_dates)) != len(raw_dates)
            or features["symbol"].astype(str).tolist() != raw["symbol"].astype(str).tolist()):
        raise ValueError("三路研究特征与原始行情的股票和日期不一致")
    # The old model projection excludes OHLCV. Independent signal and risk
    # decisions need the corresponding observed bar, with the full prefix.
    for column in raw:
        if column not in features:
            features[column] = raw[column].to_numpy(copy=True)
    return {**item, "features": features}


def _overlay(item, dates, *, model_run_id=None, model_fold=None, usage_mode="historical", device="cpu", as_of=None):
    from my_strategy.services.czsc_high_return_runtime import runtime_overlay
    return runtime_overlay(item["features"], dates, model_run_id=model_run_id,
                           model_fold=model_fold, model_mode=usage_mode, as_of=as_of,
                           holding_snapshot=None, device=device or "cpu")


def _latest(overlay, item):
    decision = overlay.get("latest_decision") or overlay.get("next_session_decision")
    if decision is None:
        rows = overlay.get("rows", overlay.get("scores", overlay.get("decisions", [])))
        if isinstance(rows, pd.DataFrame):
            rows = rows.to_dict("records")
        decision = rows[-1] if rows else {}
    row = dict(decision)
    resolution = row.get("model_resolution", overlay.get("model_resolution", {}))
    row.update(symbol=item["symbol"], data_end=item["data_end"], data_version=item["data_version"],
               model_family="high_return", model_resolution=resolution,
               model_gate=overlay.get("model_gate", {"passed": False}), publication_allowed=False,
               high_return_evidence=row.get("high_return_evidence", overlay.get("high_return_evidence")))
    row.setdefault("probability_target", overlay.get("probability_target", "按冻结退出政策形成费用后高收益区间的概率"))
    row.setdefault("exit_model", {"probability": None, "reason": "缺少兼容的真实Broker持仓历史，未评估退出优势"})
    row.setdefault("category", "buy" if row.get("candidate_action") == "BUY" else "watch")
    row.setdefault("action", "WAIT")
    row.setdefault("available_at", item["data_end"] + "T15:00:00+08:00")
    row.setdefault("reference_close", float(item["raw"].iloc[-1].close))
    row.setdefault("reference_price_date", item["data_end"])
    # No real Broker state is supplied here. Never retain a generic/entry value
    # in the exit probability field of a standalone analysis or market scan.
    row["exit_model"] = {**row["exit_model"], "probability": None}
    row["exit_probability"] = None
    return row


def _events(overlay, start=None):
    points = overlay.get("signal_points", overlay.get("signals", []))
    if isinstance(points, pd.DataFrame):
        points = points.to_dict("records")
    events = []
    for value in points or []:
        point = dict(value)
        when = str(point.get("available_at", point.get("signal_at", point.get("time", ""))))
        if not when or (start and when[:10] < start):
            continue
        side = str(point.get("side", "")).lower()
        action = "buy" if side == "buy" else "sell" if side == "sell" else point.get("action", "watch")
        events.append({**point, "time": when[:10], "available_at": when,
                       "action": action, "signal": point.get("event_type", point.get("signal", "候选信号")),
                       "reason": point.get("reason", "三路独立研究信号；实际成交以Broker账本为准")})
    return events


def _display_model_lineage(overlay):
    """Attach verified saved entry-head identity to display scores without inference."""
    import copy
    from my_strategy.services.czsc_high_return_runtime import _read_run, _head_path
    from my_strategy.services.czsc_high_return_models import load_head

    output = copy.deepcopy(overlay)
    runs, manifests = {}, {}
    for row in output.get("rows", []):
        identity = row.get("model_resolution") or {}
        if row.get("high_return_probability") is None or not identity.get("model_run_id") or not identity.get("checkpoint"):
            continue
        key = (identity["model_run_id"], identity["checkpoint"])
        if key not in manifests:
            if key[0] not in runs:
                root, report, _ = _read_run(key[0])
                runs[key[0]] = (root, {bundle["name"]: bundle for bundle in report.get("model_bundles", [])})
            root, bundles = runs[key[0]]
            bundle = bundles.get(key[1])
            if bundle is None:
                raise ValueError("图表评分的高收益模型检查点身份缺失")
            manifest = load_head(_head_path(root, bundle, "entry"))
            if manifest["head"] != "entry" or manifest["available_at"] != identity["available_at"]:
                raise ValueError("图表评分的高收益模型身份与冻结清单不一致")
            manifests[key] = manifest
        manifest = manifests[key]
        identity.update(train_label_end=manifest["train_label_end"], model_head=manifest["head"],
                        target_contract_hash=manifest["target_contract_hash"], manifest_sha256=manifest["manifest_sha256"],
                        actual_training_completed_at=manifest["training_completed_at"],
                        fold=copy.deepcopy(manifest["fold"]))
    return output


def _analysis_overlay_result(result, overlay, item, *, start, usage_mode, model_policy, calendar_id):
    """Assemble the original shadow result, shared by analysis and diagnostics."""
    import copy
    result = copy.deepcopy(result)
    result["structure_baseline"] = {"events": result["events"], "next_session_decision": result["next_session_decision"]}
    result.update(model_family="high_return", events=_events(overlay, start), signal_points=overlay.get("signal_points", []),
                  next_session_decision=_latest(overlay, item), model_gate=overlay.get("model_gate"),
                  publication_allowed=False, compute_info=overlay.get("compute_info", {}),
                  research={"model_family": "high_return", "usage_mode": usage_mode, "model_policy": model_policy,
                            "calendar_run_id": calendar_id, "model_routes": overlay.get("model_routes", []),
                            "probability_target": overlay.get("probability_target"), "publication_allowed": False})
    return result


def analyze_high_return(*, symbol, start="2026-01-01", end=None, as_of=None, research=True,
                        model_family="high_return", usage_mode="historical", model_policy="auto",
                        model_run_id=None, model_fold=None, calendar_run_id=None, device=None,
                        entry_policy="legacy", config=None, db_path=None, consensus_config=None, consensus_version=None):
    from my_strategy.services.czsc_analysis import analyze_stock
    from my_strategy.services.czsc_research import _json
    result = analyze_stock(symbol, start=start, end=end, as_of=as_of, db_path=db_path, config=config, research=False)
    dates, calendar_id = _calendar(calendar_run_id)
    item = _prepare(symbol, result["data_end"], dates, db_path)
    overlay = _overlay(item, dates, model_run_id=model_run_id, model_fold=model_fold,
                       usage_mode=usage_mode, device=device, as_of=as_of or result["data_end"])
    from my_strategy.services.czsc_agent_consensus import consensus_config as resolve_consensus_config
    from my_strategy.services.czsc_candidate_evidence import V3_VERSION, recover_daily_evidence
    effective_consensus = resolve_consensus_config(consensus_config, version=consensus_version)
    if effective_consensus.get("version") == V3_VERSION:
        # Recover the exact selected frozen daily routes only for the new chart
        # policy. Fixed-fold routing and all original trading facts remain bound.
        overlay = recover_daily_evidence({**overlay, "chart_model_fold": model_fold},
                                         item["features"], dates, usage_mode=usage_mode)
    result = _analysis_overlay_result(result, overlay, item, start=start, usage_mode=usage_mode,
                                     model_policy=model_policy, calendar_id=calendar_id)
    from my_strategy.web_dashboard.chart_evidence import decorate_analysis
    result = decorate_analysis(result, raw=item["raw"], features=item["features"], overlay=_display_model_lineage(overlay),
                               consensus_config=consensus_config, consensus_version=consensus_version)
    return _json(result)


def compare_high_return_chart_windows(*, symbol, start="2026-01-01", end=None, as_of=None,
        usage_mode="historical", model_policy="auto", model_run_id=None, model_fold=None,
        calendar_run_id=None, device="cpu", config=None, db_path=None, consensus_config=None,
        windows=(1, 3, 5)):
    """Compare actual v2 and independently assessed v3 evidence windows.

    Prepare frozen inputs and run original shadow scoring once. The v2 control
    receives the original unrecovered overlay. Every v3 variant rebuilds daily
    current-state evidence under its own age bound before arbitration; no variant
    filters another variant's already aggregated evidence. This is a read-only
    development diagnostic, not a Broker or release assessment.
    """
    from my_strategy.services.czsc_analysis import analyze_stock
    from my_strategy.services.czsc_research import _json
    from my_strategy.services.czsc_agent_consensus import consensus_config as resolve_consensus_config
    from my_strategy.services.czsc_candidate_evidence import V3_VERSION, recover_daily_evidence
    from my_strategy.web_dashboard.chart_evidence import decorate_analysis
    windows = tuple(windows)
    if (not windows or len(set(windows)) != len(windows)
            or any(isinstance(value, bool) or not isinstance(value, int) or value not in {1, 3, 5} for value in windows)):
        raise ValueError("chart diagnostic windows must be distinct allowed bar counts: 1, 3, 5")
    selected_config = resolve_consensus_config(consensus_config, version=V3_VERSION)
    variant_configs = {window: resolve_consensus_config({**selected_config, "evidence_valid_bars": window}, version=V3_VERSION)
                       for window in windows}
    base = analyze_stock(symbol, start=start, end=end, as_of=as_of, db_path=db_path, config=config, research=False)
    dates, calendar_id = _calendar(calendar_run_id)
    item = _prepare(symbol, base["data_end"], dates, db_path)
    original = _overlay(item, dates, model_run_id=model_run_id, model_fold=model_fold,
                        usage_mode=usage_mode, device=device, as_of=as_of or base["data_end"])
    recovered = recover_daily_evidence({**original, "chart_model_fold": model_fold},
                                       item["features"], dates, usage_mode=usage_mode)
    legacy_identity = _display_model_lineage(original)
    recovered_identity = _display_model_lineage(recovered)
    controls = [("v2", original, legacy_identity, None, "czsc_agent_consensus_v2")]
    controls.extend((f"v3_{window}", recovered, recovered_identity,
                     variant_configs[window], V3_VERSION) for window in windows)
    variants = {}
    for name, overlay, identity, selected, version in controls:
        assembled = _analysis_overlay_result(base, overlay, item, start=start, usage_mode=usage_mode,
                                            model_policy=model_policy, calendar_id=calendar_id)
        variants[name] = decorate_analysis(assembled, raw=item["raw"], features=item["features"],
            overlay=identity, consensus_config=selected, consensus_version=version)
    return _json({"diagnostic_only": True, "publication_allowed": False, "variants": variants,
                  "preparation": {"original_shadow_compute": original.get("compute_info"),
                                  "recovered_chart_compute": recovered.get("compute_info"),
                                  "model_run_id": recovered.get("candidate_evidence_policy", {}).get("model_run_id")},
                  "comparison_basis": "same_full_causal_input_and_original_candidates_reassessed_per_window"})


def scan_high_return(*, symbols=None, start="2026-01-01", end=None, model_run_id=None, model_fold=None,
                     usage_mode="historical", calendar_run_id=None, device="cpu", cpu_workers=4,
                     progress=None, check_cancel=None, run_callback=None):
    from my_strategy.services.czsc_research import _json
    target = latest_market_date(end)
    if not target:
        raise ValueError("截止日之前没有本地行情")
    dates, calendar_id = _calendar(calendar_run_id)
    symbols = list(dict.fromkeys(symbols or [row["symbol"] for row in list_symbols()]))
    if not symbols:
        raise ValueError("本地行情没有有效股票")
    spec = {"symbols": symbols, "start": start, "end": target, "model_run_id": model_run_id,
            "model_fold": model_fold, "usage_mode": usage_mode, "model_family": "high_return", "research": True,
            "calendar_run_id": calendar_id, "device": device, "cpu_workers": cpu_workers}
    context = create_run_context(task="czsc-high-return-scan", as_of_date=target, config=spec,
                                 data_version=str(target), stocks=symbols, start_date=start, end_date=target)
    if run_callback:
        run_callback(context.run_id)
    rows, failures, versions, compute = [], [], {}, []
    def consume(symbol, item=None, error=None):
        if check_cancel:
            check_cancel()
        if error is not None:
            failures.append({"symbol": symbol, "error": str(error)})
        else:
            try:
                if item["data_end"] != target:
                    raise ValueError(f"缺少扫描日 {target} 行情，最后日期 {item['data_end']}")
                overlay = _overlay(item, dates, model_run_id=model_run_id, model_fold=model_fold,
                                   usage_mode=usage_mode, device=device, as_of=target)
                rows.append(_latest(overlay, item))
                versions[symbol] = item["data_version"]
                compute.append(overlay.get("compute_info", {}))
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                failures.append({"symbol": symbol, "error": str(exc)})
        if progress:
            progress("三路独立买卖点与高收益影子判断", len(rows) + len(failures), len(symbols), len(failures))
    workers = min(max(int(cpu_workers or 4), 1), 16, len(symbols))
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
        pending, remaining = {}, iter(symbols)
        def submit_next():
            symbol = next(remaining, None)
            if symbol is not None:
                pending[pool.submit(_prepare, symbol, target, dates)] = symbol
        try:
            for _ in range(min(len(symbols), workers * 2)):
                submit_next()
            while pending:
                if check_cancel:
                    check_cancel()
                completed, _ = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                for future in completed:
                    symbol = pending.pop(future)
                    try:
                        item = future.result()
                    except Exception as exc:
                        consume(symbol, error=exc)
                    else:
                        consume(symbol, item=item)
                    submit_next()
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=not pending, cancel_futures=True)
    else:
        for symbol in symbols:
            try:
                item = _prepare(symbol, target, dates)
            except (ValueError, FileNotFoundError) as exc:
                consume(symbol, error=exc)
            else:
                consume(symbol, item=item)
    if not rows:
        raise ValueError(f"全部 {len(symbols)} 只股票失败：{failures[:3]}")
    order = {symbol: i for i, symbol in enumerate(symbols)}
    rows.sort(key=lambda row: order[row["symbol"]])
    result = {"run_id": context.run_id, "run_context": context.to_dict(), "request": spec,
              "rows": rows, "model_family": "high_return", "publication_allowed": False,
              "data_range": {"start": start, "end": target}, "data_version": stable_hash(versions),
              "input_versions": versions, "failures": failures,
              "coverage": {"requested": len(symbols), "success": len(rows), "failed": len(failures)},
              "category_counts": {name: sum(row.get("category") == name for row in rows) for name in ("buy", "watch", "exit", "excluded")},
              "model_gate": {"passed": False, "reason": "高收益区间模型仅作影子；新独立验证尚未完成"},
              "compute_info": {"research": True, "model_family": "high_return", "cpu_workers": workers,
                               "model_inference_rows": sum(item.get("model_inference_rows", 0) for item in compute),
                               "model_inference_batches": sum(item.get("model_inference_batches", 0) for item in compute),
                               "selected_device": device}}
    context.write_metadata({"status": "complete", "coverage": result["coverage"], "failures": failures})
    return _json(result)
