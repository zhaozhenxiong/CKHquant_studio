"""Portable CLI for the CZSC-only research workbench."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.runtime_env import configure_utf8_stdio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KHQuant CZSC 结构研究")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "scan", "backtest", "backtest-batch"):
        item = commands.add_parser(name)
        item.add_argument("--symbol", help="单股代码，例如 600519.SH")
        item.add_argument("--stocks", default="", help="逗号分隔代码；扫描/批量留空表示本地全部股票")
        item.add_argument("--start", default="2026-01-01", help="开始日期，默认2026-01-01；保留更早行情供预热")
        item.add_argument("--end", help="截止日期，默认最新本地行情日期")
        item.add_argument("--as-of", help="分析回放截止日或Asia/Shanghai时间")
        item.add_argument("--initial-cash", type=float, default=100000)
        item.add_argument("--device", help="计算设备：auto、cpu、mps、cuda 或 cuda:编号")
        item.add_argument("--research", action="store_true", default=True, help="统一缠论量价ML研究（默认）")
        item.add_argument("--model-family", choices=("ma_trend", "dual", "wyckoff", "high_return"), default="ma_trend", help="均线单模型、双专家、威科夫三专家或高收益区间影子研究")
        item.add_argument("--structure-only", dest="research", action="store_false", help="原生结构规则对照")
        item.add_argument("--usage-mode", choices=("production", "historical", "retrospective"), default="historical")
        item.add_argument("--model-policy", choices=("auto", "pinned"), default="auto")
        item.add_argument("--entry-policy", choices=("legacy", "fresh", "risk"), default="legacy",
                          help="入场契约：原目标仓位对照、有效买点研究、价格风险实验")
        item.add_argument("--model-run-id", help="研究模型运行；未认证时保持影子")
        item.add_argument("--model-fold", help="固定检查点名称")
        item.add_argument("--calendar-run-id", help="核验交易日历运行")
        if name != "analyze":
            item.add_argument("--cpu-workers", type=int, help="只读结构准备进程数（1到16，默认4）")
            item.add_argument("--batch-size", type=int, help="GPU 微批大小（1到128，默认32）")
        item.add_argument("--json", action="store_true")
    research = commands.add_parser("research-train")
    research.add_argument("--stocks", default="")
    research.add_argument("--end", required=True)
    research.add_argument("--calendar-run-id")
    research.add_argument("--device", default="auto")
    research.add_argument("--cpu-workers", type=int, default=4)
    research.add_argument("--json", action="store_true")
    dual = commands.add_parser("research-dual-train", help="双专家融合与实际持仓退出完整研究")
    dual.set_defaults(model_family="dual")
    dual.add_argument("--source-training-run-id", required=True, help="包含全列冻结特征与10日标签的已完成研究运行")
    dual.add_argument("--stocks", default="")
    dual.add_argument("--end", required=True)
    dual.add_argument("--device", default="auto")
    dual.add_argument("--cpu-workers", type=int, default=4)
    dual.add_argument("--json", action="store_true")
    wyckoff = commands.add_parser("research-wyckoff-train", help="威科夫量价三专家与真实持仓买卖契约研究")
    wyckoff.set_defaults(model_family="wyckoff")
    wyckoff.add_argument("--source-training-run-id", required=True)
    wyckoff.add_argument("--stocks", default="")
    wyckoff.add_argument("--end", required=True)
    wyckoff.add_argument("--device", default="auto")
    wyckoff.add_argument("--cpu-workers", type=int, default=4)
    wyckoff.add_argument("--json", action="store_true")
    for name, help_text in (
        ("research-points-build", "独立生成均线ML、结构、威科夫买卖点"),
        ("research-episodes-build", "按真实Broker构建操作区间与每日持仓"),
        ("research-high-return-train", "前向训练高收益入场与独立持仓退出模型"),
        ("research-high-return-backtest", "显式离线应用高收益入场与退出政策，保存完整账本"),
    ):
        item = commands.add_parser(name, help=help_text)
        item.set_defaults(model_family="high_return", research=True)
        if name == "research-high-return-backtest":
            item.add_argument("--model-run-id", required=True, help="已完成的高收益区间研究运行")
            item.add_argument("--start", default="2026-01-01")
        else:
            item.add_argument("--source-training-run-id", required=True, help="已完成并保存因果特征的冻结来源运行")
        item.add_argument("--end", help="截止日；留空采用冻结来源截止日")
        item.add_argument("--stocks", default="", help="逗号分隔股票；留空使用冻结来源股票池")
        item.add_argument("--device", default="auto", help="模型计算设备，显式不可用设备拒绝执行")
        item.add_argument("--cpu-workers", type=int, default=4)
        item.add_argument("--config", type=Path, help="独立高收益研究协议JSON；不覆盖旧模型配置")
        item.add_argument("--json", action="store_true")
    high_status = commands.add_parser("research-high-return-status", help="查看高收益模型、覆盖与影子资格")
    high_status.add_argument("--json", action="store_true")
    high_audit = commands.add_parser("research-high-return-audit", help="只读核验高收益模型、真实账本、配置与时间契约")
    high_audit.add_argument("--model-run-id", required=True, help="待审计的高收益区间研究运行")
    high_audit.add_argument("--as-of", help="资格审查信息截止时间；留空采用当前时间")
    high_audit.add_argument("--json", action="store_true")
    legacy = commands.add_parser('research-legacy-comparison', help='同冻结输入真实回放旧组合规则与固定10日MA离线对照')
    legacy.add_argument('--anchor-run-id', required=True, help='冻结来源、日历、费用和资金口径的高收益研究运行')
    legacy.add_argument('--legacy-model-run-id', help='原固定10日MA运行；留空使用已核验完整旧运行')
    legacy.add_argument('--stocks', default='', help='逗号分隔主板代码；留空完整冻结主板池')
    legacy.add_argument('--cpu-workers', type=int, default=4)
    legacy.add_argument('--json', action='store_true')
    legacy_audit = commands.add_parser('research-legacy-comparison-audit', help='只读核验旧对照来源、模型、实际概率与Broker账本')
    legacy_audit.add_argument('--comparison-run-id', required=True)
    legacy_audit.add_argument('--as-of', help='审查时间；留空当前时间')
    legacy_audit.add_argument('--json', action='store_true')
    held = commands.add_parser('research-high-return-holding', help='只读核验保存的真实Broker回放持仓并研究退出优势')
    held.add_argument('--replay-run-id', required=True, help='完整高收益回测或训练开发账本运行')
    held.add_argument('--window', required=True, help='保存的账户窗口，例如 explicit_offline 或 2025Q4')
    held.add_argument('--arm', choices=['reference','full','entry_only','exit_only','ma','structure','wyckoff'], default='full')
    held.add_argument('--symbol', required=True)
    held.add_argument('--as-of', required=True, help='已观察的收盘日；持仓历史只读取到此日')
    held.add_argument('--usage-mode', choices=['production','historical'], default='production', help='生产影子遵守实际训练完成时间；历史仅作开发回放')
    held.add_argument('--json', action='store_true')
    forward = commands.add_parser("research-high-return-forward", help="追加实际冻结后前向证据，保留完整账本与未成熟区间")
    forward.add_argument("--model-run-id", required=True, help="完整冻结的高收益模型运行")
    forward.add_argument("--source-training-run-id", required=True, help="具有冻结后因果行情与特征的新来源运行")
    forward.add_argument("--end", help="证据截止日；须有实际冻结后的已核验行情")
    forward.add_argument("--previous-run-id", help="上一份前向证据运行；用于连续追加与核验")
    forward.add_argument("--cpu-workers", type=int, default=4)
    forward.add_argument("--json", action="store_true")
    forward_source = commands.add_parser("research-high-return-forward-source", help="按冻结股票池生成新的逐时点行情与特征快照")
    forward_source.add_argument("--model-run-id", required=True, help="固定股票池和协议的高收益模型运行")
    forward_source.add_argument("--end", help="快照截止日；留空采用本地实际行情截止日")
    forward_source.add_argument("--calendar-run-id", help="覆盖新行情的已核验交易日历运行")
    forward_source.add_argument("--cpu-workers", type=int, default=4)
    forward_source.add_argument("--json", action="store_true")
    forward_status = commands.add_parser("research-high-return-forward-status", help="查看前向计划、追加登记与独立窗口成熟状态")
    forward_status.add_argument("--model-run-id", required=True)
    forward_status.add_argument("--json", action="store_true")
    for name in ("research-promote", "research-rollback", "research-status"):
        item = commands.add_parser(name)
        if name == "research-promote":
            item.add_argument("--model-run-id", required=True)
            item.add_argument("--model-fold", default="production")
        if name == "research-rollback":
            item.add_argument("--release-id", help="恢复已发布版本；留空则撤回活动模型")
        if name != "research-status":
            item.add_argument("--reason", required=True)
        else:
            item.add_argument("--model-family", choices=["ma_trend", "dual", "wyckoff", "high_return"], default="ma_trend")
        item.add_argument("--json", action="store_true")
    for name in ("dashboard", "update-data", "package", "doctor"):
        item = commands.add_parser(name, add_help=False)
        item.add_argument("args", nargs=argparse.REMAINDER)
    for name in ("verify-db",):
        item = commands.add_parser(name)
        item.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    delegates = {"dashboard": ["-m", "my_strategy.web_dashboard.scripts.serve_dashboard"], "update-data": ["-m", "my_strategy.scripts.update_daily_data"], "package": ["-m", "my_strategy.scripts.package_project"], "doctor": ["-m", "my_strategy.scripts.docker_runtime_doctor"]}
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in delegates:
        return subprocess.call([sys.executable, *delegates[values[0]], *values[1:]], cwd=PROJECT_ROOT,
                               env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    args = build_parser().parse_args(values)
    try:
        if args.command in {'research-legacy-comparison','research-legacy-comparison-audit'}:
            from my_strategy.services import czsc_legacy_rule_ma_comparison as legacy
            if args.command == 'research-legacy-comparison-audit':
                result = legacy.audit_legacy_comparison(args.comparison_run_id, as_of=args.as_of)
            else:
                if not 1 <= args.cpu_workers <= 16:
                    raise ValueError('CPU只读准备进程数应在1到16之间')
                def legacy_progress(stage, current, total, failed=0):
                    print(f'{stage} {current}/{total} failures={failed}', file=sys.stderr)
                stocks = [stock.strip() for stock in args.stocks.split(',') if stock.strip()]
                result = legacy.run_legacy_comparison(args.anchor_run_id,
                    legacy_model_run_id=args.legacy_model_run_id or legacy.DEFAULT_MA_RUN,
                    symbols=stocks or None, cpu_workers=args.cpu_workers, progress=legacy_progress)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return int(result.get('artifact_integrity_passed') is False)
        if args.command == 'research-high-return-holding':
            from my_strategy.web_dashboard.api import ReplayHoldingRequest
            from my_strategy.web_dashboard.holding_evidence import holding_exit_research
            spec = ReplayHoldingRequest(replay_run_id=args.replay_run_id, window=args.window, arm=args.arm,
                symbol=args.symbol, as_of=args.as_of, usage_mode=args.usage_mode)
            print(json.dumps(holding_exit_research(**spec.model_dump(mode='json')), ensure_ascii=False, allow_nan=False))
            return 0
        if args.command in {"research-high-return-forward", "research-high-return-forward-source", "research-high-return-forward-status"}:
            if hasattr(args, "cpu_workers") and not 1 <= args.cpu_workers <= 16:
                raise ValueError("CPU只读准备进程数应在1到16之间")
            from my_strategy.services import czsc_high_return_forward as forward_service
            def forward_progress(stage, current, total, failed=0):
                print(f"{stage} {current}/{total} failures={failed}", file=sys.stderr)
            if args.command == "research-high-return-forward-status":
                result = forward_service.forward_status(args.model_run_id)
            elif args.command == "research-high-return-forward-source":
                result = forward_service.build_forward_source_snapshot(args.model_run_id,
                    end=args.end, calendar_run_id=args.calendar_run_id, cpu_workers=args.cpu_workers,
                    progress=forward_progress)
            else:
                result = forward_service.build_forward_evidence(args.model_run_id, args.source_training_run_id,
                    end=args.end, previous_run_id=args.previous_run_id, cpu_workers=args.cpu_workers,
                    progress=forward_progress)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "research-high-return-audit":
            from my_strategy.services.czsc_high_return_audit import audit_high_return_run
            result = audit_high_return_run(args.model_run_id, as_of=args.as_of)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return int(bool(result.get("error_count")))
        from my_strategy.adapters.czsc_adapter import data_status
        if args.command in {"doctor", "verify-db"}:
            import czsc
            status = data_status()
            print(json.dumps({"status": "ok", "czsc": czsc.__version__, "data": status}, ensure_ascii=False))
            return 0
        from my_strategy.web_dashboard.api import AnalysisRequest, TaskRequest
        from my_strategy.storage.czsc_results import ResultStore
        if args.command in {"research-promote", "research-rollback", "research-status", "research-high-return-status"}:
            from my_strategy.storage.czsc_model_releases import ModelReleaseStore
            from my_strategy.services.czsc_research import research_status
            store = ModelReleaseStore()
            if args.command == "research-high-return-status" or (args.command == "research-status" and args.model_family == "high_return"):
                from my_strategy.services.czsc_high_return_research import status_high_return
                result = status_high_return()
            elif args.command == "research-status":
                if args.model_family == "wyckoff":
                    from my_strategy.services.czsc_wyckoff_runtime import wyckoff_research_status
                    result = wyckoff_research_status()
                elif args.model_family == "dual":
                    from my_strategy.services.czsc_dual_runtime import dual_research_status
                    result = dual_research_status()
                else:
                    result = research_status()
            elif args.command == "research-promote":
                result = store.promote(args.model_run_id, args.model_fold, reason=args.reason)
            else:
                result = store.rollback(args.release_id, reason=args.reason)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "analyze":
            from my_strategy.services.czsc_analysis import analyze_stock
            from my_strategy.core.run_context import create_run_context
            if not args.symbol:
                raise ValueError("analyze 须指定 --symbol")
            spec = AnalysisRequest(symbol=args.symbol, start=args.start, end=args.end, as_of=args.as_of,
                                   model_family=args.model_family,
                                   research=args.research, usage_mode=args.usage_mode, model_policy=args.model_policy,
                                   entry_policy=args.entry_policy,
                                   model_run_id=args.model_run_id, model_fold=args.model_fold,
                                   calendar_run_id=args.calendar_run_id, device=args.device).model_dump(mode="json", exclude_none=True)
            if args.research and args.model_family == "high_return":
                from my_strategy.web_dashboard.high_return import analyze_high_return
                result = analyze_high_return(**spec)
            else:
                # Consensus belongs to chart/high-return routing, not native analysis.
                calculation_spec = {key: value for key, value in spec.items()
                                    if key not in {"consensus_config", "consensus_version"}}
                result = analyze_stock(**calculation_spec)
            context = create_run_context(task="czsc-analysis", as_of_date=result["data_end"], config=spec, data_version=str(result.get("data_version", result["data_end"])), stocks=[args.symbol], source="cli", start_date=args.start, end_date=result["data_end"])
            result.update(run_id=context.run_id, run_context=context.to_dict())
            ResultStore().save("analysis", result)
        else:
            from my_strategy.web_dashboard.tasks import TaskManager
            stocks = ([args.symbol] if getattr(args, "symbol", None) else []) + [stock.strip() for stock in args.stocks.split(",") if stock.strip()]
            task_spec = {"symbols": stocks, "start": getattr(args, "start", None), "end": args.end, "initial_cash": getattr(args, "initial_cash", 100000)}
            task_spec.update({key: getattr(args, key) for key in ("device", "cpu_workers", "batch_size") if getattr(args, key, None) is not None})
            task_spec.update({key: getattr(args, key) for key in ("research", "model_family", "source_training_run_id", "model_run_id", "model_fold", "calendar_run_id", "usage_mode", "model_policy", "entry_policy") if getattr(args, key, None) is not None})
            if args.command == "research-train" and not task_spec.get("calendar_run_id"):
                from my_strategy.services.czsc_research import research_status
                task_spec["calendar_run_id"] = research_status()["calendar_run_id"]
            if getattr(args, "config", None):
                config = json.loads(args.config.read_text(encoding="utf-8"))
                if not isinstance(config, dict):
                    raise ValueError("高收益研究配置应为JSON对象")
                task_spec["research_config"] = config
            high_return_tasks = {"research-points-build": "high_return_points_build", "research-episodes-build": "high_return_episodes_build", "research-high-return-train": "high_return_train", "research-high-return-backtest": "high_return_backtest"}
            request = TaskRequest(kind=high_return_tasks.get(args.command) or ("wyckoff_research_train" if args.command == "research-wyckoff-train" else "dual_research_train" if args.command == "research-dual-train" else "research_train" if args.command == "research-train" else "scan" if args.command == "scan" else "backtest"), spec=task_spec)
            manager = TaskManager(workers=1)
            try:
                spec = request.spec.model_dump(mode="json", exclude_none=True)
                spec["source"] = "cli"
                job = manager.submit(request.kind, spec)
                last = None
                while job["status"] in {"queued", "running", "cancelling"}:
                    detail = job["progress_detail"]
                    current = (detail.get("stage"), detail.get("current"))
                    if current != last:
                        print(f'{detail.get("stage")} {detail.get("current", 0)}/{detail.get("total", 0)} failures={detail.get("failed", 0)}', file=sys.stderr)
                        last = current
                    time.sleep(.2)
                    job = manager.get(job["job_id"])
                if job["status"] != "succeeded":
                    raise RuntimeError(job["error"])
                result = job["result"]
            except KeyboardInterrupt:
                if "job" in locals():
                    manager.cancel(job["job_id"])
                return 130
            finally:
                manager.close()
        print(json.dumps(result if args.json else {key: result[key] for key in ("run_id", "symbol", "data_end", "coverage", "metrics") if key in result}, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
