"""Portable CLI for the current CZSC trend-consensus workbench."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.runtime_env import configure_utf8_stdio


DELEGATES = {
    "dashboard": ["-m", "my_strategy.web_dashboard.scripts.serve_dashboard"],
    "update-data": ["-m", "my_strategy.scripts.update_daily_data"],
    "package": ["-m", "my_strategy.scripts.package_project"],
    "doctor": ["-m", "my_strategy.scripts.docker_runtime_doctor"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KHQuant 当前趋势共识策略")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "scan"):
        item = commands.add_parser(name)
        item.add_argument("--symbol", help="单股代码，例如 600519.SH")
        item.add_argument("--stocks", default="", help="逗号分隔代码；扫描/批量留空表示本地全部股票")
        item.add_argument("--start", default="2026-01-01")
        item.add_argument("--end", help="截止日期；留空使用本地最新行情")
        item.add_argument("--as-of", help="分析回放截止日或 Asia/Shanghai 时间")
        item.add_argument("--current-position", type=float, help="仅分析的当前持仓百分比，0到100，不生成资金账户")
        item.add_argument("--device", help="计算设备：cpu、mps、cuda 或 cuda:编号")
        item.add_argument("--usage-mode", choices=("forward", "historical"), default="historical",
                          help="historical 历史研究；forward 按实际训练完成时点的当前研究")
        item.add_argument("--model-run-id", help="显式核验当前模型运行；留空使用生效选择或观察模式")
        item.add_argument("--model-fold", help="显式固定检查点名称")
        item.add_argument("--calendar-run-id", help="已核验交易日历运行")
        item.add_argument("--config", type=Path, help="当前共识参数 JSON 对象")
        item.add_argument("--cpu-workers", type=int, default=4)
        item.add_argument("--json", action="store_true")
    train = commands.add_parser("train-trend", help="从冻结原始输入完整训练当前模型依赖")
    train.add_argument("--raw-source-run-id")
    train.add_argument("--source-native-run-id")
    train.add_argument("--calendar-run-id")
    train.add_argument("--end", required=True)
    train.add_argument("--symbols", nargs="+")
    train.add_argument("--stocks", default="", help="逗号分隔代码；与 --symbols 二选一")
    train.add_argument("--cohort-file", type=Path)
    train.add_argument("--stock-batch-size", type=int, help="仅准备批大小；完整训练全部历史合格股票")
    train.add_argument("--cpu-workers", type=int)
    train.add_argument("--device", choices=("cpu", "mps", "cuda"))
    train.add_argument("--config", type=Path)
    train.add_argument("--freeze-raw-only", action="store_true")
    train.add_argument("--prepare-source-only", action="store_true")
    diagnose = commands.add_parser("model-diagnostics", help="只读核验候选、成熟标签、费用后收益及样本流失")
    diagnose.add_argument("--model-run-id", required=True)
    diagnose.add_argument("--cpu-workers", type=int, default=2)
    diagnose.add_argument("--json", action="store_true")
    for name in DELEGATES:
        item = commands.add_parser(name, add_help=False)
        item.add_argument("args", nargs=argparse.REMAINDER)
    item = commands.add_parser("verify-db", help="只读报告本地行情状态")
    item.add_argument("--json", action="store_true")
    return parser


def _subprocess(arguments: list[str]) -> int:
    return subprocess.call([sys.executable, *arguments], cwd=PROJECT_ROOT,
        env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))


def _train_arguments(args) -> list[str]:
    if args.symbols and args.stocks:
        raise ValueError("--symbols 与 --stocks 不能同时指定")
    values = ["-m", "my_strategy.scripts.train_trend", "--end", args.end]
    for key in ("raw_source_run_id", "source_native_run_id", "calendar_run_id", "stock_batch_size",
                "cohort_file", "cpu_workers", "device", "config"):
        value = getattr(args, key)
        if value is not None:
            values.extend(["--" + key.replace("_", "-"), str(value)])
    symbols = args.symbols or [s.strip() for s in args.stocks.split(",") if s.strip()]
    if symbols:
        values.extend(["--symbols", *symbols])
    for key in ("freeze_raw_only", "prepare_source_only"):
        if getattr(args, key):
            values.append("--" + key.replace("_", "-"))
    return values


def _progress(stage, current, total, failed=0):
    print(f"{stage} {current}/{total} failures={failed}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in DELEGATES:
        return _subprocess([*DELEGATES[values[0]], *values[1:]])
    args = build_parser().parse_args(values)
    try:
        if args.command == "train-trend":
            return _subprocess(_train_arguments(args))
        if args.command == "model-diagnostics":
            if not 1 <= args.cpu_workers <= 4:
                raise ValueError("CPU 只读诊断进程数应在 1 到 4 之间")
            from my_strategy.services.czsc_high_return_sample_diagnostics import run_diagnostics
            result = run_diagnostics(args.model_run_id, cpu_workers=args.cpu_workers, progress=_progress)
        elif args.command == "verify-db":
            from my_strategy.adapters.czsc_adapter import data_status, engine_contract
            result = {"status": "ok", "engine": engine_contract(), "data": data_status()}
        else:
            if not 1 <= args.cpu_workers <= 16:
                raise ValueError("CPU 只读准备进程数应在 1 到 16 之间")
            if args.current_position is not None and (args.command != 'analyze' or not 0 <= args.current_position <= 100):
                raise ValueError("当前持仓百分比仅分析可用，且必须在0到100之间")
            if args.command == "analyze" and not args.symbol:
                raise ValueError("analyze 须指定 --symbol")
            if args.as_of and args.command != "analyze":
                raise ValueError("--as-of 仅用于 analyze；扫描使用 --end")
            from my_strategy.web_dashboard import trend_strategy
            consensus = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
            if consensus is not None and not isinstance(consensus, dict):
                raise ValueError("当前共识参数应为 JSON 对象")
            spec = {"start": args.start, "end": args.end, "model_run_id": args.model_run_id,
                    "model_fold": args.model_fold, "usage_mode": args.usage_mode,
                    "calendar_run_id": args.calendar_run_id, "device": args.device or "cpu",
                    "cpu_workers": args.cpu_workers, "consensus_config": consensus, "source": "cli"}
            if args.command == "analyze":
                from my_strategy.core.run_context import create_run_context
                from my_strategy.storage.czsc_results import ResultStore
                spec.update(symbol=args.symbol, as_of=args.as_of)
                if args.current_position is not None:
                    spec["current_position_fraction"] = args.current_position / 100
                result = trend_strategy.analyze_trend_consensus(**spec)
                context = create_run_context(task="trend-consensus-analysis", as_of_date=result["data_end"],
                    config=spec, data_version=str(result.get("data_version", result["data_end"])),
                    stocks=[args.symbol], source="cli", start_date=args.start, end_date=result["data_end"])
                result.update(run_id=context.run_id, run_context=context.to_dict())
                ResultStore().save("analysis", result)
            else:
                stocks = ([args.symbol] if args.symbol else []) + [s.strip() for s in args.stocks.split(",") if s.strip()]
                spec.update(symbols=list(dict.fromkeys(stocks)) or None, progress=_progress)
                result = trend_strategy.scan_trend_consensus(**spec)
        from my_strategy.services.czsc_trend_support import _json
        output = result if getattr(args, "json", False) or args.command == "model-diagnostics" else {
            k: result[k] for k in ("status", "run_id", "symbol", "data_end", "coverage", "metrics") if k in result}
        print(json.dumps(_json(output), ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
