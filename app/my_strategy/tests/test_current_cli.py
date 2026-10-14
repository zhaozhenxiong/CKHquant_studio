"""Current CLI rejects retired identities and reaches the shared actual entrypoints."""
import json
import subprocess

import pytest

from my_strategy import cli


@pytest.mark.parametrize("command", ["research-train", "research-dual-train", "research-wyckoff-train", "research-high-return-train", "research-promote", "research-high-return-diagnostics", "backtest", "backtest-batch"])
def test_retired_commands_are_not_aliases_for_current_identity(command):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([command])
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["analyze", "scan"])
def test_current_entry_rejects_old_model_family_option(name):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([name, "--model-family", "high_return"])
    assert error.value.code == 2
    args = cli.build_parser().parse_args([name])
    assert args.model_run_id is None and not hasattr(args,"initial_cash")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([name,"--initial-cash","100000"])


def test_train_cli_delegates_frozen_input_and_effective_parameters_without_training(monkeypatch):
    recorded = []
    monkeypatch.setattr(subprocess, "call", lambda args, **kwargs: recorded.append((args, kwargs)) or 0)
    assert cli.main(["train-trend", "--end", "2026-10-12", "--raw-source-run-id", "immutable-raw", "--stocks", "600519.SH,000001.SZ",
                     "--config", "/tmp/current.json", "--stock-batch-size", "360", "--prepare-source-only"]) == 0
    arguments, context = recorded[0]
    assert arguments[1:3] == ["-m", "my_strategy.scripts.train_trend"]
    assert arguments[arguments.index("--symbols") + 1:-1] == ["600519.SH", "000001.SZ"]
    assert arguments[arguments.index("--stock-batch-size") + 1] == "360"
    assert "--cohort-size" not in arguments and "--capacity-tiers" not in arguments
    assert arguments[arguments.index("--raw-source-run-id") + 1] == "immutable-raw"
    assert arguments[-1] == "--prepare-source-only"
    assert "--device" not in arguments and "--cpu-workers" not in arguments
    assert context["env"]["PYTHONIOENCODING"] == "utf-8"


def test_two_symbol_inputs_fail_before_training_process(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: pytest.fail("ambiguous plan launched training"))
    assert cli.main(["train-trend", "--end", "2026-10-12", "--symbols", "600519.SH", "--stocks", "000001.SZ"]) == 1
    assert "不能同时" in capsys.readouterr().err


def test_scan_uses_current_shared_entry_without_changing_model_selection(monkeypatch, capsys):
    from my_strategy.web_dashboard import trend_strategy
    seen = []
    monkeypatch.setattr(trend_strategy, "scan_trend_consensus", lambda **kwargs: seen.append(kwargs) or {"run_id": "shared-current", "data_end": "2026-10-12"})
    assert cli.main(["scan", "--stocks", "600519.SH,600519.SH,000001.SZ", "--end", "2026-10-12", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "shared-current"
    assert seen[0]["symbols"] == ["600519.SH", "000001.SZ"]
    assert seen[0]["model_run_id"] is None and "model_family" not in seen[0]


def test_analysis_passes_current_head_and_optional_position_without_cash_account(monkeypatch,capsys):
    from my_strategy.web_dashboard import trend_strategy
    seen = []
    monkeypatch.setattr(trend_strategy, "analyze_trend_consensus", lambda **kwargs: seen.append(kwargs) or {"symbol":"600519.SH","data_end":"2026-10-12"})
    assert cli.main(["analyze", "--symbol", "600519.SH", "--model-run-id", "new-current-run", "--model-fold","trend_2026Q3","--current-position","25","--json"]) == 0
    assert seen[0]["model_run_id"] == "new-current-run" and seen[0]["symbol"] == "600519.SH"
    assert seen[0]["model_fold"]=="trend_2026Q3" and seen[0]["current_position_fraction"]==.25
    assert "initial_cash" not in seen[0]
    assert json.loads(capsys.readouterr().out)["symbol"]=="600519.SH"


def test_diagnostic_cli_is_read_only_and_bounds_worker_count(monkeypatch, capsys):
    from my_strategy.services import czsc_high_return_sample_diagnostics as diagnostic
    calls = []
    monkeypatch.setattr(diagnostic, "run_diagnostics", lambda run_id, **kwargs: calls.append((run_id, kwargs)) or {"publication_allowed": False})
    assert cli.main(["model-diagnostics", "--model-run-id", "new-current-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"publication_allowed": False}
    assert calls[0][0] == "new-current-run" and calls[0][1]["cpu_workers"] == 2
    assert cli.main(["model-diagnostics", "--model-run-id", "new-current-run", "--cpu-workers", "5"]) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["analyze", "scan"])
def test_current_cli_rejects_retired_public_production_label(name):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([name, "--usage-mode", "production"])
    assert error.value.code == 2
    assert cli.build_parser().parse_args([name, "--usage-mode", "forward"]).usage_mode == "forward"
