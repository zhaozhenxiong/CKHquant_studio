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


def test_train_cli_preflights_reference_source_and_effective_parameters_without_training(monkeypatch, tmp_path, capsys):
    from my_strategy.services import model_training_center
    recorded = []
    class Center:
        def preflight(self, parameters, **kwargs):
            recorded.append((parameters, kwargs))
            return {"valid": True, "plan": {"plan_id": "frozen-reference-plan"}}
        def run_training_job(self, *args, **kwargs):
            pytest.fail("preflight-only launched training")
    monkeypatch.setattr(model_training_center, "TrainingCenter", Center)
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: pytest.fail("reference CLI delegated legacy training"))
    config = tmp_path / "reference.json"
    config.write_text(json.dumps({"epochs": 3, "batch_size": 2048}))
    assert cli.main(["train-reference", "--end", "2026-10-12", "--source-native-run-id", "immutable-native",
                     "--config", str(config), "--device", "mps", "--preflight-only", "--json"]) == 0
    parameters, context = recorded[0]
    assert parameters == {"epochs": 3, "batch_size": 2048, "end": "2026-10-12", "device": "mps"}
    assert context["source_native_run_id"] == "immutable-native" and callable(context["progress"])
    assert json.loads(capsys.readouterr().out)["plan"]["plan_id"] == "frozen-reference-plan"


def test_old_symbol_training_arguments_cannot_launch_retired_training(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: pytest.fail("retired plan launched training"))
    assert cli.main(["train-trend", "--end", "2026-10-12", "--symbols", "600519.SH", "--stocks", "000001.SZ"]) == 1
    assert "旧训练入口已退役" in capsys.readouterr().err


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
