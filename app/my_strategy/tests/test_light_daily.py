"""Daily lightweight orchestration with synthetic services and no market writes."""
from __future__ import annotations

from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from my_strategy.services import light_daily


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(light_daily, "latest_market_date", lambda **_kw: "2026-10-14")
    settings = {}
    store = SimpleNamespace(get_setting=lambda name, default=None: settings.get(name, default),
                            set_setting=lambda name, value: settings.update({name: value}))
    instance = SimpleNamespace(store=store, settings=settings, close=Mock(),
        source=Mock(return_value=(None, None, [], {"source": "actual-fixture"})),
        prediction=Mock(return_value={"cache_key": "forecast", "reused": True,
                                      "quality": {"prediction_usable": True}}),
        continue_accounts=Mock(return_value=[]),
        train_score=Mock(side_effect=AssertionError("daily update must not train")),
        tune=Mock(side_effect=AssertionError("daily update must not tune")),
        backtest=Mock(side_effect=AssertionError("daily update must not replay history")))
    return instance


def test_default_post_update_route_is_lightweight_and_never_starts_full_producer(monkeypatch, tmp_path):
    from my_strategy.scripts import update_daily_data as updater
    from my_strategy.services import daily_feature_sync
    calls = []
    monkeypatch.setattr(updater, "get_data_access", lambda: SimpleNamespace(settings=SimpleNamespace(raw_db=tmp_path / "raw.db")))
    monkeypatch.setattr(light_daily, "sync_light_daily", lambda **kwargs: calls.append(kwargs) or {"status": "completed", "training_started": False})
    monkeypatch.setattr(daily_feature_sync, "sync_daily_features", lambda **_kw: pytest.fail("default daily update restarted full native work"))
    result = updater._sync_after_update(SimpleNamespace(skip_feature_sync=False),
        {"total": 1, "failed": 0, "target_date_covered": 1, "end_date": "2026-10-14"}, {})
    assert result["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["required_price_date"] == "2026-10-14"
    assert not result["training_started"]


@pytest.mark.parametrize("summary,quality", [
    ({"failed": 1, "total": 1, "target_date_covered": 1}, {}),
    ({"failed": 0, "total": 1, "target_date_covered": 1}, {"has_errors": True}),
    ({"failed": 0, "total": 1, "target_date_covered": 1}, {"error": "quality failed"}),
    ({"failed": 0, "total": 1, "target_date_covered": 0}, {}),
])
def test_failed_or_uncovered_raw_update_cannot_start_derived_work(monkeypatch, summary, quality):
    from my_strategy.scripts import update_daily_data as updater
    monkeypatch.setattr(light_daily, "sync_light_daily", lambda **_kw: pytest.fail("invalid raw update started derived work"))
    result = updater._sync_after_update(SimpleNamespace(skip_feature_sync=False), summary, quality)
    assert result["status"] in {"blocked", "skipped"}
    assert not result["training_started"]


def test_all_accounts_are_processed_before_bounded_deduplicated_watch_previews(service):
    order = []
    service.continue_accounts.side_effect = lambda *_a, **_kw: order.append("accounts") or [
        {"account_id": str(i), "status": "completed"} for i in range(25)]
    service.source.side_effect = lambda symbol, *_a, **_kw: order.append(symbol) or (None, None, [], {"symbol": symbol})
    result = light_daily.sync_light_daily(symbols=["002436.SZ", "002436.SZ", "600519.SH", "000001.SZ"],
                                          max_symbols=1, service=service)
    assert order == ["accounts", "002436.SZ"]
    assert len(result["accounts"]) == 25
    assert result["requested"] == 3
    assert (result["succeeded"], result["failed"], result["deferred"]) == (1, 0, 2)
    assert result["requested"] == result["succeeded"] + result["failed"] + result["deferred"]
    assert result["status"] == "partial"
    assert not result["training_started"] and not result["full_market_native_started"]
    assert not result["cancelled_tasks_resumed"] and not result["board_full_published"]
    assert service.source.call_args.kwargs["stop"] is service.prediction.call_args.kwargs["stop"]


def test_prediction_budget_does_not_truncate_account_risk_population(service, monkeypatch):
    now = [100.]
    monkeypatch.setattr(light_daily.time, "monotonic", lambda: now[0])
    def accounts(*_a, **kw):
        assert kw["deadline"] == 101.
        now[0] = 102.
        return [{"account_id": "held", "status": "completed"}]
    service.continue_accounts.side_effect = accounts
    result = light_daily.sync_light_daily(symbols=["002436.SZ", "600519.SH"], budget_seconds=1, service=service)
    assert result["accounts"] == [{"account_id": "held", "status": "completed"}]
    assert result["requested"] == result["deferred"] == 2
    service.source.assert_not_called()
    service.prediction.assert_not_called()


def test_single_stock_failure_remains_visible_and_does_not_stop_other_stocks(service):
    service.prediction.side_effect = [ValueError("missing verified target close"),
                                     {"cache_key": "valid", "quality": {"prediction_usable": True}}]
    result = light_daily.sync_light_daily(symbols=["002436.SZ", "600519.SH"], service=service)
    assert result["status"] == "partial"
    assert (result["requested"], result["succeeded"], result["failed"], result["deferred"]) == (2, 1, 1, 0)
    assert "missing verified" in result["rows"][0]["error"]
    assert result["rows"][1]["symbol"] == "600519.SH"


def test_cancellation_from_source_reaches_same_stop_and_never_starts_prediction(service):
    stop = threading.Event()
    def source(*_a, **kwargs):
        assert not kwargs["stop"].is_set()
        stop.set()
        assert kwargs["stop"].is_set()
        return None, None, [], {}
    service.source.side_effect = source
    result = light_daily.sync_light_daily(symbols=["002436.SZ", "600519.SH"], service=service, check_cancel=stop.is_set)
    assert result["status"] == "cancelled"
    assert result["requested"] == result["deferred"] == 2
    assert all(row["status"] == "cancelled" for row in result["rows"])
    service.prediction.assert_not_called()


def test_stale_requested_date_is_blocked_before_creating_any_service(service, monkeypatch):
    from my_strategy.services import light_research
    monkeypatch.setattr(light_research, "LightResearch", lambda **_kw: pytest.fail("stale request created a service"))
    result = light_daily.sync_light_daily(required_price_date="2026-10-13", symbols=[])
    assert result["status"] == "blocked"
    assert result["price_date"] == "2026-10-13"


def test_cancellation_before_daily_start_cannot_touch_accounts_or_forecasts(service):
    result = light_daily.sync_light_daily(symbols=["002436.SZ"], service=service, check_cancel=lambda: True)
    assert result["status"] == "cancelled"
    assert result["deferred"] == 1
    service.continue_accounts.assert_not_called()
    service.source.assert_not_called()
    service.prediction.assert_not_called()


def test_account_failure_cannot_leave_successful_last_daily_status(service):
    service.continue_accounts.side_effect = RuntimeError("compute lock is busy")
    try:
        result = light_daily.sync_light_daily(symbols=[], service=service)
        assert result["status"] in {"failed", "blocked", "busy"}
    except RuntimeError:
        pass
    assert service.settings["last_daily"]["status"] in {"failed", "blocked", "busy"}
