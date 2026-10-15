from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import kronos_assets as assets
from my_strategy.services import kronos_prediction as forecasting


def bars(n=30):
    close = np.linspace(20, 25, n)
    frame = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n),
                          "open": close - .2, "high": close + .8, "low": close - .8,
                          "close": close, "volume": np.linspace(1e5, 2e5, n),
                          "amount": close * 1e5, "has_trade_price": 1, "symbol": "000001.SZ",
                          "source": "sina_unadjusted_v1"})
    frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
    return frame


def test_input_truncation_precedes_normalization_and_identity():
    frame = bars()
    end = frame.date.iloc[20].date().isoformat()
    future = frame.date.iloc[21:26].tolist()
    first = forecasting.prepare_input(frame, future, symbol="000001.SZ", as_of=end, lookback=16)
    frame.loc[21:, "close"] = 99999
    second = forecasting.prepare_input(frame, future, symbol="000001.SZ", as_of=end, lookback=16)
    assert first[2] == second[2]
    assert first[0].close.tolist() == second[0].close.tolist()


@pytest.mark.parametrize("mutation,error", [
    (lambda f: f.drop(columns="amount"), "Actual OHLC"),
    (lambda f: f.assign(high=f.low - 1), "geometry"),
    (lambda f: f.assign(volume=np.nan), "finite"),
])
def test_invalid_inputs_fail_without_model_load(mutation, error):
    frame = bars()
    with pytest.raises(ValueError, match=error):
        forecasting.prepare_input(mutation(frame), pd.bdate_range("2026-03-01", periods=5),
                                  symbol="000001.SZ", as_of=str(frame.date.iloc[-1].date()), lookback=16)


def test_no_stale_date_or_intraday_close_substitution():
    frame = bars()
    future = pd.bdate_range("2026-03-01", periods=5)
    with pytest.raises(ValueError, match="No actual"):
        forecasting.prepare_input(frame, future, symbol="000001.SZ", as_of="2026-02-28", lookback=16)
    with pytest.raises(ValueError, match="after 15"):
        forecasting.prepare_input(frame, future, symbol="000001.SZ", as_of="2026-02-11T10:00:00+08:00", lookback=16)


def test_paths_are_filtered_as_whole_without_direction_bias_or_repair():
    data = np.tile([10., 12., 8., 11., 100., 1000.], (10, 5, 1))
    data[1, 2, 1] = 2  # invalid high; the whole path must be excluded
    data[3, 3, 4] = -1
    original = data.copy()
    result = forecasting.summarize_samples(data, list("abcde"), 10)
    assert result["quality"]["valid_samples"] == 8
    assert result["quality"]["prediction_usable"]
    assert result["summary"]["up_path_fraction"] == 1
    np.testing.assert_array_equal(data, original)
    data[4, 0, 3] = np.nan
    assert not forecasting.summarize_samples(data, list("abcde"), 10)["quality"]["prediction_usable"]
    assert not forecasting.summarize_samples(data[:7], list("abcde"), 10)["quality"]["prediction_usable"]


def test_distribution_uses_individual_paths_not_the_mean_curve():
    data = np.tile([10., 12., 8., 11., 100., 1000.], (8, 5, 1))
    data[:4, :, 3] = 9
    result = forecasting.summarize_samples(data, list("abcde"), 10)
    assert result["summary"]["up_path_fraction"] == .5
    assert result["forecast"][-1]["close_p10"] == 9
    assert result["forecast"][-1]["close_p90"] == 11


def test_asset_hash_rechecked_when_bytes_change_and_no_implicit_download(tmp_path, monkeypatch):
    raw = b'{"value": 1}'
    (tmp_path / "config.json").write_bytes(raw)
    monkeypatch.setattr(assets, "PINNED_ASSETS", ({"path": "config.json", "size": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest(), "repo": "test", "revision": "frozen"},))
    assert assets.asset_status(tmp_path)["ready"]
    (tmp_path / "config.json").write_bytes(b'{"value": 2}')
    assert not assets.asset_status(tmp_path)["ready"]
    with pytest.raises(ValueError, match="SHA"):
        assets.prepare_assets(tmp_path)


def test_invalid_asset_rejects_prediction_before_reusing_an_already_loaded_model(tmp_path, monkeypatch):
    raw = b'{"value": 1}'
    path = tmp_path / 'config.json'
    path.write_bytes(raw)
    monkeypatch.setattr(assets, 'PINNED_ASSETS', ({'path': 'config.json', 'size': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(), 'repo': 'test', 'revision': 'frozen'},))
    assert assets.asset_status(tmp_path)['ready']
    monkeypatch.setattr(assets, '_stat_identity', lambda _: ('same file metadata',))
    path.write_bytes(b'{"value": 2}')
    monkeypatch.setattr(forecasting, '_SINGLETON', {'key': 'previously-loaded', 'model': object()})
    monkeypatch.setattr(forecasting, '_load_model', lambda *_: pytest.fail('Invalid asset reused the loaded predictor'))
    frame = bars()
    future = pd.bdate_range(frame.date.iloc[-1] + pd.Timedelta(days=1), periods=5)
    with pytest.raises(FileNotFoundError, match='Kronos local assets unavailable'):
        forecasting.predict(frame, future, symbol='000001.SZ', as_of=str(frame.date.iloc[-1].date()),
                            lookback=16, device='cpu', asset_root=tmp_path)


def test_predict_preserves_samples_and_has_no_production_qualification(monkeypatch, tmp_path):
    frame = bars()
    as_of = str(frame.date.iloc[-1].date())
    future = pd.bdate_range(frame.date.iloc[-1] + pd.Timedelta(days=1), periods=5)
    def inference(tokenizer, model, x, *_args, sample_count, **kwargs):
        return np.repeat(x.cpu().numpy()[:, None, :, :], sample_count, axis=1)
    monkeypatch.setattr(forecasting, "asset_status", lambda _: {"ready": True, "files": []})
    monkeypatch.setattr(forecasting, "_source_identity", lambda: {"vendor": "fixture"})
    monkeypatch.setattr(forecasting, "_load_model", lambda *_: (None, None, inference, True))
    result = forecasting.predict(frame, future, symbol="000001.SZ", as_of=as_of, lookback=16,
                                 device="cpu", asset_root=tmp_path, calendar_verified=True,
                                 calendar_identity="test_calendar")
    assert np.asarray(result["samples"]).shape == (8, 5, 6)
    assert result["quality"]["prediction_usable"]
    assert not result["qualification"]["positive_allocation_eligible"]
    json.dumps(result, allow_nan=False)
    assert list(tmp_path.iterdir()) == []
    no_calendar = forecasting.predict(frame, future, symbol="000001.SZ", as_of=as_of,
                                      lookback=16, device="cpu", asset_root=tmp_path)
    assert not no_calendar["quality"]["prediction_usable"]
    assert no_calendar["cache_key"] != result["cache_key"]


def test_cancel_prevents_model_loading(monkeypatch):
    frame = bars()
    monkeypatch.setattr(forecasting, "asset_status", lambda _: {"ready": True, "files": []})
    monkeypatch.setattr(forecasting, "_source_identity", lambda: {})
    monkeypatch.setattr(forecasting, "_load_model", lambda *_: pytest.fail("cancelled work loaded a model"))
    def cancelled():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        forecasting.predict(frame, pd.bdate_range("2026-03-01", periods=5), symbol="000001.SZ",
                            as_of=str(frame.date.iloc[-1].date()), lookback=16, device="cpu", cancel_check=cancelled)
    with pytest.raises(InterruptedError):
        forecasting.predict(frame, pd.bdate_range("2026-03-01", periods=5), symbol="000001.SZ",
                            as_of=str(frame.date.iloc[-1].date()), lookback=16, device="cpu", cancel_check=lambda: True)


@pytest.mark.parametrize("mutation", [
    lambda frame: frame.assign(has_trade_price=0),
    lambda frame: frame.assign(source="unverified_feed"),
    lambda frame: frame.drop(columns="has_trade_price"),
])
def test_valid_generated_paths_cannot_certify_unverified_actual_prices(monkeypatch, tmp_path, mutation):
    original = bars()
    frame = mutation(original)
    frame.attrs.update(original.attrs)
    def inference(_tokenizer, _model, x, *_args, sample_count, **_kwargs):
        return np.repeat(x.cpu().numpy()[:, None, :, :], sample_count, axis=1)
    monkeypatch.setattr(forecasting, "asset_status", lambda _: {"ready": True, "files": []})
    monkeypatch.setattr(forecasting, "_source_identity", lambda: {"vendor": "fixture"})
    monkeypatch.setattr(forecasting, "_load_model", lambda *_: (None, None, inference, True))
    result = forecasting.predict(frame, pd.bdate_range("2026-03-01", periods=5),
                                 symbol="000001.SZ", as_of=str(frame.date.iloc[-1].date()),
                                 lookback=16, device="cpu", asset_root=tmp_path,
                                 calendar_verified=True, calendar_identity="test-calendar")
    assert result["quality"]["valid_samples"] == 8
    assert not result["quality"]["input_contract_verified"]
    assert not result["quality"]["prediction_usable"]
