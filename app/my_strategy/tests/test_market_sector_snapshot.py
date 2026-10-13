"""Public classification is source-backed, bounded, and cannot regress silently."""
import copy
import json

import pytest
import requests

from my_strategy.scripts.refresh_market_sector_snapshot import (VERSION, SinaIndustryProvider,
    publish_snapshot, refresh_snapshot, validate_snapshot)

POOL = ["000001.SZ", "600000.SH", "920123.BJ"]


def snapshot(*, complete=True, members=None):
    members = members if members is not None else POOL
    return {"version": VERSION, "kind": "industry", "source": "sina:public_shenwan_l1",
        "source_url": "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodes",
        "observed_at": "2026-10-12T20:00:00+08:00", "complete": complete,
        "requested_boards": 1 if complete else 2,
        "boards": [{"id": "sw1_480000", "name": "银行", "members": list(members)}],
        "failures": [] if complete else [{"id": "sw1_770000", "name": "其他", "error": "timeout"}]}


def test_complete_snapshot_is_archived_and_partial_never_replaces_it(tmp_path):
    destination = tmp_path / "market_sector_snapshot.json"
    first = publish_snapshot(snapshot(), destination=destination, universe=POOL)
    assert first["activation"]["state"] == "activated"
    before = destination.read_bytes()
    partial = publish_snapshot(snapshot(complete=False), destination=destination, universe=POOL)
    assert partial["activation"]["state"] == "retained_existing"
    assert "不能覆盖" in partial["activation"]["reason"]
    assert destination.read_bytes() == before
    assert len(list((tmp_path / "market_sector_snapshots").glob("*.json"))) == 2
    assert len(list((tmp_path / "market_sector_snapshot_attempts").glob("*.json"))) == 2
    assert json.loads((tmp_path / "market_sector_snapshot_status.json").read_text())["state"] == "retained_existing"


def test_insufficient_coverage_is_archived_without_activating_or_erasing_existing(tmp_path):
    destination = tmp_path / "market_sector_snapshot.json"
    result = publish_snapshot(snapshot(members=POOL[:1]), destination=destination, universe=POOL)
    assert result["quality"]["coverage"] == pytest.approx(1 / 3)
    assert result["activation"]["state"] == "not_activated"
    assert not destination.exists()
    assert result["quality"]["unknown_symbols"] == 2
    no_pool = publish_snapshot(snapshot(), destination=destination)
    assert no_pool["activation"]["state"] == "not_activated"
    assert "股票池" in no_pool["activation"]["reason"]


def test_complete_but_lower_coverage_cannot_regress_current_snapshot(tmp_path):
    destination = tmp_path / "market_sector_snapshot.json"
    publish_snapshot(snapshot(), destination=destination, universe=POOL)
    before = destination.read_bytes()
    result = publish_snapshot(snapshot(members=POOL[:2]), destination=destination,
        universe=POOL, min_coverage=.5)
    assert result["activation"]["state"] == "retained_existing"
    assert "下降" in result["activation"]["reason"]
    assert destination.read_bytes() == before


def test_duplicate_or_conflicting_members_are_never_counted_as_unique_coverage():
    value = snapshot()
    value["boards"].append({"id": "sw1_770000", "name": "其他", "members": [POOL[0]]})
    value["requested_boards"] = 2
    quality = validate_snapshot(value, universe=POOL)
    assert quality["conflicting_members"] == 1
    assert quality["covered_symbols"] == 2
    value["boards"][0]["members"].append(POOL[0])
    with pytest.raises(ValueError, match="重复"):
        validate_snapshot(value, universe=POOL)


@pytest.mark.parametrize("change", [
    {"complete": False}, {"requested_boards": 2}, {"observed_at": "2026-10-12"},
    {"source_url": "http://unverified.invalid"}, {"source": ""}, {"kind": "concept"},
])
def test_invalid_source_contract_is_rejected(change):
    value = snapshot(); value.update(change)
    with pytest.raises(ValueError):
        validate_snapshot(value, universe=POOL)


def test_import_preserves_source_and_time_and_detects_changed_content(tmp_path):
    destination = tmp_path / "market_sector_snapshot.json"
    first = publish_snapshot(snapshot(), destination=destination, universe=POOL)
    stored = json.loads(destination.read_text())
    second = publish_snapshot(stored, destination=destination, universe=POOL)
    assert second["source"] == first["source"]
    assert second["observed_at"] == "2026-10-12T20:00:00+08:00"
    stored["boards"][0]["name"] = "被改写"
    with pytest.raises(ValueError, match="哈希"):
        publish_snapshot(stored, destination=destination, universe=POOL)


def test_failed_source_has_bounded_requests_and_records_skipped_boards(tmp_path):
    calls = []
    class FailedProvider:
        def catalog(self):
            return [{"code": str(i), "name": str(i), "kind": "industry"} for i in range(10)]
        def members(self, code):
            calls.append(code)
            raise requests.ConnectionError("public endpoint unavailable")
        def close(self):
            pass
    result = refresh_snapshot(destination=tmp_path / "market_sector_snapshot.json", universe=POOL,
        workers=1, max_failures=2, provider_factory=FailedProvider)
    assert len(calls) == 2
    assert len(result["failures"]) == 10
    assert result["quality"]["coverage"] == 0
    assert result["activation"]["state"] == "not_activated"
    assert sum("未继续请求" in row["error"] for row in result["failures"]) == 8


def test_sina_validates_page_count_and_keeps_beijing_stock_symbols(monkeypatch):
    provider = SinaIndustryProvider(direct=True)
    records = [{"symbol": "sz000001", "name": "A"}, {"symbol": "bj920123", "name": "B"}]
    monkeypatch.setattr(provider, "_get", lambda method, **params: "2" if method.endswith("Count") else records)
    members, excluded = provider.members("sw1_480000")
    assert {row["stock"] for row in members} == {"000001.SZ", "920123.BJ"}
    assert excluded == 0
    records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match="不完整"):
        provider.members("sw1_480000")
    provider.close()


def test_access_refusal_is_not_retried_or_bypassed(monkeypatch):
    provider = SinaIndustryProvider(direct=True)
    attempts = []
    response = requests.Response(); response.status_code = 403
    def get(*args, **kwargs):
        attempts.append(kwargs)
        raise requests.HTTPError("forbidden", response=response)
    monkeypatch.setattr(provider.session, "get", get)
    with pytest.raises(requests.HTTPError):
        provider._get("getHQNodes")
    assert len(attempts) == 1
    assert not provider.session.trust_env
    provider.close()
