"""Immutable membership, failure retention, causal time and cache integrity."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3

import pytest

from my_strategy.storage import board_knowledge as kb


@pytest.fixture
def clock(monkeypatch):
    value = [datetime(2026, 10, 14, 6, tzinfo=timezone.utc)]
    monkeypatch.setattr(kb, "_now", lambda: value[0])
    return value


@pytest.fixture
def store(tmp_path, clock):
    return kb.BoardKnowledgeStore(tmp_path / "basic_kb.db")


def snapshot(*, board="THS_I_881155", members=None, source=kb.SOURCE):
    members = ["000001.SZ", "600519.SH"] if members is None else members
    fuyao = source == "fuyao:ths"
    if fuyao and not board.endswith(".TI"):
        board += ".TI"
    kind = {"I": "industry", "C": "concept", "R": "region"}[board.split("_")[1]]
    proof = {"pagination_complete": True, "reported_count": len(members), "unique_members": len(members)}
    if fuyao:
        response = {"code": 0, "request_id": "fixture-response-1", "data": {"timestamp": 1791957599,
            "total": len(members), "item": [{"thscode": symbol, "ticker": symbol.split(".")[0]} for symbol in members]}}
        repeat = dict(response, request_id="fixture-response-2")
        directory = {"code": 0, "request_id": "fixture-catalog-1", "data": {"timestamp": 1791957500,
            "item": [{"thscode": board.split("_", 2)[2], "name": "银行" if kind == "industry" else "概念"}]}}
        repeat_directory = dict(directory, request_id="fixture-catalog-2")
        catalog_rows = [{"board_id": board, "source_code": board.split("_", 2)[2],
            "name": "银行" if kind == "industry" else "概念", "kind": kind, "source": source, "taxonomy": "THS_INDEX"}]
        proof.update(business_code=0, request_id=response["request_id"], source_timestamp=response["data"]["timestamp"],
            request_path="/api/a-share-index/constituents/ths-stock-list", request_params={"thscode": board.split("_", 2)[2]},
            repeated_request_id=repeat["request_id"], repeated_source_timestamp=repeat["data"]["timestamp"],
            full_response_contract=True, membership_set_stable=True, current_only=True, historical_membership_verified=False,
            source_response=response, repeated_source_response=repeat,
            first_response_sha256=kb._hash(response), repeated_response_sha256=kb._hash(repeat),
            membership_set_hash=kb._hash(sorted(members)), repeated_membership_set_hash=kb._hash(sorted(members)),
            member_list_state="populated_verified" if members else "empty_verified",
            catalog_provenance={"full_response_contract": True, "catalog_set_stable": True, "business_code": 0,
                "request_path": "/api/a-share-index/catalog/ths-index-list", "request_params": {"tag": "industry" if kind == "industry" else "cn_concept" if kind == "concept" else "region"},
                "board_count": 1, "catalog_set_hash": kb._hash(catalog_rows), "source_response": directory,
                "repeated_source_response": repeat_directory, "first_response_sha256": kb._hash(directory),
                "repeated_response_sha256": kb._hash(repeat_directory), "request_id": directory["request_id"],
                "source_timestamp": directory["data"]["timestamp"]})
    return {"format": kb.FORMAT, "source": source, "taxonomy": "THS_INDEX" if fuyao else kb.TAXONOMY,
        "board_id": board, "source_code": board.split("_", 2)[2], "name": "银行" if kind == "industry" else "概念",
        "kind": kind, "observed_at": "2026-10-14T13:59:00+08:00", "available_at": "2026-10-14T13:59:00+08:00",
        "effective_at": "2026-10-14T13:59:00+08:00", "complete": True, "expected_count": len(members), "members": members,
        "source_url": "https://fuyao.aicubes.cn/api/a-share-index/constituents/ths-stock-list?thscode=" + board.split("_", 2)[2]
            if fuyao else "https://q.10jqka.com.cn/thshy/detail/code/881155/", "provenance": proof,
        **({"payload_sha256": kb._hash(response)} if fuyao else {})}


def projection_snapshot(*, members=None):
    value = snapshot(source="fuyao:ths", members=members)
    proof = value["provenance"]
    for field in ("source_response", "repeated_source_response"):
        proof[field] = copy.deepcopy(proof[field])
        proof[field]["data"]["item"].append({"thscode": "834683.NQ", "ticker": "834683", "name": "爹地宝贝"})
        proof[field]["data"]["total"] += 1
    value.update(membership_scope="a_share_projection_v1", source_member_count=len(value["members"])+1,
        excluded_source_members=["834683.NQ"])
    proof["reported_count"] = value["source_member_count"]
    proof["full_source_membership_set_stable"] = True
    full = sorted([*value["members"], "834683.NQ"])
    proof["full_source_membership_set_hash"] = proof["repeated_full_source_membership_set_hash"] = kb._hash(full)
    resign_source_responses(value)
    return value


def resign_source_responses(value):
    proof = value["provenance"]
    proof["first_response_sha256"] = value["payload_sha256"] = kb._hash(proof["source_response"])
    proof["repeated_response_sha256"] = kb._hash(proof["repeated_source_response"])


def test_readonly_missing_database_never_creates_files(tmp_path):
    path = tmp_path / "missing" / "basic_kb.db"
    store = kb.BoardKnowledgeStore(path, readonly=True)
    assert store.catalog() == []
    assert store.stock_boards("000001.SZ", current=True)["availability"] == "unavailable"
    assert store.get_feature("daily", "missing") is None
    assert store.latest_feature("stock", symbol="000001.SZ") is None
    assert store.status()["history_eligible"] is False
    with pytest.raises(ValueError, match="read-only"):
        store.begin_refresh()
    assert not path.parent.exists()


def test_exact_eight_tables_and_immutable_history(store):
    value = store.import_snapshot(snapshot())
    with sqlite3.connect(store.path) as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} == kb.TABLES
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            con.execute("DELETE FROM board_snapshots")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            con.execute("UPDATE board_membership SET symbol='000002.SZ'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            con.execute("UPDATE board_refresh_runs SET status='failed' WHERE run_id=?", (value["run_id"],))


def test_basic_seed_keeps_versions_and_has_no_listing_history(store, clock):
    records = [{"code": "000001", "name": "平安银行", "source": "raw-fixture", "updated_at": "2026-10-08T23:38:24+08:00"}]
    first = store.seed_stock_basic(records)
    assert first["inserted"] == 1 and first["historical_listing_verified"] is False
    assert store.seed_stock_basic(records)["inserted"] == 0
    changed = [dict(records[0], name="平安银行新名")]
    clock[0] += timedelta(minutes=1)
    store.seed_stock_basic(changed)
    assert store.status()["stock_count"] == 1
    with sqlite3.connect(store.path) as con:
        assert con.execute("SELECT COUNT(*) FROM stock_basic").fetchone()[0] == 2
    with pytest.raises(ValueError, match="duplicate"):
        store.seed_stock_basic(records * 2)


def test_import_idempotence_preserves_original_clock_and_sha(store, clock):
    original = snapshot()
    first = store.import_snapshot(original)
    clock[0] += timedelta(days=2)
    second = store.import_snapshot(original)
    assert second["snapshot_hash"] == first["snapshot_hash"]
    assert second["available_at"] == first["available_at"]
    assert second["import_reused"] is True
    assert store.status()["snapshot_count"] == 1
    before = copy.deepcopy(store.catalog())
    assert store.catalog(as_of="2026-10-14T14:00:00+08:00") == before
    sealed = store.catalog()[0]
    third = store.import_snapshot(sealed)
    assert third["snapshot_hash"] == first["snapshot_hash"] and third["import_reused"]
    assert third["available_at"] == first["available_at"]


def test_manual_past_snapshot_cannot_backfill_first_observation(store):
    value = snapshot()
    for key in ("observed_at", "available_at", "effective_at"):
        value[key] = "2026-10-12T15:00:00+08:00"
    stored = store.import_snapshot(value)
    assert stored["available_at"] == "2026-10-14T06:00:00.000000+00:00"
    assert stored["historical_membership_verified"] is False
    assert store.catalog(as_of="2026-10-12") == []
    assert not store.stock_boards("000001.SZ", as_of="2026-10-12")["available"]
    assert store.stock_boards("000001.SZ", current=True)["available"]
    with pytest.raises(ValueError, match="separately"):
        store.catalog(as_of="2026-10-12", current=True)


def test_running_refresh_is_not_available_before_success_manifest(store, clock):
    run = store.begin_refresh(requested_boards=["THS_I_881155"])
    first = store.import_snapshot(snapshot(), run)
    assert store.catalog() == []
    clock[0] += timedelta(minutes=5)
    summary = store.finish_refresh(run)
    assert summary["status"] == "succeeded"
    assert store.catalog(as_of="2026-10-14T14:02:00+08:00") == []
    assert store.catalog(as_of="2026-10-14T14:05:00+08:00")[0]["snapshot_hash"] == first["snapshot_hash"]
    assert store.finish_refresh(run) == summary
    with pytest.raises(ValueError, match="immutable"):
        store.record_failure(run, "THS_I_881155", "late error")


def test_failed_board_preserves_success_and_partial_publishes_other_complete_board(store, clock):
    first = store.import_snapshot(snapshot())
    clock[0] += timedelta(hours=1)
    run = store.begin_refresh(requested_boards=["THS_I_881155", "THS_C_885001"])
    store.record_failure(run, "THS_I_881155", "403 access refused")
    store.import_snapshot(snapshot(board="THS_C_885001", members=["000001.SZ"]), run)
    summary = store.finish_refresh(run)
    assert summary["status"] == "partial" and summary["failed_board_ids"] == ["THS_I_881155"]
    boards = store.stock_boards("000001.SZ", current=True)["boards"]
    assert len(boards) == 2
    assert next(b for b in boards if b["kind"] == "industry")["snapshot_hash"] == first["snapshot_hash"]


def test_source_failure_keeps_facts_without_fabricated_board(store):
    run = store.begin_refresh(source="fuyao:ths")
    store.record_source_failure(run, "business code 2001: product unavailable")
    summary = store.finish_refresh(run)
    assert summary["status"] == "failed" and summary["failed_board_ids"] == []
    assert summary["failures"][0]["board_id"] is None
    assert store.status()["last_refresh"]["status"] == "failed"


def test_same_board_conflict_rolls_back_uncommitted_payload(store, clock):
    run = store.begin_refresh(requested_boards=["THS_I_881155"])
    store.import_snapshot(snapshot(), run)
    with pytest.raises(ValueError, match="conflicting"):
        store.import_snapshot(snapshot(members=["000001.SZ"]), run)
    with sqlite3.connect(store.path) as con:
        assert con.execute("SELECT COUNT(*) FROM board_snapshots").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM source_payloads").fetchone()[0] == 1


def test_fuyao_full_identity_and_current_only_envelope(store):
    value = snapshot(board="THS_C_886042", source="fuyao:ths")
    value["provenance"].pop("reported_count")
    for key in ("source_response", "repeated_source_response"):
        value["provenance"][key]["data"].pop("total", None)
    value["payload_sha256"] = value["provenance"]["first_response_sha256"] = kb._hash(value["provenance"]["source_response"])
    value["provenance"]["repeated_response_sha256"] = kb._hash(value["provenance"]["repeated_source_response"])
    stored = store.import_snapshot(value)
    assert stored["board_id"] == "THS_C_886042.TI"
    assert stored["source_origin"] == "fuyao"
    assert stored["taxonomy"] == "THS_INDEX"
    assert stored["historical_membership_verified"] is False
    assert store.status()["counts_by_kind"]["concept"] == 1
    wrong = snapshot(source="fuyao:ths")
    wrong["source_url"] = "https://q.10jqka.com.cn/"
    with pytest.raises(ValueError, match="source"):
        store.import_snapshot(wrong)


def test_verified_fuyao_empty_list_is_success_and_replaces_current_members(store, clock):
    first = store.import_snapshot(snapshot(source="fuyao:ths"))
    clock[0] += timedelta(minutes=1)
    empty = snapshot(source="fuyao:ths", members=[])
    for field in ("observed_at", "available_at", "effective_at"):
        empty[field] = "2026-10-14T14:00:30+08:00"
    second = store.import_snapshot(empty)
    assert second["snapshot_hash"] != first["snapshot_hash"]
    assert store.catalog(current=True)[0]["members"] == []
    assert store.status()["board_count"] == 1 and store.status()["member_count"] == 0
    assert store.catalog(as_of="2026-10-14T14:00:00+08:00")[0]["members"] == first["members"]


@pytest.mark.parametrize("members", [["000001.SZ", "600519.SH"], []])
def test_explicit_nq_projection_preserves_complete_source_and_only_a_members(store, members):
    value = projection_snapshot(members=members)
    stored = store.import_snapshot(value)
    assert stored["members"] == members and stored["expected_count"] == len(members)
    assert stored["source_member_count"] == len(members)+1
    assert stored["excluded_source_members"] == ["834683.NQ"]
    assert stored["membership_scope"] == "a_share_projection_v1"
    assert len(stored["provenance"]["source_response"]["data"]["item"]) == len(members)+1
    assert stored["provenance"]["reported_count"] == len(members)+1
    assert store.catalog(current=True)[0]["snapshot_hash"] == stored["snapshot_hash"]
    assert store.stock_boards("000001.SZ", current=True)["available"] == ("000001.SZ" in members)


@pytest.mark.parametrize("member", ["900901.SH", "200001.SZ", "000001.BJ"])
@pytest.mark.parametrize("projected", [False, True])
def test_self_consistent_original_b_share_proof_is_rejected_in_all_scopes(store, member, projected):
    members = ["000001.SZ", member]
    value = projection_snapshot(members=members) if projected else snapshot(source="fuyao:ths", members=members)
    # Every declared list/hash and both original responses agree, so rejection
    # must independently enforce the A-share market contract in the store.
    with pytest.raises(ValueError, match="B-share or unsupported non-A"):
        store.import_snapshot(value)
    assert store.status()["snapshot_count"] == 0


@pytest.mark.parametrize("change", ["scope", "source_count", "fabricated_nq", "missing_nq", "excluded_a", "dropped_a", "full_hash", "repeated_full_hash", "stable", "nq_ticker", "duplicate_nq", "changed_repeat", "unknown", "hk", "b", "projection_without_nq"])
def test_projection_cannot_fake_exclusions_or_silently_drop_a_members(store, change):
    value = projection_snapshot();proof = value["provenance"]
    if change == "scope": value["membership_scope"] = "anything"
    elif change == "source_count": value["source_member_count"] += 1;proof["reported_count"] += 1
    elif change == "fabricated_nq": value["excluded_source_members"].append("834684.NQ")
    elif change == "missing_nq": value["excluded_source_members"] = []
    elif change == "excluded_a": value["excluded_source_members"].append(value["members"][0])
    elif change == "dropped_a":
        value["members"].pop();value["expected_count"] = proof["unique_members"] = 1
        proof["membership_set_hash"] = proof["repeated_membership_set_hash"] = kb._hash(value["members"])
    elif change == "full_hash": proof["full_source_membership_set_hash"] = "0" * 64
    elif change == "repeated_full_hash": proof["repeated_full_source_membership_set_hash"] = "0" * 64
    elif change == "stable": proof["full_source_membership_set_stable"] = False
    elif change == "nq_ticker": proof["source_response"]["data"]["item"][-1]["ticker"] = "000001"
    elif change == "duplicate_nq":
        for field in ("source_response", "repeated_source_response"):
            proof[field]["data"]["item"].append(proof[field]["data"]["item"][-1]);proof[field]["data"]["total"] += 1
        value["source_member_count"] = proof["reported_count"] = 4
    elif change == "changed_repeat":
        proof["repeated_source_response"]["data"]["item"][-1] = {"thscode": "834684.NQ", "ticker": "834684"}
        proof["repeated_full_source_membership_set_hash"] = kb._hash(sorted([*value["members"], "834684.NQ"]))
    elif change in {"unknown", "hk", "b"}:
        code = {"unknown": "834683.UNKNOWN", "hk": "834683.HK", "b": "834683.B"}[change]
        for field in ("source_response", "repeated_source_response"):
            proof[field]["data"]["item"][-1]["thscode"] = code
        value["excluded_source_members"] = [code]
        proof["full_source_membership_set_hash"] = proof["repeated_full_source_membership_set_hash"] = kb._hash(sorted([*value["members"], code]))
    elif change == "projection_without_nq":
        for field in ("source_response", "repeated_source_response"):
            proof[field]["data"]["item"].pop();proof[field]["data"]["total"] -= 1
        value["source_member_count"] = proof["reported_count"] = 2;value["excluded_source_members"] = []
        proof["full_source_membership_set_hash"] = proof["repeated_full_source_membership_set_hash"] = kb._hash(value["members"])
    resign_source_responses(value)
    with pytest.raises(ValueError): store.import_snapshot(value)
    assert store.status()["snapshot_count"] == 0


def test_read_rechecks_projection_against_original_proof_after_metadata_resigning(store):
    store.import_snapshot(projection_snapshot())
    with sqlite3.connect(store.path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM board_snapshots").fetchone()
        altered = json.loads(row["payload_json"]);altered["excluded_source_members"] = []
        body, digest = kb._json(altered), kb._hash(altered)
        con.execute("DROP TRIGGER immutable_board_snapshots_update")
        con.execute("DROP TRIGGER immutable_board_membership_update")
        con.execute("UPDATE board_snapshots SET snapshot_hash=?,payload_json=?,payload_sha256=?", (digest, body, hashlib.sha256(body.encode()).hexdigest()))
        con.execute("UPDATE board_membership SET snapshot_hash=?", (digest,))
        changed = con.execute("SELECT * FROM board_snapshots").fetchone()
        with pytest.raises(ValueError, match="original source proof"):
            store._verified_snapshot(con, changed)


@pytest.mark.parametrize("change", ["business_failure", "original_member", "repeated_member", "directory_name", "directory_sha", "directory_tag", "member_request", "invented_total", "empty_unproved"])
def test_fuyao_self_declared_proof_cannot_replace_original_response(store, change):
    value = snapshot(source="fuyao:ths")
    proof = value["provenance"]
    if change == "business_failure":
        proof["source_response"]["code"] = 3002
        proof["first_response_sha256"] = value["payload_sha256"] = kb._hash(proof["source_response"])
    elif change in {"original_member", "repeated_member"}:
        field = "source_response" if change == "original_member" else "repeated_source_response"
        # The two fixture envelopes intentionally share one data dictionary.
        proof[field]["data"]["item"][0]["thscode"] = "000002.SZ"
        proof[field]["data"]["item"][0]["ticker"] = "000002"
        proof["first_response_sha256"] = value["payload_sha256"] = kb._hash(proof["source_response"])
        proof["repeated_response_sha256"] = kb._hash(proof["repeated_source_response"])
    elif change == "directory_name":
        directory = proof["catalog_provenance"]
        directory["source_response"]["data"]["item"][0]["name"] = "另一个目录名字"
        directory["first_response_sha256"] = kb._hash(directory["source_response"])
        directory["repeated_response_sha256"] = kb._hash(directory["repeated_source_response"])
    elif change == "directory_sha": proof["catalog_provenance"]["first_response_sha256"] = "0" * 64
    elif change == "directory_tag": proof["catalog_provenance"]["request_params"] = {"tag": "cn_concept"}
    elif change == "member_request": proof["request_params"] = {"thscode": "881101.TI"}
    elif change == "invented_total":
        for field in ("source_response", "repeated_source_response"):
            proof[field]["data"].pop("total", None)
        proof["first_response_sha256"] = value["payload_sha256"] = kb._hash(proof["source_response"])
        proof["repeated_response_sha256"] = kb._hash(proof["repeated_source_response"])
    elif change == "empty_unproved":
        value["members"] = [];value["expected_count"] = 0
        proof["unique_members"] = proof["reported_count"] = 0
    with pytest.raises(ValueError): store.import_snapshot(value)
    assert store.status()["snapshot_count"] == 0


@pytest.mark.parametrize("change", ["duplicate", "member", "count", "partial", "board", "kind", "source", "taxonomy", "proof", "timezone", "backdated", "future", "supplied_sha"])
def test_bad_snapshots_never_publish_or_leave_partial_rows(store, change):
    value = snapshot()
    if change == "duplicate": value["members"] = ["000001.SZ", "sz000001"]
    elif change == "member": value["members"][0] = "foreign-stock"
    elif change == "count": value["expected_count"] += 1
    elif change == "partial": value["complete"] = False
    elif change == "board": value["board_id"] = "THS_C_881155"
    elif change == "kind": value["kind"] = "anything"
    elif change == "source": value["source"] = "eastmoney"
    elif change == "taxonomy": value["taxonomy"] = "CSRC"
    elif change == "proof": value["provenance"] = {}
    elif change == "timezone": value["observed_at"] = "2026-10-14T13:00:00"
    elif change == "backdated": value["available_at"] = "2026-10-12T15:00:00+08:00"
    elif change == "future": value["observed_at"] = "2026-10-15T15:00:00+08:00"
    elif change == "supplied_sha": value["snapshot_hash"] = "0" * 64
    with pytest.raises(ValueError): store.import_snapshot(value)
    assert store.status()["snapshot_count"] == 0


@pytest.mark.parametrize("table,column", [("board_snapshots", "available_at"), ("board_membership", "symbol"), ("source_payloads", "gzip_blob")])
def test_indexed_clock_membership_and_physical_payload_tamper_are_rejected(store, table, column):
    store.import_snapshot(snapshot())
    with sqlite3.connect(store.path) as con:
        con.execute(f"DROP TRIGGER immutable_{table}_update")
        replacement = b"broken" if column == "gzip_blob" else "000002.SZ" if column == "symbol" else "2026-10-12T00:00:00.000000+00:00"
        con.execute(f"UPDATE {table} SET {column}=? WHERE rowid=(SELECT MIN(rowid) FROM {table})", (replacement,))
    with pytest.raises(ValueError, match="drift"):
        store.catalog(current=True)


def test_feature_cache_immutable_hash_and_flatrow_latest_filters(store, clock):
    key = {"identity": "frozen-input", "mode": "current_reference"}
    value = {"price_date": "2026-10-09", "mode": "current_reference", "available_at": "2026-10-14T14:00:00+08:00",
        "boards": [{"board_id": "THS_I_881155", "date": "2026-10-09", "mode": "current_reference", "features": {"return5": .1}}],
        "stocks": [{"symbol": "000001.SZ", "date": "2026-10-09", "mode": "current_reference", "features": {"relative5": .02}}]}
    assert store.put_feature("daily", key, value) == value
    assert store.put_feature("daily", key, value) == value
    assert store.get_feature("daily", key) == value
    assert store.latest_feature("stock", symbol="000001.SZ", date="2026-10-09", mode="current_reference") == value["stocks"][0]
    assert store.latest_feature("daily", board_id="THS_I_881155", date="2026-10-09") == value["boards"][0]
    assert store.latest_feature("stock", symbol="000001.SZ", mode="causal") is None
    with pytest.raises(ValueError, match="different output"):
        store.put_feature("daily", key, dict(value, price_date="2026-10-10"))
    with sqlite3.connect(store.path) as con:
        con.execute("DROP TRIGGER immutable_board_daily_features_update")
        con.execute("UPDATE board_daily_features SET payload_json='{}'")
    with pytest.raises(ValueError, match="SHA drift"): store.get_feature("daily", key)
    with pytest.raises(ValueError, match="SHA drift"): store.latest_feature("stock", symbol="000001.SZ")


def test_parallel_writers_serialize_without_overwriting_snapshots(store, clock):
    values = [snapshot(board=f"THS_C_{885000+i}", members=["000001.SZ"]) for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(store.import_snapshot, values))
    assert len({r["snapshot_hash"] for r in results}) == 8
    assert store.status()["snapshot_count"] == 8
    assert store.status()["board_count"] == 8


def test_710_board_catalog_compacts_two_directory_proofs_and_resume_is_lossless(store):
    boards = []
    for kind, prefix, count in (("industry", "I", 320), ("concept", "C", 390)):
        prototype = snapshot(board=f"THS_{prefix}_881155", source="fuyao:ths")
        proof = copy.deepcopy(prototype["provenance"]["catalog_provenance"])
        rows = [{"board_id": f"THS_{prefix}_{881000+i}.TI", "source_code": f"{881000+i}.TI", "kind": kind,
            "name": f"目录{i}", "source": "fuyao:ths", "taxonomy": "THS_INDEX"} for i in range(count)]
        items = [{"thscode": row["source_code"], "name": row["name"]} for row in rows]
        for response in ("source_response", "repeated_source_response"):
            proof[response]["data"]["item"] = items
        proof["first_response_sha256"] = kb._hash(proof["source_response"])
        proof["repeated_response_sha256"] = kb._hash(proof["repeated_source_response"])
        proof["board_count"], proof["catalog_set_hash"] = count, kb._hash(rows)
        boards.extend({**row, "catalog_provenance": proof} for row in rows)
    run = store.begin_refresh(source="fuyao:ths", requested_boards=boards)
    with sqlite3.connect(store.path) as con:
        stored = con.execute("SELECT requested_json FROM board_refresh_runs WHERE run_id=?", (run,)).fetchone()[0]
        assert len(stored) < 230_000
        assert "source_response" not in stored
        assert con.execute("SELECT COUNT(*) FROM source_payloads").fetchone()[0] == 2
        con.row_factory = sqlite3.Row
        internal = store._run(con, run)
        assert all("catalog_provenance" not in row for row in internal["requested_boards"])
    resumed = store.get_refresh(run)
    assert resumed["requested_boards"] == boards
    assert "source_response" not in kb._json(store.status())
    # A proof blob with a valid compact reference still fails closed if damaged.
    with sqlite3.connect(store.path) as con:
        con.execute("DROP TRIGGER immutable_source_payloads_update")
        con.execute("UPDATE source_payloads SET gzip_blob=? WHERE rowid=(SELECT MIN(rowid) FROM source_payloads)", (b"wrong",))
    with pytest.raises(ValueError, match="SHA drift"):
        store.get_refresh(run)


def test_foreign_database_is_rejected_even_readonly(tmp_path):
    path = tmp_path / "foreign.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE stock_basic(code TEXT)")
    with pytest.raises(ValueError, match="identity/schema"):
        kb.BoardKnowledgeStore(path, readonly=True)
