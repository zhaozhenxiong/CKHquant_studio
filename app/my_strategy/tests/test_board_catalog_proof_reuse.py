"""Cold catalog reads share exact directory proofs within one SQLite view."""
from copy import deepcopy
from datetime import datetime, timezone
import sqlite3

import pytest

from my_strategy.storage import board_knowledge as kb
from my_strategy.tests.test_board_knowledge import snapshot


@pytest.fixture
def clock(monkeypatch):
    value = datetime(2026, 10, 14, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(kb, "_now", lambda: value)
    return value


def shared_snapshots(kind="industry", count=3):
    prefix = kb.KINDS[kind]
    values = [snapshot(board=f"THS_{prefix}_{881100+i}", source="fuyao:ths") for i in range(count)]
    rows = [{key: value[key] for key in ("board_id", "source_code", "name", "kind", "source", "taxonomy")}
        for value in values]
    catalog = deepcopy(values[0]["provenance"]["catalog_provenance"])
    for key in ("source_response", "repeated_source_response"):
        catalog[key]["data"]["item"] = [{"thscode": row["source_code"], "name": row["name"]} for row in rows]
    catalog["first_response_sha256"] = kb._hash(catalog["source_response"])
    catalog["repeated_response_sha256"] = kb._hash(catalog["repeated_source_response"])
    catalog["board_count"], catalog["catalog_set_hash"] = count, kb._hash(rows)
    for value in values:
        value["provenance"]["catalog_provenance"] = deepcopy(catalog)
    return values


def test_catalog_reuses_two_exact_proofs_and_discards_them_after_each_read(tmp_path, clock, monkeypatch):
    store = kb.BoardKnowledgeStore(tmp_path / "kb.db")
    values = shared_snapshots() + shared_snapshots("concept")
    for value in values:
        store.import_snapshot(value)
    calls = []
    verify = kb._fuyao_catalog_proof
    def counted(catalog, kind):
        calls.append(kind)
        return verify(catalog, kind)
    monkeypatch.setattr(kb, "_fuyao_catalog_proof", counted)
    expected = store.catalog(current=True)
    assert len(expected) == 6 and sorted(calls) == ["concept", "industry"]
    calls.clear()
    assert store.catalog(current=True) == expected
    assert sorted(calls) == ["concept", "industry"]
    # The standalone original-input path retains complete verification.
    calls.clear()
    for value in values:
        cache = {}
        assert kb._snapshot_input(value, verified_at=clock, catalog_proofs=cache) == kb._snapshot_input(value, verified_at=clock)
    assert len(calls) == 12


@pytest.mark.parametrize("change", ["directory_sha", "directory_count", "directory_identity", "directory_envelope", "directory_tag",
    "member_sha", "member_request", "member_set", "board_name", "future", "projection"])
def test_reused_directory_never_hides_distinct_proof_member_identity_or_time_failure(clock, change):
    first, changed = shared_snapshots(count=2)
    cache = {}
    kb._snapshot_input(first, verified_at=clock, catalog_proofs=cache)
    proof = changed["provenance"]
    catalog = proof["catalog_provenance"]
    if change == "directory_sha":
        catalog["first_response_sha256"] = "0" * 64
    elif change == "directory_count":
        catalog["board_count"] += 1
    elif change == "directory_identity":
        catalog["source_response"]["data"]["item"][0]["thscode"] = "invalid"
        catalog["first_response_sha256"] = kb._hash(catalog["source_response"])
    elif change == "directory_envelope":
        catalog["repeated_source_response"]["code"] = 2001
        catalog["repeated_response_sha256"] = kb._hash(catalog["repeated_source_response"])
    elif change == "directory_tag":
        catalog["request_params"] = {"tag": "cn_concept"}
    elif change == "member_sha":
        proof["first_response_sha256"] = "0" * 64
    elif change == "member_request":
        proof["request_params"] = {"thscode": first["source_code"]}
    elif change == "member_set":
        changed["members"] = ["000002.SZ", "600519.SH"]
    elif change == "board_name":
        changed["name"] = "目录没有的名称"
    elif change == "future":
        changed["observed_at"] = "2026-10-15T13:59:00+08:00"
    elif change == "projection":
        changed["membership_scope"] = "unverified_projection"
    with pytest.raises(ValueError) as standalone:
        kb._snapshot_input(changed, verified_at=clock)
    with pytest.raises(ValueError) as reused:
        kb._snapshot_input(changed, verified_at=clock, catalog_proofs=cache)
    assert str(standalone.value) == str(reused.value)
    assert len(cache) == 1


def test_equivalent_hash_fields_cannot_mask_distinct_directory_content(clock, monkeypatch):
    first, second = shared_snapshots(count=2)
    catalog = second["provenance"]["catalog_provenance"]
    # A valid second observation has the same member set and declared set hash,
    # but different authenticated original envelopes. It must be checked anew.
    catalog["request_id"] = catalog["source_response"]["request_id"] = "new-catalog-response"
    catalog["first_response_sha256"] = kb._hash(catalog["source_response"])
    calls = []
    verify = kb._fuyao_catalog_proof
    def counted(catalog, kind):
        calls.append(1)
        return verify(catalog, kind)
    monkeypatch.setattr(kb, "_fuyao_catalog_proof", counted)
    cache = {}
    kb._snapshot_input(first, verified_at=clock, catalog_proofs=cache)
    kb._snapshot_input(second, verified_at=clock, catalog_proofs=cache)
    assert len(calls) == len(cache) == 2


def test_failed_directory_proof_is_never_remembered(clock):
    value = shared_snapshots(count=1)[0]
    value["provenance"]["catalog_provenance"]["board_count"] += 1
    cache = {}
    for _ in range(2):
        with pytest.raises(ValueError, match="directory set"):
            kb._snapshot_input(value, verified_at=clock, catalog_proofs=cache)
        assert cache == {}


def test_catalog_reads_one_sqlite_view_and_next_call_rechecks_physical_sha(tmp_path, clock, monkeypatch):
    store = kb.BoardKnowledgeStore(tmp_path / "kb.db")
    for value in shared_snapshots(count=2):
        store.import_snapshot(value)
    expected = store.catalog(current=True)
    verify = kb.BoardKnowledgeStore._verified_snapshot
    changed = []
    def mutate_between_rows(con, row, **kwargs):
        result = verify(con, row, **kwargs)
        if not changed:
            with sqlite3.connect(store.path) as writer:
                target = writer.execute("SELECT source_payload_hash FROM board_snapshots WHERE snapshot_hash<>?", (row["snapshot_hash"],)).fetchone()[0]
                writer.execute("DROP TRIGGER immutable_source_payloads_update")
                writer.execute("UPDATE source_payloads SET blob_sha256=? WHERE payload_hash=?", ("0" * 64, target))
            changed.append(target)
        return result
    monkeypatch.setattr(kb.BoardKnowledgeStore, "_verified_snapshot", staticmethod(mutate_between_rows))
    # The concurrent commit cannot mix original source bytes into this view.
    assert store.catalog(current=True) == expected
    assert len(changed) == 1
    with pytest.raises(ValueError, match="physical SHA drift"):
        store.catalog(current=True)


def test_shared_proof_is_still_classified_per_kind(clock):
    first = shared_snapshots(count=1)[0]
    second = shared_snapshots("concept", count=1)[0]
    second["provenance"]["catalog_provenance"] = deepcopy(first["provenance"]["catalog_provenance"])
    cache = {}
    kb._snapshot_input(first, verified_at=clock, catalog_proofs=cache)
    with pytest.raises(ValueError, match="classification request"):
        kb._snapshot_input(second, verified_at=clock, catalog_proofs=cache)
