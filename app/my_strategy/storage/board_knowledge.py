"""Immutable THS public board snapshots; current facts never backfill history.

Each writer opens one IMMEDIATE transaction. Readers use SQLite mode=ro and
never create a database. Completed refresh manifests publish their verified
snapshot references; a failed board never replaces an earlier complete board.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, time, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, urlparse
import uuid
from zoneinfo import ZoneInfo

from my_strategy.core.paths import BASIC_KB_DB_PATH

SOURCE = "tonghuashun_public"
TAXONOMY = "THS_PUBLIC"
SOURCES = {SOURCE, "fuyao:ths"}
FORMAT = "ths_board_snapshot"
VERSION = 1
APPLICATION_ID = 0x424B4231
KINDS = {"industry": "I", "concept": "C", "region": "R"}
TABLES = {"stock_basic", "board_catalog_versions", "board_snapshots", "board_membership",
    "source_payloads", "board_daily_features", "stock_board_features", "board_refresh_runs"}
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_CATALOG_PROOF_REF = "_catalog_provenance_sha256"


def _now():
    return datetime.now(timezone.utc)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _hash(value):
    return _sha(_json(value).encode("utf-8"))


def _instant(value, *, cutoff=False):
    if not isinstance(value, str) or not value:
        raise ValueError("board knowledge time must include an explicit timezone")
    try:
        if cutoff and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            result = datetime.combine(datetime.fromisoformat(value).date(), time(15), _SHANGHAI)
        else:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("timezone missing")
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError) as exc:
        raise ValueError("board knowledge time must include an explicit timezone") from exc


def _stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def normalize_stock(value):
    value = str(value or "").strip().upper()
    match = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", value)
    if match:
        value = match[2] + "." + match[1]
    if re.fullmatch(r"\d{6}", value):
        exchange = "SH" if value.startswith(("6", "9")) else "SZ" if value.startswith(("0", "2", "3")) else "BJ" if value.startswith(("4", "8", "92")) else None
        if exchange is None:
            raise ValueError("invalid stock code")
        value += "." + exchange
    if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", value):
        raise ValueError("invalid stock code")
    return value


def _strict_json(value):
    def reject(value):
        raise ValueError("nonfinite board knowledge JSON")
    return json.loads(value, parse_constant=reject)


def _fuyao_response(proof, *, repeated=False):
    prefix = "repeated_" if repeated else ""
    response = proof.get(prefix + "source_response")
    digest_key = "repeated_response_sha256" if repeated else "first_response_sha256"
    if not isinstance(response, dict) or _hash(response) != proof.get(digest_key):
        raise ValueError("Fuyao original response proof SHA missing or inconsistent")
    data = response.get("data")
    if (type(response.get("code")) is not int or response["code"] != 0 or
            not isinstance(response.get("request_id"), str) or not response["request_id"] or
            not isinstance(data, dict) or type(data.get("timestamp")) is not int or data["timestamp"] <= 0 or
            not isinstance(data.get("item"), list) or any(not isinstance(row, dict) for row in data["item"])):
        raise ValueError("Fuyao original success envelope proof invalid")
    if "total" in data and (type(data["total"]) is not int or data["total"] != len(data["item"])):
        raise ValueError("Fuyao original full-response total inconsistent")
    if any(data.get(key) not in (None, False, "") for key in ("next", "next_page", "has_more", "next_cursor")):
        raise ValueError("Fuyao original response has unfinished pagination")
    if proof.get(prefix + "request_id") != response["request_id"] or proof.get(prefix + "source_timestamp") != data["timestamp"]:
        raise ValueError("Fuyao response indexed envelope identity inconsistent")
    return response


def _fuyao_proof(value, members, *, catalog_proofs=None):
    proof = value["provenance"]
    path = "/api/a-share-index/constituents/ths-stock-list"
    parsed = urlparse(value["source_url"])
    if (proof.get("request_path") != path or proof.get("request_params") != {"thscode": value["source_code"]} or
            parsed.path != path or parse_qs(parsed.query) != {"thscode": [value["source_code"]]}):
        raise ValueError("Fuyao original constituent request identity inconsistent")
    if (proof.get("full_response_contract") is not True or proof.get("membership_set_stable") is not True or
            type(proof.get("business_code")) is not int or proof["business_code"] != 0 or
            proof.get("current_only") is not True or proof.get("historical_membership_verified") is not False):
        raise ValueError("Fuyao current full-response contract proof missing")
    responses = [_fuyao_response(proof), _fuyao_response(proof, repeated=True)]
    projected = value.get("membership_scope") == "a_share_projection_v1"
    projection_fields = {"membership_scope", "source_member_count", "excluded_source_members"}
    if any(key in value for key in projection_fields) and not projected:
        raise ValueError("unknown or incomplete Fuyao member projection contract")
    full_sets, excluded_sets = [], []
    for response in responses:
        actual, full, excluded = [], [], []
        for row in response["data"]["item"]:
            code = row.get("thscode")
            if isinstance(code, str): code = code.strip().upper()
            if not isinstance(code, str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ|NQ)", code):
                raise ValueError("Fuyao original constituent identity invalid")
            if row.get("ticker") is not None and str(row["ticker"]).strip() != code.split(".")[0]:
                raise ValueError("Fuyao constituent ticker identity inconsistent")
            full.append(code)
            if code.endswith(".NQ"):
                if not projected:
                    raise ValueError("Fuyao non-A member requires an explicit projection contract")
                excluded.append(code)
            else:
                a_share = (code.endswith(".SH") and code.startswith("6") or
                    code.endswith(".SZ") and code.startswith(("0", "3")) or
                    code.endswith(".BJ") and code.startswith(("4", "8", "9")))
                if not a_share:
                    raise ValueError("Fuyao B-share or unsupported non-A constituent identity")
                actual.append(code)
        if len(set(full)) != len(full) or sorted(actual) != members:
            raise ValueError("Fuyao original member set proof inconsistent")
        full_sets.append(sorted(full));excluded_sets.append(sorted(excluded))
    if full_sets[0] != full_sets[1] or excluded_sets[0] != excluded_sets[1]:
        raise ValueError("Fuyao original full member set changed between responses")
    if projected:
        if (not excluded_sets[0] or type(value.get("source_member_count")) is not int or
                value["source_member_count"] != len(full_sets[0]) or
                value.get("excluded_source_members") != excluded_sets[0] or
                proof.get("full_source_membership_set_stable") is not True or
                proof.get("full_source_membership_set_hash") != _hash(full_sets[0]) or
                proof.get("repeated_full_source_membership_set_hash") != _hash(full_sets[1])):
            raise ValueError("Fuyao explicit A-share projection proof inconsistent")
    for key in ("full_source_membership_set_hash", "repeated_full_source_membership_set_hash"):
        if key in proof and proof[key] != _hash(full_sets[0]):
            raise ValueError("Fuyao full source member set proof inconsistent")
    if (proof.get("membership_set_hash") != _hash(members) or
            proof.get("repeated_membership_set_hash") != _hash(members) or
            value.get("payload_sha256") != _hash(responses[0]) or
            proof.get("member_list_state") != ("populated_verified" if members else "empty_verified")):
        raise ValueError("Fuyao member set or empty-list proof inconsistent")
    if ("reported_count" in proof) != ("total" in responses[0]["data"]):
        raise ValueError("Fuyao reported count must come from the original response")
    catalog = proof.get("catalog_provenance")
    # A single published catalog usually repeats two large directory proofs
    # across hundreds of independently authenticated snapshots. Share only the
    # completed directory validation, by exact content, within this read.
    key = (value["kind"], _json(catalog))
    identities = None if catalog_proofs is None else catalog_proofs.get(key)
    if identities is None:
        identities = _fuyao_catalog_proof(catalog, value["kind"])
        if catalog_proofs is not None:
            catalog_proofs[key] = identities
    expected = tuple(value[key] for key in ("board_id", "source_code", "name", "kind", "source", "taxonomy"))
    if expected not in identities:
        raise ValueError("Fuyao original directory set or board identity proof inconsistent")


def _fuyao_catalog_proof(catalog, kind):
    if (not isinstance(catalog, dict) or catalog.get("full_response_contract") is not True or
            catalog.get("catalog_set_stable") is not True or
            type(catalog.get("business_code")) is not int or catalog["business_code"] != 0):
        raise ValueError("Fuyao original directory proof missing")
    tag = {"industry": "industry", "concept": "cn_concept", "region": "region"}[kind]
    if (catalog.get("request_path") != "/api/a-share-index/catalog/ths-index-list" or
            catalog.get("request_params") != {"tag": tag}):
        raise ValueError("Fuyao original directory classification request inconsistent")
    # Directory proof originally records the first envelope fields only; the
    # repeated original still has its own complete success envelope and SHA.
    second = catalog.get("repeated_source_response")
    catalog_second = dict(catalog, repeated_request_id=second.get("request_id") if isinstance(second, dict) else None,
        repeated_source_timestamp=second.get("data", {}).get("timestamp") if isinstance(second, dict) and isinstance(second.get("data"), dict) else None)
    directories = [_fuyao_response(catalog), _fuyao_response(catalog_second, repeated=True)]
    sets = []
    for response in directories:
        rows = []
        for row in response["data"]["item"]:
            code, name = row.get("thscode"), row.get("name")
            if isinstance(code, str): code = code.strip().upper()
            if not isinstance(code, str) or not re.fullmatch(r"\d{6}\.TI", code) or not isinstance(name, str) or not name.strip():
                raise ValueError("Fuyao original directory identity invalid")
            rows.append({"board_id": "THS_" + KINDS[kind] + "_" + code,
                "source_code": code, "name": name.strip(), "kind": kind,
                "source": "fuyao:ths", "taxonomy": "THS_INDEX"})
        if not rows or len({row["board_id"] for row in rows}) != len(rows):
            raise ValueError("Fuyao original directory empty or duplicated")
        sets.append(sorted(rows, key=lambda row: row["board_id"]))
    if (sets[0] != sets[1] or type(catalog.get("board_count")) is not int or
            catalog["board_count"] != len(sets[0]) or catalog.get("catalog_set_hash") != _hash(sets[0])):
        raise ValueError("Fuyao original directory set or board identity proof inconsistent")
    return frozenset(tuple(row[key] for key in ("board_id", "source_code", "name", "kind", "source", "taxonomy")) for row in sets[0])


def _snapshot_input(snapshot, *, verified_at=None, catalog_proofs=None):
    if not isinstance(snapshot, dict):
        raise ValueError("board snapshot must be an object")
    value = _strict_json(_json(snapshot))
    supplied_hash = value.pop("snapshot_hash", None)
    if supplied_hash is not None and supplied_hash != _hash(value):
        raise ValueError("supplied snapshot hash does not match its input")
    source = value.get("source")
    if source != "fuyao:ths" and any(key in value for key in ("membership_scope", "source_member_count", "excluded_source_members")):
        raise ValueError("board member projection requires the identified Fuyao original responses")
    taxonomy = "THS_INDEX" if source == "fuyao:ths" else TAXONOMY
    if value.get("format") != FORMAT or source not in SOURCES or value.get("taxonomy") != taxonomy:
        raise ValueError("only identified THS public current snapshots are supported")
    kind, code = value.get("kind"), value.get("source_code")
    pattern = r"\d{6}\.TI" if source == "fuyao:ths" else r"\d{6}"
    if kind not in KINDS or not isinstance(code, str) or not re.fullmatch(pattern, code):
        raise ValueError("THS board kind or source code invalid")
    if value.get("board_id") != "THS_" + KINDS[kind] + "_" + code:
        raise ValueError("THS board identity does not match its kind and source code")
    if not isinstance(value.get("name"), str) or not value["name"].strip():
        raise ValueError("board name missing")
    parsed = urlparse(value.get("source_url") or "")
    domain_valid = parsed.hostname == "fuyao.aicubes.cn" if source == "fuyao:ths" else bool(parsed.hostname and (
        parsed.hostname == "10jqka.com.cn" or parsed.hostname.endswith(".10jqka.com.cn")))
    if parsed.scheme not in {"http", "https"} or not domain_valid:
        raise ValueError("board source must be the identified THS public site")
    if parsed.username or parsed.password:
        raise ValueError("board source URL must not contain credentials")
    count, members = value.get("expected_count"), value.get("members")
    if (value.get("complete") is not True or type(count) is not int or count < 0 or
            not isinstance(members, list) or len(members) != count):
        raise ValueError("incomplete board snapshot cannot replace a success")
    normalized = [normalize_stock(member) for member in members]
    if len(set(normalized)) != count:
        raise ValueError("board members must be unique after code normalization")
    proof = value.get("provenance")
    if (not isinstance(proof, dict) or proof.get("pagination_complete") is not True or
            type(proof.get("unique_members")) is not int or proof["unique_members"] != count):
        raise ValueError("board member completeness proof missing or inconsistent")
    if "reported_count" in proof:
        source_count = value.get("source_member_count") if value.get("membership_scope") == "a_share_projection_v1" else count
        if type(proof["reported_count"]) is not int or proof["reported_count"] != source_count:
            raise ValueError("board member reported count proof inconsistent")
    elif proof.get("full_response_contract") is not True or proof.get("membership_set_stable") is not True:
        raise ValueError("unpaginated member list needs a full-response contract and repeated-set proof")
    if source == "fuyao:ths":
        _fuyao_proof(value, sorted(normalized), catalog_proofs=catalog_proofs)
    elif count == 0:
        raise ValueError("empty board list requires verified original successful full responses")
    if value.get("payload_sha256") is not None and not re.fullmatch(r"[a-f0-9]{64}", str(value["payload_sha256"])):
        raise ValueError("source payload proof SHA invalid")
    observed, available, effective = (_instant(value.get(key)) for key in ("observed_at", "available_at", "effective_at"))
    now = _now() if verified_at is None else verified_at
    if now > _now():
        raise ValueError("board verified import clock cannot be in the future")
    if observed > now or effective > now or available > now:
        raise ValueError("current board snapshot cannot claim a future observation")
    if available < observed or available < effective:
        raise ValueError("board availability cannot precede observation or effective time")
    input_hash = _hash(value)
    result = dict(value, members=sorted(normalized), name=value["name"].strip(),
        observed_at=_stamp(observed), available_at=_stamp(max(available, now)),
        effective_at=_stamp(max(effective, observed)), historical_membership_verified=False,
        imported_at=_stamp(now), classification_basis="current_snapshot",
        source_origin="fuyao" if source == "fuyao:ths" else "tonghuashun_public")
    return value, input_hash, result


_SCHEMA = """
CREATE TABLE stock_basic (
 row_hash TEXT PRIMARY KEY, symbol TEXT NOT NULL, name TEXT NOT NULL,
 source TEXT NOT NULL, observed_at TEXT NOT NULL, available_at TEXT NOT NULL,
 payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
CREATE INDEX idx_board_stock_identity ON stock_basic(symbol, available_at);
CREATE TABLE source_payloads (
 payload_hash TEXT PRIMARY KEY, gzip_blob BLOB NOT NULL, blob_sha256 TEXT NOT NULL,
 encoding TEXT NOT NULL CHECK(encoding='json_gzip'), recorded_at TEXT NOT NULL);
CREATE TABLE board_catalog_versions (
 catalog_hash TEXT PRIMARY KEY, board_id TEXT NOT NULL, source_code TEXT NOT NULL,
 name TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL, taxonomy TEXT NOT NULL,
 observed_at TEXT NOT NULL, available_at TEXT NOT NULL, payload_json TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL);
CREATE TABLE board_refresh_runs (
 run_id TEXT PRIMARY KEY, source TEXT NOT NULL, requested_json TEXT NOT NULL,
 requested_sha256 TEXT NOT NULL, observed_at TEXT NOT NULL, started_at TEXT NOT NULL,
 finished_at TEXT, status TEXT NOT NULL CHECK(status IN ('running','succeeded','partial','failed')),
 snapshot_refs_json TEXT NOT NULL, failures_json TEXT NOT NULL,
 summary_json TEXT, summary_sha256 TEXT);
CREATE TABLE board_snapshots (
 snapshot_hash TEXT PRIMARY KEY, board_id TEXT NOT NULL, catalog_hash TEXT NOT NULL,
 input_hash TEXT NOT NULL UNIQUE, source_payload_hash TEXT NOT NULL,
 origin_run_id TEXT NOT NULL, observed_at TEXT NOT NULL, available_at TEXT NOT NULL,
 effective_at TEXT NOT NULL, member_count INTEGER NOT NULL CHECK(member_count>=0),
 payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
 FOREIGN KEY(catalog_hash) REFERENCES board_catalog_versions(catalog_hash),
 FOREIGN KEY(source_payload_hash) REFERENCES source_payloads(payload_hash),
 FOREIGN KEY(origin_run_id) REFERENCES board_refresh_runs(run_id));
CREATE INDEX idx_board_snapshot_time ON board_snapshots(board_id, available_at, effective_at);
CREATE TABLE board_membership (
 snapshot_hash TEXT NOT NULL, symbol TEXT NOT NULL,
 PRIMARY KEY(snapshot_hash,symbol), FOREIGN KEY(snapshot_hash) REFERENCES board_snapshots(snapshot_hash));
CREATE INDEX idx_board_member_stock ON board_membership(symbol,snapshot_hash);
CREATE TABLE board_daily_features (
 key_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, key_json TEXT NOT NULL,
 payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, recorded_at TEXT NOT NULL);
CREATE TABLE stock_board_features (
 key_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, key_json TEXT NOT NULL,
 payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, recorded_at TEXT NOT NULL);
"""


class BoardKnowledgeStore:
    def __init__(self, db_path=None, readonly=False):
        self.path = Path(db_path or BASIC_KB_DB_PATH).expanduser().resolve()
        self.readonly = bool(readonly)
        if self.readonly:
            if self.path.is_file():
                with self._read() as con:
                    self._verify_schema(con)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=FULL")
            con.execute("BEGIN IMMEDIATE")
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables and con.execute("PRAGMA user_version").fetchone()[0] == 0:
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        con.execute(statement)
                for table in TABLES - {"board_refresh_runs"}:
                    for operation in ("UPDATE", "DELETE"):
                        con.execute(f"CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable board knowledge'); END")
                con.execute("CREATE TRIGGER immutable_refresh_update BEFORE UPDATE ON board_refresh_runs WHEN OLD.status<>'running' BEGIN SELECT RAISE(ABORT,'immutable completed refresh'); END")
                con.execute("CREATE TRIGGER immutable_refresh_delete BEFORE DELETE ON board_refresh_runs BEGIN SELECT RAISE(ABORT,'immutable refresh history'); END")
                con.execute(f"PRAGMA user_version={VERSION}")
                con.execute(f"PRAGMA application_id={APPLICATION_ID}")
            self._verify_schema(con)
            con.commit()

    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    @contextmanager
    def _read(self):
        if not self.path.is_file():
            yield None
            return
        con = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        try:
            yield con
        finally:
            con.close()

    @contextmanager
    def _write(self):
        if self.readonly:
            raise ValueError("board knowledge store is read-only")
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _verify_schema(con):
        if (con.execute("PRAGMA user_version").fetchone()[0] != VERSION or
                con.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID or
                {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} != TABLES):
            raise ValueError("board knowledge database identity/schema mismatch")

    def seed_stock_basic(self, records):
        if not isinstance(records, (list, tuple)):
            raise ValueError("stock basic records must be a list")
        rows = []
        now = _stamp(_now())
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("stock basic record must be an object")
            value = _strict_json(_json(record))
            symbol = normalize_stock(value.get("code") or value.get("stock") or value.get("symbol"))
            if symbol in seen:
                raise ValueError("duplicate stock basic symbol")
            seen.add(symbol)
            name = str(value.get("name") or value.get("stock_name") or "").strip()
            if not name:
                raise ValueError("stock name missing")
            observed = _stamp(_instant(value.get("updated_at") or now))
            if observed > now:
                raise ValueError("stock basic cannot claim future observation")
            digest = _hash(value)
            rows.append((digest, symbol, name, str(value.get("source") or "raw_sqlite:stock_basic"), observed, now, _json(value), digest))
        with self._write() as con:
            before = con.total_changes
            con.executemany("INSERT OR IGNORE INTO stock_basic VALUES(?,?,?,?,?,?,?,?)", rows)
            inserted = con.total_changes - before
        return {"records": len(rows), "inserted": inserted, "stock_basic_hash": _hash([r[0] for r in rows]),
            "historical_listing_verified": False}

    @staticmethod
    def _source_payload(con, digest):
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("board source payload reference SHA invalid")
        row = con.execute("SELECT * FROM source_payloads WHERE payload_hash=?", (digest,)).fetchone()
        if row is None or row["encoding"] != "json_gzip" or _sha(bytes(row["gzip_blob"])) != row["blob_sha256"]:
            raise ValueError("board source payload physical SHA drift")
        try:
            value = _strict_json(gzip.decompress(bytes(row["gzip_blob"])).decode("utf-8"))
        except (ValueError, OSError, EOFError) as exc:
            raise ValueError("board source payload parse failed") from exc
        if _hash(value) != digest:
            raise ValueError("board source payload identity drift")
        return value

    @classmethod
    def _put_source_payload(cls, con, value):
        raw = _json(value).encode();digest = _sha(raw)
        if con.execute("SELECT 1 FROM source_payloads WHERE payload_hash=?", (digest,)).fetchone():
            cls._source_payload(con, digest)
        else:
            blob = gzip.compress(raw, compresslevel=1, mtime=0)
            con.execute("INSERT INTO source_payloads VALUES(?,?,?,?,?)", (digest, blob, _sha(blob), "json_gzip", _stamp(_now())))
        return digest

    def _begin(self, con, source, requested_boards, observed_at):
        if source not in SOURCES:
            raise ValueError("unsupported board source")
        requested = [] if requested_boards is None else requested_boards
        if not isinstance(requested, list):
            raise ValueError("requested boards must be a list")
        ids = [x.get("board_id") if isinstance(x, dict) else x for x in requested]
        if any(not isinstance(x, str) or not re.fullmatch(r"THS_[ICR]_\d{6}(?:\.TI)?", x) for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("requested board identities invalid or duplicated")
        # A 710-board frozen catalog shares two large directory proofs. Store
        # each distinct proof once; per-board mutations parse only small refs.
        compact = []
        references = {}
        for board in requested:
            if not isinstance(board, dict):
                compact.append(board);continue
            item = dict(board)
            if _CATALOG_PROOF_REF in item:
                raise ValueError("external requested catalog cannot supply internal proof references")
            proof = item.pop("catalog_provenance", None)
            if proof is not None:
                if not isinstance(proof, dict):
                    raise ValueError("requested catalog proof must be an object")
                # Provider rows share the same proof object; a hash also merges
                # separately deserialized equivalent proofs on resume.
                identity = id(proof)
                digest = references.get(identity)
                if digest is None:
                    digest = self._put_source_payload(con, proof);references[identity] = digest
                item[_CATALOG_PROOF_REF] = digest
            compact.append(item)
        now = _stamp(_now());observed = _stamp(_instant(observed_at or now))
        if observed > now:
            raise ValueError("refresh observation cannot be in the future")
        run_id = uuid.uuid4().hex
        body = _json(compact)
        con.execute("INSERT INTO board_refresh_runs VALUES(?,?,?,?,?,?,NULL,'running','{}','[]',NULL,NULL)",
            (run_id, source, body, _sha(body.encode()), observed, now))
        return run_id

    def begin_refresh(self, source=SOURCE, requested_boards=None, observed_at=None):
        with self._write() as con:
            return self._begin(con, source, requested_boards, observed_at)

    @staticmethod
    def _run(con, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("invalid board refresh id")
        row = con.execute("SELECT * FROM board_refresh_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("board refresh not found")
        result = dict(row)
        if _sha(result["requested_json"].encode()) != result["requested_sha256"]:
            raise ValueError("board refresh requested catalog SHA drift")
        result["requested_boards"] = _strict_json(result.pop("requested_json"))
        result["snapshot_refs"] = _strict_json(result.pop("snapshot_refs_json"))
        result["failures"] = _strict_json(result.pop("failures_json"))
        if result["summary_json"] is not None:
            if _sha(result["summary_json"].encode()) != result["summary_sha256"]:
                raise ValueError("board refresh summary SHA drift")
            result["summary"] = _strict_json(result.pop("summary_json"))
            summary = result["summary"]
            if (summary.get("run_id") != run_id or summary.get("status") != result["status"] or
                    summary.get("source") != result["source"] or summary.get("snapshot_refs") != result["snapshot_refs"] or
                    summary.get("failures") != result["failures"] or summary.get("finished_at") != result["finished_at"] or
                    summary.get("requested_sha256") != result["requested_sha256"]):
                raise ValueError("completed refresh indexed identity drift")
        result["board_ids"] = [x.get("board_id") if isinstance(x, dict) else x for x in result["requested_boards"]]
        result["completed_board_ids"] = sorted(result["snapshot_refs"])
        return result

    def get_refresh(self, run_id):
        with self._read() as con:
            if con is None:
                return None
            run = self._run(con, run_id)
            proofs = {}
            requested = []
            for board in run["requested_boards"]:
                if not isinstance(board, dict):
                    requested.append(board);continue
                item = dict(board)
                digest = item.pop(_CATALOG_PROOF_REF, None)
                if digest is not None:
                    if digest not in proofs:
                        proofs[digest] = self._source_payload(con, digest)
                    item["catalog_provenance"] = proofs[digest]
                requested.append(item)
            run["requested_boards"] = requested
            return run

    @staticmethod
    def _running(run):
        if run["status"] != "running":
            raise ValueError("completed board refresh is immutable; retry in a new refresh")

    def record_failure(self, run_id, board_id, error):
        if not isinstance(board_id, str) or not re.fullmatch(r"THS_[ICR]_\d{6}(?:\.TI)?", board_id) or not isinstance(error, str) or not error:
            raise ValueError("board failure identity/error invalid")
        with self._write() as con:
            run = self._run(con, run_id);self._running(run)
            if run["board_ids"] and board_id not in run["board_ids"]:
                raise ValueError("failure board is outside frozen requested catalog")
            failures = [*run["failures"], {"board_id": board_id, "error": error, "observed_at": _stamp(_now())}]
            con.execute("UPDATE board_refresh_runs SET failures_json=? WHERE run_id=?", (_json(failures), run_id))
        return {"run_id": run_id, "board_id": board_id, "status": "failed"}

    def record_source_failure(self, run_id, error, scope="catalog"):
        if not isinstance(error, str) or not error or scope not in {"catalog", "source", "authentication"}:
            raise ValueError("source failure scope/error invalid")
        with self._write() as con:
            run = self._run(con, run_id);self._running(run)
            failures = [*run["failures"], {"board_id": None, "scope": scope, "error": error, "observed_at": _stamp(_now())}]
            con.execute("UPDATE board_refresh_runs SET failures_json=? WHERE run_id=?", (_json(failures), run_id))
        return {"run_id": run_id, "scope": scope, "status": "failed"}

    def _finish(self, con, run_id):
        run = self._run(con, run_id)
        if run["status"] != "running":
            return run["summary"]
        success = set(run["snapshot_refs"])
        missing = set(run["board_ids"]) - success
        failed = {x["board_id"] for x in run["failures"] if x.get("board_id")} - success
        failed |= missing
        source_failure = any(not x.get("board_id") for x in run["failures"])
        status = "succeeded" if success and not failed and not source_failure else "partial" if success else "failed"
        summary = {"run_id": run_id, "source": run["source"], "status": status,
            "requested_boards": len(run["board_ids"]), "completed_boards": len(success),
            "completed_board_ids": sorted(success), "failed_board_ids": sorted(failed),
            "failures": run["failures"], "missing_board_ids": sorted(missing),
            "requested_sha256": run["requested_sha256"],
            "snapshot_refs": run["snapshot_refs"], "finished_at": _stamp(_now())}
        encoded = _json(summary)
        con.execute("UPDATE board_refresh_runs SET status=?,finished_at=?,summary_json=?,summary_sha256=? WHERE run_id=?",
            (status, summary["finished_at"], encoded, _sha(encoded.encode()), run_id))
        return summary

    def finish_refresh(self, run_id):
        with self._write() as con:
            return self._finish(con, run_id)

    def reuse_current_snapshot(self, run_id, board_id, snapshot_hash):
        """Attach today's verified immutable observation to a new frozen run.

        Fresh directory response envelopes can differ, but their entire board
        set must be identical. Never rewrite a snapshot's original source proof,
        availability or observation time to make a cache look newly collected.
        """
        with self._write() as con:
            run = self._run(con, run_id); self._running(run)
            requested = next((item for item in run['requested_boards']
                if isinstance(item, dict) and item.get('board_id') == board_id), None)
            if requested is None or run['source'] != 'fuyao:ths':
                raise ValueError('reused snapshot is outside frozen Fuyao catalog')
            now = _now(); stamp = _stamp(now)
            row = con.execute("""SELECT s.* FROM board_snapshots s
                WHERE s.snapshot_hash=? AND EXISTS(SELECT 1 FROM board_refresh_runs r,
                    json_each(r.snapshot_refs_json) refs WHERE refs.value=s.snapshot_hash
                    AND r.status IN ('succeeded','partial') AND r.finished_at<=?)""",
                (snapshot_hash, stamp)).fetchone()
            if row is None: raise ValueError('reused snapshot has no completed verified publication')
            value = self._verified_snapshot(con, row)
            fields = ('board_id', 'source_code', 'name', 'kind', 'source', 'taxonomy')
            if any(requested.get(key) != value[key] for key in fields):
                raise ValueError('reused snapshot differs from frozen board identity')
            if (_instant(value['observed_at']).astimezone(_SHANGHAI).date() != now.astimezone(_SHANGHAI).date()
                    or value['available_at'] > stamp or value['effective_at'] > stamp):
                raise ValueError('only an already available observation from today can be reused')
            proof_ref = requested.get(_CATALOG_PROOF_REF)
            fresh = self._source_payload(con, proof_ref) if proof_ref else None
            original = value['provenance'].get('catalog_provenance')
            identities = _fuyao_catalog_proof(fresh, value['kind'])
            if (tuple(value[key] for key in fields) not in identities or
                    not isinstance(original, dict) or fresh['catalog_set_hash'] != original.get('catalog_set_hash')):
                raise ValueError('reused snapshot complete directory set differs from frozen catalog')
            refs = dict(run['snapshot_refs']); prior = refs.get(board_id)
            if prior and prior != snapshot_hash: raise ValueError('refresh already bound a different board snapshot')
            refs[board_id] = snapshot_hash
            con.execute('UPDATE board_refresh_runs SET snapshot_refs_json=? WHERE run_id=?', (_json(refs), run_id))
        return value

    def import_snapshot(self, snapshot, run_id=None):
        original, input_hash, value = _snapshot_input(snapshot)
        with self._write() as con:
            own_run = run_id is None
            if own_run:
                run_id = self._begin(con, value["source"], [value["board_id"]], value["observed_at"])
            run = self._run(con, run_id);self._running(run)
            if run["source"] != value["source"] or run["board_ids"] and value["board_id"] not in run["board_ids"]:
                raise ValueError("snapshot is outside frozen refresh source/catalog")
            requested = next((board for board in run["requested_boards"] if isinstance(board, dict) and board.get("board_id") == value["board_id"]), None)
            if requested is not None:
                if any(key in requested and requested[key] != value[key] for key in ("source_code", "name", "kind", "source", "taxonomy")):
                    raise ValueError("snapshot identity differs from the frozen requested catalog")
                frozen_proof = requested.get(_CATALOG_PROOF_REF)
                if frozen_proof and frozen_proof != _hash(value["provenance"].get("catalog_provenance")):
                    raise ValueError("snapshot directory proof differs from the frozen requested catalog")
            existing = con.execute("SELECT * FROM board_snapshots WHERE input_hash=?", (input_hash,)).fetchone()
            if existing is None and snapshot.get("snapshot_hash"):
                existing = con.execute("SELECT * FROM board_snapshots WHERE snapshot_hash=?", (snapshot["snapshot_hash"],)).fetchone()
                if existing is not None and existing["payload_json"] != _json(original):
                    raise ValueError("signed local snapshot input identity drift")
            if existing:
                value = self._verified_snapshot(con, existing)
                digest = value["snapshot_hash"]
            else:
                self._put_source_payload(con, original)
                catalog = {key: value[key] for key in ("board_id", "source_code", "name", "kind", "source", "taxonomy", "observed_at", "available_at")}
                catalog_hash = _hash(catalog);catalog_json = _json(catalog)
                con.execute("INSERT OR IGNORE INTO board_catalog_versions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (catalog_hash, *[catalog[k] for k in ("board_id", "source_code", "name", "kind", "source", "taxonomy", "observed_at", "available_at")], catalog_json, _sha(catalog_json.encode())))
                digest = _hash(value);encoded = _json(value)
                con.execute("INSERT INTO board_snapshots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (digest, value["board_id"], catalog_hash,
                    input_hash, input_hash, run_id, value["observed_at"], value["available_at"], value["effective_at"], len(value["members"]), encoded, _sha(encoded.encode())))
                con.executemany("INSERT INTO board_membership VALUES(?,?)", [(digest, symbol) for symbol in value["members"]])
                value["snapshot_hash"] = digest
            refs = dict(run["snapshot_refs"])
            prior = refs.get(value["board_id"])
            if prior and prior != digest:
                raise ValueError("one refresh cannot publish conflicting versions of a board")
            refs[value["board_id"]] = digest
            con.execute("UPDATE board_refresh_runs SET snapshot_refs_json=? WHERE run_id=?", (_json(refs), run_id))
            if own_run:
                self._finish(con, run_id)
        return {**value, "run_id": run_id, "import_reused": existing is not None}

    @staticmethod
    def _verified_snapshot(con, row, *, catalog_proofs=None):
        encoded = row["payload_json"]
        if _sha(encoded.encode()) != row["payload_sha256"]:
            raise ValueError("board snapshot payload SHA drift")
        value = _strict_json(encoded)
        if (_hash(value) != row["snapshot_hash"] or any(value.get(key) != row[key] for key in
                ("board_id", "observed_at", "available_at", "effective_at")) or
                value.get("expected_count") != row["member_count"]):
            raise ValueError("board snapshot identity drift")
        members = [x[0] for x in con.execute("SELECT symbol FROM board_membership WHERE snapshot_hash=? ORDER BY symbol", (row["snapshot_hash"],))]
        if members != value["members"] or len(members) != row["member_count"]:
            raise ValueError("board membership identity drift")
        original = BoardKnowledgeStore._source_payload(con, row["source_payload_hash"])
        if _hash(original) != row["input_hash"] or row["source_payload_hash"] != row["input_hash"]:
            raise ValueError("board source payload identity drift")
        # Reconstruct at the first local verified import clock, so reads retain
        # immutable availability while checking projection and original proof.
        _, digest, derived = _snapshot_input(original, verified_at=_instant(value.get("imported_at")),
            catalog_proofs=catalog_proofs)
        if digest != row["input_hash"] or derived != value:
            raise ValueError("board snapshot differs from its original source proof")
        catalog = con.execute("SELECT * FROM board_catalog_versions WHERE catalog_hash=?", (row["catalog_hash"],)).fetchone()
        if catalog is None or _sha(catalog["payload_json"].encode()) != catalog["payload_sha256"] or _hash(_strict_json(catalog["payload_json"])) != row["catalog_hash"]:
            raise ValueError("board catalog identity drift")
        catalog_body = _strict_json(catalog["payload_json"])
        if any(catalog_body[key] != value.get(key) or catalog_body[key] != catalog[key] for key in catalog_body):
            raise ValueError("board catalog indexed identity drift")
        return {**value, "snapshot_hash": row["snapshot_hash"]}

    def catalog(self, kind=None, as_of=None, current=False):
        if kind is not None and kind not in KINDS:
            raise ValueError("unknown board kind")
        if current and as_of is not None:
            raise ValueError("current reference and causal as_of must be requested separately")
        cutoff = _stamp(_now() if as_of is None else _instant(as_of, cutoff=True))
        with self._read() as con:
            if con is None:
                return []
            # Bind all reused proofs and per-board physical/index checks to one
            # SQLite view. The proof dictionary is discarded after this call.
            con.execute("BEGIN")
            catalog_proofs = {}
            for manifest in con.execute("SELECT run_id FROM board_refresh_runs WHERE status IN ('succeeded','partial')").fetchall():
                self._run(con, manifest[0])
            rows = con.execute("""SELECT DISTINCT s.* FROM board_snapshots s
                JOIN board_refresh_runs r ON r.status IN ('succeeded','partial')
                JOIN json_each(r.snapshot_refs_json) refs ON refs.value=s.snapshot_hash
                WHERE s.available_at<=? AND s.effective_at<=? AND r.finished_at<=?
                ORDER BY s.available_at DESC,s.observed_at DESC,s.snapshot_hash DESC""", (cutoff, cutoff, cutoff)).fetchall()
            result, seen = [], set()
            for row in rows:
                if row["board_id"] in seen:
                    continue
                value = self._verified_snapshot(con, row, catalog_proofs=catalog_proofs)
                if kind is not None and value["kind"] != kind:
                    continue
                seen.add(row["board_id"])
                result.append(value)
            return sorted(result, key=lambda value: value["board_id"])

    membership_snapshots = catalog

    def stock_boards(self, symbol, as_of=None, current=False):
        symbol = normalize_stock(symbol)
        values = self.catalog(as_of=as_of, current=current)
        boards = [{key: value[key] for key in ("board_id", "source_code", "name", "kind", "source", "taxonomy",
            "snapshot_hash", "observed_at", "available_at", "effective_at", "historical_membership_verified")} for value in values if symbol in value["members"]]
        cutoff = _now() if as_of is None else _instant(as_of, cutoff=True)
        return {"symbol": symbol, "boards": boards, "availability": "current_reference" if current and boards else "available" if boards else "unavailable",
            "available": bool(boards), "source": sorted({b["source"] for b in boards}) if boards else None, "date": str(cutoff.astimezone(_SHANGHAI).date()),
            "as_of": _stamp(cutoff), "current": bool(current),
            "observed_at": max((b["observed_at"] for b in boards), default=None),
            "available_at": max((b["available_at"] for b in boards), default=None),
            "historical_membership_verified": False,
            "limitations": ["Current observations do not establish membership before their availability."]}

    def status(self, run_id=None):
        if run_id is not None:
            with self._read() as con:
                if con is None:
                    return None
                run = self._run(con, run_id)
                return {key: value for key, value in run.items() if key != "requested_boards"}
        with self._read() as con:
            if con is None:
                return {"available": False, "status": "unavailable", "path": str(self.path), "stock_count": 0,
                    "snapshot_count": 0, "board_count": 0, "member_count": 0, "counts_by_kind": {kind: 0 for kind in KINDS},
                    "latest_run": None, "last_refresh": None, "current_only": True, "history_eligible": False,
                    "feature_cache": {"daily": 0, "stock": 0}}
            row = con.execute("SELECT run_id FROM board_refresh_runs ORDER BY started_at DESC,run_id DESC LIMIT 1").fetchone()
            latest = None if row is None else self._run(con, row[0])
            if latest is not None:
                latest = {key: value for key, value in latest.items() if key != "requested_boards"}
                latest["requested_board_count"] = len(latest["board_ids"])
            stocks = con.execute("SELECT COUNT(DISTINCT symbol) FROM stock_basic").fetchone()[0]
            snapshots = con.execute("SELECT COUNT(*) FROM board_snapshots").fetchone()[0]
            feature_counts = {"daily": con.execute("SELECT COUNT(*) FROM board_daily_features").fetchone()[0],
                "stock": con.execute("SELECT COUNT(*) FROM stock_board_features").fetchone()[0]}
        boards = self.catalog(current=True)
        return {"available": True, "status": "ready" if boards else "empty", "path": str(self.path), "stock_count": stocks,
            "snapshot_count": snapshots, "board_count": len(boards), "member_count": sum(len(b["members"]) for b in boards),
            "covered_stock_count": len({symbol for b in boards for symbol in b["members"]}),
            "counts_by_kind": {kind: sum(b["kind"] == kind for b in boards) for kind in KINDS},
            "complete_board_count": len(boards), "latest_run": latest, "last_refresh": latest,
            "historical_membership_verified": False, "history_eligible": False, "current_only": True,
            "feature_cache": feature_counts,
            "source": sorted({b["source"] for b in boards}) or ([latest["source"]] if latest else [])}

    @staticmethod
    def _feature_table(kind):
        if kind in {"board_daily", "daily"}:
            return "board_daily_features"
        if kind in {"stock", "stock_daily"}:
            return "stock_board_features"
        raise ValueError("unsupported board feature cache kind")

    def put_feature(self, kind, key, payload):
        table = self._feature_table(kind)
        if not isinstance(payload, dict) or not isinstance(key, (str, dict, list, tuple)):
            raise ValueError("board feature key/payload must be explicit JSON")
        encoded, key_json = _json(payload), _json(key)
        for field in ("observed_at", "available_at", "decision_at"):
            if payload.get(field) is not None:
                _instant(payload[field])
        digest, key_hash = _sha(encoded.encode()), _hash({"kind": kind, "key": key})
        with self._write() as con:
            prior = con.execute(f"SELECT * FROM {table} WHERE key_hash=?", (key_hash,)).fetchone()
            if prior and (prior["key_json"] != key_json or prior["payload_sha256"] != digest or prior["payload_json"] != encoded):
                raise ValueError("same immutable board feature input produced a different output")
            con.execute(f"INSERT OR IGNORE INTO {table} VALUES(?,?,?,?,?,?)", (key_hash, kind, key_json, encoded, digest, _stamp(_now())))
        return _strict_json(encoded)

    def get_feature(self, kind, key):
        table = self._feature_table(kind);key_json = _json(key);key_hash = _hash({"kind": kind, "key": key})
        with self._read() as con:
            if con is None:
                return None
            row = con.execute(f"SELECT * FROM {table} WHERE key_hash=?", (key_hash,)).fetchone()
            if row is None:
                return None
            if row["kind"] != kind or row["key_json"] != key_json or _sha(row["payload_json"].encode()) != row["payload_sha256"]:
                raise ValueError("board feature cached identity or payload SHA drift")
            return _strict_json(row["payload_json"])

    def latest_feature(self, kind="stock", symbol=None, board_id=None, date=None, mode=None):
        """Read an existing feature payload; this method never computes features."""
        table = self._feature_table(kind)
        if symbol is not None:
            symbol = normalize_stock(symbol)
        with self._read() as con:
            if con is None:
                return None
            tables = [table, "board_daily_features"] if table == "stock_board_features" else [table]
            rows = [row for selected in tables for row in con.execute(f"SELECT * FROM {selected} ORDER BY recorded_at DESC,key_hash DESC").fetchall()]
            candidates = []
            for row in rows:
                if _sha(row["payload_json"].encode()) != row["payload_sha256"]:
                    raise ValueError("board feature cached payload SHA drift")
                key = _strict_json(row["key_json"])
                if _hash({"kind": row["kind"], "key": key}) != row["key_hash"]:
                    raise ValueError("board feature cached key identity drift")
                bundle = _strict_json(row["payload_json"])
                if mode is not None and bundle.get("mode") not in {None, mode}:
                    continue
                values = bundle.get("stocks" if table == "stock_board_features" else "boards")
                if values is None:
                    values = [bundle]
                if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
                    raise ValueError("board feature cached row schema invalid")
                for payload in values:
                    if symbol is not None and payload.get("symbol") != symbol:
                        continue
                    if board_id is not None and payload.get("board_id") != board_id:
                        continue
                    row_date = payload.get("date", payload.get("price_date", bundle.get("price_date")))
                    if date is not None and row_date != date:
                        continue
                    if mode is not None and payload.get("mode", bundle.get("mode")) != mode:
                        continue
                    candidates.append((str(row_date or ""), row["recorded_at"], payload))
            return max(candidates, key=lambda item: item[:2])[2] if candidates else None
