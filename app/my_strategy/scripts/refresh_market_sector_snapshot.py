"""Capture and verify a current public industry snapshot without touching prices.

A partial fetch is archived but cannot replace a complete active snapshot.
Membership is current classification, never historical point-in-time evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import math
from pathlib import Path
import re
import uuid

import requests

from my_strategy.core.paths import METADATA_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now

VERSION = "market_sector_snapshot_v1"
SINA_BASE = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center."
SOURCE_INFO = {
    "sina": ("sina:public_shenwan_l1", SINA_BASE + "getHQNodes"),
    "eastmoney": ("eastmoney:public_industry_boards", "https://quote.eastmoney.com/center/boardlist.html#industry_board"),
}


class SinaIndustryProvider:
    """Ordinary anonymous public requests; access refusals are never bypassed."""
    def __init__(self, *, direct=False):
        self.session = requests.Session()
        self.session.trust_env = not direct
        self.session.headers["User-Agent"] = "Mozilla/5.0"

    def close(self):
        self.session.close()

    def _get(self, method, **params):
        for attempt in range(2):
            try:
                response = self.session.get(SINA_BASE + method, params=params, timeout=12)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in (401, 403, 429):
                    raise
                if attempt:
                    raise
            except (requests.RequestException, ValueError):
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def catalog(self):
        value = self._get("getHQNodes")
        groups = value[1][0][1]
        items = next((group[1] for group in groups if group[0] == "申万一级"), None)
        if not isinstance(items, list) or not 25 <= len(items) <= 40:
            raise ValueError("公开申万一级行业目录数量异常")
        boards = [{"code": item[2], "name": item[0], "kind": "industry"} for item in items]
        if any(not re.fullmatch(r"sw1_\d{6}", board["code"]) or not board["name"] for board in boards):
            raise ValueError("公开行业目录身份异常")
        if len({board["code"] for board in boards}) != len(boards):
            raise ValueError("公开行业目录存在重复代码")
        return boards

    def members(self, code):
        if not re.fullmatch(r"sw1_\d{6}", code):
            raise ValueError("公开行业代码异常")
        total = int(self._get("getHQNodeStockCount", node=code))
        if total <= 0 or total > 10000:
            raise ValueError("公开行业成员总数异常")
        records = []
        for page in range(1, math.ceil(total / 80) + 1):
            chunk = self._get("getHQNodeData", node=code, page=page, num=80,
                sort="symbol", asc=1, symbol="", _s_r_a="page")
            if not isinstance(chunk, list) or not chunk:
                raise ValueError("公开行业成员分页缺失")
            records.extend(chunk)
        if len(records) != total or len({row.get("symbol") for row in records}) != total:
            raise ValueError("公开行业成员分页重复或不完整")
        if int(self._get("getHQNodeStockCount", node=code)) != total:
            raise ValueError("公开行业成员数量在分页期间发生变化")
        members = []
        for row in records:
            match = re.fullmatch(r"(sh|sz|bj)(\d{6})", str(row.get("symbol") or ""))
            if match:
                exchange, digits = match.groups()
                valid = (exchange == "sh" and digits.startswith("6") or
                    exchange == "sz" and digits.startswith(("0", "3")) or
                    exchange == "bj" and digits.startswith(("4", "8", "9")))
                if valid:
                    members.append({"stock": digits + "." + exchange.upper(), "name": row.get("name") or ""})
        return members, len(records) - len(members)


def _provider(source, direct):
    if source == "sina":
        return SinaIndustryProvider(direct=direct)
    if source == "eastmoney":
        from my_strategy.data_manager.sector_provider import EastmoneySectorProvider, EASTMONEY_ENDPOINTS
        class BoundedEastmoney(EastmoneySectorProvider):
            def _page(self, fs, page):
                for attempt in range(2):
                    try:
                        response = self.session.get(EASTMONEY_ENDPOINTS[attempt], params={
                            "pn": page, "pz": 100, "np": 1, "fltt": 2,
                            "fs": fs, "fields": "f12,f13,f14", "fid": "f12", "po": 0}, timeout=12)
                        response.raise_for_status()
                        payload = response.json()
                        if payload.get("rc") != 0 or not isinstance(payload.get("data"), dict):
                            raise ValueError("公开行业响应格式异常")
                        return payload["data"]
                    except requests.HTTPError as exc:
                        if exc.response is not None and exc.response.status_code in (401, 403, 429):
                            raise
                        if attempt:
                            raise
                    except (requests.RequestException, ValueError):
                        if attempt:
                            raise
                raise AssertionError("unreachable")
            def catalog(self):
                rows = self.rows("m:90 t:2 f:!50")
                if len(rows) < 20:
                    raise ValueError("公开行业目录数量异常")
                result = [{"code": str(row["f12"]), "name": str(row["f14"]).strip(), "kind": "industry"} for row in rows]
                if any(not re.fullmatch(r"BK\d+", row["code"]) or not row["name"] for row in result):
                    raise ValueError("公开行业目录身份异常")
                return result
        return BoundedEastmoney(direct=direct)
    raise ValueError("不支持的公开行业来源")


def validate_snapshot(value, *, universe=None):
    """Validate schema and calculate one-class-per-stock coverage without guessing."""
    if value.get("version") != VERSION or value.get("kind") != "industry":
        raise ValueError("行业快照版本或类型不兼容")
    if not isinstance(value.get("source"), str) or not value["source"] or not isinstance(value.get("source_url"), str) or not value["source_url"].startswith("https://"):
        raise ValueError("行业快照必须包含可核验公开来源及HTTPS地址")
    try:
        observed = datetime.fromisoformat(value["observed_at"])
        if observed.tzinfo is None:
            raise ValueError("行业快照获取时间需要时区")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("行业快照获取时间无效") from exc
    if type(value.get("complete")) is not bool or not isinstance(value.get("failures"), list) or not isinstance(value.get("boards"), list):
        raise ValueError("行业快照缺少完整性或失败记录")
    requested = value.get("requested_boards")
    if type(requested) is not int or requested < 1 or requested != len(value["boards"]) + len(value["failures"]):
        raise ValueError("行业目录、成功成员与失败记录数量不一致")
    if value["complete"] != (not value["failures"]):
        raise ValueError("行业完整性标识与失败记录不一致")
    identities, assignments, conflicts = set(), {}, set()
    for board in value["boards"]:
        if not isinstance(board, dict) or not isinstance(board.get("id"), str) or not board["id"] or board["id"] == "unknown" or not isinstance(board.get("name"), str) or not board["name"]:
            raise ValueError("行业板块身份不完整")
        if board["id"] in identities:
            raise ValueError("行业板块代码重复")
        identities.add(board["id"])
        if not isinstance(board.get("members"), list) or not board["members"] or any(not isinstance(member, str) for member in board["members"]):
            raise ValueError("行业成员必须为非空股票代码数组")
        if len(set(board["members"])) != len(board["members"]):
            raise ValueError("行业成员为空或重复")
        for symbol in board["members"]:
            if not isinstance(symbol, str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", symbol):
                raise ValueError("行业成员代码格式无效")
            if symbol in assignments:
                conflicts.add(symbol)
            assignments[symbol] = board["id"]
    for failure in value["failures"]:
        if not isinstance(failure, dict) or not failure.get("id") or not failure.get("error"):
            raise ValueError("行业失败记录身份或原因缺失")
        if failure["id"] in identities:
            raise ValueError("同一行业同时记录成功及失败")
        identities.add(failure["id"])
    known = set(assignments) - conflicts
    pool = set(universe) if universe is not None else None
    covered = known & pool if pool is not None else known
    return {"source_members": len(assignments), "unambiguous_members": len(known),
        "conflicting_members": len(conflicts), "conflicting_symbols": sorted(conflicts),
        "universe_size": len(pool) if pool is not None else None,
        "universe_hash": stable_hash(sorted(pool)) if pool is not None else None,
        "covered_symbols": len(covered), "unknown_symbols": len(pool - known) if pool is not None else None,
        "coverage": len(covered) / len(pool) if pool else None,
        "successful_boards": len(value["boards"]), "failed_boards": len(value["failures"])}


def _atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _status(destination, value):
    stamp = stable_hash(value)
    archive = destination.parent / "market_sector_snapshot_attempts" / (stamp + ".json")
    if not archive.exists():
        _atomic(archive, value)
    _atomic(destination.with_name("market_sector_snapshot_status.json"), value)


def publish_snapshot(value, *, destination=None, universe=None, min_coverage=.9):
    if not 0 < min_coverage <= 1:
        raise ValueError("分类最低覆盖率须在0至1之间")
    destination = Path(destination or METADATA_ROOT / "market_sector_snapshot.json")
    value = dict(value)
    supplied_hash = value.pop("snapshot_hash", None)
    if supplied_hash and stable_hash(value) != supplied_hash:
        raise ValueError("行业快照内容与所声明哈希不一致")
    value.pop("quality", None)
    quality = validate_snapshot(value, universe=universe)
    value["quality"] = quality
    value["snapshot_hash"] = stable_hash(value)
    archive = destination.parent / "market_sector_snapshots" / (value["snapshot_hash"] + ".json")
    if archive.exists():
        if json.loads(archive.read_text(encoding="utf-8")) != value:
            raise ValueError("不可覆盖已有行业快照归档")
    else:
        _atomic(archive, value)
    reason = None
    if quality["coverage"] is None:
        reason = "未提供可核验扫描股票池，归档但不激活"
    elif quality["coverage"] < min_coverage:
        reason = f"分类覆盖{quality['coverage']:.2%}低于最低{min_coverage:.2%}，保留已有快照"
    old = None
    if destination.is_file():
        try:
            old = json.loads(destination.read_text(encoding="utf-8"))
            old_quality = validate_snapshot(old, universe=universe)
            if old.get("complete") is True and not value["complete"]:
                reason = "部分来源结果不能覆盖已有完整快照"
            elif quality["coverage"] is not None and old_quality["coverage"] is not None and quality["coverage"] < old_quality["coverage"]:
                reason = "分类覆盖下降，保留已有快照"
        except (ValueError, OSError, TypeError, KeyError) as exc:
            reason = "已有快照无法核验，保留原文件：" + str(exc)
    activated = reason is None
    if activated:
        _atomic(destination, value)
    status = {"attempted_at": local_now().isoformat(), "source": value["source"],
        "state": "activated" if activated else "retained_existing" if old else "not_activated",
        "reason": reason, "complete": value["complete"], "quality": quality,
        "snapshot_hash": value["snapshot_hash"], "archive_path": str(archive),
        "active_snapshot_hash": value["snapshot_hash"] if activated else (old or {}).get("snapshot_hash")}
    _status(destination, status)
    return {**value, "activation": status}


def refresh_snapshot(*, destination=None, direct=False, workers=2, progress=None,
                     source="sina", universe=None, min_coverage=.9, max_failures=5,
                     provider_factory=None):
    destination = Path(destination or METADATA_ROOT / "market_sector_snapshot.json")
    factory = provider_factory or (lambda: _provider(source, direct))
    catalog_provider = factory()
    try:
        catalog = [board for board in catalog_provider.catalog() if board["kind"] == "industry"]
        if not catalog or len({board["code"] for board in catalog}) != len(catalog):
            raise ValueError("行业目录为空或代码重复")
    except Exception as exc:
        _status(destination, {"attempted_at": local_now().isoformat(), "source": source,
            "state": "source_failed", "reason": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        catalog_provider.close()

    def fetch(board):
        provider = factory()
        try:
            members, excluded = provider.members(board["code"])
            identities = [member["stock"] for member in members]
            if len(set(identities)) != len(identities):
                raise ValueError("行业成员有重复代码")
            return {"id": board["code"], "name": board["name"], "members": sorted(identities),
                "source_non_a_share_excluded": excluded, "membership_hash": stable_hash(sorted(identities))}
        finally:
            provider.close()

    boards, failures = [], []
    count = max(1, min(int(workers), 4)); max_failures = max(1, int(max_failures))
    remaining = iter(catalog)
    with ThreadPoolExecutor(max_workers=count) as pool:
        pending = {}
        def enqueue():
            board = next(remaining, None)
            if board is not None:
                pending[pool.submit(fetch, board)] = board
        for _ in range(count):
            enqueue()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                board = pending.pop(future)
                try:
                    boards.append(future.result())
                except Exception as exc:
                    failures.append({"id": board["code"], "name": board["name"], "error": f"{type(exc).__name__}: {exc}"})
                if progress:
                    progress(len(boards) + len(failures), len(catalog), len(failures))
                if len(failures) < max_failures:
                    enqueue()
        for board in remaining:
            failures.append({"id": board["code"], "name": board["name"],
                "error": "source_failure_limit_reached;未继续请求"})
    source_name, url = SOURCE_INFO[source]
    value = {"version": VERSION, "kind": "industry", "source": source_name, "source_url": url,
        "observed_at": local_now().isoformat(), "classification_basis": "current_snapshot",
        "note": "按当前行业分类归组历史信号；不是当时可知的历史成员关系。",
        "complete": not failures, "requested_boards": len(catalog),
        "catalog_hash": stable_hash(catalog), "boards": sorted(boards, key=lambda board: board["id"]),
        "failures": sorted(failures, key=lambda board: board["id"])}
    return publish_snapshot(value, destination=destination, universe=universe, min_coverage=min_coverage)


def main(argv=None):
    parser = argparse.ArgumentParser(description="核验并保存当前公开行业成员快照，不改行情或策略")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--direct", action="store_true", help="普通匿名直接连接，不绕过访问限制")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source", choices=tuple(SOURCE_INFO), default="sina")
    parser.add_argument("--min-coverage", type=float, default=.9)
    parser.add_argument("--max-failures", type=int, default=5)
    parser.add_argument("--import-snapshot", type=Path, help="校验已有来源JSON，保留原始来源及获取时间")
    parser.add_argument("--universe-db", type=Path, help="以只读方式取得固定扫描股票池")
    parser.add_argument("--universe-json", type=Path, help="固定股票代码JSON数组，用于可复现核验")
    args = parser.parse_args(argv)
    try:
        if args.universe_json:
            universe = json.loads(args.universe_json.read_text(encoding="utf-8"))
            if not isinstance(universe, list) or any(not isinstance(symbol, str) for symbol in universe):
                raise ValueError("股票池JSON须为股票代码数组")
        else:
            from my_strategy.adapters.czsc_adapter import list_symbols
            universe = [row["symbol"] for row in list_symbols(db_path=args.universe_db)]
        if args.import_snapshot:
            value = publish_snapshot(json.loads(args.import_snapshot.read_text(encoding="utf-8")),
                destination=args.destination, universe=universe, min_coverage=args.min_coverage)
        else:
            value = refresh_snapshot(destination=args.destination, direct=args.direct, workers=args.workers,
                source=args.source, universe=universe, min_coverage=args.min_coverage, max_failures=args.max_failures,
                progress=lambda done, total, failed: print(f"行业成员 {done}/{total} · 失败 {failed}", flush=True))
        print(json.dumps(value["activation"], ensure_ascii=False))
        return 0 if value["activation"]["state"] == "activated" else 2
    except (ValueError, OSError, requests.RequestException) as exc:
        status = {"attempted_at": local_now().isoformat(), "source": args.source,
            "state": "source_failed", "error": f"{type(exc).__name__}: {exc}"}
        _status(Path(args.destination or METADATA_ROOT / "market_sector_snapshot.json"), status)
        print(json.dumps(status, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
