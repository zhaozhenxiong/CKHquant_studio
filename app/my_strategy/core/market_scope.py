"""Public request scope; frozen historical inputs and numerical code stay intact."""
from __future__ import annotations

import re


def normalize_public_symbol(value: str) -> str:
    """Normalize exchange aliases, respecting explicit SH/SZ and Shanghai B shares."""
    text = str(value).strip().upper()
    prefix = re.fullmatch(r"(SH|SZ|BJ|BSE)[.:-]?(\d{6})", text)
    suffix = re.fullmatch(r"(\d{6})[.:-](SH|SZ|BJ|BSE)", text)
    if prefix:
        exchange, digits = prefix.groups()
    elif suffix:
        digits, exchange = suffix.groups()
    elif re.fullmatch(r"\d{6}", text):
        digits = text
        exchange = "SH" if text.startswith(("6", "900")) else "BJ" if text.startswith(("4", "8", "9")) else "SZ"
    else:
        raise ValueError("股票代码应为六位数字及 .SH/.SZ 后缀")
    return f"{digits}.{'BJ' if exchange == 'BSE' else exchange}"


def require_supported_symbol(value: str) -> str:
    symbol = normalize_public_symbol(value)
    if symbol.endswith(".BJ"):
        raise ValueError("北交所股票已从新请求范围剔除，请选择沪深股票")
    return symbol


def filter_new_stock_basic(frame):
    """Exclude Beijing before new basic-directory data reaches DB or parquet."""
    if frame is None or frame.empty:
        return frame.copy() if frame is not None else None
    column = "code" if "code" in frame else "stock"
    normalized = frame[column].map(normalize_public_symbol)
    excluded = normalized.str.endswith(".BJ")
    for name in ("exchange", "market"):
        if name in frame:
            excluded |= frame[name].fillna("").astype(str).str.strip().str.upper().isin(
                {"BJ", "BSE", "北交所", "北京证券交易所"})
    result = frame.loc[~excluded].copy()
    result[column] = normalized.loc[~excluded]
    return result.reset_index(drop=True)


def supported_catalog(*, query: str | None = None, limit: int | None = None, db_path=None):
    """Filter before limiting, without altering the archived securities catalog."""
    from my_strategy.adapters.czsc_adapter import list_symbols

    rows = list_symbols(query=query, limit=10000, **({"db_path": db_path} if db_path is not None else {}))
    eligible = [row for row in rows if not normalize_public_symbol(row["symbol"]).endswith(".BJ")]
    return eligible if limit is None else eligible[:max(1, min(int(limit), 10000))]


def new_scan_symbols(symbols=None):
    """Resolve a new legacy scan's population before entering its numerical code."""
    selected = list(dict.fromkeys(require_supported_symbol(symbol) for symbol in symbols or ()))
    if not selected:
        selected = [row["symbol"] for row in supported_catalog()]
    if not selected:
        raise ValueError("没有可扫描的沪深股票")
    return selected
