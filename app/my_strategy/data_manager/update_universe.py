"""Market scope for daily downloads; historical research populations stay intact."""
from __future__ import annotations

import re

import pandas as pd

from .stock_pool import normalize_stock_code


def normalize_update_code(code: str) -> str:
    text = str(code).strip().upper()
    prefix = re.fullmatch(r"(SH|SZ|BJ|BSE)[.:-]?(\d{6})", text)
    if prefix:
        exchange, digits = prefix.groups()
        return f"{digits}.{'BJ' if exchange == 'BSE' else exchange}"
    suffix = re.fullmatch(r"(\d{6})[.:-](SH|SZ|BJ|BSE)", text)
    if suffix:
        digits, exchange = suffix.groups()
        return f"{digits}.{'BJ' if exchange == 'BSE' else exchange}"
    # A bare Shanghai B-share code must not be inferred as Beijing from '9'.
    if re.fullmatch(r"900\d{3}", text):
        return f"{text}.SH"
    return normalize_stock_code(text)


def filter_update_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exclude Beijing before limits, requests and progress denominators."""
    out = pool.copy()
    if "stock" not in out:
        out["stock"] = pd.Series(dtype=str)
    out["stock"] = out["stock"].map(normalize_update_code).astype(str)
    out = out.drop_duplicates("stock").reset_index(drop=True)
    excluded_mask = out["stock"].str.endswith(".BJ")
    # Explicit exchange columns are authoritative, including custom pools.
    for column in ("exchange", "market"):
        if column in out:
            excluded_mask |= out[column].fillna("").astype(str).str.strip().str.upper().isin(
                {"BJ", "BSE", "北交所", "北京证券交易所"}
            )
    excluded = out.loc[excluded_mask, "stock"].tolist()
    return out.loc[~excluded_mask].reset_index(drop=True), {
        "policy": "exclude_beijing_exchange", "requested": len(out),
        "eligible": int((~excluded_mask).sum()), "excluded": len(excluded),
        "excluded_stocks": excluded, "reason": "北交所不参与每日行情更新；历史清理由维护流程单独处理",
    }
