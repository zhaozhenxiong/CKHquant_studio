"""Content identity and contained paths for immutable current artifacts."""
from __future__ import annotations
from datetime import datetime, time
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from my_strategy.core.tz import LOCAL_TZ

def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def model_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()

def identity_path(root: Path, identity: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", identity):
        raise ValueError("invalid CZSC model identity")
    path = (root / identity).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("CZSC model path outside run storage")
    return path

def signal_time(value: Any) -> datetime:
    text = str(value)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if len(text) == 10:
        parsed = datetime.combine(parsed.date(), time(15), LOCAL_TZ)
    elif parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)
