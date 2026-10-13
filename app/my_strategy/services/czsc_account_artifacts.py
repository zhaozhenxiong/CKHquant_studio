"""Lossless, relocatable evidence for actual Broker accounts.

The familiar account JSON paths contain small, explicit references to one
immutable compressed graph. Equal ledger prefixes, account states, model
outputs and trade aliases share graph nodes before compression. Reading an
artifact restores the same JSON object that the former ``_save`` wrote;
independent copies are returned for repeated references so mutations do not
change another observation. Trading logic and training labels are untouched.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_support import _json


ACCOUNT_REFERENCE_SCHEMA = "czsc_account_reference_v1"
ACCOUNT_GRAPH_SCHEMA = "czsc_account_graph_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_GRAPH_BYTES = 512 * 1024 * 1024
_MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_NODES = 2_000_000


def _dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _filename(value: Any) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or "/" in value or "\\" in value or "\x00" in value):
        raise ValueError("account artifact requires a contained relative filename")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("account artifact requires a SHA-256 digest")
    return value


def _atomic(path: Path, value: bytes) -> None:
    if path.is_symlink():
        raise ValueError("account artifact destination cannot be a symbolic link")
    handle = tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class _Encoder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.interned: dict[bytes, int] = {}
        self.containers = 0

    def encode(self, value: Any) -> Any:
        if isinstance(value, dict):
            node = {"d": {key: self.encode(value[key]) for key in sorted(value)}}
        elif isinstance(value, list):
            node = {"l": [self.encode(item) for item in value]}
        else:
            return value
        self.containers += 1
        signature = _dump(node)
        index = self.interned.get(signature)
        if index is None:
            index = len(self.nodes)
            self.interned[signature] = index
            self.nodes.append(node)
        return {"ref": index}


def write_account_artifacts(directory: str | Path, documents: dict[str, Any]) -> dict[str, Any]:
    """Write logical account documents together, sharing all repeated evidence.

    ``documents`` keys are the existing ``*.json`` basenames. The returned
    file index includes every reference and its shared graph; it can be merged
    directly into a dataset manifest. No source or old artifact is rewritten.
    """
    if not documents:
        raise ValueError("account evidence requires at least one document")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in documents:
        if not _filename(name).endswith(".json"):
            raise ValueError("account evidence document must use a JSON basename")
    encoder = _Encoder()
    roots = {}
    evidence = {}
    for name in sorted(documents):
        logical = _json(documents[name])
        expanded = _dump(logical)
        if len(expanded) > _MAX_EXPANDED_BYTES:
            raise ValueError("account logical evidence exceeds the supported size")
        roots[name] = encoder.encode(logical)
        evidence[name] = {"expanded_sha256": _sha(expanded),
                          "semantic_sha256": stable_hash(logical),
                          "expanded_bytes": len(expanded)}
    graph = _dump({"schema": ACCOUNT_GRAPH_SCHEMA, "objects": encoder.nodes,
                   "documents": roots})
    if len(graph) > _MAX_GRAPH_BYTES or len(encoder.nodes) > _MAX_NODES:
        raise ValueError("account graph exceeds the supported size")
    compressed = io.BytesIO()
    # An empty filename and zero mtime make the immutable asset deterministic.
    with gzip.GzipFile(filename="", fileobj=compressed, mode="wb", mtime=0,
                       compresslevel=6) as handle:
        handle.write(graph)
    content = compressed.getvalue()
    asset_sha = _sha(content)
    asset_name = "account_evidence." + asset_sha + ".json.gz"
    asset_path = directory / asset_name
    if asset_path.is_symlink():
        raise ValueError("account graph cannot be a symbolic link")
    if asset_path.exists():
        if _sha(asset_path.read_bytes()) != asset_sha:
            raise ValueError("existing immutable account graph is corrupted")
    else:
        _atomic(asset_path, content)
    files = {asset_name: {"path": str(asset_path), "sha256": asset_sha,
                          "bytes": len(content), "schema": ACCOUNT_GRAPH_SCHEMA}}
    for name in sorted(documents):
        descriptor = {"schema": ACCOUNT_REFERENCE_SCHEMA, "document": name,
                      "asset": asset_name, "asset_sha256": asset_sha,
                      "graph_sha256": _sha(graph), "graph_bytes": len(graph),
                      **evidence[name]}
        value = _dump(descriptor)
        path = directory / name
        _atomic(path, value)
        files[name] = {"path": str(path), "sha256": _sha(value), "bytes": len(value),
                       "schema": ACCOUNT_REFERENCE_SCHEMA, **evidence[name]}
    stats = {"schema": ACCOUNT_REFERENCE_SCHEMA, "graph_schema": ACCOUNT_GRAPH_SCHEMA,
             "deduplication": "equal_composite_content_including_ledger_prefixes_and_model_outputs",
             "compression": "gzip", "logical_document_count": len(documents),
             "expanded_canonical_bytes": sum(v["expanded_bytes"] for v in evidence.values()),
             "unique_graph_nodes": len(encoder.nodes), "composite_occurrences": encoder.containers,
             "deduplicated_composite_occurrences": encoder.containers - len(encoder.nodes),
             "graph_canonical_bytes": len(graph), "graph_compressed_bytes": len(content),
             "stored_bytes": sum(v["bytes"] for v in files.values()),
             "physical_bytes": sum((directory / name).stat().st_blocks * 512 for name in files),
             "asset": asset_name, "asset_sha256": asset_sha}
    return {"files": files, "storage_stats": stats}


def write_account_artifact(path: str | Path, value: Any) -> dict[str, Any]:
    """Write one account, using the same format as shared dataset evidence."""
    path = Path(path)
    return write_account_artifacts(path.parent, {path.name: value})


def _parse(content: bytes) -> Any:
    def reject(value):
        raise ValueError("non-finite number in account evidence: " + value)
    try:
        return json.loads(content, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid account evidence JSON") from error


def _check_token(token: Any, limit: int) -> None:
    if isinstance(token, dict):
        index = token.get("ref")
        if (set(token) != {"ref"} or type(index) is not int or not 0 <= index < limit):
            raise ValueError("invalid or cyclic account graph reference")
    elif token is not None and not isinstance(token, (str, bool, int, float)):
        raise ValueError("invalid account graph scalar")
    elif isinstance(token, float) and not math.isfinite(token):
        raise ValueError("non-finite account graph scalar")


def _load(path: str | Path, expected_sha256: str | None, allow_plain: bool):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("account reference cannot be a symbolic link")
    content = path.read_bytes()
    if expected_sha256 is not None and _sha(content) != _digest(expected_sha256):
        raise ValueError("account reference SHA-256 mismatch")
    descriptor = _parse(content)
    if not isinstance(descriptor, dict) or descriptor.get("schema") != ACCOUNT_REFERENCE_SCHEMA:
        if allow_plain and not (isinstance(descriptor, dict) and "schema" in descriptor):
            return descriptor, None
        raise ValueError("unsupported account artifact schema; plain JSON is archive-only")
    expected = {"schema", "document", "asset", "asset_sha256", "graph_sha256", "graph_bytes",
                "expanded_sha256", "semantic_sha256", "expanded_bytes"}
    if set(descriptor) != expected:
        raise ValueError("invalid account reference fields")
    document = _filename(descriptor["document"])
    if document != path.name:
        raise ValueError('account reference document does not match its filename')
    asset_name = _filename(descriptor["asset"])
    for key in ("asset_sha256", "graph_sha256", "expanded_sha256", "semantic_sha256"):
        _digest(descriptor[key])
    if asset_name != "account_evidence." + descriptor["asset_sha256"] + ".json.gz":
        raise ValueError("account graph filename does not bind its content")
    for key, maximum in (("graph_bytes", _MAX_GRAPH_BYTES), ("expanded_bytes", _MAX_EXPANDED_BYTES)):
        if type(descriptor[key]) is not int or not 0 <= descriptor[key] <= maximum:
            raise ValueError("invalid account evidence size")
    asset = path.parent / asset_name
    if asset.is_symlink() or asset.resolve().parent != path.parent.resolve():
        raise ValueError("account graph escapes its reference directory")
    compressed = asset.read_bytes()
    if _sha(compressed) != descriptor["asset_sha256"]:
        raise ValueError("account graph SHA-256 mismatch")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as handle:
            graph_bytes = handle.read(descriptor["graph_bytes"] + 1)
    except (OSError, EOFError) as error:
        raise ValueError("invalid compressed account graph") from error
    if len(graph_bytes) != descriptor["graph_bytes"] or _sha(graph_bytes) != descriptor["graph_sha256"]:
        raise ValueError("account graph payload SHA-256 or size mismatch")
    graph = _parse(graph_bytes)
    if (not isinstance(graph, dict) or set(graph) != {"schema", "objects", "documents"}
            or graph["schema"] != ACCOUNT_GRAPH_SCHEMA):
        raise ValueError("unsupported account graph schema")
    nodes = graph["objects"]
    roots = graph["documents"]
    if not isinstance(nodes, list) or len(nodes) > _MAX_NODES or not isinstance(roots, dict):
        raise ValueError("invalid account graph tables")
    sizes = []
    depths = []
    def token_size(token):
        return sizes[token['ref']] if isinstance(token, dict) else len(_dump(token))
    def token_depth(token):
        return depths[token['ref']] if isinstance(token, dict) else 0
    for index, node in enumerate(nodes):
        if isinstance(node, dict) and set(node) == {"d"} and isinstance(node["d"], dict):
            tokens = node["d"].values()
        elif isinstance(node, dict) and set(node) == {"l"} and isinstance(node["l"], list):
            tokens = node["l"]
        else:
            raise ValueError("invalid account graph node")
        for token in tokens:
            _check_token(token, index)
        if 'd' in node:
            size = 2 + max(0, len(node['d']) - 1) + sum(
                len(_dump(key)) + 1 + token_size(value) for key, value in node['d'].items())
        else:
            size = 2 + max(0, len(node['l']) - 1) + sum(token_size(value) for value in node['l'])
        depth = 1 + max((token_depth(token) for token in tokens), default=0)
        if size > _MAX_EXPANDED_BYTES or depth > 256:
            raise ValueError('account graph expands beyond supported size or depth')
        sizes.append(size)
        depths.append(depth)
    for name, root in roots.items():
        _filename(name)
        _check_token(root, len(nodes))
    if document not in roots:
        raise ValueError("account graph document is missing")
    if token_size(roots[document]) != descriptor['expanded_bytes']:
        raise ValueError('account logical size does not match the verified graph')
    return descriptor, graph


def validate_account_artifact(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Verify stored bytes, schemas and graph references without expansion.

    This quick integrity check does not recompute the logical object hash;
    ``read_account_artifact`` performs that full semantic check.
    """
    descriptor, graph = _load(path, expected_sha256, False)
    return {**descriptor, "verified": True, "expanded_semantics_verified": False,
            "graph_nodes": len(graph["objects"])}


def read_account_artifact(path: str | Path, expected_sha256: str | None = None,
                          *, allow_plain: bool = False) -> Any:
    """Restore lossless JSON evidence; old JSON requires explicit archive use."""
    descriptor, graph = _load(path, expected_sha256, allow_plain)
    if graph is None:
        return descriptor
    nodes = graph["objects"]

    def decode(token):
        if not isinstance(token, dict):
            return token
        node = nodes[token["ref"]]
        # Each appearance expands independently, matching a normal JSON read.
        if "d" in node:
            return {key: decode(value) for key, value in node["d"].items()}
        return [decode(value) for value in node["l"]]

    value = decode(graph["documents"][descriptor["document"]])
    expanded = _dump(value)
    if (len(expanded) != descriptor["expanded_bytes"]
            or _sha(expanded) != descriptor["expanded_sha256"]
            or stable_hash(value) != descriptor["semantic_sha256"]):
        raise ValueError("restored account evidence semantic SHA-256 mismatch")
    return value
