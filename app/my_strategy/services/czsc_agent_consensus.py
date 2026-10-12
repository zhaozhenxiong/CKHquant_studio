"""Version router for immutable v1/v2/v3 and native candidate arbitration v4.

Only a new unversioned request reads the editable default configuration. Saved
versioned requests replay with their frozen defaults and explicit saved overrides.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

from my_strategy.core.config_loader import load_config
from my_strategy.services import czsc_agent_consensus_v1 as _v1
from my_strategy.services import czsc_agent_consensus_v2 as _v2
from my_strategy.services import czsc_agent_consensus_v3 as _v3
from my_strategy.services import czsc_agent_consensus_v4 as _v4

CONSENSUS_VERSION = _v4.CONSENSUS_VERSION
CANDIDATE_VERSION = _v3.CONSENSUS_VERSION
WEIGHTED_VERSION = _v2.CONSENSUS_VERSION
LEGACY_VERSION = _v1.CONSENSUS_VERSION
SEMANTICS = _v2.SEMANTICS
SOURCES = _v2.SOURCES
_DEFAULTS = _v4._DEFAULTS
_MODULES = {LEGACY_VERSION: _v1, WEIGHTED_VERSION: _v2,
            CANDIDATE_VERSION: _v3, CONSENSUS_VERSION: _v4}


def consensus_config(config=None, *, version=None):
    if version is not None and (not isinstance(version, str) or version not in _MODULES):
        raise ValueError("unknown consensus version")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("consensus configuration must be a mapping")
    supplied = config.get("version") if config is not None else None
    if config is not None and "version" in config and (not isinstance(supplied, str) or supplied not in _MODULES):
        raise ValueError("unknown consensus configuration version")
    if version is not None and supplied is not None and supplied != version:
        raise ValueError("conflicting consensus versions")
    inferred = None
    if config is not None:
        if set(config) == set(_v1._DEFAULTS):
            inferred = LEGACY_VERSION
        elif set(config) == set(_v2._DEFAULTS) - {"version"}:
            inferred = WEIGHTED_VERSION
        elif set(config) == set(_v3._DEFAULTS) - {"version"}:
            inferred = CANDIDATE_VERSION
    selected = version or supplied or inferred
    values = copy.deepcopy(dict(config or {}))
    if selected is None:
        stored = load_config("czsc_agent_consensus")
        if not isinstance(stored, Mapping) or stored.get("version") not in _MODULES:
            raise ValueError("invalid consensus configuration file contract")
        selected = stored["version"]
        if set(stored) != set(_MODULES[selected]._DEFAULTS):
            raise ValueError("invalid consensus configuration file contract")
        if not isinstance(stored.get("source_weights"), Mapping) or set(stored["source_weights"]) != set(SOURCES):
            raise ValueError("invalid consensus configuration file source weights")
        values = {**copy.deepcopy(dict(stored)), **values}
        if config is not None and isinstance(config.get("source_weights"), Mapping):
            values["source_weights"] = {**stored["source_weights"], **config["source_weights"]}
    if selected == LEGACY_VERSION:
        values.pop("version", None)
        return _v1.consensus_config(values)
    return _MODULES[selected].consensus_config(values, version=selected)


def build_consensus(rows, *, symbol, model_family="high_return", config=None, version=None):
    values = consensus_config(config, version=version)
    selected = values.get("version", LEGACY_VERSION)
    module = _MODULES[selected]
    if selected == LEGACY_VERSION:
        return module.build_consensus(rows, symbol=symbol, model_family=model_family, config=values)
    return module.build_consensus(rows, symbol=symbol, model_family=model_family, config=values, version=selected)
