"""Candidate-centred causal chart arbitration, with frozen v1/v2 replay.

A fresh observable fact opens an assessment. Other agents assess that candidate
using today's verified state and bounded, currently revalidated original facts.
No account, inference, risk-plan reconstruction or trading policy is owned here.
"""
from __future__ import annotations

import copy
import datetime as dt
from collections import Counter
from collections.abc import Mapping
from decimal import Decimal

from my_strategy.core.config_loader import load_config
from my_strategy.services import czsc_agent_consensus_v1 as _v1
from my_strategy.services import czsc_agent_consensus_v2 as _v2

CONSENSUS_VERSION = "czsc_agent_consensus_v3"
WEIGHTED_VERSION = "czsc_agent_consensus_v2"
LEGACY_VERSION = "czsc_agent_consensus_v1"
SEMANTICS = _v2.SEMANTICS
SOURCES = _v2.SOURCES
_ZONE = _v2._ZONE
_DEFAULTS = {**copy.deepcopy(_v2._DEFAULTS), "version": CONSENSUS_VERSION, "evidence_valid_bars": 3}
_finite, _hash, _decimal, _sum = _v2._finite, _v2._hash, _v2._decimal, _v2._sum
_date, _instant, _observable = _v2._date, _v2._instant, _v2._observable
_codes, _side, _opportunity = _v2._codes, _v2._side, _v2._opportunity


def _selected_version(config, version):
    versions = {LEGACY_VERSION, WEIGHTED_VERSION, CONSENSUS_VERSION}
    if version is not None and (not isinstance(version, str) or version not in versions):
        raise ValueError("unknown consensus version")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("consensus configuration must be a mapping")
    supplied = config.get("version") if config is not None else None
    if config is not None and "version" in config and (not isinstance(supplied, str) or supplied not in versions):
        raise ValueError("unknown consensus configuration version")
    if version is not None and supplied is not None and supplied != version:
        raise ValueError("conflicting consensus versions")
    inferred = None
    if config is not None:
        if set(config) == set(_v1._DEFAULTS):
            inferred = LEGACY_VERSION
        elif set(config) == set(_v2._DEFAULTS) - {"version"}:
            inferred = WEIGHTED_VERSION
    return version or supplied or inferred or CONSENSUS_VERSION


def consensus_config(config=None, *, version=None):
    selected = _selected_version(config, version)
    if selected in {LEGACY_VERSION, WEIGHTED_VERSION}:
        return _v2.consensus_config(config, version=selected)
    stored = load_config("czsc_agent_consensus")
    if not isinstance(stored, Mapping) or set(stored) != set(_DEFAULTS) or stored.get("version") != CONSENSUS_VERSION:
        raise ValueError("invalid consensus configuration file contract")
    if not isinstance(stored["source_weights"], Mapping) or set(stored["source_weights"]) != set(SOURCES):
        raise ValueError("invalid consensus configuration file source weights")
    if config is not None and set(config) - set(_DEFAULTS):
        raise ValueError("unknown consensus configuration keys")
    values = {**copy.deepcopy(stored), **copy.deepcopy(dict(config or {}))}
    if isinstance(values.get("source_weights"), Mapping):
        values["source_weights"] = {**stored["source_weights"], **values["source_weights"]}
    window = _finite(values.pop("evidence_valid_bars"))
    if window is None or window != int(window) or int(window) not in {1, 3, 5}:
        raise ValueError("evidence_valid_bars must be 1, 3 or 5")
    values["version"] = WEIGHTED_VERSION
    values = _v2.consensus_config(values, version=WEIGHTED_VERSION)
    values.update(version=CONSENSUS_VERSION, evidence_valid_bars=int(window))
    return values


def _fact_reasons(event, *, symbol, day, close_at):
    # Risk eligibility is a separate final-entry gate. An explicit direction
    # fact may remain visible after an entry-risk rejection.
    fact = dict(event)
    if event.get("direction_eligible") is True:
        fact["candidate_eligible"] = True
    reasons = _v2._event_reasons(fact, symbol=symbol, day=day, close_at=close_at)
    if event.get("input_eligible") is not True:
        reasons.append("event_input_unavailable")
    return list(dict.fromkeys(reasons))


def _state_reasons(evaluation, *, day, close_at, price):
    state = evaluation.get("direction_state")
    if not isinstance(state, Mapping):
        return {}, ["evidence_state_unavailable"]
    reasons = []
    if _date(state.get("date")) != day:
        reasons.append("evidence_state_date_mismatch")
    if not state.get("available_at") or not _observable(state["available_at"], close_at):
        reasons.append("evidence_state_not_observable")
    if state.get("input_eligible") is not True or evaluation.get("input_eligible") is False:
        reasons.append("evidence_state_input_unavailable")
    if state.get("invalidated") is True:
        reasons.append("current_state_invalidated")
    if not state.get("source_id"):
        reasons.append("evidence_snapshot_identity_unavailable")
    current = state.get("current_state")
    if not isinstance(current, Mapping):
        reasons.append("current_state_unavailable")
    else:
        current_close = _finite(current.get("close"))
        if current_close is None or price is None or current_close != price:
            reasons.append("current_state_price_mismatch")
        for key in ("available_at", "structure_confirmed_at", "confirmed_at", "wyckoff_available_at",
                    "wyckoff_observed_at", "wyckoff_range_formed_at"):
            if current.get(key) is not None and not _observable(current[key], close_at):
                reasons.append("current_state_not_observable")
    return state, list(dict.fromkeys(reasons))


def _full_ma_identity_reasons(evaluation, *, day, close_at):
    reasons = _v2._score_reasons(evaluation, day, close_at, require_lineage=True,
                               expected_heads={"entry", "ma_entry"}, require_target=True)
    identity = _v2._identity(evaluation)
    # Availability alone cannot identify which model supplied a probability.
    if not (identity.get("checkpoint_id") or identity.get("model_id") or identity.get("checkpoint")):
        reasons.append("ma_full_identity_unavailable")
    if not (identity.get("checkpoint_hash") or identity.get("model_hash") or identity.get("model_sha256")
            or identity.get("checkpoint_sha256") or identity.get("artifact_hash") or identity.get("manifest_sha256")):
        reasons.append("ma_full_identity_unavailable")
    if not (identity.get("train_label_end") or identity.get("label_cutoff")):
        reasons.append("ma_full_identity_unavailable")
    if not (identity.get("head") or identity.get("model_head")):
        reasons.append("ma_full_identity_unavailable")
    if identity.get("verified_frozen_contract") is not True or not all(identity.get(key) for key in
            ("model_run_id", "target_contract_hash", "feature_schema_hash", "signal_points_config_hash", "risk_plan_policy_hash")):
        reasons.append("ma_frozen_model_contract_unverified")
    if _date(evaluation.get("score_date")) != day:
        reasons.append("score_date_mismatch")
    return list(dict.fromkeys(reasons))


def _policy_reasons(evidence, provenance, *, source):
    identity = _v2._identity(evidence)
    provenance = provenance if isinstance(provenance, Mapping) else {}
    if not all(provenance.get(key) and identity.get(key) == provenance[key]
               for key in ("signal_points_config_hash", "risk_plan_policy_hash")):
        return [source + "_candidate_risk_contract_mismatch"]
    return []


def _assess(source, evaluation, active, *, day, close_at, price, config, provenance):
    """Produce qualified current opinions and synthetic snapshot references.

    Synthetic IDs identify this assessment only; they do not create an
    opportunity. Original event IDs separately remain the auditable anchors.
    """
    agent = _v2._base_agent(source, evaluation)
    agent["probability"] = _finite(evaluation.get("probability"), unit=True)
    agent["strength"] = {"buy": _finite(evaluation.get("buy_score"), unit=True),
                         "sell": _finite(evaluation.get("sell_score"), unit=True)} if source == "wyckoff" else None
    agent.update(origin_event_ids=[], snapshot_id=None, evidence_mode="unavailable", current_state={})
    state, reasons = _state_reasons(evaluation, day=day, close_at=close_at, price=price)
    if reasons:
        agent.update(status="unavailable", reason_codes=reasons)
        return agent, [], []
    current = state["current_state"]
    agent.update(snapshot_id=str(state["source_id"]), current_state=copy.deepcopy(current))
    origins = _codes(state.get("origin_event_ids"))
    unknown = [identifier for identifier in origins if identifier not in active]
    if unknown:
        reasons.append("active_evidence_expired_or_invalid")
    supported = []
    snapshot_origins = set()
    for side in ("BUY", "SELL"):
        if state.get(side.lower() + "_valid") is not True:
            continue
        matching = [active[identifier]["event"] for identifier in origins if identifier in active
                    and active[identifier]["event"].get("source") == source
                    and _side(active[identifier]["event"]) == side]
        if source == "ma":
            if side == "BUY":
                errors = [*_full_ma_identity_reasons(evaluation, day=day, close_at=close_at),
                          *_policy_reasons(evaluation, provenance, source="ma")]
                if evaluation.get("status") != "scored" or agent["probability"] is None:
                    errors.append("ma_entry_score_unavailable")
                trend = [_finite(current.get(key)) for key in ("close_above_ma20", "ma20_slope5", "ma_alignment", "ret_5")]
                causal_buy = all(number is not None for number in trend) and trend[0] > 0 and trend[1] > 0 and (trend[2] > 0 or trend[3] > 0)
                if current.get("ma_buy") is not True or not causal_buy:
                    errors.append("ma_causal_trend_not_supported")
                if agent["probability"] is not None and agent["probability"] < config["ma_entry_threshold"]:
                    errors.append("ma_entry_below_threshold")
                if errors:
                    reasons.extend(errors)
                    continue
                agent["threshold"] = config["ma_entry_threshold"]
            else:
                matching = [event for event in matching if event.get("event_type") == "ma_trend_failure"]
                trend = [_finite(current.get(key)) for key in ("close_above_ma20", "ma20_slope5", "ma5_ma10_gap", "ret_5")]
                causal_sell = all(number is not None for number in trend) and (trend[0] <= 0 and trend[1] < 0 or trend[2] < 0 and trend[3] < 0)
                if current.get("ma_sell") is not True or not causal_sell or not matching:
                    reasons.append("ma_causal_sell_origin_unavailable")
                    continue
        elif source == "structure":
            if not matching or current.get("support_valid") is not True or current.get("invalidated") is True or not current.get("structure_confirmed_at"):
                reasons.append("structure_current_support_invalid")
                continue
            boundary = _finite(current.get("support_floor" if side == "BUY" else "resistance_ceiling"))
            if boundary is None or side == "BUY" and price <= boundary or side == "SELL" and price >= boundary:
                reasons.append("structure_current_price_invalidated")
                continue
        else:
            buy, sell = (_finite(current.get(key), unit=True) for key in ("buy_score", "sell_score"))
            strength, opposite = (buy, sell) if side == "BUY" else (sell, buy)
            agent["strength"] = {"buy": buy, "sell": sell}
            if not matching or current.get("current_direction_valid") is not True or not all(current.get(key) for key in ("wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at")):
                reasons.append("wyckoff_current_direction_not_supported")
                continue
            if strength is None or opposite is None or strength < config["wyckoff_min_strength"] or strength <= opposite:
                reasons.append("wyckoff_strength_below_threshold_or_opposing")
                continue
            agent["threshold"] = config["wyckoff_min_strength"]
        if source != "ma" and not matching:
            continue
        # Use each original fact's dependency so conflicting assisted versus
        # independent structure opinions retain their actual family.
        if matching:
            supported.extend(copy.deepcopy(matching))
            snapshot_origins.update(str(event["event_id"]) for event in matching)
        else:
            supported.append({"event_id": "snapshot:" + _hash({"source": source, "side": side,
                "date": day, "source_id": state["source_id"]}), "source": source, "side": side,
                "event_type": "current_verified_ma_snapshot", "dependencies": [], "snapshot_only": True})
    sides = {_side(event) for event in supported}
    agent["event_ids"] = sorted({str(event["event_id"]) for event in supported})
    agent["origin_event_ids"] = sorted(snapshot_origins)
    agent["opportunity_ids"] = sorted({_opportunity(event) for event in supported if not event.get("snapshot_only")})
    agent["supported_sides"] = sorted(sides)
    if len(sides) > 1:
        agent.update(judgment="CONFLICT", status="conflict")
        reasons.append("opposing_events_within_source")
    elif sides:
        agent.update(judgment=next(iter(sides)), status="supported")
        reasons.append("currently_revalidated_candidate_support")
    else:
        agent["status"] = "not_supported"
        reasons.append("no_current_direction_support")
    agent["evidence_mode"] = "current_snapshot" if supported and all(event.get("snapshot_only") for event in supported) else "active_event" if supported else "unavailable"
    agent["reason_codes"] = list(dict.fromkeys(reasons))
    return agent, supported, sorted(snapshot_origins)


def build_consensus(rows, *, symbol, model_family="high_return", config=None, version=None):
    selected = _selected_version(config, version)
    values = consensus_config(config, version=selected)
    if selected in {LEGACY_VERSION, WEIGHTED_VERSION}:
        return _v2.build_consensus(rows, symbol=symbol, model_family=model_family, config=values, version=selected)
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("consensus requires a symbol")
    if isinstance(rows, (Mapping, str, bytes)):
        raise ValueError("consensus rows must be a chronological sequence")
    rows = list(rows)
    previous = None
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("symbol", "")) != symbol:
            raise ValueError("consensus rows require the exact symbol")
        day = _date(row.get("date"))
        if day is None or str(row.get("date")) != day or previous is not None and day <= previous:
            raise ValueError("consensus rows require unique ascending ISO dates")
        previous = day
    config_hash = _hash(values)
    decisions, final_events = [], []
    active, known, consumed = {}, {}, set()
    last_emit = {"BUY": None, "SELL": None}
    state, last_direction = "neutral", None
    aggregate, reason_counts = Counter(), Counter()
    for index, row in enumerate(rows):
        day = row["date"]
        close_at = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(15), tzinfo=_ZONE)
        available_at = close_at.isoformat()
        reasons, ignored = [], []
        state_before = state
        price = _finite(row.get("reference_close", row.get("reference_price")))
        price_date = _date(row.get("reference_price_date"))
        exact_price = price is not None and price > 0 and price_date == day
        if not exact_price:
            reasons.append("exact_signal_day_reference_price_unavailable")
        if last_direction is not None and index - last_direction >= values["signal_valid_bars"]:
            state, last_direction = "neutral", None
            reasons.append("direction_evidence_expired")
        active = {identifier: fact for identifier, fact in active.items()
                  if index - fact["index"] < values["evidence_valid_bars"]}
        bundle = row.get("candidate_evidence")
        if not isinstance(bundle, Mapping) or bundle.get("version") != "czsc_candidate_evidence_v1":
            bundle = {}
            reasons.append("candidate_evidence_contract_unavailable")
        facts = bundle.get("facts", [])
        if not isinstance(facts, (list, tuple)):
            raise ValueError("candidate facts must be a sequence")
        daily_ids, fresh = set(), []
        observable_count = 0
        for event in facts:
            if not isinstance(event, Mapping):
                ignored.append({"event_id": None, "reason_codes": ["invalid_event"]})
                continue
            event_reasons = _fact_reasons(event, symbol=symbol, day=day, close_at=close_at)
            identifier = str(event.get("event_id") or "")
            source = event.get("source")
            if source not in SOURCES:
                event_reasons.append("unknown_evidence_source")
            if identifier and (identifier in known or identifier in daily_ids):
                event_reasons.append("duplicate_event_identity")
            daily_ids.add(identifier)
            if event_reasons:
                ignored.append({"event_id": identifier or None, "reason_codes": list(dict.fromkeys(event_reasons))})
                continue
            observable_count += 1
            known[identifier] = copy.deepcopy(event)
            active[identifier] = {"event": copy.deepcopy(event), "index": index}
            if _opportunity(event) in consumed:
                ignored.append({"event_id": identifier, "reason_codes": ["opportunity_already_consumed"]})
            elif values["source_weights"][source] > 0:
                fresh.append(event)
        evaluations = bundle.get("evaluations", {})
        if not isinstance(evaluations, Mapping):
            evaluations = {}
        invalidated_details = []
        for source in SOURCES:
            evaluation = evaluations.get(source)
            if not isinstance(evaluation, Mapping):
                continue
            validity, validation_reasons = _state_reasons(evaluation, day=day, close_at=close_at, price=price)
            # Explicit, causal invalidations are permanent within this run.
            # A missing input or a temporary weak score alone cannot erase a fact.
            if any(code != "current_state_invalidated" for code in validation_reasons):
                continue
            for identifier in _codes(validity.get("invalidated_event_ids")):
                fact = active.get(identifier)
                if fact and fact["event"].get("source") == source:
                    invalidated_details.append({"event_id": identifier, "source": source,
                        "origin_date": _date(fact["event"].get("date")), "invalidated_at": available_at})
                    del active[identifier]
                elif identifier not in known or known[identifier].get("source") != source:
                    ignored.append({"event_id": identifier, "reason_codes": ["invalid_invalidation_reference"]})
        if invalidated_details:
            reasons.append("active_evidence_permanently_invalidated")
        by_source, agents = {}, {}
        for source in SOURCES:
            evaluation = evaluations.get(source)
            evaluation = evaluation if isinstance(evaluation, Mapping) else {}
            agents[source], by_source[source], _ = _assess(source, evaluation, active,
                day=day, close_at=close_at, price=price, config=values, provenance=bundle.get("provenance"))
            if by_source[source]:
                indices = [active[event["event_id"]]["index"] for event in by_source[source] if not event.get("snapshot_only")]
                agents[source]["evidence_mode"] = "mixed" if indices and min(indices) < index <= max(indices) else "fresh_event" if indices and min(indices) == index else agents[source]["evidence_mode"]
        # A raw fact opens an assessment, but only a currently revalidated
        # original fact can trigger the final point. An invalidated new fact
        # cannot borrow an older valid opinion merely because its source agrees.
        revalidated_ids = {source: {str(event["event_id"]) for event in by_source[source]
                                   if not event.get("snapshot_only")} for source in SOURCES}
        qualified_fresh = []
        for event in fresh:
            identifier = str(event["event_id"])
            if identifier in active and identifier in revalidated_ids[event["source"]]:
                qualified_fresh.append(event)
            else:
                ignored.append({"event_id": identifier,
                    "reason_codes": ["fresh_candidate_invalidated_or_not_currently_supported"]})
        fresh = qualified_fresh
        families, dependency_reasons, contributions, family_details, family_conflicts, supporting_ids = _v2._supports(agents, by_source, values)
        reasons.extend(dependency_reasons)
        buy_support, sell_support = (_sum(families[side].values()) for side in ("BUY", "SELL"))
        has_buy_candidate = any(_side(event) == "BUY" for event in fresh)
        old_evidence = row.get("agents") if isinstance(row.get("agents"), Mapping) else {}
        quality_evidence = evaluations.get("high_return", old_evidence.get("high_return", {}))
        quality_evidence = quality_evidence if isinstance(quality_evidence, Mapping) else {}
        quality_contract_reasons = []
        identity = _v2._identity(quality_evidence)
        if has_buy_candidate:
            quality_contract_reasons.extend(_policy_reasons(quality_evidence, bundle.get("provenance"), source="high_return"))
            if identity.get("verified_frozen_contract") is not True or not all(identity.get(key) for key in
                    ("model_run_id", "checkpoint", "manifest_sha256", "target_contract_hash", "feature_schema_hash", "signal_points_config_hash", "risk_plan_policy_hash")):
                quality_contract_reasons.append("high_return_frozen_model_contract_unverified")
            if _date(quality_evidence.get("score_date", identity.get("score_date"))) != day:
                quality_contract_reasons.append("high_return_score_date_mismatch")
        agents["high_return"] = _v2._gate_agent(quality_evidence, day=day, close_at=close_at,
            config=values, required=True, has_buy_candidate=has_buy_candidate)
        if quality_contract_reasons:
            agents["high_return"].update(judgment="ABSTAIN", status="unavailable", reason_codes=quality_contract_reasons)
        quality_bonus = values["high_return_quality_weight"] if agents["high_return"]["status"] == "passed" else 0.
        buy_total = _sum([buy_support, quality_bonus])
        contributions["high_return"] = {"configured_weight": values["high_return_quality_weight"],
            "buy_contribution": quality_bonus, "sell_contribution": 0., "quality_contribution": quality_bonus, "families": []}
        agents["high_return"].update(copy.deepcopy(contributions["high_return"]))
        buy_pass = (_decimal(buy_support) >= _decimal(values["min_buy_base_support"])
            and _decimal(buy_support) - _decimal(sell_support) >= _decimal(values["min_direction_margin"])
            and len(families["BUY"]) >= values["min_independent_families"])
        sell_pass = (_decimal(sell_support) >= _decimal(values["min_sell_support"])
            and _decimal(sell_support) - _decimal(buy_support) >= _decimal(values["min_direction_margin"])
            and len(families["SELL"]) >= values["min_independent_families"])
        proposed = "BUY" if buy_pass else "SELL" if sell_pass else "WAIT"
        if proposed == "WAIT":
            if not (len(families["BUY"]) >= values["min_independent_families"] or len(families["SELL"]) >= values["min_independent_families"]):
                reasons.append("insufficient_independent_families")
            if buy_support and _decimal(buy_support) < _decimal(values["min_buy_base_support"]):
                reasons.append("buy_base_support_below_threshold")
            if sell_support and _decimal(sell_support) < _decimal(values["min_sell_support"]):
                reasons.append("sell_base_support_below_threshold")
            reasons.append("direction_conflict_or_insufficient_margin" if buy_support and sell_support else "insufficient_independent_support")
        action = proposed
        if not fresh:
            reasons.append("no_fresh_candidate_trigger")
            action = "WAIT"
        elif proposed in {"BUY", "SELL"} and not any(_side(event) == proposed for event in fresh):
            reasons.append("fresh_candidate_direction_mismatch")
            action = "WAIT"
        risk = _v2._risk_reasons(row.get("raw_risk_plan"), symbol=symbol, day=day, close_at=close_at, required=True)
        if proposed == "BUY":
            reasons.extend(risk)
            if risk:
                action = "WAIT"
            if agents["high_return"]["status"] != "passed":
                reasons.extend(agents["high_return"]["reason_codes"])
                action = "WAIT"
            if _decimal(buy_total) < _decimal(values["min_buy_support"]):
                reasons.append("buy_total_support_below_threshold")
                action = "WAIT"
        if not exact_price:
            action = "WAIT"
        selected_ids = sorted(supporting_ids.get(proposed, set()))
        selected_origins = sorted({identifier for identifier in selected_ids if identifier in known})
        selected_snapshots = sorted({agents[source]["snapshot_id"] for source in SOURCES
            if agents[source]["snapshot_id"] and any(str(event["event_id"]) in selected_ids for event in by_source[source])})
        # An active supportive fact can belong to a consumed old opportunity;
        # only a fresh, qualified same-direction opportunity can emit again.
        fresh_supporting = [event for event in fresh if _side(event) == proposed
            and (str(event["event_id"]) in selected_ids or agents[event["source"]]["judgment"] == proposed)]
        selected_opportunities = sorted({_opportunity(event) for event in fresh_supporting})
        if action in {"BUY", "SELL"} and not selected_opportunities:
            reasons.append("no_fresh_qualified_opportunity")
            action = "WAIT"
        if action in {"BUY", "SELL"} and last_emit[action] is not None and index - last_emit[action] < values["cooldown_bars"]:
            reasons.append("independent_opportunity_cooldown")
            action = "WAIT"
        emitted = action in {"BUY", "SELL"}
        if emitted:
            state, last_direction = ("bullish" if action == "BUY" else "bearish"), index
            last_emit[action] = index
            consumed.update(selected_opportunities)
            reasons.append("multi_agent_direction_confirmed")
        active_details = []
        supported_origins = {str(event["event_id"]) for events in by_source.values() for event in events if not event.get("snapshot_only")}
        for identifier in sorted(supported_origins):
            fact = active[identifier]
            event = fact["event"]
            age = index - fact["index"]
            active_details.append({"event_id": identifier, "source": event["source"], "side": _side(event),
                "origin_date": _date(event.get("date")), "age_bars": age,
                "remaining_bars": values["evidence_valid_bars"] - age, "opportunity_id": _opportunity(event),
                "dependencies": copy.deepcopy(event.get("dependencies", [])), "revalidated": True})
        funnel = {"raw_events": len(facts), "observable_events": observable_count, "fresh_candidates": len(fresh),
            "active_evidence": len(active_details), "directional_sources": sum(agents[source]["judgment"] in {"BUY", "SELL"} for source in SOURCES),
            "independent_buy_families": len(families["BUY"]), "independent_sell_families": len(families["SELL"]),
            "l1_qualified": int(proposed != "WAIT" and bool(selected_opportunities)), "risk_eligible": int(has_buy_candidate and not risk),
            "quality_passed": int(agents["high_return"]["status"] == "passed"), "final_emitted": int(emitted)}
        aggregate.update(funnel)
        reason_counts.update(list(dict.fromkeys(reasons)))
        agents["arbiter"] = {"judgment": action, "type": "direction_arbitration", "status": "emitted" if emitted else "wait",
            "reason_codes": list(dict.fromkeys(reasons)), "probability": None, "exit_probability": None,
            "counts_as_vote": False, "semantics": SEMANTICS, "configured_weight": 0.,
            "buy_contribution": 0., "sell_contribution": 0., "quality_contribution": 0.}
        decision = {"date": day, "symbol": symbol, "action": action, "proposed_action": proposed,
            "reason_codes": list(dict.fromkeys(reasons)), "agents": agents, "version": CONSENSUS_VERSION,
            "buy_support": buy_support, "sell_support": sell_support,
            "buy_base_score": buy_support, "sell_base_score": sell_support,
            "buy_quality_bonus": quality_bonus, "buy_total_score": buy_total,
            "direction_margin": float(abs(_decimal(buy_support) - _decimal(sell_support))),
            "contributions": contributions, "family_details": family_details, "family_conflicts": family_conflicts,
            "buy_independent_families": sorted(families["BUY"]), "sell_independent_families": sorted(families["SELL"]),
            "buy_independent_count": len(families["BUY"]), "sell_independent_count": len(families["SELL"]),
            "evidence_event_ids": selected_origins, "evidence_snapshot_ids": selected_snapshots,
            "opportunity_ids": selected_opportunities, "candidate_event_ids": sorted(str(event["event_id"]) for event in fresh),
            "active_evidence": active_details, "invalidated_evidence": invalidated_details, "funnel": funnel, "raw_risk_plan": copy.deepcopy(row.get("raw_risk_plan")),
            "state_before": state_before, "state_after": state, "emitted": emitted,
            "reference_price": price, "reference_price_date": price_date, "available_at": available_at,
            "config_hash": config_hash, "semantics": SEMANTICS, "ignored_events": ignored}
        decisions.append(decision)
        if emitted:
            identifier = _hash({"version": CONSENSUS_VERSION, "symbol": symbol, "date": day,
                "action": action, "config_hash": config_hash, "event_ids": selected_origins,
                "snapshot_ids": selected_snapshots})
            final_events.append({"event_id": identifier, "symbol": symbol, "date": day, "time": day,
                "evidence_date": day, "signal_at": available_at, "available_at": available_at,
                "first_available_at": available_at, "source_frequency": "日线",
                "source": "consensus", "side": action, "action": action, "event_type": "multi_agent_direction",
                "version": CONSENSUS_VERSION, "candidate_eligible": True, "evidence_visible": True,
                "original_event_ids": selected_origins, "source_event_ids": selected_origins,
                "anchor_dates": sorted({_date(known[event_id].get("anchor_date") or known[event_id].get("original_event_date") or known[event_id].get("date")) for event_id in selected_origins
                    if _date(known[event_id].get("anchor_date") or known[event_id].get("original_event_date") or known[event_id].get("date"))}), "evidence_snapshot_ids": selected_snapshots,
                "opportunity_ids": selected_opportunities, "reference_price": price,
                "reference_price_date": day, "reference_price_source": "signal_day_close",
                "reference_price_basis": row.get("reference_price_basis", "unavailable"),
                "reason_codes": decision["reason_codes"], "config_hash": config_hash,
                "semantics": SEMANTICS, "consensus": copy.deepcopy(decision)})
    return {"version": CONSENSUS_VERSION, "semantics": SEMANTICS, "config": values,
        "config_hash": config_hash, "rows": decisions, "events": final_events,
        "funnel": {**dict(aggregate), "reason_counts": dict(sorted(reason_counts.items())), "bars": len(rows)},
        "initial_state": "neutral", "final_state": state}
