"""Native candidate-centred causal chart arbitration v4.

A fresh observable fact opens an assessment. Other agents assess that candidate
using today's verified state and bounded, currently revalidated original facts.
Current continuous opinions cannot create candidates or revive invalidated origins.
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

CONSENSUS_VERSION = "czsc_agent_consensus_v4"
CANDIDATE_POLICY = "independent_points_union_v4_native"
CANDIDATE_ENVELOPE = "risk_qualified_fresh_native_union_v4"
NATIVE_VERSIONS = {"czsc_native_evidence_v4_20261012", "pure_confirmed_czsc_v4_20261012", "wyckoff_current_pv_v4_20261012"}
WEIGHTED_VERSION = "czsc_agent_consensus_v2"
LEGACY_VERSION = "czsc_agent_consensus_v1"
SEMANTICS = _v2.SEMANTICS
SOURCES = _v2.SOURCES
_ZONE = _v2._ZONE
_DEFAULTS = {**copy.deepcopy(_v2._DEFAULTS), "version": CONSENSUS_VERSION, "evidence_valid_bars": 3}
_finite, _hash, _decimal, _sum = _v2._finite, _v2._hash, _v2._decimal, _v2._sum
_date, _instant, _observable = _v2._date, _v2._instant, _v2._observable
_codes, _side, _opportunity = _v2._codes, _v2._side, _v2._opportunity


def consensus_config(config=None, *, version=None):
    if version is not None and version != CONSENSUS_VERSION:
        raise ValueError("unknown consensus version")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("consensus configuration must be a mapping")
    if config is not None and config.get("version", CONSENSUS_VERSION) != CONSENSUS_VERSION:
        raise ValueError("conflicting consensus versions")
    if config is not None and set(config) - set(_DEFAULTS):
        raise ValueError("unknown consensus configuration keys")
    values = {**copy.deepcopy(_DEFAULTS), **copy.deepcopy(dict(config or {}))}
    window = _finite(values.pop("evidence_valid_bars"))
    if window is None or window != int(window) or int(window) not in {1, 3, 5}:
        raise ValueError("evidence_valid_bars must be 1, 3 or 5")
    values["version"] = WEIGHTED_VERSION
    values = _v2.consensus_config(values, version=WEIGHTED_VERSION)
    values.update(version=CONSENSUS_VERSION, evidence_valid_bars=int(window))
    return values


def _policy_value(mapping, key):
    mapping = mapping if isinstance(mapping, Mapping) else {}
    aliases = {"candidate_policy_hash": "signal_points_config_hash", "risk_policy_hash": "risk_plan_policy_hash"}
    return mapping.get(key) or mapping.get(aliases.get(key, key))


def _model_reasons(evaluation, provenance, *, source, day, close_at):
    identity = _v2._identity(evaluation)
    reasons = _v2._score_reasons(evaluation, day, close_at, require_lineage=True,
        expected_heads={"ma_entry"} if source == "ma" else {"entry"}, require_target=True)
    if identity.get("verified_frozen_contract") is not True or not all(identity.get(key) for key in
            ("model_run_id", "manifest_sha256", "target_contract_hash", "feature_schema_hash", "train_label_end")):
        reasons.append(source + "_frozen_model_contract_unverified")
    if not (identity.get("checkpoint") or identity.get("model_id")):
        reasons.append(source + "_checkpoint_identity_unavailable")
    if _date(evaluation.get("score_date", identity.get("score_date"))) != day:
        reasons.append(source + "_score_date_mismatch")
    if not (identity.get("model_head") or identity.get("head")):
        reasons.append(source + "_model_head_unavailable")
    expected_policy = provenance.get("candidate_policy_version") if isinstance(provenance, Mapping) else None
    if expected_policy != CANDIDATE_POLICY or identity.get("candidate_policy_version") != expected_policy:
        reasons.append(source + "_candidate_population_mismatch")
    for key in ("candidate_policy_hash", "risk_policy_hash"):
        expected = _policy_value(provenance, key)
        if not expected or _policy_value(identity, key) != expected:
            reasons.append(source + "_" + key + "_mismatch")
    return list(dict.fromkeys(reasons))


def _scope_reasons(evaluation, *, fresh_ids):
    envelope = evaluation.get("candidate_envelope")
    if isinstance(envelope, str):
        envelope = {"version": envelope, "in_scope": evaluation.get("in_scope"),
                    "candidate_event_ids": evaluation.get("candidate_event_ids")}
    if not isinstance(envelope, Mapping) or envelope.get("in_scope") is not True:
        return ["ma_candidate_population_out_of_scope"]
    if envelope.get("version", envelope.get("name")) != CANDIDATE_ENVELOPE:
        return ["ma_candidate_envelope_mismatch"]
    identifiers = set(_codes(envelope.get("candidate_event_ids")))
    if not identifiers or not identifiers.issubset(fresh_ids):
        return ["ma_candidate_envelope_event_mismatch"]
    return []


def _evaluation(row, bundle, source):
    evaluations = bundle.get("evaluations") if isinstance(bundle.get("evaluations"), Mapping) else {}
    evaluation = evaluations.get(source)
    if not isinstance(evaluation, Mapping):
        evaluation = (row.get("agents") or {}).get(source, {})
    evaluation = copy.deepcopy(dict(evaluation)) if isinstance(evaluation, Mapping) else {}
    native = row.get("native_evaluation")
    if source in {"structure", "wyckoff"} and isinstance(native, Mapping) and isinstance(native.get(source), Mapping):
        state = copy.deepcopy(dict(native[source]))
        state.setdefault("date", native.get("date"))
        if state.get("snapshot_id"):
            state["source_id"] = state["snapshot_id"]
        state.setdefault("available_at", native.get("available_at"))
        evaluation.update(direction_state=state, native_evidence_version=state.get("version", native.get("version")))
    # A native daily object may also be supplied directly under evaluations.
    if source in {"structure", "wyckoff"} and not isinstance(evaluation.get("direction_state"), Mapping) and isinstance(evaluation.get("current_state"), Mapping):
        evaluation["direction_state"] = copy.deepcopy(evaluation)
        if evaluation.get("snapshot_id"):
            evaluation["direction_state"]["source_id"] = evaluation["snapshot_id"]
    return evaluation


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
    if not state.get("source_id"):
        reasons.append("evidence_snapshot_identity_unavailable")
    current = state.get("current_state")
    if not isinstance(current, Mapping):
        reasons.append("current_state_unavailable")
    else:
        current_close = _finite(current.get("close"))
        if current_close is None or price is None or current_close != price:
            reasons.append("current_state_price_mismatch")
        for key in ("snapshot_at", "available_at", "structure_confirmed_at", "first_confirmed_at",
                    "origin_observed_at", "wyckoff_available_at", "wyckoff_range_formed_at"):
            if current.get(key) is not None and not _observable(current[key], close_at):
                reasons.append("current_state_not_observable")
    return state, list(dict.fromkeys(reasons))


def _fact_reasons(event, *, symbol, day, close_at):
    fact = dict(event)
    if event.get("direction_eligible") is True:
        fact["candidate_eligible"] = True
    reasons = _v2._event_reasons(fact, symbol=symbol, day=day, close_at=close_at)
    if event.get("input_eligible") is not True:
        reasons.append("event_input_unavailable")
    if not event.get("pattern_id") or not event.get("original_event_id"):
        reasons.append("original_pattern_identity_unavailable")
    for key in ("first_confirmed_at", "first_available_at", "origin_observed_at"):
        if event.get(key) is not None and not _observable(event[key], close_at):
            reasons.append("origin_event_not_observable")
    return list(dict.fromkeys(reasons))


def _opportunity(event):
    # Daily risk evaluations and new IDs cannot refresh the original pattern.
    pattern = event.get("pattern_id")
    if pattern:
        return "pattern:" + str(pattern) + ":" + str(event.get("activation_kind") or "confirmation") + ":" + str(_side(event))
    return _v2._opportunity(event)


def _original_fact_reasons(event, *, symbol, source, close_at):
    reasons = []
    if str(event.get("symbol", "")) != symbol or event.get("source") != source:
        reasons.append("foreign_support_fact")
    if not (event.get("event_id") and event.get("pattern_id") and event.get("original_event_id")):
        reasons.append("original_pattern_identity_unavailable")
    if _side(event) is None:
        reasons.append("unknown_event_direction")
    if not event.get("available_at") or not _observable(event["available_at"], close_at):
        reasons.append("support_origin_not_observable")
    for key in ("first_confirmed_at", "first_available_at", "origin_observed_at"):
        if event.get(key) is not None and not _observable(event[key], close_at):
            reasons.append("support_origin_not_observable")
    return list(dict.fromkeys(reasons))


def _assess(source, evaluation, active, *, day, close_at, price, config, provenance, fresh_ids, symbol, invalidated):
    agent = _v2._base_agent(source, evaluation)
    agent.update(probability=_finite(evaluation.get("probability"), unit=True),
        origin_event_ids=[], snapshot_id=None, evidence_mode="unavailable", current_state={},
        input_eligible=False, candidate_envelope=copy.deepcopy(evaluation.get("candidate_envelope")),
        in_scope=evaluation.get("in_scope"))
    state, reasons = _state_reasons(evaluation, day=day, close_at=close_at, price=price)
    if reasons:
        agent.update(status="input_ineligible", reason_codes=reasons)
        return agent, [], []
    current = state["current_state"]
    agent.update(snapshot_id=str(state["source_id"]), current_state=copy.deepcopy(current), input_eligible=True)
    support_facts = state.get("support_facts", evaluation.get("support_facts", []))
    support_facts = support_facts if isinstance(support_facts, (list, tuple)) else []
    candidates = [copy.deepcopy(fact["event"]) for fact in active.values() if fact["event"].get("source") == source
        and str(fact["event"]["event_id"]) not in invalidated and str(fact["event"].get("original_event_id")) not in invalidated]
    for event in support_facts:
        if not isinstance(event, Mapping):
            reasons.append("invalid_support_fact")
            continue
        errors = _original_fact_reasons(event, symbol=symbol, source=source, close_at=close_at)
        if errors:
            reasons.extend(errors)
            continue
        if str(event["event_id"]) not in invalidated and str(event.get("original_event_id")) not in invalidated:
            candidates.append(copy.deepcopy(event))
    origin_ids = set(_codes(state.get("origin_event_ids")))
    origin_ids.update(str(event["event_id"]) for event in support_facts if isinstance(event, Mapping) and event.get("event_id"))
    origin_ids.update(_codes(state.get("active_buy_event_ids")))
    origin_ids.update(_codes(state.get("active_sell_event_ids")))
    if any(code in reasons for code in ("foreign_support_fact", "original_pattern_identity_unavailable", "unknown_event_direction", "support_origin_not_observable")):
        agent.update(status="input_ineligible", reason_codes=list(dict.fromkeys(reasons)))
        return agent, [], []
    supported, snapshot_origins = [], set()
    native_version = evaluation.get("native_evidence_version", state.get("version"))
    native_ok = native_version in NATIVE_VERSIONS
    if source != "ma" and not native_ok:
        agent.update(status="unavailable", reason_codes=["native_evidence_contract_unverified"])
        return agent, [], []
    if source == "wyckoff":
        agent["strength"] = {"buy": _finite(current.get("buy_score"), unit=True),
                             "sell": _finite(current.get("sell_score"), unit=True)}
        agent["event_strength"] = {side.lower(): max((_finite(event.get("event_strength"), unit=True) or 0.
            for event in candidates if _side(event) == side and event.get("confirmed") is True), default=0.) for side in ("BUY", "SELL")}
        agent["direction_policy"] = "valid_confirmed_event_or_current_continuous_same_family_conflict_abstains"
    for side in ("BUY", "SELL"):
        if source != "wyckoff" and state.get(side.lower() + "_valid") is not True:
            continue
        matching = [event for event in candidates if _side(event) == side and
                    (str(event["event_id"]) in origin_ids or str(event.get("original_event_id")) in origin_ids)]
        if source == "ma":
            trend = [_finite(current.get(key)) for key in ("close_above_ma20", "ma20_slope5", "ma_alignment", "ret_5")]
            long_buy = all(value is not None for value in trend) and trend[0] > 0 and trend[1] > 0 and (trend[2] > 0 or trend[3] > 0)
            long_failure = all(value is not None for value in trend) and trend[0] <= 0 and trend[1] < 0
            gap = _finite(current.get("ma5_ma10_gap"))
            tactical_weakening = gap is not None and trend[3] is not None and gap < 0 and trend[3] < 0
            agent.update(trend_state="bullish" if long_buy else "bearish" if long_failure else "neutral",
                         tactical_weakening=tactical_weakening)
            if side == "BUY":
                errors = _scope_reasons(evaluation, fresh_ids=fresh_ids)
                if evaluation.get("status") != "scored" or agent["probability"] is None:
                    errors.append("ma_entry_score_unavailable")
                else:
                    errors.extend(_model_reasons(evaluation, provenance, source="ma", day=day, close_at=close_at))
                if current.get("ma_buy") is not True or not long_buy:
                    errors.append("ma_causal_long_trend_not_supported")
                if agent["probability"] is not None and agent["probability"] < config["ma_entry_threshold"]:
                    errors.append("ma_entry_below_threshold")
                if errors:
                    reasons.extend(errors)
                    continue
                agent.update(threshold=config["ma_entry_threshold"], in_scope=True)
                if tactical_weakening:
                    reasons.append("ma_short_tactical_weakening_with_bullish_long_trend")
            else:
                matching = [event for event in matching if event.get("event_type") == "ma_trend_failure"]
                if current.get("ma_sell") is not True or not long_failure:
                    reasons.append("ma_tactical_weakening_is_not_confirmed_long_trend_failure")
                    continue
                agent.update(type="causal_rule_evidence", direction_evidence="current_long_trend_failure", exit_probability=None)
                reasons.append("ma_current_long_trend_failure_rule_support")
        elif source == "structure":
            if current.get("support_valid" if side == "BUY" else "resistance_valid") is not True or not current.get("structure_confirmed_at"):
                reasons.append("structure_current_support_invalid")
                continue
            boundary = _finite(current.get("support_floor" if side == "BUY" else "resistance_ceiling"))
            if boundary is None or price is None or side == "BUY" and price <= boundary or side == "SELL" and price >= boundary:
                reasons.append("structure_current_price_invalidated")
                continue
        else:
            buy, sell = (agent["strength"][key] for key in ("buy", "sell"))
            strength, opposite = (buy, sell) if side == "BUY" else (sell, buy)
            continuous = (state.get(side.lower() + "_valid") is True and current.get("continuous_direction_valid") is True
                and strength is not None and opposite is not None and strength >= config["wyckoff_min_strength"] and strength > opposite)
            valid_events = []
            for event in matching:
                event_strength = _finite(event.get("event_strength"), unit=True)
                boundary = _finite(event.get("invalidation_price"))
                if (event.get("confirmed") is True and event_strength is not None and event_strength >= config["wyckoff_min_strength"]
                    and boundary is not None and price is not None and (price >= boundary if side == "BUY" else price <= boundary)):
                    valid_events.append(event)
            matching = valid_events
            if not continuous and not valid_events:
                reasons.append("wyckoff_" + side.lower() + "_event_and_continuous_support_below_threshold")
                continue
            reasons.append("wyckoff_valid_confirmed_event_support" if valid_events else "wyckoff_current_continuous_support")
            agent["threshold"] = config["wyckoff_min_strength"]
        if matching:
            supported.extend(matching)
            snapshot_origins.update(str(event["event_id"]) for event in matching)
        else:
            supported.append({"event_id": "snapshot:" + _hash({"source": source, "side": side,
                "date": day, "source_id": state["source_id"]}), "source": source, "side": side,
                "event_type": "current_verified_ma_snapshot" if source == "ma" else "current_native_snapshot", "dependencies": copy.deepcopy(state.get("dependencies", [])),
                "snapshot_only": True})
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
        agent["status"] = "out_of_scope" if "ma_candidate_population_out_of_scope" in reasons else "not_supported"
        reasons.append("no_current_direction_support")
    agent["evidence_mode"] = "current_snapshot" if supported and all(event.get("snapshot_only") for event in supported) else "continuous_state" if supported else "unavailable"
    agent["reason_codes"] = list(dict.fromkeys(reasons))
    return agent, supported, sorted(snapshot_origins)


def _raw_candidate_reasons(event, evaluation, *, symbol, day, close_at, price, invalidated):
    """Revalidate a raw opportunity independently of every expert's vote.

    Entry probabilities express opinion quality. A low score or an abstaining
    source cannot delete a still-valid original opportunity from the population
    on which the other independent experts and L2 were trained.
    """
    reasons = []
    identifier = str(event.get("event_id"))
    original = str(event.get("original_event_id"))
    if identifier in invalidated or original in invalidated:
        return ["fresh_original_fact_permanently_invalidated"]
    if "candidate_state_valid" in event:
        if event.get("candidate_state_valid") is not True:
            reasons.append("fresh_original_fact_current_state_invalid")
        if _date(event.get("candidate_state_date")) != day:
            reasons.append("candidate_state_date_mismatch")
        observed = event.get("candidate_state_available_at")
        if not observed or not _observable(observed, close_at) or _date(observed) != day:
            reasons.append("candidate_state_not_observable_today")
    if reasons:
        return reasons
    state, state_reasons = _state_reasons(evaluation, day=day, close_at=close_at, price=price)
    if state_reasons:
        return state_reasons
    side, source = _side(event), event.get("source")
    current = state["current_state"]
    if source == "ma":
        assisted = _v2._assisted_structure(event) or event.get("auxiliary") is True
        if assisted:
            # The shared source-validation adapter checks this original assisted
            # definition, since a MA trend snapshot cannot identify its pattern.
            if event.get("candidate_state_valid") is not True:
                return ["assisted_original_pattern_state_unverified"]
            return []
        values = [_finite(current.get(key)) for key in ("close_above_ma20", "ma20_slope5", "ma_alignment", "ret_5")]
        buy = all(value is not None for value in values) and values[0] > 0 and values[1] > 0 and (values[2] > 0 or values[3] > 0)
        sell = all(value is not None for value in values) and values[0] <= 0 and values[1] < 0
        if side == "BUY" and (current.get("ma_buy") is not True or not buy):
            return ["fresh_ma_long_trend_not_currently_valid"]
        if side == "SELL" and (current.get("ma_sell") is not True or not sell or event.get("event_type") != "ma_trend_failure"):
            return ["fresh_ma_long_failure_not_currently_valid"]
        return []
    native_version = evaluation.get("native_evidence_version", state.get("version"))
    if native_version not in NATIVE_VERSIONS:
        return ["native_evidence_contract_unverified"]
    support_facts = state.get("support_facts", evaluation.get("support_facts", []))
    support_facts = support_facts if isinstance(support_facts, (list, tuple)) else []
    matches = []
    for fact in support_facts:
        if not isinstance(fact, Mapping) or _side(fact) != side:
            continue
        original_match = str(fact.get("original_event_id") or fact.get("event_id")) == original
        pattern_match = str(fact.get("pattern_id")) == str(event.get("pattern_id"))
        if not original_match or not pattern_match:
            continue
        if _original_fact_reasons(fact, symbol=symbol, source=source, close_at=close_at):
            continue
        if str(fact.get("event_id")) in invalidated or str(fact.get("original_event_id")) in invalidated:
            continue
        if fact.get("confirmed") is not True or fact.get("input_eligible") is not True:
            continue
        boundary = _finite(fact.get("invalidation_price"))
        if boundary is None:
            boundary = _finite(fact.get("support_low" if side == "BUY" else "resistance_high"))
        if boundary is not None and price is not None and (price >= boundary if side == "BUY" else price <= boundary):
            matches.append(fact)
    if not matches:
        return ["fresh_original_native_pattern_not_currently_valid"]
    return []


def _quality_agent(evidence, provenance, *, day, close_at, config, has_buy_candidate):
    agent = _v2._base_agent("high_return", evidence)
    raw_status = str(evidence.get("status") or "model_unavailable")
    agent.update(type="model_quality_gate", probability=_finite(evidence.get("probability"), unit=True),
        threshold=config["high_return_threshold"], model_threshold=_finite(evidence.get("threshold"), unit=True),
        counts_as_vote=False, required=True, threshold_scope="consensus_research_quality_filter",
        raw_status=raw_status, in_scope=evidence.get("in_scope"))
    # Unscored stages keep their causal reason, without running score identity
    # checks against a checkpoint that never supplied a probability.
    statuses = {"no_candidate": "no_candidate", "candidate_risk_ineligible": "risk_blocked_unscored",
        "risk_ineligible": "risk_blocked_unscored", "risk_blocked_unscored": "risk_blocked_unscored",
        "feature_ineligible": "input_ineligible", "input_unavailable": "input_ineligible",
        "input_ineligible": "input_ineligible", "model_unavailable": "model_unavailable",
        "score_unavailable": "model_unavailable", "unavailable": "model_unavailable",
        "out_of_scope": "out_of_scope"}
    if raw_status != "scored" or agent["probability"] is None:
        status = statuses.get(raw_status, "model_unavailable")
        reasons = _codes(evidence.get("reason_codes")) or [raw_status if raw_status != "scored" else "high_return_score_unavailable"]
        agent.update(status=status, reason_codes=reasons)
        return agent
    if not has_buy_candidate:
        agent.update(status="no_candidate", reason_codes=["no_buy_candidate"])
        return agent
    reasons = _model_reasons(evidence, provenance, source="high_return", day=day, close_at=close_at)
    if evidence.get("in_scope") is not True:
        reasons.append("high_return_candidate_population_out_of_scope")
    if reasons:
        agent.update(status="identity_error", reason_codes=list(dict.fromkeys(reasons)))
    elif agent["probability"] < agent["threshold"]:
        agent.update(judgment="WAIT", status="rejected", reason_codes=["high_return_below_consensus_threshold"])
    else:
        agent.update(judgment="BUY", status="passed", reason_codes=["high_return_quality_gate_passed"])
    return agent


def build_consensus(rows, *, symbol, model_family="high_return", config=None, version=None):
    values = consensus_config(config, version=version)
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
    seen_opportunities, permanently_invalidated = set(), set()
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
        if not isinstance(bundle, Mapping) or bundle.get("version") != "czsc_candidate_evidence_v4":
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
            elif _opportunity(event) in seen_opportunities:
                ignored.append({"event_id": identifier, "reason_codes": ["pattern_already_assessed"]})
            elif values["source_weights"][source] > 0:
                seen_opportunities.add(_opportunity(event))
                fresh.append(event)
        evaluations = bundle.get("evaluations", {})
        if not isinstance(evaluations, Mapping):
            evaluations = {}
        invalidated_details = []
        for source in SOURCES:
            evaluation = _evaluation(row, bundle, source)
            validity, validation_reasons = _state_reasons(evaluation, day=day, close_at=close_at, price=price)
            # Explicit, causal invalidations are permanent within this run.
            # A missing input or a temporary weak score alone cannot erase a fact.
            if validation_reasons:
                continue
            for identifier in _codes(validity.get("invalidated_event_ids")):
                original = known.get(identifier)
                if not original or original.get("source") != source:
                    ignored.append({"event_id": identifier, "reason_codes": ["invalid_invalidation_reference"]})
                    continue
                permanently_invalidated.add(identifier)
                fact = active.get(identifier)
                if fact and fact["event"].get("source") == source:
                    invalidated_details.append({"event_id": identifier, "source": source,
                        "origin_date": _date(fact["event"].get("date")), "invalidated_at": available_at})
                    del active[identifier]
        if invalidated_details:
            reasons.append("active_evidence_permanently_invalidated")
        by_source, agents = {}, {}
        for source in SOURCES:
            evaluation = _evaluation(row, bundle, source)
            agents[source], by_source[source], _ = _assess(source, evaluation, active,
                day=day, close_at=close_at, price=price, config=values, provenance=bundle.get("provenance"),
                fresh_ids={str(event["event_id"]) for event in fresh}, symbol=symbol, invalidated=permanently_invalidated)
        # A raw fact opens an assessment, but only a currently revalidated
        # original fact can trigger the final point. An invalidated new fact
        # cannot borrow an older valid opinion merely because its source agrees.
        assessment_fresh = copy.deepcopy(fresh)
        qualified_fresh = []
        for event in fresh:
            identifier = str(event["event_id"])
            errors = _raw_candidate_reasons(event, _evaluation(row, bundle, event["source"]),
                symbol=symbol, day=day, close_at=close_at, price=price, invalidated=permanently_invalidated)
            if identifier in active and not errors:
                qualified_fresh.append(event)
            else:
                ignored.append({"event_id": identifier,
                    "reason_codes": ["fresh_candidate_invalidated_or_not_currently_supported", *errors]})
        fresh = qualified_fresh
        for source in SOURCES:
            own = [event for event in facts if isinstance(event, Mapping) and event.get("source") == source]
            agents[source]["raw_direction_counts"] = {side.lower(): sum(_side(event) == side and event.get("direction_eligible") is True for event in own) for side in ("BUY", "SELL")}
            agents[source]["risk_eligible_counts"] = {side.lower(): sum(_side(event) == side and event.get("candidate_eligible") is True for event in own) for side in ("BUY", "SELL")}
        families, dependency_reasons, contributions, family_details, family_conflicts, supporting_ids = _v2._supports(agents, by_source, values)
        reasons.extend(dependency_reasons)
        buy_support, sell_support = (_sum(families[side].values()) for side in ("BUY", "SELL"))
        has_buy_candidate = any(_side(event) == "BUY" for event in assessment_fresh)
        quality_evidence = _evaluation(row, bundle, "high_return")
        agents["high_return"] = _quality_agent(quality_evidence, bundle.get("provenance"),
            day=day, close_at=close_at, config=values, has_buy_candidate=has_buy_candidate)
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
        selected_origins = sorted({identifier for identifier in selected_ids if not identifier.startswith("snapshot:")})
        selected_snapshots = sorted({agents[source]["snapshot_id"] for source in SOURCES
            if agents[source]["snapshot_id"] and any(str(event["event_id"]) in selected_ids for event in by_source[source])})
        # An active supportive fact can belong to a consumed old opportunity;
        # only a fresh, qualified same-direction opportunity can emit again.
        # A valid fresh fact triggers a common assessment; its source need not
        # contribute a vote when the other independent experts support it.
        fresh_supporting = [event for event in fresh if _side(event) == proposed]
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
            fact = active.get(identifier)
            event = fact["event"] if fact else next(event for events in by_source.values() for event in events if str(event["event_id"]) == identifier)
            age = index - fact["index"] if fact else None
            active_details.append({"event_id": identifier, "source": event["source"], "side": _side(event),
                "origin_date": _date(event.get("date")), "age_bars": age,
                "remaining_bars": values["evidence_valid_bars"] - age if age is not None else None,
                "support_scope": "current_revalidated_state", "opportunity_id": _opportunity(event),
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
            "trigger_event_ids": sorted(str(event["event_id"]) for event in fresh_supporting),
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
                "anchor_dates": sorted({_date(event.get("anchor_date") or event.get("original_event_date") or event.get("date")) for events in by_source.values() for event in events if str(event["event_id"]) in selected_origins
                    if _date(event.get("anchor_date") or event.get("original_event_date") or event.get("date"))}), "evidence_snapshot_ids": selected_snapshots,
                "opportunity_ids": selected_opportunities, "trigger_event_ids": decision["trigger_event_ids"], "reference_price": price,
                "reference_price_date": day, "reference_price_source": "signal_day_close",
                "reference_price_basis": row.get("reference_price_basis", "unavailable"),
                "reason_codes": decision["reason_codes"], "config_hash": config_hash,
                "semantics": SEMANTICS, "consensus": copy.deepcopy(decision)})
    return {"version": CONSENSUS_VERSION, "semantics": SEMANTICS, "config": values,
        "config_hash": config_hash, "rows": decisions, "events": final_events,
        "funnel": {**dict(aggregate), "reason_counts": dict(sorted(reason_counts.items())), "bars": len(rows)},
        "initial_state": "neutral", "final_state": state}
