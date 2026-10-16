"""Display-only scan projections, separate from numeric source identities."""
def _scan_display_metadata(value):
    """Keep saved display identities and scores, omitting raw model inputs/contracts."""
    if not isinstance(value, dict):
        return value
    retained = {key: item for key, item in value.items()
                if not isinstance(item, (dict, list))}
    for key in ("reason_codes", "reasons", "model_input_reason_codes", "original_opportunity_ids",
                "original_event_ids", "origin_event_ids", "sources", "point_types", "label_target", "expert_probabilities",
                "native_labels", "source_boundaries", "source_ids", "source_event_ids"):
        if key in value:
            retained[key] = value[key]
    for key in ("model_identity", "fallback_model", "rule_identity", "current_state", "current_valid_zone",
                "reference_price_band", "reference_risk_plan", "position_reference", "reference_quality_identity"):
        if key in value:
            retained[key] = _scan_display_metadata(value[key])
    if "model_availability_diagnostics" in value:
        def availability(record):
            if not isinstance(record, dict):
                return record
            result = _scan_display_metadata({key: item for key, item in record.items()
                                            if key != "model_availability_diagnostics"})
            if "candidates" in record:
                result["candidates"] = [_scan_display_metadata(item) for item in record["candidates"]]
            if "dual_fallback" in record:
                result["dual_fallback"] = availability(record["dual_fallback"])
            return result
        retained["model_availability_diagnostics"] = availability(value["model_availability_diagnostics"])
    # Targets are rendered directly, including older descriptive target objects.
    if "target" in value:
        retained["target"] = value["target"]
    return retained


_SCAN_EVENT_FIELDS = frozenset(("event_id", "opportunity_id", "original_event_id", "original_pattern_id",
    "symbol", "source", "source_frequency", "event_type", "signal", "name", "side", "action", "time",
    "anchor_at", "available_at", "first_available_at", "first_confirmed_at", "origin_observed_at",
    "signal_at", "observed_at", "confirmed_at", "reference_price_date", "reference_price_source",
    "point_type", "confirmed", "input_eligible", "candidate_eligible", "direction_eligible",
    "risk_candidate_eligible", "risk_reason_codes", "reason_codes", "reason", "reference_price",
    "stop_price", "risk_stop_price", "stop_distance", "price_floor", "price_ceiling", "target_weight", "sources",
    "candidate_direction_valid", "evidence_visible", "pattern_id", "version", "family", "rule_name",
    "is_confirmed", "invalidates_event_id", "reference_layer", "source_label",
    "consensus_action", "consensus_qualified", "consensus_reason_codes", "reference_only",
    "signal_date", "date", "account_scope", "analysis_mode"))


def _scan_events(values):
    if not isinstance(values, list):
        return values
    return [{key: item for key, item in value.items() if key in _SCAN_EVENT_FIELDS}
            if isinstance(value, dict) else value for value in values]


def _scan_row_view(row):
    if not isinstance(row, dict):
        return row
    result = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
    for key in ("reason_codes", "original_opportunity_ids"):
        if key in row:
            result[key] = row[key]
    for key in ("model_resolution", "exit_model", "policy_entry_model", "reference_recommendation",
                "reference_risk_plan", "entry_plan", "raw_risk_plan", "position_reference", "reference_quality_identity"):
        if key in row:
            result[key] = _scan_display_metadata(row[key])
    if isinstance(row.get('method_evaluation'), dict):
        result['method_evaluation'] = {key: value for key, value in row['method_evaluation'].items()
                                       if key != 'source_identity'}
    if isinstance(row.get('method_context'), dict): result['method_context'] = row['method_context']
    if isinstance(row.get("agents"), dict):
        result["agents"] = {key: _scan_display_metadata(value) for key, value in row["agents"].items()}
    candidate = row.get("candidate_evidence")
    if isinstance(candidate, list):
        result["candidate_evidence"] = _scan_events(candidate)
    elif isinstance(candidate, dict):
        result["candidate_evidence"] = {key: value for key, value in candidate.items()
                                        if not isinstance(value, (dict, list))}
        for key in ("facts", "risk_details"):
            if key in candidate:
                result["candidate_evidence"][key] = _scan_events(candidate[key])
    for key in ("events", "signal_points", "source_reference_points", "reference_points"):
        if key in row:
            result[key] = _scan_events(row[key])
    return result


def compact_scan_view(result):
    """Project current reference evidence without changing the saved report."""
    return {**result, "rows": [_scan_row_view(row) for row in result.get("rows", [])],
            "scan_view": {"schema": "current_scan_view", "compact": True,
                          "analysis_mode": "reference", "raw_evidence": "per_stock_analysis"}}
