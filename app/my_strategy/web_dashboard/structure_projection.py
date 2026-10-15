"""Explicit source-only presentation contract; no uninitialized portfolio targets."""
from copy import deepcopy


def structure_projection(result):
    out = deepcopy(result)
    capabilities = {"source_evidence": True, "direction_model": False,
                    "reference_position": False, "simulated_account": False}
    def project_row(row):
        row = deepcopy(row)
        for key in ("position_reference", "reference_recommendation", "model_gate",
                    "target_position_fraction", "previous_reference_target", "reference_action",
                    "target_change_action", "direction_probability", "direction_probabilities"):
            row.pop(key, None)
        row.update(structure_only=True, capabilities=capabilities, reference_points=[],
                   action="WAIT", emitted=False, reference_position_schema=None)
        row["agents"] = {key: value for key, value in row.get("agents", {}).items()
                         if key in {"structure", "wyckoff"}}
        if isinstance(row.get("consensus"), dict):
            row["consensus"] = project_row(row["consensus"])
        for event in row.get("source_reference_points", []) + row.get("events", []):
            if event.get("reference_layer") == "source" or event.get("source") in {"structure", "wyckoff"}:
                event["source_relationship_text"] = "已确认结构与量价来源；实际仓位变化在模拟账户中核验"
                event.pop("target_change_action", None)
                event.pop("reference_position_schema", None)
        return row
    out["structure_context"] = out.pop("research", {})
    out["capabilities"] = capabilities
    for key in ("position_reference", "reference_recommendation", "model_gate"):
        out.pop(key, None)
    out.update(structure_only=True, models_enabled=False, reference_points=[], direction_points=[],
               display_name="结构与量价来源", projection_version="structure_sources_v1")
    for key in ("trend_consensus", "consensus_decisions"):
        if isinstance(out.get(key), dict):
            section = out[key]
            section.update(capabilities=capabilities, structure_only=True, events=[])
            section["rows"] = [project_row(row) for row in section.get("rows", [])]
    evidence = out.get("chart_evidence", {})
    evidence.update(structure_only=True, capabilities=capabilities)
    evidence['reference_points'] = []
    evidence['direction_points'] = []
    statistics = evidence.get('statistics', {})
    for key in list(statistics):
        if key.startswith(('final_', 'reference_', 'target_')):
            statistics.pop(key)
    evidence["rows"] = [project_row(row) for row in evidence.get("rows", [])]
    out["next_session_decision"] = project_row(out.get("next_session_decision") or {})
    for events in (out.get("source_reference_points", []), evidence.get("source_reference_points", [])):
        for event in events:
            event["source_relationship_text"] = "已确认结构与量价来源；实际仓位变化在模拟账户中核验"
            event.pop("target_change_action", None)
            event.pop("reference_position_schema", None)
    out["warnings"] = [warning for warning in out.get("warnings", [])
                       if "最终B/S表示逐股参考目标" not in warning]
    out["warnings"].append("来源 B/S 是已确认结构与量价事件；持仓、减仓及成交在独立模拟账户中计算。")
    return out
