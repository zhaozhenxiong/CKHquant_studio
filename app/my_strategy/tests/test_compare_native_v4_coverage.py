import pandas as pd

from my_strategy.scripts.compare_native_v4_coverage import candidate_coverage, compare_candidates, summarize_entries


def test_overlapping_source_members_do_not_inflate_union():
    candidate = {"sources": ["ma", "structure"], "candidate_eligible": True, "selected_by_reference": True,
                 "source_eligibility": {"ma": True, "structure": False},
                 "source_selected_by_reference": {"ma": True},
                 "source_reasons": {"structure": ["stop_distance"]}}
    result = candidate_coverage([candidate])
    assert result["union_opportunities"] == result["union_risk_qualified"] == 1
    assert sum(row["member_opportunities"] for row in result["by_source_overlapping_membership"].values()) == 2
    assert result["by_source_overlapping_membership"]["structure"]["source_risk_qualified"] == 0


def test_pending_and_existing_inventory_are_not_negative_labels():
    frame = pd.DataFrame({"label_available": [True, False, False], "label": [1., None, None],
                          "input_eligible": [True, True, False], "selected_by_reference": [True, True, False],
                          "risk_available": [True, None, False], "source_ma": [1, 1, 1],
                          "source_structure": [1, 0, 0], "source_wyckoff": [0, 0, 0],
                          "label_reason": ["available", "pending", "existing_inventory"],
                          "outcome_class": ["high_return", None, None]})
    result = summarize_entries(frame)
    assert result["mature"] == result["positive"] == 1
    assert result["mature_positive_fraction"] == 1
    assert result["unavailable_rows"] == 2
    assert result["by_source_overlapping_membership"]["ma"]["unavailable_reasons"] == {"pending": 1, "existing_inventory": 1}


def test_risk_arms_match_stock_date_with_distinct_opportunity_ids():
    shared = {"sources": ["ma"], "source_eligibility": {"ma": True}, "candidate_eligible": True, "stop_price": 10}
    main = {"ABC:2026-01-01": {**shared, "opportunity_id": "new"}}
    control = {"ABC:2026-01-01": {**shared, "opportunity_id": "old", "candidate_eligible": False,
                                      "source_eligibility": {"ma": False}, "stop_price": None}}
    result = compare_candidates(main, control)
    assert result["shared_stock_dates"] == 1
    assert result["risk_qualification_transitions"] == {"main_True_control_False": 1}
    assert result["raw_source_membership_differences"] == []
    assert len(result["changed_union_stops"]) == 1
