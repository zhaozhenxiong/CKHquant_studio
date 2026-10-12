"""Diagnostic arms cannot change model identity, scope, risk or continuation."""
import copy
from pathlib import Path

import pytest

from my_strategy.scripts import diagnose_native_v4 as diag


def test_predefined_arms_reuse_one_linear_point_one_account_set():
    arms = diag.diagnostic_arms()
    assert len(arms) == 7
    assert {a['architecture'] for a in arms if 'architecture' in a['comparisons']} == {'linear', 'tree', 'mlp'}
    assert {a['entry_threshold'] for a in arms if 'threshold_grid' in a['comparisons']} == set(diag.THRESHOLDS)
    assert next(a for a in arms if a['id'] == 'linear_entry_0p1')['comparisons'] == ['architecture', 'threshold_grid']


def test_architecture_catalog_uses_real_tree_identity_and_rejects_population_drift(tmp_path):
    root = tmp_path
    for arch in ['linear', 'tree']:
        path = root / 'models/entry/fold' / arch
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{}')
    report = {'model_bundles': [{'name': 'fold', 'available_at': '2025-03-31', 'architecture': 'linear',
              'policy': {'entry_threshold': .1}, 'heads': {'entry': {'path': 'models/entry/fold/linear', 'manifest_sha256': 'linear-sha'}}}],
              'source_snapshot': {'snapshot_hash': 'source'}}
    def loader(path):
        arch = Path(path).name
        return {'head': 'entry', 'architecture': arch, 'manifest_sha256': arch+'-sha',
                'target_contract_hash': 'same-real-target', 'target_contract': {'source_snapshot_hash': 'source'},
                'available_at': '2025-03-31', 'fold': {'name': 'fold'}, 'training_completed_at': '2026-10-12'}
    catalog, coverage = diag.architecture_catalog(root, report, 'tree', loader=loader)
    assert catalog[0]['heads']['entry']['path'] == 'models/entry/fold/tree'
    assert catalog[0]['heads']['entry']['manifest_sha256'] == 'tree-sha'
    assert coverage[0]['architecture'] == 'tree'
    assert report['model_bundles'][0]['heads']['entry']['manifest_sha256'] == 'linear-sha'
    absent, coverage = diag.architecture_catalog(root, report, 'mlp', loader=loader)
    assert absent == []
    assert coverage[0]['status'] == 'missing_architecture_head_no_linear_substitution'
    def mismatched(path):
        result = loader(path)
        if Path(path).name == 'tree': result['target_contract_hash'] = 'other-risk-population'
        return result
    with pytest.raises(ValueError, match='targets differ'):
        diag.architecture_catalog(root, report, 'tree', loader=mismatched)


def test_entry_replacement_leaves_native_facts_ma_risk_and_exit_state_untouched():
    original = {'date': '2026-01-06', 'agents': {'ma': {'probability': .6}, 'high_return': {'probability': .1}},
                'candidate_evidence': {'facts': [{'event_id': 'real'}], 'provenance': {'risk_policy_hash': 'frozen'},
                                       'evaluations': {'ma': {'probability': .6}, 'high_return': {'probability': .1}}},
                'raw_risk_plan': {'plan_id': 'frozen-stop'}, 'rule_sell': True, 'source_exit_event_ids': ['exit'],
                'entry_probability': .1}
    before = copy.deepcopy(original)
    row = diag.replace_entry_evaluation(original, {'probability': .2, 'model_identity': {'manifest_sha256': 'tree-real'}})
    assert original == before
    assert row['raw_risk_plan'] is original['raw_risk_plan']
    assert row['candidate_evidence']['facts'] is original['candidate_evidence']['facts']
    assert row['agents']['ma'] is original['agents']['ma']
    assert row['rule_sell'] and row['source_exit_event_ids'] == ['exit']
    assert row['entry_probability'] == .2
    assert row['model_resolution']['manifest_sha256'] == 'tree-real'


def test_causal_window_excludes_future_and_before_window_entries():
    symbol = '002436.SZ'
    def row(day):
        return {'date': day, 'rule_sell': False, 'raw_risk_plan': {'status': 'active', 'entry_allowed': True,
                'symbol': symbol, 'signal_date': day, 'policy': 'risk', 'valid_for_sessions': 1,
                'plan_id': day, 'stop_price': 9., 'target_weight': .2}}
    rows = [row('2026-01-05'), row('2026-01-06'), row('2026-01-07'), row('2026-01-08')]
    consensus = {'events': [{'date': day, 'action': 'BUY', 'event_id': day} for day in ['2026-01-05', '2026-01-06', '2026-01-08']]}
    decisions, blocked, events = diag.causal_window(consensus, rows, '2026-01-06', '2026-01-07', symbol, 1.)
    assert [r['date'] for r in decisions] == ['2026-01-05', '2026-01-06', '2026-01-07']
    assert decisions[0]['action'] == 'HOLD'
    assert decisions[1]['action'] == 'BUY'
    assert [e['date'] for e in events] == ['2026-01-06']
    assert blocked == []
    assert len(consensus['events']) == 3


def test_classification_counts_only_labels_mature_by_each_development_window_end():
    rows = [{'date': '2026-06-04', 'label_end': '2026-06-25', 'probability': .2, 'label': 1.,
             'source_ma': False, 'source_structure': True, 'source_wyckoff': True},
            {'date': '2026-06-29', 'label_end': '2026-07-03', 'probability': .8, 'label': 0.,
             'source_ma': True, 'source_structure': False, 'source_wyckoff': False}]
    result = diag.classification_metrics(rows, {'2026Q2': ('2026-04-01', '2026-06-30')}, '2026-10-08')
    union = next(r for r in result if r['source'] == 'union')
    assert union['rows'] == 1
    assert union['positive_rate'] == 1.
    assert next(r for r in result if r['source'] == 'ma')['rows'] == 0
    assert union['population'] == 'mature_actual_raw_union_reference_candidate_account_features'
