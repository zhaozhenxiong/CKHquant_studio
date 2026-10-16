"""This release's final installed contract requires the new read-only features."""
from copy import deepcopy
import pytest
import verify_installation as installation
import verify_followup_installation as followup


@pytest.fixture(scope='module')
def report():
    return followup.run_probe()


def leaves(value, prefix=()):
    for key, item in value.items():
        if isinstance(item, dict):
            yield from leaves(item, prefix+(key,))
        else:
            yield prefix+(key,)


def test_new_installed_contract_reads_real_synthetic_routes_and_preserves_store(report):
    assert installation.verify_v103_followup(report) is report
    assert report['account_chart']['readonly_store_unchanged'] is True
    assert report['capital_history']['cached_slices_no_reload'] is True


def test_each_new_evidence_field_is_required_and_type_checked(report):
    for keys in leaves(report):
        changed = deepcopy(report); target = changed
        for key in keys[:-1]:
            target = target[key]
        original = target.pop(keys[-1])
        with pytest.raises(RuntimeError, match='v1.0.3'):
            installation.verify_v103_followup(changed)
        target[keys[-1]] = not original if type(original) is bool else True if type(original) is int else 'wrong'
        with pytest.raises(RuntimeError, match='v1.0.3'):
            installation.verify_v103_followup(changed)


@pytest.mark.parametrize('section', ['navigation', 'capital_history', 'account_chart'])
def test_old_release_receipts_cannot_certify_new_features(report, section):
    changed = deepcopy(report); changed.pop(section)
    with pytest.raises(RuntimeError, match='v1.0.3'):
        installation.verify_v103_followup(changed)


def test_probe_never_imports_test_fixtures_or_loads_weights():
    from pathlib import Path
    source = Path(followup.__file__).read_text(encoding='utf-8')
    assert 'my_strategy.tests' not in source
    assert 'LightResearch(' not in source and "'POST'" not in source
