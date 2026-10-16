"""Synthetic HTTP installation evidence, independent of any business workspace."""
from copy import deepcopy
from pathlib import Path

import pytest

import verify_installation as installation
import verify_views_installation as views


@pytest.fixture(scope='module')
def view_report():
    value=views.run_probe()
    html=(Path(installation.__file__).parent/'my_strategy/web_dashboard/static/index.html').read_text(encoding='utf-8')
    value['navigation']=installation.verify_v102_navigation(html)
    return value


def test_real_installed_read_routes_and_strict_gate(view_report):
    assert installation.verify_v102_views(view_report) is view_report
    assert view_report['readonly_state_unchanged'] is True


def leaves(value,prefix=()):
    for key,item in value.items():
        if isinstance(item,dict):yield from leaves(item,prefix+(key,))
        else:yield prefix+(key,)


def test_every_v102_field_is_required_and_type_checked(view_report):
    for keys in leaves(view_report):
        changed=deepcopy(view_report);target=changed
        for key in keys[:-1]:target=target[key]
        original=target.pop(keys[-1])
        with pytest.raises(RuntimeError,match='v1.0.2'):installation.verify_v102_views(changed)
        target[keys[-1]]=not original if type(original) is bool else True if type(original) is int else 'incorrect'
        with pytest.raises(RuntimeError,match='v1.0.2'):installation.verify_v102_views(changed)
    for section in ('navigation','prediction_chart','method_display','board_summary'):
        changed=deepcopy(view_report);changed.pop(section)
        with pytest.raises(RuntimeError,match='v1.0.2'):installation.verify_v102_views(changed)


@pytest.mark.parametrize('identifier', ['light-sim-form','light-saved','light-accounts-state','light-results',
    'light-daily','light-history-window','layer-method','layer-sources','analysis-prediction','board-method-panel'])
def test_shipped_navigation_requires_actual_control_ancestry(identifier):
    html=(Path(installation.__file__).parent/'my_strategy/web_dashboard/static/index.html').read_text(encoding='utf-8')
    with pytest.raises(RuntimeError,match='v1.0.2'):
        installation.verify_v102_navigation(html.replace('id="'+identifier+'"','id="missing-'+identifier+'"'))
    # ID exists elsewhere, but location is still wrong.
    changed=html.replace('id="'+identifier+'"','id="missing-'+identifier+'"')+'<div id="'+identifier+'"></div>'
    with pytest.raises(RuntimeError,match='v1.0.2'):installation.verify_v102_navigation(changed)


def test_no_test_module_or_hidden_business_computation_dependency():
    source=Path(views.__file__).read_text(encoding='utf-8')
    assert 'my_strategy.tests' not in source
    assert "LightResearch(" not in source and 'TaskManager(' not in source
    assert "'POST'" not in source


def test_http_worker_joins_when_a_read_contract_fails(monkeypatch):
    import threading
    from my_strategy.services import prediction_chart
    original=prediction_chart.prediction_chart
    def broken(*args,**kwargs):
        result=original(*args,**kwargs); result['history']=[]; return result
    monkeypatch.setattr(prediction_chart,'prediction_chart',broken)
    with pytest.raises(RuntimeError,match='history/forecast window'):
        views.run_probe()
    assert not any(thread.name=='synthetic-view-server' for thread in threading.enumerate())


def test_http_probe_rejects_any_store_write_from_a_get(monkeypatch):
    import threading
    from my_strategy.services import prediction_chart
    def broken(light,reference):
        light.store.put({'unexpected':reference},{'wrong':True})
    monkeypatch.setattr(prediction_chart,'prediction_chart',broken)
    with pytest.raises(RuntimeError):views.run_probe()
    assert not any(thread.name=='synthetic-view-server' for thread in threading.enumerate())
