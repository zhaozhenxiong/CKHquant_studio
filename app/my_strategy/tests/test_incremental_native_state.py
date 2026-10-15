import copy
import json
import pandas as pd
import pytest

from my_strategy.tests.test_current_native_evidence import _raw
from my_strategy.services.czsc_native_evidence import build_native_evidence
from my_strategy.services.czsc_incremental_state import StateNotRestorable


@pytest.mark.parametrize('cut',[121,187,310])
def test_explicit_state_restart_matches_full_native(cut):
    bars=_raw(340,varied=True);dates=bars.date.dt.strftime('%Y-%m-%d').tolist()
    prefix=build_native_evidence(bars.iloc[:cut],market_dates=dates[:cut],capture_state=True)
    state=json.loads(json.dumps(prefix['incremental_state']))
    try:
        appended=build_native_evidence(bars,market_dates=dates,incremental_state=state,capture_state=True)
    except StateNotRestorable:
        pytest.fail('representative frozen-engine state could not restore')
    complete=build_native_evidence(bars,market_dates=dates)
    pd.testing.assert_frame_equal(appended['frame'],complete['frame'].iloc[cut:],check_exact=True)
    assert appended['evaluations']==complete['evaluations'][cut:]
    assert appended['events']==[e for e in complete['events'] if e['date']>=str(bars.date.iloc[cut].date())]
    assert appended['incremental']['native_prefix_reused']==cut


def test_changed_old_bar_or_calendar_cannot_resume():
    bars=_raw(150,varied=True);dates=bars.date.dt.strftime('%Y-%m-%d').tolist()
    saved=build_native_evidence(bars.iloc[:130],market_dates=dates[:130],capture_state=True)['incremental_state']
    changed=bars.copy();changed.loc[changed.index[9],'volume']+=1
    with pytest.raises(ValueError,match='prefix'):
        build_native_evidence(changed,market_dates=dates,incremental_state=saved)
    with pytest.raises(ValueError,match='prefix'):
        build_native_evidence(bars,market_dates=dates[1:],incremental_state=saved)


def test_state_tampering_is_rejected():
    bars=_raw(150,varied=True);dates=bars.date.dt.strftime('%Y-%m-%d').tolist()
    state=build_native_evidence(bars.iloc[:130],market_dates=dates[:130],capture_state=True)['incremental_state']
    state=copy.deepcopy(state);state['structure']['session']['symbol']='600519.SH'
    with pytest.raises(ValueError,match='SHA'):
        build_native_evidence(bars,market_dates=dates,incremental_state=state)
