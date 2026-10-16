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


def test_v1_read_compatibility_replays_wyckoff_without_changing_values():
    from my_strategy.core.run_context import stable_hash
    bars=_raw(190,varied=True);dates=bars.date.dt.strftime('%Y-%m-%d').tolist()
    prefix=build_native_evidence(bars.iloc[:151],market_dates=dates[:151],capture_state=True)
    legacy=copy.deepcopy(prefix['incremental_state']);legacy['format']='native_evidence_state_v1'
    legacy.pop('wyckoff');legacy.pop('wyckoff_features');legacy.pop('state_sha256')
    legacy['state_sha256']=stable_hash(legacy)
    resumed=build_native_evidence(bars,market_dates=dates,incremental_state=legacy,capture_state=True)
    full=build_native_evidence(bars,market_dates=dates)
    pd.testing.assert_frame_equal(resumed['frame'],full['frame'].iloc[151:],check_exact=True)
    assert resumed['incremental']['wyckoff_mode']=='single_stock_vector_replay'


def test_multiple_json_restarts_keep_every_wyckoff_event_and_causal_value():
    bars=_raw(340,varied=True);dates=bars.date.dt.strftime('%Y-%m-%d').tolist()
    state=None;frames=[];events=[];start=0
    for end in (121,151,187,241,310,340):
        pack=build_native_evidence(bars.iloc[:end],market_dates=dates[:end],incremental_state=state,capture_state=True)
        if state:
            assert pack['incremental']['wyckoff_rows_replayed']==0
            assert pack['incremental']['wyckoff_rows_appended']==end-start
        state=json.loads(json.dumps(pack['incremental_state']))
        frames.append(pack['frame']);events.extend(pack['events']);start=end
    full=build_native_evidence(bars,market_dates=dates)
    pd.testing.assert_frame_equal(pd.concat(frames),full['frame'],check_exact=True)
    assert sorted(events,key=lambda e:(e['date'],e['source'],e['event_id']))==full['events']


def test_standalone_wyckoff_features_state_is_exact_and_checks_prefix():
    from my_strategy.services.czsc_price_volume_features import prepare_wyckoff_inputs
    bars=_raw(220,varied=True);dates=bars.date.dt.strftime('%Y-%m-%d').tolist();state={}
    prefix=prepare_wyckoff_inputs(bars.iloc[:151],config={'market_dates':dates[:151]},state_sink=state)
    resumed=prepare_wyckoff_inputs(bars,config={'market_dates':dates},resume_state=json.loads(json.dumps(state)))
    full=prepare_wyckoff_inputs(bars,config={'market_dates':dates})
    pd.testing.assert_frame_equal(resumed,full.iloc[151:],check_exact=True)
    changed=bars.copy();changed.loc[changed.index[130],'volume']*=1.1
    with pytest.raises(ValueError,match='prefix'):
        prepare_wyckoff_inputs(changed,config={'market_dates':dates},resume_state=state)
