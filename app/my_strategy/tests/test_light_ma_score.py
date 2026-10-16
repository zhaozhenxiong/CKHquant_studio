"""MA input lineage, paired comparison, and candidate routing regressions."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.light_ma_features import MA_TREND_COLUMNS, prepare_ma_features
from my_strategy.services.light_score_training import prepare_score_features, make_direction_training_frame, chronological_split_ranges
from my_strategy.services.light_score_v2 import fit_light_score, score_light_samples
from my_strategy.services.light_research import LightResearch


@pytest.fixture(scope='module')
def history():
    rng=np.random.default_rng(54)
    days=pd.bdate_range('2012-01-02',periods=1300)
    close=10*np.exp(np.cumsum(rng.normal(0,.012,len(days))))
    return pd.DataFrame({'symbol':'000001.SZ','date':days,'open':close*.999,
        'high':close*1.02,'low':close*.98,'close':close,'volume':10000.,'amount':100000.,
        'source':'tushare','has_trade_price':1})


@pytest.fixture(scope='module')
def data(history):
    dates=history.date.dt.strftime('%Y-%m-%d').tolist()
    frame=prepare_score_features(history,feature_profile='ma_trend_v1',calendar_dates=dates)
    frame=make_direction_training_frame(frame,dates)
    frame=frame.loc[frame.score_input_eligible].copy()
    return frame,chronological_split_ranges(frame)


@pytest.fixture(scope='module')
def model(data):
    frame,ranges=data
    return fit_light_score(frame,ranges,feature_profile='ma_trend_v1',
        source_identity={'test':'ma'},label_contract={'horizon':5},
        trials=[{'widths':[8,4],'max_epochs':3,'patience':2}])


def test_ma_features_match_shared_definition_and_are_causal(history):
    from my_strategy.services.czsc_market_features import _trailing_features
    dates=history.date.dt.strftime('%Y-%m-%d').tolist()
    result=prepare_ma_features(history,calendar_dates=dates)
    prefix=prepare_ma_features(history.iloc[:210],calendar_dates=dates)
    pd.testing.assert_frame_equal(result.loc[:209,list(MA_TREND_COLUMNS)],prefix[list(MA_TREND_COLUMNS)])
    expected=_trailing_features(history)
    pd.testing.assert_frame_equal(result[list(MA_TREND_COLUMNS)],expected[list(MA_TREND_COLUMNS)])
    assert not result.score_input_eligible.iloc[:119].any()
    assert result.score_input_eligible.iloc[119:].all()
    assert 'ma5_bias' not in history


def test_ma_quality_veto_matches_existing_contract(history):
    dates=history.date.dt.strftime('%Y-%m-%d').tolist()
    bad=history.iloc[:300].copy()
    bad.loc[130,'source']='unverified'
    quality=prepare_ma_features(bad,calendar_dates=dates)
    assert quality.score_input_eligible.iloc[129]
    assert not quality.score_input_eligible.iloc[130:250].any()
    assert quality.score_input_eligible.iloc[250]
    inactive=history.iloc[:200].copy()
    inactive.loc[:99,['volume','amount']]=0
    result=prepare_ma_features(inactive,calendar_dates=dates)
    assert not result.score_input_eligible.iloc[:159].any()
    assert result.score_input_eligible.iloc[159]
    gap=prepare_ma_features(history.drop(index=130),calendar_dates=dates)
    assert not gap.score_input_eligible.iloc[130:250].any()
    assert gap.score_input_eligible.iloc[250]
    assert not prepare_ma_features(history).score_input_eligible.any()


def test_ma_profile_is_explicit_and_paired_l2_uses_same_input(model,data):
    assert model['features']==list(MA_TREND_COLUMNS)
    assert model['feature_profile']=='ma_trend_v1'
    assert model['architecture']['input_dimensions']==20
    assert model['preprocessing']['optional_columns']==[]
    assert model['identity']['feature_contract']['kronos_input'] is False
    row=data[0].iloc[[-1]]
    mlp=score_light_samples(model,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    l2=score_light_samples(model,row,as_of=row.iloc[0].as_of,historical_reconstruction=True,model_kind='l2')
    assert mlp['features']==l2['features']==list(MA_TREND_COLUMNS)
    assert mlp['scorer_id']!=l2['scorer_id']
    assert mlp['model_version']==l2['model_version']
    assert mlp['qualification']['paper_allocation_eligible'] is False


def test_ma_ineligible_scores_stay_missing_and_network_is_reused(model,data,monkeypatch):
    from my_strategy.services import light_score_v2 as scoring
    row=data[0].iloc[[-1]].copy()
    scoring._NETWORKS.clear()
    calls=[];original=scoring._network
    monkeypatch.setattr(scoring,'_network',lambda *a,**k:(calls.append(True),original(*a,**k))[1])
    for _ in range(2):score_light_samples(model,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    assert len(calls)==1
    row['score_input_eligible']=False;row['ma5_bias']=np.nan
    for kind in ('mlp','l2'):
        value=score_light_samples(model,row,as_of=row.iloc[0].as_of,historical_reconstruction=True,model_kind=kind)
        assert value['status']=='unavailable' and value['scores']==[None]
        assert value['row_statuses']==['ma_input_ineligible']
        assert value['qualification']['allow_increase'] is False
    broken=deepcopy(model);broken['state_dict'][next(iter(broken['state_dict']))][0][0]+=1
    with pytest.raises(ValueError,match='identity mismatch'):
        score_light_samples(broken,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)


def test_ma_cannot_claim_kronos_learning_or_bypass_input_quality(history,data):
    with pytest.raises(ValueError,match='A only'):
        prepare_score_features(history,feature_profile='ma_trend_v1',variant='B')
    frame,ranges=data
    with pytest.raises(ValueError,match='verified eligible'):
        fit_light_score(frame.assign(score_input_eligible=False),ranges,feature_profile='ma_trend_v1',
            source_identity={'test':'bad'},label_contract={'horizon':5})


def test_ma_resolver_and_runtime_preserve_frozen_profile(tmp_path,model,history,data,monkeypatch):
    light=LightResearch(tmp_path/'research')
    dates=history.date.dt.strftime('%Y-%m-%d').tolist()
    monkeypatch.setattr(light,'calendar',lambda:(dates,{'calendar':'test'}))
    value={'artifact':model,'model_type':'mlp','status':'completed','training_end':dates[-1]}
    ref=light.store.put({'fixture':'ma'},value,namespace='mlp_score_models',pin='test')
    try:
        day=data[0].iloc[-1].as_of
        chosen=light.resolve_score_model(feature_profile='ma_trend_v1',as_of=day)
        assert chosen['model_ref']['cache_key']==ref['cache_key']
        assert light.list_score_models()[0]['feature_profile']=='ma_trend_v1'
        assert light.resolve_score_model(feature_profile='mixed_v1',as_of=day) is None
        assert light.resolve_score_model(feature_profile='ma_trend_v1',as_of='2010-01-01') is None
        with pytest.raises(ValueError,match='不一致'):
            light.resolve_score_model(ref['cache_key'],feature_profile='mixed_v1')
        result=light.scorer(chosen,{'000001.SZ':history},{})('000001.SZ',day,{'unrelated':'forecast'})
        assert result['feature_profile']==result['l2_comparison']['feature_profile']=='ma_trend_v1'
        assert result['consumes_kronos'] is False
        assert not light.store.get_setting('current_score')
    finally:light.close()


def test_ma_training_reads_raw_without_native_rebuild_or_promotion(tmp_path,history,monkeypatch):
    from my_strategy.services import light_research, light_score_v2
    light=LightResearch(tmp_path/'research')
    dates=history.date.dt.strftime('%Y-%m-%d').tolist()
    monkeypatch.setattr(light,'calendar',lambda:(dates,{'calendar':'test'}))
    monkeypatch.setattr(light_research,'load_bars',lambda *a,**k:history.copy())
    monkeypatch.setattr(light,'source',lambda *a,**k:pytest.fail('MA training rebuilt native sources'))
    original=light_score_v2.fit_light_score
    monkeypatch.setattr(light_score_v2,'fit_light_score',lambda *a,**k:original(*a,**k,
        trials=[{'widths':[8,4],'max_epochs':2,'patience':2}]))
    try:
        trained=light.train_score(['000001.SZ'],dates[-1],feature_profile='ma_trend_v1')
        assert trained['status']=='completed' and trained['selected'] is False
        assert trained['artifact']['architecture']['input_dimensions']==20
        assert trained['core_excluded_rows']==119
        assert trained['identity']['sources'][0]['kind']=='ma_raw_source_v2'
        assert trained['artifact']['l2_baseline']['model_type']=='multinomial_logistic_L2'
        assert light.current_score() is None
        cached=light.train_score(['000001.SZ'],dates[-1],feature_profile='ma_trend_v1')
        assert cached['reused'] and cached['model_ref']['cache_key']==trained['model_ref']['cache_key']
        with pytest.raises(ValueError,match='仅训练'):
            light.train_score(['000001.SZ'],dates[-1],feature_profile='ma_trend_v1',model_type='l2')
    finally:light.close()


def test_ma_backtest_and_saved_daily_keep_explicit_model(tmp_path,model,history,monkeypatch):
    from my_strategy.services import light_research
    from my_strategy.services.light_account_continuation import continue_saved_account
    light=LightResearch(tmp_path/'research')
    dates=history.date.dt.strftime('%Y-%m-%d').tolist()
    monkeypatch.setattr(light,'calendar',lambda:(dates,{'calendar':'test'}))
    event={'event_id':'ma-test-source','symbol':'000001.SZ','date':dates[-16],
           'side':'BUY','confirmed':True,'risk_boundary':float(history.close.iloc[-16])*.9}
    def source(symbol,end,**kwargs):
        frame=history.loc[history.date<=pd.Timestamp(end)].copy()
        return frame,frame,[event],{'frozen_source':'same'}
    monkeypatch.setattr(light,'source',source)
    monkeypatch.setattr(light,'prediction',lambda symbol,day,**kwargs:{'as_of':day,
        'quality':{'prediction_usable':True},'summary':{'terminal_return_median':.02,'up_path_fraction':.7}})
    monkeypatch.setattr(light_research,'prediction_identity',lambda *a,**k:{'fixture':'offline'})
    ref=light.store.put({'fixture':'ma'}, {'artifact':model,'model_type':'mlp'},namespace='mlp_score_models',pin='test')
    try:
        previous=light.backtest(['000001.SZ'],dates[-18],dates[-10],score_model_id=ref['cache_key'],
            feature_profile='ma_trend_v1',device='cpu',save=True)
        assert previous['identity']['feature_profile']=='ma_trend_v1'
        assert set(previous['comparisons'])=={'structure_rules','structure_kronos','structure_kronos_score','structure_kronos_l2'}
        account=light.enable_account(previous['result_id'],shadow_research=True)
        continued=continue_saved_account(light,account,previous,dates[-1])
        assert continued['score_model']==previous['score_model']
        assert continued['feature_profile']=='ma_trend_v1'
        observations=[e['score'] for e in continued['source_events']['000001.SZ'] if e.get('score')]
        assert observations and all(s['feature_profile']=='ma_trend_v1' for s in observations)
        assert all(s['l2_comparison']['feature_profile']=='ma_trend_v1' for s in observations)
        assert light.current_score() is None
    finally:light.close()
