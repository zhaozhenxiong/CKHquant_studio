"""Cross-entrypoint identity, causal cutoff and bounded scan regressions."""
from copy import deepcopy
from types import SimpleNamespace

from filelock import FileLock
import pandas as pd
import pytest

from my_strategy.services import analysis_methods as methods
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.research_store import ResearchStore
from my_strategy.web_dashboard.api import AnalysisRequest, TaskRequest, compact_scan_view


@pytest.fixture
def environment(tmp_path, monkeypatch):
    day = '2026-10-14'
    frame = pd.DataFrame({'symbol': ['000001.SZ'], 'date': [pd.Timestamp(day)], 'close': [12.]})
    frame.attrs['data_version'] = 'raw-prefix'
    calls = []
    def prediction(symbol, end, **kwargs):
        calls.append((symbol, end, kwargs.get('seed')))
        return {'status': 'completed', 'as_of': end, 'summary': {'terminal_return_median': .02},
                'quality': {'prediction_usable': True}, 'result_ref': {'cache_key': 'forecast'}}
    score = {'status': 'scored', 'score': 70., 'probabilities': {'down':.1, 'flat':.2, 'up':.7},
             'qualification': {'paper_allocation_eligible': False}, 'model_version': 'ma-model'}
    light = SimpleNamespace(raw_db=tmp_path/'raw.db', compute_lock=FileLock(str(tmp_path/'compute.lock')),
        store=ResearchStore(tmp_path/'research'),
        calendar=lambda: ([day], {'sha256': 'calendar'}),
        source=lambda *a, **kw: (frame.copy(), frame.copy(), [], {'input_sha': 'source'}),
        resolve_score_model=lambda *a, **kw: {'model_id': 'a'*64, 'artifact': {'model_version': 'ma-model'}},
        prediction=prediction,
        scorer=lambda *a, **kw: lambda *args: {**deepcopy(score), 'l2_comparison': {**deepcopy(score), 'score': 60.}})
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import kronos_prediction, light_research
    from my_strategy.web_dashboard import trend_strategy
    monkeypatch.setattr(czsc_adapter, 'latest_market_date', lambda *a, **kw: day)
    monkeypatch.setattr(czsc_adapter, 'load_bars', lambda *a, **kw: frame.copy())
    monkeypatch.setattr(czsc_adapter, 'list_symbols', lambda *a, **kw: [
        {'symbol': '000001.SZ', 'name': 'A'}, {'symbol': '600000.SH', 'name': 'B'}, {'symbol': '920001.BJ', 'name': 'C'}])
    monkeypatch.setattr(kronos_prediction, 'asset_status', lambda: {'ready': True})
    monkeypatch.setattr(light_research, 'prediction_identity', lambda *a: {'model': 'frozen-kronos'})
    monkeypatch.setattr(methods, 'database_identity', lambda *a: {'database': 'raw-sha'})
    monkeypatch.setattr(trend_strategy, 'analyze_trend_consensus', lambda **kw: {
        'symbol': kw['symbol'], 'data_end': kw['end'], 'capabilities': {}, 'data_version': 'raw-prefix'})
    store = ResultStore(tmp_path/'results.db', tmp_path/'runs')
    yield light, store, calls
    light.store.close()


def spec(method='structure_kronos_ma_mlp', **kw):
    return dict(analysis_method=method, start='2026-01-01', end='2026-10-14',
        usage_mode='historical', **kw)


@pytest.mark.parametrize('method', sorted(methods.COMBINED))
def test_single_and_scan_use_identical_method_evaluation(environment, method):
    light, store, calls = environment
    single = methods.analyze_method(light, spec(method, symbol='000001.SZ'))
    scan = methods.scan_methods(light, store, spec(method, symbols=['000001.SZ']))
    assert single['method_evaluation'] == scan['rows'][0]['method_evaluation']
    assert scan['status'] == 'succeeded' and scan['coverage']['complete']
    assert single['method_context'] == scan['method_context']
    assert single['method_evaluation']['account_scope'] == 'none'
    evaluation = scan['rows'][0]['method_evaluation']
    if method == 'structure_kronos':
        assert evaluation['score'] is None and evaluation['model_id'] is None
    else:
        assert evaluation['score']['status'] == 'scored'
        assert evaluation['model_id'] == 'a'*64
        assert scan['method_context']['score_model_id'] == 'a'*64


def test_same_input_l2_is_selected_without_calling_legacy_score(environment):
    light, store, calls = environment
    result = methods.analyze_method(light, spec('structure_kronos_ma_l2', symbol='000001.SZ'))
    assert result['method_evaluation']['score']['score'] == 60
    assert result['method_evaluation']['model_id'] == 'a'*64


def test_budget_preserves_full_population_and_explicit_resume_does_only_remaining(environment):
    light, store, calls = environment
    first = methods.scan_methods(light, store, spec(max_predictions=1))
    assert first['population'] == ['000001.SZ', '600000.SH']
    assert first['status'] == 'partial' and first['coverage']['deferred'] == 1
    second = methods.scan_methods(light, store, spec(max_predictions=1, resume_scan_id=first['run_id']))
    assert second['run_id'] != first['run_id'] and second['status'] == 'succeeded'
    assert second['coverage']['success'] == 2 and len(calls) == 2
    assert store.get(first['run_id'])['coverage']['deferred'] == 1


def test_resume_rejects_data_or_method_drift(environment, monkeypatch):
    light, store, calls = environment
    first = methods.scan_methods(light, store, spec(max_predictions=1))
    with pytest.raises(ValueError, match='方法'):
        methods.scan_methods(light, store, spec('structure_kronos', resume_scan_id=first['run_id']))
    monkeypatch.setattr(methods, 'database_identity', lambda *a: {'database': 'revised'})
    with pytest.raises(ValueError, match='行情'):
        methods.scan_methods(light, store, spec(resume_scan_id=first['run_id']))


def test_missing_ma_model_keeps_forecast_but_does_not_fall_back(environment):
    light, store, calls = environment
    light.resolve_score_model = lambda *a, **kw: None
    value = methods.analyze_method(light, spec(symbol='000001.SZ'))['method_evaluation']
    assert value['score'].get('score') is None
    assert value['model_id'] is None
    assert value['status'] == 'partial' and 'ma_score_model_missing' in value['missing_reasons']
    assert value['kronos']['quality']['prediction_usable'] and value['direction'] == 'WAIT'


def test_cancelled_scan_keeps_checkpoint_and_is_not_reactivated(environment):
    light, store, calls = environment
    class Cancelled(RuntimeError): pass
    def cancelled():
        if calls: raise Cancelled('cancel')
    first = methods.scan_methods(light, store, spec(max_predictions=2, cpu_workers=1), check_cancel=cancelled)
    assert first['status'] == 'cancelled'
    assert store.get(first['run_id'])['status'] == 'cancelled'
    assert first['coverage']['success'] == 1 and first['coverage']['deferred'] == 1


def test_compact_scan_retains_nested_forecast_and_score(environment):
    light, store, calls = environment
    result = methods.scan_methods(light, store, spec(symbols=['000001.SZ']))
    view = compact_scan_view(result)['rows'][0]['method_evaluation']
    assert view['score']['probabilities'] == {'down':.1, 'flat':.2, 'up':.7}
    assert view['kronos']['summary']['terminal_return_median'] == .02
    assert 'source_identity' not in view


def test_unknown_method_rejected_and_new_modes_never_resolve_legacy_checkpoint():
    with pytest.raises(ValueError): AnalysisRequest(symbol='000001.SZ', analysis_method='typo')
    value = AnalysisRequest(symbol='000001.SZ', analysis_method='structure_kronos_ma_mlp')
    assert value.research is False
    with pytest.raises(ValueError):
        AnalysisRequest(symbol='000001.SZ', analysis_method='structure_kronos_ma_mlp', model_run_id='legacy')
    with pytest.raises(ValueError): TaskRequest(kind='update', spec={'analysis_method': 'structure_kronos'})


@pytest.mark.parametrize('extra', [
    {'calendar_run_id':'old-calendar'}, {'current_position_fraction':.5},
    {'model_policy':'pinned'}, {'consensus_config':{}}, {'consensus_version':'reference_position_v1'},
])
def test_combined_request_rejects_unused_legacy_parameters(extra):
    with pytest.raises(ValueError):
        AnalysisRequest(symbol='000001.SZ', analysis_method='structure_kronos', **extra)
    if 'current_position_fraction' not in extra:
        with pytest.raises(ValueError):
            TaskRequest(kind='scan',spec={'analysis_method':'structure_kronos',**extra})


def test_scan_provenance_records_actual_cli_entrypoint(environment):
    light, store, _ = environment
    result = methods.scan_methods(light, store, spec(max_predictions=1), invocation_source='cli')
    assert result['run_context']['source'] == 'cli'


def test_intraday_cutoff_cannot_use_same_day_close(environment):
    light, store, calls = environment
    with pytest.raises(ValueError, match='核验交易日'):
        methods.analyze_method(light, spec(symbol='000001.SZ', as_of='2026-10-14T14:59:59+08:00'))
    assert calls == []


@pytest.mark.parametrize('probabilities,direction', [
    ({'down':.1,'flat':.2,'up':.7},'BUY'),
    ({'up':.1,'flat':.2,'down':.7},'SELL'),
    ({'down':.2,'flat':.6,'up':.2},'WAIT'),
    ({'down':.5,'flat':0.,'up':.5},'WAIT'),
    ({'down':.1,'flat':.2,'up':float('nan')},'WAIT'),
    ({'down':.1,'up':.9},'WAIT'),
])
def test_named_probability_classes_are_validated(probabilities,direction):
    assert methods._score_direction({'probabilities':probabilities}) == direction


def test_cancel_is_observed_inside_prediction_without_coordinator_poll(environment):
    light,store,calls=environment
    cancelled=[False]
    def check():
        if cancelled[0]:raise RuntimeError('cancel request')
    def prediction(symbol,end,**kwargs):
        cancelled[0]=True
        assert kwargs['stop'].is_set()
        raise InterruptedError('prediction stopped')
    light.prediction=prediction
    result=methods.scan_methods(light,store,spec(cpu_workers=1),check_cancel=check)
    assert result['status']=='cancelled' and result['coverage']['deferred']==2
    assert result['coverage']['success']==0


def test_task_manager_cancel_during_source_remains_resumable_not_failed(environment):
    light, store, calls = environment
    from my_strategy.web_dashboard.tasks import Cancelled
    requested = [False]
    original = light.source
    def check():
        if requested[0]: raise Cancelled('用户取消任务')
    def source(*args, **kwargs):
        requested[0] = True
        kwargs['stop'].check()
        return original(*args, **kwargs)
    light.source = source
    cancelled = methods.scan_methods(light, store, spec(cpu_workers=1), check_cancel=check)
    assert cancelled['status'] == 'cancelled'
    assert cancelled['coverage']['failed'] == 0
    assert cancelled['coverage']['deferred'] == 2
    assert cancelled['coverage']['evaluated'] == 0
    light.source = original
    resumed = methods.scan_methods(light, store, spec(resume_scan_id=cancelled['run_id']))
    assert resumed['status'] == 'succeeded' and resumed['coverage']['success'] == 2


def test_resume_resolves_original_end_before_latest_date(environment,monkeypatch):
    light,store,calls=environment
    first=methods.scan_methods(light,store,spec(max_predictions=1))
    from my_strategy.adapters import czsc_adapter
    light.calendar=lambda:(['2026-10-14','2026-10-15'],{'sha256':'calendar'})
    monkeypatch.setattr(czsc_adapter,'latest_market_date',lambda end=None,**kw:end or '2026-10-15')
    resumed=spec(max_predictions=1,resume_scan_id=first['run_id'])
    resumed.pop('end')
    result=methods.scan_methods(light,store,resumed)
    assert result['data_end']=='2026-10-14' and result['status']=='succeeded'
    assert calls[-1][1]=='2026-10-14'


def test_single_refuses_mixed_raw_vintages(environment,monkeypatch):
    light,store,calls=environment
    identity=iter([{'database':'before'},{'database':'after'}])
    monkeypatch.setattr(methods,'database_identity',lambda *a:next(identity))
    with pytest.raises(ValueError,match='分析期间'):
        methods.analyze_method(light,spec(symbol='000001.SZ'))


def test_checkpoint_not_published_after_raw_changes(environment,monkeypatch):
    light,store,calls=environment
    identity=iter([{'database':'before'},{'database':'after'}])
    monkeypatch.setattr(methods,'database_identity',lambda *a:next(identity))
    with pytest.raises(ValueError,match='检查点未发布'):
        methods.scan_methods(light,store,spec(symbols=[f'{i:06d}.SZ' for i in range(1,11)]))
    assert store.list()==[]


def test_generation_identity_covers_ma_and_native_dependencies():
    identity=methods._code_identity()
    assert 'light_ma_features.py' in identity['research']
    assert 'czsc_market_profiles.py' in identity['source']['code']
    assert 'engine' in identity['source'] and 'config' in identity['source']


def test_switching_models_reuses_structure_chart_without_run_or_pin(environment,monkeypatch):
    light,store,calls=environment
    from my_strategy.web_dashboard import trend_strategy
    built=[]
    def chart(**kwargs):
        built.append(kwargs)
        return {'symbol':kwargs['symbol'],'data_end':kwargs['end'],'capabilities':{},'data_version':'raw-prefix'}
    monkeypatch.setattr(trend_strategy,'analyze_trend_consensus',chart)
    for method in ('structure_kronos','structure_kronos_ma_mlp','structure_kronos_ma_l2'):
        result=methods.analyze_method(light,spec(method,symbol='000001.SZ'))
        assert result['analysis_method']==method
    assert len(built)==1
    entries=light.store.list('method_structure_charts')
    assert len(entries)==1 and entries[0]['pin_count']==0
    assert store.list()==[]
    # A model replacement changes scores, without invalidating structure inputs.
    light.resolve_score_model=lambda *a,**kw:{'model_id':'b'*64,'artifact':{'model_version':'new-model'}}
    result=methods.analyze_method(light,spec(symbol='000001.SZ'))
    assert result['method_context']['score_model_id']=='b'*64 and len(built)==1
    assert entries[0]['identity']['kind']=='method_structure_chart_v1'
    raw,frame,events,_=light.source()
    light.source=lambda *a,**kw:(raw,frame,events,{'input_sha':'revised-source'})
    methods.analyze_method(light,spec(symbol='000001.SZ'))
    assert len(built)==2


def test_completion_refills_cpu_while_first_stock_waits_and_keeps_prediction_order(environment):
    import threading
    light,store,calls=environment
    population=['000001.SZ','000002.SZ','000003.SZ','000004.SZ','000005.SZ','000006.SZ']
    fourth_started=threading.Event();started=[];guard=threading.Lock()
    original=light.source
    def source(symbol,*args,**kwargs):
        with guard:started.append(symbol)
        if symbol==population[3]:fourth_started.set()
        if symbol==population[0]:assert fourth_started.wait(2),'completed workers did not refill'
        return original(symbol,*args,**kwargs)
    light.source=source
    result=methods.scan_methods(light,store,spec(symbols=population,cpu_workers=2,max_predictions=6))
    assert result['status']=='succeeded'
    assert [symbol for symbol,_,_ in calls]==population
    assert [row['symbol'] for row in result['rows']]==population
    # CPU start order is OS scheduling; only serial prediction order is fixed.
    assert set(started[:4])==set(population[:4])
    assert result['compute_info']['preparation']['maximum_buffered_stocks']<=4


def test_ready_later_stock_never_uses_earlier_deferred_prediction_budget(environment, monkeypatch):
    import threading
    light,store,calls=environment
    clock=[100.]
    monkeypatch.setattr(methods,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    ready=threading.Event()
    original=light.source
    def source(symbol,*args,**kwargs):
        if symbol=='000001.SZ':
            assert ready.wait(2)
            clock[0]+=2.
            kwargs['stop'].check()
        value=original(symbol,*args,**kwargs)
        ready.set()
        return value
    light.source=source
    result=methods.scan_methods(light,store,spec(cpu_workers=2,budget_seconds=1))
    assert result['status']=='partial' and result['coverage']['deferred']==2
    assert calls==[]
