"""Bounded local research: causal structures, one Kronos instance and saved accounts.

No run directory is created by this module. Raw prices and published source runs
are read-only; previews live in the expiring content store.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import wraps
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
from filelock import FileLock, Timeout as FileLockTimeout

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, normalize_symbol, latest_market_date
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now

VERSION = 'agent_mlp_sim_v1'
SCORE_COLUMNS = ['ret_5', 'ret_20', 'ma20_bias', 'ma20_slope5', 'ma60_bias',
    'atr14_ratio', 'volatility20', 'vol_ratio20', 'amount_ratio20', 'drawdown20',
    'native_structure_buy_support', 'native_structure_sell_support',
    'native_w_demand_score', 'native_w_supply_score']


def plain(value):
    from my_strategy.services.czsc_trend_support import _json
    return _json(value)


def code_identity():
    base = Path(__file__).parent
    names = ['light_research.py', 'position_simulation.py', 'light_account_continuation.py', 'simple_score.py', 'kronos_tuning.py',
             'agent_evidence.py','light_daily_scheduler.py','light_prediction_inputs.py','light_source_preparation.py','method_scan_acceleration.py','method_compute.py','analysis_methods.py','light_score_v2.py','light_score_training.py','light_ma_features.py','light_qualification.py','light_experiment_registry.py',
             'kronos_prediction.py', 'czsc_native_evidence.py', 'czsc_market_features.py']
    identities = {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in names if (base / name).is_file()}
    for name in ['broker.py','cost_model.py','ledger.py','order.py']:
        identities['execution/'+name]=hashlib.sha256((base.parent/'execution'/name).read_bytes()).hexdigest()
    return identities


def score_code_identity():
    import inspect
    from my_strategy.services.incremental_source import numerical_input_contract
    names=('light_score_v2.py','light_score_training.py','light_ma_features.py','czsc_market_features.py','czsc_market_profiles.py',
           'light_qualification.py','light_experiment_registry.py','simple_score.py')
    result={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}
    result['actual_loaded_training_adapter']=hashlib.sha256(inspect.getsource(LightResearch.train_score.__wrapped__).encode()).hexdigest()
    result['observation_exclusion_adapter']=hashlib.sha256(inspect.getsource(training_observation_manifest).encode()).hexdigest()
    result['legacy_training_adapter']=hashlib.sha256(inspect.getsource(LightResearch._train_legacy_score.__wrapped__).encode()).hexdigest()
    result['input_contract']=numerical_input_contract()
    result['legacy_score_columns']=list(SCORE_COLUMNS)
    return result


def structure_identity():
    from my_strategy.services.czsc_trend_source import _native_code_hashes
    from my_strategy.services.czsc_analysis import strategy_config
    from my_strategy.services.czsc_native_evidence import NATIVE_SCHEMA_HASH
    import inspect
    return {'generating':_native_code_hashes(),'strategy':strategy_config(),
            'schema':NATIVE_SCHEMA_HASH,'numpy':np.__version__,'pandas':pd.__version__,
            'source_adapter':hashlib.sha256(inspect.getsource(LightResearch.source).encode()).hexdigest()}


def prediction_identity(device='auto'):
    from my_strategy.services.kronos_prediction import asset_status, _device, _source_identity
    import torch
    return {'assets':asset_status(),'source':_source_identity(),
            'device':_device(device),'dtype':'float32','torch_version':torch.__version__}


def serialized_compute(function):
    @wraps(function)
    def wrapped(self,*args,**kwargs):
        try:
            with self.compute_lock.acquire(timeout=0):
                return function(self,*args,**kwargs)
        except FileLockTimeout as exc:
            raise RuntimeError('已有轻量任务使用计算资源，请待其结束后再试') from exc
    return wrapped


class DeadlineStop:
    def __init__(self, stop, deadline): self.stop,self.deadline=stop,deadline
    def is_set(self):
        return bool(self.stop and self.stop.is_set()) or time.monotonic()>=self.deadline
    def check(self):
        if self.stop and self.stop.is_set():raise InterruptedError('已取消轻量研究任务')
        if time.monotonic()>=self.deadline:raise TimeoutError('已达到本次计算总时间预算')


def training_observation_manifest(manifest,full,mature,eligible,dates,end):
    """Retain the frozen B denominator and explain only label/input exclusions."""
    derived=deepcopy(manifest)
    keys=lambda frame:set(zip(frame.symbol.astype(str),frame.as_of.astype(str)))
    available,mature_keys,eligible_keys=keys(full),keys(mature),keys(eligible)
    lookup={day:i for i,day in enumerate(dates)}
    label_exclusions={(item['symbol'],item['as_of']):item['reason_codes'] for item in mature.attrs.get('label_audit',{}).get('exclusions',[])}
    for item in derived['observations']:
        key=(str(item['symbol']),str(item.get('as_of',item.get('date')))[:10])
        reason=None
        if key not in available:reason='outside_frozen_window'
        elif key not in mature_keys:
            at=lookup.get(key[1]);tail=dates[at+5] if at is not None and at+5<len(dates) else None
            reason='immature_label' if tail is None or tail>end else 'calendar_gap'
            codes=label_exclusions.get(key)
            if codes:reason=codes[0];item['training_exclusion_reason_codes']=list(codes)
        elif key not in eligible_keys:reason='core_feature_unavailable'
        if reason:item['training_exclusion_reason']=reason
        else:
            item.pop('training_exclusion_reason',None)
            item.pop('training_exclusion_reason_codes',None)
    derived['training_exclusion_contract']={'label_horizon':5,'end':end,
        'original_manifest_sha256':stable_hash(manifest),'eligible_rows':len(eligible_keys)}
    return derived


class LightResearch:
    def __init__(self, root=None, *, raw_db=None):
        from my_strategy.storage.research_store import ResearchStore
        from my_strategy.core.paths import METADATA_ROOT
        self.store = ResearchStore(root=root or METADATA_ROOT/'light_research')
        self.compute_lock = FileLock(str(self.store.root/'compute.lock'))
        self.raw_db = raw_db
        self.lock = threading.RLock()
        self.queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix='light-research')
        self.jobs = {}
        self.stops = {}
        from my_strategy.storage.task_journal import TaskJournal, bind_publication_intents
        self.journal = TaskJournal(self.store.root / 'task_journal.db')
        self.journal.import_legacy(self.store)
        self._publication_context=bind_publication_intents(self.store,self.journal)
        self._progress_flushed = {}
        self._closed = False

    def close(self):
        with self.lock:
            self._closed = True
            for job_id,stop in self.stops.items():
                stop.set()
                job=self.jobs.get(job_id)
                if job and job['status']=='queued':
                    job.update(status='cancelled',cancel_requested=True,finished_at=local_now().isoformat())
                    self.journal.update(job)
        self.queue.shutdown(wait=True, cancel_futures=True)
        from my_strategy.services.kronos_prediction import release_prediction_resources
        release_prediction_resources()
        self.store.close()

    def calendar(self):
        from my_strategy.services.czsc_trend_support import _calendar
        dates, run_id = _calendar(None)
        return dates, {'run_id': run_id, 'sha256': stable_hash(dates)}

    def current_score(self):
        pointer=self.store.get_setting('current_score')
        if not pointer:return None
        if 'artifact' in pointer:return pointer  # First lightweight build migration.
        result=self.store.get_by_key(pointer['cache_key'])
        return {**result,'model_ref':pointer} if result else None

    def select_score(self, result, reference):
        previous=self.current_score()
        previous_ref=(previous or {}).get('model_ref')
        if previous and previous_ref is None and previous.get('identity'):
            legacy_key=self.store.cache_key(previous['identity'],'score_models')
            if self.store.get_by_key(legacy_key): previous_ref={'cache_key':legacy_key}
        if previous_ref and previous_ref['cache_key']!=reference['cache_key']:
            rollback=self.store.get_setting('rollback_score')
            if rollback:self.store.unpin(rollback.get('model_ref',rollback)['cache_key'],'simple_score_rollback')
            self.store.pin(previous_ref['cache_key'],'simple_score_rollback')
            self.store.unpin(previous_ref['cache_key'],'simple_score_current')
            self.store.set_setting('rollback_score',previous_ref)
        self.store.pin(reference['cache_key'],'simple_score_current')
        self.store.set_setting('current_score',reference)
        self.store.limit_unpinned('score_models',keep=2,dry_run=False)
        return {**result,'model_ref':reference}

    def _check(self, stop=None, deadline=None):
        if stop and stop.is_set(): raise InterruptedError('已取消轻量研究任务')
        if deadline and time.monotonic() > deadline: raise TimeoutError('已达到本次推理时间预算')

    def _read_checkpoint(self,identity,namespace):
        reference=self.store.get_setting('checkpoint:'+namespace+':'+stable_hash(identity))
        return self.store.get_by_key(reference['cache_key']) if reference else None

    def _write_checkpoint(self,identity,value,namespace):
        pointer='checkpoint:'+namespace+':'+stable_hash(identity)
        previous=self.store.get_setting(pointer)
        # Each resumable request keeps one live object. Do not evict another
        # account's checkpoint when this account advances.
        reference=self.store.put({'scope':identity,'state_sha256':stable_hash(value)},value,
            namespace=namespace,pin=pointer)
        self.store.set_setting(pointer,reference)
        if previous and previous['cache_key']!=reference['cache_key']:
            self.store.unpin(previous['cache_key'],pointer)
        catalog_key='checkpoint_catalog:'+namespace
        catalog=[v for v in self.store.get_setting(catalog_key,[]) if v['pointer']!=pointer]
        if value.get('stage') in {'completed','partial'} and not value.get('pending'):
            self.store.unpin(reference['cache_key'],pointer)
            self.store.set_setting(pointer,None)
        else:catalog.append({'pointer':pointer,'reference':reference})
        for retired in catalog[:-30]:
            self.store.unpin(retired['reference']['cache_key'],retired['pointer'])
            self.store.set_setting(retired['pointer'],None)
        self.store.set_setting(catalog_key,catalog[-30:])
        self.store.limit_unpinned(namespace,keep=1,dry_run=False)
        return reference

    def _register_experiments(self,kind,configurations):
        from my_strategy.services.light_experiment_registry import merge_experiment_registry,persist_experiment_registry
        limits={'mlp':6,'kronos':2,'policy':6}
        if kind not in limits:raise ValueError('Unknown experiment kind')
        entries=[]
        for config in configurations:
            item={'kind':kind,'configuration':config,'strategy_identity':stable_hash(config)}
            item['experiment_id']=stable_hash(item)
            entries.append(item)
        registry=merge_experiment_registry(self.store.get_setting('light_experiment_registry'),entries)
        return persist_experiment_registry(self.store,registry)

    def source(self, symbol, end, *, stop=None, raw_snapshot=None):
        from my_strategy.services.incremental_source import IncrementalSource
        dates, calendar_identity = self.calendar()
        return IncrementalSource(store=self.store, raw_db=self.raw_db).source(symbol,end,stop=stop,
            calendar_dates=dates,calendar_identity=calendar_identity,raw_snapshot=raw_snapshot)

    def score_model(self, cache_key=None, *, model_type='mlp'):
        if cache_key:
            value=self.store.get_by_key(cache_key)
            if value is None or not isinstance(value.get('artifact'),dict): raise ValueError('评分候选不存在或已过期')
            return {**value,'model_ref':{'cache_key':cache_key},'model_id':cache_key}
        if model_type in {'l2','simple'}: return self.current_score()
        candidates=self.list_score_models(model_type='mlp')
        return self.score_model(candidates[0]['model_ref']['cache_key']) if candidates else None

    def resolve_score_model(self, cache_key=None, *, feature_profile='ma_trend_v1', as_of=None,
                            research_mode='historical_candidate'):
        """Resolve a matching frozen candidate without changing any selection.

        Explicit ids fail on a profile/date mismatch. Automatic research lookup
        skips incompatible candidates; paper eligibility is still checked by
        the scorer, never inferred from training completion or recency.
        """
        from my_strategy.services.light_qualification import verify_model
        if feature_profile not in {'mixed_v1','ma_trend_v1'}: raise ValueError('Unknown score feature profile')
        if research_mode not in {'historical_candidate','paper'}: raise ValueError('Unknown research mode')
        candidates = [self.score_model(cache_key)] if cache_key else [self.score_model(row['model_id'])
            for row in self.list_score_models(model_type='mlp') if row['feature_profile']==feature_profile]
        for model in candidates:
            artifact=model['artifact']
            reason=None
            if artifact.get('schema')!='light_score_v2' or artifact.get('feature_profile','mixed_v1')!=feature_profile:
                reason='评分模型与所选特征方案不一致'
            elif artifact.get('status')!='completed':reason='评分候选尚未完成'
            if not reason and as_of is not None:
                from my_strategy.services.simple_score import _dates
                decision=_dates([as_of],daily_close=True)[0]
                if decision<=_dates([artifact['available_after']])[0]:reason='评分模型开发校准证据晚于分析日期'
                elif research_mode=='paper' and decision<_dates([artifact['created_at']])[0]:reason='评分模型在该日期尚未实际冻结'
            if reason:
                if cache_key:raise ValueError(reason)
                continue
            verify_model(artifact)
            return model
        return None

    def list_score_models(self, model_type=None):
        result=[]
        namespaces=['mlp_score_models'] if model_type=='mlp' else ['score_models'] if model_type=='l2' else ['mlp_score_models','score_models']
        for namespace in namespaces:
            for item in self.store.list(namespace,limit=20):
                model=self.store.get_by_key(item['cache_key'])
                if not model: continue
                artifact=model.get('artifact') or {}
                population=artifact.get('population') or {}
                mature=sum(v.get('mature_rows',0) for v in population.values())
                result.append({'model_ref':{'cache_key':item['cache_key']},'model_id':item['cache_key'],'model_type':model.get('model_type','l2'),
                    'variant':artifact.get('variant'),'model_version':artifact.get('model_version'),
                    'feature_profile':artifact.get('feature_profile','mixed_v1'),
                    'training_end':model.get('training_end'),'symbols':model.get('symbols'),
                    'row_count':model.get('row_count'),'status':model.get('status'),'population':population,
                    'mature_rows':mature if population else None,
                    'excluded_rows':max(0,model.get('row_count',0)-mature) if population else None,
                    'feature_dimensions':(artifact.get('architecture') or {}).get('input_dimensions'),
                    'qualification':artifact.get('qualification'),'prediction_checks':artifact.get('prediction_checks'),
                    'feature_columns':model.get('feature_columns'),'created_at':item['created_at']})
        return sorted(result,key=lambda x:x['created_at'],reverse=True)

    def scorer(self, model, frames, evidence, *, score_source_events=None,
               research_mode='historical_candidate', qualification_event=None):
        if not model: return lambda symbol,day,forecast: {'status':'unavailable','reason':'score_model_not_selected'}
        # Full source history is separate from the account's replay-window
        # evidence. DataFrame attrs are deep-copied by pandas row operations.
        source_events=score_source_events if score_source_events is not None else evidence
        artifact=model['artifact']
        is_mlp=artifact.get('schema')=='light_score_v2' or artifact.get('model_type')=='torch_mlp'
        if is_mlp:
            from my_strategy.services.light_score_training import prepare_score_features
            variant=artifact.get('variant','A')
            feature_profile=artifact.get('feature_profile','mixed_v1')
            calendar=self.calendar()[0] if feature_profile=='ma_trend_v1' else None
            prepared={symbol:prepare_score_features(frame,events=source_events.get(symbol,()),variant=variant,
                      feature_profile=feature_profile,calendar_dates=calendar)
                      for symbol,frame in frames.items()}
        else: prepared=frames
        def evaluate(symbol,day,forecast):
            if is_mlp:
                from my_strategy.services.light_score_v2 import score_light_samples
                frame=prepared[symbol]
                row=frame.loc[frame.date.dt.strftime('%Y-%m-%d')==day].copy()
                if len(row)!=1: raise ValueError('评分日没有唯一真实行情')
                timing={}
                if variant=='B':
                    from my_strategy.services.light_score_training import prepare_score_features,KRONOS_FEATURES
                    # Same fixed trailing/structure features, with only this day's
                    # previously generated forecast attached. No model-generated population.
                    joined=prepare_score_features(frames[symbol],events=source_events.get(symbol,()),
                        forecasts={(symbol,day):forecast},variant='B',decision_time=(forecast or {}).get('actual_available_at'))
                    selected=joined.loc[joined.date.dt.strftime('%Y-%m-%d')==day]
                    for column in [*KRONOS_FEATURES,'kronos_state','feature_available_at']: row[column]=selected[column].to_numpy()
                    if (forecast or {}).get('actual_available_at'):timing={'decision_time':forecast['actual_available_at']}
                scored=score_light_samples(artifact,row,as_of=day,historical_reconstruction=research_mode=='historical_candidate',
                    mode=research_mode,qualification_event=qualification_event,**timing)
            else:
                from my_strategy.services.simple_score import score_samples
                frame=prepared[symbol]
                row=frame.loc[pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')==day]
                scored=score_samples(artifact,row,as_of=day,historical_reconstruction=True)
            result={'status':scored.get('status','scored'),'score':scored['scores'][0],
                'probabilities':scored['probabilities'][0],'qualification':scored['qualification'],
                'model_version':scored['model_version'],'model_type':'torch_mlp' if is_mlp else 'l2',
                'feature_profile':artifact.get('feature_profile','mixed_v1'),'scorer_id':scored.get('scorer_id'),
                'variant':artifact.get('variant'),'reason':next(iter(scored.get('row_statuses',[])),None)}
            if is_mlp and feature_profile=='ma_trend_v1':
                result['input_reason_codes']=row.iloc[0].get('score_reason_codes',[])
            if is_mlp:
                try:
                    baseline=score_light_samples(artifact,row,as_of=day,historical_reconstruction=research_mode=='historical_candidate',mode=research_mode,model_kind='l2',**timing)
                    result['l2_comparison']={'status':baseline.get('status','scored'),'score':baseline['scores'][0],
                        'probabilities':baseline['probabilities'][0],'qualification':baseline['qualification'],
                        'model_version':baseline['model_version'],'model_type':'l2','variant':variant,'consumes_kronos':variant=='B',
                        'feature_profile':feature_profile,'scorer_id':baseline.get('scorer_id')}
                except ValueError as exc:
                    result['l2_comparison']={'status':'unavailable','reason':str(exc),'model_type':'l2','variant':variant}
                result['consumes_kronos']=variant=='B'
            return result
        return evaluate

    @serialized_compute
    def prediction(self, symbol, end=None, *, stop=None, raw_snapshot=None, _prepared_input=None, **parameters):
        from my_strategy.services.kronos_prediction import predict
        parameters={**dict(lookback=128,horizon=5,samples=8,seed=42,device='auto',temperature=.8,top_p=.9),**parameters}
        symbol = normalize_symbol(symbol)
        end = end or latest_market_date(db_path=self.raw_db)
        if _prepared_input is not None:
            from my_strategy.services.light_prediction_inputs import PreparedPredictionInput
            if (not isinstance(_prepared_input,PreparedPredictionInput) or raw_snapshot is not None
                    or (_prepared_input.symbol,_prepared_input.as_of)!=(symbol,end)):
                raise ValueError('Prepared prediction input does not match the causal request')
            raw=_prepared_input.raw
        else:raw = load_bars(symbol, end=end, db_path=self.raw_db) if raw_snapshot is None else raw_snapshot
        if raw.empty or set(raw.symbol.astype(str))!={symbol} or str(raw.iloc[-1].date)[:10] != end:
            raise ValueError(f'{symbol} 缺少 {end} 行情，不能使用旧日报价')
        dates, cal = ((_prepared_input.calendar_dates,_prepared_input.calendar_identity)
                      if _prepared_input is not None else self.calendar())
        future = [d for d in dates if d > end][:int(parameters.get('horizon', 5))]
        from my_strategy.services.kronos_prediction import asset_status
        identity = {'kind': 'prediction-v1', 'symbol': symbol, 'as_of': end,
            'data': raw.attrs['data_version'], 'calendar': cal,
            'exact_price_input_sha256': (_prepared_input.exact_price_input_sha256
                if _prepared_input is not None else stable_hash(plain(raw.to_dict('records')))),
            'model': prediction_identity(parameters.get('device','auto')),
            'parameters': parameters, 'code': 'kronos_request_cache_v2'}
        # Asset status can include volatile load-state fields. The predictor also
        # supplies its full immutable identity; this outer key is a request cache.
        saved = self.store.get(identity, namespace='predictions')
        if saved:return {**saved,'result_ref':{'cache_key':self.store.cache_key(identity,'predictions')},'reused':True}
        self.store.gc(dry_run=False)
        self._check(stop)
        result = predict(raw, future, symbol=symbol, as_of=end, calendar_identity=cal,
            calendar_verified=True, cancel_check=(stop.is_set if stop else None), **parameters)
        self._check(stop)
        result = plain(result)
        # A preview keeps summaries and a bounded 8 paths, not a model/run copy.
        result['samples'] = result.get('samples', [])[:8]
        result['retained_diagnostic_samples'] = len(result['samples'])
        result['reused'] = False
        ref = self.store.put(identity, result, namespace='predictions')
        result['result_ref'] = ref
        return result

    def account_prediction_inputs(self, raw_frames, dates, calendar_identity, *, stop=None):
        from my_strategy.services.light_prediction_inputs import AccountPredictionInputs
        return AccountPredictionInputs(raw_frames, dates, calendar_identity, stop=stop)

    @serialized_compute
    def tune(self, symbols, validation_dates, test_dates, *, device='auto', budget_seconds=300, stop=None):
        from my_strategy.services.kronos_tuning import tune_kronos,DEFAULT_CONFIGURATIONS
        self._register_experiments('kronos',list(DEFAULT_CONFIGURATIONS))
        dates, calendar_identity = self.calendar()
        cases=[]
        for symbol in symbols:
            frame=load_bars(symbol,db_path=self.raw_db)
            for day in validation_dates+test_dates:
                self._check(stop)
                future=[d for d in dates if d>day][:5]
                actual=frame.loc[frame.date.dt.strftime('%Y-%m-%d').isin(future)].copy()
                cases.append({'symbol':symbol,'as_of':day,'frame':frame,
                    'future_dates':future,'actual_future':actual,
                    'calendar_identity':calendar_identity,'calendar_verified':True})
        identity={'kind':'kronos-tuning-v1','symbols':symbols,'validation_dates':validation_dates,
            'test_dates':test_dates,'device':device,'budget_seconds':budget_seconds,
            'inputs':[{'symbol':c['symbol'],'as_of':c['as_of'],'data':c['frame'].attrs['data_version']} for c in cases],
            'calendar':calendar_identity,'model':prediction_identity(device),'code':code_identity()}
        cached=self.store.get(identity,namespace='tuning')
        if cached:return {**cached,'reused':True}
        result=tune_kronos(cases,validation_dates,test_dates,device=device,
            budget_seconds=budget_seconds,stop=(stop.is_set if stop else None))
        self._check(stop)
        result=plain(result)
        if result.get('status')=='completed':
            self.store.gc(dry_run=False)
            self.store.put(identity,result,namespace='tuning')
        return result

    @serialized_compute
    def _train_legacy_score(self, symbols, end='2025-12-31', *, stop=None, progress=None, select=False, budget_seconds=600):
        from my_strategy.services.simple_score import fit_simple_score
        from my_strategy.services.light_score_training import direction_label_contract
        if not 0<float(budget_seconds)<=600:raise ValueError('Training budget/thread limit exceeded')
        started=time.monotonic();deadline=started+float(budget_seconds);bounded_stop=DeadlineStop(stop,deadline)
        label_audits=[]
        if select:raise ValueError('训练候选不能自动晋升；须独立资格事件')
        population = []
        sources = []
        for symbol in symbols:
            self._check(stop,deadline)
            raw, frame, events, identity = self.source(symbol, end, stop=bounded_stop)
            frame = frame.copy()
            from my_strategy.services.light_score_training import make_direction_training_frame
            dates, _ = self.calendar()
            frame = make_direction_training_frame(frame, dates)
            label_audits.append(frame.attrs['label_audit'])
            frame = frame.loc[frame[SCORE_COLUMNS].notna().all(axis=1)].copy()
            population.append(frame); sources.append(identity)
            if progress: progress(stage='simple_score_inputs', symbol=symbol)
        data = pd.concat(population, ignore_index=True).sort_values(['as_of','symbol'])
        days = sorted(data.as_of.unique())
        if len(days) < 400: raise ValueError('简单评分需要至少400个有成熟标签的历史日期')
        boundaries = [0, int(len(days)*.55), int(len(days)*.68), int(len(days)*.79), int(len(days)*.89), len(days)]
        ranges = {name:[days[boundaries[i]], days[boundaries[i+1]-1]]
                  for i,name in enumerate(['train','tune','calibration','policy','test'])}
        identity = {'kind':'simple-score-v1', 'sources':sources, 'splits':ranges,
                    'features':SCORE_COLUMNS,'label_contract':direction_label_contract(), 'code':score_code_identity()}
        saved = self.store.get(identity, namespace='score_models')
        if saved:
            ref={'cache_key':self.store.cache_key(identity,'score_models')}
            selected=self.select_score(saved,ref) if select else {**saved,'model_ref':ref}
            return {**selected, 'reused':True}
        artifact = fit_simple_score(data, SCORE_COLUMNS, ranges,
            source_identity=identity, cancel_check=bounded_stop.check,label_contract=direction_label_contract())
        artifact = plain(artifact)
        self._check(stop,deadline)
        result = {'artifact':artifact, 'identity':identity, 'symbols':symbols,
            'training_end':end, 'split_ranges':ranges, 'row_count':len(data),
            'feature_columns':SCORE_COLUMNS,'label_audits':label_audits,
            'timing':{'budget_scope':'preparation_and_training','budget_seconds':budget_seconds,'total_compute_seconds':time.monotonic()-started},
            'status':'completed', 'reused':False}
        reference = self.store.put(identity, result, namespace='score_models')
        return self.select_score(result,reference) if select else {**result,'model_ref':reference,'model_type':'l2'}

    @serialized_compute
    def train_score(self, symbols, end='2025-12-31', *, model_type='mlp',variant='A',feature_profile='mixed_v1',device='cpu',
                    budget_seconds=600,select=False,observation_manifest=None,stop=None,progress=None):
        if feature_profile not in {'mixed_v1','ma_trend_v1'}:raise ValueError('Unknown score feature profile')
        if feature_profile=='ma_trend_v1' and (variant!='A' or model_type!='mlp'):
            raise ValueError('均线方案仅训练 A 型 MLP 及其同输入 L2；Kronos 保留外部证据')
        if model_type=='l2':
            return self._train_legacy_score(symbols,end,stop=stop,progress=progress,select=select,budget_seconds=budget_seconds)
        if model_type!='mlp': raise ValueError('model_type must be mlp or l2')
        if not 1<=len(symbols)<=20: raise ValueError('首批学习池限定1至20股')
        if select: raise ValueError('训练候选不能自动晋升；须独立资格事件')
        if variant=='B' and not observation_manifest:
            return {'status':'blocked','reason':'fusion_data_pending','variant':'B',
                'message':'需先冻结候选/基线持仓观察人口并补齐有界Kronos历史证据；未自动启动历史预测或训练。'}
        from my_strategy.services.light_score_training import prepare_score_features,make_direction_training_frame,chronological_split_ranges,direction_label_contract
        from my_strategy.services.light_score_v2 import fit_light_score
        from my_strategy.services.light_experiment_registry import persist_experiment_registry
        if not 0<float(budget_seconds)<=600:raise ValueError('Training budget/thread limit exceeded')
        started=time.monotonic();deadline=started+float(budget_seconds)
        bounded_stop=DeadlineStop(stop,deadline)
        frames=[];sources=[];dates,calendar_identity=self.calendar()
        forecasts={(o['symbol'],o.get('date',o.get('as_of'))):o.get('forecast')
                   for o in (observation_manifest or {}).get('observations',[])}
        for symbol in symbols:
            self._check(stop,deadline)
            if feature_profile=='ma_trend_v1':
                # MA training needs only raw trailing prices and their quality
                # evidence; changing score inputs never rebuilds native sources.
                frame=load_bars(symbol,end=end,db_path=self.raw_db)
                events=[]
                fields=[k for k in ('symbol','date','open','high','low','close','volume','amount','source','has_trade_price','available_at') if k in frame]
                source={'kind':'ma_raw_source_v2','symbol':symbol,'end':end,'calendar':calendar_identity,
                    'data_sha256':stable_hash(plain(frame[fields].to_dict('records'))),
                    'price_basis':frame.attrs.get('price_basis'),'data_version':frame.attrs.get('data_version')}
            else:
                _,frame,events,source=self.source(symbol,end,stop=bounded_stop)
            frame=prepare_score_features(frame,events,forecasts,variant=variant,
                feature_profile=feature_profile,calendar_dates=dates)
            frames.append(frame);sources.append(source)
            if progress:progress(stage='mlp_inputs',symbol=symbol,completed=len(frames),total=len(symbols))
        full=pd.concat(frames,ignore_index=True)
        self._check(stop,deadline)
        data=make_direction_training_frame(full,dates,horizon=5,threshold=.003)
        label_audit=data.attrs['label_audit']
        if variant=='B':
            population={(s,str(d)[:10]) for s,d in forecasts}
            data=data.loc[[(str(r.symbol),str(r.as_of)) in population for r in data.itertuples()]].copy()
        if len(data)>100000: raise ValueError('成熟样本超过10万，需重新冻结有界研究池')
        core_valid=data.score_input_eligible.eq(True) if feature_profile=='ma_trend_v1' else data[SCORE_COLUMNS].notna().all(axis=1)
        core_excluded=int((~core_valid).sum())
        if variant=='B':
            observation_manifest=training_observation_manifest(observation_manifest,full,data,
                data.loc[core_valid],dates,end)
        data=data.loc[core_valid].copy()
        if variant=='B' and data.as_of.nunique()<50:
            return plain({'status':'blocked','reason':'fusion_data_pending','variant':'B','model_type':'mlp',
                'message':'融合观察尚不足50个不同成熟日期，继续有界采集后再训练。',
                'row_count':len(data),'mature_dates':data.as_of.nunique(),
                'observation_manifest':observation_manifest,'selected':False})
        ranges=chronological_split_ranges(data)
        identity={'kind':'light-score-v2','variant':variant,'feature_profile':feature_profile,'sources':sources,'splits':ranges,
            'device':device,'observation_manifest':observation_manifest,'label_contract':direction_label_contract(),'code':score_code_identity()}
        cached=self.store.get(identity,namespace='mlp_score_models')
        if cached:return {**cached,'model_ref':{'cache_key':self.store.cache_key(identity,'mlp_score_models')},'reused':True}
        self._check(stop,deadline)
        preparation_seconds=time.monotonic()-started
        try:
            artifact=fit_light_score(data,ranges,variant=variant,feature_profile=feature_profile,source_identity=identity,
                label_contract=direction_label_contract(),device=device,budget_seconds=max(.001,deadline-time.monotonic()),
                cancel_check=bounded_stop.check,observation_manifest=observation_manifest,
                progress=progress,
                experiment_registry=self.store.get_setting('light_experiment_registry'),
                registration_callback=lambda value:persist_experiment_registry(self.store,value))
        except ValueError as exc:
            if variant!='B' or not ('insufficient mature rows' in str(exc) or 'five examples of every class' in str(exc)):raise
            return plain({'status':'blocked','reason':'fusion_data_pending','variant':'B','model_type':'mlp',
                'message':str(exc),'observation_manifest':observation_manifest,'row_count':len(data),'selected':False})
        self._check(stop,deadline)
        if artifact.get('status')!='completed':
            return plain({'status':'blocked','reason':'fusion_data_pending','model_type':'mlp','variant':variant,'artifact':artifact,'selected':False})
        result=plain({'status':'completed','model_type':'mlp','variant':variant,'feature_profile':feature_profile,'artifact':artifact,
            'identity':identity,'symbols':symbols,'training_end':end,'split_ranges':ranges,
            'row_count':len(data),'candidate_row_count':len(full),'label_audit':label_audit,
            'timing':{'budget_scope':'preparation_and_training','budget_seconds':budget_seconds,'preparation_seconds':preparation_seconds,
                'total_compute_seconds':time.monotonic()-started},'core_excluded_rows':core_excluded,'feature_columns':artifact.get('features'),
            'reused':False,'selected':False})
        ref=self.store.put(identity,result,namespace='mlp_score_models',pin='frozen_mlp_candidate')
        self.store.limit_unpinned('mlp_score_models',keep=6,dry_run=False)
        return {**result,'model_ref':ref,'model_id':ref['cache_key']}

    @serialized_compute
    def collect_score_observations(self,symbols=None,start=None,end=None,*,manifest_id=None,
                                   max_calls=200,budget_seconds=300,device='auto',samples=32,
                                   lookback=128,horizon=5,stop=None,progress=None):
        from my_strategy.services.position_simulation import run_position_simulation
        from my_strategy.services.agent_evidence import volume_candidates
        if not 1<=int(max_calls)<=200 or not 1<=float(budget_seconds)<=300:
            raise ValueError('每批最多200次预测、300秒')
        deadline=time.monotonic()+budget_seconds
        params=dict(device=device,samples=samples,lookback=lookback,horizon=horizon,seed=42,temperature=.8,top_p=.9)
        if manifest_id:
            manifest=self._read_checkpoint({'manifest_id':manifest_id},'score_observation_manifests')
            if not manifest or manifest.get('schema')!='score_observation_manifest_v1':raise ValueError('冻结观察人口不存在')
            symbols=manifest['symbols'];start=manifest['start'];end=manifest['end'];params=manifest['parameters']
        else:manifest=None
        self._register_experiments('kronos',[params])
        if not symbols or not start or not end or not 1<=len(symbols)<=20:
            raise ValueError('需明确1至20股历史观察池、开始和结束日期')
        dates,cal=self.calendar();frames={};events={};sources=[]
        for symbol in sorted(set(symbols)):
            self._check(stop)
            try:
                _,frame,source_events,source=self.source(symbol,end,stop=DeadlineStop(stop,deadline))
            except (InterruptedError,TimeoutError):
                if stop and stop.is_set():raise
                if time.monotonic()<deadline:raise
                return {'status':'blocked','reason':'observation_population_pending',
                    'message':'本批来源准备达到时间预算；已完成的逐股缓存可在新请求复用。',
                    'prepared_symbols':sorted(frames),'pending_symbol':symbol,'selected':False}
            if 'symbol' not in frame:frame=frame.assign(symbol=symbol)
            frames[symbol]=frame;sources.append(source)
            events[symbol]=[e for e in source_events+volume_candidates(frame,symbol) if start<=str(e['date'])[:10]<=end]
        population_config=dict(policy='structure_rules',candidate_sessions=3,research_mode='historical_candidate',
                               enable_agent_evidence=False)
        population_identity={'symbols':sorted(frames),'start':start,'end':end,'sources':sources,'calendar':cal,
            'baseline':population_config,'parameters':params,'code':code_identity(),'model':prediction_identity(params['device'])}
        if manifest:
            if manifest['population_identity']!=population_identity:raise ValueError('冻结观察人口输入或代码已改变，需新清单')
        else:
            baseline=run_position_simulation(frames,events,config=population_config,start=start,end=end,
                calendar=dates,include_benchmarks=False,check_cancel=(stop.is_set if stop else None))
            candidates={}
            for opportunity in baseline['opportunities']:
                candidates.setdefault(opportunity['date'],set()).add(opportunity['symbol'])
            observations=[]
            for row in baseline['daily']:
                held={s for s,p in row['positions'].items() if p['shares'] or p.get('receivable_shares')}
                needed=sorted(held)+sorted(candidates.get(row['date'],set())-held)
                for symbol in needed:
                    item=dict(symbol=symbol,as_of=row['date'],status='deferred',forecast=None,
                        population_reason='baseline_holding' if symbol in held else 'frozen_candidate')
                    data=frames[symbol];observed=pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')
                    at=dates.index(row['date']);following=dates[at+1:at+6]
                    selected=data.loc[observed==row['date']]
                    if len(following)<5 or following[-1]>end:
                        item['training_exclusion_reason']='immature_label'
                    elif any(day not in set(observed) for day in following) or len(selected)!=1:
                        item['training_exclusion_reason']='calendar_gap'
                    elif not np.isfinite(selected.reindex(columns=SCORE_COLUMNS).to_numpy(dtype=float)).all():
                        item['training_exclusion_reason']='core_feature_unavailable'
                    observations.append(item)
            manifest={'schema':'score_observation_manifest_v1','model_independent':True,
                'source_policy_id':stable_hash(population_config),'frozen_at':local_now().isoformat(),
                'symbols':sorted(frames),'start':start,'end':end,'parameters':params,'total_calls':0,'batch_count':0,
                'observations':observations,'population_identity':population_identity,'attempts':[]}
        manifest_id=stable_hash(population_identity)
        key={'manifest_id':manifest_id}
        def save_manifest():
            from collections import Counter
            manifest['coverage']=dict(Counter(o['status'] for o in manifest['observations']))
            manifest['requested']=len(manifest['observations'])
            manifest['status']='completed' if not manifest['coverage'].get('deferred') else 'partial'
            manifest['reason']='research_total_budget_reached' if manifest['total_calls']>=2000 else 'batch_budget_or_pending'
            ref=self._write_checkpoint(key,plain(manifest),'score_observation_manifests')
            return {**plain(manifest),'manifest_id':manifest_id,'manifest_ref':ref}
        if manifest['batch_count']>=10 or manifest['total_calls']>=2000:
            return save_manifest()
        manifest['batch_count']+=1;made=0
        try:
            for item in manifest['observations']:
                if item['status']!='deferred':continue
                self._check(stop)
                if made>=max_calls or manifest['total_calls']>=2000 or time.monotonic()>=deadline:break
                made+=1;manifest['total_calls']+=1
                if progress:progress(stage='mlp_frozen_observations',symbol=item['symbol'],date=item['as_of'],
                    completed=made,total=min(max_calls,2000-manifest['total_calls']+made))
                attempt=dict(symbol=item['symbol'],as_of=item['as_of'],ordinal=manifest['total_calls'])
                try:
                    forecast=self.prediction(item['symbol'],item['as_of'],stop=DeadlineStop(stop,deadline),**params)
                    forecast=deepcopy(forecast);forecast.pop('reused',None);forecast.pop('resource',None)
                    item.update(forecast=forecast,status='succeeded' if (forecast.get('quality') or {}).get('prediction_usable') is True else 'unusable')
                except InterruptedError:
                    attempt['status']='cancelled' if stop and stop.is_set() else 'deadline'
                    manifest['attempts'].append(attempt)
                    if attempt['status']=='deadline' and time.monotonic()>=deadline:break
                    raise
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    item.update(status='failed',forecast={'as_of':item['as_of'],'quality':{'prediction_usable':False},'reason':str(exc)})
                attempt['status']=item['status'];manifest['attempts'].append(attempt)
                save_manifest()
        except InterruptedError:
            saved=save_manifest()
            raise InterruptedError('已取消本批采集；冻结清单 '+saved['manifest_id']+' 保留，旧任务不会恢复')
        return save_manifest()

    @serialized_compute
    def backtest(self, symbols, start, end, *, initial_cash=1000000, samples=8,
                 lookback=128, horizon=5, seed=42, device='auto', temperature=.8,
                 top_p=.9,max_predictions=60,budget_seconds=300,save=False,
                 stop=None,progress=None,corporate_actions=None,score_model_id=None,
                 score_model_type='mlp',feature_profile=None,research_mode='historical_candidate',candidate_sessions=3,
                 fusion_mode='evidence',execution_mode='historical',qualification_event=None,cost_stress=1.0):
        from my_strategy.services.light_daily_scheduler import run_daily_schedule,ScheduleCancelled
        from my_strategy.services.agent_evidence import volume_candidates
        if research_mode not in {'historical_candidate','paper'}:raise ValueError('Unknown research mode')
        if execution_mode=='forward' and research_mode!='paper':raise ValueError('前向模拟须使用paper模式及独立资格')
        started=time.monotonic();deadline=started+float(budget_seconds)
        prediction_stop=DeadlineStop(stop,deadline)
        dates,cal=self.calendar();frames={};raw_frames={};evidence={};sources=[];score_source_events={}
        symbols=sorted(set(normalize_symbol(s) for s in symbols))
        from my_strategy.services.light_source_preparation import prepare_account_sources
        source_preparation={}
        for symbol, prepared in prepare_account_sources(self,symbols,end,stop=prediction_stop,progress=progress,stats=source_preparation):
            raw,frame,events,source=prepared
            raw_frames[symbol]=raw
            if 'symbol' not in frame:frame=frame.assign(symbol=symbol)
            score_source_events[symbol]=events
            frames[symbol]=frame;sources.append(source)
            evidence[symbol]=[{**e} for e in events+volume_candidates(frame,symbol)
                              if start<=str(e['date'])[:10]<=end]
        if feature_profile is not None and score_model_type!='mlp':
            raise ValueError('特征方案选择使用 MLP 候选及其配对 L2；旧简单 L2 不兼容')
        model=(self.resolve_score_model(score_model_id,feature_profile=feature_profile,as_of=end,research_mode=research_mode)
               if feature_profile is not None else self.score_model(score_model_id,model_type=score_model_type))
        params=dict(samples=samples,lookback=lookback,horizon=horizon,seed=seed,device=device,temperature=temperature,top_p=top_p)
        configs={policy:dict(strategy=policy,initial_cash=float(initial_cash),research_only=True,
            candidate_sessions=candidate_sessions,fusion_mode=fusion_mode,research_mode=research_mode,
            enable_agent_evidence=True,cost_stress=cost_stress) for policy in ('structure_rules','structure_kronos','structure_kronos_score')}
        if model and (model.get('artifact') or {}).get('model_type')=='torch_mlp':
            configs['structure_kronos_l2']={**configs['structure_kronos_score'],'strategy':'structure_kronos_l2'}
        self._register_experiments('policy',[{'accounts':configs,'cost_stress':float(cost_stress)}])
        identity={'kind':'agent-simulation-v1','version':VERSION,'sources':sources,'calendar':cal,
            'start':start,'end':end,'cash':float(initial_cash),'prediction':params,'model':prediction_identity(device),
            'score':model['artifact']['model_version'] if model else None,'score_model_type':score_model_type,
            'max_predictions':max_predictions,'budget_seconds':budget_seconds,'configs':configs,
            'corporate_actions':corporate_actions or [],'code':code_identity(),
            'execution_mode':execution_mode,'qualification_event':qualification_event}
        from my_strategy.services.light_account_compatibility import execution_adapter_identity
        identity['execution_adapter']=execution_adapter_identity()
        if feature_profile is not None:identity['feature_profile']=feature_profile
        cached=self.store.get(identity,namespace='simulations')
        if cached:
            if save:self.save_result(cached)
            return {**cached,'reused':True,'saved':save or self.store.is_pinned(cached['result_id'])}
        self.store.gc(dry_run=False)
        checkpoint_identity={'request':identity}
        checkpoint=self._read_checkpoint(checkpoint_identity,'simulation_checkpoints')
        if checkpoint and checkpoint.get('stage') not in {'cancelled','day_committed','interrupted'}:checkpoint=None
        scorer=self.scorer(model,frames,evidence,score_source_events=score_source_events,
            research_mode=research_mode,qualification_event=qualification_event)
        def save_checkpoint(value):
            # One content-addressed live checkpoint identity; overwritten index,
            # immutable objects reclaimed normally. No per-day run directories.
            if value.get('stage')!='observing':
                self._write_checkpoint(checkpoint_identity,plain(value),'simulation_checkpoints')
        preparation_seconds=time.monotonic()-started
        with self.account_prediction_inputs(raw_frames,dates,cal,stop=prediction_stop) as inputs:
            from my_strategy.services.light_prediction_inputs import account_prediction_callback
            predict_account,report_progress=account_prediction_callback(self,inputs,params,prediction_stop,progress)
            scheduled=run_daily_schedule(frames,evidence,configs=configs,calendar=dates,start=start,end=end,
                prediction=predict_account,score=scorer,
                max_predictions_per_day=max_predictions,deadline=deadline,stop=stop,
                checkpoint=checkpoint,checkpoint_callback=save_checkpoint,progress=report_progress,
                checkpoint_stages={'day_committed','cancelled','interrupted','completed','partial'},
                corporate_actions=corporate_actions,execution_mode=execution_mode,checkpoint_interval_seconds=5.)
            input_preparation=dict(inputs.stats)
        self._check(stop)
        comparisons=scheduled['comparisons'];evidence=scheduled['evidence']
        if scheduled['status']=='partial':identity={**identity,'partial_observation_scope':stable_hash(evidence),
            'partial_stage':scheduled['scheduler'].get('stage'),'complete_through':scheduled['scheduler'].get('complete_through')}
        result_id=self.store.cache_key(identity,'simulations')
        existing=self.store.get(identity,namespace='simulations')
        if existing:
            if save:self.save_result(existing)
            return {**existing,'reused':True,'saved':save or self.store.is_pinned(result_id)}
        result=plain({'status':scheduled['status'],'result_id':result_id,'version':VERSION,
            'created_at':local_now().isoformat(),'currency':'CNY','symbols':symbols,'start':start,'end':end,'identity':identity,
            'comparisons':comparisons,'prediction_coverage':scheduled['coverage'],'prediction_errors':scheduled['errors'],
            'score_model':model.get('model_ref') if model else None,
            'score_model_type':model.get('model_type',score_model_type) if model else score_model_type,
            'score_variant':(model or {}).get('artifact',{}).get('variant'),
            'feature_profile':(model or {}).get('artifact',{}).get('feature_profile','mixed_v1') if model else feature_profile,
            'qualification_mode':research_mode,'scheduler':scheduled['scheduler'],
            'timing':{'budget_scope':'preparation_and_replay','preparation_seconds':preparation_seconds,'source_preparation':source_preparation,
                'total_compute_seconds':time.monotonic()-started,'budget_seconds':budget_seconds,
                'input_preparation':input_preparation},
            'risk_comparisons':scheduled.get('risk_comparisons'),
            'comparison_scope':scheduled.get('comparison_scope','completed_evaluation'),
            'diagnostics':{'opportunity_funnel':{k:v['opportunity_funnel'] for k,v in comparisons.items()},
                           'score_model_missing':model is None},
            'chart_bars':{s:plain(f.loc[(pd.to_datetime(f.date)>=pd.Timestamp(start))&(pd.to_datetime(f.date)<=pd.Timestamp(end)),
                ['date','open','high','low','close']].to_dict('records')) for s,f in frames.items()},
            'source_events':evidence,'corporate_actions':corporate_actions or [],'saved':save,'reused':False,
            'limitations':['历史候选试验与自动纸上账户资格分别核验；没有独立前瞻收益认证。',
                'Kronos预训练范围未知；公司行为和历史ST/涨跌停缺完整资料，收益为未复权研究回放。',
                '持仓/候选逐日共用因果队列；缓存也占每日额度，预算延后保持partial并独立处理风险。']})
        from my_strategy.services.light_experiment_report import build_experiment_report
        result['experiment_report']=build_experiment_report(result)
        self.store.put(identity,result,namespace='simulations',pin='saved_simulation' if save else None)
        self.store.put({'id':result_id},{'identity':identity},namespace='result_lookup',pin='saved_simulation' if save else None)
        if save and model:self.store.pin(model['model_ref']['cache_key'],'saved_simulation:'+result_id)
        return result

    def result(self, result_id):
        lookup = self.store.get({'id':result_id},namespace='result_lookup')
        if not lookup: raise KeyError(result_id)
        result = self.store.get(lookup['identity'],namespace='simulations')
        if result is None: raise KeyError(result_id)
        return {**result,'saved':self.store.is_pinned(result_id)}

    def save_result(self,result):
        result_id=result['result_id']
        self.store.pin(result_id,'saved_simulation')
        self.store.pin(self.store.cache_key({'id':result_id},'result_lookup'),'saved_simulation')
        if result.get('score_model'):
            self.store.pin(result['score_model']['cache_key'],'saved_simulation:'+result_id)

    def enable_account(self, result_id, enabled=True, *, shadow_research=False):
        result=self.result(result_id)
        if enabled and result.get('version')!=VERSION:
            raise ValueError('旧版本账户保留原始结果；启用每日续算需建立当前版本的新分支')
        if enabled and result.get('version')==VERSION:
            if result.get('status')!='completed':raise ValueError('不完整评估不能启用每日账户，请先续完观察阶段')
            if result.get('qualification_mode')=='historical_candidate' and not shadow_research:
                raise ValueError('历史候选未取得自动增仓资格；需建立paper前向分支，或显式启用影子研究续算')
            if not shadow_research:
                from my_strategy.services.light_qualification import resolve_qualification
                model=self.score_model((result.get('score_model') or {}).get('cache_key'))
                if not model or model['artifact'].get('schema')!='light_score_v2':
                    raise ValueError('自动纸上账户需明确的合格评分模型')
                qualified=resolve_qualification(model['artifact'],mode='paper',as_of=result['end'],
                    qualification_event=result['identity'].get('qualification_event'))
                if not qualified.get('paper_allocation_eligible'):raise ValueError('自动纸上账户资格尚未通过')
        identity=result['identity']
        self.store.pin(result_id,'saved_simulation')
        self.store.pin(self.store.cache_key({'id':result_id},'result_lookup'),'saved_simulation')
        if result.get('score_model'):
            self.store.pin(result['score_model']['cache_key'],'saved_simulation:'+result_id)
        accounts=self.store.get_setting('enabled_accounts',{})
        # A branch's initial configuration/model is immutable. Subsequent daily
        # results update its pointer, never its starting cash or configuration.
        entry=accounts.get(result_id) or {'account_id':result_id,'last_result_id':result_id,
            'symbols':result['symbols'],'start':result['start'],'last_date':result['end'],
            'cash':identity['cash'],'prediction':identity['prediction'],
            'max_predictions':identity['max_predictions'],'budget_seconds':identity['budget_seconds'],
            'code':identity['code'],'score_identity':identity['score'],'model':identity['model']}
        desired='shadow_research' if shadow_research else 'paper'
        if enabled and entry.get('continuation_mode') and entry['continuation_mode']!=desired:
            raise ValueError('已保存账户的执行身份不能原地改变，请建立分支')
        entry['enabled']=bool(enabled)
        if enabled:
            entry['continuation_mode']=desired
            entry['role']=desired
        accounts[result_id]=entry
        self.store.set_setting('enabled_accounts',accounts)
        return entry

    @serialized_compute
    def continue_accounts(self, end, *, stop=None, deadline=None, progress=None):
        from my_strategy.services.light_account_continuation import continue_saved_account
        accounts=self.store.get_setting('enabled_accounts',{});out=[]
        for account_id, account in accounts.items():
            if not account.get('enabled'):continue
            self._check(stop)
            if end < account['last_date']:
                out.append({'account_id':account_id,'status':'blocked','reason':'cannot rewind account'});continue
            from my_strategy.services.light_account_compatibility import account_code_compatible
            if not account_code_compatible(account['code'],code_identity()):
                out.append({'account_id':account_id,'status':'blocked',
                            'reason':'模型或政策版本改变，需要建立新的账户分支'});continue
            try:
                previous=self.result(account['last_result_id'])
                from my_strategy.services.light_account_compatibility import execution_adapter_identity
                adapter_identity=previous.get('identity',{}).get('execution_adapter')
                if adapter_identity is not None and adapter_identity!=execution_adapter_identity():
                    raise ValueError('账户执行适配版本改变，需要建立新的账户分支')
                result=continue_saved_account(self,account,previous,end,stop=stop,deadline=deadline,progress=progress)
                self._check(stop)
                if result['result_id']!=previous['result_id']:
                    identity=result['identity'];result['result_id']=self.store.cache_key(identity,'simulations')
                    self.store.put(identity,result,namespace='simulations',pin='saved_account:'+account_id)
                    self.store.put({'id':result['result_id']},{'identity':identity},namespace='result_lookup',pin='saved_account:'+account_id)
                account.update(last_result_id=result['result_id'],
                    last_date=(result.get('scheduler') or {}).get('complete_through') or result['end'],
                    risk_through=(result.get('scheduler') or {}).get('risk_through'))
                self.store.set_setting('enabled_accounts',accounts)
                out.append({'account_id':account_id,'status':result.get('status','completed'),'result_id':result['result_id'],
                            'role':account.get('role',account.get('continuation_mode')),
                            'reused':result['result_id']==previous['result_id'],
                            'dependency_errors':result.get('dependency_errors',[])})
            except (ValueError,KeyError) as exc:
                out.append({'account_id':account_id,'status':'blocked','reason':str(exc)})
        return out

    def submit(self, kind, spec):
        with self.lock:
            if self._closed: raise RuntimeError('任务服务正在关闭')
            fingerprint = stable_hash({'kind':kind,'spec':spec})
            for job in self.jobs.values():
                if job['fingerprint']==fingerprint and job['status'] in {'queued','running','cancelling'}:
                    return deepcopy(job)
            job_id=uuid.uuid4().hex
            job={'job_id':job_id,'kind':kind,'fingerprint':fingerprint,'status':'queued',
                 'created_at':local_now().isoformat(),'spec':spec,'progress':{}}
            job, created = self.journal.create(job)
            if not created: return deepcopy(job)
            from my_strategy.storage.task_journal import JournalStop
            self.jobs[job_id]=job
            self.stops[job_id]=JournalStop(self.journal,job_id,threading.Event())
            try:
                self.queue.submit(self._run,job_id)
            except Exception as exc:
                job.update(status='failed',error=str(exc),finished_at=local_now().isoformat())
                self.journal.update(job)
                raise
            return deepcopy(job)

    def _run(self,job_id):
        job=self.jobs[job_id]
        self._publication_context.job=job
        heartbeat_done=threading.Event()
        def heartbeat():
            while not heartbeat_done.wait(5.):
                try: self.journal.heartbeat(job_id)
                except Exception as exc:
                    with self.lock: job['heartbeat_error']=str(exc)
        heartbeat_thread=threading.Thread(target=heartbeat,name='task-heartbeat',daemon=True)
        heartbeat_thread.start()
        def update(**value):
            with self.lock:
                job['progress']=value
                now=time.monotonic()
                if now-self._progress_flushed.get(job_id,0)>=1.:
                    self.journal.update(job); self._progress_flushed[job_id]=now
        try:
            self._check(self.stops[job_id]); job['status']='running'
            self.journal.update(job)
            spec=job['spec']; kind=job['kind']
            if kind=='prediction': result=self.prediction(**spec,stop=self.stops[job_id])
            elif kind=='score_training': result=self.train_score(**spec,stop=self.stops[job_id],progress=update)
            elif kind=='tuning': result=self.tune(**spec,stop=self.stops[job_id])
            elif kind=='simulation': result=self.backtest(**spec,stop=self.stops[job_id],progress=update)
            elif kind=='score_observations': result=self.collect_score_observations(**spec,stop=self.stops[job_id],progress=update)
            elif kind=='incremental':
                from my_strategy.services.incremental_sync import execute_plan
                with self.compute_lock.acquire(timeout=0):
                    result=execute_plan(store=self.store,raw_db_path=self.raw_db,**spec,
                        stop=self.stops[job_id],progress=lambda value:update(**value))
            elif kind=='daily':
                from my_strategy.services.light_daily import sync_light_daily
                result=sync_light_daily(**spec,service=self,check_cancel=self.stops[job_id].is_set,
                    progress=lambda value:update(**value))
            else: raise ValueError('未知轻量任务')
            # Bind committed business results before publishing the terminal receipt.
            if result.get('result_id'):
                job['result_id']=result['result_id']
                job['result_ref']={'type':'simulation','result_id':result['result_id']}
            else:
                reference=next((result[key] for key in ('result_ref','model_ref','manifest_ref')
                    if isinstance(result.get(key),dict) and result[key].get('cache_key')),None)
                if reference is None:
                    reference=self.store.put({'job_id':job_id},result,namespace='job_results',ttl_seconds=30*86400)
                job['result_ref']={'type':'cas',**reference}
            self.journal.update(job)
            self._check(self.stops[job_id])
            reported=result.get('status')
            outcome=reported if reported in {'partial','cancelled','failed','blocked'} else 'succeeded'
            job.update(status=outcome,result=result)
        except InterruptedError as exc: job.update(status='cancelled',error=str(exc))
        except Exception as exc: job.update(status='failed',error=str(exc))
        finally:
            self._publication_context.job=None
            heartbeat_done.set(); heartbeat_thread.join(timeout=1.)
            from my_strategy.services.kronos_prediction import release_prediction_resources
            job['finished_at']=local_now().isoformat()
            self.journal.update(job)
            release_prediction_resources()
            if job['kind']=='incremental' and job['status']=='cancelled':
                key='incremental_plan:'+job['spec']['plan_id']
                current=self.store.get_setting(key,{})
                self.store.set_setting(key,{**current,'status':'cancelled','job_id':job_id})
            # The journal is the task record; business results remain in their verified stores.
            # Avoid emitting a new CAS object for every task-state transition.
            # Bound in-memory terminal history just like the visible task list.
            terminal=[key for key,value in self.jobs.items() if value['status'] in {'succeeded','failed','cancelled','partial','blocked'}]
            for key in terminal[:-30]:
                self.jobs.pop(key,None);self.stops.pop(key,None)

    def get_job(self,job_id):
        with self.lock:
            if job_id in self.jobs: return deepcopy(self.jobs[job_id])
        saved=(self.journal.get(job_id) if hasattr(self,'journal') else None) or self.store.get({'job_id':job_id},namespace='jobs')
        if not saved: raise KeyError(job_id)
        if hasattr(self,'journal') and saved['status'] in {'interrupted','failed','cancelled'} and not saved.get('result_ref'):
            saved['recovered_outputs']=self.journal.recovered_outputs(job_id,self.store)
        if saved.get('result_id'): saved['result']=self.result(saved['result_id'])
        elif (saved.get('result_ref') or {}).get('type')=='cas':
            saved['result']=self.store.get_by_key(saved['result_ref']['cache_key'])
            if saved['result'] is None: saved['result_unavailable']='结果已过期或不可用；任务记录保留'
            else: saved['result']={**saved['result'],**saved.get('result_metadata',{})}
        return saved

    def cancel(self,job_id):
        with self.lock:
            saved=self.journal.cancel(job_id)
            if job_id in self.jobs:
                self.jobs[job_id].update(saved)
                if saved['status']=='cancelling': self.stops[job_id].set()
                return deepcopy(self.jobs[job_id])
            return saved
