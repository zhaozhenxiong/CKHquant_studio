"""Bounded Torch MLP research score with a same-input L2 comparator.

Pure service: compact JSON weights, no pickle, current-model writes, automatic
promotion, Kronos inference, or full-market work. A failed/insufficient B dataset
returns an explicit pending report; successful fitting is not policy acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from my_strategy.services.light_qualification import content_hash, prediction_checks, resolve_qualification, verify_model
from my_strategy.services.light_score_training import BASE_FEATURES,EXTRA_FEATURES,KRONOS_FEATURES,KRONOS_STATES
from my_strategy.services.simple_score import _dates,_metrics,_probabilities

SCHEMA='light_score_v2'
SPLITS=('train','tune','calibration','policy','test')
CLASSES=('down','flat','up')


def _check(cancel,deadline):
    if cancel and bool(cancel() if callable(cancel) else cancel.is_set()): raise InterruptedError('Light MLP training cancelled')
    if time.monotonic()>=deadline: raise TimeoutError('Light MLP training budget exhausted; model not published')


def _device(requested):
    import torch
    if requested=='auto':
        requested='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested not in {'cpu','mps','cuda'}: raise ValueError('Unsupported training device')
    if requested=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable')
    if requested=='mps' and not torch.backends.mps.is_available(): raise ValueError('MPS is unavailable')
    return torch.device(requested)


def _network(inputs,widths,dropout):
    import torch
    layers=[]
    last=inputs
    for width in widths:
        layers.extend([torch.nn.Linear(last,width),torch.nn.GELU(),torch.nn.Dropout(dropout)])
        last=width
    layers.append(torch.nn.Linear(last,3))
    return torch.nn.Sequential(*layers)


def _bounds(frame,split_ranges,gap_sessions):
    if set(split_ranges)!=set(SPLITS): raise ValueError('Require five chronological splits')
    if isinstance(gap_sessions,bool) or not isinstance(gap_sessions,int) or gap_sessions<0: raise ValueError('Invalid split gap')
    bounds={};previous=None
    for name in SPLITS:
        start,end=_dates(split_ranges[name])
        if len(str(split_ranges[name][1]))==10: end+=pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)
        if start>end or (previous is not None and start<=previous): raise ValueError('Overlapping or unordered time splits')
        bounds[name]=(start,end);previous=end
    decisions=_dates(frame.as_of,daily_close=True)
    mature=_dates(frame.label_mature_at,daily_close=True)
    if (mature<=decisions).any(): raise ValueError('Labels must mature after decisions')
    if 'feature_available_at' in frame and (_dates(frame.feature_available_at,daily_close=True)>decisions).any():
        raise ValueError('Features became available after decisions')
    masks={};population={}
    for i,name in enumerate(SPLITS):
        start,end=bounds[name]
        assigned=(decisions>=start)&(decisions<=end)
        mature_here=assigned&(mature<=end)
        gap=np.zeros(len(frame),dtype=bool)
        if i and gap_sessions:
            first_days=sorted(set(decisions[assigned].normalize()))[:gap_sessions]
            gap=np.asarray(decisions.normalize().isin(first_days))
        used=mature_here&~gap
        masks[name]=used
        population[name]=dict(assigned_rows=int(assigned.sum()),mature_rows=int(used.sum()),
            excluded_immature_rows=int((assigned&~mature_here).sum()),gap_excluded_rows=int((mature_here&gap).sum()),
            class_counts={c:int(((frame.label.to_numpy()==j)&used).sum()) for j,c in enumerate(CLASSES)})
    return bounds,masks,population,decisions,mature


def _preprocess_fit(frame,features,optional,mask_columns,train,variant):
    values=frame[features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float,copy=True)
    optional=set(optional)
    for j,column in enumerate(features):
        if column not in optional and not np.isfinite(values[:,j]).all():
            raise ValueError('Missing/nonfinite core feature: '+column)
    values[~np.isfinite(values)]=np.nan
    medians=[];low=[];high=[]
    for j in range(len(features)):
        known=values[train,j][np.isfinite(values[train,j])]
        medians.append(float(np.median(known)) if len(known) else 0.)
        low.append(float(np.quantile(known,.01)) if len(known) else 0.)
        high.append(float(np.quantile(known,.99)) if len(known) else 0.)
    filled=np.where(np.isfinite(values),values,np.array(medians))
    filled=np.clip(filled,low,high)
    mean=filled[train].mean(axis=0);scale=filled[train].std(axis=0);scale[scale<1e-12]=1
    contract=dict(features=features,optional_columns=sorted(optional),mask_columns=list(mask_columns),
        medians=medians,clip_low=low,clip_high=high,mean=mean.tolist(),scale=scale.tolist(),fitted_on='train_only',
        kronos_states=list(KRONOS_STATES) if variant=='B' else [])
    return contract,_preprocess_apply(frame,contract)


def _preprocess_apply(frame,contract):
    features=contract['features'];data=frame.copy()
    optional=set(contract['optional_columns'])
    for column in features:
        if column not in data:
            if column not in optional: raise ValueError('Missing core score feature: '+column)
            data[column]=np.nan
    raw=data[features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    for j,column in enumerate(features):
        if column not in optional and not np.isfinite(raw[:,j]).all(): raise ValueError('Nonfinite core score feature: '+column)
    missing=~np.isfinite(raw)
    values=np.clip(np.where(missing,np.asarray(contract['medians']),raw),contract['clip_low'],contract['clip_high'])
    result=[(values-np.asarray(contract['mean']))/np.asarray(contract['scale'])]
    if contract['mask_columns']:
        result.append(missing[:,[features.index(k) for k in contract['mask_columns']]].astype(float))
    if contract['kronos_states']:
        if 'kronos_state' not in data or not data.kronos_state.isin(contract['kronos_states']).all():
            raise ValueError('Missing or unknown Kronos observation state')
        result.append(np.stack([(data.kronos_state==name).to_numpy(dtype=float) for name in contract['kronos_states']],axis=1))
    return np.concatenate(result,axis=1).astype(np.float32)


def _b_coverage(frame,masks,population,manifest):
    failures=[]
    if not isinstance(manifest,dict): return dict(status='pending',reasons=['frozen_observation_manifest_missing'])
    if manifest.get('model_independent') is not True or not manifest.get('source_policy_id') or not manifest.get('frozen_at'):
        failures.append('observation_population_not_frozen_independently')
    rows=manifest.get('observations') or []
    keys=[(str(row.get('symbol')),str(row.get('as_of',row.get('date')))[:10]) for row in rows]
    data_keys=set(zip(frame.symbol.astype(str),frame.as_of.astype(str).str[:10]))
    if len(keys)!=len(set(keys)) or not data_keys<=set(keys): failures.append('observation_population_identity_mismatch')
    allowed_exclusions={'immature_label','calendar_gap','core_feature_unavailable','outside_frozen_window'}
    if any(row.get('training_exclusion_reason') and row['training_exclusion_reason'] not in allowed_exclusions for row in rows):
        failures.append('unrecognized_population_exclusion')
    required_keys={key for key,row in zip(keys,rows) if not row.get('training_exclusion_reason')}
    if required_keys!=data_keys: failures.append('frozen_training_population_not_complete')
    states={'succeeded','unusable','failed','deferred'}
    if not rows or any(row.get('status') not in states for row in rows): failures.append('observation_outcomes_incomplete')
    valid=sum(row.get('status')=='succeeded' for row in rows)
    coverage=valid/len(rows) if rows else 0
    if coverage<.8: failures.append('overall_valid_prediction_coverage_below_80_percent')
    calls=manifest.get('total_calls')
    attempted=sum(row.get('status')!='deferred' for row in rows)
    if not isinstance(calls,int) or isinstance(calls,bool) or calls<attempted or calls>2000:
        failures.append('historical_prediction_budget_not_reconciled')
    outcome_by_key=dict(zip(keys,[row.get('status') for row in rows]))
    if any((outcome_by_key.get(key)=='succeeded')!=(state=='valid') for key,state in zip(zip(frame.symbol.astype(str),frame.as_of.astype(str).str[:10]),frame.kronos_state)):
        failures.append('forecast_state_does_not_match_manifest')
    observation_by_key=dict(zip(keys,rows))
    for _,row in frame.iterrows():
        if row.kronos_state!='valid': continue
        key=(str(row.symbol),str(row.as_of)[:10])
        forecast=(observation_by_key.get(key) or {}).get('forecast') or {}
        summary=forecast.get('summary') or {};quality=forecast.get('quality') or {}
        observed_input=(forecast.get('identity') or {}).get('input') or {}
        if (quality.get('prediction_usable') is not True or observed_input.get('symbol')!=key[0]
                or str(observed_input.get('as_of'))[:10]!=key[1] or not observed_input.get('data_sha256')):
            failures.append('frozen_forecast_identity_missing_or_mismatched');break
        expected=[summary.get('terminal_return_median'),summary.get('terminal_return_p10'),summary.get('terminal_return_p90'),
                  None,summary.get('up_path_fraction'),quality.get('valid_fraction')]
        if all(isinstance(expected[i],(int,float)) and np.isfinite(expected[i]) for i in (0,1,2,4,5)):
            expected[3]=expected[2]-expected[1]
        else:
            failures.append('valid_forecast_summary_missing');break
        actual=row[KRONOS_FEATURES].to_numpy(dtype=float)
        if (not np.isfinite(actual).all() or not np.allclose(actual,expected,rtol=1e-10,atol=1e-12)
                or not expected[1]<=expected[0]<=expected[2] or not .8<=expected[5]<=1):
            failures.append('forecast_feature_values_do_not_match_manifest');break
    if frame.symbol.nunique()<5: failures.append('fewer_than_five_stocks')
    segments={}
    for name,mask in masks.items():
        usable=mask&(frame.kronos_state.to_numpy()=='valid')
        classes={c:int(((frame.label.to_numpy()==j)&usable).sum()) for j,c in enumerate(CLASSES)}
        count=int(usable.sum());segments[name]=dict(valid_mature_rows=count,class_counts=classes,all_mature_rows=int(mask.sum()))
        if count<(500 if name=='train' else 100): failures.append(name+'_valid_mature_rows_insufficient')
        if min(classes.values())<10: failures.append(name+'_valid_class_support_insufficient')
    return dict(status='pending' if failures else 'complete',reasons=failures,manifest_sha256=content_hash(manifest),
        requested=len(rows),valid=valid,valid_fraction=coverage,total_calls=calls,segments=segments)


def _registry(existing,trials,variant,identity):
    from my_strategy.services.light_experiment_registry import merge_experiment_registry
    entries=[]
    for trial in trials:
        item=dict(kind='mlp',variant=variant,configuration=trial,input_identity=identity)
        item['experiment_id']=content_hash(item)
        entries.append(item)
    return merge_experiment_registry(existing,entries)


def fit_light_score(frame,split_ranges,*,variant='A',source_identity,label_contract,
                    feature_columns=None,optional_columns=None,trials=None,budget_seconds=600,
                    device='cpu',cancel_check=None,observation_manifest=None,experiment_registry=None,
                    min_segment_rows=30,gap_sessions=0,cpu_threads=2,progress=None,registration_callback=None):
    """Fit/choose on train+tune, calibrate independently, then report policy/test.

    Feature values must already be point-in-time observations. Targets cannot be
    inputs. Defaults use two small candidates, with at most six across a supplied
    shared registry. Exhaustion/cancellation raises without publishing a model.
    """
    if variant not in {'A','B'}: raise ValueError('Unknown light-score variant')
    if not source_identity or not label_contract: raise ValueError('Frozen source and label contracts are required')
    if not 0<budget_seconds<=600 or not 1<=cpu_threads<=2: raise ValueError('Training budget/thread limit exceeded')
    if len(frame)>100000 or len(frame)==0 or min_segment_rows<10: raise ValueError('Light score population limits exceeded')
    start=time.monotonic();deadline=start+budget_seconds
    _check(cancel_check,deadline)
    data=frame.copy().reset_index(drop=True)
    required={'symbol','as_of','label','label_mature_at'}
    if not required<=set(data): raise ValueError('Missing label/temporal population fields')
    if data.duplicated(['symbol','as_of']).any(): raise ValueError('Duplicate stock/date training rows')
    if not data.label.isin([0,1,2]).all(): raise ValueError('Labels must be 0/1/2')
    features=list(feature_columns or BASE_FEATURES+EXTRA_FEATURES+(KRONOS_FEATURES if variant=='B' else []))
    forbidden={'symbol','as_of','date','label','label_mature_at','feature_available_at','future_return','realized_pnl'}
    if len(set(features))!=len(features) or not features or any(key in forbidden for key in features): raise ValueError('Invalid feature columns')
    optional=list(optional_columns if optional_columns is not None else [k for k in EXTRA_FEATURES+KRONOS_FEATURES if k in features])
    if not set(optional)<=set(features): raise ValueError('Optional feature not in schema')
    masks_features=[k for k in optional if k not in KRONOS_FEATURES]
    dimensions=len(features)+len(masks_features)+(4 if variant=='B' else 0)
    if dimensions>48: raise ValueError('Light MLP input dimension limit is 48 including masks')
    for key in features:
        if key not in data:
            if key not in optional: raise ValueError('Missing core feature: '+key)
            data[key]=np.nan
    if variant=='B' and ('kronos_state' not in data or not data.kronos_state.isin(KRONOS_STATES).all()):
        return dict(schema=SCHEMA,status='pending',variant=variant,reason='kronos_observation_states_missing',model_created=False)
    bounds,masks,population,decisions,maturity=_bounds(data,split_ranges,gap_sessions)
    for name in SPLITS:
        if population[name]['mature_rows']<(max(60,min_segment_rows) if name=='train' else min_segment_rows):
            raise ValueError(name+': insufficient mature rows')
    if min(population['train']['class_counts'].values())<5: raise ValueError('Training requires five examples of every class')
    coverage=_b_coverage(data,masks,population,observation_manifest) if variant=='B' else None
    if coverage and coverage['status']=='pending':
        return dict(schema=SCHEMA,status='pending',variant=variant,reason='fusion_data_incomplete',coverage=coverage,population=population,model_created=False)
    defaults=dict(widths=[32,16],learning_rate=.001,weight_decay=.01,dropout=.1,max_epochs=40,patience=6,batch_size=256,seed=42)
    specs=[{**defaults,**spec} for spec in (trials or [{'widths':[32,16]},{'widths':[16,8]}])]
    if not 1<=len(specs)<=6 or len({tuple(s['widths']) for s in specs})>2: raise ValueError('At most six trials and two widths allowed')
    for s in specs:
        if len(s['widths'])!=2 or any(not isinstance(w,int) or isinstance(w,bool) or not 1<=w<=64 for w in s['widths']): raise ValueError('Invalid network widths')
        if not 1<=s['max_epochs']<=50 or not 1<=s['patience']<=6 or not 1<=s['batch_size']<=256: raise ValueError('Epoch/batch/early-stop budget exceeded')
        if not 0<=s['dropout']<=.5 or not 0<s['learning_rate']<=.01 or not 0<=s['weight_decay']<=1: raise ValueError('Invalid training hyperparameter')
    canonical=data[[*features,'symbol','as_of','label','label_mature_at']].replace({np.nan:None}).to_dict('records')
    identity=dict(variant=variant,source=source_identity,label_contract=label_contract,split_ranges=split_ranges,
        features=features,optional_columns=optional,gap_sessions=gap_sessions,data_sha256=content_hash(canonical),
        observation_manifest_sha256=content_hash(observation_manifest) if observation_manifest else None,
        implementation_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('light_score_v2.py','light_score_training.py','light_qualification.py','light_experiment_registry.py')})
    registry=_registry(experiment_registry,specs,variant,identity)
    if registration_callback:
        committed=registration_callback(deepcopy(registry))
        if committed is not None:
            from my_strategy.services.light_experiment_registry import merge_experiment_registry
            registry=merge_experiment_registry(committed,registry['entries'])
    preprocessing,x=_preprocess_fit(data,features,optional,masks_features,masks['train'],variant)
    labels=data.label.to_numpy(dtype=np.int64)
    counts=np.bincount(labels[masks['train']],minlength=3);prior=(counts+.5)/(counts.sum()+1.5)
    import torch
    target=_device(device)
    old_threads=torch.get_num_threads()
    results=[];chosen=None
    try:
        torch.set_num_threads(cpu_threads)
        for number,spec in enumerate(specs):
            _check(cancel_check,deadline)
            # Isolate CPU/CUDA RNG state from other consumers in this process.
            devices=list(range(torch.cuda.device_count())) if target.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(spec['seed']))
                if target.type=='mps': torch.mps.manual_seed(int(spec['seed']))
                net=_network(dimensions,spec['widths'],spec['dropout']).to(target)
                optimizer=torch.optim.AdamW(net.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
                tx=torch.as_tensor(x[masks['train']],dtype=torch.float32,device=target)
                ty=torch.as_tensor(labels[masks['train']],dtype=torch.long,device=target)
                vx=torch.as_tensor(x[masks['tune']],dtype=torch.float32,device=target)
                vy=torch.as_tensor(labels[masks['tune']],dtype=torch.long,device=target)
                generator=np.random.default_rng(int(spec['seed']))
                best_loss=float('inf');best_state=None;best_epoch=0;stale=0;epochs=[]
                for epoch in range(spec['max_epochs']):
                    _check(cancel_check,deadline);net.train();total=0.
                    permutation=generator.permutation(len(tx))
                    for offset in range(0,len(tx),spec['batch_size']):
                        _check(cancel_check,deadline)
                        index=torch.as_tensor(permutation[offset:offset+spec['batch_size']],dtype=torch.long,device=target)
                        optimizer.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.cross_entropy(net(tx[index]),ty[index])
                        if not torch.isfinite(loss): raise ValueError('Nonfinite MLP training loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);optimizer.step()
                        total+=float(loss.detach().cpu())*len(index)
                    net.eval()
                    with torch.no_grad(): tune_loss=float(torch.nn.functional.cross_entropy(net(vx),vy).cpu())
                    epochs.append(dict(epoch=epoch+1,train_loss=total/len(tx),tune_log_loss=tune_loss))
                    if progress: progress(stage='mlp_training',candidate=number+1,candidates=len(specs),epoch=epoch+1,epochs=spec['max_epochs'])
                    if tune_loss<best_loss-1e-6:
                        best_loss=tune_loss;best_epoch=epoch+1;stale=0
                        best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                    else: stale+=1
                    if stale>=spec['patience']: break
                if best_state is None: raise ValueError('No valid MLP candidate')
                results.append(dict(configuration=spec,best_epoch=best_epoch,tune_log_loss=best_loss,epochs=epochs))
                if chosen is None or best_loss<chosen['loss']: chosen=dict(loss=best_loss,state=best_state,configuration=spec,best_epoch=best_epoch)
        net=_network(dimensions,chosen['configuration']['widths'],chosen['configuration']['dropout'])
        net.load_state_dict(chosen['state']);net.eval()
        with torch.no_grad(): logits=net(torch.as_tensor(x)).numpy().astype(float)
    finally:
        torch.set_num_threads(old_threads)
    _check(cancel_check,deadline)
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    def temperature_for(values):
        def objective(log_t):
            _check(cancel_check,deadline)
            return _metrics(labels[masks['calibration']],_probabilities(values[masks['calibration']],np.exp(log_t)))['log_loss']
        result=minimize_scalar(objective,bounds=(-2.302585093,2.302585093),method='bounded')
        if not result.success or not np.isfinite(result.fun): raise ValueError('Independent calibration failed')
        return float(np.exp(result.x))
    temperature=temperature_for(logits)
    linear_trials=[];linear_best=None
    for c in (.1,1.,10.):
        _check(cancel_check,deadline)
        linear=LogisticRegression(C=c,solver='lbfgs',max_iter=20,random_state=42,warm_start=True)
        converged=False
        # Bounded warm-start chunks leave cancellation/budget checks between
        # optimizer calls; numerical BLAS threads obey the same CPU budget.
        with threadpool_limits(limits=cpu_threads):
            for chunk in range(15):
                _check(cancel_check,deadline)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    linear.fit(x[masks['train']],labels[masks['train']])
                converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                if converged: break
        metric=_metrics(labels[masks['tune']],linear.predict_proba(x[masks['tune']]))
        linear_trials.append(dict(C=c,converged=converged,optimizer_chunks=chunk+1,tune=metric))
        if converged and (linear_best is None or metric['log_loss']<linear_best[0]): linear_best=(metric['log_loss'],linear)
    if linear_best is None: raise ValueError('Same-input L2 comparator did not converge')
    linear=linear_best[1];linear_logits=x@linear.coef_.T+linear.intercept_
    linear_temperature=temperature_for(linear_logits)
    metrics={};linear_metrics={};patterns={}
    for name,mask in masks.items():
        metrics[name]=_metrics(labels[mask],_probabilities(logits[mask],temperature))
        metrics[name]['frozen_train_prior_baseline']=_metrics(labels[mask],np.tile(prior,(int(mask.sum()),1)))
        linear_metrics[name]=_metrics(labels[mask],_probabilities(linear_logits[mask],linear_temperature))
        if variant=='B':
            patterns[name]={}
            for state in KRONOS_STATES:
                selected=mask&(data.kronos_state.to_numpy()==state)
                patterns[name][state]=_metrics(labels[selected],_probabilities(logits[selected],temperature)) if selected.any() else dict(rows=0,class_counts={c:0 for c in CLASSES})
                if selected.any():
                    patterns[name][state]['frozen_train_prior_baseline']=_metrics(labels[selected],np.tile(prior,(int(selected.sum()),1)))
                    patterns[name][state]['l2_baseline']=_metrics(labels[selected],_probabilities(linear_logits[selected],linear_temperature))
    supported=[]
    if variant=='B':
        supported=[state for state in KRONOS_STATES if all(patterns[s][state]['rows']>=10 and min(patterns[s][state]['class_counts'].values())>=3 for s in ('train','calibration'))]
    _check(cancel_check,deadline)
    artifact=dict(schema=SCHEMA,status='completed',model_type='torch_mlp',variant=variant,features=features,classes=list(CLASSES),
        identity=identity,architecture=dict(input_dimensions=dimensions,widths=chosen['configuration']['widths'],dropout=chosen['configuration']['dropout'],activation='GELU'),
        state_dict={key:value.tolist() for key,value in chosen['state'].items()},preprocessing=preprocessing,
        temperature=temperature,calibration_kind='independent_positive_temperature',train_prior=prior.tolist(),
        trials=results,selected_configuration=chosen['configuration'],selected_epoch=chosen['best_epoch'],
        metrics=metrics,population=population,observation_coverage=coverage,missing_pattern_metrics=patterns,supported_missing_patterns=supported,
        l2_baseline=dict(model_type='multinomial_logistic_L2',coefficients=linear.coef_.tolist(),intercept=linear.intercept_.tolist(),
            temperature=linear_temperature,C=float(linear.C),trials=linear_trials,metrics=linear_metrics),
        available_after=bounds['calibration'][1].isoformat(),evaluation_end=bounds['test'][1].isoformat(),
        research_admission=dict(temporal_contract_passed=True,mode='historical_candidate',labels_mature_inside_split=True,
            preprocessing_train_only=True,selection_tune_only=True,calibration_independent=True),
        qualification=dict(paper_allocation_eligible=False,independent_forward_status='pending',reason='requires_separate_broker_policy_evidence'),
        experiment_registry=registry,created_at=datetime.now(timezone.utc).isoformat(),
        runtime=dict(device=str(target),inference_verification_device='cpu',torch_version=str(torch.__version__),
            duration_seconds=time.monotonic()-start,cpu_threads=cpu_threads,budget_seconds=budget_seconds,
            candidate_count=len(specs),parameter_count=sum(v.numel() for v in chosen['state'].values()),weights_format='json_arrays_no_pickle'))
    artifact['prediction_checks']=prediction_checks(artifact)
    artifact['model_version']=content_hash(artifact)
    return artifact


def score_light_samples(artifact,frame,*,as_of,historical_reconstruction=False,mode='historical_candidate',qualification_event=None,model_kind='mlp',decision_time=None):
    verify_model(artifact)
    # as_of identifies the completed market bar; decision_time preserves when an
    # after-close B forecast could actually be used. Neither rewrites the frozen
    # artifact nor claims it was produced by this scorer's current source SHA.
    bar_close=_dates([as_of],daily_close=True)[0]
    requested=_dates([decision_time])[0] if decision_time is not None else bar_close
    if requested<bar_close:raise ValueError('Decision time precedes the completed as_of bar')
    if requested<=_dates([artifact['available_after']])[0]: raise ValueError('Model development/calibration evidence unavailable at decision')
    actual_frozen=requested>=_dates([artifact['created_at']])[0]
    if not actual_frozen and not historical_reconstruction: raise ValueError('Historical application requires explicit reconstruction')
    if 'feature_available_at' in frame and (_dates(frame.feature_available_at,daily_close=True)>requested).any():
        raise ValueError('Features unavailable at decision')
    if 'as_of' in frame and (_dates(frame.as_of,daily_close=True)>requested).any(): raise ValueError('Future observations cannot be scored')
    x=_preprocess_apply(frame,artifact['preprocessing'])
    if model_kind not in {'mlp','l2'}: raise ValueError('Unknown comparison model')
    if model_kind=='l2':
        if mode!='historical_candidate': raise ValueError('L2 comparator has no independent paper-policy qualification')
        linear=artifact['l2_baseline'];logits=x@np.asarray(linear['coefficients']).T+np.asarray(linear['intercept'])
        p=_probabilities(logits,linear['temperature'])
    else:
        import torch
        architecture=artifact['architecture']
        with torch.random.fork_rng(devices=[]):
            net=_network(architecture['input_dimensions'],architecture['widths'],architecture['dropout'])
            weights={key:torch.as_tensor(value,dtype=torch.float32) for key,value in artifact['state_dict'].items()}
            net.load_state_dict(weights,strict=True);net.eval()
            with torch.no_grad(): logits=net(torch.as_tensor(x)).numpy().astype(float)
        p=_probabilities(logits,artifact['temperature'])
    qualification=resolve_qualification(artifact,mode=mode,as_of=as_of if decision_time is None else requested.isoformat(),qualification_event=qualification_event)
    row_statuses=[];scores=[];probabilities=[]
    for i in range(len(frame)):
        supported=artifact['variant']!='B' or frame.iloc[i]['kronos_state'] in artifact['supported_missing_patterns']
        row_statuses.append('scored' if supported else 'unsupported_missing_pattern')
        scores.append(float(100*(p[i,2]+.5*p[i,1])) if supported else None)
        probabilities.append(dict(zip(CLASSES,p[i].tolist())) if supported else None)
    if any(status!='scored' for status in row_statuses):
        qualification={**qualification,'allow_increase':False,'positive_allocation_eligible':False,'reason':'unsupported_missing_pattern'}
    result=dict(schema=SCHEMA,status='scored' if all(s=='scored' for s in row_statuses) else 'unavailable',
        model_version=artifact['model_version'],scorer_id=content_hash(dict(model=artifact['model_version'],head=model_kind)),
        model_type=artifact['model_type'] if model_kind=='mlp' else 'multinomial_logistic_L2',variant=artifact['variant'],as_of=str(as_of),
        features=artifact['features'],scores=scores,probabilities=probabilities,row_statuses=row_statuses,
        score_semantics='100 * (P(up) + 0.5 * P(flat))',qualification=qualification,
        historical_reconstruction=bool(historical_reconstruction),model_actually_frozen=actual_frozen,
        temporal_available=True,research_only=True)
    if decision_time is not None:result['decision_time']=requested.isoformat()
    return result


train_mlp_score=fit_light_score
score_mlp=score_light_samples
