"""Append-only, frozen-model forward evidence with continuous real accounts.

No training artifact is rewritten. Every append replays the same account from
its actual freeze and checks the complete observed input and ledger prefix.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, PROCESSED_DATA_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_high_return_research import (_source, _read_features, _file_hash,
    _save, _compact_evaluation, _json, _code_files)
from my_strategy.storage.czsc_results import ResultStore

KIND='czsc-high-return-forward-evidence'
SOURCE_KIND='czsc-high-return-forward-source'
OUTCOMES={'label','label_end','label_reason','label_available','label_available_at','net_return',
          'entry_date','exit_date','entry_shares','fees'}


def _root(run_id):
    root=(ARTIFACT_RUNS_ROOT/run_id).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()) or root==ARTIFACT_RUNS_ROOT.resolve():
        raise ValueError('invalid forward run identity')
    return root


def _model(run_id):
    root=_root(run_id); report=json.loads((root/'reports/high_return.json').read_text())
    freeze=report.get('protocol_freeze') or {}
    if report.get('model_family')!='high_return' or report.get('status')!='completed' or not freeze.get('independent_start'):
        raise ValueError('a completed independently frozen interval model is required')
    if freeze.get('freeze_hash')!=stable_hash({k:v for k,v in freeze.items() if k!='freeze_hash'}):
        raise ValueError('actual protocol freeze hash mismatch')
    if json.loads((root/'validation/protocol_freeze.json').read_text())!=freeze:
        raise ValueError('original saved protocol freeze changed')
    if freeze.get('model_bundles_hash')!=stable_hash(report['model_bundles']):
        raise ValueError('frozen model catalog changed')
    if report.get('run_id')!=run_id or freeze.get('config_hash')!=stable_hash(report['config']) or freeze.get('source_snapshot_hash')!=report['source_snapshot']['snapshot_hash'] or report['config'].get('initial_cash')!=100000:
        raise ValueError('frozen model identity/config/source or actual account capital changed')
    from my_strategy.services.czsc_research_models import signal_time
    completed=signal_time(freeze['actual_completed_at'])
    if completed>local_now() or freeze['independent_start']<=completed.date().isoformat():
        raise ValueError('independent data must start strictly after actual completed freeze')
    calendar=report['calendar']['dates']
    if freeze.get('calendar_hash')!=stable_hash(calendar): raise ValueError('frozen training calendar changed')
    extra=freeze.get('independent_calendar')
    if extra:
        path=Path(extra['path'])
        if _file_hash(path)!=extra['sha256']: raise ValueError('independent start calendar changed')
        value=json.loads(path.read_text());calendar=value['dates']
        if value.get('verified') is not True or extra['dates_hash']!=stable_hash(calendar): raise ValueError('independent calendar is not verified')
    if next((d for d in calendar if d>completed.date().isoformat()),None)!=freeze['independent_start']:
        raise ValueError('independent start must be the first verified session after actual freeze')
    from my_strategy.services.czsc_high_return_models import load_head
    for bundle in report['model_bundles']:
        for head,binding in bundle['heads'].items():
            path=(root/binding['path']).resolve()
            if not path.is_relative_to(root): raise ValueError('frozen model head outside run')
            manifest=load_head(path)
            if manifest['head']!=head or manifest['manifest_sha256']!=binding['manifest_sha256'] or signal_time(manifest['training_completed_at'])>completed:
                raise ValueError('frozen model head/time binding changed')
    if set(freeze.get('code_hashes',{}))!={str(p) for p in _code_files()}:
        raise ValueError('frozen transitive execution/feature code coverage changed')
    for path,sha in freeze.get('code_hashes',{}).items():
        if _file_hash(Path(path))!=sha: raise ValueError('frozen model execution/feature code changed')
    from my_strategy.services.czsc_high_return_forward_protocol import forward_protocol
    if freeze.get('forward_protocol')!=forward_protocol(report['config']):
        raise ValueError('forward statistical protocol was not included in the actual freeze')
    return root,report,freeze


def _prefix(frame,end):
    selected=frame.loc[frame.date<=end].drop(columns=[c for c in frame if c in OUTCOMES or c.startswith('label_')])
    return selected.reindex(sorted(selected.columns),axis=1).to_json(orient='split',index=False,double_precision=15,date_format='iso')


def _records_prefix(previous,new,config,end):
    old={r['symbol']:r for r in previous}; current={r['symbol']:r for r in new}
    if set(old)!=set(current): raise ValueError('forward source must retain the entire frozen universe')
    hashes={}
    for symbol,record in old.items():
        left=_prefix(_read_features(record,config,end),end)
        right=_prefix(_read_features(current[symbol],config,end),end)
        if left!=right: raise ValueError('forward source rewrote observed causal input prefix: '+symbol)
        hashes[symbol]=hashlib.sha256(left.encode()).hexdigest()
    return hashes


def _evaluation_bindings(evaluation):
    path=Path(evaluation['output_root'])/'evaluation.json'
    if evaluation.get('evaluation_sha256') and _file_hash(path)!=evaluation['evaluation_sha256']:
        raise ValueError('immutable continuous evaluation changed')
    value=json.loads(path.read_text())
    if len(value['windows'])!=1: raise ValueError('one continuous frozen account required')
    return next(iter(value['windows'].values()))['ledger_bindings']


def _account_prefix(previous,current,end):
    old=_evaluation_bindings(previous); new=_evaluation_bindings(current);count=0
    for arm in ('reference','full'):
        left={v['symbol']:v for v in old[arm]};right={v['symbol']:v for v in new[arm]}
        if set(left)!=set(right): raise ValueError('forward accounts changed the frozen stock pool')
        for symbol,binding in left.items():
            for kind in ('daily','ledger','rejections'):
                pa=Path(binding['path'])/(kind+'.csv');pb=Path(right[symbol]['path'])/(kind+'.csv')
                if _file_hash(pa)!=binding[kind+'_sha256'] or _file_hash(pb)!=right[symbol][kind+'_sha256']:
                    raise ValueError('forward actual account artifact hash changed')
                a=pd.read_csv(pa);b=pd.read_csv(pb)
                a=a.loc[a.date.astype(str)<=end].reset_index(drop=True)
                b=b.loc[b.date.astype(str)<=end].reset_index(drop=True)
                try: pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
                except AssertionError as exc: raise ValueError('forward '+kind+' rewrote actual account prefix: '+symbol) from exc
                count+=1
    return count


def window_trade_metrics(bindings,dates):
    """Costs in this window and whole economic round trips closed in it."""
    if not dates: return {'completed_round_trips':0,'fees':0.,'trade_count':0}
    selected=set(dates);closed=[];fees=0.;trades=0
    for binding in bindings:
        path=Path(binding['path'])/'ledger.csv'
        if _file_hash(path)!=binding['ledger_sha256']: raise ValueError('actual fill ledger changed')
        ledger=pd.read_csv(path)
        within=ledger.loc[ledger.date.astype(str).isin(selected)]
        fees+=float(within.fee.sum()) if len(within) else 0.;trades+=len(within)
        path=Path(binding['path'])/'episodes.csv'
        if _file_hash(path)!=binding['episodes_sha256']: raise ValueError('actual economic episodes changed')
        episodes=pd.read_csv(path)
        if 'exit_date' not in episodes: continue
        complete=episodes.loc[episodes.exit_date.notna()&episodes.exit_date.astype(str).isin(selected)]
        returns=pd.to_numeric(complete.net_return,errors='raise')
        if not np.isfinite(returns).all(): raise ValueError('closed economic returns must all be finite')
        closed.extend(returns.tolist())
    return {'completed_round_trips':len(closed),'fees':fees,'trade_count':trades,
            'closed_round_trip_expectancy':float(np.mean(closed)) if closed else None,
            'trade_metric_scope':'whole_economic_roundtrips_closed_in_window_and_fill_fees_in_window'}


def _registry(model_run_id):
    db=PROCESSED_DATA_ROOT/'czsc/results.db';items=[]
    if not db.is_file(): return items
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        rows=conn.execute('SELECT run_id,summary,report_path FROM czsc_runs WHERE kind=? ORDER BY created_at',(KIND,)).fetchall()
    for run_id,summary,report_path in rows:
        if json.loads(summary).get('model_run_id')!=model_run_id: continue
        indexed=(ARTIFACT_RUNS_ROOT/report_path).resolve();root=_root(run_id)
        if not indexed.is_relative_to(root): raise ValueError('forward registry path outside run')
        report=json.loads(indexed.read_text());path=root/'reports/forward.json'
        if _json(json.loads(path.read_text()))!=report: raise ValueError('forward registry/report changed')
        if report.get('run_id')!=run_id or report.get('model_run_id')!=model_run_id: raise ValueError('forward registry identity changed')
        items.append(report)
    return items


def _economic_gate(reference,full,config):
    a,b=reference['metrics'],full['metrics']
    from my_strategy.services.czsc_high_return_forward_protocol import forward_protocol
    if min(a.get('completed_round_trips',0),b.get('completed_round_trips',0))<forward_protocol(config)['min_window_round_trips']:
        return False
    values=(a.get('period_net_return'),b.get('period_net_return'),a.get('max_drawdown'),b.get('max_drawdown'),
            a.get('closed_round_trip_expectancy'),b.get('closed_round_trip_expectancy'))
    if any(v is None or not np.isfinite(v) for v in values): return False
    return bool(b['period_net_return']>a['period_net_return'] and b['max_drawdown']<=a['max_drawdown']+min(.02,config.get('drawdown_tolerance',.02))
        and b['closed_round_trip_expectancy']>a['closed_round_trip_expectancy'])


def _bound_evaluation(evaluation):
    return {**_compact_evaluation(evaluation),'evaluation_sha256':_file_hash(Path(evaluation['output_root'])/'evaluation.json')}


def _chain_tip(items,freeze):
    if not items: return None
    indexed={r['run_id']:r for r in items}
    if len(indexed)!=len(items): raise ValueError('forward append identities are not unique')
    parents={r.get('previous_run_id') for r in items}
    tips=[r for r in items if r['run_id'] not in parents]
    if len(tips)!=1: raise ValueError('forward registry contains competing or cyclic append branches')
    tip=tips[0];current=tip;seen=set()
    while current:
        identity=current['run_id']
        if identity in seen: raise ValueError('forward append chain cycle')
        seen.add(identity)
        if any(current.get(key)!=freeze.get(key) for key in ('freeze_hash','model_bundles_hash','config_hash')):
            raise ValueError('forward append changed actual frozen model identity')
        parent=current.get('previous_run_id')
        if not parent: break
        if parent not in indexed: raise ValueError('forward append parent is missing from registry')
        previous=indexed[parent];path=_root(parent)/'reports/forward.json'
        if Path(current.get('append_parent_report_path','')).resolve()!=path.resolve() or current.get('append_parent_report_sha256')!=_file_hash(path):
            raise ValueError('forward append parent report changed')
        if previous['observed_end']>=current['observed_end']: raise ValueError('forward append dates are not strictly increasing')
        current=previous
    if seen!=set(indexed): raise ValueError('forward append registry contains unlinked evidence')
    return tip


def _register_forward(report,freeze):
    """Commit a single append atomically; competing jobs cannot branch it."""
    store=ResultStore()
    path=_root(report['run_id'])/'reports/czsc.json'
    summary={key:report[key] for key in ('metrics','coverage','data_range','strategy_version','model_gate','model_run_id','category_counts') if key in report}
    with sqlite3.connect(store.db_path,timeout=30) as conn:
        conn.execute('BEGIN IMMEDIATE')
        tip=_chain_tip(_registry(report['model_run_id']),freeze)
        if (tip['run_id'] if tip else None)!=report.get('previous_run_id'):
            raise ValueError('another forward append committed first; name the new tip and replay again')
        _save(path,report)
        conn.execute('INSERT INTO czsc_runs VALUES (?, ?, ?, ?, ?, ?)',
            (report['run_id'],KIND,'',local_now().isoformat(),json.dumps(summary,ensure_ascii=False),path.relative_to(ARTIFACT_RUNS_ROOT).as_posix()))
        conn.commit()
    return report


def build_forward_evidence(model_run_id,source_training_run_id,*,end=None,previous_run_id=None,cpu_workers=4,
        progress=None,check_cancel=None,run_callback=None):
    from my_strategy.services.czsc_high_return_evaluation import evaluate_bundles
    from my_strategy.services.czsc_high_return_forward_protocol import (forward_protocol,forward_windows,
        slice_continuous_variant,paired_forward_statistics)
    model_root,model,freeze=_model(model_run_id);cfg=model['config'];check=check_cancel or (lambda:None)
    if isinstance(cpu_workers,bool) or not isinstance(cpu_workers,int) or cpu_workers<1: raise ValueError('forward workers must be positive integers')
    records,calendar,snapshot,observed_end=_source(source_training_run_id,end,[r['symbol'] for r in model['dataset_records']])
    from my_strategy.services.czsc_research_models import signal_time
    if signal_time(observed_end)>local_now(): raise ValueError('future quotes cannot become observed forward evidence')
    start=freeze['independent_start']
    if observed_end<start: raise ValueError('new verified data after actual freeze is not available; independent validation remains pending')
    from my_strategy.services.czsc_research_calendar import load_verified_calendar
    verified,_=load_verified_calendar(snapshot['calendar_run_id'],runs_root=ARTIFACT_RUNS_ROOT)
    if calendar!=verified: raise ValueError('forward calendar source evidence mismatch')
    training_calendar=model['calendar']['dates']
    if calendar[:len(training_calendar)]!=training_calendar: raise ValueError('forward calendar rewrote the verified historical prefix')
    parent=None; chain=_registry(model_run_id);tip=_chain_tip(chain,freeze)
    if chain:
        if previous_run_id!=tip['run_id']: raise ValueError('append must name the single current forward-chain tip')
    elif previous_run_id: raise ValueError('forward append parent is not registered for this model')
    if previous_run_id:
        parent=next(r for r in chain if r['run_id']==previous_run_id)
        if parent['freeze_hash']!=freeze['freeze_hash'] or parent['observed_end']>=observed_end:
            raise ValueError('forward append must retain actual freeze and increase observed end')
        parent_calendar=json.loads(Path(parent['calendar']['path']).read_text())['dates']
        if calendar[:len(parent_calendar)]!=parent_calendar: raise ValueError('forward calendar is not an append of parent')
    previous_records=parent['new_dataset_records'] if parent else model['dataset_records']
    previous_end=parent['observed_end'] if parent else model['data_end']
    prefix_hashes=_records_prefix(previous_records,records,cfg,previous_end)
    context=create_run_context(task=KIND,as_of_date=observed_end,config=cfg,data_version=snapshot['snapshot_hash'],
        stocks=[r['symbol'] for r in records],scope='full',start_date=start,end_date=observed_end)
    if run_callback: run_callback(context.run_id)
    calendar_path=context.subdir('data_snapshots')/'calendar.json'
    _save(calendar_path,{'verified':True,'dates':calendar,'source':snapshot})
    _save(context.subdir('data_snapshots')/'source.json',snapshot)
    evaluation=evaluate_bundles(records,calendar,model_root=model_root,bundles=model['model_bundles'],
        config={**cfg,'source_snapshot_hash':snapshot['snapshot_hash']},
        windows=[{'name':'continuous_forward_from_actual_freeze','start':start,'end':observed_end}],
        arms=['reference','full'],cpu_workers=cpu_workers,progress=progress,check_cancel=check,
        output_root=context.subdir('backtests','forward'))
    if any(v['cash_retained_failures'] for v in evaluation['variants'].values()):
        raise RuntimeError('forward input/account failures must be resolved; no successful append is registered')
    prefix_files=_account_prefix(parent['continuous_evaluation'],evaluation,previous_end) if parent else 0
    stress=[]
    for multiplier in (1.5,2.):
        check();value=evaluate_bundles(records,calendar,model_root=model_root,bundles=model['model_bundles'],
            config={**cfg,'source_snapshot_hash':snapshot['snapshot_hash']},
            windows=[{'name':'continuous_forward_from_actual_freeze','start':start,'end':observed_end}],
            arms=['reference','full'],cpu_workers=cpu_workers,progress=progress,check_cancel=check,
            output_root=context.subdir('backtests',f'cost_x{multiplier}'),cost_multiplier=multiplier)
        if any(v['cash_retained_failures'] for v in value['variants'].values()): raise RuntimeError('forward cost stress failures')
        if parent:
            prior=next(v for v in parent['cost_stress'] if v['cost_multiplier']==multiplier)
            prefix_files+=_account_prefix(prior,value,previous_end)
        stress.append(_bound_evaluation(value))
    bindings=_evaluation_bindings(evaluation);all_dates=[r['date'] for r in evaluation['variants']['full']['daily']]
    windows=forward_windows(calendar,start,observed_end,cfg)
    for window in windows:
        dates=window['dates'];variants={}
        for arm in ('reference','full'):
            variants[arm]=slice_continuous_variant(evaluation['variants'][arm],dates,all_dates,initial_cash=100000.)
            variants[arm]['metrics'].update(window_trade_metrics(bindings[arm],dates))
        window['variants']=variants
        window['statistics']=paired_forward_statistics(variants['reference'],variants['full'],cfg)
        window['economic_gate']=_economic_gate(variants['reference'],variants['full'],cfg)
        window['cost_stress']=[]
        for item in stress:
            stress_bindings=_evaluation_bindings(item)
            stressed={}
            for arm in ('reference','full'):
                stressed[arm]=slice_continuous_variant(item['variants'][arm],dates,all_dates,initial_cash=100000.)
                stressed[arm]['metrics'].update(window_trade_metrics(stress_bindings[arm],dates))
            window['cost_stress'].append({'cost_multiplier':item['cost_multiplier'],'variants':stressed,
                'economic_gate':_economic_gate(stressed['reference'],stressed['full'],cfg)})
    protocol=forward_protocol(cfg)
    qualified=all(w['complete'] and w['statistics']['qualified'] and w['economic_gate']
        and all(item['economic_gate'] and min(item['variants'][a]['metrics']['completed_round_trips']
            for a in ('reference','full'))>=protocol['min_window_round_trips'] for item in w['cost_stress']) for w in windows)
    if sum(w['variants']['full']['metrics']['completed_round_trips'] for w in windows)<protocol['min_total_round_trips']:
        qualified=False
    now=local_now().isoformat()
    report={'run_id':context.run_id,'model_run_id':model_run_id,'model_family':'high_return','status':'completed_shadow_evidence',
        'freeze_hash':freeze['freeze_hash'],'model_bundles_hash':freeze['model_bundles_hash'],'config_hash':freeze['config_hash'],
        'source_snapshot':snapshot,'source_snapshot_hash':snapshot['snapshot_hash'],'original_training_source_hash':model['source_snapshot']['snapshot_hash'],
        'new_dataset_records':records,'calendar':{'path':str(calendar_path),'sha256':_file_hash(calendar_path),'dates_hash':stable_hash(calendar)},
        'data_end':observed_end,'observed_end':observed_end,'independent_start':start,'independent':True,
        'created_at':context.created_at,'actual_completed_at':now,'previous_run_id':previous_run_id,
        'append_parent_report_path':str(_root(previous_run_id)/'reports/forward.json') if parent else None,
        'append_parent_report_sha256':_file_hash(_root(previous_run_id)/'reports/forward.json') if parent else None,
        'prefix_audit':{'unchanged_verified':True,'source_prefix_hashes':prefix_hashes,'prefix_through':previous_end,
            'same_calendar_prefix':True,'ledger_and_daily_prefix_verified':True,'files_checked':prefix_files,
            'method':'entire_frozen_account_replay_from_actual_freeze_then_exact_parent_prefix_check'},
        'continuous_evaluation':_bound_evaluation(evaluation),'cost_stress':stress,'forward_protocol':protocol,'forward_windows':windows,
        'technical_statistics_qualified':qualified,'model_gate':False,'publication_allowed':False,
        'formal_blocks':['historical_universe','price_basis_total_return','historical_st'],
        'qualification':'post_actual_freeze_shadow_evidence_not_a_publication_certificate'}
    _model(model_run_id)
    report=_json(report);_save(context.subdir('reports')/'forward.json',report)
    result=_register_forward(report,freeze)
    context.write_metadata({'status':'complete','model_run_id':model_run_id,'freeze_hash':freeze['freeze_hash'],'publication_allowed':False})
    return result


def forward_status(model_run_id):
    from my_strategy.services.czsc_high_return_forward_protocol import forward_windows
    root,model,freeze=_model(model_run_id);items=_registry(model_run_id)
    if items:
        latest=_chain_tip(items,freeze);plans=latest['forward_windows']
    else:
        latest=None;calendar=model['calendar']['dates']
        extra=freeze.get('independent_calendar')
        if extra:
            if _file_hash(Path(extra['path']))!=extra['sha256']: raise ValueError('independent start calendar changed')
            calendar=json.loads(Path(extra['path']).read_text())['dates']
        plans=forward_windows(calendar,freeze['independent_start'],model['data_end'],model['config'])
    return {'model_run_id':model_run_id,'freeze_hash':freeze['freeze_hash'],'status':'forward_shadow' if latest else 'pending_future_data',
        'append_registry_count':len(items),'latest_forward_run_id':latest['run_id'] if latest else None,
        'observed_end':latest['observed_end'] if latest else model['data_end'],
        'mature_technical_windows':sum(bool(w['complete']) for w in plans),
        'statistical_status':'qualified' if latest and latest['technical_statistics_qualified'] else 'pending_or_failed',
        'forward_protocol':freeze['forward_protocol'],'forward_windows':[{k:v for k,v in w.items() if k not in {'variants','dates'}} for w in plans],
        'publication_allowed':False,'formal_blocks':['historical_universe','price_basis_total_return','historical_st']}


def _source_worker(payload):
    symbol,end,calendar,directory=payload
    from my_strategy.adapters.czsc_adapter import load_bars
    from my_strategy.services.czsc_research_features import build_features
    from my_strategy.services.czsc_research import _calendar_quality
    from my_strategy.services.czsc_research_profiles import bind_feature_profile
    from my_strategy.services.czsc_wyckoff_features import prepare_wyckoff_inputs
    raw=load_bars(symbol,end=end);features=_calendar_quality(build_features(raw),raw,calendar)
    for column in ('open','high','low','close','volume','amount','source','has_trade_price'):
        features[column]=raw[column].to_numpy(copy=True)
    features['date']=pd.to_datetime(features.date).dt.strftime('%Y-%m-%d')
    features=prepare_wyckoff_inputs(bind_feature_profile(features,'ma_trend_v1'))
    if any(c in features for c in OUTCOMES): raise ValueError('forward input preparation may not create outcome labels')
    path=Path(directory)/(symbol.replace('.','_')+'.parquet');features.to_parquet(path,index=False)
    return {'symbol':symbol,'path':str(path),'sha256':_file_hash(path),'data_end':features.date.iloc[-1],
        'data_version':raw.attrs['data_version'],'rows':len(features)}


def build_forward_source_snapshot(model_run_id,*,end=None,calendar_run_id=None,cpu_workers=4,
        progress=None,check_cancel=None,run_callback=None):
    """New raw-source X without any retired fixed-horizon labels or fitting."""
    from my_strategy.adapters.czsc_adapter import latest_market_date
    from my_strategy.services.czsc_research_calendar import resolve_execution_calendar
    model_root,model,freeze=_model(model_run_id);target=latest_market_date(end)
    from my_strategy.services.czsc_research_models import signal_time
    if not target or signal_time(target)>local_now(): raise ValueError('observed raw end required')
    if isinstance(cpu_workers,bool) or not isinstance(cpu_workers,int) or cpu_workers<1: raise ValueError('source workers must be positive integers')
    dates,metadata=resolve_execution_calendar(calendar_run_id or model['source_snapshot']['calendar_run_id'],runs_root=ARTIFACT_RUNS_ROOT)
    if target not in dates: raise ValueError('raw end has no independently verified trading session')
    symbols=[r['symbol'] for r in model['dataset_records']]
    context=create_run_context(task=SOURCE_KIND,as_of_date=target,config=model['config'],data_version=target,
        stocks=symbols,scope='full',start_date=model['config']['feature_start'],end_date=target)
    if run_callback: run_callback(context.run_id)
    directory=context.subdir('dataset');payloads=[(s,target,dates,str(directory)) for s in symbols];records=[]
    check=check_cancel or (lambda:None)
    def accept(record,n):
        check();records.append(record)
        if progress: progress('forward_causal_input_snapshot',n,len(symbols),0)
    if cpu_workers>1:
        from my_strategy.services.czsc_high_return_evaluation import _worker_init
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn'),initializer=_worker_init) as pool:
            for n,record in enumerate(pool.map(_source_worker,payloads),1): accept(record,n)
    else:
        for n,payload in enumerate(payloads,1):check();accept(_source_worker(payload),n)
    records.sort(key=lambda r:r['symbol'])
    _records_prefix(model['dataset_records'],records,model['config'],model['data_end'])
    path=context.subdir('data_snapshots')/'calendar.json';_save(path,{'verified':True,'dates':dates,'source':metadata})
    report={'run_id':context.run_id,'model_run_id':model_run_id,'model_family':'high_return','data_end':target,'status':'complete',
        'dataset_records':records,'calendar':{'path':str(path),'file_sha256':_file_hash(path),
            'run_id':metadata.get('run_id') or metadata.get('calendar_run_id') or calendar_run_id or model['source_snapshot']['calendar_run_id']},
        'outcome_labels_generated':False,'training_performed':False,'same_frozen_universe':True,'historical_causal_prefix_unchanged':True,
        'coverage':{'requested':len(symbols),'successful':len(records),'failed':0},'publication_allowed':False}
    _model(model_run_id)
    report=_json(report);_save(context.subdir('reports')/'source.json',report);context.write_metadata({'status':'complete','coverage':report['coverage']})
    return ResultStore().save(SOURCE_KIND,report)
