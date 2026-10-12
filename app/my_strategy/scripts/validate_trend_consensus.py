"""Frozen trend strategy actual Broker development accounts and old control.

No result here is an independent forward acceptance window or publication.
Normal and double slippage apply the same frozen nominal-trained models;
stress results are execution robustness diagnostics, not relabeled targets.
"""
from __future__ import annotations
import argparse
import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context,stable_hash
from my_strategy.services.czsc_research import _save,_file_hash
from my_strategy.services.czsc_trend_pipeline import (read_run,read_snapshot,forward_trend,predict_entry,pinned_exit_policy,completed_round_trips)
from my_strategy.services.czsc_trend_features import original_evidence_events,event_prefix_rows
from my_strategy.services.czsc_trend_model import bind_trend_inputs,TrendSession
from my_strategy.services.czsc_trend_interval_model import HeadSession
from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
from my_strategy.services.czsc_trend_account import replay_trend_account,prepare_account_context

WINDOWS={'2025Q2':('2025-04-01','2025-06-30'),'2025Q4':('2025-10-01','2025-12-31'),
    '2026Q2':('2026-04-01','2026-06-30'),'2026Q3':('2026-07-01','2026-09-30'),
    '2026YTD':('2026-01-01','2026-10-08')}
OLD_CONTROL='20261012-121418-949260-20261009-czsc-native-v4-broker-validation-NO_GIT'


def verify_account(account,start,end,cash):
    actions={a['intent_id']:a for a in account['final_action_intents']};inventory=0;running=float(cash)
    for fill in account['ledger']:
        assert start<=fill['date']<=end and fill['intent_id'] in actions
        intent=actions[fill['intent_id']];stamp=pd.Timestamp(intent['original_signal_at'])
        assert fill['date']>=stamp.date().isoformat()
        if stamp.hour>=15:assert fill['date']>stamp.date().isoformat()
        inventory+=fill['shares'] if fill['action']=='BUY' else -fill['shares']
        assert inventory>=0
        running+=fill['cash_flow'];assert np.isclose(running,fill['cash_after'],atol=1e-6)
    assert inventory==account['terminal_state']['shares']
    assert np.isclose(running,account['terminal_state']['cash'],atol=1e-6)
    assert account['daily'][-1]['equity']==account['metrics']['final_equity']
    return {'intent_to_fill':True,'close_never_same_day_open':True,'cash_reconciliation':True,'quantity_reconciliation':True}


def _entry_evaluator(frame,calendar,prepared,root,bundles,cfg,session):
    values={r['date']:r for r in frame.to_dict('records')}
    ages=dict(zip(frame.date,event_prefix_rows(frame,original_evidence_events(prepared),calendar)))
    def evaluate(snapshot,candidate):
        day=snapshot['date'];p,identity,status=predict_entry(values[day],snapshot,candidate,candidate['entry_plan'],ages[day],root,bundles,cfg,session)
        return {'date':day,'available_at':snapshot['available_at'],'usage_mode':'historical','probability':p,
            'model_identity':identity,'status':status,'in_scope':status=='scored','reason_codes':[] if status=='scored' else [status]}
    evaluate.recipe_hash=stable_hash({'models':bundles,'capital':cfg['initial_cash'],'continuation':'E1','population':'actual_flat_admission'})
    return evaluate


def _old_prefix(symbol,window,multiplier,end,cash):
    directory=ARTIFACT_RUNS_ROOT/OLD_CONTROL/'accounts'/symbol.replace('.','_')/window/'v4_age3'/('slippage_'+str(multiplier))
    files={name:directory/(name+'.json') for name in ('ledger','daily','rejections','summary')}
    if not all(p.exists() for p in files.values()):return None
    ledger=[r for r in json.loads(files['ledger'].read_text()) if r['date']<=end]
    daily=[r for r in json.loads(files['daily'].read_text()) if r['date']<=end]
    rejections=[r for r in json.loads(files['rejections'].read_text()) if r['date']<=end]
    values=pd.Series([cash,*[r['equity'] for r in daily]])
    return {'symbol':symbol,'window':window,'variant':'old_frozen_source_union_control','slippage_multiplier':multiplier,
        'status':'completed_immutable_prior_Broker_prefix','ledger':ledger,'daily':daily,'rejections':rejections,
        'metrics':{'total_return':float(values.iloc[-1]/cash-1),'max_drawdown':float((values/values.cummax()-1).min()),
            'final_equity':float(values.iloc[-1]),'trade_count':len(ledger),'fees':sum(r['fee'] for r in ledger),
            'rejection_count':len(rejections)},'completed_round_trips':completed_round_trips(ledger),
        'source_run_id':OLD_CONTROL,'source_files':{k:{'path':str(p),'sha256':_file_hash(p)} for k,p in files.items()}}


def _worker(payload):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    record,report,calendar,windows,root,out=payload;symbol=record['symbol'];cfg=copy.deepcopy(report['config'])
    frame,pack=read_snapshot(record);frame=bind_trend_inputs(frame[frame.date<=report['data_end']].copy(),calendar)
    routes=forward_trend(frame,root,report['trend_bundles'],cfg,TrendSession(),market_dates=calendar)
    prepared=prepare_trend_decisions(frame,calendar,native_pack=pack,trend_probabilities=routes,config=cfg,mode='E0')
    cache=prepare_account_context(frame,calendar);session=HeadSession();results=[];failures=[]
    variants=[('E0_rule',None,[]),('E1_pinned_exit',True,[]),('final_linear',True,report['model_bundles'])]
    for architecture in ('tree','mlp'):
        variants.append(('final_'+architecture,True,[b for b in report['architecture_bundles'] if b['architecture']==architecture and b['heads'].get('entry')]))
    for window in windows:
        start,end=WINDOWS[window];end=min(end,report['data_end'])
        for multiplier in (1,2):
            old=_old_prefix(symbol,window,multiplier,end,cfg['initial_cash'])
            if old:results.append(old)
            else:
                failures.append({'symbol':symbol,'window':window,'variant':'old_control','error':'immutable paired baseline unavailable'})
                results.append({'symbol':symbol,'window':window,'variant':'old_frozen_source_union_control','slippage_multiplier':multiplier,
                    'status':'failed_cash_retained','ledger':[],'rejections':[],
                    'daily':[{'date':d,'equity':cfg['initial_cash']} for d in calendar if start<=d<=end],
                    'metrics':{'total_return':0.,'max_drawdown':0.,'final_equity':cfg['initial_cash'],'trade_count':0,'fees':0.,'rejection_count':0},
                    'completed_round_trips':0})
            for variant,learned,bundles in variants:
                execution=copy.deepcopy(cfg);execution['account']['slippage']*=multiplier
                mode='E0' if variant=='E0_rule' else 'E1' if variant=='E1_pinned_exit' else 'final'
                callback=pinned_exit_policy(frame,calendar,prepared,root,report['ma_exit_bundles'],cfg,session) if learned else None
                quality=_entry_evaluator(frame,calendar,prepared,root,bundles,cfg,session) if mode=='final' else None
                name=f'{symbol}:{window}:{variant}:slip{multiplier}'
                directory=Path(out)/'accounts'/symbol.replace('.','_')/window/variant/('slippage_'+str(multiplier))
                try:
                    account=replay_trend_account(frame,calendar,prepared,config=execution,exit_policy=callback,
                        entry_evaluator=quality,start=start,end=end,mode=mode,run_id=name,account_id=name,account_context=cache)
                    checked=verify_account(account,start,end,cfg['initial_cash'])
                    row={'symbol':symbol,'window':window,'variant':variant,'slippage_multiplier':multiplier,'status':'completed',
                        'metrics':account['metrics'],'completed_round_trips':completed_round_trips(account['ledger']),
                        'checks':checked,'ledger':account['ledger'],'daily':account['daily'],'rejections':account['rejections'],
                        'open_position':account['actual_position'],'nominal_model_target_preserved':True,
                        'execution_stress':multiplier==2,'model_labels_retrained_for_stress':False}
                    for key in ('ledger','daily','rejections','actual_position','entry_diagnostics','exit_policy_diagnostics','final_action_intents','holding_snapshots'):
                        _save(directory/(key+'.json'),account[key])
                    _save(directory/'summary.json',{k:v for k,v in row.items() if k not in {'ledger','daily','rejections'}})
                except Exception as exc:
                    failures.append({'symbol':symbol,'window':window,'variant':variant,'slippage_multiplier':multiplier,'error':repr(exc)})
                    dates=[d for d in calendar if start<=d<=end]
                    row={'symbol':symbol,'window':window,'variant':variant,'slippage_multiplier':multiplier,'status':'failed_cash_retained',
                        'metrics':{'total_return':0.,'max_drawdown':0.,'final_equity':cfg['initial_cash'],'trade_count':0,'fees':0.,'rejection_count':0},
                        'completed_round_trips':0,'ledger':[],'daily':[{'date':d,'equity':cfg['initial_cash']} for d in dates],
                        'rejections':[],'failure':repr(exc)}
                    _save(directory/'summary.json',row)
                results.append(row)
        print('VALIDATED '+symbol+' '+window,flush=True)
    _save(Path(out)/'accounts'/symbol.replace('.','_')/'stock_summary.json',{'symbol':symbol,'failures':failures,
        'accounts':[{k:v for k,v in r.items() if k not in {'ledger','daily','rejections'}} for r in results]})
    return results,failures


def _worker_safe(payload):
    """Keep every requested account when stock-level preparation is unusable."""
    try:
        return _worker(payload)
    except Exception as exc:
        record,report,calendar,windows,root,out=payload
        symbol=record['symbol'];cash=report['config']['initial_cash'];results=[];failures=[]
        for window in windows:
            start,end=WINDOWS[window];end=min(end,report['data_end'])
            for multiplier in (1,2):
                for variant in ('old_frozen_source_union_control','E0_rule','E1_pinned_exit','final_linear','final_tree','final_mlp'):
                    old=None
                    if variant=='old_frozen_source_union_control':
                        try:old=_old_prefix(symbol,window,multiplier,end,cash)
                        except Exception:pass
                    if old:
                        results.append(old);continue
                    error={'symbol':symbol,'window':window,'variant':variant,'slippage_multiplier':multiplier,
                        'stage':'stock_preparation','error':repr(exc)}
                    failures.append(error)
                    row={'symbol':symbol,'window':window,'variant':variant,'slippage_multiplier':multiplier,
                        'status':'failed_cash_retained','metrics':{'total_return':0.,'max_drawdown':0.,'final_equity':cash,
                            'trade_count':0,'fees':0.,'rejection_count':0},'completed_round_trips':0,'ledger':[],
                        'daily':[{'date':d,'equity':cash} for d in calendar if start<=d<=end],
                        'rejections':[],'failure':repr(exc)}
                    directory=Path(out)/'accounts'/symbol.replace('.','_')/window/variant/('slippage_'+str(multiplier))
                    for key in ('ledger','daily','rejections'):_save(directory/(key+'.json'),row[key])
                    _save(directory/'summary.json',{k:v for k,v in row.items() if k not in {'ledger','daily','rejections'}})
                    results.append(row)
        _save(Path(out)/'accounts'/symbol.replace('.','_')/'stock_summary.json',{'symbol':symbol,'failures':failures,
            'accounts':[{k:v for k,v in r.items() if k not in {'ledger','daily','rejections'}} for r in results]})
        return results,failures


def aggregate(accounts,symbols,cash,calendar=None):
    outputs=[]
    for window,variant,cost in sorted(set((r['window'],r['variant'],r['slippage_multiplier']) for r in accounts)):
        group=[r for r in accounts if (r['window'],r['variant'],r['slippage_multiplier'])==(window,variant,cost)]
        start,end=WINDOWS[window]
        dates=sorted({day['date'] for row in group for day in row['daily']})
        if calendar is not None:
            dates=[d for d in calendar if start<=d<=end]
        byday={d:0. for d in dates}
        for row in group:
            known={day['date']:float(day['equity']) for day in row['daily']}
            if len(known)!=len(row['daily']):raise ValueError('duplicate account daily dates')
            # A missing mark does not erase an independent account's capital.
            # Start flat with its cash and carry only earlier observed equity;
            # a later quote never fills a prior missing date.
            equity=float(cash)
            for day in dates:
                if day in known:equity=known[day]
                byday[day]+=equity
        initial=cash*len(symbols);values=pd.Series([initial,*[byday[d] for d in sorted(byday)]])
        outputs.append({'window':window,'variant':variant,'slippage_multiplier':cost,'accounts':len(group),'requested':len(symbols),
            'completed_accounts':sum(r['status'].startswith('completed') for r in group),
            'failed_cash_accounts':sum(not r['status'].startswith('completed') for r in group),
            'total_return':float(values.iloc[-1]/initial-1),'max_drawdown':float((values/values.cummax()-1).min()),
            'completed_round_trips':sum(r['completed_round_trips'] for r in group),
            'trade_count':sum(r['metrics']['trade_count'] for r in group),'fees':sum(r['metrics']['fees'] for r in group),
            'daily':[{'date':d,'equity':byday[d]} for d in sorted(byday)],'basis':'independent_equal_cash_no_shared_capital',
            'daily_alignment':'initial_cash_then_carry_prior_observed_equity_never_backfill_future_marks'})
    return outputs


def validate(model_run_id=None,*,cohort='controls',windows=None,cpu_workers=4):
    root,report=read_run(model_run_id)
    if report is None:raise ValueError('complete new model unavailable')
    records=report.get('validation_records',report['dataset_records'])
    if cohort=='controls':records=[r for r in records if r['symbol'] in {'000333.SZ','002436.SZ','600519.SH'}]
    windows=windows or list(WINDOWS);calendar=json.loads((root/'data_snapshots'/'calendar.json').read_text())['dates']
    context=create_run_context(task='czsc-trend-consensus-Broker-validation',as_of_date=report['data_end'],
        config={'model_run_id':root.name,'cohort':cohort,'windows':windows,'costs':[1,2],'locked_models':True},
        data_version=report['source_snapshot']['snapshot_hash'],scope='batch',stocks=[r['symbol'] for r in records])
    print('VALIDATION_RUN_ID '+context.run_id,flush=True)
    payloads=[(r,report,calendar,windows,str(root),str(context.run_dir())) for r in records];accounts=[];failures=[]
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for result,errors in pool.map(_worker_safe,payloads):accounts.extend(result);failures.extend(errors)
    else:
        for p in payloads:
            result,errors=_worker_safe(p);accounts.extend(result);failures.extend(errors)
    results=aggregate(accounts,[r['symbol'] for r in records],report['config']['initial_cash'],calendar)
    quarter=[r for r in results if r['variant']=='final_linear' and r['slippage_multiplier']==1 and r['window']!='2026YTD']
    count=sum(r['completed_round_trips'] for r in quarter)
    floor=count>=200 and len(quarter)==4 and all(r['completed_round_trips']>=50 for r in quarter)
    improvement=[]
    for new in [r for r in results if r['variant']=='final_linear']:
        old=next((r for r in results if r['variant']=='old_frozen_source_union_control' and r['window']==new['window'] and r['slippage_multiplier']==new['slippage_multiplier']),None)
        if old:improvement.append({'window':new['window'],'cost':new['slippage_multiplier'],'return_difference':new['total_return']-old['total_return'],
            'drawdown_difference':new['max_drawdown']-old['max_drawdown']})
    output={'run_id':context.run_id,'model_run_id':root.name,'status':'completed' if not failures else 'completed_with_failures',
        'cohort':cohort,'validation_count':len(records),'source_data_end':report['data_end'],'aggregates':results,'paired_control':improvement,
        'failures':failures,'sample_gate':{'passed':floor,'nonoverlapping_completed_round_trips':count,'required_total':200,'required_each_window':50,
            'YTD_and_arms_do_not_add_independent_samples':True},'independent_validation':{'status':'pending','development_windows_are_read':True},
        'architecture_selection':'preregistered_linear_main_tree_mlp_controlled_ablations','promotion_allowed':False}
    output['validation_script_sha256']=_file_hash(Path(__file__))
    _save(context.subdir('reports')/'trend_Broker_validation.json',output);context.write_metadata({'status':output['status'],'failures':failures})
    print('VALIDATION_COMPLETED '+context.run_id,flush=True);return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-run-id');parser.add_argument('--cohort',choices=['controls','training'],default='controls')
    parser.add_argument('--windows',nargs='+',choices=list(WINDOWS));parser.add_argument('--cpu-workers',type=int,default=4);args=parser.parse_args()
    result=validate(args.model_run_id,cohort=args.cohort,windows=args.windows,cpu_workers=args.cpu_workers)
    print(json.dumps({'run_id':result['run_id'],'sample_gate':result['sample_gate'],'failures':len(result['failures'])},ensure_ascii=False))
if __name__=='__main__':main()
