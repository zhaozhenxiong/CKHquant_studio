#!/usr/bin/env python3
"""Reaggregate paid immutable accounts without rerunning any decision model."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context,stable_hash
from my_strategy.services.czsc_research import _save,_file_hash
from my_strategy.services.czsc_trend_pipeline import read_run,completed_round_trips
from my_strategy.scripts.validate_trend_consensus import aggregate,WINDOWS


def _read_verified(path,expected=None):
    path=Path(path);before=_file_hash(path)
    if expected is not None and before!=expected:raise ValueError('immutable account asset SHA drift')
    value=json.loads(path.read_text())
    if _file_hash(path)!=before:raise ValueError('account asset changed during read')
    return value,{'path':str(path),'sha256':before,'prior_expected_sha256':expected}


def verify_saved_account(row,cash,end):
    inventory=0;running=float(cash)
    for fill in row['ledger']:
        if fill['date']>end:raise ValueError('future fill in reused account prefix')
        if fill['action'] not in ('BUY','SELL','REDUCE'):raise ValueError('unknown ledger action')
        if fill.get('position_before',inventory)!=inventory:raise ValueError('ledger inventory prefix drift')
        inventory+=int(fill['shares']) if fill['action']=='BUY' else -int(fill['shares'])
        running+=float(fill['cash_flow'])
        if inventory<0 or fill.get('position_after',inventory)!=inventory:raise ValueError('ledger inventory mismatch')
        if not np.isclose(running,fill['cash_after'],atol=1e-6):raise ValueError('ledger cash mismatch')
    if len(row['ledger'])!=row['metrics']['trade_count']:raise ValueError('paid trade count differs from prior summary')
    if not np.isclose(sum(float(f['fee']) for f in row['ledger']),row['metrics']['fees'],atol=1e-6):
        raise ValueError('paid fees differ from prior summary')
    if row['daily'] and not np.isclose(row['daily'][-1]['equity'],row['metrics']['final_equity'],atol=1e-6):
        raise ValueError('actual last observed equity differs from prior summary')
    if row['completed_round_trips']!=completed_round_trips(row['ledger']):raise ValueError('complete inventory interval count drift')
    return {'cash_ledger_reconciled':True,'inventory_reconciled':True,'prior_metrics_reconciled':True}


def reaggregate(parent_validation_run_id):
    parent=(ARTIFACT_RUNS_ROOT/parent_validation_run_id).resolve()
    if not parent.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):raise ValueError('validation root outside artifact store')
    old,report_asset=_read_verified(parent/'reports/trend_Broker_validation.json')
    model_root,model=read_run(old['model_run_id']);cash=float(model['config']['initial_cash'])
    calendar=json.loads((model_root/'data_snapshots/calendar.json').read_text())['dates']
    records=model['validation_records']
    if old['cohort']=='controls':records=[r for r in records if r['symbol'] in {'000333.SZ','002436.SZ','600519.SH'}]
    context=create_run_context(task='czsc-trend-Broker-equity-reaggregation',as_of_date=old['source_data_end'],
        config={'parent_validation_run_id':parent.name,'model_run_id':model_root.name,
            'cause':'missing_account_daily_row_was_incorrectly_summed_as_zero_asset',
            'models_or_decisions_or_paid_ledgers_regenerated':False},
        data_version=report_asset['sha256'],scope='batch',stocks=[r['symbol'] for r in records])
    print('REAGGREGATION_RUN_ID '+context.run_id,flush=True)
    accounts=[];assets=[report_asset];checks=[]
    for record in records:
        symbol=record['symbol'];stock_dir=parent/'accounts'/symbol.replace('.','_')
        stock,meta=_read_verified(stock_dir/'stock_summary.json');assets.append(meta)
        for saved in stock['accounts']:
            row=copy.deepcopy(saved);start,end=WINDOWS[row['window']];end=min(end,old['source_data_end'])
            if row.get('source_files'):
                for key in ('ledger','daily'):
                    src=row['source_files'][key]
                    values,asset=_read_verified(src['path'],src['sha256']);assets.append(asset)
                    row[key]=[v for v in values if start<=v['date']<=end]
            else:
                directory=stock_dir/row['window']/row['variant']/('slippage_'+str(row['slippage_multiplier']))
                for key in ('ledger','daily'):
                    path=directory/(key+'.json')
                    if path.exists():row[key],asset=_read_verified(path);assets.append(asset)
                    elif row['status']=='failed_cash_retained':
                        row[key]=[] if key=='ledger' else [{'date':d,'equity':cash} for d in calendar if start<=d<=end]
                    else:raise ValueError('paid account asset unavailable')
                summary,asset=_read_verified(directory/'summary.json');assets.append(asset)
                if summary['metrics']!=row['metrics']:raise ValueError('parent stock/account summary mismatch')
            row['rejections']=[]
            check=verify_saved_account(row,cash,end);checks.append({'symbol':symbol,'window':row['window'],
                'variant':row['variant'],'slippage_multiplier':row['slippage_multiplier'],**check})
            accounts.append(row)
    results=aggregate(accounts,[r['symbol'] for r in records],cash,calendar)
    if len(accounts)!=sum(a['accounts'] for a in results):raise ValueError('account completeness mismatch')
    quarter=[r for r in results if r['variant']=='final_linear' and r['slippage_multiplier']==1 and r['window']!='2026YTD']
    count=sum(r['completed_round_trips'] for r in quarter)
    pairs=[]
    for new in [r for r in results if r['variant']=='final_linear']:
        baseline=next(r for r in results if r['variant']=='old_frozen_source_union_control' and
            r['window']==new['window'] and r['slippage_multiplier']==new['slippage_multiplier'])
        pairs.append({'window':new['window'],'cost':new['slippage_multiplier'],
            'return_difference':new['total_return']-baseline['total_return'],
            'drawdown_difference':new['max_drawdown']-baseline['max_drawdown']})
    out=copy.deepcopy(old);out.update(run_id=context.run_id,parent_validation_run_id=parent.name,
        status='completed_reaggregation' if not old['failures'] else 'completed_reaggregation_with_parent_failures',
        aggregates=results,paired_control=pairs,actual_accounts_reused=len(accounts),actual_accounts_regenerated=0,
        ledger_and_model_changes=False,original_report_preserved=True,
        aggregation_repair='per_account_initial_cash_then_causal_carry_of_prior_observed_net_equity',
        supersedes_parent_aggregate_metrics_only=True,validation_script_sha256=_file_hash(Path(__file__)),
        aggregator_sha256=_file_hash(Path(sys.modules[aggregate.__module__].__file__)))
    out['sample_gate']['passed']=count>=200 and len(quarter)==4 and all(r['completed_round_trips']>=50 for r in quarter)
    out['sample_gate']['nonoverlapping_completed_round_trips']=count
    _save(context.subdir('validation')/'reused_asset_hashes.json',assets)
    _save(context.subdir('validation')/'account_cash_inventory_checks.json',checks)
    _save(context.subdir('reports')/'trend_Broker_validation.json',out)
    context.write_metadata({'status':out['status'],'actual_accounts_reused':len(accounts),'actual_accounts_regenerated':0})
    print('REAGGREGATION_COMPLETED '+context.run_id,flush=True);return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-validation-run-id',required=True)
    args=parser.parse_args();out=reaggregate(args.parent_validation_run_id)
    print(json.dumps({'run_id':out['run_id'],'actual_accounts_reused':out['actual_accounts_reused'],
        'actual_accounts_regenerated':out['actual_accounts_regenerated'],'sample_gate':out['sample_gate']},ensure_ascii=False))
if __name__=='__main__':main()
