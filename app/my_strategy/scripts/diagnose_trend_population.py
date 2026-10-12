#!/usr/bin/env python3
"""Read-only E0/E1 population and real liquidation-reason diagnostics."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from my_strategy.services.czsc_trend_pipeline import read_run
from my_strategy.services.czsc_trend_dataset import training_frame
from my_strategy.services.czsc_trend_interval_model import split_validation,time_masks
from my_strategy.services.czsc_research import _save,_file_hash


def diagnose(model_run_id):
    root,report=read_run(model_run_id);cfg=report['config']
    calendar=json.loads((root/'data_snapshots/calendar.json').read_text())['dates'];market={d:i for i,d in enumerate(calendar)}
    stages={};clone_reasons={};diagnostics=[]
    for stage in ('E0','E1'):
        frame=pd.read_parquet(root/'dataset'/stage/'entries.parquet')
        frame['held_market_sessions']=frame.exit_date.map(market)-frame.entry_date.map(market)
        reasons={}
        for record in report['dataset_records']:
            directory=root/'dataset'/stage/record['symbol'].replace('.','_')
            for clone in json.loads((directory/'candidate_clones.json').read_text()):
                final=next((f for f in reversed(clone['ledger']) if f['action']=='SELL' and f.get('position_after')==0),None)
                reasons[(record['symbol'],clone['signal_date'])]=final['reason'] if final else 'unfilled_or_open'
            if stage=='E1':
                account=json.loads((directory/'reference_account.json').read_text())
                for row in account['exit_policy_diagnostics']:
                    diagnostics.append({'symbol':record['symbol'],'date':row['date'],'probability':row.get('probability'),
                        'threshold':row.get('calibrated_threshold'),'applied':row.get('applied',False),
                        'request_exit':row.get('request_exit',False),'status':row.get('status'),
                        'checkpoint':(row.get('model_identity') or {}).get('checkpoint')})
        frame['liquidation_reason']=[reasons.get((row.symbol,row.date),'unknown') for row in frame.itertuples()]
        stages[stage]=frame;clone_reasons[stage]=reasons
    def summary(frame):
        mature=frame[frame.label_available.fillna(False)]
        holding=mature.held_market_sessions.dropna()
        return {'rows':len(frame),'mature':len(mature),'positive':int(mature.label.eq(1).sum()),
            'net_return_at_least_minimum':int(mature.net_return.ge(cfg['high_return']['min_net_return']).sum()),
            'R_at_least_minimum':int(mature.r_multiple.ge(cfg['high_return']['min_r_multiple']).sum()),
            'median_held_market_sessions':float(holding.median()) if len(holding) else None,
            'held_no_more_than_one_market_session':int(holding.le(1).sum()),
            'median_net_return':float(mature.net_return.median()) if len(mature) else None,
            'liquidation_reasons':dict(Counter(mature.liquidation_reason)),
            'unavailable_reasons':dict(Counter(frame.loc[~frame.label_available.fillna(False),'label_reason']))}
    yearly={stage:{year:summary(group) for year,group in frame.groupby(frame.date.str[:4])} for stage,frame in stages.items()}
    e0=stages['E0'];e1=stages['E1']
    paired=e0[e0.label.eq(1)].merge(e1,on=['symbol','date'],suffixes=('_E0','_E1'))
    lost=paired[paired.label_available_E1.fillna(False)&paired.label_E1.eq(0)]
    recent=lost[lost.date.ge('2026-01-01')]
    examples=recent.sort_values('net_return_E0',ascending=False).head(12)[[
        'symbol','date','entry_date_E0','exit_date_E0','net_return_E0','r_multiple_E0','held_market_sessions_E0',
        'entry_date_E1','exit_date_E1','net_return_E1','r_multiple_E1','held_market_sessions_E1','liquidation_reason_E1']].to_dict('records')
    final={'entries':e1,'exits':pd.read_parquet(root/'dataset/E1/exits.parquet')}
    entry=training_frame(final,'entry',cfg,calendar,'E1');fold=split_validation(cfg['production'],calendar)
    partitions={name:summary(entry.loc[mask]) for name,mask in zip(('train','calibration','policy'),time_masks(entry,fold))}
    scored=pd.DataFrame(diagnostics);recent_scored=scored[scored.date.ge('2026-01-01')]
    output={'model_run_id':root.name,'script_sha256':_file_hash(Path(__file__)),'observational_only':True,
        'thresholds_or_policy_changed':False,'yearly':yearly,'production_fold':fold,'production_E1_partitions':partitions,
        'E0_positive_matched_E1':len(paired),'E0_positive_matched_E1_became_negative':len(lost),
        'recent_E0_positive_matched_E1_became_negative':len(recent),'recent_paired_examples':examples,
        'recent_reference_MA_exit':{'observed_holding_evaluations':len(recent_scored),
            'scored_rows':int(recent_scored.probability.notna().sum()),
            'exit_requests':int(recent_scored.request_exit.sum()),'actually_selected_exit':int(recent_scored.applied.sum()),
            'median_probability':float(recent_scored.probability.median()),'statuses':dict(Counter(recent_scored.status))},
        'interpretation':'actual_paid_ledger_outcomes_not_relabeling_or_model_selection;paired_opportunities_can_have_different_reference_cash'}
    directory=root/'validation/population_diagnostic';path=directory/'E0_E1_population.json'
    if path.exists():raise ValueError('existing diagnostic is immutable; choose another output run')
    _save(path,output);print(json.dumps(output,ensure_ascii=False));return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-run-id',required=True)
    args=parser.parse_args();diagnose(args.model_run_id)
if __name__=='__main__':main()
