#!/usr/bin/env python3
"""Read-only stop-reference precision sensitivity; paid ledger stays fixed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from my_strategy.services.czsc_trend_pipeline import read_run
from my_strategy.services.czsc_backtest import _tick
from my_strategy.execution.cost_model import CostModel,FLAT_COST_VERSION
from my_strategy.services.czsc_research import _save,_file_hash


def audit(model_run_id):
    root,report=read_run(model_run_id);cfg=report['config'];a=cfg['account'];h=cfg['high_return']
    cost=CostModel(commission=a['commission'],stamp_tax=a['stamp'],slippage=a['slippage'],
        min_commission=a['min_fee'],version=FLAT_COST_VERSION,impact_tiers=())
    stages={}
    for stage in ('E0','E1'):
        path=root/'dataset'/stage/'entries.parquet';frame=pd.read_parquet(path);differences=[];flips=[];eligibility=[]
        for row in frame.itertuples(index=False):
            values=[row.original_stop_price,row.entry_cash,row.entry_shares,row.risk_cash]
            if not np.isfinite(values).all() or row.entry_shares<=0:continue
            price=cost.execution_price('SELL',float(row.original_stop_price))
            risks={}
            for key,value in (('frozen_tick_reference',_tick(price)),('continuous_reference',price)):
                notional=row.entry_shares*value
                risks[key]=row.entry_cash-(notional-cost.fees('SELL',notional))
            if not np.isclose(risks['frozen_tick_reference'],row.risk_cash,atol=1e-6):
                raise ValueError('stored initial risk does not reproduce frozen tick reference')
            fraction=risks['continuous_reference']/row.entry_cash
            available=h['min_risk_fraction']<=fraction<=h['max_risk_fraction']
            item={'symbol':row.symbol,'date':row.date,'original_stop_price':row.original_stop_price,
                'stop_execution_continuous':price,'stop_execution_frozen_tick':_tick(price),
                'frozen_risk_cash':row.risk_cash,'continuous_risk_cash':risks['continuous_reference'],
                'risk_cash_difference':risks['continuous_reference']-row.risk_cash,
                'risk_cash_relative_difference':(risks['continuous_reference']/row.risk_cash-1),
                'net_return_unchanged':row.net_return,'net_profit_unchanged':row.net_profit,
                'frozen_r_multiple':row.r_multiple,
                'continuous_r_multiple':row.net_profit/risks['continuous_reference'] if np.isfinite(row.net_profit) else None,
                'frozen_label':row.label,'frozen_risk_available':row.risk_available,
                'continuous_risk_available':available}
            differences.append(item)
            if available!=row.risk_available:eligibility.append(item)
            if row.label_available and available:
                new_high=row.net_return>=h['min_net_return'] and item['continuous_r_multiple']>=h['min_r_multiple']
                if float(new_high)!=row.label:flips.append(item)
        change=pd.DataFrame(differences)
        stages[stage]={'source_entries_sha256':_file_hash(path),'entries':len(frame),'reproduced_risk_rows':len(change),
            'risk_eligibility_changes':len(eligibility),'mature_label_changes':len(flips),
            'risk_cash_absolute_difference_max':float(change.risk_cash_difference.abs().max()),
            'risk_cash_relative_difference_max':float(change.risk_cash_relative_difference.abs().max()),
            'mature_label_flip_examples':flips[:30],'risk_eligibility_change_examples':eligibility[:30],
            'near_2R_and_8pct_examples':sorted([x for x in differences if np.isfinite(x['net_return_unchanged']) and
                x['net_return_unchanged']>=h['min_net_return']],key=lambda x:abs(x['frozen_r_multiple']-h['min_r_multiple']))[:15]}
    target=next((json.loads((root/b['heads']['entry']['path']/'manifest.json').read_text())['target_contract']
        for b in report['model_bundles']),{})
    result={'model_run_id':root.name,'script_sha256':_file_hash(Path(__file__)),'paid_ledger_changes':False,
        'generation_sources_changed':False,'stages':stages,
        'execution_observed':'CostModel continuous modeled average fills; no exchange-tick rounding in new account',
        'initial_R_observed':'frozen stop-reference SELL price is rounded to .01 after CostModel slippage',
        'explicit_price_precision_field_in_target':any('precision' in k or 'tick' in k for k in target),
        'interpretation':'counterfactual_reference_R_sensitivity_only_not_relabeling_or_execution_replay'}
    path=root/'validation/risk_precision_sensitivity.json'
    if path.exists():raise ValueError('risk precision sensitivity report is immutable')
    _save(path,result);print(json.dumps(result,ensure_ascii=False));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-run-id',required=True)
    args=parser.parse_args();audit(args.model_run_id)
if __name__=='__main__':main()
