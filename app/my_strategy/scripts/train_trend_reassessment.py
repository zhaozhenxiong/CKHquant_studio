#!/usr/bin/env python3
"""Rebuild the isolated v2 reassessment family from new immutable raw inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def _identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}',value):
        raise argparse.ArgumentTypeError('invalid snapshot run identifier')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-source-run-id',type=_identifier)
    parser.add_argument('--source-native-run-id',type=_identifier)
    parser.add_argument('--calendar-run-id',type=_identifier)
    parser.add_argument('--end',required=True)
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--cohort-size',type=int,default=360)
    parser.add_argument('--cohort-file',type=Path,help='A fixed ordered symbol list, without outcomes')
    parser.add_argument('--capacity-tiers',nargs='+',type=int,default=[60,180,360])
    parser.add_argument('--cpu-workers',type=int,default=6)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--freeze-raw-only',action='store_true')
    parser.add_argument('--prepare-source-only',action='store_true')
    args=parser.parse_args()
    from datetime import date
    date.fromisoformat(args.end)
    if args.cpu_workers < 1 or args.cohort_size < 1 or any(tier < 1 for tier in args.capacity_tiers):
        parser.error('worker count, cohort size and capacity tiers must be positive')
    if args.freeze_raw_only and args.source_native_run_id:
        parser.error('raw-only mode does not accept a native snapshot')
    from my_strategy.adapters.czsc_adapter import normalize_symbol
    args.symbols=[normalize_symbol(symbol) for symbol in args.symbols] if args.symbols else None
    from my_strategy.services.czsc_trend_v2_contracts import strategy_config, TRAIN_KIND
    from my_strategy.services.czsc_trend_v2_source import select_cohort, freeze_raw_source, prepare_native_source
    cfg=strategy_config(json.loads(args.config.read_text()) if args.config else None)
    cfg['training_capacity_tiers']=list(dict.fromkeys(args.capacity_tiers))
    native=args.source_native_run_id
    if not native:
        raw_id=args.raw_source_run_id
        if not raw_id:
            preferred=json.loads(args.cohort_file.read_text()) if args.cohort_file else None
            metadata={key:value for key,value in preferred.items() if key!='symbols'} if isinstance(preferred,dict) else {}
            if isinstance(preferred,dict):preferred=preferred.get('symbols')
            if preferred is not None and not isinstance(preferred,list):
                parser.error('cohort file must contain a symbol list')
            preferred=[normalize_symbol(symbol) for symbol in preferred] if preferred else None
            symbols=args.symbols or select_cohort(args.cohort_size,preferred=preferred)
            _,raw=freeze_raw_source(symbols=symbols,end=args.end,calendar_run_id=args.calendar_run_id,
                                   cpu_workers=args.cpu_workers,selection_metadata=metadata)
            raw_id=raw['run_id']
        if args.freeze_raw_only:
            print(json.dumps({'raw_source_run_id':raw_id,'training_started':False},ensure_ascii=False),flush=True)
            return
        source=prepare_native_source(raw_id,config=cfg,cpu_workers=args.cpu_workers)
        if source['data_end'] != args.end:
            raise ValueError('training end differs from immutable raw source end')
        native=source['run_id']
    if args.prepare_source_only:
        print(json.dumps({'source_native_run_id':native,'training_started':False},ensure_ascii=False),flush=True)
        return
    from my_strategy.services.czsc_trend_v2_pipeline import train_trend_consensus
    report=train_trend_consensus(source_native_run_id=native,end=args.end,symbols=args.symbols,
        config=cfg,cpu_workers=args.cpu_workers,device=args.device)
    from my_strategy.services.czsc_research import _json
    from my_strategy.storage.czsc_results import ResultStore
    ResultStore().save(TRAIN_KIND,_json(report))
    print(json.dumps({'run_id':report['run_id'],'pipeline':report['pipeline'],
        'model_gate':report['model_gate'],'publication_allowed':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
