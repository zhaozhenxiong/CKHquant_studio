"""Explicit offline native-v4 training; never publishes or changes aliases."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from my_strategy.services.czsc_high_return_pipeline_v4 import train_native_research

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True,type=Path);p.add_argument('--end',default='2026-10-08')
    p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--device',default='cpu');p.add_argument('--risk-policy',default='source_confirmed_boundary_prior_atr_v4')
    args=p.parse_args();cohort=json.loads(args.cohort.read_text())
    train_native_research(symbols=cohort['symbols'],end=args.end,cpu_workers=args.cpu_workers,device=args.device,
        config={'risk_policy':args.risk_policy})
if __name__=='__main__':main()
