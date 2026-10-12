#!/usr/bin/env python3
"""Train immutable independent trend and real-account interval model families."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from my_strategy.services.czsc_trend_pipeline import train_trend_consensus


def install_finalization_path_adapter():
    """Normalize only SHA-helper arguments; frozen decision sources stay intact."""
    from my_strategy.services import czsc_trend_pipeline as pipeline
    from my_strategy.services.czsc_research import _file_hash
    pipeline._file_hash=lambda path:_file_hash(Path(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-native-run-id',default='20261012-120726-523936-20261008-czsc-high-return-native-v4-train-NO_GIT')
    parser.add_argument('--end',default='2026-10-08')
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--cohort',choices=['controls','training'],default='training')
    parser.add_argument('--cpu-workers',type=int,default=6)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume-run-id',help='Audit/finalize an existing complete data/model run, then continue capacity tiers')
    args=parser.parse_args();symbols=args.symbols
    install_finalization_path_adapter()
    if symbols is None and args.cohort=='controls':symbols=['002436.SZ','000333.SZ','600519.SH']
    config=json.loads(args.config.read_text()) if args.config else None
    if args.resume_run_id:
        from my_strategy.services.czsc_trend_finalization import finalize_existing_run
        from my_strategy.services.czsc_trend_pipeline import read_run,expand_training_snapshot,_train_one
        from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
        source_root=ARTIFACT_RUNS_ROOT/args.resume_run_id
        if not (source_root/'reports/trend_consensus.json').exists():finalize_existing_run(args.resume_run_id)
        _,report=read_run(args.resume_run_id)
        registered=json.loads((source_root/'config/initial_config.json').read_text())
        base=json.loads((ARTIFACT_RUNS_ROOT/args.source_native_run_id/'reports/native_v4.json').read_text())
        for tier in (180,360):
            if report['required_recent_heads_ready']:break
            if len(report['dataset_records'])>=tier:continue
            records=expand_training_snapshot(base,tier,args.end,args.cpu_workers)
            report=_train_one(source_native_run_id=args.source_native_run_id,end=args.end,config=registered,
                cpu_workers=args.cpu_workers,device=args.device,native_records=records)
    else:
        report=train_trend_consensus(source_native_run_id=args.source_native_run_id,symbols=symbols,end=args.end,
            config=config,cpu_workers=args.cpu_workers,device=args.device)
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.services.czsc_research import _json
    from my_strategy.services.czsc_trend_contracts import TRAIN_KIND
    ResultStore().save(TRAIN_KIND, _json(report))
    print(json.dumps({'run_id':report['run_id'],'pipeline':report['pipeline'],'model_gate':report['model_gate']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
