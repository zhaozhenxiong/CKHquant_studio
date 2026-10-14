"""Supervised source-only preparation from an explicit actual producer root.

All preparation code is imported by a fresh interpreter from the chosen root.
No training pipeline or pickle is loaded; cache misses use safe full replay.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end',required=True)
    parser.add_argument('--raw-db',type=Path)
    parser.add_argument('--cache-db',type=Path)
    parser.add_argument('--source-root',type=Path,help='Actual producer application/project code root, not artifact storage')
    parser.add_argument('--artifacts-root',type=Path,help='ARTIFACT_ROOT containing runs; independent from producer code')
    parser.add_argument('--calendar-run-id')
    parser.add_argument('--raw-source-run-id')
    parser.add_argument('--source-native-run-id',help='Cold-verify an already completed source; never replay or fit')
    parser.add_argument('--resume-native-run-id')
    parser.add_argument('--cpu-workers',type=int,default=6)
    parser.add_argument('--stock-batch-size',type=int,default=360)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--sync-id')
    parser.add_argument('--json',action='store_true',help='Final summary JSON is always emitted')
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    return parser


def _application(value=None):
    root=Path(value or Path(__file__).resolve().parents[2]).expanduser().resolve()
    if (root/'app/my_strategy').is_dir():root/= 'app'
    if not (root/'my_strategy/services/czsc_trend_source.py').is_file():
        raise ValueError('native producer source root missing')
    return root


def _summary_reader():
    # The new adapter may be staged beside this script while the deliberately
    # frozen producer is elsewhere. Load this fixed sibling adapter; all of its
    # generating imports still resolve from the fresh producer-root process.
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'services/board_native_summary.py'
    spec=importlib.util.spec_from_file_location('_khquant_native_dependency_reader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _validate_calendar(args,expected_hash):
    from my_strategy.services.czsc_trend_support import _calendar
    from my_strategy.core.run_context import stable_hash
    dates,_=_calendar(args.calendar_run_id);dates=[day for day in dates if day<=args.end]
    if not dates or dates[-1]!=args.end or stable_hash(dates)!=expected_hash:
        raise ValueError('native source differs from requested verified calendar prefix')


def _validate_current_population(args,reader,contract):
    from contextlib import closing
    import sqlite3
    warehouse=reader.raw_path(args.raw_db)
    with closing(sqlite3.connect(warehouse.as_uri()+'?mode=ro',uri=True,timeout=30)) as connection:
        connection.execute('PRAGMA query_only=ON');connection.execute('BEGIN')
        def progress(**value):
            value['completed']=value.pop('current');value['inflight']=0
            print('SOURCE_BATCH '+json.dumps(value,separators=(',',':')),flush=True)
        reader._verify_population_current(contract,connection,progress=progress)
        connection.rollback()


def _validate_requested_dependencies(args,proof):
    """Old completed sources must still describe today's requested inputs."""
    header=proof['header'];_validate_calendar(args,header['identity']['calendar_hash'])
    reader=_summary_reader()
    contract=reader._load_contract(args.source_root,args.artifacts_root,header['run_id'],
        proof['raw_source_run_id'],args.end,header['config'])
    _validate_current_population(args,reader,contract)


def _validate_raw_source(args,raw_id,cfg):
    from my_strategy.services.czsc_trend_source import _load_native_raw_source
    _,source=_load_native_raw_source(raw_id,cfg)
    if source['data_end']!=args.end:raise ValueError('native raw source target date differs')
    _validate_calendar(args,source['calendar_hash'])
    _validate_current_population(args,_summary_reader(),{'population':source['liquidity'],'dates':source['dates']})


def run_preparation(args):
    """Called only in the fresh source-root worker; useful for isolated tests."""
    from datetime import date
    date.fromisoformat(args.end)
    if args.cpu_workers<1 or args.stock_batch_size<1:raise ValueError('native preparation resources must be positive')
    from my_strategy.services.czsc_trend_contracts import strategy_config
    from my_strategy.services.czsc_trend_source import (build_liquidity_universe,freeze_raw_source,
        prepare_native_source,verify_native_checkpoint,_native_code_hashes)
    cfg=strategy_config(json.loads(args.config.read_text()) if args.config else None)
    cfg.update(full_qualified_pool=True,stock_batch_size=args.stock_batch_size)
    raw_id=args.raw_source_run_id
    completed_source=None
    if args.source_native_run_id and args.resume_native_run_id:
        raise ValueError('completed source and native resume are mutually exclusive')
    if args.source_native_run_id:
        proof=verify_native_checkpoint(args.source_native_run_id,raw_source_run_id=raw_id,config=cfg)
        if not proof['completed_report']:raise ValueError('native source is not completed')
        _validate_requested_dependencies(args,proof)
        raw_id=proof['raw_source_run_id'];completed_source=args.source_native_run_id
    if args.resume_native_run_id:
        proof=verify_native_checkpoint(args.resume_native_run_id,raw_source_run_id=raw_id,config=cfg,
            cpu_workers=args.cpu_workers,prepare_batch_size=args.stock_batch_size)
        _validate_requested_dependencies(args,proof)
        raw_id=proof['raw_source_run_id']
        if proof['completed_report']:completed_source=args.resume_native_run_id
    if not raw_id:
        print('STAGE raw_source_freeze',flush=True)
        population=build_liquidity_universe(feature_start=cfg['feature_start'],end=args.end,
            calendar_run_id=args.calendar_run_id,policy=cfg['liquidity_universe'],db_path=args.raw_db)
        _,raw=freeze_raw_source(symbols=population['selected_symbols'],end=args.end,
            calendar_run_id=args.calendar_run_id,db_path=args.raw_db,cpu_workers=args.cpu_workers,
            feature_start=cfg['feature_start'],policy=cfg['liquidity_universe'],preflight_population=population,
            prepare_batch_size=args.stock_batch_size,selection_metadata={'sync_id':args.sync_id})
        raw_id=raw['run_id']
    if completed_source:
        from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
        source=json.loads((ARTIFACT_RUNS_ROOT/completed_source/'reports/native_source.json').read_text())
        print('NATIVE_SOURCE_RUN_ID '+completed_source,flush=True)
    else:
        _validate_raw_source(args,raw_id,cfg)
        print('RAW_SOURCE_RUN_ID '+raw_id,flush=True)
        print('STAGE native_source_prepare',flush=True)
        source=prepare_native_source(raw_id,config=cfg,cpu_workers=args.cpu_workers,
            prepare_batch_size=args.stock_batch_size,resume_native_run_id=args.resume_native_run_id,
            cache_db_path=args.cache_db)
    if source.get('status')!='completed' or source.get('data_end')!=args.end or source.get('failures'):
        raise ValueError('native source preparation incomplete or target date differs')
    result=dict(status='completed',price_date=args.end,raw_source_run_id=raw_id,
        source_native_run_id=source['run_id'],training_started=False,sync_id=args.sync_id,
        source_root=str(_application(args.source_root)),artifacts_root=str(args.artifacts_root) if args.artifacts_root else None,
        cache_db_path=str(args.cache_db) if args.cache_db else None,
        code_hashes=_native_code_hashes(),completed=len(source['dataset_records']))
    print('NATIVE_CACHE_RESULT '+json.dumps(result,ensure_ascii=False,separators=(',',':')),flush=True)
    return result


def main(argv=None):
    args=build_parser().parse_args(argv)
    root=_application(args.source_root)
    if args._worker:
        sys.path.insert(0,str(root))
        run_preparation(args)
        return 0
    # A direct script worker avoids inheriting an already imported -m package
    # path and truthfully uses the declared producer bytes (including WAL code).
    env=os.environ.copy();env['KHQUANT_PROJECT_ROOT']=str(root);env['PYTHONPATH']=str(root)
    if args.raw_db:env['KHQUANT_RAW_DB']=str(args.raw_db.resolve())
    if args.artifacts_root:env['KHQUANT_ARTIFACT_ROOT']=str(args.artifacts_root.resolve())
    command=[sys.executable,str(Path(__file__).resolve()),*(argv if argv is not None else sys.argv[1:]),'--_worker']
    # stdout/stderr remain the supervisor's on-disk stream. The caller manages
    # cancellation for this complete process tree; there is no undrained pipe.
    return subprocess.call(command,cwd=root,env=env)


if __name__=='__main__':raise SystemExit(main())
