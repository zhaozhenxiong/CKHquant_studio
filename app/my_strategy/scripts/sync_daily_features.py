"""Synchronize current derived features without running model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('raw-db', 'kb-db', 'fluent-db', 'native-cache-db', 'source-root', 'artifacts-root', 'config'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--calendar-run-id')
    parser.add_argument('--resume-sync-id')
    parser.add_argument('--parent-update-id')
    parser.add_argument('--source-native-run-id')
    parser.add_argument('--raw-source-run-id')
    parser.add_argument('--cpu-workers', type=int, default=6)
    parser.add_argument('--stock-batch-size', type=int, default=360)
    parser.add_argument('--kb-request-budget', type=int, default=2000)
    parser.add_argument('--kb-min-interval', type=float, default=1.3)
    return parser


def main(argv=None):
    from my_strategy.services.daily_feature_sync import sync_daily_features, SyncCancelled
    args = build_parser().parse_args(argv)
    cancelled = threading.Event()
    def stop(_signal, _frame): cancelled.set()
    previous = {number: signal.signal(number, stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = sync_daily_features(raw_db_path=args.raw_db, kb_db_path=args.kb_db,
            fluent_db_path=args.fluent_db, native_cache_db_path=args.native_cache_db,
            calendar_run_id=args.calendar_run_id, sync_id=args.resume_sync_id,
            parent_update_id=args.parent_update_id, cpu_workers=args.cpu_workers,
            source_native_run_id=args.source_native_run_id, raw_source_run_id=args.raw_source_run_id,
            stock_batch_size=args.stock_batch_size, kb_request_budget=args.kb_request_budget,
            kb_min_interval=args.kb_min_interval, source_root=args.source_root,
            artifacts_root=args.artifacts_root, config=json.loads(args.config.read_text()) if args.config else None,
            check_cancel=cancelled.is_set,
            progress=lambda value: print(json.dumps({'event': 'feature_sync', **value}, ensure_ascii=False), file=sys.stderr, flush=True))
    except (SyncCancelled, KeyboardInterrupt):
        result = {'status': 'cancelled', 'sync_id': args.resume_sync_id, 'training_started': False,
            'error': 'daily sync interrupted before checkpoint acquisition'}
    finally:
        for number, handler in previous.items(): signal.signal(number, handler)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] in {'completed', 'partial'} else 130 if result['status'] == 'cancelled' else 2


if __name__ == '__main__': raise SystemExit(main())
