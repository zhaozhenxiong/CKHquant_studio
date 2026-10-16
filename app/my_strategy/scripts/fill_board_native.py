"""Preview or explicitly execute a bounded full SH/SZ board-native task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preview', 'execute'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--price-date')
    parser.add_argument('--raw-db', type=Path);parser.add_argument('--kb-db', type=Path)
    parser.add_argument('--fluent-db', type=Path);parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--cpu-workers', type=int, help='1..12; preview defaults to 4, execute keeps prior setting')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-stocks', type=int, help='Optional per-attempt limit, useful for a measured 20-stock pilot')
    parser.add_argument('--cache-quota-bytes', type=int,
        help='Explicit local attempt capacity, 1..64 GiB in bytes; default remains 8 GiB, same parent/worker limit')
    parser.add_argument('--budget-seconds', type=float, default=1800,
        help='Soft dispatch budget; in-flight stocks finish and final verification is outside it')
    args = parser.parse_args(argv)
    from my_strategy.core.paths import METADATA_ROOT
    from my_strategy.services.board_native_bulk import execute_bulk, preview_bulk, _receipt
    from my_strategy.storage.research_store import ResearchStore
    if args.cpu_workers is not None and not 1 <= args.cpu_workers <= 12:
        parser.error('--cpu-workers must be 1..12')
    if args.cache_quota_bytes is not None and not 1 <= args.cache_quota_bytes <= 64*1024**3:
        parser.error('--cache-quota-bytes must be a positive byte count at most 64 GiB')
    stop = threading.Event()
    def cancel(_number, _frame):stop.set()
    previous = {number: signal.signal(number, cancel) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ResearchStore(args.cache_root or METADATA_ROOT / 'light_research',
                           quota_bytes=args.cache_quota_bytes or 8*1024**3) as store:
            if args.action == 'preview':
                state = preview_bulk(store=store, checkpoint=args.checkpoint, raw_db_path=args.raw_db,
                    kb_db_path=args.kb_db, fluent_db_path=args.fluent_db, price_date=args.price_date,
                    batch_size=args.batch_size, cpu_workers=args.cpu_workers or 4, stop=stop)
                result = {**_receipt(state, args.checkpoint.resolve()), 'counts': state['counts'],
                    'contract_sha256': state['contract_sha256'], 'executor': state['executor']}
            else:
                result = execute_bulk(store=store, checkpoint=args.checkpoint,
                    budget_seconds=args.budget_seconds, max_stocks=args.max_stocks, cpu_workers=args.cpu_workers,
                    cache_quota_bytes=args.cache_quota_bytes, stop=stop,
                    progress=lambda value: print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True))
            print(json.dumps({'event': 'board_native_bulk_result', **result}, ensure_ascii=False, allow_nan=False), flush=True)
            return 0 if result['status'] in {'planned', 'completed', 'partial'} else 130 if result['status'] == 'cancelled' else 2
    except (InterruptedError, KeyboardInterrupt):
        print(json.dumps({'event': 'board_native_bulk_result', 'status': 'cancelled',
            'training_started': False, 'legacy_tasks_resumed': False}), flush=True)
        return 130
    finally:
        for number, handler in previous.items():signal.signal(number, handler)


if __name__ == '__main__':raise SystemExit(main())
