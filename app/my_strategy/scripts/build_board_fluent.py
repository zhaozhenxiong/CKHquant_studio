"""Build indexed latest board observations; no model or remote-data operations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb-db',type=Path);parser.add_argument('--raw-db',type=Path)
    parser.add_argument('--fluent-db',type=Path);parser.add_argument('--calendar-run-id')
    parser.add_argument('--price-date');parser.add_argument('--with-native',action='store_true')
    parser.add_argument('--native-cache-db',type=Path);parser.add_argument('--source-native-run-id')
    parser.add_argument('--raw-source-run-id');parser.add_argument('--source-root',type=Path)
    parser.add_argument('--artifacts-root',type=Path);parser.add_argument('--sync-id')
    parser.add_argument('--no-migrate',action='store_true');parser.add_argument('--no-publish',action='store_true')
    args=parser.parse_args(argv)
    from my_strategy.services.board_fluent import build_board_fluent
    result=build_board_fluent(kb_db_path=args.kb_db,raw_db_path=args.raw_db,fluent_db_path=args.fluent_db,
        calendar_run_id=args.calendar_run_id,price_date=args.price_date,with_native=args.with_native,
        native_cache_db_path=args.native_cache_db,source_native_run_id=args.source_native_run_id,
        raw_source_run_id=args.raw_source_run_id,source_root=args.source_root,artifacts_root=args.artifacts_root,
        sync_id=args.sync_id,publish=not args.no_publish,migrate=not args.no_migrate)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
