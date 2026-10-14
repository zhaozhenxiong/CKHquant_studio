"""Build current board context from the latest qualified local price session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb-db',type=Path)
    parser.add_argument('--raw-db',type=Path)
    parser.add_argument('--calendar-run-id')
    parser.add_argument('--price-date',help='Must equal the latest qualified local price session')
    parser.add_argument('--mode',choices=['current_reference'],default='current_reference')
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--min-members',type=int,default=5)
    parser.add_argument('--min-coverage',type=float,default=.8)
    parser.add_argument('--no-cache',action='store_true')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    from my_strategy.adapters.czsc_adapter import _connection
    from my_strategy.services.czsc_trend_support import _calendar, VERIFIED_SOURCES
    from my_strategy.services.board_features import build_board_features
    from my_strategy.storage.board_knowledge import BoardKnowledgeStore
    dates,calendar_id=_calendar(args.calendar_run_id)
    with _connection(args.raw_db) as conn:
        day=conn.execute('SELECT MAX(date) FROM stock_daily_normalized WHERE has_trade_price=1 AND source IN ('+
            ','.join('?' for _ in VERIFIED_SOURCES)+')',sorted(VERIFIED_SOURCES)).fetchone()[0]
    if not day:raise ValueError('no qualified local price session')
    if args.price_date is not None and args.price_date!=day:
        raise ValueError('当前板块特征只计算最新本地行情日：'+day)
    store=BoardKnowledgeStore(args.kb_db,readonly=False)
    result=build_board_features(store,raw_db_path=args.raw_db,calendar_dates=dates,price_date=day,mode=args.mode,
        symbols=args.symbols,policy=dict(min_members=args.min_members,min_coverage=args.min_coverage),use_cache=not args.no_cache)
    summary={key:result[key] for key in ('mode','price_date','membership_date','training_eligible','qualification','schema_hash','cache_key','diagnostics')}
    summary['calendar_run_id']=calendar_id
    print(json.dumps(result if args.json else summary,ensure_ascii=False,allow_nan=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
