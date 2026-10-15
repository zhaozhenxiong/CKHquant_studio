"""Read-only health checks for the current lightweight KHQuant workspace.

No model load, training probe, database initialization, native preparation or
network request occurs. Missing optional caches and retired registries are not
workspace errors. Use --deep explicitly for SQLite integrity/full-row counts.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date
import importlib.metadata
import json
from pathlib import Path
import sqlite3

REQUIRED_PATHS=('AGENTS.md','my_strategy/configs','my_strategy/core/paths.py',
    'my_strategy/services/light_research.py','my_strategy/services/incremental_source.py',
    'my_strategy/services/position_simulation.py','my_strategy/web_dashboard/app.py',
    'my_strategy/scripts/update_daily_data.py')


def find_project_root(candidate):
    path=Path(candidate).expanduser().resolve()
    if path.is_file():path=path.parent
    for root in (path,path/'app',*path.parents):
        if (root/'AGENTS.md').is_file() and (root/'my_strategy').is_dir():return root
    raise FileNotFoundError('No KHQuant application root with AGENTS.md and my_strategy found')


def inspect_database(path,required_tables=(),*,deep=False):
    path=Path(path).resolve();report={'path':str(path),'exists':path.is_file(),'missing_tables':[],'error':None}
    if not path.is_file():return report
    try:
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as connection:
            connection.execute('PRAGMA query_only=ON')
            tables={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            report['missing_tables']=sorted(set(required_tables)-tables)
            if deep:report['quick_check']=connection.execute('PRAGMA quick_check').fetchone()[0]
    except sqlite3.Error as exc:report['error']=str(exc)
    return report


def inspect_raw(path,target_date=None,min_history=60,*,deep=False):
    from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES
    report=inspect_database(path,{'stock_daily_normalized'},deep=deep)
    if not report['exists'] or report['error'] or report['missing_tables']:return report
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=5)) as connection:
        connection.execute('PRAGMA query_only=ON')
        columns={row[1] for row in connection.execute('PRAGMA table_info(stock_daily_normalized)')}
        report['missing_columns']=sorted({'stock','date','open','high','low','close','volume','amount','source','has_trade_price'}-columns)
        if report['missing_columns']:return report
        sources=sorted(VERIFIED_SOURCES);placeholders=','.join('?' for _ in sources)
        # These are source/price-contract counts, not a replacement for per-bar
        # OHLC/calendar/liquidity validation in the actual source service.
        contract='has_trade_price=1 AND source IN ('+placeholders+')'
        report['latest_date']=connection.execute('SELECT MAX(date) FROM stock_daily_normalized').fetchone()[0]
        report['latest_qualified_source_date']=connection.execute('SELECT MAX(date) FROM stock_daily_normalized WHERE '+contract,sources).fetchone()[0]
        day=target_date or report['latest_qualified_source_date']
        report.update(target_date=day,min_history=min_history,coverage_basis='verified_source_and_trade_price_contract',
            per_bar_quality_checked=False)
        if day:
            rows=connection.execute('SELECT DISTINCT stock FROM stock_daily_normalized WHERE date=? AND '+contract,[day,*sources]).fetchall()
            symbols=[row[0] for row in rows if (row[0].endswith('.SH') and row[0].startswith('6')) or
                     (row[0].endswith('.SZ') and row[0].startswith(('0','3')))]
            report['target_source_qualified_stocks']=len(rows)
            report['target_sh_sz_a_stocks']=len(symbols)
            # Each indexed symbol query stops after the requested warmup. This
            # avoids a full 20-million-row GROUP BY for an ordinary health read.
            report['warmup_source_qualified_sh_sz_a_stocks']=sum(connection.execute(
                'SELECT COUNT(*) FROM (SELECT 1 FROM stock_daily_normalized WHERE stock=? AND date<=? AND '+contract+' LIMIT ?)',
                [symbol,day,*sources,min_history]).fetchone()[0]>=min_history for symbol in symbols)
        if deep:report['daily_rows']=connection.execute('SELECT COUNT(*) FROM stock_daily_normalized').fetchone()[0]
    return report


def verify(project_root,*,raw_db=None,metadata_root=None,fluent_db=None,target_date=None,min_history=60,strict=False,deep=False):
    from my_strategy.core.paths import METADATA_ROOT,BASIC_FLUENT_DB_PATH
    from my_strategy.storage.db_paths import raw_db_path
    if type(min_history) is not int or not 1<=min_history<=10000:raise ValueError('min_history must be 1..10000')
    if target_date and date.fromisoformat(target_date).isoformat()!=target_date:raise ValueError('target_date must be ISO YYYY-MM-DD')
    root=find_project_root(project_root);metadata=Path(metadata_root or METADATA_ROOT)
    missing=[name for name in REQUIRED_PATHS if not (root/name).exists()]
    raw=inspect_raw(raw_db or raw_db_path(),target_date,min_history,deep=deep)
    optional=[inspect_database(metadata/'light_research'/'research.db',{'results','settings','objects'},deep=deep),
        inspect_database(metadata/'personal_portfolio.db',{'watchlist'},deep=deep),
        inspect_database(fluent_db or BASIC_FLUENT_DB_PATH,{'feature_generations'},deep=deep)]
    errors=['required path missing: '+name for name in missing]
    for item in [raw,*optional]:
        if item['error'] or item['missing_tables'] or item.get('missing_columns') or item.get('quick_check','ok')!='ok':
            errors.append('database check failed: '+item['path'])
    if strict and not raw['exists']:errors.append('raw market database missing')
    if strict and target_date and not raw.get('target_sh_sz_a_stocks'):errors.append('requested target date has no qualified SH/SZ A-share source rows')
    dependencies={}
    for name in ('numpy','pandas','torch','fastapi'):
        try:dependencies[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:dependencies[name]=None
    return {'status':'failed' if errors else 'ok','architecture':'structure_kronos_lightweight',
        'project_root':str(root),'required_paths_missing':missing,'raw_database':raw,'optional_databases':optional,
        'dependency_versions':dependencies,'errors':errors,'read_only':True,'training_started':False,
        'native_preparation_started':False,'models_loaded':False,'legacy_registry_required':False,
        'independent_forward_qualification_checked':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',default='.')
    parser.add_argument('--raw-db',type=Path)
    parser.add_argument('--target-date')
    parser.add_argument('--min-history',type=int,default=60)
    parser.add_argument('--strict',action='store_true')
    parser.add_argument('--deep',action='store_true')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    try:report=verify(args.project_root,raw_db=args.raw_db,target_date=args.target_date,min_history=args.min_history,
        strict=args.strict,deep=args.deep)
    except (ValueError,FileNotFoundError,sqlite3.Error) as exc:
        report={'status':'failed','errors':[str(exc)],'read_only':True}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
