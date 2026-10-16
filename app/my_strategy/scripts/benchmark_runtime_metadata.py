"""Small synthetic runtime benchmark; all databases/results live in a temp dir.

Run from the app root. No services, training, network or production data access.
"""
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import threading
import time
import tracemalloc
from types import SimpleNamespace


def measure(call, n=7):
    elapsed=[]
    for _ in range(n):
        at=time.perf_counter();call();elapsed.append(time.perf_counter()-at)
    tracemalloc.start();call();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return {'median_seconds':statistics.median(elapsed),'p95_seconds':sorted(elapsed)[min(n-1,int(n*.95))],
            'peak_python_bytes':peak,'samples':n}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from my_strategy.services.task_maintenance import TaskMaintenance
    from my_strategy.storage.task_journal import TaskJournal
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.storage.scan_view import compact_scan_view
    from my_strategy.storage.status_summary import read_status
    with tempfile.TemporaryDirectory(prefix='khquant-runtime-benchmark-') as directory:
        root=Path(directory); raw=root/'raw.db';cache=root/'status.db'
        with sqlite3.connect(raw) as conn:
            conn.execute('CREATE TABLE prices(stock TEXT,date TEXT,source TEXT,has_trade_price INTEGER)')
            conn.executemany('INSERT INTO prices VALUES(?,?,?,?)',((str(i%500),'2026-10-15','fuyao_unadjusted_v1',1) for i in range(200000)))
        def aggregate():
            with closing(sqlite3.connect(raw)) as conn:
                conn.row_factory=sqlite3.Row
                return dict(conn.execute("SELECT MIN(date) AS start,MAX(date) AS end,COUNT(*) AS rows,COUNT(DISTINCT stock) AS symbols,SUM(CASE WHEN source LIKE '%fuyao%' AND source <> 'fuyao_unadjusted_v1' THEN 1 ELSE 0 END) AS legacy_fuyao_bars,SUM(CASE WHEN has_trade_price IS NULL OR has_trade_price <> 1 THEN 1 ELSE 0 END) AS unverified_trade_price_bars FROM prices").fetchone())
        status_before=measure(aggregate)
        read_status(raw,aggregate,cache_path=cache)
        status_after=measure(lambda:read_status(raw,aggregate,cache_path=cache))
        task_db=root/'tasks.db'
        with sqlite3.connect(task_db) as conn:conn.execute('CREATE TABLE czsc_tasks(job_id TEXT PRIMARY KEY,fingerprint TEXT,status TEXT,payload TEXT)')
        results=ResultStore(root/'results.db',root/'runs')
        with ResearchStore(root/'research') as store:
            journal=TaskJournal(root/'research/task_journal.db')
            tasks=SimpleNamespace(db_path=task_db,results=results,_lock=threading.RLock())
            light=SimpleNamespace(store=store,journal=journal,jobs={},stops={},lock=threading.RLock())
            maintenance=TaskMaintenance(tasks,light,metadata_root=root)
            with sqlite3.connect(task_db) as conn:
                conn.executemany('INSERT INTO czsc_tasks VALUES(?,?,?,?)',((str(i),str(i),'succeeded',json.dumps({'job_id':str(i),'kind':'scan','status':'succeeded','created_at':f'2026-10-16T01:{i%60:02d}:00+08:00','progress':{'completed':5000,'total':5000},'spec':{'symbols':['600000.SH']},'error':None})) for i in range(10000)))
            def old_list():
                all_rows=maintenance._all()
                rows=sorted(all_rows,key=lambda r:(r.get('status') in {'queued','running','cancelling'},str(r.get('finished_at') or r.get('updated_at') or r.get('created_at') or ''),r['backend'],r['job_id']),reverse=True)
                return rows[:20]
            history_before=measure(old_list);history_after=measure(lambda:maintenance.list(page_size=20))
            source={'run_id':'large-scan','coverage':{'requested':5000,'success':5000},'rows':[
                {'symbol':f'{i:06d}.SZ','name':f'Stock {i}','category':'watch','reference_action':'WAIT','raw_history':[i]*300,
                 'events':[{'side':'BUY','time':'2026-10-15','vector':[0.1]*100}]} for i in range(5000)]}
            results.save('scan',source)
            scan_before=measure(lambda:compact_scan_view(results.get('large-scan')))
            scan_after=measure(lambda:results.get_scan_page('large-scan',page=1,page_size=20))
            report_size=(root/'runs/large-scan/reports/czsc.json').stat().st_size
        report={'kind':'synthetic_single_process_runtime_benchmark','production_data_accessed':False,
            'raw_status':{'rows':200000,'before':status_before,'after_hot':status_after,'external_summary_bytes':cache.stat().st_size},
            'task_history':{'rows':10000,'page_size':20,'before':history_before,'after':history_after,'metadata_db_bytes':task_db.stat().st_size},
            'compact_scan':{'rows':5000,'page_size':20,'report_bytes':report_size,'before_full_projection':scan_before,'after_page':scan_after,'result_index_db_bytes':results.db_path.stat().st_size}}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
