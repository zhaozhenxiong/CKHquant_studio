"""Small, reproducible CPU-only cache benchmark using disposable fixtures.

Never opens the configured business cache. The baseline is an explicit saved
research_store.py. Identical real CAS files are seeded without timing setup;
1k/5k metadata scaling, hot put/get/GC/list and batch publication are measured.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time


def timing(action, count):
    samples = []
    for _ in range(count):
        start = time.perf_counter(); action(); samples.append((time.perf_counter()-start)*1000)
    ordered = sorted(samples)
    return {'n': count, 'median_ms': statistics.median(samples),
            'p95_ms': ordered[min(len(ordered)-1, int(len(ordered)*.95))], 'total_ms': sum(samples)}


def allocation_metrics(paths):
    """Report unavailable disk allocation explicitly instead of logical bytes."""
    blocks = [getattr(path.stat(), 'st_blocks', None) for path in paths]
    unknown = sum(value is None for value in blocks)
    return {'allocated_object_bytes': None if unknown else sum(blocks)*512,
        'allocation_metrics_available': not unknown, 'allocation_unknown_objects': unknown}


def run(baseline_source, sizes):
    with tempfile.TemporaryDirectory(prefix='khquant-store-benchmark-') as temporary:
        root = Path(temporary)
        for name, suffix in {'KHQUANT_DATA_ROOT':'data', 'KHQUANT_ARTIFACT_ROOT':'artifacts',
            'KHQUANT_METADATA_ROOT':'metadata', 'KHQUANT_RAW_DB':'raw.db',
            'KHQUANT_PROCESSED_DB':'warehouse.db', 'KHQUANT_CURRENT_STRATEGY_CONFIG':'strategy.json'}.items():
            os.environ[name] = str(root / suffix)
        from my_strategy.storage.research_store import ResearchStore, canonical
        spec = importlib.util.spec_from_file_location('baseline_research_store', baseline_source)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        report = {'schema':'research_store_benchmark_v1', 'clock':1000.,
            'baseline_sha256': hashlib.sha256(Path(baseline_source).read_bytes()).hexdigest(),
            'new_sha256': hashlib.sha256(Path(__file__).parents[1].joinpath('storage/research_store.py').read_bytes()).hexdigest(),
            'scope':'single CPU process, synthetic CAS data, no business databases or numerical models', 'sizes':[]}
        for count in sizes:
            fixture = root / ('seed-'+str(count))
            with old.ResearchStore(fixture, clock=lambda:1000.) as store:
                with store._tx():
                    for n in range(count):
                        identity, namespace = {'seed':n}, 'seed'+str(n % 3)
                        content = gzip.compress(canonical({'seed':n, 'values':[n, n+1, n+2]}).encode(), mtime=0)
                        sha = hashlib.sha256(content).hexdigest(); key = store.cache_key(identity, namespace)
                        path = store._path(sha); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
                        store.connection.execute("INSERT INTO objects VALUES(?,?,'active')", (sha,len(content)))
                        store.connection.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?)',
                            (key,namespace,canonical(identity),sha,1000.,1000.,100000.))
                        if n % 100 == 0: store.connection.execute('INSERT INTO pins VALUES(?,?,?)',(key,'saved',1000.))
                        if n % 100 == 1: store.connection.execute('INSERT INTO leases VALUES(?,?,?)',(key,'reader',100000.))
            row = {'seed_count':count, 'variants':{}}
            for label, factory in [('before',old.ResearchStore),('after',ResearchStore)]:
                location = root / (label+'-'+str(count)); shutil.copytree(fixture,location)
                opened = time.perf_counter()
                with factory(location,clock=lambda:1000.) as store:
                    observed = {'open_migrate_ms':(time.perf_counter()-opened)*1000}
                    store.gc(False,scan_orphans='auto')
                    sequence = [0]
                    def put():
                        sequence[0]+=1
                        return store.put({'new':sequence[0]},{'value':sequence[0]})
                    observed['single_put'] = timing(put,30)
                    observed['cache_byte_query'] = timing(store._cache_bytes,30)
                    observed['indexed_gc_no_pressure'] = timing(lambda:store.gc(False,scan_orphans='auto'),20)
                    observed['namespace_list_20'] = timing(lambda:store.list('seed2',20),30)
                    trace=[]; store.connection.set_trace_callback(trace.append)
                    observed['hot_get'] = timing(lambda:store.get({'new':1}),100)
                    store.connection.set_trace_callback(None)
                    observed['hot_get_writer_transactions'] = sum(sql=='BEGIN IMMEDIATE' for sql in trace)
                    batch=[0]
                    def publish_batch():
                        batch[0]+=1
                        items=[{'identity':{'batch':batch[0],'part':part},'result':{'batch':batch[0],'part':part}} for part in range(32)]
                        if label=='after':
                            store.put_many(items,lease_owner='build')
                        else:
                            for item in items:
                                ref=store.put(item['identity'],item['result']); store.acquire_lease(ref['cache_key'],'build')
                    observed['publish_32_and_lease'] = timing(publish_batch,3)
                    store.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                    observed['storage']={'sqlite_bytes':(location/'research.db').stat().st_size,
                        'object_count':store.connection.execute('SELECT COUNT(*) FROM objects').fetchone()[0],
                        'logical_object_bytes':store.connection.execute('SELECT SUM(size_bytes) FROM objects').fetchone()[0],
                        **allocation_metrics(store.objects.rglob('*.json.gz'))}
                    observed['sqlite_integrity']=store.connection.execute('PRAGMA integrity_check').fetchone()[0]
                    row['variants'][label]=observed
            row['median_speedup']={name:row['variants']['before'][name]['median_ms']/row['variants']['after'][name]['median_ms']
                for name in ('single_put','cache_byte_query','indexed_gc_no_pressure','namespace_list_20','hot_get','publish_32_and_lease')}
            report['sizes'].append(row)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[1000,5000])
    args=parser.parse_args()
    if any(n<1 or n>10000 for n in args.sizes): parser.error('sizes must be 1..10000')
    report=run(args.baseline_source,args.sizes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
