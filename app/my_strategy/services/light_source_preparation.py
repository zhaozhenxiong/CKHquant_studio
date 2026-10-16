"""Bounded spawn preparation before the unchanged single-owner account replay.

The source function, raw/calendar admission, native state, producer identities
and frozen population order are unchanged. Only exact original source methods
are eligible. Injected/custom services retain their original serial execution.
"""
from __future__ import annotations

import inspect
import time


def prepare_account_sources(light, symbols, end, *, stop, progress=None, stats=None):
    from my_strategy.services.light_research import LightResearch
    from my_strategy.services.method_compute import source_workers
    started=time.monotonic()
    stats = stats if stats is not None else {}
    original = getattr(light.source, '__func__', None) is LightResearch.source
    original = original and inspect.getsourcefile(light.source) == inspect.getsourcefile(LightResearch)
    workers = source_workers(population=len(symbols)) if original else 1
    stats.update(cpu_workers=workers,source_tasks_received=0,source_tasks_submitted=0,
        scheduling='spawn_fixed_population' if workers>1 else 'original_serial')
    if workers == 1:
        for symbol in symbols:
            stop.check()
            if progress: progress(stage='structure',symbol=symbol,cpu_workers=1)
            stats['source_tasks_submitted'] += 1
            value=light.source(symbol,end,stop=stop)
            stats['source_tasks_received'] += 1
            yield symbol,value
        stats['elapsed_seconds']=time.monotonic()-started
        return
    from my_strategy.services.analysis_methods import scan_methods
    from my_strategy.services.method_scan_acceleration import _PreparationExecutor, _Execution
    execution = _Execution(scan_methods)
    pool = _PreparationExecutor(light,execution,max_workers=workers)
    pending = {}; iterator = iter(symbols)
    def fill():
        while len(pending) < workers:
            symbol = next(iterator, None)
            if symbol is None: break
            pending[symbol] = pool.submit_source(symbol,end,stop)
    try:
        fill()
        while pending:
            symbol = next(iter(pending)); future = pending[symbol]
            if progress: progress(stage='structure',symbol=symbol,cpu_workers=workers,
                source_tasks_received=execution.received,source_tasks_submitted=execution.submitted)
            while not future.done():
                stop.check()
                try: value = future.result(timeout=.2)
                except TimeoutError: continue
            value = future.result()
            pending.pop(symbol);fill()
            yield symbol, value
    finally:
        for future in pending.values():future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        stats.update(execution.report(),scheduling='spawn_fixed_population',
            elapsed_seconds=time.monotonic()-started)
