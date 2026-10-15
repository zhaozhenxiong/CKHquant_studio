"""Bounded post-update work; never resumes a historical full-market job."""
from __future__ import annotations
import time

from my_strategy.adapters.czsc_adapter import latest_market_date
from my_strategy.core.paths import METADATA_ROOT
from my_strategy.storage.personal_portfolio import PersonalStore


class _Cancellation:
    def __init__(self, callback): self.callback = callback
    def is_set(self): return bool(self.callback and self.callback())


def sync_light_daily(*, raw_db_path=None, required_price_date=None, symbols=None,
                     max_symbols=20, budget_seconds=300, progress=None, check_cancel=None,
                     service=None):
    from my_strategy.services.light_research import LightResearch
    latest = latest_market_date(db_path=raw_db_path)
    date = required_price_date or latest
    if date != latest:
        return {'status':'blocked','reason':'requested date is not latest verified raw date',
                'price_date':date,'training_started':False}
    own = service is None
    service = service or LightResearch(root=METADATA_ROOT/'light_research',raw_db=raw_db_path)
    if symbols is None:
        watch = PersonalStore(METADATA_ROOT/'personal_portfolio.db').watchlist()
        symbols = [r['symbol'] for r in watch.get('items',watch.get('symbols',[])) if isinstance(r,dict)]
    # Every requested symbol remains visible, including those outside this budget.
    symbols = list(dict.fromkeys(symbols or []))
    started = time.monotonic(); deadline = started + budget_seconds
    stop = _Cancellation(check_cancel); rows = []; account_rows = []; cancelled = False; failure = None
    try:
        if stop.is_set(): raise InterruptedError('每日轻量续算已取消')
        # Existing accounts and their hard risk exits always precede watch previews.
        # The prediction deadline never truncates the population of held symbols.
        account_rows = service.continue_accounts(date,stop=stop,deadline=deadline,
            progress=(lambda **value:progress(value)) if progress else None)
        if stop.is_set(): raise InterruptedError('每日轻量续算已取消')
        for index, symbol in enumerate(symbols):
            if stop.is_set(): raise InterruptedError('每日轻量续算已取消')
            if index >= max_symbols or time.monotonic() > deadline:
                rows.append({'symbol':symbol,'status':'budget_skipped'}); continue
            if progress: progress({'stage':'light_daily','symbol':symbol,'completed':len(rows),'total':len(symbols)})
            try:
                _, _, _, identity = service.source(symbol,date,stop=stop)
                if stop.is_set(): raise InterruptedError('每日轻量续算已取消')
                if time.monotonic() > deadline:
                    rows.append({'symbol':symbol,'status':'budget_skipped'}); continue
                prediction = service.prediction(symbol,date,stop=stop,samples=8,lookback=128,horizon=5,device='auto')
                if stop.is_set(): raise InterruptedError('每日轻量续算已取消')
                rows.append({'symbol':symbol,'status':'completed','source':identity,
                             'prediction_key':prediction.get('cache_key'),'reused':prediction.get('reused'),
                             'prediction_usable':prediction.get('quality',{}).get('prediction_usable',False)})
            except InterruptedError: raise
            except Exception as exc:
                rows.append({'symbol':symbol,'status':'failed','error':str(exc)})
    except InterruptedError:
        cancelled = True
    except Exception as exc:
        failure = str(exc)
    finally:
        completed_symbols = {row['symbol'] for row in rows}
        rows.extend({'symbol':s,'status':'cancelled' if cancelled else 'budget_skipped'}
                    for s in symbols if s not in completed_symbols)
        succeeded = sum(r['status']=='completed' for r in rows)
        failed = sum(r['status']=='failed' for r in rows)
        deferred = len(rows)-succeeded-failed
        result = {'status':'cancelled' if cancelled else 'failed' if failure else 'partial' if failed or deferred or
                  any(r['status']!='completed' for r in account_rows) else 'completed',
            'mode':'lightweight','price_date':date,'error':failure,'requested':len(symbols),
            'succeeded':succeeded,'failed':failed,'deferred':deferred,
            'failed_or_deferred':failed+deferred,'rows':rows,'accounts':account_rows,
            'accounts_processed_first':True,'full_market_native_started':False,
            'cancelled_tasks_resumed':False,'training_started':False,'board_full_published':False,
            'duration_seconds':round(time.monotonic()-started,3)}
        service.store.set_setting('last_daily',result)
        if own: service.close()
    return result
