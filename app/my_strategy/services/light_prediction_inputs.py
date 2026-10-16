"""Bounded CPU preparation for the account's unchanged serial predictions.

Workers never call a model, scorer, store or Broker. They prepare an exact
load_bars-compatible prefix of the request's already verified raw snapshot.
Only the causal scheduler decides which prepared prefix is actually consumed.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import threading

import pandas as pd

from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
from my_strategy.core.run_context import stable_hash


@dataclass(frozen=True)
class PreparedPredictionInput:
    symbol: str
    as_of: str
    raw: pd.DataFrame
    calendar_dates: tuple[str, ...]
    calendar_identity: dict
    exact_price_input_sha256: str


class AccountPredictionInputs:
    """At most two CPU workers and eight outstanding raw prefixes per request.

    Per-symbol lookahead is only two observed dates. It does not precompute
    forecasts for future dates or change logical quotas and observation order.
    """
    def __init__(self, raw_frames, calendar_dates, calendar_identity, *, stop=None,
                 workers=2, max_pending=8, lookahead=2):
        if workers not in (1, 2) or not workers <= max_pending <= 8 or not 0 <= lookahead <= 2:
            raise ValueError('Invalid bounded prediction input preparation')
        self.raw = {s:f.copy(deep=True) for s,f in raw_frames.items()}
        self.days = {s:pd.to_datetime(f.date).dt.strftime('%Y-%m-%d').tolist() for s,f in self.raw.items()}
        self.calendar = tuple(calendar_dates)
        self.calendar_identity = deepcopy(calendar_identity)
        self.stop, self.max_pending, self.lookahead = stop, max_pending, lookahead
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='account-input')
        self.pending = OrderedDict()
        self.closed = False
        self.lock = threading.Lock()
        self.stats = dict(workers=workers, max_pending=max_pending, peak_pending=0, prepared=0,
                          consumed=0, database_reads=0, model_calls_in_workers=0)

    def _check(self):
        if self.closed or self.stop is not None and self.stop.is_set():
            raise InterruptedError('Account prediction input preparation stopped')

    def _prepare(self, symbol, day):
        self._check()
        raw = self.raw[symbol]
        count = bisect_right(self.days[symbol], day)
        prefix = raw.iloc[:count].copy().reset_index(drop=True)
        if prefix.empty or self.days[symbol][count-1] != day:
            raise ValueError(f'{symbol} 缺少 {day} 行情，不能使用旧日报价')
        # load_bars derives these over the requested prefix, not the final day.
        legacy = int(prefix.source.map(legacy_fuyao_source).sum()) if 'source' in prefix else 0
        prefix.attrs.update(price_basis='mixed_legacy_adjustment_unverified' if legacy else 'unadjusted',
            legacy_fuyao_bars=legacy,
            unverified_trade_price_bars=int((prefix.has_trade_price != 1).sum()) if 'has_trade_price' in prefix else len(prefix),
            volume_unit='shares', amount_unit='CNY')
        prefix.attrs['data_version'] = hashlib.sha256(prefix.to_csv(index=False, float_format='%.10g').encode()).hexdigest()
        from my_strategy.services.czsc_trend_support import _json
        request_sha = stable_hash(_json(prefix.to_dict('records')))
        self._check()
        with self.lock: self.stats['prepared'] += 1
        return PreparedPredictionInput(symbol, day, prefix, self.calendar,
                                       self.calendar_identity, request_sha)

    def _submit(self, symbol, day):
        key = (symbol, day)
        if key in self.pending: return
        while len(self.pending) >= self.max_pending:
            _, future = self.pending.popitem(last=False)
            if not future.cancel():
                try: future.result()
                except (ValueError, InterruptedError): pass  # Unrequested lookahead is not an observation.
        self.pending[key] = self.pool.submit(self._prepare, symbol, day)
        self.stats['peak_pending'] = max(self.stats['peak_pending'], len(self.pending))

    def get(self, symbol, day):
        self._check()
        if symbol not in self.raw: raise ValueError('Prediction symbol outside frozen account input')
        self._submit(symbol, day)
        value = self.pending.pop((symbol, day)).result()
        self._check()
        self.stats['consumed'] += 1
        at = bisect_right(self.days[symbol], day)
        for upcoming in self.days[symbol][at:at+self.lookahead]:
            self._submit(symbol, upcoming)
        return value

    def close(self):
        if self.closed: return
        self.closed = True
        for future in self.pending.values(): future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def account_prediction_callback(service, inputs, parameters, stop, progress=None):
    """Keep progress and model inference on the causal coordinator thread."""
    current = {}
    def report(**value):
        current.clear(); current.update(value)
        if progress: progress(**value)
    def predict(symbol, day):
        if progress: progress(**{**current, 'phase':'input_preparation', 'cpu_workers':inputs.stats['workers']})
        prepared = inputs.get(symbol, day)
        if progress: progress(**{**current, 'phase':'model_inference', 'cpu_workers':inputs.stats['workers']})
        return service.prediction(symbol, day, stop=stop, _prepared_input=prepared, **parameters)
    return predict, report
