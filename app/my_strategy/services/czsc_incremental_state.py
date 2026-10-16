"""Explicit, versioned native state; no pickle or executable deserialization.

The frozen CZSC runtime only exports retained raw bars. Restoration is admitted
only when reconstructing those bars reproduces its complete exposed geometry.
An unsupported retained state fails closed so callers can replay one stock.
"""
from __future__ import annotations

from datetime import datetime
import copy

from my_strategy.adapters.czsc_adapter import native_runtime, engine_contract
from my_strategy.core.run_context import stable_hash

VERSION = 'czsc_explicit_state_v1'


class StateNotRestorable(ValueError):
    pass


def bar_record(bar):
    return dict(symbol=str(bar.symbol), dt=bar.dt.isoformat(), freq=str(bar.freq), id=int(bar.id),
                open=float(bar.open), high=float(bar.high), low=float(bar.low), close=float(bar.close),
                vol=float(bar.vol), amount=float(bar.amount))


def restore_bar(value):
    if set(value) != {'symbol','dt','freq','id','open','high','low','close','vol','amount'}:
        raise StateNotRestorable('native bar state schema mismatch')
    czsc=native_runtime()
    freq={'日线':czsc.Freq.D,'周线':czsc.Freq.W,'月线':czsc.Freq.M}.get(value['freq'])
    if freq is None: raise StateNotRestorable('unsupported native state frequency')
    return czsc.RawBar(**{**value,'dt':datetime.fromisoformat(value['dt']),'freq':freq})


def geometry(analysis):
    """Include retained inputs, forming bars, every pen and its dependencies."""
    def pen(item):
        return {'sdt':item.sdt.isoformat(),'edt':item.edt.isoformat(),'direction':str(item.direction),
                'high':float(item.high),'low':float(item.low),
                'raw':[bar_record(bar) for bar in item.raw_bars]}
    return {'raw':[bar_record(bar) for bar in analysis.bars_raw],
            'pens':[pen(item) for item in analysis.bi_list],
            'finished':[pen(item) for item in analysis.finished_bis],
            'ubi':[[bar_record(raw) for raw in bar.raw_bars] for bar in analysis.bars_ubi],
            'zones':[[item.sdt.isoformat(),item.edt.isoformat(),float(item.zd),float(item.zg)] for item in analysis.zs_list]}


def dump_session(session):
    if session.collect_confirmations or session.position is not None:
        raise StateNotRestorable('state export is restricted to causal evidence sessions')
    value={'format':VERSION,'engine':engine_contract(),'symbol':session.symbol,
           'config':copy.deepcopy(session.config),
           'bg':{freq:[bar_record(bar) for bar in bars] for freq,bars in session.bg.bars.items()},
           'kas':{freq:{'raw':[bar_record(bar) for bar in item.bars_raw],
                        'geometry_hash':stable_hash(geometry(item))} for freq,item in session.kas.items()},
           # advance_structure consumes only the last two closed weekly prices.
           'closed':{freq:[bar_record(bar) for bar in bars[-2:]] for freq,bars in session.closed.items()},
           'pending':{freq:bar_record(bar) for freq,bar in session.pending.items()}}
    value['sha256']=stable_hash(value)
    return value


def restore_session(value, *, cancel_check=None):
    from my_strategy.services.czsc_analysis import CzscSession
    if (value.get('format')!=VERSION or value.get('engine')!=engine_contract() or
        value.get('sha256')!=stable_hash({k:v for k,v in value.items() if k!='sha256'})):
        raise StateNotRestorable('native state schema, engine or SHA mismatch')
    session=CzscSession(value['symbol'],value['config'],0,collect_confirmations=False,create_position=False)
    # Only invoke the known frozen class with validated data; never call an
    # externally supplied constructor or pickle reduction.
    session.bg.__setstate__({'market':'A股','freq_bars':{
        freq:[restore_bar(bar) for bar in bars] for freq,bars in value['bg'].items()}})
    for freq,item in value['kas'].items():
        if cancel_check:
            if hasattr(cancel_check,'check'):cancel_check.check()
            elif cancel_check.is_set() if hasattr(cancel_check,'is_set') else cancel_check():
                raise InterruptedError('原生状态恢复已取消')
        analysis=session.czsc.CZSC([restore_bar(bar) for bar in item['raw']],
            min_bi_len=int(session.config['min_bi_len']),max_bi_num=int(session.config['max_bi_num']))
        if stable_hash(geometry(analysis))!=item['geometry_hash']:
            raise StateNotRestorable('retained native geometry cannot be restored exactly')
        session.kas[freq]=analysis
    session.closed={freq:[restore_bar(bar) for bar in bars] for freq,bars in value['closed'].items()}
    session.pending={freq:restore_bar(bar) for freq,bar in value['pending'].items()}
    return session
