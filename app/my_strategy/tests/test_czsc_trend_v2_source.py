"""The new snapshot boundary preserves actual raw basis and runtime semantics."""
import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_trend_v2_source as source
from my_strategy.services.czsc_trend_v2_contracts import strategy_config
from my_strategy.services.czsc_research import _file_hash, _json


def test_cohort_uses_only_preregistered_prior_input_coverage(tmp_path):
    db=tmp_path/'raw.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,source TEXT,has_trade_price INTEGER)')
        for symbol,size,origin in [('000001.SZ',65,'sina_unadjusted_v1'),('600519.SH',64,'sina_unadjusted_v1'),
                                   ('002436.SZ',90,'legacy_unverified'),('300001.SZ',90,'sina_unadjusted_v1')]:
            conn.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,1)',
                [(symbol,str(day.date()),origin) for day in pd.bdate_range('2022-01-03',periods=size)])
    assert source.select_cohort(1,db_path=db)==['000001.SZ']
    with pytest.raises(ValueError,match='coverage'):
        source.select_cohort(1,db_path=db,preferred=['600519.SH'])
    with pytest.raises(ValueError,match='registered cohort'):
        source.select_cohort(2,db_path=db)


@pytest.mark.parametrize('legacy',[False,True])
def test_fresh_native_projection_matches_real_runtime_price_basis(monkeypatch,tmp_path,legacy):
    from my_strategy.services import czsc_research_runtime
    from my_strategy.web_dashboard.high_return import _prepare
    from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
    from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS
    dates=pd.bdate_range('2024-01-02',periods=280)
    close=20+np.sin(np.arange(len(dates))*.18)*2+np.arange(len(dates))*.008
    raw=pd.DataFrame({'symbol':'002436.SZ','date':dates,'open':close-.12,'high':close+.35,'low':close-.4,
        'close':close,'volume':500000.,'amount':close*500000.,'source':'sina_unadjusted_v1','has_trade_price':1})
    if legacy:raw.loc[0,'source']='fuyao_legacy_qfq'
    attrs={'price_basis':'mixed_legacy_adjustment_unverified' if legacy else 'unadjusted',
        'legacy_fuyao_bars':int(legacy),'unverified_trade_price_bars':0,'volume_unit':'shares','amount_unit':'CNY',
        'data_version':'frozen-raw-observations'}
    raw.attrs=attrs; observed=raw.copy();observed['future_label']=99
    monkeypatch.setattr(source,'load_bars',lambda *args,**kwargs:copy.deepcopy(observed))
    record,error=source._raw_worker(('002436.SZ',str(dates[-1].date()),str(tmp_path/'raw'),None))
    assert error is None and record['raw_attrs']==attrs
    saved=pd.read_parquet(record['raw_path']);assert 'future_label' not in saved
    assert saved.attrs=={}
    monkeypatch.setattr(czsc_research_runtime,'load_bars',lambda *args,**kwargs:copy.deepcopy(raw))
    calendar=[str(day.date()) for day in dates]
    expected=_prepare('002436.SZ',calendar[-1],calendar)
    expected_pack=build_native_evidence_v4(expected['features'],strategy_config(),market_dates=calendar)
    result=source._native_worker((record,strategy_config(),calendar,str(tmp_path/'native')))
    actual=pd.read_parquet(result['path']);pack=json.loads(open(result['native_path']).read())
    fields=[*MA_TREND_COLUMNS,'model_input_eligible','structure_input_eligible','wyckoff_input_eligible']
    pd.testing.assert_frame_equal(actual[fields],expected_pack['frame'][fields],check_dtype=False)
    assert pack['schema_hash']==expected_pack['schema_hash'] and pack['config_hash']==expected_pack['config_hash']
    assert pack==_json({key:value for key,value in expected_pack.items() if key!='frame'})
    assert result['raw_attrs']==attrs and _file_hash(Path(result['raw_path']))==record['raw_sha256']
    if legacy:assert not actual.wyckoff_input_eligible.any()
