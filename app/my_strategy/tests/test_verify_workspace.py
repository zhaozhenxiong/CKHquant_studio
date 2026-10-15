import hashlib
import sqlite3

import pytest

from my_strategy.scripts import verify_workspace as health


@pytest.fixture
def workspace(tmp_path):
    root=tmp_path/'app'
    for name in health.REQUIRED_PATHS:
        path=root/name
        if name.endswith('configs'):path.mkdir(parents=True)
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    return root


def test_modern_empty_workspace_needs_no_legacy_results_or_model_registry(workspace,tmp_path):
    report=health.verify(tmp_path,raw_db=tmp_path/'missing.db',metadata_root=tmp_path/'metadata',fluent_db=tmp_path/'fluent.db')
    assert report['status']=='ok' and report['legacy_registry_required'] is False
    assert report['project_root']==str(workspace)
    assert not (tmp_path/'metadata').exists() and not (tmp_path/'missing.db').exists()


def test_current_raw_coverage_is_read_only_and_rejects_stale_unverified_rows(workspace,tmp_path):
    path=tmp_path/'raw.db'
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER,PRIMARY KEY(stock,date))')
        con.executemany('INSERT INTO stock_daily_normalized(stock,date,source,has_trade_price) VALUES(?,?,?,?)',[
            ('002436.SZ','2026-10-13','tushare',1),('002436.SZ','2026-10-14','tushare',1),
            ('600519.SH','2026-10-15','unverified',1),('600519.SH','2026-10-14','tushare',0),
            ('920001.BJ','2026-10-14','tushare',1)])
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    result=health.verify(workspace,raw_db=path,metadata_root=tmp_path/'metadata',fluent_db=tmp_path/'fluent.db',
        target_date='2026-10-14',min_history=2,strict=True)
    raw=result['raw_database']
    assert result['status']=='ok' and raw['latest_date']=='2026-10-15'
    assert raw['latest_qualified_source_date']=='2026-10-14' and raw['target_sh_sz_a_stocks']==1
    assert raw['warmup_source_qualified_sh_sz_a_stocks']==1 and 'daily_rows' not in raw
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert not (tmp_path/'metadata').exists()


def test_strict_missing_target_fails_without_creating_storage(workspace,tmp_path):
    result=health.verify(workspace,raw_db=tmp_path/'raw.db',metadata_root=tmp_path/'metadata',
        fluent_db=tmp_path/'fluent.db',target_date='2026-10-14',strict=True)
    assert result['status']=='failed'
    assert not (tmp_path/'raw.db').exists()


@pytest.mark.parametrize('options',[{'target_date':'2026-99-99'},{'min_history':0}])
def test_bad_health_inputs_are_rejected(workspace,options):
    with pytest.raises(ValueError):health.verify(workspace,**options)
