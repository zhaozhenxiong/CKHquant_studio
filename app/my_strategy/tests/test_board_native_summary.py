"""Isolated causal cache proofs, independent coverage, and source-only lifecycle."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import board_native_summary as summary


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(summary._json(value))


@pytest.fixture
def proof(tmp_path,monkeypatch):
    from my_strategy.services.czsc_trend_contracts import strategy_config
    from my_strategy.services.czsc_trend_source import _raw_worker,_native_worker
    from my_strategy.services import czsc_trend_source
    from my_strategy.services.czsc_native_cache import NativeStructureCache,make_native_cache_identity
    from my_strategy.adapters.czsc_adapter import engine_contract
    app=tmp_path/'producer';local=summary._app_root()
    for name in summary.NATIVE_CODE_FILES:
        path=app/'my_strategy'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(local/'my_strategy'/name,path)
    actual_engine=engine_contract();monkeypatch.setattr(summary,'_engine_contract',lambda root:actual_engine)
    monkeypatch.setattr(czsc_trend_source,'resolve_frozen_asset',lambda value:Path(value))
    artifacts=tmp_path/'artifacts';raw_id='raw-source-fixture';native_id='native-source-fixture'
    raw_root=artifacts/'runs'/raw_id;native_root=artifacts/'runs'/native_id
    dates=pd.bdate_range('2023-01-02',periods=85).strftime('%Y-%m-%d').tolist();day=dates[-1];stock='000001.SZ'
    warehouse=tmp_path/'raw.db'
    frame=pd.DataFrame({'stock':[stock]*len(dates),'date':dates,'open':100+np.sin(np.arange(85)/4)*3,
        'high':105+np.sin(np.arange(85)/4)*3,'low':95+np.sin(np.arange(85)/4)*3,
        'close':101+np.sin(np.arange(85)/4)*3,'volume':[1_000_000.]*85,'amount':[100_000_000.]*85,
        'source':['tushare']*85,'has_trade_price':[1]*85})
    with sqlite3.connect(warehouse) as con:frame.to_sql('stock_daily_normalized',con,index=False)
    cfg=strategy_config();cfg.update(full_qualified_pool=True,stock_batch_size=360)
    record,error=_raw_worker((stock,day,str(raw_root/'data_snapshots/raw'),str(warehouse),dates,cfg,cfg['feature_start']))
    assert error is None
    result=_native_worker((record,cfg,dates,str(native_root/'data_snapshots/native')))
    generation=summary.native_generation_contract(source_root=app,config=cfg)
    catalog=[dict(symbol=stock,rows=record['raw_rows'],raw_content_sha256=record['raw_content_sha256'],
        data_start=record['raw_data_start'],data_end=record['data_end'],volume_unit='shares',amount_unit='CNY')]
    population=dict(input_catalog=catalog,input_catalog_hash=stable_hash(catalog),candidate_symbols=[stock],requested_symbols=None)
    population['population_hash']=stable_hash(population)
    raw_report=dict(run_id=raw_id,status='completed',failures=[],raw_records=[record],selected_symbols=[stock],
        data_end=day,dates=dates,liquidity=population)
    raw_report['snapshot_hash']=stable_hash(raw_report)
    raw_report_path=raw_root/'reports/raw_source.json';write_json(raw_report_path,raw_report)
    identity=dict(strategy_version='trend_consensus',raw_source_run_id=raw_id,
        raw_report_sha256=summary._file_sha(raw_report_path),raw_snapshot_hash=raw_report['snapshot_hash'],
        raw_records_hash=stable_hash([record]),config_hash=stable_hash(cfg),native_config_hash=generation['native_config_hash'],
        native_schema_hash=generation['native_schema_hash'],native_input_contract_hash=stable_hash(dict(
            feature_start=cfg['feature_start'],liquidity_universe=cfg['liquidity_universe'],
            native_schema_hash=generation['native_schema_hash'],native_config_hash=generation['native_config_hash'])),
        code_hashes=generation['code_hashes'],calendar_hash=stable_hash(dates),population_hash=population['population_hash'],
        selected_symbols=[stock],cpu_workers=1,prepare_batch_size=360,engine_contract=actual_engine,
        feature_schema_hashes=generation['feature_schema_hashes'])
    header=dict(format='native_source_checkpoint',run_id=native_id,config=cfg,identity=identity,created_at='2023-05-01T16:00:00+08:00')
    header['checkpoint_hash']=stable_hash(header);write_json(native_root/'reports/native_checkpoint.json',header)
    report=dict(run_id=native_id,status='completed',failures=[],data_end=day,native_checkpoint_hash=header['checkpoint_hash'],dataset_records=[result])
    write_json(native_root/'reports/native_source.json',report)
    write_json(native_root/'reports/native_prepare_status.json',{'status':'completed'})
    write_json(native_root/'metadata.json',dict(task='trend-native-source',as_of_date=day))
    cache_path=tmp_path/'native_structure.db'
    cache_identity=make_native_cache_identity(record,identity,dates)
    with NativeStructureCache(cache_path) as cache:cache.store(cache_identity,result)
    kwargs=dict(price_date=day,source_root=app,artifacts_root=artifacts,source_native_run_id=native_id,
        raw_source_run_id=raw_id,raw_db_path=warehouse,cache_db_path=cache_path,symbols=[stock])
    return dict(kwargs=kwargs,record=record,result=result,header=header,raw_report=raw_report,
        app=app,cache=cache_path,warehouse=warehouse,day=day,stock=stock,artifacts=artifacts,native_root=native_root)


def test_real_isolated_native_cache_target_summary_and_dual_proofs(proof):
    value=summary.collect_native_summaries(**proof['kwargs'])
    assert value['complete'] is True and value['diagnostics']['verified']==1,value['diagnostics']
    row=value['summaries'][proof['stock']]
    assert row['training_eligible'] is False and row['status']=='available'
    assert row['native_cache_key'] and row['features_sha256']==proof['result']['sha256']
    assert row['native_sha256']==proof['result']['native_sha256']
    assert row['row_hash']==stable_hash({k:v for k,v in row.items() if k!='row_hash'})
    pack=json.loads(gzip.decompress(Path(proof['result']['native_path']).read_bytes()))
    native=pack['evaluations'][-1]
    assert row['structure']['direction_state']==native['structure']['direction_state']
    assert row['wyckoff']['active_buy_event_ids']==native['wyckoff']['active_buy_event_ids']
    assert row['structure']['reason_codes']==native['structure']['reason_codes']
    assert summary.validate_native_reuse(value['identity'],raw_db_path=proof['warehouse'],artifacts_root=proof['artifacts'])['valid'] is True


def test_default_discovery_current_completed_source_and_outside_population(proof):
    kwargs=dict(proof['kwargs']);kwargs.pop('source_native_run_id');kwargs['symbols']=[proof['stock'],'600000.SH']
    value=summary.collect_native_summaries(**kwargs)
    assert value['complete'] is True,value['diagnostics']
    missing=value['summaries']['600000.SH']
    assert missing['structure'] is None and missing['native_sha256'] is None
    assert missing['reason_codes']==['native_stock_outside_frozen_population']


def test_historical_storage_producer_requires_exact_archived_bytes(proof):
    name='services/czsc_native_cache.py'
    live=proof['app']/'my_strategy'/name
    old=live.read_bytes();digest=hashlib.sha256(old).hexdigest()
    archived=proof['app']/'my_strategy/data/metadata/producer_snapshots'/digest/name
    archived.parent.mkdir(parents=True);archived.write_bytes(old)
    live.write_bytes(old+b'\n# storage-only deployment update\n')
    value=summary.collect_native_summaries(**proof['kwargs'])
    assert value['complete'],value['diagnostics']
    assert value['identity']['generation_contract']['code_hashes'][name]==digest
    assert value['identity']['generation_contract']['historical_storage_producer_paths'][name]==str(archived)
    archived.write_bytes(b'wrong archived producer')
    unavailable=summary.collect_native_summaries(**proof['kwargs'])
    assert not unavailable['complete']


@pytest.mark.parametrize('change',['blob','metadata','producer','identity','schema_version','missing_entry','frozen_raw','liquidity','warehouse_old_bar','native_report','checkpoint','code'])
def test_expected_stock_integrity_drift_never_publishes_complete_or_zero(proof,change):
    if change in {'blob','metadata','producer','identity','schema_version','missing_entry'}:
        with sqlite3.connect(proof['cache']) as con:
            if change=='blob':con.execute("UPDATE native_structure_entries SET native_blob=?",[b'wrong'])
            elif change=='metadata':con.execute("UPDATE native_structure_entries SET metadata_json='{}'")
            elif change=='producer':con.execute("UPDATE native_structure_entries SET producer_json='{}'")
            elif change=='identity':con.execute("UPDATE native_structure_entries SET identity_json='{}'")
            elif change=='schema_version':con.execute('PRAGMA user_version=3')
            else:con.execute('DELETE FROM native_structure_entries')
    elif change=='frozen_raw':Path(proof['record']['raw_path']).write_bytes(b'wrong')
    elif change=='liquidity':Path(proof['record']['liquidity_path']).write_bytes(b'wrong')
    elif change=='warehouse_old_bar':
        with sqlite3.connect(proof['warehouse']) as con:con.execute("UPDATE stock_daily_normalized SET amount=amount+1 WHERE date=(SELECT MIN(date) FROM stock_daily_normalized)")
    elif change=='native_report':
        report_path=proof['native_root']/'reports/native_source.json';report=json.loads(report_path.read_text());report['dataset_records'][0]['sha256']='0'*64;write_json(report_path,report)
    elif change=='checkpoint':
        path=proof['native_root']/'reports/native_checkpoint.json';header=json.loads(path.read_text());header['identity']['calendar_hash']='0'*64;write_json(path,header)
    else:
        path=proof['app']/'my_strategy/services/czsc_native_cache.py';path.write_text(path.read_text()+'\n# changed producer\n')
    value=summary.collect_native_summaries(**proof['kwargs'])
    assert value['complete'] is False
    row=value['summaries'][proof['stock']]
    assert row['status']=='unavailable' and row['structure'] is None and row['wyckoff'] is None


def test_old_prefix_revision_invalidates_indexed_reuse_without_native_blob_decode(proof,monkeypatch):
    value=summary.collect_native_summaries(**proof['kwargs']);assert value['complete']
    monkeypatch.setattr(summary,'_read_cache_entry',lambda *args:pytest.fail('reuse must not decode native blobs'))
    assert summary.validate_native_reuse(value['identity'],raw_db_path=proof['warehouse'],artifacts_root=proof['artifacts'])['valid']
    with sqlite3.connect(proof['warehouse']) as con:con.execute("UPDATE stock_daily_normalized SET amount=amount+1 WHERE date=(SELECT MIN(date) FROM stock_daily_normalized)")
    assert not summary.validate_native_reuse(value['identity'],raw_db_path=proof['warehouse'],artifacts_root=proof['artifacts'])['valid']


def test_calendar_identity_cannot_drift_from_actual_frozen_source(proof):
    value=summary.collect_native_summaries(**proof['kwargs']);assert value['complete']
    identity=value['identity']
    assert identity['calendar_hash']==proof['header']['identity']['calendar_hash']
    assert identity['calendar_run_id'] is None
    drift=copy.deepcopy(identity);drift['calendar_hash']='0'*64
    result=summary.validate_native_reuse(drift,raw_db_path=proof['warehouse'],artifacts_root=proof['artifacts'])
    assert not result['valid'] and result['native_blob_decode'] is False
    assert 'manifest contract changed' in result['reason']


def test_cancellation_propagates_and_progress_reports_read_only_stages(proof):
    events=[]
    result=summary.collect_native_summaries(**proof['kwargs'],progress=lambda **value:events.append(value))
    assert result['complete'] and {v['stage'] for v in events}=={'native_raw_identity','native_summary'}
    class Stop(RuntimeError):pass
    def cancel():raise Stop('cancel requested')
    with pytest.raises(Stop):summary.collect_native_summaries(**proof['kwargs'],check_cancel=cancel)


def source_row(symbol,day,structure=True,wyckoff=True):
    def part(eligible):return dict(input_eligible=eligible,direction_state='BUY',buy_score=.7 if eligible else None,
        sell_score=.1 if eligible else None,active_buy_event_ids=[symbol+':original'] if eligible else [],
        active_sell_event_ids=[],reason_codes=[] if eligible else ['warmup'])
    row=dict(symbol=symbol,price_date=day,training_eligible=False,status='available',structure=part(structure),wyckoff=part(wyckoff))
    row['row_hash']=stable_hash(row);return row


def test_aggregate_source_eligibility_coverage_and_original_ids_are_independent():
    members=[f'{i:06d}.SZ' for i in range(1,7)];day='2026-10-12'
    snapshot=dict(board_id='THS_I_881155.TI',members=members,expected_count=6,complete=True,snapshot_hash='a'*64,available_at='2026-10-14T15:00:00+08:00')
    native=dict(identity=dict(price_date=day),complete=True,summaries={symbol:source_row(symbol,day,structure=i<5,wyckoff=i<3) for i,symbol in enumerate(members)})
    row=summary.aggregate_board_native(snapshot,native)
    assert row['structure_available'] and row['structure_coverage']==5/6 and row['structure_buy_score_mean']==.7
    assert not row['wyckoff_available'] and row['wyckoff_coverage']==.5 and row['wyckoff_buy_score_mean'] is None
    assert row['status']=='partial' and row['training_eligible'] is False
    assert row['structure_active_buy_event_ids']==sorted(symbol+':original' for symbol in members[:5])
    native['complete']=False
    assert summary.aggregate_board_native(snapshot,native)['structure_buy_score_mean'] is None
    native['summaries'][members[0]]['structure']['buy_score']=1
    with pytest.raises(ValueError,match='row identity'):summary.aggregate_board_native(snapshot,native)


def test_missing_database_contract_never_creates_files(tmp_path):
    value=summary.collect_native_summaries(price_date='2026-10-12',symbols=['000001.SZ'],source_root=tmp_path/'missing')
    assert not value['complete'] and value['summaries']['000001.SZ']['structure'] is None
    assert not (tmp_path/'missing').exists()


def test_refresh_cli_is_source_only_and_producer_root_is_a_fresh_worker(tmp_path,monkeypatch,capsys):
    from my_strategy.scripts import refresh_native_cache as cli
    from my_strategy.services import czsc_trend_source as source
    calls=[]
    monkeypatch.setattr(cli,'_validate_raw_source',lambda *a:None)
    monkeypatch.setattr(source,'prepare_native_source',lambda raw,**kw:(calls.append((raw,kw)) or dict(status='completed',data_end='2026-10-12',failures=[],run_id='native-done',dataset_records=[{}])))
    args=cli.build_parser().parse_args(['--end','2026-10-12','--raw-source-run-id','raw-done','--cache-db',str(tmp_path/'cache.db')])
    result=cli.run_preparation(args)
    assert result['training_started'] is False and result['source_native_run_id']=='native-done'
    assert calls[0][0]=='raw-done' and calls[0][1]['cache_db_path']==tmp_path/'cache.db'
    assert 'NATIVE_CACHE_RESULT ' in capsys.readouterr().out
    monkeypatch.setattr(cli.subprocess,'call',lambda command,**kw:(calls.append((command,kw)) or 0))
    assert cli.main(['--end','2026-10-12','--raw-source-run-id','raw-done','--artifacts-root',str(tmp_path/'artifacts')])==0
    command,opts=calls[-1]
    assert command[-1]=='--_worker' and opts['env']['KHQUANT_ARTIFACT_ROOT']==str(tmp_path/'artifacts')
    assert opts['env']['PYTHONPATH']==str(cli._application())


@pytest.mark.parametrize('source_option',['--source-native-run-id','--resume-native-run-id'])
def test_ready_source_checks_requested_calendar_and_current_full_raw(proof,monkeypatch,capsys,source_option):
    from my_strategy.scripts import refresh_native_cache as cli
    from my_strategy.services import czsc_trend_source as source,czsc_trend_support as support
    from my_strategy.core import paths
    header=proof['header'];dates=proof['raw_report']['dates'];native_id=header['run_id']
    monkeypatch.setattr(cli,'_summary_reader',lambda:summary)
    monkeypatch.setattr(source,'verify_native_checkpoint',lambda *a,**kw:dict(
        header=header,raw_source_run_id=proof['kwargs']['raw_source_run_id'],completed_report=True))
    monkeypatch.setattr(source,'prepare_native_source',lambda *a,**kw:pytest.fail('completed recovery must not prepare'))
    monkeypatch.setattr(paths,'ARTIFACT_RUNS_ROOT',proof['artifacts']/'runs')
    monkeypatch.setattr(support,'_calendar',lambda run:(dates+['2023-05-01'],'explicit-calendar'))
    args=cli.build_parser().parse_args(['--end',proof['day'],'--source-root',str(proof['app']),
        '--artifacts-root',str(proof['artifacts']),'--raw-db',str(proof['warehouse']),
        source_option,native_id,'--calendar-run-id','explicit-calendar'])
    result=cli.run_preparation(args)
    assert result['source_native_run_id']==native_id and result['training_started'] is False
    assert 'NATIVE_CACHE_RESULT ' in capsys.readouterr().out
    monkeypatch.setattr(support,'_calendar',lambda run:(dates[1:],'wrong-calendar'))
    with pytest.raises(ValueError,match='requested verified calendar'):cli.run_preparation(args)
    monkeypatch.setattr(support,'_calendar',lambda run:(dates,'explicit-calendar'))
    with sqlite3.connect(proof['warehouse']) as con:
        con.execute('UPDATE stock_daily_normalized SET close=close+1 WHERE date=(SELECT MIN(date) FROM stock_daily_normalized)')
    with pytest.raises(ValueError,match='full candidate raw prefix drift'):cli.run_preparation(args)


def test_loaded_generating_semantics_cannot_be_disguised_by_business_root(proof,monkeypatch):
    monkeypatch.setattr(summary,'PROJECT_ROOT',proof['app'])
    path=proof['app']/'my_strategy/services/czsc_liquidity.py';path.write_text(path.read_text()+'\n# changed actual semantic source\n')
    with pytest.raises(ValueError,match='reader generating semantics differ'):
        summary.native_generation_contract(source_root=proof['app'])
