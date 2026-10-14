"""Independent reference head fitting, real Broker fee evidence and sealing."""
from dataclasses import replace
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.broker import BrokerSimulator
from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.order import Order
from my_strategy.services import czsc_reference_quality as quality
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.services.czsc_trend_support import _save,_file_hash
from my_strategy.services.czsc_account_artifacts import write_account_artifact


def model_rows():
    dates=pd.bdate_range('2023-01-02',periods=400)
    rows=[]
    for i,stamp in enumerate(dates):
        day=stamp.date().isoformat();atom='atom:'+day;label=i%2
        rows.append({**{k:0. for k in quality.COLUMNS},quality.COLUMNS[0]:4. if label else -4.,
            'symbol':'000001.SZ','date':day,'label_end':(stamp+pd.offsets.BDay(3)).date().isoformat(),
            'label':label,'label_available':True,'input_eligible':True,'attempt_id':atom,'economic_group_id':atom,
            'opportunity_id':atom,'original_event_ids':json.dumps([atom]),
            'original_event_first_pre_gate_dates':json.dumps({atom:day})})
    data=quality._frame(rows)
    fold={'name':'reference_unit','train_end':data.date.iloc[209],'calibration_start':data.date.iloc[220],
        'calibration_end':data.date.iloc[279],'policy_start':data.date.iloc[290],'policy_end':data.date.iloc[349],
        'validation_end':data.date.iloc[349]}
    return data,fold


def reference_row(day='2026-10-12',symbol='000001.SZ'):
    at=day+'T15:00:00+08:00';events=[]
    for source in ('ma','structure','wyckoff'):
        events.append({'source':source,'symbol':symbol,'date':day,'side':'BUY','confirmed':True,'input_eligible':True,
            'event_id':source+':event:'+day,'original_event_id':source+':origin:'+day,
            'opportunity_id':source+':opportunity:'+day,'available_at':at})
    ids=sorted(e['opportunity_id'] for e in events)
    plan={'symbol':symbol,'signal_date':day,'available_at':at,'reference_price':10.,'stop_price':9.5,
        'price_floor':9.51,'price_ceiling':10.4,'target_weight':.2,'atr14':.4,'status':'active',
        'entry_allowed':True,'policy':'risk','valid_for_sessions':1,'sources':['ma','structure','wyckoff'],
        'source_event_ids':sorted(e['event_id'] for e in events),'original_opportunity_ids':ids,'plan_id':'plan:'+day,
        'holding_boundaries':[]}
    row={'symbol':symbol,'date':day,'available_at':at,'reference_price':10.,'direction':'BUY',
        'buy_base_score':.65,'sell_base_score':.05,'buy_base_qualified':True,'entry_eligible':True,
        'buy_independent_families':['ma','structure','wyckoff'],'active_candidates':events,
        'original_opportunity_ids':ids,'entry_plan':plan,
        'agents':{'ma':{'status':'scored','p_up':.8,'p_down':.1,'p_neutral':.1,
            'model_identity':{'checkpoint':'unit_trend','manifest_sha256':'unit-trend-sha'}},
            'structure':{'status':'scored'},'wyckoff':{'status':'scored'}}}
    risk={'status':'available','date':day,'available_at':at,'reference_price':10.,'original_buy_plan_id':plan['plan_id'],
        'original_buy_plan_hash':stable_hash(plan),'stop_price':9.5,'risk_position_cap_fraction':.2,
        'risk_budget_fraction':.01,'stop_distance_fraction':.05,'reason_codes':[],
        'reference_price_date':day,'invalidation_price':9.5,'reference_price_band':{'low':9.51,'high':10.4},
        'max_position_fraction':.95}
    return row,risk


def _real_probe(day,next_day,sell_day,symbol,positive,cfg,multiplier):
    costs=CostModel(version='flat_v1',impact_tiers=(),commission=cfg['account']['commission']*multiplier,
        min_commission=cfg['account']['min_fee']*multiplier,stamp_tax=cfg['account']['stamp']*multiplier,
        slippage=cfg['account']['slippage']*multiplier)
    broker=BrokerSimulator(quality.PROBE_CASH,costs,'synthetic-reference-probe')
    buy=round(costs.execution_price('BUY',10.),2)
    sell=round(costs.execution_price('SELL',12. if positive else 9.7),2)
    broker.submit(Order(next_day,symbol,'BUY',1900,buy,signal_date=day),mark_price=10.,equity_before=quality.PROBE_CASH)
    broker.submit(Order(sell_day,symbol,'SELL',1900,sell,signal_date=next_day),mark_price=sell,equity_before=broker.equity({symbol:sell}))
    ledger=broker.ledger.to_frame().to_dict('records')
    assert len(ledger)==2 and ledger[1]['position_after']==0
    from my_strategy.services.czsc_trend_labels import episode_outcome
    from my_strategy.services.czsc_trend_support import _json
    outcome=episode_outcome(ledger,stop_price=9.5,cost_model=costs,config=cfg)
    return {'ledger':ledger,'rejections':[],'cost_model':costs.as_config_dict(),'outcome':_json(outcome),'actual_initial_stop':9.5}


def qualified_reference_fixture(tmp_path,monkeypatch):
    """Synthetic contract fixture with 60 stocks and genuine Ledger fee audit.

    No production selection is created. All qualification metrics are actually
    reconstructed from 1x/2x Broker records and prior fitted probabilities.
    """
    from my_strategy.services import czsc_trend_pipeline
    from my_strategy.services.model_training_center import aggregate_account_metrics,selection_gate
    cfg=strategy_config();cfg.update(source_snapshot_hash='synthetic-source',pipeline_contract_hash='synthetic-pipeline',E1_policy_hash='synthetic-E1')
    cfg['consensus'].update(buy_base_threshold=.45,sell_threshold=.45,margin=.15)
    root=tmp_path/'reference-fixture';root.mkdir()
    calendar=pd.bdate_range('2023-01-02','2026-10-14').strftime('%Y-%m-%d').tolist()
    _save(root/'data_snapshots/calendar.json',{'dates':calendar,'verified':True})
    trends=[{'name':'unit_trend','available_at':'2023-01-01','heads':{'trend_direction':{'manifest_sha256':'unit-trend-sha'}}}]
    protocol=quality.reference_protocol(cfg);records=[];summaries=[];population_frames=[]
    initial_data,fold=model_rows()
    def make_probe(day,next_day,sell_day,symbol,positive):
        row,risk=reference_row(day,symbol);plan=row['entry_plan'];atoms=plan['original_opportunity_ids']
        values={k:0. for k in quality.COLUMNS};values[quality.COLUMNS[0]]=4. if positive else -4.
        values['reference_buy_base_support']=.65
        identifier=symbol+':'+day
        meta={'symbol':symbol,'date':day,'attempt_id':identifier,'economic_group_id':stable_hash(atoms),
            'opportunity_id':stable_hash(atoms),'original_event_ids':json.dumps(atoms),
            'original_event_first_pre_gate_dates':json.dumps({a:day for a in atoms}),
            'risk_identity_hash':stable_hash(plan),'opportunity_hash':stable_hash(atoms),'input_eligible':True}
        accounts={str(multiplier):_real_probe(day,next_day,sell_day,symbol,positive,cfg,multiplier) for multiplier in (1.,2.)}
        return {'probe_id':identifier,'signal_date':day,'scope':quality.POPULATION,'probe_cash':quality.PROBE_CASH,
            'original_plan':plan,'reference_risk':risk,'feature_row':values,'metadata':meta,'accounts':accounts}
    for number in range(60):
        symbol=f'{number+1:06d}.SZ';directory=root/'dataset/reference_quality/stocks'/symbol.replace('.','_');directory.mkdir(parents=True)
        frame=pd.DataFrame({'date':calendar,'symbol':symbol,'close':10.,'source':'tushare','has_trade_price':1})
        featurepath=directory/'source.parquet';frame.to_parquet(featurepath,index=False)
        record={'symbol':symbol,'path':str(featurepath),'sha256':_file_hash(featurepath)};records.append(record)
        probes=[]
        if number==0:
            for i,day in enumerate(initial_data.date):
                index=calendar.index(day)
                probes.append(make_probe(day,calendar[index+1],calendar[index+3],symbol,i%2==1))
        for window in cfg['development_windows']:
            days=[d for d in calendar if window['test_start']<=d<=window['test_end']]
            for index in range(8):
                probes.append(make_probe(days[index*5],days[index*5+1],days[index*5+2],symbol,index%2==0))
        observations=quality._frame([{**p['feature_row'],**p['metadata'],**p['accounts']['1.0']['outcome']} for p in probes])
        observations.to_parquet(directory/'observations.parquet',index=False);population_frames.append(observations)
        write_account_artifact(directory/'probes.json',{'population':quality.POPULATION,'symbol':symbol,'probes':probes})
        files={p.name:{'path':str(p.relative_to(root)),'sha256':_file_hash(p)} for p in directory.iterdir() if p.name!='source.parquet'}
        summaries.append({'symbol':symbol,'files':files,'source_record':record,'source_record_sha256':record['sha256'],'rows':len(probes)})
    monkeypatch.setattr(czsc_trend_pipeline,'read_snapshot',lambda r:(pd.read_parquet(r['path']),{}))
    data=pd.concat(population_frames,ignore_index=True)
    pooled=root/'dataset/reference_quality/observations.parquet';data.to_parquet(pooled,index=False)
    manifest={'population':quality.POPULATION,'protocol':protocol,'protocol_hash':protocol['protocol_hash'],
        'source_snapshot_hash':cfg['source_snapshot_hash'],'pipeline_contract_hash':cfg['pipeline_contract_hash'],
        'continuation_policy_hash':cfg['E1_policy_hash'],'calendar_hash':stable_hash(calendar),'records':summaries,'failures':[],
        'trend_catalog_hash':stable_hash(trends),'ma_exit_catalog_hash':stable_hash([]),
        'pooled_path':str(pooled.relative_to(root)),'pooled_sha256':_file_hash(pooled)}
    manifest['population_hash']=stable_hash(manifest);mpath=root/'dataset/reference_quality/manifest.json';_save(mpath,manifest)
    population={k:manifest[k] for k in ('population_hash','protocol_hash','calendar_hash','trend_catalog_hash','ma_exit_catalog_hash')}
    population.update(manifest_path=str(mpath.relative_to(root)),manifest_sha256=_file_hash(mpath),pooled_path=manifest['pooled_path'],
        pooled_sha256=manifest['pooled_sha256'],symbols=[r['symbol'] for r in records])
    contract=quality._target_contract(cfg,population)
    path=root/'models/reference_quality/reference_unit';m=quality.train_reference_head(data,path,fold,cfg,contract)
    bundles=[{'name':fold['name'],'fold':fold,'available_at':m['available_at'],
        'heads':{quality.HEAD:{'path':str(path.relative_to(root)),'manifest_sha256':m['manifest_sha256']}}}]
    results=[quality._qualification_worker((r,calendar,cfg,str(root),population,bundles,contract)) for r in summaries]
    windows=[{'window':window,'arms':{name:{'metrics':aggregate_account_metrics([r['windows'][i]['arms'][name] for r in results])}
        for name in ('candidate','rules','candidate_stress','rules_stress')}} for i,window in enumerate(cfg['development_windows'])]
    gate=selection_gate(windows,protocol['qualification'],60,False)
    assert gate['passed'],gate['reasons']
    q={'head':quality.HEAD,'population':quality.POPULATION,'population_hash':population['population_hash'],'protocol_hash':protocol['protocol_hash'],
        'contract_hash':contract['contract_hash'],'windows':windows,'assets':[{'symbol':r['symbol'],'files':r['files']} for r in results],
        **gate,'status':'passed','publication_allowed':False,'certification_status':'pending','core_selection_gate_changed':False}
    qpath=root/'validation/reference_quality/qualification.json';_save(qpath,q)
    block={'head':quality.HEAD,'protocol':protocol,'protocol_hash':protocol['protocol_hash'],'population':population,
        'target_contract':contract,'bundles':bundles,'unavailable_heads':[],
        'qualification':{'path':str(qpath.relative_to(root)),'sha256':_file_hash(qpath),'report_hash':stable_hash(q),'status':'passed','passed':True},
        'status':'qualified','publication_allowed':False,'required_for_core_selection':False,'label_counts':quality._label_counts(data)}
    report={'config':cfg,'dataset_records':records,'reference_quality':block,'trend_bundles':trends,'ma_exit_bundles':[]}
    row,risk=reference_row();feature={k:0. for k in quality.COLUMNS};feature[quality.COLUMNS[0]]=4.
    feature.update(symbol=row['symbol'],date=row['date'],model_input_eligible=True,native_structure_input_eligible=True,native_w_input_eligible=True)
    return root,report,cfg,row,risk,feature,{}


def test_qualified_population_real_ledger_gate_and_sealed_score(tmp_path,monkeypatch):
    root,report,cfg,row,risk,feature,prefix=qualified_reference_fixture(tmp_path,monkeypatch)
    session=quality.ReferenceQualitySession(root,report,config=cfg)
    score=session.score(feature,row,prefix,risk_plan=risk)
    assert score.status=='scored' and score.probability>.1
    assert quality.validate_reference_quality_score(score,row,risk,cfg,run_id=root.name)==[]
    assert quality.validate_reference_quality_score(replace(score,probability=1.),row,risk,cfg,run_id=root.name)
    assert quality.validate_reference_quality_score({'verified':True,'probability':1.},row,risk,cfg)
    assert report['reference_quality']['required_for_core_selection'] is False
    altered=copy.deepcopy(row);altered['agents']['ma']['model_identity']['checkpoint']='older-pinned'
    assert session.score(feature,altered,prefix,risk_plan=risk).reason_codes==('reference_quality_trend_route_out_of_population',)
    # A self-rehashed manifest cannot invent its training population counts.
    asset=report['reference_quality']['bundles'][0]['heads'][quality.HEAD]
    manifest_path=root/asset['path']/'manifest.json';original=manifest_path.read_bytes()
    forged=json.loads(original);forged['sample_counts']['train']['rows']+=1
    forged.pop('manifest_sha256');forged['manifest_sha256']=stable_hash(forged);_save(manifest_path,forged)
    changed=copy.deepcopy(report['reference_quality']);changed['bundles'][0]['heads'][quality.HEAD]['manifest_sha256']=forged['manifest_sha256']
    with pytest.raises(ValueError,match='samples or input digests'):
        quality.audit_reference_quality(root,changed,cfg,[r['symbol'] for r in report['dataset_records']],expected_records=report['dataset_records'])
    manifest_path.write_bytes(original)
    forged=json.loads(original);forged['train_label_end']=forged['fold']['calibration_start']
    forged.pop('manifest_sha256');forged['manifest_sha256']=stable_hash(forged);_save(manifest_path,forged)
    with pytest.raises(ValueError,match='label cutoffs'):quality.load_reference_quality_head(manifest_path.parent)
    manifest_path.write_bytes(original)
    # The full-population summary is measured again, even if a forged JSON
    # summary and its SHA are mutually consistent.
    qpath=root/report['reference_quality']['qualification']['path'];q=json.loads(qpath.read_text())
    q['windows'][0]['arms']['candidate']['metrics']['total_return']+=.01
    original_q=qpath.read_bytes();_save(qpath,q);changed=copy.deepcopy(report['reference_quality'])
    changed['qualification'].update(sha256=_file_hash(qpath),report_hash=stable_hash(q))
    with pytest.raises(ValueError,match='summary differs'):
        quality.audit_reference_quality(root,changed,cfg,[r['symbol'] for r in report['dataset_records']],expected_records=report['dataset_records'])
    qpath.write_bytes(original_q)
    # Saved high labels have no authority over real fee-bearing Broker fills.
    from my_strategy.services.czsc_account_artifacts import read_account_artifact
    manifest=json.loads((root/report['reference_quality']['population']['manifest_path']).read_text())
    record=manifest['records'][0];artifact=read_account_artifact(root/record['files']['probes.json']['path'])
    artifact['probes'][0]['accounts']['1.0']['outcome']['label']=1-artifact['probes'][0]['accounts']['1.0']['outcome']['label']
    with pytest.raises(ValueError,match='saved outcome differs'):
        quality._validate_probe_rows(root,record,artifact,pd.read_parquet(record['source_record']['path']),cfg,
            json.loads((root/'data_snapshots/calendar.json').read_text())['dates'])


def test_reference_features_never_read_cash_inventory_or_future_prefix():
    row,risk=reference_row();f={k:1. for k in quality.COLUMNS}
    f.update(symbol=row['symbol'],date=row['date'],model_input_eligible=True,native_structure_input_eligible=True,native_w_input_eligible=True)
    a=quality.reference_feature_row(f,row,{},risk)
    f.update(cash=1e100,equity=-1e100,holding_sessions=999,current_position_fraction=1.)
    b=quality.reference_feature_row(f,row,{},risk)
    assert a.keys()==b.keys() and all(a[k]==b[k] or np.isnan(a[k]) and np.isnan(b[k]) for k in a)
    assert not any('cash' in c or 'account_return' in c or 'holding' in c or 'current_position' in c for c in quality.COLUMNS)


def test_reference_head_fit_cannot_use_policy_or_actual_flat_identity(tmp_path):
    cfg=strategy_config();cfg.update(source_snapshot_hash='s',pipeline_contract_hash='p',E1_policy_hash='e1')
    p={'population_hash':'population','protocol_hash':'protocol','trend_catalog_hash':'trend','ma_exit_catalog_hash':'exit'}
    contract=quality._target_contract(cfg,p);data,fold=model_rows()
    first=quality.train_reference_head(data,tmp_path/'a',fold,cfg,contract)
    data.loc[data.date>=fold['policy_start'],quality.COLUMNS[0]]*=1000
    data.loc[data.date>=fold['policy_start'],'label']=1-data.loc[data.date>=fold['policy_start'],'label']
    second=quality.train_reference_head(data,tmp_path/'b',fold,cfg,contract)
    assert first['preprocess']==second['preprocess'] and first['calibration']==second['calibration']
    assert np.array_equal(joblib.load(tmp_path/'a/model.joblib').coef_,joblib.load(tmp_path/'b/model.joblib').coef_)
    changed=copy.deepcopy(first);changed['population']='actual_flat_pre_gate_daily_attempt_forced_entry_clone'
    changed.pop('manifest_sha256');changed['manifest_sha256']=stable_hash(changed);_save(tmp_path/'a/manifest.json',changed)
    with pytest.raises(ValueError,match='identity mismatch'):quality.load_reference_quality_head(tmp_path/'a')


def test_probe_worker_uses_real_independent_seeds_next_open_and_complete_costed_episodes(tmp_path,monkeypatch):
    from my_strategy.services import czsc_trend_pipeline,czsc_trend_consensus
    from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS,STRUCTURE_COLUMNS
    from my_strategy.services.czsc_price_volume_features import W_INPUT_COLUMNS
    from my_strategy.services.czsc_account_artifacts import read_account_artifact
    days=pd.bdate_range('2025-01-02',periods=95).strftime('%Y-%m-%d').tolist()
    frame=pd.DataFrame({k:0. for k in (*MA_TREND_COLUMNS,*STRUCTURE_COLUMNS,*W_INPUT_COLUMNS)},index=range(len(days)))
    frame=frame.assign(symbol='000001.SZ',date=days,open=10.,high=10.2,low=9.8,close=10.,
        volume=4000000.,amount=40000000.,source='tushare',has_trade_price=1,model_input_eligible=True,
        native_structure_input_eligible=True,native_w_input_eligible=True)
    for index,price in enumerate((10.2,10.5,10.8,11.2,11.6,12.,12.),start=71):
        frame.loc[index,['open','close','high','low']]=[frame.close.iloc[index-1],price,price+.2,min(price,frame.close.iloc[index-1])-.05]
    frame.loc[78:,['open','close','high','low']]=[12.,12.,12.2,11.8]
    frame.attrs.update(volume_unit='shares',amount_unit='CNY')
    cfg=strategy_config();cfg.update(source_snapshot_hash='synthetic-raw-source',pipeline_contract_hash='synthetic-pipeline',
        candidate_policy_hash='synthetic-candidate',E1_policy_hash='synthetic-E1',training_end=days[-1])
    cfg['feature_start']=days[0]
    prepared=[]
    for index,day in enumerate(days):
        row,_=reference_row(day)
        row.update(reference_price=float(frame.close.iloc[index]),entry_plan=None,active_candidates=[],original_opportunity_ids=[],
            direction='WAIT',action='WAIT',entry_eligible=False,buy_base_qualified=False,buy_pass=False,sell_pass=index==76,
            candidate_evidence=[])
        if index==70:
            row,_=reference_row(day)
            for event in row['active_candidates']:
                event.update(pattern_id=event['original_event_id'],risk_stop_price=9.5,invalidation_price=9.5)
            row['entry_plan']=czsc_trend_consensus._risk_plan(frame.iloc[index].to_dict(),row['active_candidates'],index,frame.to_dict('records'),cfg)
            row['original_opportunity_ids']=row['entry_plan']['original_opportunity_ids']
            row.update(action='BUY',buy_pass=True,sell_pass=False,candidate_evidence=copy.deepcopy(row['active_candidates']))
        prepared.append(row)
    pack={'evaluations':[{'date':day,'structure':{'support_facts':[]},'wyckoff':{'support_facts':[]}} for day in days]}
    monkeypatch.setattr(czsc_trend_pipeline,'read_snapshot',lambda record:(frame.copy(),copy.deepcopy(pack)))
    monkeypatch.setattr(czsc_trend_pipeline,'forward_trend',lambda *a,**k:{})
    monkeypatch.setattr(czsc_trend_consensus,'prepare_trend_decisions',lambda *a,**k:{'rows':copy.deepcopy(prepared),'config_hash':'synthetic-consensus','direction_points':[]})
    record={'symbol':'000001.SZ','sha256':'synthetic-stock'}
    result=quality._probe_worker((record,days,cfg,str(tmp_path),[],[]))
    assert result['rows']==1
    artifact=read_account_artifact(tmp_path/result['files']['probes.json']['path'])
    probe=artifact['probes'][0]
    assert artifact['seed_ledgers']=={'1.0':[],'2.0':[]}
    assert probe['seed_state_hashes']['1.0']!=probe['seed_state_hashes']['2.0']
    assert artifact['seed_contexts']['1.0']['config_hash']!=artifact['seed_contexts']['2.0']['config_hash']
    for multiplier in ('1.0','2.0'):
        account=probe['accounts'][multiplier]
        assert account['ledger'][0]['date']==days[71] and account['ledger'][0]['signal_date']==days[70]
        assert account['ledger'][-1]['date']==days[77] and account['ledger'][-1]['position_after']==0
        assert account['outcome']['label_available'] and account['outcome']['label']==1.
        assert account['outcome']['net_return']>=.08 and account['outcome']['r_multiple']>=2.
        bindings=[b['exit_model_binding'] for b in account['exit_model_bindings'] if b['exit_model_binding']]
        assert bindings and all(b['mode']=='explicit_rule_E0_no_prior_exit_head' and b['checkpoint'] is None for b in bindings)
    assert probe['accounts']['2.0']['cost_model']['commission']==2*probe['accounts']['1.0']['cost_model']['commission']
    assert probe['accounts']['2.0']['outcome']['net_return']<probe['accounts']['1.0']['outcome']['net_return']
