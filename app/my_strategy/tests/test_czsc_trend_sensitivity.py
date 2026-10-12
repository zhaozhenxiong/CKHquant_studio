from pathlib import Path
from types import SimpleNamespace
import json

from my_strategy.scripts import train_trend_sensitivity as sensitivity
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_model import build_trend_labels
from my_strategy.tests.test_czsc_trend_model import _frame


def test_sensitivity_rebuilds_H20_labels_and_reports_unqualified_heads_without_fake_scores(tmp_path,monkeypatch):
    frame,calendar=_frame(110)
    frame.loc[80,'close']=110.
    cfg=strategy_config({'feature_start':calendar[0]})
    cfg.update(source_snapshot_hash='frozen-source',pipeline_contract_hash='main-protocol')
    main=tmp_path/'main';(main/'data_snapshots').mkdir(parents=True)
    (main/'dataset/trend').mkdir(parents=True)
    (main/'data_snapshots/calendar.json').write_text(json.dumps({'dates':calendar}))
    h10=build_trend_labels(frame,calendar,cfg,source_snapshot_hash='frozen-source',pipeline_contract_hash='main-protocol')
    h10.to_parquet(main/'dataset/trend/observations.parquet',index=False)
    report={'config':cfg,'dataset_records':[{'symbol':'TEST.SZ'}],'data_end':calendar[-1],
        'source_snapshot':{'snapshot_hash':'frozen-source'},'trend_bundles':[]}
    monkeypatch.setattr(sensitivity,'read_run',lambda rid:(main,report))
    monkeypatch.setattr(sensitivity,'read_snapshot',lambda record:(frame.copy(),{}))
    control=tmp_path/'control';control.mkdir()
    def subdir(*parts):
        path=control.joinpath(*parts);path.mkdir(parents=True,exist_ok=True);return path
    context=SimpleNamespace(run_id='H20-unit',run_dir=lambda:control,subdir=subdir,
        write_metadata=lambda metadata:None)
    monkeypatch.setattr(sensitivity,'create_run_context',lambda **kwargs:context)
    fold={'name':'trend_unit','train_end':calendar[70],'calibration_start':calendar[71],
        'calibration_end':calendar[89],'policy_start':calendar[90],'policy_end':calendar[-1]}
    monkeypatch.setattr(sensitivity,'chronological_folds',lambda *args:[fold])
    captured={}
    def fit(data,path,fold,cfg,*,contract):
        captured['labels']=data.copy();captured['contract']=contract
        assert set(data.horizon_sessions)=={20}
        assert contract['parameters']['horizon_sessions']==20
        assert contract['pipeline_contract_hash']!='main-protocol'
        assert data.loc[data.date==calendar[70],'label_class'].iloc[0]=='NEUTRAL'
        assert h10.loc[h10.date==calendar[70],'label_class'].iloc[0]=='UP'
        assert data.observation_id.iloc[70]!=h10.observation_id.iloc[70]
        raise ValueError('insufficient mature trend classes policy')
    monkeypatch.setattr(sensitivity,'train_trend_head',fit)
    out=sensitivity.train_sensitivity('main')
    assert out['trained_head_count']==0 and len(out['unavailable_heads'])==1
    assert out['automatic_selection_eligible'] is out['publication_allowed'] is False
    assert out['H10_models_reused_for_H20'] is out['E0_E1_L2_retrained'] is False
    assert out['calibration_comparison'][0]['H20_status'].startswith('unavailable')
    assert 'H20' not in out['calibration_comparison'][0]
    assert out['coverage']['H20']['mature']<out['coverage']['H10']['mature']
    assert Path(control/'reports/trend_sensitivity.json').exists()
    protocol=json.loads((control/'config/protocol.json').read_text())
    assert out['protocol_hash']==stable_hash(protocol)==out['target_contract']['pipeline_contract_hash']
    assert protocol['main_H10_pipeline_hash_provenance_only']=='main-protocol'
    assert 'pipeline_contract_hash' not in protocol['config']
    assert 'consensus' not in protocol['config']
    assert len(captured['labels'])==len(h10)


def test_stage_source_is_explicit_immutable_and_never_completes_parent(tmp_path,monkeypatch):
    import pytest
    frame, calendar = _frame(110)
    cfg = strategy_config({'feature_start':calendar[0]})
    source = {'effective_data_end':calendar[-1], 'records_hash':stable_hash([{'symbol':'TEST.SZ'}])}
    source['snapshot_hash'] = stable_hash(source)
    cfg.update(source_snapshot_hash=source['snapshot_hash'], pipeline_contract_hash='H10-frozen')
    root = tmp_path/'running'
    def save(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    fold={'name':'trend_unit','train_end':calendar[70],'calibration_start':calendar[71],
        'calibration_end':calendar[89],'policy_start':calendar[90],'policy_end':calendar[-1]}
    bundles=[{'name':fold['name'],'fold':fold,'available_at':fold['policy_end'],
        'heads':{'trend_direction':{'path':'models/head','manifest_sha256':'verified-head'}}}]
    paths=[]
    for i in range(9):
        p=tmp_path/f'generator-{i}.py';p.write_text(f'GENERATOR={i}\n');paths.append(p)
    save('config/initial_config.json',cfg)
    save('config/generating_code_hashes.json',{str(p):sensitivity._file_hash(p) for p in paths})
    save('data_snapshots/source.json',source)
    save('data_snapshots/records.json',[{'symbol':'TEST.SZ'}])
    save('data_snapshots/calendar.json',{'verified':True,'dates':calendar})
    save('models/trend_bundles.json',bundles)
    labels=build_trend_labels(frame,calendar,cfg,source_snapshot_hash=source['snapshot_hash'],pipeline_contract_hash='H10-frozen')
    (root/'dataset/trend').mkdir(parents=True)
    labels.to_parquet(root/'dataset/trend/observations.parquet',index=False)
    monkeypatch.setattr(sensitivity,'ARTIFACT_RUNS_ROOT',tmp_path)
    monkeypatch.setattr(sensitivity,'chronological_folds',lambda *args:[fold])
    monkeypatch.setattr(sensitivity,'load_trend_head',lambda path:{
        'manifest_sha256':'verified-head','fold':fold,
        'actual_training_completed_at':'2026-10-12T10:00:00+08:00',
        'target_contract':labels.attrs['label_contract']})
    monkeypatch.setattr(sensitivity,'read_run',lambda rid:(_ for _ in ()).throw(ValueError('parent incomplete')))
    with pytest.raises(ValueError,match='parent incomplete'):
        sensitivity.read_source('running')
    directory,report,audit=sensitivity.read_source('running',allow_trend_stage_source=True)
    assert directory==root and audit['main_pipeline_completed'] is False
    assert audit['main_status']=='in_progress_trend_stage_frozen'
    assert report['trend_bundles']==bundles and 'E1_policy_hash' not in report
    assert not (root/'reports/trend_consensus.json').exists()
    save('models/trend_bundles.json',[])
    with pytest.raises(ValueError,match='every completed'):
        sensitivity.read_source('running',allow_trend_stage_source=True)
    save('models/trend_bundles.json',bundles)
    paths[0].write_text('changed\n')
    with pytest.raises(ValueError,match='nine-generator freeze'):
        sensitivity.read_source('running',allow_trend_stage_source=True)
