"""Completed stages can be reused only with unchanged causal provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.services import czsc_high_return_research as research
from my_strategy.tests.test_czsc_high_return_research import Context, frozen_source, write_json


@pytest.fixture
def complete_stage(frozen_source, tmp_path):
    _, calendar, _, _, config = frozen_source
    records, _, snapshot, _ = research._source('frozen-source')
    root=research.ARTIFACT_RUNS_ROOT/'complete-stage'
    bundles=[{'available_at':'2023-01-01','heads':{'ma_entry':{'path':'entry'}}}]
    for relative,value in [('data_snapshots/source.json',snapshot),
            ('data_snapshots/source_records.json',records),
            ('data_snapshots/calendar.json',{'verified':True,'dates':calendar}),
            ('config/research.json',config),('models/ma_bundles.json',bundles)]:
        write_json(root/relative,value)
    code={str(p):research._file_hash(p) for p in research._code_files()}
    write_json(root/'config/initial_code_hashes.json',code)
    for name in code:
        path=Path(name); destination=root/'config/source_code'/path.relative_to(research.MY_STRATEGY_ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
    cached=[]
    for source in records:
        folder=root/'dataset/final'/source['symbol'].replace('.','_')
        folder.mkdir(parents=True,exist_ok=True)
        item={'symbol':source['symbol'],'path':str(folder),'source_path':source['path'],
            'source_sha256':source['sha256'],'source_hash_verified':True,'tradable_board':True,
            'episode_status':'completed'}
        for key in ['points','contracts','entries','exits','episodes']:
            path=folder/(key+('.json' if key in {'points','contracts'} else '.parquet'))
            if path.suffix=='.json':write_json(path,{'fixture':key})
            else:pd.DataFrame({'value':[1]}).to_parquet(path,index=False)
            item[key+'_path']=str(path);item[key+'_sha256']=research._file_hash(path)
        cached.append(item)
    final={'records':cached,'failures':[],'requested':2,'successful':2,
        'supported_mainboard':2,'hash_verified_count':2,'stage':'episodes',
        'sources':['ma','structure','wyckoff','union']}
    write_json(root/'dataset/final/manifest.json',final)
    return root,records,calendar,snapshot,config,bundles,Context(tmp_path/'new-run'),final


def invoke(stage):
    root,records,calendar,snapshot,config,bundles,context,_=stage
    return research._verified_final_artifacts(root.name,records=records,calendar=calendar,
        snapshot=snapshot,config=config,bundles=bundles,context=context)


def test_complete_stage_references_original_bytes_and_records_verification(complete_stage):
    root,*_,context,final=complete_stage
    before={str(p):research._file_hash(p) for p in (root/'dataset/final').rglob('*') if p.is_file()}
    assert invoke(complete_stage)==final
    assert before=={str(p):research._file_hash(p) for p in (root/'dataset/final').rglob('*') if p.is_file()}
    evidence=json.loads((context.root/'dataset/final/artifact_reuse.json').read_text())
    assert evidence['complete_stage_only'] and evidence['records_referenced_without_rewrite']
    assert evidence['verified_source_files']==2 and evidence['verified_artifact_files']==10
    assert json.loads((context.root/'dataset/final/manifest.json').read_text())==final


def test_independent_verifier_is_read_only_and_matches_saved_provenance(complete_stage):
    invoke(complete_stage)
    root,records,calendar,snapshot,config,bundles,context,final=complete_stage
    before={str(p):research._file_hash(p) for base in (root,context.root)
        for p in base.rglob('*') if p.is_file()}
    actual,binding=research._verified_final_artifacts(root.name,records=records,calendar=calendar,
        snapshot=snapshot,config=config,bundles=bundles)
    assert actual==final
    assert binding==json.loads((context.root/'dataset/final/artifact_reuse.json').read_text())
    assert before=={str(p):research._file_hash(p) for base in (root,context.root)
        for p in base.rglob('*') if p.is_file()}


@pytest.mark.parametrize('field,value',[
    ('failures',[{'symbol':'000001.SZ','error':'partial'}]),('successful',1),
    ('hash_verified_count',1),('requested',1),('stage','points'),
    ('sources',['union']),('supported_mainboard',1),('records',[]),
])
def test_partial_or_different_policy_stage_is_rejected(complete_stage,field,value):
    root,*_,final=complete_stage
    altered=copy.deepcopy(final);altered[field]=value
    write_json(root/'dataset/final/manifest.json',altered)
    with pytest.raises(ValueError,match='coverage/policies'):invoke(complete_stage)


@pytest.mark.parametrize('relative,mutation,reason',[
    ('data_snapshots/source.json',lambda d:d.update(snapshot_hash='wrong'),'source/calendar/behavior'),
    ('data_snapshots/calendar.json',lambda d:d.update(dates=d['dates'][:-1]),'source/calendar/behavior'),
    ('data_snapshots/source_records.json',lambda d:d.reverse(),'source/calendar/behavior'),
    ('config/research.json',lambda d:d.update(initial_cash=200000),'source/calendar/behavior'),
    ('models/ma_bundles.json',lambda d:d[0].update(available_at='2024-01-01'),'MA provenance'),
])
def test_source_calendar_behavior_and_MA_identity_cannot_drift(complete_stage,relative,mutation,reason):
    path=complete_stage[0]/relative;value=json.loads(path.read_text());mutation(value);write_json(path,value)
    with pytest.raises(ValueError,match=reason):invoke(complete_stage)


@pytest.mark.parametrize('key',['points','contracts','entries','exits','episodes'])
def test_target_and_point_bytes_are_verified(complete_stage,key):
    path=Path(complete_stage[-1]['records'][0][key+'_path']);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='file hash/path'):invoke(complete_stage)


def test_source_bytes_are_rechecked(complete_stage):
    path=Path(complete_stage[1][0]['path']);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source hash/identity'):invoke(complete_stage)


def test_artifact_file_cannot_escape_original_stage(complete_stage,tmp_path):
    root,*_,final=complete_stage
    changed=copy.deepcopy(final);source=Path(changed['records'][0]['points_path'])
    outside=tmp_path/'points.json';outside.write_bytes(source.read_bytes())
    changed['records'][0]['points_path']=str(outside)
    write_json(root/'dataset/final/manifest.json',changed)
    with pytest.raises(ValueError,match='file hash/path'):invoke(complete_stage)


def test_generating_dependency_drift_is_rejected(complete_stage,monkeypatch):
    original=research._file_hash
    dependency=next(p for p in research._code_files() if p.name=='czsc_trade_episodes.py')
    monkeypatch.setattr(research,'_file_hash',lambda p:'changed' if Path(p)==dependency else original(p))
    with pytest.raises(ValueError,match='generating dependency'):invoke(complete_stage)


@pytest.mark.parametrize('replacement',[
    ('def _mainboard(symbol):','def _mainboard(symbol):\n    return True'),
    ('import copy\n','import copy\nALTERED_GLOBAL=True\n'),
    ('from my_strategy.storage.czsc_results import ResultStore\n',
        'from my_strategy.storage.czsc_results import ResultStore\n\ndef _file_hash(path):\n    return "fake"\n'),
    ('from my_strategy.storage.czsc_results import ResultStore\n',
        'from my_strategy.storage.czsc_results import ResultStore\n\nclass pd:\n    pass\n'),
])
def test_archived_generating_functions_and_globals_cannot_drift(complete_stage,replacement):
    root=complete_stage[0];path=root/'config/source_code/services/czsc_high_return_research.py'
    path.write_text(path.read_text().replace(*replacement,1))
    code_path=root/'config/initial_code_hashes.json';code=json.loads(code_path.read_text())
    code[str(Path(research.__file__))]=research._file_hash(path);write_json(code_path,code)
    with pytest.raises(ValueError,match='generating functions'):invoke(complete_stage)


def test_outside_root_is_rejected(complete_stage):
    with pytest.raises(ValueError,match='artifact root'):
        research._verified_final_artifacts('..',records=[],calendar=[],snapshot={},config={},bundles=[],context=complete_stage[-2])
