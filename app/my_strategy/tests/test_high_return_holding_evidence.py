"""Real account references reject invented inventory and future observations."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from my_strategy.web_dashboard import holding_evidence as evidence
from my_strategy.web_dashboard.api import ReplayHoldingRequest


@pytest.fixture
def account():
    dates = ['2025-10-16','2025-10-17','2025-10-20','2025-10-21']
    bar = {'open':4.,'high':4.2,'low':3.9,'close':4.1,'volume':100000.,'amount':400000.,
        'source':'sina_unadjusted_v1','has_trade_price':1}
    features = pd.DataFrame([{'date':d,'symbol':'000010.SZ',**bar} for d in dates])
    ledger = [{'date':'2025-10-17','symbol':'000010.SZ','action':'BUY','shares':100,'price':4.,'fee':5.,
        'position_before':0,'position_after':100,'cash_flow':-405.,'cash_after':99595.,
        'holding_cost_after':4.05,'signal_date':'2025-10-16'}]
    daily = [{'date':d,'shares':100 if d>='2025-10-17' else 0,'cash':99595. if d>='2025-10-17' else 100000.,
        'equity':100005. if d>='2025-10-17' else 100000.,'close':4.1,'sell_pending':None} for d in dates[:3]]
    snapshots = [{'date':d,'symbol':'000010.SZ','available_at':d+'T15:00:00+08:00','entry_date':'2025-10-17',
        'plan_id':'real-plan','shares':100,'entry_shares':100,'actual_cost_per_share':4.05,'bar':bar.copy()}
        for d in dates[1:3]]
    replay = {'daily':daily,'ledger':ledger,'holding_snapshots':snapshots}
    saved = {'daily':pd.DataFrame(daily),'ledger':pd.DataFrame(ledger),'holding_snapshots':copy.deepcopy(snapshots)}
    return saved, replay, features, dates


def validate(account):
    saved,replay,features,dates = account
    return evidence._validate_prefix(saved,replay,features,dates,symbol='000010.SZ',as_of='2025-10-20',initial_cash=100000.)


def test_true_prefix_contains_every_observed_holding_day_and_no_future(account):
    saved,replay,features,dates=account
    saved['holding_snapshots'].append({**saved['holding_snapshots'][-1],'date':'2025-10-21','shares':999999})
    current=validate(account)
    assert current['shares']==100 and [s['date'] for s in current['history']]==dates[1:3]
    assert all(s['date']<='2025-10-20' for s in current['history'])
    # None/NaN is CSV serialization, not a disagreement about real inventory.
    saved['daily']['sell_pending']=float('nan')
    assert validate(account)['actual_cost_per_share']==4.05


@pytest.mark.parametrize('kind', ['shares','cost','cash','bar','missing_history','date'])
def test_forged_prefix_never_becomes_compatible(account,kind):
    saved,replay,features,dates=account
    if kind=='shares': saved['daily'].loc[2,'shares']=200
    elif kind=='cash': saved['ledger'].loc[0,'cash_after']=99600.
    elif kind=='cost': saved['holding_snapshots'][-1]['actual_cost_per_share']=4.
    elif kind=='bar': saved['holding_snapshots'][-1]['bar']['high']=999.
    elif kind=='missing_history': saved['holding_snapshots'].pop(0)
    else: saved['ledger'].loc[0,'signal_date']='2025-10-17'
    with pytest.raises(ValueError): validate(account)


def test_coordinated_fake_replay_cannot_bypass_independent_cash_or_calendar_checks(account):
    saved,replay,features,dates=account
    saved['ledger'].loc[0,'holding_cost_after']=4.
    replay['ledger'][0]['holding_cost_after']=4.
    with pytest.raises(ValueError,match='含费成本'): validate(account)


def test_same_day_fill_cannot_become_real_observed_inventory(account):
    saved,replay,features,dates=account
    saved['ledger'].loc[0,'signal_date']='2025-10-17'
    replay['ledger'][0]['signal_date']='2025-10-17'
    with pytest.raises(ValueError,match='下一核验开盘'): validate(account)


def test_flat_account_has_no_snapshot_or_probability(account):
    saved,replay,features,dates=account
    saved['daily']=pd.DataFrame([{'date':d,'shares':0,'cash':100000.,'equity':100000.,'close':4.1,'sell_pending':None} for d in dates[:3]])
    replay['daily']=saved['daily'].to_dict('records'); replay['ledger']=[];replay['holding_snapshots']=[]
    saved['ledger']=saved['ledger'].iloc[:0];saved['holding_snapshots']=[]
    assert validate(account) is None


def test_reference_cannot_leave_run_directory_or_invent_manual_cost(tmp_path):
    with pytest.raises(ValueError): evidence._root('../foreign',tmp_path)
    with pytest.raises(ValueError): evidence._within(tmp_path.parent/'foreign',tmp_path)
    values={'replay_run_id':'run','window':'2025Q4','symbol':'000010.SZ','as_of':'2025-10-20'}
    with pytest.raises(ValueError): ReplayHoldingRequest(**values,shares=100,average_cost=4.)
    with pytest.raises(ValueError): ReplayHoldingRequest(**{**values,'usage_mode':'retrospective'})
    assert ReplayHoldingRequest(**values).usage_mode=='production'


def test_readonly_endpoint_and_cli_share_reference_contract(monkeypatch,capsys):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from my_strategy.web_dashboard.api import router
    from my_strategy.cli import main
    calls=[]
    def operation(**kwargs):
        calls.append(kwargs)
        return {'status':'flat','exit_probability':None,'publication_allowed':False}
    monkeypatch.setattr(evidence,'holding_exit_research',operation)
    app=FastAPI();app.include_router(router)
    spec={'replay_run_id':'real-run','window':'2025Q4','arm':'full','symbol':'000010.SZ','as_of':'2025-10-20','usage_mode':'historical'}
    with TestClient(app) as client:
        response=client.post('/api/research/high-return-holding',json=spec)
        assert response.status_code==200 and response.json()['exit_probability'] is None
        assert client.post('/api/research/high-return-holding',json={**spec,'shares':100}).status_code==422
        assert client.post('/api/research/high-return-holding',json={**spec,'replay_run_id':'../bad'}).status_code==422
    assert calls==[spec]
    assert main(['research-high-return-holding','--replay-run-id','real-run','--window','2025Q4','--symbol','000010.SZ',
        '--as-of','2025-10-20','--usage-mode','historical','--json'])==0
    assert json.loads(capsys.readouterr().out)['publication_allowed'] is False
    assert calls==[spec,spec]


def test_incompatible_saved_source_returns_explicit_null(monkeypatch):
    def reject(*args): raise ValueError('真实账户文件哈希不一致：holding_snapshots')
    monkeypatch.setattr(evidence,'_load_reference',reject)
    result=evidence.holding_exit_research(replay_run_id='real-run',window='2025Q4',arm='full',symbol='000010.SZ',as_of='2025-10-20')
    assert result['status']=='incompatible' and result['exit_probability'] is None
    assert not result['holding_contract_compatible'] and '哈希不一致' in result['reason']


@pytest.mark.parametrize('case',['hash','future'])
def test_real_file_hash_and_observed_date_guards_precede_scoring(monkeypatch,tmp_path,case):
    from my_strategy.services import czsc_high_return_forward,czsc_high_return_research
    from my_strategy.core.run_context import stable_hash
    root=tmp_path/'run'; reports=root/'reports';reports.mkdir(parents=True)
    directory=root/'account';directory.mkdir()
    files={'daily':'csv','ledger':'csv','rejections':'csv','episodes':'csv','account':'json',
        'holding_snapshots':'jsonl','entry_diagnostics':'jsonl','entry_gate_diagnostics':'jsonl','exit_policy_diagnostics':'jsonl'}
    binding={'symbol':'000010.SZ','window':'2025Q4','arm':'full','status':'complete','path':str(directory)}
    for key,extension in files.items():
        path=directory/(key+'.'+extension);path.write_text('{}' if extension=='json' else '')
        binding[key+'_sha256']=czsc_high_return_research._file_hash(path)
    cfg={'initial_cash':100000.}
    evaluation={'model_bundles_hash':stable_hash([]),'source_snapshot_hash':'source','cost_multiplier':1,
        'windows':{'2025Q4':{'ledger_bindings':{'full':[binding]},'fold':{'test_start':'2025-10-01'}}}}
    ep=root/'evaluation.json';ep.write_text(json.dumps(evaluation))
    report={'run_id':'run','model_family':'high_return','status':'completed','data_end':'2025-10-20',
        'config':cfg,'evaluation':czsc_high_return_research._compact_evaluation(evaluation),
        'complete_ledger_evaluation_path':str(ep)}
    (reports/'high_return.json').write_text(json.dumps(report));(root/'metadata.json').write_text('{"status":"complete"}')
    model={'config':cfg,'model_bundles':[],'source_snapshot':{'snapshot_hash':'source'}}
    monkeypatch.setattr(czsc_high_return_forward,'_model',lambda run:(root,model,{}))
    if case=='hash': (directory/'holding_snapshots.jsonl').write_text('{"shares":999999}\n')
    date='2025-10-21' if case=='future' else '2025-10-20'
    with pytest.raises(ValueError,match='尚无实际观察' if case=='future' else '文件哈希不一致'):
        evidence._load_reference('run','2025Q4','full','000010.SZ',date,tmp_path)


def test_holding_ui_has_real_reference_fields_and_escapes_report():
    root=Path(__file__).parents[1]/'web_dashboard/static'
    html=(root/'index.html').read_text();js=(root/'js/workbench.js').read_text()
    for identity in ('holding-replay-run','holding-replay-window','holding-replay-arm','holding-replay-symbol','holding-replay-asof','holding-replay-mode'):
        assert 'id="'+identity+'"' in html
    assert "request('/api/research/high-return-holding'" in js
    assert "esc(reference.account_sha256)" in js and "esc(data.reason)" in js
    assert '手动记录仅含股数与平均成本' in html


def test_holding_renderer_distinguishes_null_and_shows_cost_prefix_identity():
    node=shutil.which('node')
    if not node: pytest.skip('Node is unavailable')
    path=Path(__file__).parents[1]/'web_dashboard/static/js/workbench.js'
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const host={hidden:true,innerHTML:''};global.document={getElementById:()=>host};global.window={};
vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);','globalThis.renderHeld=renderHoldingReplay;'));
const report={status:'scored',reason:'<script>forged</script>',request:{usage_mode:'historical'},holding_state:{shares:4900,actual_cost_per_share:3.941182,entry_date:'2025-10-17'},observed_holding_features:{holding_sessions:2,running_peak_return:.06,running_trough_return:-.02,peak_giveback:.01},exit_probability:.6827349,exit_advantage:.010308,account_reference:{replay_run_id:'verified-run',window:'2025Q4',arm:'full',symbol:'000010.SZ',source:'union',model_run_id:'model',account_sha256:'sha-proof'},exit_model:{checkpoint:'2025Q4',actual_training_completed_at:'2026-10-10'},prefix_through_date:'2025-10-20',probability_target:'remaining_value_advantage'};
renderHeld(report);assert(!host.hidden);assert(host.innerHTML.includes('68.27%'));assert(host.innerHTML.includes('3.941182'));assert(host.innerHTML.includes('2025-10-20'));assert(host.innerHTML.includes('sha-proof'));assert(!host.innerHTML.includes('<script>'));assert(host.innerHTML.includes('&lt;script&gt;'));assert(host.innerHTML.split('<details>')[0].includes('历史回放开发研究'));assert(!host.innerHTML.split('<details>')[0].includes('sha-proof'));assert(!host.innerHTML.includes('<details open'));
renderHeld({...report,status:'model_unavailable',exit_probability:null,exit_advantage:null});assert(host.innerHTML.includes('观察日模型不可用'));assert(!host.innerHTML.includes('68.27%'));assert(host.innerHTML.includes('<strong>—</strong>'));
renderHeld({status:'flat',reason:'账户已平仓',holding_state:null});assert(host.innerHTML.includes('真实账户为空仓'));assert(!host.innerHTML.includes('退出优势概率'));
"""
    result=subprocess.run([node,'-e',script,str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
