"""Exercise shipped method requests, restoration and task-history actions without computation."""
import shutil
import subprocess

import pytest

from test_czsc_display_dates_ui import HARNESS, STATIC, run_js


def test_all_methods_have_identical_single_and_scan_contracts():
    run_js(r'''
const expected=['structure','structure_kronos','structure_kronos_ma_mlp','structure_kronos_ma_l2','legacy_reference'];
assert.deepEqual(Object.keys(KHQuantMethods.definitions),expected);
for(const prefix of ['analysis','scan']){
 for(const [suffix,value] of Object.entries({lookback:'256',horizon:'10',samples:'32',seed:'0',budget:'120','max-predictions':'2',device:'cpu','score-model':'ma-frozen'}))element(prefix+'-'+suffix).value=value;
 for(const method of expected.slice(0,4)){
  element(prefix+'-mode').value=method;const actual=workbench.researchSpec(prefix);
  assert.equal(actual.analysis_method,method);assert.equal(actual.analysis_mode,'reference');assert.equal(actual.research,false);
  assert.equal(actual.score_model_id,method.includes('_ma_')?'ma-frozen':undefined);
  if(method!=='structure'){assert.equal(actual.seed,0);assert.equal(actual.lookback,256);assert.equal(actual.max_predictions,2);}
 }
 element(prefix+'-mode').value='unknown';assert.throws(()=>workbench.researchSpec(prefix),/未知分析方法/);
}
''')


def test_saved_method_identity_and_drilldown_keep_model_and_forecast_settings():
    run_js(r'''
const context={analysis_method:'structure_kronos_ma_l2',score_model_id:'ma-existing',lookback:256,horizon:10,samples:32,seed:7,budget_seconds:100,max_predictions:2,device:'cpu'};
const result={analysis_method:context.analysis_method,method_context:context,request:{symbol:'002436.SZ',start:'2026-01-01',end:'2026-10-12'},data_end:'2026-10-12',research:false};
const saved=workbench.resultResearchContext(result);
workbench.applyResearchContext('analysis',saved);
assert.equal(element('analysis-mode').value,context.analysis_method);
assert.equal(element('analysis-score-model').value,context.score_model_id);
assert.equal(element('analysis-seed').value,'7');
assert.deepEqual(workbench.researchSpec('analysis'),{research:false,analysis_mode:'reference',...context});
await workbench.analyzeSymbol('002436.SZ','2026-10-12',result);
for(const [key,value] of Object.entries(context))assert.equal(lastAnalysisQuery[key],value,key);
assert.equal(lastAnalysisQuery.as_of,'2026-10-12');
assert.equal(lastAnalysisQuery.account_start,undefined);
''')


def test_method_evidence_does_not_hide_missing_inputs_or_rename_old_mlp():
    run_js(r'''
const html=KHQuantMethods.evaluationHtml({method:'structure_kronos_ma_mlp',status:'partial',as_of:'2026-10-12',direction:'WAIT',score:{score:null,status:'missing'},missing_reasons:['缺少均线候选'],kronos:{status:'missing',reason:'预算已用完'}});
assert(html.includes('均线模型 MLP'));assert(html.includes('评分不可用'));assert(html.includes('缺少均线候选'));assert(html.includes('预算已用完'));assert(!html.includes('评分 0'));
const complete=KHQuantMethods.evaluationHtml({method:'structure_kronos_ma_l2',status:'complete',direction:'BUY',score:{score:68,model_type:'l2',probabilities:[.1,.2,.7]}});
assert(complete.includes('68'));assert(complete.includes('上涨 70%'));assert(complete.includes('不代表模拟成交或目标仓位'));
''')


def history_js(body):
    node=shutil.which('node')
    if not node:
        pytest.skip('Node required for UI contract checks')
    setup=HARNESS + r'''
global.CustomEvent=class {constructor(type){this.type=type;}};window.dispatchEvent=()=>{};
const historySource=fs.readFileSync(path.join(process.argv[1],'task_history.js'),'utf8').replace('globalThis.KHQuantTaskHistory={refresh};boot();','globalThis.historyUI={state,refresh,deleteTasks,previewCleanup,executeCleanup,render};');
vm.runInThisContext(historySource);
const calls=[];const row={backend:'light',job_id:'score-one',kind:'score_training',status:'failed',created_at:'2026-10-15',can_delete:true};
global.fetch=async(url,options={})=>{calls.push({url,options});let value;
 if(url.startsWith('/api/task-history/overview?'))value={items:[row],total:1,counts:{terminal:1},active_items:[],active_total:0};
 else if(url.endsWith('/delete'))value={deleted:[row],skipped:[],failed:[]};
 else if(url.endsWith('/preview'))value={plan_id:'frozen-plan',plan_sha256:'sha',candidates:[{id:'record',type:'task_record',size_bytes:0,reason:'终态记录'}],protected:[],candidate_bytes:0};
 else if(url.endsWith('/execute'))value={deleted:['record'],skipped:[],failed:[],deleted_bytes:0};
 else throw Error('unexpected '+url);
 return {ok:true,json:async()=>value};};
'''
    result=subprocess.run([node,'-e',setup+'\n(async()=>{'+body+'\n})().catch(e=>{console.error(e);process.exitCode=1;});',str(STATIC)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_task_history_keeps_expansion_and_deletes_only_explicit_records():
    history_js(r'''
historyUI.state.open.add('light:score-one');await historyUI.refresh();
assert.equal(calls.length,1);assert(calls[0].url.startsWith('/api/task-history/overview?'));
assert(element('task-list').innerHTML.includes('data-history-key="light:score-one" open'));
await historyUI.refresh();assert(historyUI.state.open.has('light:score-one'));
await historyUI.deleteTasks([{backend:'light',job_id:'score-one'}]);
const call=calls.find(c=>c.url.endsWith('/delete'));
assert.deepEqual(JSON.parse(call.options.body),{tasks:[{backend:'light',job_id:'score-one'}]});
assert(!calls.some(c=>c.url.endsWith('/execute')));
assert(element('task-history-state').textContent.includes('结果产物保留'));
''')


def test_cleanup_executes_only_the_reviewed_plan_and_is_not_automatic():
    history_js(r'''
await historyUI.refresh();await historyUI.previewCleanup();
assert(!calls.some(c=>c.url.endsWith('/execute')));assert.equal(element('task-cleanup-plan').hidden,false);
await historyUI.executeCleanup();
const call=calls.find(c=>c.url.endsWith('/execute'));
assert.deepEqual(JSON.parse(call.options.body),{plan_id:'frozen-plan',plan_sha256:'sha'});
assert.equal(historyUI.state.plan,null);
''')


def test_light_training_and_analysis_transfer_preserve_profile_and_dates_without_starting():
    run_js(r'''
globalThis.submitted=[];
const lightSource=fs.readFileSync(path.join(process.argv[1],'light_research.js'),'utf8')
 .replace("async function submit(path,spec){", "async function submit(path,spec){submitted.push({path,spec});return;")
 .replace('globalThis.KHQuantLightResearch={refresh,openTask};boot();', 'globalThis.lightUI={updateFeatureControls,accountLabel};globalThis.KHQuantLightResearch={refresh,openTask};boot();');
vm.runInThisContext(lightSource);
element('light-feature-profile').value='ma_trend_v1';element('light-score-type').value='l2';element('light-score-variant').value='B';lightUI.updateFeatureControls();
assert.equal(element('light-score-type').value,'mlp');assert.equal(element('light-score-variant').value,'A');assert(element('light-score-type').disabled);assert(element('light-observe-panel').hidden);
element('light-training-symbols').value='002436.SZ';element('light-score-end').value='2025-12-31';
await fire('light-train','click');assert.equal(submitted[0].spec.feature_profile,'ma_trend_v1');assert.equal(submitted[0].spec.model_type,'mlp');assert.equal(submitted[0].spec.variant,'A');
element('analysis-mode').value='structure_kronos_ma_l2';element('analysis-symbol').value='002436.SZ';element('analysis-start').value='2026-01-01';element('analysis-end').value='2026-10-12';element('analysis-score-model').value='frozen-ma';
for(const [key,value] of Object.entries({lookback:'256',horizon:'10',samples:'32',seed:'7',device:'cpu',budget:'120','max-predictions':'1'}))element('analysis-'+key).value=value;
const before=submitted.length;await fire('open-light-analysis','click');assert.equal(submitted.length,before);assert.equal(location.hash,'simulation');
await fire('open-account-analysis','click');assert.equal(submitted.length,before);assert.equal(location.hash,'watchlist?tab=accounts');
assert.equal(element('light-symbols').value,'002436.SZ');assert.equal(element('light-start').value,'2026-01-01');assert.equal(element('light-end').value,'2026-10-12');assert.equal(element('light-predict-date').value,'2026-10-12');assert.equal(element('light-score-model').value,'frozen-ma');assert.equal(element('light-seed').value,'7');
assert(lightUI.accountLabel('structure_kronos_score',{lineage:{feature_profile:'ma_trend_v1'}}).includes('均线模型 MLP'));
assert(lightUI.accountLabel('structure_kronos_score',{}).includes('历史融合'));
''')


def test_cleanup_preview_reports_waiting_and_restores_button_after_failure():
    history_js(r'''
await historyUI.refresh();
const previousFetch=global.fetch;let finishPreview;let previewRequests=0;
global.fetch=(url,options)=>url.endsWith('/preview')?(previewRequests++,new Promise(resolve=>{finishPreview=resolve;})):previousFetch(url,options);
const pending=historyUI.previewCleanup();
assert.equal(element('task-cleanup-preview').disabled,true);
assert.equal(element('task-cleanup-preview').attributes['aria-busy'],'true');
assert(element('task-history-state').textContent.includes('正在核验引用与可回收空间'));
await historyUI.previewCleanup();assert.equal(previewRequests,1);
finishPreview({ok:false,json:async()=>({detail:'引用核验失败'})});
await assert.rejects(pending,/引用核验失败/);
assert.equal(element('task-cleanup-preview').disabled,false);
assert.equal(element('task-cleanup-preview').attributes['aria-busy'],undefined);
assert.equal(historyUI.state.busy,false);
global.fetch=previousFetch;await historyUI.previewCleanup();
assert.equal(element('task-cleanup-preview').disabled,false);
assert(element('task-history-state').textContent.includes('清理预览已完成'));
''')


def test_synthetic_method_summary_preserves_numeric_and_qualification_contract():
    import json
    from pathlib import Path
    # Public fixture: invented values and a deterministic synthetic model identifier.
    evidence=json.loads((Path(__file__).parent/'fixtures/method_evaluation_ma_mlp.json').read_text())
    run_js('const evidence='+json.dumps(evidence,ensure_ascii=False)+';'+r'''
const html=KHQuantMethods.evaluationHtml(evidence),summary=html.split('<details>')[0];
assert(summary.includes('MLP 已评分 60 / 100'));assert(!summary.includes('TORCH_MLP'));
assert(summary.includes('中位涨跌 -1.25%'));assert(summary.includes('P10 -4%'));assert(summary.includes('P90 2.5%'));
assert(summary.includes('上涨采样比例 37.5%'));assert(summary.includes('有效路径 24 / 32'));
assert(summary.includes('历史研究候选；独立前瞻资格待验证'));assert(summary.includes('采样分位数与上涨比例未经校准'));
for(const text of ['interval_semantics','probability_semantics','return_basis','scored','empirical_sample_quantiles'])assert(!summary.includes(text),text);
assert(summary.includes(evidence.model_id.slice(0,12)+'…'));assert(!summary.includes(evidence.model_id));assert(html.includes(evidence.model_id));
evidence.score.qualification={mode:'paper',independent_forward_status:'qualified'};
assert(KHQuantMethods.evaluationHtml(evidence).includes('自动模拟用途；独立前瞻资格已通过'));
delete evidence.score.qualification;assert(!KHQuantMethods.evaluationHtml(evidence).includes('独立前瞻资格待验证'));
''')


def runtime_js(body):
    node=shutil.which('node')
    if not node:
        pytest.skip('Node required for UI contract checks')
    harness=HARNESS.replace("'async function analyze(query, preserveDates = false, locate = null) { globalThis.lastAnalysisQuery=query; return;'", "'async function analyze(query, preserveDates = false, locate = null) {'")
    harness=harness.replace('globalThis.workbench={state,bind,refreshLatestDates,','renderAnalysis=()=>{};renderEvents=()=>{};renderReplayEvidence=()=>{};updateReplay=()=>{};emptyChart=()=>{};resetChartSelection=()=>{};globalThis.workbench={state,bind,analyze,showResult,runMethodCaption,scanTimingCell,scanActionCell,scanReasonText,refreshLatestDates,')
    result=subprocess.run([node,'-e',harness+'\n(async()=>{'+body+'\n})().catch(e=>{console.error(e);process.exitCode=1;});',str(STATIC)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_auto_resolved_model_transfers_only_while_original_form_still_matches():
    runtime_js(r'''
for(const [key,value] of Object.entries({mode:'structure_kronos_ma_mlp',symbol:'000001.SZ',start:'2026-01-01',end:'2026-10-14',lookback:'128',horizon:'5',samples:'32',seed:'42',device:'cpu',budget:'300','max-predictions':'1','score-model':''}))element('analysis-'+key).value=value;
const actualId='64963ef05012c1c1bf7adb2a4f6eeb3a859ef07d9e06a82097bc9db776ec449d';
let submitted;global.fetch=async(url,options)=>{assert.equal(url,'/api/analysis');submitted=JSON.parse(options.body);return response({symbol:'000001.SZ',analysis_method:submitted.analysis_method,method_context:{...submitted,score_model_id:actualId},request:submitted,data_end:submitted.end,structure_only:true,frequencies:{'日线':{bars:[]}},chart_evidence:{rows:[]}});};
await workbench.analyze(KHQuantMethods.analysisContext());assert.equal(submitted.score_model_id,undefined);assert.equal(workbench.state.query.score_model_id,actualId);
assert.equal(KHQuantMethods.analysisContext().score_model_id,actualId);
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'light_research.js'),'utf8'));
await fire('open-account-analysis','click');assert.equal(element('light-score-model').value,actualId);
element('analysis-end').value='2026-10-13';assert.equal(KHQuantMethods.analysisContext().score_model_id,undefined);
await fire('open-account-analysis','click');assert.equal(element('light-score-model').value,'','edited context must clear the previously transferred model');
element('analysis-end').value='2026-10-14';element('analysis-symbol').value='600519.SH';assert.equal(KHQuantMethods.analysisContext().score_model_id,undefined);
''')


def test_open_scan_clears_board_route_and_deferred_rows_never_claim_missing_prices():
    runtime_js(r'''
const stored={khquant_scan_tab:'board'};global.sessionStorage={getItem:key=>stored[key],setItem:(key,value)=>stored[key]=value};
global.location={href:'http://127.0.0.1:8124/?board=1&foo=2#data',search:'?board=1&foo=2',pathname:'/',hash:'#data'};
global.history={state:{},replaceState(_,__,url){const target=new URL(url,location.href);Object.assign(location,{href:target.href,search:target.search,pathname:target.pathname,hash:target.hash});}};
const code=fs.readFileSync(path.join(process.argv[1],'board_panel.js'),'utf8').replace("if(!$('scan-board-panel'))return;",'globalThis.boardTest={showStock,selectTab};return;');
vm.runInThisContext(code);globalThis.KHQuantBoardPanel={showStock:boardTest.showStock};
workbench.showResult('scan',{analysis_method:'structure_kronos',method_context:{analysis_method:'structure_kronos'},rows:[]},true);
assert.equal(location.search,'?foo=2');assert.equal(location.hash,'scan');assert.equal(stored.khquant_scan_tab,'stock');assert.equal(element('scan-stock-panel').hidden,false);assert.equal(element('scan-board-panel').hidden,true);
const deferred={method_evaluation:{method:'structure_kronos',status:'deferred',direction:'WAIT',missing_reasons:['batch_budget_deferred']}};
assert(workbench.scanActionCell(deferred).includes('待处理'));assert(!workbench.scanActionCell(deferred).includes('观察'));
assert(workbench.scanTimingCell(deferred).includes('尚未计算'));assert(!workbench.scanTimingCell(deferred).includes('缺失'));
assert.equal(workbench.scanReasonText(deferred),'本批预算已用尽，等待续批');
assert(workbench.scanTimingCell({data_end:'2026-10-14',method_evaluation:{status:'complete'}}).includes('结构时间见依据'));
assert(!workbench.runMethodCaption({summary:{analysis_method:'structure_kronos_ma_mlp',strategy_version:'account_free_methods_v1'}}).includes('归档'));
assert(workbench.runMethodCaption({summary:{strategy_version:'old_unknown'}}).includes('只读归档'));
''')
