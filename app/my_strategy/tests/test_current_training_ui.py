"""Exercise shipped current training UI with form changes and HTTP receipts."""
from pathlib import Path
import shutil
import subprocess

import pytest


STATIC = Path(__file__).resolve().parents[1] / "web_dashboard/static"


def run_js(body):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped training UI checks")
    result = subprocess.run([node, "-e", HARNESS + body, str(STATIC / "js/training.js")],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


HARNESS = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map(),requests=[],timers=[],storage=new Map(),selectionEvents=[];
function element(id) {
  if(!elements.has(id)) elements.set(id,{id,value:'',hidden:false,disabled:false,checked:false,dataset:{},open:false,scrollTop:0,style:{},handlers:{},_html:'',textContent:'',
    addEventListener(type,handler){this.handlers[type]=handler;},reportValidity(){return true;},
    querySelector(){return null;},querySelectorAll(){return this.controls || [];},get innerHTML(){return this._html;},
    set innerHTML(html){this._html=html;const controls=[];
      for(const match of html.matchAll(/<input\b([^>]+)>|<select\b([^>]+)>([\s\S]*?)<\/select>/g)) {
        const attrs=match[1] || match[2],id=attrs.match(/\bid="([^"]+)"/)?.[1],key=attrs.match(/data-parameter="([^"]+)"/)?.[1];
        if(!id || !key) continue;
        const control=element(id);control.dataset={parameter:key};control.type=attrs.match(/\btype="([^"]+)"/)?.[1] || 'select';
        control.value=attrs.match(/\bvalue="([^"]*)"/)?.[1] || match[3]?.match(/value="([^"]*)" selected/)?.[1] || '';
        control.checked=/\bchecked\b/.test(attrs);control.disabled=/\bdisabled\b/.test(attrs);control.readOnly=/\breadonly\b/.test(attrs);
        controls.push(control);
      }
      if(id==='training-run-fields') element('training-form').controls=controls;
      if(id==='parameters-fields') element('parameters-form').controls=controls;
    }});
  return elements.get(id);
}
global.document={getElementById:element};
global.sessionStorage={getItem:()=>null};global.localStorage={getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)};
global.window={dispatchEvent:event=>selectionEvents.push(event.type)};global.location={hash:''};
global.setTimeout=fn=>{timers.push(fn);return timers.length;};global.clearTimeout=()=>{};
const fields=[
  {key:'end',label:'数据截止',type:'date',default:null,group:'数据',impact:'rebuild'},
  {key:'device',label:'训练设备',type:'enum',choices:['cpu','mps','cuda'],default:'cpu',group:'资源'},
  {key:'cpu_workers',label:'CPU并发',type:'integer',min:1,max:4,group:'资源',impact:'retrain'},
  {key:'training_batch_size',label:'神经训练行批',type:'integer',min:1,max:65536,default:2048,group:'资源'},
  {key:'commission',label:'佣金',type:'number',unit:'比例',min:0,max:.01,group:'费用',impact:'rebuild'},
  {key:'quality_threshold',label:'质量门槛',type:'number',unit:'概率',min:0,max:1,group:'策略',impact:'retrain'},
  {key:'ma_weight',label:'趋势权重',type:'number',min:0,max:1,group:'策略',impact:'retrain'},
  {key:'liquidity_min_median_amount',label:'成交额中位数下限',type:'number',unit:'元CNY',min:30000000,default:30000000,group:'股票资格',impact:'rebuild'},
  {key:'liquidity_lookback_sessions',label:'流动性核验交易日',type:'integer',min:20,max:120,default:20,group:'股票资格',impact:'rebuild'},
  {key:'stock_batch_size',label:'每批处理股票数',type:'integer',min:1,max:10000,default:360,group:'资源',impact:'resource'},
  {key:'min_net_return',label:'高收益目标',type:'number',unit:'比例',default:.08,min:.08,locked:true,group:'约束'},
  {key:'auto_replace',label:'自动替换',type:'boolean',group:'选择',impact:'selection'}
];
const defaults={end:null,device:'cpu',cpu_workers:2,training_batch_size:2048,commission:.0003,quality_threshold:.1,ma_weight:.35,stock_batch_size:360,liquidity_lookback_sessions:20,liquidity_min_median_amount:30000000,min_net_return:.08,auto_replace:true};
let availableDevices=['cpu','mps'];
let revision=3,status={models:[],active_model:null,certification:{status:'pending',publication_allowed:false},jobs:[]};
let custom=async()=>undefined;
global.fetch=async(path,options={})=>{
  const request={path,method:options.method || 'GET',body:options.body ? JSON.parse(options.body) : null};requests.push(request);
  let response=await custom(request);
  if(response===undefined) {
    if(path.endsWith('/capabilities')) response={schema:{fields},devices:availableDevices,defaults,stages:[{key:'entry',label:'高收益入场'},{key:'return_loss',label:'辅助收益',status:'not_connected'}]};
    else if(path.endsWith('/profiles')) response={items:[],default_profile:{name:'我的默认配置',revision,parameters:defaults}};
    else if(path.endsWith('/status')) response=status;
    else if(path.endsWith('/selection-events')) response={items:[]};
    else throw new Error('Unexpected request '+path);
  }
  const failed=response?.http_error;
  return {ok:!failed,status:failed || 200,json:async()=>failed ? {detail:response.detail} : response};
};
const source=fs.readFileSync(process.argv[1],'utf8').replace('globalThis.KHQuantTraining = {refresh,showResult};',
  'globalThis.trainingTest={state,refresh,preflight,start,save,pollJobs,events,showResult,showComparison,renderJob,showDiagnostics,revalidate,validationSpec,syncTaskLayout,taskLayoutBudget,renderJobs};');
vm.runInThisContext(source);
const ui=trainingTest;
const submit=async id=>element(id).handlers.submit({preventDefault(){}});
const change=(id,value)=>{element(id).value=value;const form=id.startsWith('run-') ? 'training-form' : 'parameters-form';element(form).handlers.input();};
'''


def test_current_shell_has_one_model_and_separate_training_defaults_views():
    from html.parser import HTMLParser

    class Shell(HTMLParser):
        def __init__(self):
            super().__init__()
            self.active = None
            self.options = {}
            self.routes = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("data-route"):
                self.routes.append(attrs["data-route"])
            if tag == "select":
                self.active = attrs.get("id")
                self.options[self.active] = []
            if tag == "option" and self.active:
                self.options[self.active].append(attrs.get("value"))

        def handle_endtag(self, tag):
            if tag == "select":
                self.active = None

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    shell = Shell()
    shell.feed(html)
    assert {"training", "parameters"}.issubset(shell.routes)
    for prefix in ("analysis", "scan"):
        assert shell.options[f"{prefix}-model-family"] == ["current"]
        assert f"{prefix}-entry-policy" not in shell.options
        assert "production" not in shell.options[f"{prefix}-usage-mode"]
    assert shell.options["analysis-usage-mode"] == ["historical", "forward"]
    assert shell.options["scan-usage-mode"] == ["historical", "forward"]
    assert 'id="view-holdings"' not in html and 'id="view-backtest"' not in html
    assert 'id="analysis-position"' in html and 'id="position-reference-card"' in html


def test_form_units_nullable_latest_and_fixed_constraint_cannot_change():
    run_js(r'''
(async()=>{
  await ui.refresh();
  const html=element('parameters-fields').innerHTML;
  assert(html.includes('需重建来源、标签及受影响后代'));
  assert(html.includes('新任务重新训练'));assert(html.includes('新任务的研究选择设置'));
  assert(!html.includes('rebuild'));assert(!html.includes('retrain'));
  assert.equal(element('default-commission').value,'0.03');assert.equal(element('default-quality_threshold').value,'10');
  assert.equal(element('default-ma_weight').value,'35');assert(element('default-min_net_return').readOnly);
  assert.equal(element('run-liquidity_min_median_amount').value,'3000');
  assert.equal(element('run-liquidity_lookback_sessions').value,'20');
  assert.equal(element('run-stock_batch_size').value,'360');
  assert(html.includes('单位：万元人民币'));assert(html.includes('仅计算分批'));
  const endTag=element('training-run-fields').innerHTML.match(/<input id="run-end"[^>]+>/)[0];assert(!endTag.includes('required'));
  change('run-end','');change('run-stock_batch_size','240');
  custom=async request=>request.path.endsWith('/preflight') ? {valid:false,errors:['缺少核验日历'],plan:{plan_id:'blocked',parameters:request.body.parameters}} : undefined;
  await submit('training-form');
  const sent=requests.find(request=>request.path.endsWith('/preflight')).body.parameters;
  assert.equal(sent.end,null);assert.equal(sent.stock_batch_size,240);assert(!('capacity_tiers' in sent));assert(!('cohort_size' in sent));assert.equal(sent.commission,.0003);
  assert.equal(element('training-start').disabled,true);assert(element('training-preflight-result').innerHTML.includes('缺少核验日历'));
  element('default-min_net_return').value='0';
  custom=async request=>request.method==='PUT' ? {} : undefined;
  await submit('parameters-form');
  assert.equal(requests.find(request=>request.method==='PUT').body.parameters.min_net_return,.08);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_modified_parameters_require_new_frozen_preflight_before_job_submission():
    run_js(r'''
(async()=>{
  await ui.refresh();
  custom=async request=>request.path.endsWith('/preflight') ? {valid:true,plan:{plan_id:'frozen-plan',config_hash:'sha',parameters:request.body.parameters,cohort:['000001.SZ'],data_end:'2026-10-12',calendar_run_id:'verified'}} : undefined;
  await submit('training-form');assert.equal(element('training-start').disabled,false);
  change('run-cpu_workers','3');assert.equal(element('training-start').disabled,true);
  await element('training-start').handlers.click();
  assert(!requests.some(request=>request.path.endsWith('/jobs')));
  assert(element('training-error').textContent.includes('请重新预检'));
  await submit('training-form');
  custom=async request=>request.path.endsWith('/jobs') ? {job_id:'job',run_id:'wrapper',status:'queued'} :
    request.path==='/api/tasks/job' ? {job_id:'job',run_id:'wrapper',status:'running',progress:22,progress_detail:{stage:'E0_train_bootstrap',current:3,total:12,children:{raw_source_run_id:'raw',source_native_run_id:'native'}}} : undefined;
  await element('training-start').handlers.click();
  const sent=requests.find(request=>request.path.endsWith('/jobs')).body;
  assert.equal(sent.plan_id,'frozen-plan');assert.equal(sent.parameters.cpu_workers,3);
  assert(element('training-jobs').innerHTML.includes('构建初始真实参考账户'));
  assert(element('training-jobs').innerHTML.includes('原始来源：raw'));assert(element('training-jobs').innerHTML.includes('原生来源：native'));
  assert(!element('training-jobs').innerHTML.includes('E0_train_bootstrap'));
  assert.equal(storage.get('khquant_current_training_job'),'job');assert.equal(ui.state.plan,null);
  assert(element('training-start-state').textContent.includes('任务已提交'));
  assert(element('training-start-state').textContent.includes('任务号：job'));
  assert(!element('training-start-state').textContent.includes('参数已改变'));
  assert.equal(element('training-preflight-result').textContent,element('training-start-state').textContent);
  assert(element('training-preflight-result').textContent.includes('查看详情；新的训练需重新预检'));
  assert(!element('training-preflight-result').textContent.includes('正在提交冻结计划'));
  assert.equal(element('training-start').disabled,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_native_preparation_waits_for_actual_first_batch_counts_and_preserves_fold_state():
    run_js(r'''
(async()=>{
  await ui.refresh();
  const job={job_id:'current-full-job',kind:'model_training',status:'running',progress:0,
    progress_detail:{stage:'training',current:0,total:0,message:'NATIVE_SOURCE_RUN_ID frozen-native',
      children:{source_native_run_id:'frozen-native'}}};
  ui.state.jobs.set(job.job_id,job);ui.renderJobs();
  const panel=element('training-task-panel'),scroll=element('training-job-scroll');
  assert.equal(panel.open,false,'preparation must not expand the task panel');
  const summary=element('training-task-summary').innerHTML,html=element('training-jobs').innerHTML;
  assert(summary.includes('结构数据准备中 · 首批完成后更新计数'));
  assert(html.includes('结构数据准备中 · 首批完成后更新计数'));
  assert(html.includes('原生来源：frozen-native'));assert(!summary.includes('0 / 0'));assert(!html.includes('0 / 0'));
  assert(!summary.includes('360'));assert(!summary.includes('3194'),'resource batch and population cannot invent progress');
  assert(html.includes('width:0%'),'keep the reported progress rather than invent a preparation percentage');
  ui.state.jobs.set(job.job_id,{...job,progress_detail:{...job.progress_detail,stage:'native_source_prepare'}});ui.renderJobs();
  assert(element('training-task-summary').innerHTML.includes('结构数据准备中 · 首批完成后更新计数'));
  assert(!element('training-jobs').innerHTML.includes('native_source_prepare'));
  panel.open=true;scroll.scrollTop=147;
  ui.state.jobs.set(job.job_id,{...job,progress:8,progress_detail:{...job.progress_detail,stage:'native_source',current:360,total:3194}});
  ui.renderJobs();
  assert.equal(panel.open,true);assert.equal(scroll.scrollTop,147);
  assert(element('training-task-summary').innerHTML.includes('准备原生特征 · 360 / 3194'));
  assert(element('training-jobs').innerHTML.includes('width:8%'));
  assert(!element('training-task-summary').innerHTML.includes('首批完成后更新计数'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_actual_training_stage_codes_display_natural_chinese_and_keep_unknown_stage():
    run_js(r'''
(async()=>{
  await ui.refresh();
  const stages=[['calendar','核验交易日历'],['raw_source_freeze','冻结原始行情'],['frozen_source_verification','核验冻结来源'],['model_head','模型头训练完成'],
    ['native_source_prepare','结构数据准备'],['independent_three_class_trend','训练三分类趋势'],
    ['independent_policy_selection','核验来源政策'],['source_policy','核验来源政策'],
    ['missing_trend_dependency','趋势模型依赖不足'],['train_independent_actual_holding_MA_exit','训练固定持仓退出'],
    ['E0_actual_rule_bootstrap_no_HR','构建初始真实参考账户'],
    ['E1_pinned_exit_rebuild_entry_and_shadow_exit_labels','构建固定退出头续持账户'],
    ['sequential_L2_actual_flat_pre_gate_population_and_forced_E1_labels','逐折构建评分前尝试与真实克隆目标'],
    ['L2_population','逐折训练高收益与退出观察']];
  for(const [stage,label] of stages) {
    const html=ui.renderJob({job_id:'stage-job',kind:'model_training',status:'running',progress:7,
      progress_detail:{stage,current:360,total:3194}});
    assert(html.includes(label),stage);assert(!html.includes('>'+stage+' · '),stage);
    assert(html.includes('360 / 3194'));assert(html.includes('width:7%'));
  }
  assert(ui.renderJob({job_id:'unknown',status:'running',progress_detail:{stage:'unknown_actual_phase'}}).includes('unknown_actual_phase'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_device_suggestion_is_explicit_current_run_only_and_invalidates_plan():
    run_js(r'''
(async()=>{
  await ui.refresh();
  assert(element('run-device').innerHTML.includes('MPS · 可用'));
  assert(element('run-device').innerHTML.includes('CUDA · 当前不可用，不自动回退'));
  assert(element('training-run-fields').innerHTML.includes('不是同时运行的股票数'));
  assert(element('parameters-fields').innerHTML.includes('每次训练批的样本行数'));
  assert.equal(element('default-training_batch_size').value,'2048');
  custom=async request=>request.path.endsWith('/preflight') ? {valid:true,plan:{plan_id:'resource-plan',parameters:request.body.parameters}} : undefined;
  await submit('training-form');assert.equal(element('training-start').disabled,false);
  const before=requests.length;
  await element('training-device-suggest').handlers.click();
  assert.equal(element('run-device').value,'mps');assert.equal(element('default-device').value,'cpu');
  assert.equal(ui.state.parameters.device,'cpu','saved defaults must remain unchanged');
  assert.equal(ui.state.plan,null);assert(element('training-start').disabled);
  assert(element('training-start-state').textContent.includes('明确选择 MPS'));
  assert(!requests.slice(before).some(request=>request.method!=='GET'),'suggestion does not save or submit');
  await ui.refresh();assert.equal(element('run-device').value,'mps','refresh preserves the explicit current-run choice');
  await submit('training-form');
  assert.equal(requests.filter(request=>request.path.endsWith('/preflight')).at(-1).body.parameters.device,'mps');
  assert(!requests.some(request=>request.body?.parameters?.device==='auto'));
  assert(!requests.some(request=>request.method==='PUT' || request.path.endsWith('/jobs')));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_unavailable_explicit_mps_is_preserved_and_unknown_availability_has_no_suggestion():
    run_js(r'''
(async()=>{
  defaults.device='mps';availableDevices=['cpu'];await ui.refresh();
  assert.equal(element('run-device').value,'mps');assert.equal(element('default-device').value,'mps');
  assert(element('run-device').innerHTML.includes('MPS · 当前不可用，不自动回退'));
  assert(element('training-resource-status').textContent.includes('不会自动回退CPU'));
  custom=async request=>request.path.endsWith('/preflight') ? {valid:false,errors:['explicit MPS unavailable; no CPU fallback'],plan:{plan_id:'blocked'}} : undefined;
  await submit('training-form');assert(element('training-start').disabled);
  assert.equal(requests.find(request=>request.path.endsWith('/preflight')).body.parameters.device,'mps');
  assert.equal(element('run-device').value,'mps');
  availableDevices=undefined;await ui.refresh();
  assert.equal(element('run-device').value,'mps');assert(element('training-device-suggest').disabled);
  assert(element('training-resource-availability').textContent.includes('可用性尚未报告'));
  assert(!requests.some(request=>request.path.endsWith('/jobs') || request.method==='PUT'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_stage_rules_and_actual_device_report_are_separate_from_gpu_availability():
    run_js(r'''
(async()=>{
  await ui.refresh();
  const job={job_id:'resources',kind:'model_training',status:'running',progress:0,
    spec:{parameters:{device:'mps',cpu_workers:6,stock_batch_size:360,training_batch_size:2048}},
    progress_detail:{stage:'independent_policy_selection',current:10,total:3194}};
  let html=ui.renderJob(job);
  assert(html.includes('当前政策评估：CPU'));assert(html.includes('冻结请求设备：MPS'));
  assert(html.includes('本阶段实际设备报告：未报告'));
  assert(html.includes('股票准备分组 360 股 · CPU准备进程 6 · 神经训练行批 2048 行'));
  assert(!html.includes('实际设备报告：MPS'));
  html=ui.renderJob({...job,progress_detail:{stage:'train_independent_actual_holding_MA_exit',device:'mps'}});
  assert(html.includes('线性/MLP训练按冻结请求设备执行，树头在CPU'));
  assert(html.includes('本阶段实际设备报告：MPS'));
  html=ui.renderJob({...job,progress_detail:{stage:'model_head',actual_device:'mps'}});
  assert(html.includes('模型头训练完成'));assert(html.includes('本阶段实际设备报告：MPS'));
  html=ui.renderJob({...job,progress_detail:{stage:'unknown_phase'}});
  assert(html.includes('本阶段执行方式尚未报告'));
  assert(html.includes('本阶段实际设备报告：未报告'));
  assert(!html.includes('GPU利用率：'));assert(!html.includes('预计完成：'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_full_population_preflight_uses_daily_eligibility_and_resource_batch_only():
    run_js(r'''
(async()=>{
  status.data_end='2026-10-12';await ui.refresh();
  assert.equal(element('run-end').value,'','empty latest date stays nullable');
  change('run-liquidity_min_median_amount','4200');change('run-stock_batch_size','100');
  custom=async request=>request.path.endsWith('/preflight') ? {valid:true,plan:{plan_id:'full',parameters:request.body.parameters,
    requested_end:null,effective_end:'2026-10-12',date_status:{code:'latest_verified_frozen'},cohort:['a','b','c'],
    liquidity_universe:{selected_symbols:['a','b','c'],latest_count:2,daily_counts:{'2026-10-12':2},policy:{min_median_amount:42000000},policy_hash:'policy'}}} : undefined;
  await submit('training-form');const request=requests.find(request=>request.path.endsWith('/preflight'));
  assert.equal(request.body.parameters.liquidity_min_median_amount,42000000);assert.equal(request.body.parameters.stock_batch_size,100);
  assert(!('cohort_size' in request.body.parameters));assert(!('capacity_tiers' in request.body.parameters));
  const html=element('training-preflight-result').innerHTML;
  assert(html.includes('历史合格并集'));assert(html.includes('3 股 · 全部参与'));assert(html.includes('2 股'));
  assert(html.includes('每批 100 股仅分配资源，不截断人口'));assert(html.includes('4200 万元'));
  assert(html.includes('policy_hash'));assert(element('training-population-status').innerHTML.includes('尚未报告'));
  assert(!element('training-run-fields').innerHTML.includes('cohort_size'));assert(!element('parameters-fields').innerHTML.includes('capacity_tiers'));
  await ui.refresh();assert.equal(element('run-liquidity_min_median_amount').value,'4200','status refresh preserves this-run edits');
  assert.equal(element('run-stock_batch_size').value,'100');
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_invalid_date_and_network_failure_explain_real_blocker_next_to_start():
    run_js(r'''
(async()=>{
  await ui.refresh();assert(element('training-start-state').textContent.includes('尚未预检'));
  custom=async request=>request.path.endsWith('/preflight') ? {valid:false,errors:['2026-10-11 不是核验交易日'],plan:{plan_id:'invalid',
    requested_end:'2026-10-11',effective_end:null,date_status:{code:'not_verified_session',nearest_verified_before:'2026-10-09',nearest_verified_after:'2026-10-12'}}} : undefined;
  await submit('training-form');assert(element('training-start').disabled);
  assert(element('training-start-state').textContent.includes('2026-10-11 不是核验交易日'));
  assert(element('training-preflight-result').innerHTML.includes('最近前一核验交易日 2026-10-09'));
  assert(element('training-preflight-result').innerHTML.includes('最近后一核验交易日 2026-10-12'));
  assert(element('training-preflight-result').innerHTML.includes('有效数据截止 未核验'));
  custom=async request=>{if(request.path.endsWith('/preflight'))throw new Error('网络中断，请重试');};
  await submit('training-form');assert(element('training-start').disabled);
  assert(element('training-start-state').textContent.includes('计划检查失败'));
  assert(!element('training-start-state').textContent.includes('参数已改变'));
  assert(element('training-error').textContent.includes('网络中断'));
  assert(!requests.some(request=>request.path.endsWith('/jobs')));
  custom=async request=>request.path.endsWith('/preflight') ? {valid:true,plan:{}} : undefined;
  await submit('training-form');assert(element('training-start').disabled,'valid flag without plan receipt cannot enable start');
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_task_fold_keeps_live_summary_expansion_scroll_and_actual_completion_counts():
    run_js(r'''
(async()=>{
  await ui.refresh();assert.equal(element('training-task-panel').open,false);
  ui.state.jobs.set('active',{job_id:'active',kind:'model_training',status:'running',created_at:'3',progress_detail:{stage:'trend_fit',current:14,total:25}});
  ui.state.jobs.set('failed',{job_id:'failed',status:'failed',created_at:'2',error:'source missing'});
  ui.state.jobs.set('blocked',{job_id:'blocked',status:'succeeded',created_at:'1',result:{selection:{status:'blocked'},
    historical_union_count:3100,latest_eligible_count:2200,trained_stock_count:3097}});
  element('training-task-panel').open=true;element('training-job-scroll').scrollTop=223;ui.renderJobs();
  assert.equal(element('training-task-panel').open,true);assert.equal(element('training-job-scroll').scrollTop,223);
  const summary=element('training-task-summary').innerHTML;
  assert(summary.includes('活动 1'));assert(summary.includes('失败 1'));assert(summary.includes('资格／诊断阻断 1'));
  assert(summary.includes('训练与核验趋势检查点'));assert(summary.includes('14 / 25'));
  const html=element('training-jobs').innerHTML;
  assert(html.includes('3100'));assert(html.includes('2200'));assert(html.includes('3097'));
  element('training-task-panel').open=false;ui.renderJobs();assert.equal(element('training-task-panel').open,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_task_scroll_budget_tracks_qualification_boundary_and_releases_on_mobile():
    run_js(r'''
const panel=element('training-task-panel'),heading=element('training-task-heading'),body=element('training-job-scroll'),boundary=element('training-qualification-panel');
const variables=new Map();for(const item of [panel,body])item.style={setProperty:(key,value)=>variables.set(key,value),removeProperty:key=>variables.delete(key)};
panel.getBoundingClientRect=()=>({top:550,width:527});heading.getBoundingClientRect=()=>({height:90});boundary.getBoundingClientRect=()=>({bottom:885,height:310});
global.getComputedStyle=item=>item===panel ? {paddingTop:'20px',paddingBottom:'20px',borderTopWidth:'1px',borderBottomWidth:'1px'} : {marginTop:'16px'};
global.matchMedia=()=>({matches:true});ui.syncTaskLayout();
assert.equal(variables.get('--task-panel-size'),'335px');assert.equal(variables.get('--task-body-size'),'187px');
ui.state.jobs.set('many',{job_id:'many',status:'failed',logs:Array(100).fill('long log')});panel.open=true;ui.renderJobs();ui.syncTaskLayout();
assert.equal(variables.get('--task-panel-size'),'335px','opening long details cannot expand past the same qualification boundary');
global.matchMedia=()=>({matches:false});ui.syncTaskLayout();assert.equal(variables.size,0);
''')
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    opening = html.split('<details id="training-task-panel"', 1)[1].split(">", 1)[0]
    assert "open" not in opening
    css = (STATIC / "css/training.css").read_text(encoding="utf-8")
    assert "@media(max-width:800px)" in css and "@media(max-width:480px)" in css
    assert "max-block-size:60dvh" in css


def test_dirty_default_revision_survives_refresh_and_conflict_preserves_edits():
    run_js(r'''
(async()=>{
  await ui.refresh();change('default-commission','0.04');revision=4;
  await ui.refresh();assert.equal(element('default-commission').value,'0.04');assert.equal(ui.state.revision,3);
  custom=async request=>request.method==='PUT' ? {http_error:409,detail:'默认参数已被修改，请刷新后重试'} : undefined;
  await submit('parameters-form');
  const sent=requests.find(request=>request.method==='PUT').body;
  assert.equal(sent.expected_revision,3);assert.equal(sent.parameters.commission,.0004);
  assert(element('parameters-error').textContent.includes('默认参数已被修改'));
  assert.equal(element('default-commission').value,'0.04');assert.equal(ui.state.dirty,true);assert.equal(ui.state.revision,3);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_finished_job_discloses_model_diagnostics_and_blocked_selection_events():
    run_js(r'''
(async()=>{
  await ui.refresh();ui.state.jobs.set('job',{job_id:'job',status:'running'});
  custom=async request=>request.path==='/api/tasks/job' ? {job_id:'job',run_id:'wrapper',status:'succeeded',progress:100,
    result:{training_run_id:'trained',selection:{status:'blocked',reasons:['calibration_positive_class_missing <detail>']},
      diagnostics:{status:'completed',run_id:'diag',report_paths:{json:'reports/diagnostics.json',markdown:'reports/diagnostics.md'}},children:{model_run_id:'trained'}}} :
    request.path.endsWith('/selection-events') ? {items:[{kind:'select',created_at:'2026-10-13',payload:{run_id:'qualified',previous_run_id:'previous',effective_from:'2026-10-14',reason:'通过冻结证据'}}]} : undefined;
  await ui.pollJobs();const html=element('training-jobs').innerHTML;
  assert(html.includes('训练已完成；替换结果：阻断'));assert(html.includes('calibration_positive_class_missing &lt;detail&gt;'));
  assert(html.includes('模型运行：trained'));assert(html.includes('诊断运行 diag'));assert(html.includes('reports/diagnostics.md'));
  assert(!html.includes('undefined'));assert(selectionEvents.includes('khquant-selection-updated'));
  const eventHtml=element('training-selection-events').innerHTML;
  assert(eventHtml.includes('更新当前研究模型'));assert(eventHtml.includes('生效 2026-10-14'));
  assert(eventHtml.includes('模型 qualified'));assert(eventHtml.includes('前任 previous'));
  assert(element('training-overview').innerHTML.includes('尚未取得'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_diagnostics_button_reads_verified_report_and_keeps_fold_populations_separate():
    run_js(r'''
(async()=>{
  await ui.refresh();
  const segment={start:'2026-07-01',end:'2026-08-13',selected:{rows:369,positive:0,negative:369,unknown:0},
    target:{candidates:422,available_labels:411,fee_net_profitable:220,net_return_target_pass:3,r_multiple_target_pass:0,both_target_pass:0,
      exclusive_target_reasons:{net_nonpositive:191,net_below_target:217,r_below_target:3,episode_unclosed:11},
      fees_total:153.5,fee_excluded_same_denominator_high:0,fee_only_high_to_negative:0,
      risk_fraction_quantiles:{'0.5':.0813},required_net_return_quantiles:{'0.5':.1626}},
    funnel:[{stage:'decision_segment',rows:422,positive:0,removed:0},{stage:'label_unavailable',rows:411,positive:0,removed:11,positive_removed:0},
      {stage:'label_end_after_cutoff',rows:369,positive:0,removed:42,positive_removed:0}],
    ledger_audit:{last_exit_reasons:{signal_pinned_ma_exit:396},fee_and_slippage_excluded_same_denominator_high:0,holding_session_distance_quantiles:{'0.5':1}},
    reference_quality:{quality_rejection_removes_training_candidate:false,high_labels_retained_despite_reference_rejection:0,reason:{model_unavailable:400}},
    feature_observations:{rows_with_missing_features:21},economic_components:211};
  const report={run_id:'diag',model_run_id:'model',data_end:'2026-10-12',cohort_size:360,artifact_integrity_passed:true,source_report_unchanged:true,
    rules:{high_return:{min_net_return:.08,min_r_multiple:2}},folds:[{fold:{name:'production'},segments:{calibration:segment,
      train:{...segment,selected:{rows:12098,positive:206,negative:11892,unknown:0}},policy:{...segment}},
      qualification:{passed:false,row_gate_passed:true,both_classes_passed:false,min_train_rows:100,min_calibration_rows:30},
      daily_candidate_path:{calibration:{exclusive_first_block:{no_active_buy_opportunity:15142,pre_gate_attempt:422}}}},
      {fold:{name:'other-fold'},segments:{calibration:{...segment,target:{...segment.target,candidates:17}}},qualification:{passed:false,row_gate_passed:false}}],
    input_assets:[{path:'frozen<script>',sha256:'input-sha'}],output_assets:[{path:'samples.parquet',sha256:'output-sha'}]};
  custom=async request=>request.path.startsWith('/api/training/diagnostics/') ? report : undefined;
  await ui.showResult({run_id:'wrapper',training_run_id:'model',diagnostics:{status:'completed',run_id:'diag',report_sha256:'a'.repeat(64)}});
  assert(element('training-jobs').innerHTML.includes('查看样本流失诊断'));
  const target={dataset:{trainingDiagnostics:'diag',diagnosticSha256:'a'.repeat(64)},closest(selector){return selector==='[data-training-diagnostics]' ? this : null;}};
  await element('training-jobs').handlers.click({target});
  assert(requests.some(request=>request.path==='/api/training/diagnostics/diag?expected_sha256='+'a'.repeat(64)));
  const html=element('training-sample-diagnostics').innerHTML;
  assert(html.includes('每折人口分别统计'));assert(html.includes('不能相加'));
  assert(html.includes('production'));assert(html.includes('other-fold'));
  assert(html.includes('校准段'));assert(html.includes('训练段'));assert(html.includes('政策段'));
  assert(html.includes('369'));assert(html.includes('12098'));assert(html.includes('206'));
  assert(html.includes('8.00%'));assert(html.includes('2R'));assert(html.includes('费用后盈利'));
  assert(html.includes('目标是本折校准正类不足的直接原因'));
  assert(html.includes('清仓晚于本段截止'));assert(html.includes('排除正类'));
  assert(html.includes('质量门槛拒绝没有直接删除强制入场克隆'));
  assert(html.includes('入场时固定 MA 退出头'));assert(html.includes('396'));
  assert(html.includes('16.26%'));assert(html.includes('退出日索引减入场日索引'));
  assert(html.includes('frozen&lt;script&gt;'));assert(!html.includes('frozen<script>'));
  assert(html.includes('完整诊断 JSON'));assert(html.includes('input-sha'));assert(html.includes('samples.parquet'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_diagnostic_wrong_identity_does_not_render_other_model_report():
    run_js(r'''
(async()=>{
  await ui.refresh();custom=async request=>request.path.startsWith('/api/training/diagnostics/') ? {run_id:'wrong',model_run_id:'other',folds:[]} : undefined;
  await assert.rejects(()=>ui.showDiagnostics('expected'),/身份与所选运行不一致/);
  assert(element('training-sample-diagnostics').textContent.includes('诊断读取或身份核验失败'));
  assert(!element('training-sample-diagnostics').innerHTML.includes('模型 other'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_revalidation_uses_original_plan_and_completed_model_without_retraining():
    run_js(r'''
(async()=>{
  status.models=[{run_id:'complete-model',status:'completed',archived:false,validation_available:true},{run_id:'small-model',status:'capacity_insufficient',validation_available:false}];
  await ui.refresh();
  const failed={job_id:'old-job',kind:'model_training',status:'failed',spec:{plan_id:'original-plan'},
    progress_detail:{children:{model_run_id:'complete-model'}}};
  const incomplete={...failed,job_id:'incomplete',progress_detail:{children:{model_run_id:'small-model'}}};
  assert.equal(ui.validationSpec(incomplete),null);
  await assert.rejects(()=>ui.revalidate(incomplete),/完整模型资产/);
  assert(!requests.some(request=>request.path.endsWith('/validation-jobs')));
  ui.state.jobs.set(failed.job_id,failed);
  assert(ui.renderJob(failed).includes('重新核验已完成模型'));
  custom=async request=>request.path.endsWith('/validation-jobs') ? {job_id:'validation-job',kind:'model_validation',status:'queued'} :
    request.path==='/api/tasks/validation-job' ? {job_id:'validation-job',kind:'model_validation',status:'succeeded',
      spec:{plan_id:'original-plan',model_run_id:'complete-model'},result:{training_run_id:'complete-model',original_plan_id:'original-plan',
        plan_id:'validation-plan',training_started:false,weights_unchanged:true,selection:{status:'blocked',reasons:['校准缺少正类']},
        diagnostics:{status:'completed',run_id:'new-diagnostic'}}} : undefined;
  const target={dataset:{trainingValidate:'old-job'},disabled:false,closest(selector){return selector==='[data-training-validate]' ? this : null;}};
  await element('training-jobs').handlers.click({target});
  const request=requests.find(request=>request.path.endsWith('/validation-jobs'));
  assert.deepEqual(request.body,{model_run_id:'complete-model',plan_id:'original-plan'});
  assert(!requests.some(request=>request.path.endsWith('/jobs') || request.path.endsWith('/preflight')));
  const html=element('training-jobs').innerHTML;
  assert(html.includes('已完成模型复核'));assert(html.includes('复核已完成；替换结果：阻断'));
  assert(html.includes('本次没有启动训练，模型权重保持不变'));assert(html.includes('校准缺少正类'));
  assert(html.includes('new-diagnostic'));assert(html.includes('查看样本流失诊断'));
  assert.equal(ui.state.jobs.get('old-job').status,'failed','original failed task remains unchanged');
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_archived_completed_model_blocks_revalidation_and_keeps_original_diagnostic():
    run_js(r'''
(async()=>{
  const reason='原人口或生成/执行代码与当前不同 <旧包>；完整训练须使用新来源';
  status.models=[{run_id:'archived-model',status:'completed',archived:true,validation_available:false,
    archive_reason:reason,required_recent_heads_ready:true,automatic_selection_eligible:true}];
  await ui.refresh();
  const job={job_id:'archived-job',kind:'model_training',status:'failed',spec:{plan_id:'original-plan'},
    result:{training_run_id:'archived-model',training_status:'completed',training_started:false,weights_unchanged:true,
      diagnostics:{status:'completed',run_id:'original-diagnostic'}}};
  assert.equal(ui.validationSpec(job),null,'old completion receipt cannot override current archive status');
  await assert.rejects(()=>ui.revalidate(job),/原人口或生成\/执行代码与当前不同/);
  assert(!requests.some(request=>request.path.endsWith('/validation-jobs')));
  const html=ui.renderJob(job), table=element('training-selection').innerHTML;
  assert(/data-training-validate="archived-job" disabled/.test(html));
  assert(html.includes('原任务与原诊断保留，可继续查看'));assert(html.includes('查看样本流失诊断'));
  assert(html.includes('original-diagnostic'));assert(html.includes('&lt;旧包&gt;'));assert(!html.includes('<旧包>'));
  assert(table.includes('历史归档 · 只读'));assert(table.includes('历史头仅作档案，不提供当前资格'));
  assert(table.includes('需新完整来源重新训练'));assert(!table.includes('样本资格通过，仍需账本比较'));
  assert.equal(ui.validationSpec({...job,result:{...job.result,training_run_id:'not-in-current-catalog'}}),null,
    'completion receipt alone cannot certify current validation availability');
  status.models=[{run_id:'archived-model',status:'completed',archived:false,validation_available:false,archive_reason:'当前资产不可复核'}];
  ui.state.status=status;
  assert.equal(ui.validationSpec(job),null);await assert.rejects(()=>ui.revalidate(job),/当前资产不可复核/);
  assert(!requests.some(request=>request.path.endsWith('/validation-jobs')));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_reference_quality_stage_has_independent_counts_and_cpu_rule():
    run_js(r'''
for(const [stage,label] of [['reference_quality_population','生成独立参考机会人口'],['train_reference_quality','训练独立参考质量头'],['reference_quality_fee_qualification','核验参考人口完整费用资格']]) {
 const html=ui.renderJob({job_id:'reference',status:'running',progress:25,progress_detail:{stage,current:12,total:3194,actual_device:'cpu'},spec:{parameters:{device:'mps'}}});
 assert(html.includes(label));assert(html.includes('12 / 3194'));assert(html.includes('CPU'));assert(html.includes('当前阶段 25%'));
 assert(!html.includes('本阶段已使用GPU'));
}
''')


def test_reference_quality_failure_is_separate_from_core_model_selection():
    run_js(r'''
(async()=>{
 const q={status:'trained',heads:4,unavailable_heads:[{fold:'latest',reason:'reference_quality_no_positive_class'}],population:'independent_reference',population_hash:'reference-pop-proof',fee_status:'rejected',fee_qualified:false,qualification_path:'validation/reference_quality/qualification.json',qualification_sha256:'ref-fee-proof',required_for_core_selection:false,publication_allowed:false};
 const result={training_run_id:'model',selection:{status:'accepted',reasons:[]},reference_quality:q,result_reasons:['reference_quality_not_qualified']};
 let html=ui.renderJob({job_id:'done',status:'succeeded',run_id:'run',result});
 for(const text of ['独立参考质量','完成头 4','完整费用资格 未通过或证据不足','参考质量与核心自动选模分别核验','参考仓位目标保持待核验','reference-pop-proof','ref-fee-proof','reference_quality_no_positive_class'])assert(html.includes(text),text);
 assert(!html.includes('<ul class="training-blockers">'));assert(html.includes('reference-quality-diagnostics'));
 status={models:[{run_id:'model',required_recent_heads_ready:true,automatic_selection_eligible:true,reference_quality:q}],active_model:null,certification:{status:'pending'},jobs:[]};
 await ui.refresh();html=element('training-selection').innerHTML;
 assert(html.includes('近期关键头齐备'));assert(html.includes('独立参考质量'));assert(html.includes('参考质量与核心自动选模分别核验'));
 await ui.showResult(result);assert(element('training-jobs').innerHTML.includes('完成头 4'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_reference_quality_fee_success_is_not_production_certification():
    run_js(r'''
const html=ui.renderJob({status:'succeeded',job_id:'q',result:{reference_quality:{status:'completed',heads:12,fee_status:'passed',fee_qualified:true,qualification_sha256:'<img>',publication_allowed:false,required_for_core_selection:false}}});
assert(html.includes('完成头 12'));assert(html.includes('完整费用资格 已通过'));assert(html.includes('不授予正式生产资格'));assert(html.includes('&lt;img&gt;'));assert(!html.includes('<img>'));
''')
