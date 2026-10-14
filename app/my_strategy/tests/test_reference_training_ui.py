"""Exercise the shipped UI with the real reference-training schema."""
import json

from my_strategy.services.reference_training import capabilities
from my_strategy.tests.test_current_training_ui import run_js


def new_schema():
    data = capabilities()
    return "fields.splice(0,fields.length,..." + json.dumps(data["schema"]["fields"], ensure_ascii=False) + ");Object.keys(defaults).forEach(key=>delete defaults[key]);Object.assign(defaults," + json.dumps(data["defaults"]) + ");availableDevices=['auto','mps','cpu'];\n"


def test_reference_form_reads_numeric_choices_nullable_source_and_gpu_auto():
    run_js(new_schema() + r'''
(async()=>{
  await ui.refresh();
  assert.equal(element('run-device').value,'auto');
  const html=element('parameters-fields').innerHTML;
  assert(!html.includes('commission'));assert(!html.includes('8%'));assert(!html.includes('质量门槛'));
  assert.equal(element('default-horizon_sessions').value,'10');
  assert.equal(element('default-max_stock_weight').value,'20');
  const source=element('training-run-fields').innerHTML.match(/<input id="run-source_native_run_id"[^>]+>/)[0];
  assert(!source.includes('required'));
  change('default-horizon_sessions','20');change('default-risk_budget','0.5');
  custom=async request=>request.method==='PUT' ? {items:[],default_profile:{revision:4,parameters:request.body.parameters}} : undefined;
  await submit('parameters-form');
  const saved=requests.find(request=>request.method==='PUT').body.parameters;
  assert.equal(saved.horizon_sessions,20);assert.equal(saved.risk_budget,.005);
  assert.equal(saved.end,null);assert.equal(saved.source_native_run_id,null);
  assert.equal(saved.device,'auto');assert(!('initial_cash' in saved));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_reference_preflight_counts_and_job_receipt_keep_full_population():
    run_js(new_schema() + r'''
(async()=>{
  await ui.refresh();
  custom=async request=>request.path.endsWith('/preflight') ? {valid:true,plan:{plan_id:'reference-plan',historical_union_count:3194,latest_eligible_count:2842,data_end:'2026-10-12',parameters:request.body.parameters,calendar_run_id:'verified'}} : undefined;
  await submit('training-form');
  assert(element('training-preflight-result').innerHTML.includes('3194'));
  assert(element('training-preflight-result').innerHTML.includes('2842'));
  assert.equal(element('training-start').disabled,false);
  custom=async request=>request.path.endsWith('/jobs') ? {job_id:'reference-job',status:'queued',kind:'model_training'} :
    request.path==='/api/tasks/reference-job' ? {job_id:'reference-job',run_id:'new-reference-run',kind:'model_training',status:'running',progress:10,progress_detail:{stage:'reference_model',current:1,total:5,actual_device:'mps'}} : undefined;
  await element('training-start').handlers.click();
  const sent=requests.find(request=>request.path.endsWith('/jobs')).body;
  assert.equal(sent.plan_id,'reference-plan');assert.equal(sent.parameters.device,'auto');
  assert.equal(sent.parameters.source_native_run_id,null);
  assert(element('training-jobs').innerHTML.includes('MPS'));
  assert(element('training-jobs').innerHTML.includes('data-training-cancel'));
  assert(!element('training-jobs').innerHTML.includes('data-training-resume'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_measured_predictive_failure_is_distinct_from_missing_evidence():
    run_js(r'''
const base={job_id:'reference-complete',kind:'model_training',status:'succeeded',result:{strategy_version:'reference_position_v1',technical_validation:{status:'passed'},predictive_gate:{status:'pending',passed:false,log_loss:1.104321,brier:.670123,baseline_log_loss:1.091234,baseline_brier:.660456}}};
const failed=ui.renderJob(base);
assert(failed.includes('预测基线：未通过基线（不生成积极增仓目标）'));
assert(failed.includes('LogLoss：模型 1.104321 / 基线 1.091234'));
assert(failed.includes('Brier：模型 0.670123 / 基线 0.660456'));
assert(!failed.includes('预测基线：待补齐证据'));
const pending=ui.renderJob({...base,result:{...base.result,predictive_gate:{status:'pending',passed:false,log_loss:null}}});
assert(pending.includes('预测基线：待核验'));
assert(!pending.includes('未通过基线（不生成积极增仓目标）'));
assert(!pending.includes('training-predictive-metrics'));
''')
