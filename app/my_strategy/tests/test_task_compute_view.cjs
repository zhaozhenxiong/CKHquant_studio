const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../web_dashboard/static/js/task_compute_view.js'), 'utf8'), context);
const caption = context.KHQuantTaskComputeView.caption;
const task = execution => ({status:'running',spec:{analysis_method:'structure_kronos',device:'mps'},compute_info:{method:'structure_kronos',cpu_workers:2,prediction_execution:execution}});
test('requested MPS and old timings never become actual GPU execution proof', () => {
  const value = caption({status:'partial',spec:{analysis_method:'structure_kronos',device:'mps'},compute_info:{method:'structure_kronos',cpu_workers:2,timing:{phase_wall_seconds:{waiting_preparation:281.65,evaluation:16.4}}}});
  assert.match(value,/结构等待 281\.7 秒/);
  assert.match(value,/预测与评价 16\.4 秒/);
  assert.match(value,/请求 MPS/);
  assert.doesNotMatch(value,/已核验|CPU 信号运算/);
});
test('all-cache batch shows old MPS provenance without claiming current MPS work', () => {
  const value = caption(task({fresh_calls:0,reused_calls:91,execution_verified_calls:0,actual_devices:{},cached_provenance_devices:{mps:91}}));
  assert.match(value,/预测缓存 91 次/);
  assert.match(value,/缓存原计算：MPS/);
  assert.doesNotMatch(value,/本批已核验/);
});
test('verified noncached execution shows actual mixed device counts', () => {
  const value = caption(task({fresh_calls:3,reused_calls:5,execution_verified_calls:3,actual_devices:{mps:2,cpu:1}}));
  assert.match(value,/本批新预测请求 3 次/);
  assert.match(value,/本批已核验 MPS 2 次 \/ CPU 1 次/);
});
test('malformed, inconsistent, unknown or cache-only device claims fail closed', () => {
  for (const execution of [
    {fresh_calls:0,reused_calls:5,execution_verified_calls:5,actual_devices:{mps:5}},
    {fresh_calls:1,execution_verified_calls:1,actual_devices:{mps:2}},
    {fresh_calls:2,execution_verified_calls:2,actual_devices:{metal:2}},
    {fresh_calls:2,execution_verified_calls:2,actual_devices:{mps:'2'}},
    {fresh_calls:2,execution_verified_calls:2,actual_devices:{}},
    {fresh_calls:2,execution_verified_calls:1.5,actual_devices:{mps:1.5}},
  ]) assert.doesNotMatch(caption(task(execution)),/本批已核验/);
});
test('running phase and elapsed measurements use progress detail before terminal info', () => {
  const value = caption({...task({fresh_calls:2,reused_calls:0}),progress_detail:{current_phase:'source_preparation',cpu_workers:4,phase_wall_seconds:{waiting_preparation:2.5,evaluation:NaN}}});
  assert.match(value,/CPU 结构准备 4 进程/);
  assert.match(value,/当前等待结构准备/);
  assert.match(value,/结构等待 2\.5 秒/);
  assert.doesNotMatch(value,/NaN|预测与评价 .* 秒/);
});
test('terminal tasks do not display a current running stage; input is immutable', () => {
  const value = {...task({fresh_calls:1,reused_calls:0,execution_verified_calls:1,actual_devices:{'cuda:0':1}}),status:'succeeded',progress_detail:{current_phase:'model_inference'}};
  const original = JSON.stringify(value);
  assert.match(caption(value),/CUDA:0 1 次/);
  assert.doesNotMatch(caption(value),/当前预测/);
  assert.equal(JSON.stringify(value),original);
  assert.equal(caption(null),'');
});
const resourceCaption = context.KHQuantTaskComputeView.resourceCaption;
const resource = () => ({actual_device:'mps',elapsed_seconds:2.3,timings:{model_load_seconds:1.2,inference_seconds:1.0},execution:{schema:'kronos_execution_v1',fresh_inference:true,actual_device:'mps',inference_calls:1,tensor_devices_verified:true,synchronization_completed:true,model_parameter_devices:['mps'],tokenizer_parameter_devices:['mps'],input_tensor_devices:['mps']}});
test('prediction proof requires actual tensors, parameters and completed device synchronization', () => {
  assert.match(resourceCaption({resource:resource()}),/本次已核验 MPS 推理.*模型加载 1\.2 秒.*推理 1 秒/);
  for (const mutate of [r=>r.execution.synchronization_completed=false,r=>r.execution.model_parameter_devices=[],r=>r.execution.input_tensor_devices=['cpu'],r=>r.execution.actual_device='cpu',r=>r.execution.inference_calls=0]) {
    const value=resource();mutate(value);
    assert.doesNotMatch(resourceCaption({resource:value}),/已核验/);
  }
});
test('cached prediction never reuses old execution proof as this request execution', () => {
  const value=resourceCaption({reused:true,resource:resource()});
  assert.match(value,/预测缓存 · 原计算 MPS/);
  assert.match(value,/原模型加载 1\.2 秒/);
  assert.doesNotMatch(value,/本次已核验/);
});
test('final receipt supersedes an older last progress notification', () => {
  const value = {...task({fresh_calls:3,reused_calls:1,execution_verified_calls:3,actual_devices:{mps:3}}),status:'succeeded',progress_detail:{prediction_execution:{fresh_calls:1,reused_calls:1,execution_verified_calls:1,actual_devices:{mps:1}},phase_wall_seconds:{evaluation:1}}};
  value.compute_info.timing={phase_wall_seconds:{evaluation:3}};
  assert.match(caption(value),/本批新预测请求 3 次.*本批已核验 MPS 3 次.*预测与评价 3 秒/);
});
