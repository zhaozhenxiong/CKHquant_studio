from my_strategy.tests.test_current_training_ui import run_js


def test_native_progress_uses_only_measured_finite_values():
    run_js(r'''
const job={job_id:'native-job',kind:'model_training',status:'running',spec:{parameters:{}},
  progress_detail:{stage:'native_source_prepare',current:20,total:60,in_flight:8,
    cache_hits:10,reused:2,stocks_per_minute:24.5,eta_seconds:120}};
let html=trainingTest.renderJob(job);
assert(html.includes('计算中 8 股'));assert(html.includes('数据库复用 10 股'));
assert(html.includes('断点复用 2 股'));assert(html.includes('实测 24.5 股/分钟'));
assert(html.includes('本阶段剩余约 2 分钟'));
Object.assign(job.progress_detail,{in_flight:Infinity,cache_hits:'10',reused:-1,
  stocks_per_minute:NaN,eta_seconds:Infinity});
html=trainingTest.renderJob(job);
assert(!html.includes('本阶段剩余约'));assert(!html.includes('实测 NaN'));
assert(!html.includes('数据库复用 10 股'));
''')


def test_only_new_verified_checkpoint_jobs_offer_resume():
    run_js(r'''
const job={job_id:'cancelled-native',kind:'model_training',status:'cancelled',
  spec:{parameters:defaults},progress_detail:{stage:'native_source_prepare',current:10,total:60,
    children:{raw_source_run_id:'raw-input',source_native_run_id:'native-input'}}};
assert(!trainingTest.renderJob(job).includes('data-training-resume'));
job.progress_detail.checkpoint_available=true;
assert(trainingTest.renderJob(job).includes('继续结构准备与训练'));
job.status='running';assert(!trainingTest.renderJob(job).includes('data-training-resume'));
''')


def test_resume_click_cold_checks_then_submits_a_new_frozen_plan():
    run_js(r'''
(async()=>{
const job={job_id:'cancelled-native',kind:'model_training',status:'cancelled',
  spec:{parameters:defaults},progress_detail:{checkpoint_available:true,
    children:{raw_source_run_id:'raw-input',source_native_run_id:'native-input'}}};
trainingTest.state.jobs.set(job.job_id,job);
custom=async req=>{
  if(req.path==='/api/training/preflight')return {valid:true,plan:{plan_id:'resume-plan'}};
  if(req.path==='/api/training/jobs')return {job_id:'resumed',kind:'model_training',status:'queued'};
  if(req.path==='/api/tasks/resumed')return {job_id:'resumed',kind:'model_training',status:'queued'};
};
const target={dataset:{trainingResume:job.job_id},closest(selector){return selector==='[data-training-resume]'?this:null;}};
await element('training-jobs').handlers.click({target});
const preflight=requests.find(r=>r.path==='/api/training/preflight');
assert.deepEqual(preflight.body,{parameters:defaults,raw_source_run_id:'raw-input',resume_native_run_id:'native-input'});
const submitted=requests.find(r=>r.path==='/api/training/jobs');
assert.deepEqual(submitted.body,{plan_id:'resume-plan',parameters:defaults});
assert.equal(storage.get('khquant_current_training_job'),'resumed');
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
