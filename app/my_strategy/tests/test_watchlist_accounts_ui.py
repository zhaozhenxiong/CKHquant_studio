"""Exercise the shipped account workspace and immutable prediction view contracts."""
from html.parser import HTMLParser
import shutil
import subprocess

import pytest

from test_czsc_display_dates_ui import HARNESS, STATIC


SETUP = HARNESS + r'''
const memory=new Map();global.sessionStorage={getItem:k=>memory.get(k)||null,setItem:(k,v)=>memory.set(k,String(v)),removeItem:k=>memory.delete(k)};
document.createElement=tag=>newElement(tag);document.hidden=false;
chart.remove=()=>{};chart.timeScale().setVisibleRange=()=>{};chart.timeScale().unsubscribeVisibleLogicalRangeChange=()=>{};
ResizeObserver.prototype.disconnect=()=>{};
for(const [id,value] of Object.entries({'light-feature-profile':'ma_trend_v1','light-score-type':'mlp','light-score-variant':'A','light-training-symbols':'000002.SZ','light-training-device':'cpu','light-symbol':'600519.SH','light-predict-date':'2000-12-29','light-lookback':'128','light-horizon':'5','light-samples':'8','light-seed':'42','light-device':'cpu','light-symbols':'000001.SZ','light-start':'2000-01-01','light-end':'2000-12-29','light-cash':'1000000','light-sim-feature-profile':'ma_trend_v1','light-sim-score-type':'mlp','light-sim-lookback':'256','light-sim-horizon':'10','light-sim-samples':'32','light-sim-seed':'7','light-sim-device':'cpu','light-candidate-days':'3','light-fusion':'evidence','light-cost-stress':'1','light-score-end':'1999-12-31','light-account':'structure_rules','light-chart-mode':'equity','light-daily-date':'2001-01-02'}))element(id).value=value;
element('view-simulation').hidden=true;element('view-watchlist').hidden=true;element('watchlist-accounts-panel').hidden=true;
const calls=[];let answer=async(url,options)=>{
 if(url==='/api/light/status')return {assets:{ready:true},score_candidates:[],jobs:[]};
 if(url==='/api/light/saved')return {results:[]};
 if(url==='/api/light/accounts')return {accounts:{},last_daily:null};
 if(url==='/api/light/score/observations')return {manifests:[]};
 if(url==='/api/data/status')return {latest_date:'2001-01-02'};
 if(url.startsWith('/api/light/jobs/'))return {job_id:'one',kind:'simulation',status:'running',progress:{}};
 if(options.method==='POST')return {job_id:'one',kind:url.endsWith('/simulate')?'simulation':url.endsWith('/predict')?'prediction':'score_training',status:'queued'};
 throw Error('unexpected '+url);
};
global.fetch=async(url,options={})=>{calls.push({url,options});const value=await answer(url,options);return {ok:true,json:async()=>value,text:async()=>JSON.stringify(value)};};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'light_research.js'),'utf8'));
const value={result_id:'saved-account',saved:true,version:'agent_mlp_sim_v1',symbols:['000001.SZ'],start:'2000-01-01',end:'2000-12-29',status:'completed',identity:{cash:1000000},comparisons:{structure_rules:{daily:[],trades:[],orders:[],metrics:{}}}};
'''


def run(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for UI behavior checks')
    result = subprocess.run([node, '-e', SETUP+'\n(async()=>{\n'+script+'\n})().catch(e=>{console.error(e);process.exitCode=1;});', str(STATIC)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_account_modules_have_one_owner_and_prediction_precedes_training():
    class Tree(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack=[]; self.parents={}; self.ids=[]
        def handle_starttag(self, tag, attrs):
            attrs=dict(attrs); identity=attrs.get('id'); self.ids.append(identity) if identity else None
            if identity:self.parents[identity]=tuple(self.stack)
            if tag not in {'input','meta','link','img','br','hr'}:self.stack.append((tag,identity))
        def handle_endtag(self, tag):
            for i in range(len(self.stack)-1,-1,-1):
                if self.stack[i][0]==tag:self.stack=self.stack[:i];break
    html=(STATIC.parent/'index.html').read_text();tree=Tree();tree.feed(html)
    assert len(tree.ids)==len(set(tree.ids))
    for identity in ('light-sim-form','light-saved','light-daily','light-results','light-export','light-trade-chart'):
        assert ('div','watchlist-accounts-panel') in tree.parents[identity]
        assert ('section','view-simulation') not in tree.parents[identity]
    assert html.index('id="light-predict-form"')<html.index('id="light-prediction"')<html.index('id="light-training-panel"')
    assert '预测与模型' in html and 'id="open-account-analysis"' in html


def test_route_parser_preserves_legacy_routes_and_account_deep_links():
    run(r'''
for(const name of ['simulation','watchlist','scan']){location.hash='#'+name;assert.equal(KHQuantNavigation.parse().view,name);}
location.hash='#watchlist?tab=accounts&result=saved%2Bkey';assert.deepEqual(KHQuantNavigation.parse(),{view:'watchlist',tab:'accounts',result:'saved+key'});
location.hash='#watchlist?tab=unknown&result=unused';assert.equal(KHQuantNavigation.parse().tab,'stocks');assert.equal(KHQuantNavigation.parse().result,null);
assert(!calls.some(c=>c.options.method==='POST'));
''')


def test_simulation_parameters_do_not_read_prediction_or_training_drafts():
    run(r'''
const before=KHQuantLightResearch.simulationSpec();
for(const id of ['light-lookback','light-horizon','light-samples','light-seed','light-device','light-feature-profile','light-score-type','light-training-symbols'])element(id).value='changed';
assert.deepEqual(KHQuantLightResearch.simulationSpec(),before);
assert.equal(before.lookback,256);assert.equal(before.horizon,10);assert.equal(before.samples,32);assert.equal(before.seed,7);assert.equal(before.feature_profile,'ma_trend_v1');assert.deepEqual(before.symbols,['000001.SZ']);
''')


def test_training_has_own_pool_and_bringing_accounts_never_starts_compute():
    run(r'''
KHQuantLightResearch.transferToAccounts({symbols:['600519.SH'],end:'2000-11-30'});
assert.equal(element('light-training-symbols').value,'000002.SZ');assert.equal(element('light-predict-date').value,'2000-12-29');assert(location.hash.startsWith('watchlist?tab=accounts'));assert(!calls.some(c=>c.options.method==='POST'));
assert.throws(()=>KHQuantLightResearch.transferToAccounts({symbols:Array.from({length:11},(_,i)=>String(i))}),/1至10/);
await fire('light-train','click');await settle();const posted=calls.find(c=>c.url.endsWith('/score/train'));
assert.deepEqual(JSON.parse(posted.options.body).symbols,['000002.SZ']);
''')


def test_analysis_transfer_preserves_frozen_model_and_only_target_draft():
    run(r'''
const context={symbol:'000001.SZ',start:'2000-01-01',end:'2000-12-29',analysis_method:'structure_kronos_ma_l2',score_model_id:'synthetic-frozen-model',lookback:128,horizon:5,samples:8,seed:3,device:'cpu'};
const training=element('light-training-symbols').value,prediction=element('light-symbol').value;
KHQuantLightResearch.transferToAccounts(context);assert.equal(element('light-score-model').value,context.score_model_id);assert.equal(element('light-sim-seed').value,'3');assert.equal(element('light-symbol').value,prediction);assert.equal(element('light-training-symbols').value,training);
const before=KHQuantLightResearch.simulationSpec();KHQuantLightResearch.transferToPrediction({...context,symbol:'600519.SH',seed:9});assert.deepEqual(KHQuantLightResearch.simulationSpec(),before);assert.equal(location.hash,'simulation');assert.equal(element('light-seed').value,'9');assert.equal(calls.filter(c=>c.options.method==='POST').length,0);
''')


def test_account_job_busy_cancel_and_daily_date_use_correct_owner():
    run(r'''
const startButtons=[element('account-submit'),element('predict-submit')];document.querySelectorAll=selector=>selector.includes('button[data-start]')?startButtons:[];
await fire('light-sim-form','submit',{preventDefault(){}});await settle();assert(startButtons.every(button=>button.disabled));assert.equal(element('light-account-cancel').hidden,false);assert.equal(element('light-cancel').hidden,true);assert(element('light-progress').innerHTML.includes('#watchlist?tab=accounts'));
await fire('light-cancel','click');assert(!calls.some(c=>c.url.endsWith('/cancel')));
await fire('light-account-cancel','click');assert.equal(calls.filter(c=>c.url==='/api/light/jobs/one/cancel').length,1);
await fire('light-sim-form','submit',{preventDefault(){}});assert.equal(calls.filter(c=>c.url==='/api/light/simulate').length,1);
''')


def test_observation_progress_distinguishes_total_days_and_per_day_stocks():
    run(r'''
const old=answer;answer=async(url,options)=>url==='/api/light/jobs/one'?{job_id:'one',kind:'simulation',status:'running',progress:{stage:'agent_observation',phase:'model_inference',symbol:'002156.SZ',date:'2026-07-17',completed:0,total:1,completed_sessions:128,total_sessions:188,observations_completed:103,cpu_workers:2}}:old(url,options);
await fire('light-sim-form','submit',{preventDefault(){}});await settle();
const text=element('light-account-progress').textContent;
assert(text.includes('模型推理'));assert(text.includes('已评估交易日 128/188'));assert(text.includes('累计预测 103 次'));assert(text.includes('当日股票 0/1'));assert(text.includes('CPU输入准备 2 线程'));assert(!text.includes('进程'));assert(!text.includes('agent_observation'));
''')


def test_structure_progress_distinguishes_processes_from_input_threads():
    run(r'''
const old=answer;answer=async(url,options)=>url==='/api/light/jobs/one'?{job_id:'one',kind:'simulation',status:'running',progress:{stage:'structure',symbol:'002156.SZ',cpu_workers:2}}:old(url,options);
await fire('light-sim-form','submit',{preventDefault(){}});await settle();
const text=element('light-account-progress').textContent;
assert(text.includes('准备结构与量价'));assert(text.includes('CPU准备 2 进程'));assert(!text.includes('线程'));assert(!text.includes('已评估交易日'));assert(!text.includes('累计预测'));
''')


def test_legacy_observation_progress_does_not_invent_missing_totals():
    run(r'''
const old=answer;answer=async(url,options)=>url==='/api/light/jobs/one'?{job_id:'one',kind:'simulation',status:'running',progress:{stage:'agent_observation',date:'2026-07-17',completed:0,total:1}}:old(url,options);
await fire('light-sim-form','submit',{preventDefault(){}});await settle();
const text=element('light-account-progress').textContent;assert(text.includes('当日股票 0/1'));assert(!text.includes('已评估交易日'));assert(!text.includes('累计预测'));assert(!text.includes('CPU准备'));
''')


def test_daily_uses_its_explicit_date_and_saved_open_is_read_only():
    run(r'''
const old=answer;answer=async(url,options)=>url.startsWith('/api/light/results/')?value:old(url,options);
await KHQuantLightResearch.openSaved('saved-account');assert.equal(calls.filter(c=>c.options.method==='POST').length,0);assert(location.hash.includes('result=saved-account'));assert.equal(element('light-new-account').open,false);
element('light-end').value='2099-01-01';await fire('light-daily','click');await settle();assert.deepEqual(JSON.parse(calls.find(c=>c.url==='/api/light/daily').options.body),{required_price_date:'2001-01-02'});
''')


def test_task_result_goes_to_accounts_and_late_saved_response_cannot_replace_newer():
    run(r'''
let finishOld;const old=answer;answer=async(url,options)=>url==='/api/light/results/old'?await new Promise(resolve=>finishOld=resolve):url.startsWith('/api/light/results/')?{...value,result_id:'new'}:url==='/api/light/jobs/read'?{job_id:'read',kind:'simulation',result:{...value,saved:false}}:old(url,options);
const pending=KHQuantLightResearch.openSaved('old');await settle();await KHQuantLightResearch.openSaved('new');finishOld({...value,result_id:'old'});await pending;assert(location.hash.includes('result=new'));
await KHQuantLightResearch.openTask('read');assert.equal(location.hash,'watchlist?tab=accounts');assert(!calls.some(c=>c.options.method==='POST'));
''')


def test_prediction_chart_uses_verified_history_and_never_recomputes_samples():
    run(r'''
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'prediction_chart.js'),'utf8'));
const original={as_of:'2000-12-29',input:{lookback:128},history:[{date:'2000-12-29',open:10,high:11,low:9,close:10,volume:100,amount:1000}],history_verification:{status:'verified',input_sha256_match:true},forecast:[{date:'2001-01-02',close:10.5,close_p10:9.5,close_p90:11.5}],samples:[[1],[2]]};
const before=JSON.stringify(original),series=KHQuantPredictionChart.prepareSeries(original);assert.equal(series.history.length,1);assert.equal(series.forecast[0].middle,10.5);assert.equal(JSON.stringify(original),before);
assert.equal(KHQuantPredictionChart.prepareSeries({...original,history_verification:{status:'unavailable'}}).history.length,0);
assert.throws(()=>KHQuantPredictionChart.prepareSeries({...original,forecast:[{date:'2000-12-29',close:10}]}),/交易日/);
assert.throws(()=>KHQuantPredictionChart.prepareSeries({...original,history:[{...original.history[0],date:'2001-01-02'}]}),/历史/);
''')


def test_prediction_view_race_and_display_window_make_only_read_requests():
    run(r'''
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'prediction_chart.js'),'utf8'));
let finish;const requests=[],base={symbol:'000001.SZ',as_of:'2000-12-29',input:{as_of:'2000-12-29'},forecast:[{date:'2001-01-02',close:10,close_p10:9,close_p90:11}],summary:{terminal_return_median:.1},quality:{requested_samples:32},result_ref:{cache_key:'outer-one'}};
const read=async url=>{requests.push(url);if(url.includes('outer-one'))return await new Promise(resolve=>finish=resolve);return {...base,prediction_ref:'outer-two',history:[],history_verification:{status:'unavailable',reason:'第二结果历史缺失'}};};
const first=KHQuantPredictionChart.render(base,read);await settle();await KHQuantPredictionChart.render({...base,result_ref:{cache_key:'outer-two'}},read);finish({...base,prediction_ref:'outer-one',history:[],history_verification:{status:'unavailable',reason:'旧结果'}});await first;
assert(element('light-prediction-history-state').textContent.includes('第二结果'));const before=requests.length;await fire('light-history-window','change');await fire('light-history-reset','click');assert.equal(requests.length,before);assert(requests.every(url=>url.startsWith('/predictions/outer-')));assert.equal(calls.filter(c=>c.options.method==='POST').length,0);
''')


def test_saved_failure_hides_old_result_and_same_route_can_retry():
    run(r'''
const old=answer;answer=async(url,options)=>url.startsWith('/api/light/results/')?value:old(url,options);
await KHQuantLightResearch.openSaved('saved-account');assert.equal(element('light-results').hidden,false);
const previousFetch=global.fetch;let failed=true;global.fetch=async(url,options)=>url==='/api/light/results/missing'&&failed?{ok:false,json:async()=>({detail:'结果已过期'})}:previousFetch(url,options);
location.hash='#watchlist?tab=accounts&result=missing';await KHQuantLightResearch.openSaved('missing',{navigateTo:false});assert.equal(element('light-results').hidden,true);assert.equal(element('light-enable-daily').disabled,true);assert(element('light-account-error').textContent.includes('过期'));
failed=false;KHQuantLightResearch.onRoute({view:'watchlist',tab:'accounts',result:'missing'});await settle();assert.equal(element('light-results').hidden,false);assert.equal(element('light-account-error').hidden,true);
''')


def test_pending_saved_read_does_not_steal_navigation_after_user_leaves():
    run(r'''
let finish;const old=answer;answer=async(url,options)=>url==='/api/light/results/slow'?await new Promise(resolve=>finish=resolve):old(url,options);
location.hash='#watchlist?tab=accounts';const pending=KHQuantLightResearch.openSaved('slow');await settle();location.hash='#simulation';finish({...value,result_id:'slow'});await pending;
assert.equal(location.hash,'#simulation');assert(!calls.some(c=>c.options.method==='POST'));
''')


def test_old_enabled_account_does_not_imply_paper_qualification():
    run(r'''
const old=answer;answer=async(url,options)=>url==='/api/light/accounts'?{accounts:{legacy:{account_id:'legacy',enabled:true,symbols:['000001.SZ'],last_date:'2000-12-29'}}}:old(url,options);
await KHQuantLightResearch.refresh();assert(element('light-accounts-state').innerHTML.includes('用途未记录'));assert(!element('light-accounts-state').innerHTML.includes('纸上账户'));
''')


def test_account_charts_use_theme_colors_and_recreate_after_hidden_page():
    run(r'''
const created=[];let removed=0;LightweightCharts.createChart=(node,options)=>{created.push({node,options});return chart;};chart.remove=()=>{removed++;};
const old=answer;answer=async(url,options)=>url.startsWith('/api/light/results/')?value:old(url,options);
element('view-watchlist').hidden=false;element('watchlist-accounts-panel').hidden=false;
await KHQuantLightResearch.openSaved('saved-account',{navigateTo:false});assert.equal(created.length,2);assert(created.every(call=>call.options.layout.textColor==='#123456'&&call.options.grid.vertLines.color==='#123456'));
element('watchlist-accounts-panel').hidden=true;KHQuantLightResearch.onRoute({view:'watchlist',tab:'stocks'});assert.equal(removed,2);
global.getComputedStyle=()=>({getPropertyValue:name=>name==='--panel'?'#111111':'#eeeeee'});element('watchlist-accounts-panel').hidden=false;KHQuantLightResearch.onRoute({view:'watchlist',tab:'accounts'});assert.equal(created.length,4);assert.equal(created.at(-1).options.layout.background.color,'#111111');assert.equal(created.at(-1).options.layout.textColor,'#eeeeee');
''')


def test_visible_chart_window_changes_only_the_display_range():
    run(r'''
const plotted=[],ranges=[];let crosshair;
const ts={fitContent(){},timeToCoordinate:()=>50,subscribeVisibleLogicalRangeChange(){},setVisibleRange:r=>ranges.push(r)};
LightweightCharts.createChart=()=>({remove(){},timeScale:()=>ts,subscribeCrosshairMove:fn=>crosshair=fn,addCandlestickSeries:()=>({setData:rows=>plotted.push({kind:'history',rows})}),addLineSeries:()=>({setData:rows=>plotted.push({kind:'forecast',rows})})});
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'prediction_chart.js'),'utf8'));
element('view-simulation').hidden=false;element('light-prediction').hidden=false;
const history=Array.from({length:128},(_,index)=>({date:new Date(Date.UTC(2000,8,1+index)).toISOString().slice(0,10),open:10,high:12,low:9,close:11,volume:100,amount:1000})),as_of=history.at(-1).date;
const result={symbol:'000001.SZ',as_of,input:{as_of,lookback:128,price_basis:'unadjusted',volume_unit:'shares',amount_unit:'CNY'},forecast:[{date:'2001-01-08',close:11.5,close_p10:10,close_p90:13}],summary:{terminal_return_median:.05},quality:{requested_samples:32},result_ref:{cache_key:'outer'}};
let reads=0;const before=JSON.stringify(result);await KHQuantPredictionChart.render(result,async()=>{reads++;return {...result,prediction_ref:'outer',history,history_verification:{status:'verified',input_sha256_match:true}};});
assert.equal(plotted.filter(item=>item.kind==='history').at(-1).rows.length,128);assert.equal(ranges.at(-1).from,history.at(-60).date);
element('light-history-window').value='30';await fire('light-history-window','change');assert.equal(ranges.at(-1).from,history.at(-30).date);
element('light-history-window').value='all';await fire('light-history-window','change');assert.equal(ranges.at(-1).from,history[0].date);assert.equal(reads,1);assert.equal(JSON.stringify(result),before);
crosshair({time:as_of});assert(element('light-chart-readout').textContent.includes('shares'));assert(element('light-chart-readout').textContent.includes('CNY'));crosshair({time:'2001-01-08'});assert(element('light-chart-readout').textContent.includes('P10 10 / P90 13'));
''')


def test_forecast_boundary_stays_above_canvas_without_intercepting_pointer():
    css=(STATIC.parent/'css/light_research.css').read_text()
    declaration=css.split('.light-forecast-boundary{',1)[1].split('}',1)[0]
    assert 'z-index:3' in declaration
    assert 'pointer-events:none' in declaration


def test_prediction_boundary_label_stays_inside_host_at_both_edges():
    run(r'''
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'prediction_chart.js'),'utf8'));
for(const width of [280,339,760,851])for(const coordinate of [0,1,8,width/2,width-1,width])for(const labelWidth of [80,160,200]){
 const left=KHQuantPredictionChart.boundaryLabelLeft(coordinate,width,labelWidth);
 assert(left>=0);assert(left+labelWidth<=width,`label exceeds host: ${left}+${labelWidth}>${width}`);
}
''')


def test_original_prediction_entry_preserves_get_with_factory_abort_signal():
    run(r'''
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'prediction_chart.js'),'utf8'));
const savedPrediction={symbol:'000001.SZ',as_of:'2000-12-29',result_ref:{cache_key:'synthetic-prediction-key'},forecast:[{date:'2001-01-02',close:10,close_p10:9,close_p90:11}]};
const previous=answer;answer=async(url,options)=>url==='/api/light/jobs/prediction-read'?{job_id:'prediction-read',kind:'prediction',status:'succeeded',result:savedPrediction}:url.endsWith('/chart')?{...savedPrediction,prediction_ref:'synthetic-prediction-key',history:[],history_verification:{status:'unavailable',reason:'合成原预测'}}:previous(url,options);
await KHQuantLightResearch.openTask('prediction-read');await settle();
const request=calls.find(call=>call.url==='/api/light/predictions/synthetic-prediction-key/chart');
assert(request);assert.equal(request.options.method,'GET');assert.equal(request.options.body,undefined);assert(request.options.signal instanceof AbortSignal);assert(element('light-prediction-history-state').textContent.includes('合成原预测'));assert(!calls.some(call=>call.options.method==='POST'));
''')
