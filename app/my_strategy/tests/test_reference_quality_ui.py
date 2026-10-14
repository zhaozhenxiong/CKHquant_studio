"""Independent reference population scores must never borrow trading scores."""
from pathlib import Path
import shutil
import subprocess
import pytest

STATIC = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js'


def run_js(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for actual shipped reference renderer')
    result = subprocess.run([node, '-e', script, str(STATIC)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


FIXTURE = r'''
const assert=require('node:assert/strict'),p=require('path'),ui=require(p.join(process.argv[1],'chart_evidence.js'));
const date='2026-10-09',symbol='000001.SZ',available_at=date+'T15:00:00+08:00';
const risk={status:'available',reference_price:35,reference_price_date:date,stop_price:31.5,invalidation_price:32,stop_distance_fraction:.1,risk_budget_fraction:.01,risk_position_cap_fraction:.1};
const identity={model_run_id:'reference-run',head:'reference_quality',checkpoint:'policy-fold',available_at:'2026-10-08T15:00:00+08:00',actual_training_completed_at:'2026-10-08T16:00:00+08:00'};
const plan={status:'increase_exposure',action:'BUY',target_position_fraction:.08,current_position_fraction:.02,change_fraction:.06,quality_status:'scored',reference_quality_probability:.8,reference_quality_identity:identity,reference_population_hash:'population-proof',reference_quality_qualification_hash:'qualification-proof',reference_quality_threshold:.1,reference_buy_total_score:.58,reference_buy_total_threshold:.5,reason_codes:[]};
const row={date,symbol,available_at,analysis_mode:'reference',reference_only:true,reference_close:35,reference_price_date:date,action:'BUY',emitted:true,buy_base_score:.5,sell_base_score:.1,reference_risk_plan:risk,position_reference:plan,source_reference_points:[],agents:{reference_quality:{kind:'model_probability',status:'scored',in_scope:true,probability:.8,probability_scope:'independent_reference_buy_opportunity_fee_net_quality',applied_to_reference_position:true,applied_to_actual_entry:false},high_return:{probability:.99,status:'scored',exit_probability:.97},ma:{kind:'trend_probability',status:'model_unavailable',p_up:null,p_down:null,p_neutral:null},structure:{status:'scored'},wyckoff:{status:'scored'}}};
'''


def test_reference_q_probability_threshold_total_and_evidence_remain_separate():
    run_js(FIXTURE + r'''
const saved=JSON.stringify(row),view=ui.referenceQualityView(row);
assert.equal(view.probability,.8);assert.equal(view.qualityThreshold,.1);assert.equal(view.total,.58);assert.equal(view.totalThreshold,.5);assert.equal(view.approved,true);
const html=ui.positionHtml(row);
for(const phrase of ['独立参考质量Q','参考质量概率','80%','原质量门槛','10%','58 / 50 分','完整费用后至少8%且2R','参考仓位资格通过','参考目标 8%','建议变化 6 个百分点','reference-run','policy-fold','population-proof','qualification-proof'])assert(html.includes(phrase),phrase);
for(const phrase of ['99%','97%','持仓退出概率','实际入场概率','入场概率 80%'])assert(!html.includes(phrase),phrase);
assert.equal(ui.currentAgentView(row,'reference_quality').summary,'参考质量概率 80%');
assert.equal(ui.currentAgentView(row,'reference_quality').exit,'');
assert(!ui.currentAgentView(row,'high_return').summary.includes('99'));
assert.equal(JSON.stringify(row),saved);
''')


def test_null_pending_identity_failure_and_zero_are_never_replaced_by_actual_entry_probability():
    run_js(FIXTURE + r'''
const pending={...row,reference_quality_probability:.88,reference_buy_total_score:.588,position_reference:{...plan,status:'pending',target_position_fraction:null,quality_status:'pending',reference_quality_probability:null,reference_buy_total_score:null,reason_codes:['no_strictly_prior_reference_quality_head']}};
let view=ui.referenceQualityView(pending),html=ui.positionHtml(pending);
assert.equal(view.probability,null);assert.equal(view.total,null);assert.equal(view.approved,false);
assert(html.includes('参考质量概率'));assert(html.includes('未评分'));assert(html.includes('推荐待资格核验'));assert(html.includes('该日缺少严格先验参考质量头'));
for(const phrase of ['80%','88%','99%','参考目标','建议变化'])assert(!html.includes(phrase),phrase);
html=ui.positionHtml({...row,position_reference:{...plan,quality_status:'identity_error'}});assert(html.includes('参考评分来源核验失败'));assert(!html.includes('参考目标'));assert(!html.includes('80%'));
const zero={...pending,position_reference:{...pending.position_reference,quality_status:'scored',reference_quality_probability:0,reference_buy_total_score:0,reason_codes:['reference_quality_below_threshold','reference_total_support_below_threshold']}};
view=ui.referenceQualityView(zero);assert.equal(view.probability,0);assert.equal(view.total,0);
html=ui.positionHtml(zero);assert(html.includes('<strong>0%</strong>'));assert(html.includes('0 / 50 分'));assert(html.includes('参考质量概率未达到原门槛'));assert(!html.includes('参考目标'));
const oldActual={...row,position_reference:{status:'pending',target_position_fraction:null},agents:{high_return:row.agents.high_return}};
assert.equal(ui.referenceQualityView(oldActual).present,false);assert(!ui.positionHtml(oldActual).includes('99%'));
''')


def test_popup_replay_history_and_compact_scan_share_one_q_consumer():
    run_js(FIXTURE + r'''
const consensus={version:'trend_consensus',config:{version:'trend_consensus'},rows:[row]},evidence={rows:[row]};
const entry={original_date:date,event_date:date,displayed:true,row:{symbol,date,reference_only:true,reference_layer:'consensus',source:'reference_consensus',action:'BUY',reference_price:35,reference_price_date:date,first_available_at:available_at}};
const group={kind:'intention',visible:true,time:date,action:'BUY',sources:['reference_consensus'],entries:[entry]};
for(const frequency of ['日线','周线','月线']) {
 const html=ui.popupHtml({evidence,consensus,groups:[group],barDate:date,frequency});
 assert(html.includes('独立参考质量Q'));assert(html.includes('参考质量Q 80%'));assert(html.includes('58 / 50 分'));assert(!html.includes('99%'));
}
const history=ui.historyHtml([row]);assert(history.includes('参考质量Q 80%'));assert(history.includes('质量门槛 10%'));assert(history.includes('参考总支持 58 分 / 门槛 50 分'));
const scan=ui.positionSummaryHtml({...row,reference_quality_probability:.99});assert(scan.includes('参考质量Q 80%'));assert(!scan.includes('99%'));
const latest=ui.positionHtml({next_session_decision:row});assert(latest.includes('参考目标 8%'));assert(latest.includes('参考质量概率'));
const agent=ui.agentsHtml(row,false,row);assert(agent.includes('独立参考质量Q'));assert(agent.includes('参考质量Q 80%'));
assert(ui.agentOverviewHtml(row,row).includes('三路参考评分'));
''')


def test_reference_q_identity_and_unknown_rejection_codes_are_escape_safe():
    run_js(FIXTURE + r'''
const hostile={...row,position_reference:{...plan,reference_population_hash:'<img src=x onerror=alert(1)>',reference_quality_identity:{...identity,model_run_id:'<script>secret</script>'},reason_codes:['unknown_reference_contract_rejection_<img>']}};
const html=ui.referenceQualityHtml(hostile);
assert(html.includes('&lt;img'));assert(html.includes('&lt;script&gt;'));assert(!html.includes('<script>'));assert(!html.includes('<img'));
assert(html.includes('unknown_reference_contract_rejection_&lt;img&gt;'));
''')
