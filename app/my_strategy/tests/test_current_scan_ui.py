"""Current scan coverage, unavailable scores, filters and escaped diagnostics.

Retains the rendering contract from the archived research_ui module without
restoring its fixed-horizon shadow model or old entry policy choices.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_current_scan_preserves_failure_dates_and_filterable_unavailable_scores():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped scan rendering checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
global.KHQuantChartEvidence=require(require('path').join(require('path').dirname(process.argv[1]),'chart_evidence.js'));
const elements=new Map();function element(id){if(!elements.has(id))elements.set(id,{value:'',textContent:'',innerHTML:'',hidden:false,disabled:false,querySelector:()=>element(id+'-child')});return elements.get(id);}
global.document={getElementById:element};
global.window={addEventListener:()=>{}};
const source=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);','globalThis.scanTest={state,renderScan};');
vm.runInThisContext(source);const {state,renderScan}=scanTest;
element('scan-filter').value='all';
const first={symbol:'000001.SZ',name:'测试<script>',model_family:'current',strategy_version:'trend_consensus',category:'watch',action:'WAIT',
  data_end:'2026-10-12',point_confirmed_at:null,reason:'风险<detail>',agents:{
    ma:{kind:'trend_probability',status:'model_unavailable',p_up:.5,p_down:.2,p_neutral:.3},
    high_return:{kind:'model_probability',evaluation_status:'model_unavailable',probability:null}},holding:{shares:0}};
const second={...first,symbol:'600519.SH',name:'排除',category:'excluded',data_end:'2026-10-09',reason:'missing_target_date'};
state.scan={model_family:'current',strategy_version:'trend_consensus',data_end:'2026-10-12',
  data_range:{requested_end:'2026-10-12',end:'2026-10-12'},coverage:{requested:3,success:2,failed:1},
  category_counts:{watch:1,excluded:1},model_gate:{passed:false},rows:[first,second],
  failures:[{symbol:'000002.SZ',error:'缺少行情<script>'}]};
renderScan();let html=element('scan-rows').innerHTML;
assert(html.includes('继续观察'));assert(html.includes('趋势概率未评分'));
assert(html.includes('行情 2026-10-09'));assert(html.includes('结构确认 未确认'));
assert(html.includes('测试&lt;script&gt;'));assert(html.includes('风险&lt;detail&gt;'));assert(!html.includes('<script>'));
assert(!html.includes('10日研究概率'));assert(!html.includes('高收益区间概率 0%'));
assert(element('scan-coverage').textContent.includes('覆盖 2 / 3 只成功 · 失败 1'));
assert(element('scan-coverage').textContent.includes('请求截止 2026-10-12 · 实际行情 2026-10-12'));
assert(element('scan-failures-child').innerHTML.includes('000002.SZ'));
assert(element('scan-failures-child').innerHTML.includes('缺少行情&lt;script&gt;'));
element('scan-filter').value='excluded';renderScan();
assert(!element('scan-rows').innerHTML.includes('000001.SZ'));assert(element('scan-rows').innerHTML.includes('600519.SH'));
element('scan-filter').value='all';element('scan-search').value='测试';renderScan();
assert(element('scan-rows').innerHTML.includes('000001.SZ'));assert(!element('scan-rows').innerHTML.includes('600519.SH'));
element('scan-search').value='';
const scored={...first,symbol:'600519.SH',name:'贵州茅台',strategy_version:'reference_position_v1',
  analysis_mode:'reference',position_reference:{schema:'reference_position_v1',status:'stale',stale:true,
    target_position_fraction:0,previous_reference_target:0,change_fraction:0,
    probability_scored:true,risk_geometry_available:false,predictive_gate_passed:false,
    reason_codes:['reference_risk_geometry_unavailable','reference_predictive_gate_pending']},
  reference_risk_plan:{status:'unavailable'},
  agents:{ma:{status:'scored',probability_scored:true,p_up:.2854868173599243,p_down:.41692468523979187,
    p_neutral:.2975884974002838,risk_geometry_available:false,predictive_gate_passed:false}}};
state.scan={...state.scan,analysis_mode:'reference',strategy_version:'reference_position_v1',rows:[scored]};
renderScan();html=element('scan-rows').innerHTML;
for(const text of ['上涨 28.55%','下跌 41.69%','中性 29.76%','风险计划不可用','预测基线未通过']) assert(html.includes(text),text);
assert(!html.includes('缺兼容模型'));assert(!html.includes('B · 参考增仓'));assert(!html.includes('S · 参考减仓'));
'''
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
