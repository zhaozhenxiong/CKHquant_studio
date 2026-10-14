"""Check the shipped event prices and latest-session guidance without future inference."""
from pathlib import Path
import shutil
import subprocess

import pytest




def test_failed_analysis_clears_prior_buy_guidance():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped failed-analysis checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{textContent:'',innerHTML:'',hidden:false,
    querySelector:(selector) => element(id+selector),replaceChildren() { this.innerHTML=''; }});
  return elements.get(id);
}
global.document={getElementById:element};
global.window={addEventListener:()=>{}};
global.sessionStorage={getItem:() => null};
global.fetch=async () => { throw new Error('network unavailable'); };
global.KHQuantMarkers={describe:() => ''};
global.KHQuantChartEvidence=require(require('path').join(require('path').dirname(process.argv[1]),'chart_evidence.js'));
const source=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.failureTest={state,renderEvents,analyze};');
vm.runInThisContext(source);
const {state,renderEvents,analyze}=failureTest;
(async () => {
  state.analysis={symbol:'600027.SH',analysis_mode:'reference',reference_points:[{symbol:'600027.SH',date:'2026-09-30',first_available_at:'2026-09-30T15:00:00+08:00',action:'BUY',reference_only:true,reference_price:4.95,reference_price_date:'2026-09-30'}],events:[{action:'BUY',time:'2026-09-30',reference_price:4.95}],
    next_session_decision:{action:'BUY',reference_close:4.95,reference_price_date:'2026-09-30',
      next_market_session:'2026-10-08',basis:'最新收盘触发开多'}};
  renderEvents();
  assert(element('event-context').innerHTML.includes('收盘参考价 4.95 元'));
  assert(element('event-list').innerHTML.includes('买入参考价 4.95 元'));
  await analyze({symbol:'600027.SH',as_of:'2026-09-30'});
  assert.equal(state.analysis,null);
  assert.equal(element('event-context').innerHTML,'参考判断不可用，请重新分析。');
  assert(!element('event-context').innerHTML.includes('买入'));
  assert(!element('event-context').innerHTML.includes('4.95'));
  assert(element('event-list').innerHTML.includes('分析未完成'));
  assert(!element('event-list').innerHTML.includes('买入参考价'));
  assert.equal(element('event-count').textContent,'0');
  assert.equal(element('analysis-state').textContent,'分析失败');
  assert.equal(element('page-errorspan').textContent,'network unavailable');
  assert.equal(element('analysis-form[type=submit]').disabled,false);
})().catch((error) => { console.error(error); process.exitCode=1; });
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
