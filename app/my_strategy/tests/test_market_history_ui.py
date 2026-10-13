"""Saved scan charts preserve missingness, filters and asynchronous run identity."""
from pathlib import Path
import shutil
import subprocess

import pytest


UI = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/market_history.js"


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for market history UI")
    result = subprocess.run([node, "-e", SETUP + script, str(UI)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


SETUP = r"""
const assert=require('node:assert/strict'),ui=require(process.argv[1]);
function day(date,missing=false){return {date,universe_size:10,
 coverage:{price_available:missing?0:8,input_available:missing?0:7,model_scored:missing?0:6,failed:1,missing_price:missing?10:2},
 counts:{direction_buy:missing?null:4,direction_sell:missing?null:2,candidates:3,risk_blocked:1,hr_scored_passed:0,hr_scored_rejected:1,hr_unscored:9,account_buy:0,account_sell:0,filled_buy:0,filled_sell:0,holding:0},
 hr_status:{no_candidate:7,model_unavailable:2,scored_rejected:1},
 events:{new_total:1,active_total:2,invalidated:0,new_by_type:{spring:1},active_by_type:{spring:2}},
 mean_support:{continuous_demand:.5,continuous_supply:.3,event_demand:0,event_supply:0,effective_demand:null,effective_supply:null}};}
const summary={version:'market_history_v1',run_id:'run-a',universe_size:10,dates:['2026-10-08','2026-10-09'],daily:[day('2026-10-08',true),day('2026-10-09')],
 classification:{mode:'current_snapshot',classified_symbols:8,unknown_symbols:2},unknown_sector_id:'unknown',
 sectors:[{id:'unknown',name:'未知板块',member_count:2,daily:[{...day('2026-10-08',true),universe_size:2},{...day('2026-10-09'),universe_size:2,counts:{direction_buy:1}}]}],
 event_types:[{id:'spring',label:'弹簧测试',side:'BUY'}],notes:['保留缺失覆盖']};
class Container{
 constructor(){this.listeners={};this.innerHTML='';this.children=new Map();}
 addEventListener(type,fn){this.listeners[type]=fn;}
 removeEventListener(type,fn){if(this.listeners[type]===fn)delete this.listeners[type];}
 contains(){return true;}
 querySelector(selector){if(!this.children.has(selector))this.children.set(selector,{innerHTML:'',textContent:'',classList:{add(){}}});return this.children.get(selector);}
 click(dataset){const target={dataset,disabled:false,closest(){return this;}};this.listeners.click({target});}
 change(dataset,value){this.listeners.change({target:{dataset,value}});}
}
const fact={symbol:'002436.SZ',name:'测试',sector:'未知板块',quality_status:'model_unavailable',price_available:true,direction:'SELL',account_action:'HOLD',holding_shares:0,new_events:[{event_type:'spring'}],active_events:[],reason_codes:['no_prior_high_return_head']};
function response(url,rows=[fact]){const q=new URL(url,'http://localhost').searchParams;return {run_id:'run-a',rows,total:50,page:Number(q.get('page')),page_size:25,filters:{date:q.get('date'),sector:q.get('sector')||'',event_type:q.get('event_type')||'',state:q.get('state')||''}};}
"""


def test_aggregate_missingness_proportions_unknown_industry_and_causal_semantics():
    run_js(r"""
const original=JSON.stringify(summary),parsed=ui.normalize(summary);
assert.equal(ui.metricValue(summary.daily[0],'direction_buy','rate'),null);
assert.equal(ui.metricText(summary.daily[0],'direction_buy','count'),'—');
assert.equal(ui.metricText(summary.daily[1],'account_buy','rate'),'0.0%');
assert.equal(ui.metricText(summary.daily[1],'direction_buy','rate'),'40.0%');
const missingButKnown={...summary.daily[0],coverage:{...summary.daily[0].coverage,account_available:1},counts:{...summary.daily[0].counts,account_sell:1,holding:1}};
assert.equal(ui.metricText(missingButKnown,'account_sell','count'),'1');
assert.equal(ui.metricText(missingButKnown,'holding','count'),'1');
assert.equal(ui.metricText(missingButKnown,'direction_sell','count'),'—');
assert.equal(ui.metricText({...missingButKnown,coverage:{...missingButKnown.coverage,account_available:0}},'account_sell','count'),'—');
assert.equal(ui.metricText({...summary.daily[1],universe_size:0},'direction_buy','rate'),'—');
assert.equal(ui.supportText(null),'未评估');assert.equal(ui.supportText(0),'0.0');
const state={date:'2026-10-09',mode:'rate',metric:'direction_buy',allSectors:true,sector:'',event:'',filterState:'',page:1};
const html=ui.viewHtml(parsed,state);
assert(html.includes('未知板块'));assert(html.includes('50%')); // Industry denominator is 2, not the scan's 10.
assert(html.includes('缺模型不是零分'));assert(html.includes('空仓卖方方向不代表卖出成交'));
assert(html.includes('当前分类快照映射到历史'));assert(html.includes('需 未评估 · 供 未评估'));
assert(html.includes('灰色纹理表示无当日行情覆盖'));assert(!html.includes('NaN'));
assert.equal(JSON.stringify(summary),original);
assert.equal(ui.normalize({...summary,dates:['2026-10-09','2026-10-08']}),null);
assert.equal(ui.normalize({...summary,daily:[summary.daily[0],summary.daily[0]]}),null);
const escaped=ui.factsHtml({rows:[{...fact,name:'<script>x</script>',reason_codes:['<img src=x>']}],total:1,page:1,page_size:25},parsed,state);
assert(escaped.includes('&lt;script&gt;'));assert(escaped.includes('&lt;img'));assert(!escaped.includes('<script>'));
""")


def test_drill_filter_pagination_and_analysis_callback_keep_run_and_date():
    run_js(r"""
(async()=>{
 const container=new Container(),calls=[],analysis=[];
 const controller=ui.render(container,summary,{runId:'run-a',request:async url=>{calls.push(url);return response(url);},onAnalyze:context=>analysis.push(context)});
 await controller.ready;assert(calls[0].includes('date=2026-10-09'));assert(calls[0].includes('page=1'));
 container.click({mhSector:'unknown',mhDate:'2026-10-08',mhState:'direction_sell'});await controller.ready;
 container.click({mhEvent:'spring'});await controller.ready;
 let query=new URL(calls.at(-1),'http://localhost').searchParams;
 assert.equal(query.get('date'),'2026-10-08');assert.equal(query.get('sector'),'unknown');assert.equal(query.get('state'),'direction_sell');assert.equal(query.get('event_type'),'spring');
 container.click({mhPage:'2'});await controller.ready;query=new URL(calls.at(-1),'http://localhost').searchParams;
 assert.equal(query.get('page'),'2');assert.equal(query.get('sector'),'unknown');
 container.click({mhAnalyze:'0'});await Promise.resolve();await Promise.resolve();
 assert.equal(analysis.length,1);assert.equal(analysis[0].symbol,'002436.SZ');assert.equal(analysis[0].date,'2026-10-08');assert.equal(analysis[0].runId,'run-a');
 container.click({mhMode:'rate'});await controller.ready;assert.equal(controller.getState().mode,'rate');assert.equal(controller.getState().sector,'unknown');
 container.change({mhDateSelect:''},'2026-10-09');await controller.ready;assert.equal(controller.getState().page,1);assert.equal(controller.getState().event,'spring');
 container.click({mhReset:''});await controller.ready;assert.equal(controller.getState().sector,'');assert.equal(controller.getState().filterState,'');
 const same=ui.render(container,summary,{runId:'run-a'});assert.equal(same,controller); // Filtering the scan list must not reset this view.
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_async_old_date_or_old_run_cannot_overwrite_current_facts_and_wrong_identity_is_visible():
    run_js(r"""
(async()=>{
 const container=new Container(),pending=[];
 const controller=ui.render(container,summary,{request:url=>new Promise(resolve=>pending.push({url,resolve}))});
 const first=controller.ready;container.change({mhDateSelect:''},'2026-10-08');const second=controller.ready;
 pending[1].resolve(response(pending[1].url,[{...fact,name:'新日期'}]));await second;
 pending[0].resolve(response(pending[0].url,[{...fact,name:'旧日期'}]));await first;
 assert(container.querySelector('[data-mh-facts]').innerHTML.includes('新日期'));assert(!container.querySelector('[data-mh-facts]').innerHTML.includes('旧日期'));
 const refreshing=controller.refresh();ui.clear(container,'旧结果无历史');pending[2].resolve(response(pending[2].url));await refreshing;
 assert(container.innerHTML.includes('旧结果无历史'));assert.deepEqual(container.listeners,{});
 const wrong=ui.render(container,summary,{request:async url=>({...response(url),run_id:'foreign'})});await wrong.ready;
 assert(container.querySelector('[data-mh-facts]').innerHTML.includes('与当前扫描运行不匹配'));
 const mismatched=ui.render(container,{...summary},{request:async url=>({...response(url),filters:{date:'2020-01-01'}})});await mismatched.ready;
 assert(container.querySelector('[data-mh-facts]').innerHTML.includes('筛选与当前选择不匹配'));
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_unavailable_old_result_keeps_explicit_empty_state_without_requests():
    run_js(r"""
const container=new Container();let requested=false;
assert.equal(ui.render(container,null,{request:()=>{requested=true;}}),null);
assert(container.innerHTML.includes('旧结果保留原扫描列表'));assert.equal(requested,false);
assert.equal(ui.render(container,{dates:[],daily:[]}),null);assert(container.innerHTML.includes('无法生成历史图表'));
const url=ui.filterUrl('saved/run', {date:'2026-10-09',page:2,sector:'行业 & 板块',event:'spring&test',filterState:'hr_unscored'});
assert(url.startsWith('/api/runs/saved%2Frun/market-history?'));
const params=new URL(url,'http://localhost').searchParams;
assert.equal(params.get('sector'),'行业 & 板块');assert.equal(params.get('event_type'),'spring&test');
assert.throws(()=>ui.filterUrl('',{}));
""")


def test_redrawing_keeps_focused_control_or_heat_cell_without_stealing_outside_focus():
    run_js(r"""
class FocusContainer extends Container {
 constructor(){super();this.ownerDocument={body:{tagName:'BODY'}};this.ownerDocument.activeElement=this.ownerDocument.body;}
 get innerHTML(){return this._html;}
 set innerHTML(html){
  const lostFocus=this.controls?.includes(this.ownerDocument?.activeElement);this._html=html;this.controls=[];
  if(lostFocus&&this.ownerDocument)this.ownerDocument.activeElement=this.ownerDocument.body;
  for(const match of html.matchAll(/<(button|select|rect)\b([^>]*)>/g)){
   const dataset={};for(const attribute of match[2].matchAll(/data-([a-z0-9-]+)(?:="([^"]*)")?/g)){
    const key=attribute[1].replace(/-([a-z])/g,(_,c)=>c.toUpperCase());dataset[key]=attribute[2]||'';
   }
   if(!Object.keys(dataset).some(key=>key.startsWith('mh')))continue;
   const node={tagName:match[1].toUpperCase(),dataset,disabled:/\bdisabled\b/.test(match[2]),focus:()=>{this.ownerDocument.activeElement=node;}};
   this.controls.push(node);
  }
 }
 contains(node){return this.controls.includes(node);}
 querySelectorAll(){return this.controls;}
 querySelector(selector){const attr=selector.match(/^\[data-([a-z-]+)\]$/);if(attr){const key=attr[1].replace(/-([a-z])/g,(_,c)=>c.toUpperCase()),node=this.controls.find(item=>Object.hasOwn(item.dataset,key));if(node)return node;}return super.querySelector(selector);}
}
(async()=>{
 const container=new FocusContainer(),controller=ui.render(container,summary,{request:async url=>response(url)});await controller.ready;
 for(const [key,value] of [['mhDateSelect','2026-10-08'],['mhSectorSelect','unknown'],['mhStateSelect','direction_sell'],['mhEventSelect','spring'],['mhHeatSector','unknown'],['mhMetric','account_sell']]){
  const before=container.controls.find(node=>Object.hasOwn(node.dataset,key));before.focus();container.change({[key]:''},value);await controller.ready;
  assert.notEqual(container.ownerDocument.activeElement,before);assert(Object.hasOwn(container.ownerDocument.activeElement.dataset,key));
 }
 let before=container.controls.find(node=>node.dataset.mhMode==='rate');before.focus();container.click({mhMode:'rate'});await controller.ready;
 assert.equal(container.ownerDocument.activeElement.dataset.mhMode,'rate');
 before=container.controls.find(node=>node.tagName==='BUTTON'&&node.dataset.mhSector==='unknown'&&node.dataset.mhDate==='2026-10-09');before.focus();container.click({...before.dataset});await controller.ready;
 assert.equal(container.ownerDocument.activeElement.dataset.mhSector,'unknown');assert.equal(container.ownerDocument.activeElement.dataset.mhDate,'2026-10-09');
 before=container.controls.find(node=>node.tagName==='RECT'&&node.dataset.mhDate==='2026-10-08');before.focus();container.click({...before.dataset});await controller.ready;
 assert.equal(container.ownerDocument.activeElement.tagName,'RECT');assert.equal(container.ownerDocument.activeElement.dataset.mhDate,'2026-10-08');
 const outside={tagName:'INPUT'};container.ownerDocument.activeElement=outside;container.click({mhMode:'count'});await controller.ready;
 // Repaint must not turn a focus outside this panel into a panel control.
 assert.equal(container.ownerDocument.activeElement,outside);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")
