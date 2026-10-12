'use strict';
const assert = require('node:assert/strict');
const {build,periodKey,describe} = require('../static/js/chart_markers.js');
const dailyBars = ['2026-01-02','2026-01-05','2026-01-09','2026-01-12','2026-01-16','2026-01-19','2026-02-02'].map((time) => ({time}));
const events = [{time:'2026-01-05',available_at:'2026-01-05T15:00:00+08:00',action:'BUY'},
                {time:'2026-01-09',available_at:'2026-01-09T15:00:00+08:00',action:'SELL'},
                {time:'2025-12-31',available_at:'2025-12-31T15:00:00+08:00',action:'BUY'},
                {time:'2026-02-03',available_at:'2026-02-03T15:00:00+08:00',action:'BUY'}];
const trades = [{date:'2026-01-05',symbol:'000001.SZ',action:'BUY'},
                {date:'2026-01-09',symbol:'000001.SZ',action:'SELL'},
                {date:'2026-01-09',symbol:'600000.SH',action:'BUY'}];
const common = {dailyBars,events,trades,symbol:'000001.SZ',start:'2026-01-01',asOf:'2026-02-02',up:'red',down:'green'};
const daily = build({...common,bars:dailyBars,frequency:'日线'});
assert.equal(daily.intentCount,2); assert.equal(daily.tradeCount,2);
assert.deepEqual(daily.markers.map((marker) => marker.shape),['arrowUp','circle','arrowDown','square']);
assert.deepEqual(daily.markers.map((marker) => marker.color),['red','red','green','green']);
assert.deepEqual(daily.markers.map((marker) => marker.text),['B','实B','S','实S']);
const weekly = build({...common,bars:[{time:'2026-01-09'},{time:'2026-01-16'}],frequency:'周线'});
assert.equal(weekly.markers.length,4);
assert(weekly.mappings.every((mapping) => mapping.bar_date === '2026-01-09' && mapping.closed_at === '2026-01-12'));
assert.equal(weekly.markers[0].text,'B');
assert(describe(weekly.mappings,'2026-01-09','周线').includes('买入意图 2026-01-05 · 周线闭合 2026-01-12'));
assert(describe(weekly.mappings,'2026-01-09','周线').includes('卖出成交 2026-01-09'));
assert(!describe(weekly.mappings,null,'周线').includes('2026-01-05'));
assert.equal(describe(weekly.mappings,'2026-01-16','周线'),'2026-01-16 · 该K线无买卖标记');
const monthly = build({...common,bars:[{time:'2026-01-30'}],frequency:'月线'});
assert.equal(monthly.markers.length,4);
assert(monthly.mappings.every((mapping) => mapping.closed_at === '2026-02-02'));
const unfinished = build({...common,dailyBars:dailyBars.filter((bar) => bar.time <= '2026-01-30'),asOf:'2026-01-30',bars:[{time:'2026-01-30'}],frequency:'月线'});
assert.equal(unfinished.markers.length,0); assert.equal(unfinished.deferredCount,4);
const visibleForming = build({...common,bars:[{time:'2026-01-30',is_closed:false,bucket_time:'2026-01-31'}],frequency:'月线'});
assert.equal(visibleForming.markers.length,0); assert.equal(visibleForming.deferredCount,4);
const closedAndForming = build({...common,bars:[{time:'2026-01-30',is_closed:true},{time:'2026-02-02',is_closed:false,bucket_time:'2026-02-28'}],frequency:'月线'});
assert.equal(closedAndForming.markers.length,4);
assert(closedAndForming.mappings.every((mapping) => mapping.bar_date === '2026-01-30'));
const replay = build({...common,dailyBars:dailyBars.filter((bar) => bar.time <= '2026-01-09'),asOf:'2026-01-09',bars:[{time:'2026-01-09'}],frequency:'周线'});
assert.equal(replay.markers.length,0);
assert.equal(periodKey('2025-12-31','周线'),'2026-01');
// The display merges agent evidence, while keeping every distinct original
// event and fill available to the popover and its source/date drill-down.
const point = (event_id, source, side, date, extra = {}) => ({event_id, source, side,
  symbol:'000001.SZ', date, time:date, available_at:`${date}T15:00:00+08:00`,
  event_type:`${source}_${side}`, evidence_visible:true, candidate_eligible:true,
  reference_price:23.45, reference_price_date:date, reference_price_source:'signal_day_close',
  ...extra});
const first = point('ma-1','ma','BUY','2026-01-05', {
  probability:.61, probability_target:'forward_ma_entry', selected_by_reference:false,
  model_lineage:{checkpoint:'before-signal',available_at:'2026-01-02T15:00:00+08:00'},
  anchor_at:'2026-01-02T15:00:00+08:00'});
const multiEvents = [first, {...first},
  point('ma-2','ma','BUY','2026-01-05'),
  point('structure-1','structure','BUY','2026-01-05'),
  point('wyckoff-1','wyckoff','BUY','2026-01-05'),
  point('ma-sell','ma','SELL','2026-01-05'),
  point('structure-sell','structure','SELL','2026-01-05'),
  point('unseen','wyckoff','BUY','2026-01-05',{evidence_visible:false}),
  point('rejected','structure','BUY','2026-01-05',{candidate_eligible:false}),
  point('excluded-only','wyckoff','SELL','2026-01-09',{candidate_eligible:false}),
  point('other-symbol','ma','BUY','2026-01-05',{symbol:'600000.SH'})];
const fill = {date:'2026-01-05', symbol:'000001.SZ', action:'BUY',
  price:23.5, shares:100, fee:5, signal_date:'2026-01-02', order_id:'actual-1'};
const merged = build({...common,bars:dailyBars,frequency:'日线',events:multiEvents,
  trades:[fill,{...fill,order_id:'actual-2',shares:200}, {...fill,symbol:'600000.SH'},
    {...fill,symbol:undefined}]});
assert.equal(merged.intentCount,6);
assert.equal(merged.tradeCount,2);
assert.equal(merged.ignoredCount,3);
assert.equal(merged.markers.length,3);
assert.equal(merged.groups.length,4); // Excluded-only evidence has no default marker.
assert.deepEqual(merged.markers.map((marker) => marker.text),['B','实B','S']);
const buyGroup = merged.groups.find((group) => group.kind === 'intention' && group.action === 'BUY');
assert.equal(buyGroup.event_count,4);
assert.equal(buyGroup.source_count,3); // Two MA events count as one agent.
assert.equal(buyGroup.total_count,6);
assert.equal(buyGroup.excluded_count,2);
assert.deepEqual(buyGroup.sources,['ma','structure','wyckoff']);
assert.equal(buyGroup.entries.filter((entry) => entry.event_id === 'ma-1').length,1);
const mappedFirst = buyGroup.entries.find((entry) => entry.event_id === 'ma-1');
assert.equal(mappedFirst.displayed,true); // Reference veto is evidence, not missing eligibility.
assert.equal(mappedFirst.selected_by_reference,false);
assert.equal(mappedFirst.row,first);
assert.equal(mappedFirst.probability,.61);
assert.equal(mappedFirst.reference_price,23.45);
assert.equal(mappedFirst.anchor_at,'2026-01-02T15:00:00+08:00');
assert.deepEqual(mappedFirst.model_lineage,first.model_lineage);
assert.equal(mappedFirst.original_date,'2026-01-05');
assert.equal(mappedFirst.event_date,'2026-01-05');
assert.equal(mappedFirst.group_id,buyGroup.id);
assert.equal(merged.markers.find((marker) => marker.id === buyGroup.id).shape,'arrowUp');
assert.equal(merged.groups.find((group) => group.time === '2026-01-09').visible,false);
assert(!describe(merged.mappings,'2026-01-09','日线').includes('卖出意图'));
const fillGroup = merged.groups.find((group) => group.kind === 'execution');
assert.equal(fillGroup.trade_count,2);
assert.equal(fillGroup.entries[0].row,fill);
assert.equal(fillGroup.entries[1].shares,200);
assert.equal(fillGroup.entries[0].fee,5);
assert.equal(fillGroup.entries[0].price,23.5);
assert.deepEqual(fillGroup.entries.map((entry) => entry.order_id),['actual-1','actual-2']);
const includingExcluded = build({...common,bars:dailyBars,frequency:'日线',events:multiEvents,trades:[],includeExcluded:true});
assert.equal(includingExcluded.intentCount,9);
assert.equal(includingExcluded.ignoredCount,3);
assert.equal(includingExcluded.groups.find((group) => group.time === '2026-01-09').visible,true);
assert(includingExcluded.mappings.every((entry) => entry.displayed));
assert(describe(includingExcluded.mappings,'2026-01-09','日线').includes('卖出意图'));
assert.equal(includingExcluded.groups.find((group) => group.id === buyGroup.id).id,buyGroup.id);

// Multiple daily points become one B/S per observable closed display period.
const weeklyPoints = [point('week-buy-1','ma','BUY','2026-01-05'),
  point('week-buy-2','structure','BUY','2026-01-09'),
  point('week-sell-1','wyckoff','SELL','2026-01-09')];
const mergedWeekly = build({...common,bars:[{time:'2026-01-09'}],frequency:'周线',events:weeklyPoints,trades:[]});
assert.equal(mergedWeekly.markers.length,2);
assert.deepEqual(mergedWeekly.markers.map((marker) => marker.text),['B','S']);
assert.deepEqual(mergedWeekly.groups[0].entries.map((entry) => entry.original_date),['2026-01-05','2026-01-09']);
assert(mergedWeekly.mappings.every((entry) => entry.closed_at === '2026-01-12'));
const mergedMonth = build({...common,bars:[{time:'2026-01-30'}],frequency:'月线',events:weeklyPoints,trades:[]});
assert.equal(mergedMonth.markers.length,2);
assert(mergedMonth.mappings.every((entry) => entry.closed_at === '2026-02-02'));
const replayMerged = build({...common,bars:[{time:'2026-01-09'}],frequency:'周线',events:weeklyPoints,trades:[],asOf:'2026-01-09'});
assert.equal(replayMerged.markers.length,0);
assert.equal(replayMerged.deferredCount,3);

const yearDaily = ['2025-12-31','2026-01-02','2026-01-05'].map((time) => ({time}));
const yearEvents = [point('year-ma','ma','BUY','2025-12-31'),point('year-structure','structure','BUY','2026-01-02')];
const yearMerged = build({...common,dailyBars:yearDaily,bars:[{time:'2026-01-02'}],
  frequency:'周线',events:yearEvents,trades:[],start:'2025-12-01',asOf:'2026-01-05'});
assert.equal(periodKey('2026-01-02','周线'),'2026-01');
assert.equal(periodKey('2021-01-01','周线'),'2020-53');
assert.equal(yearMerged.markers.length,1);
assert.equal(yearMerged.markers[0].text,'B');
assert.equal(yearMerged.groups[0].closed_at,'2026-01-05');
assert.deepEqual(yearMerged.groups[0].entries.map((entry) => entry.original_date),['2025-12-31','2026-01-02']);

// The display adapter resolves the first observable decision day. It takes
// precedence over a source timestamp while the factual event date is retained.
const delayedEvidence = point('delayed-evidence','ma','BUY','2026-01-05', {
  evidence_date:'2026-01-09', original_event_date:'2026-01-05',
  available_at:'2026-01-05T15:00:00+08:00', signal_at:'2026-01-09T15:00:00+08:00',
  reference_price_date:'2026-01-09'});
const alignedEvidence = build({...common,bars:dailyBars,frequency:'日线',events:[delayedEvidence],
  trades:[{...fill,evidence_date:'2026-01-09'}]});
const delayedMapping = alignedEvidence.mappings.find((entry) => entry.kind === 'intention');
assert.equal(delayedMapping.bar_date,'2026-01-09');
assert.equal(delayedMapping.original_date,'2026-01-09');
assert.equal(delayedMapping.event_date,'2026-01-05');
assert.equal(delayedMapping.original_event_date,'2026-01-05');
assert.equal(delayedMapping.row,delayedEvidence);
assert.equal(delayedMapping.row.available_at,'2026-01-05T15:00:00+08:00');
assert.equal(delayedMapping.row.signal_at,'2026-01-09T15:00:00+08:00');
assert.equal(alignedEvidence.markers.find((marker) => marker.text === 'B').time,'2026-01-09');
assert.equal(alignedEvidence.mappings.find((entry) => entry.kind === 'execution').bar_date,'2026-01-05');
const beforeEvidence = build({...common,bars:dailyBars,frequency:'日线',events:[delayedEvidence],trades:[],asOf:'2026-01-05'});
assert.equal(beforeEvidence.markers.length,0);

// No fabricated agent identity, no mutation of caller records, stable IDs.
assert.equal(daily.groups.find((group) => group.kind === 'intention').source_count,0);
assert.equal(first.action,undefined);
const reordered = build({...common,bars:dailyBars,frequency:'日线',events:[...multiEvents].reverse(),trades:[fill]});
assert.deepEqual(reordered.groups.map((group) => group.id),merged.groups.map((group) => group.id));
console.log(JSON.stringify({status:'passed',checks:'B/S grouping, source/event counts, full evidence and fill retention, exclusions, stable identities, colors/shapes, source dates, completed buckets, replay, cross-year ISO weeks, start cutoff, symbol isolation'}));
