"""Shipped chart projections preserve per-stock target-change point identity."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_reference_position_markers_project_without_relabeling_as_old_consensus():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for shipped chart marker contract')
    static = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js'
    result = subprocess.run([node, '-e', r'''
const assert=require('node:assert/strict'),path=require('path');
const markers=require(path.join(process.argv[1],'chart_markers.js'));
const date='2026-10-12',symbol='000001.SZ';
const event={symbol,date,available_at:date+'T15:00:00+08:00',first_available_at:date+'T15:00:00+08:00',
 event_id:'target-change',source:'reference_consensus',reference_layer:'consensus',side:'SELL',
 target_position_fraction:.075,previous_reference_target:.1,change_fraction:-.025,reference_only:true};
for(const frequency of ['日线','周线','月线']) {
 const result=markers.build({bars:[{time:date,is_closed:frequency==='日线'}],dailyBars:[{time:date}],
  events:[event,event],symbol,frequency,asOf:date,consensusVersion:'reference_position_v1',up:'r',down:'g'});
 assert.equal(result.markers.length,1);assert.equal(result.markers[0].text,'参S');
 assert.equal(result.markers[0].color,'g');assert.equal(result.groups[0].events[0].change_fraction,-.025);
 assert.equal(result.groups[0].events[0].target_position_fraction,.075);
 assert.equal(result.groups[0].entries.length,1);
 if(frequency!=='日线') assert.equal(result.formingProjectedCount,1);
}
const legacy=markers.build({bars:[{time:date,is_closed:true}],dailyBars:[{time:date}],events:[event],symbol,
 frequency:'日线',asOf:date,consensusVersion:'trend_consensus',up:'r',down:'g'});
assert.equal(legacy.markers[0].text,'共S');
''', str(static)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
