"""Saved/finished scans use the transport view while preserving real UI context."""
import shutil
import subprocess

import pytest

from test_czsc_display_dates_ui import HARNESS, STATIC


def run_js(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for shipped scan interactions')
    harness = HARNESS.replace(
        'globalThis.workbench={state,bind,refreshLatestDates,',
        'globalThis.workbench={state,bind,openRun,refreshTasks,refreshRuns,renderScan,taskActions,refreshLatestDates,')
    result = subprocess.run([node, '-e', harness + '\n(async()=>{\n' + script +
        '\n})().catch(error=>{console.error(error);process.exitCode=1;});', str(STATIC)],
        capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr


SCAN = r'''
const version='czsc_trend_consensus_v2';
const scan={run_id:'final-scan',strategy_version:version,model_family:'high_return',research:true,
 model_run_id:'final-360',model_policy:'auto',calendar_run_id:'verified-calendar',
 account_start:'2023-01-01',account_initial_cash:100000,
 consensus_config:{version,sell_threshold:.35},data_range:{start:'2026-01-01',end:'2026-10-12'},
 scan_view:{version:'czsc_scan_view_v1',compact:true,account_context_status:'consistent'},
 rows:[{symbol:'002675.SZ',name:'东诚药业',strategy_version:version,model_family:'high_return',category:'exit',
  data_end:'2026-10-12',available_at:'2026-10-12T15:00:00+08:00',reason_codes:['actual_held'],
  account_context:{account_id:'research:real-held-prefix',account_start:'2023-01-01',initial_cash:100000},
  account_action:{action:'SELL',emitted:true,holding_shares:600},
  agents:{ma:{probabilities:{up:.2,down:.6,neutral:.2}},high_return:{status:'held',scored:false}}},
 {symbol:'000016.SZ',name:'缺行情样本',strategy_version:version,category:'excluded',
  reason_codes:['missing_expected_session'],account_action:{action:'WAIT',emitted:false}}]};
element('scan-filter').value='all';
'''


def test_saved_compact_scan_renders_search_and_drills_same_held_account():
    run_js(SCAN + r'''
const calls=[];global.fetch=async(url)=>{calls.push(url);return response(scan);};
await workbench.openRun('saved/final','scan');
assert.equal(calls[0],'/api/runs/saved%2Ffinal?view=scan');
assert.equal(element('scan-count').textContent,'2');
assert(element('scan-rows').innerHTML.includes('S · 账户退出意图'));
element('scan-search').value='东诚';workbench.renderScan();
assert(element('scan-rows').innerHTML.includes('002675.SZ'));
assert(!element('scan-rows').innerHTML.includes('000016.SZ'));
await workbench.analyzeSymbol('002675.SZ','2026-10-12',workbench.state.scan);
assert.equal(lastAnalysisQuery.model_run_id,'final-360');
assert.equal(lastAnalysisQuery.calendar_run_id,'verified-calendar');
assert.equal(lastAnalysisQuery.consensus_version,version);
assert.equal(lastAnalysisQuery.account_start,'2023-01-01');
assert.equal(lastAnalysisQuery.account_initial_cash,100000);
assert.equal(lastAnalysisQuery.as_of,'2026-10-12');
assert.deepEqual(lastAnalysisQuery.consensus_config,scan.consensus_config);
assert.equal(scan.rows[0].account_context.account_id,'research:real-held-prefix');
''')


def test_partial_saved_request_keeps_compact_top_level_account_and_pinned_policy():
    run_js(SCAN + r'''
scan.request={start:'2026-01-01',research:true,model_family:'high_return',model_run_id:'final-360'};
scan.account_start='2021-01-01';scan.account_initial_cash=150000;
scan.model_policy='pinned';scan.model_fold='trend_2025Q2';
scan.consensus_config={version,sell_threshold:.48};
scan.rows=scan.rows.slice(0,1);
scan.rows[0].account_context={account_id:'research:custom-real-prefix',account_start:scan.account_start,
 initial_cash:scan.account_initial_cash,model_identity:{model_run_id:'final-360',model_policy:'pinned',model_fold:'trend_2025Q2'}};
await workbench.analyzeSymbol('002675.SZ','2026-10-12',scan);
assert.equal(lastAnalysisQuery.account_start,'2021-01-01');
assert.equal(lastAnalysisQuery.account_initial_cash,150000);
assert.equal(lastAnalysisQuery.calendar_run_id,'verified-calendar');
assert.equal(lastAnalysisQuery.model_policy,'pinned');
assert.equal(lastAnalysisQuery.model_fold,'trend_2025Q2');
assert.deepEqual(lastAnalysisQuery.consensus_config,scan.consensus_config);
''')


def test_finished_scan_task_reads_compact_job_before_delivery_and_delivers_once():
    run_js(SCAN + r'''
const job={job_id:'finished scan',kind:'scan',status:'succeeded',run_id:'final-scan',result:null};
workbench.state.tracked.scan=job.job_id;
const calls=[];global.fetch=async(url)=>{calls.push(url);
 if(url==='/api/tasks')return response({tasks:[job]});
 if(url==='/api/tasks/finished%20scan?view=scan')return response({...job,result:scan});
 if(url==='/api/runs?limit=30')return response({runs:[]});
 throw new Error('unexpected full result request: '+url);};
await workbench.refreshTasks();await settle();
assert.equal(workbench.state.scan.run_id,'final-scan');
assert(element('scan-rows').innerHTML.includes('S · 账户退出意图'));
assert.equal(calls.filter(url=>url.includes('?view=scan')).length,1);
await workbench.refreshTasks();await settle();
assert.equal(calls.filter(url=>url.includes('?view=scan')).length,1,'completed job must not redeliver a large result');
assert(workbench.taskActions(job).includes('data-run-kind="scan"'));
''')


def test_missing_saved_name_is_resolved_by_symbol_without_changing_analysis_options():
    run_js(SCAN + r'''
delete scan.rows[0].name;workbench.state.scan=scan;
element('symbol-options').innerHTML='retained analysis suggestions';
element('scan-search').value='东诚';
global.fetch=async(url)=>{
 assert.equal(url,'/api/symbols?q=%E4%B8%9C%E8%AF%9A&limit=200');
 return response({symbols:[{symbol:'002675.SZ',name:'东诚药业'},
  {symbol:'002676.SZ',name:'另一只股票'}]});};
await fire('scan-search','input');
assert(element('scan-rows').innerHTML.includes('002675.SZ'));
assert(element('scan-rows').innerHTML.includes('东诚药业'));
assert(!element('scan-rows').innerHTML.includes('002676.SZ'));
assert(!element('scan-rows').innerHTML.includes('000016.SZ'));
assert.equal(element('symbol-options').innerHTML,'retained analysis suggestions');
scan.rows[0].name='保存的原名称';element('scan-search').value='东诚';workbench.renderScan();
assert(!element('scan-rows').innerHTML.includes('002675.SZ'),'saved name remains authoritative when provided');
''')


def test_late_name_lookup_does_not_restore_an_old_search():
    run_js(SCAN + r'''
delete scan.rows[0].name;delete scan.rows[1].name;workbench.state.scan=scan;
const requests=pendingNetwork();
element('scan-search').value='东诚';const old=fire('scan-search','input');
element('scan-search').value='康佳';const current=fire('scan-search','input');
requests[1].resolve(response({symbols:[{symbol:'000016.SZ',name:'康佳样本'}]}));await current;
assert(element('scan-rows').innerHTML.includes('000016.SZ'));
requests[0].resolve(response({symbols:[{symbol:'002675.SZ',name:'东诚药业'}]}));await old;
assert.equal(element('scan-search').value,'康佳');
assert(element('scan-rows').innerHTML.includes('000016.SZ'));
assert(!element('scan-rows').innerHTML.includes('002675.SZ'));
''')


def test_newer_saved_scan_wins_and_unknown_kind_keeps_default_full_read():
    run_js(SCAN + r'''
const requests=pendingNetwork();
const earlier=workbench.openRun('earlier','scan'),later=workbench.openRun('later','scan');
requests[1].resolve(response({...scan,run_id:'later'}));await later;
requests[0].resolve(response({...scan,run_id:'earlier'}));await earlier;
assert.equal(workbench.state.scan.run_id,'later');
global.fetch=async(url)=>{assert.equal(url,'/api/runs/legacy-unknown');return response(scan);};
await workbench.openRun('legacy-unknown');
assert.equal(workbench.state.scan.run_id,'final-scan');
''')


@pytest.mark.parametrize('status', ['mixed', 'unavailable'])
def test_unverifiable_compact_trend_account_cannot_silently_create_a_drill(status):
    run_js(SCAN + r'''
element('analysis-symbol').value='original-stock';global.location.hash='scan';
scan.scan_view.account_context_status=''' + repr(status) + r''';
await assert.rejects(workbench.analyzeSymbol('002675.SZ','2026-10-12',scan),/研究账户来源/);
assert.equal(element('analysis-symbol').value,'original-stock');
assert.equal(location.hash,'scan');assert.equal(globalThis.lastAnalysisQuery,undefined);
''')
