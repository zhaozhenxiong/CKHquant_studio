"""Data workspace contracts: real task kinds and latest-state-safe list reads."""
from __future__ import annotations

import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
from typing import get_args

import pytest

from test_czsc_display_dates_ui import HARNESS, STATIC
from test_light_api import api


WEB = Path(__file__).resolve().parents[1] / "web_dashboard"


class TaskFilter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_filter = False
        self.values = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "select":
            self.in_filter = values.get("id") == "task-kind-filter"
        elif tag == "option" and self.in_filter:
            self.values.append(values.get("value"))

    def handle_endtag(self, tag):
        if tag == "select":
            self.in_filter = False


def filter_values():
    parser = TaskFilter()
    parser.feed((WEB / "static/index.html").read_text(encoding="utf-8"))
    return parser.values


def test_all_current_submitted_task_kinds_can_be_filtered_and_named():
    """Derive kinds from backend submissions, rather than the UI's own labels."""
    from my_strategy.web_dashboard.api import TaskRequest

    kinds = set(get_args(TaskRequest.model_fields["kind"].annotation))
    for name in ("api.py", "board_api.py", "light_api.py"):
        tree = ast.parse((WEB / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "submit_compute" and len(node.args) > 2
                    and isinstance(node.args[2], ast.Constant)):
                kinds.add(node.args[2].value)

    values = filter_values()
    assert len(values) == len(set(values)), "Task filters must not duplicate a kind"
    assert set(values) == kinds | {"all", "analysis"}
    source = (STATIC / "task_history.js").read_text(encoding="utf-8")
    labels = re.search(r"const kindNames=\{([^;]+)\};", source).group(1)
    named = set(re.findall(r"(?:^|,)\s*(\w+)\s*:", labels))
    assert kinds <= named, f"Backend task kinds need readable labels: {kinds - named}"


def test_parameter_comparison_filter_matches_actual_http_submission(api):
    response = api.client.post("/api/light/tune", json={"device": "cpu"})
    assert response.status_code == 202
    kind, spec = api.light.submit.call_args.args
    assert kind == "tuning"
    assert spec["device"] == "cpu"
    assert kind in filter_values()


def run_js(body):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for data workspace checks")
    harness = HARNESS.replace(
        "state,bind,refreshLatestDates,refreshData,",
        "state,bind,refreshTasks,refreshRuns,cancelTask,refreshLatestDates,refreshData,",
    )
    setup = r'''
let requests=[];
global.fetch=(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}));
const run=id=>({run_id:id,kind:'analysis',created_at:'2026-10-16T01:00:00Z',summary:{}});
const visibleRuns=()=>[...element('run-list').innerHTML.matchAll(/data-run-id="([^"]+)"/g)].map(match=>match[1]);
'''
    result = subprocess.run(
        [node, "-e", harness + setup + "\n(async()=>{\n" + body +
         "\n})().catch(error=>{console.error(error);process.exitCode=1;});", str(STATIC)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recent_runs_page_locally_and_clamp_after_refresh_shrinks_list():
    run_js(r'''
const load=workbench.refreshRuns();
assert.equal(requests.length,1);
requests[0].resolve(response({runs:Array.from({length:14},(_,i)=>run('run-'+i))}));await load;
assert.deepEqual(visibleRuns(),Array.from({length:6},(_,i)=>'run-'+i));
assert(element('run-page-prev').disabled);assert(!element('run-page-next').disabled);
await fire('run-page-next','click');
assert.deepEqual(visibleRuns(),Array.from({length:6},(_,i)=>'run-'+(i+6)));
await fire('run-page-next','click');
assert.deepEqual(visibleRuns(),['run-12','run-13']);assert(element('run-page-next').disabled);
assert(element('run-page-label').textContent.includes('最近 14 条'));
assert.equal(requests.length,1,'Paging only the loaded recent records must not fetch or submit');
const shorter=workbench.refreshRuns(true);requests[1].resolve(response({runs:[run('short')]}));await shorter;
assert.deepEqual(visibleRuns(),['short']);
assert(element('run-page-prev').disabled);assert(element('run-page-next').disabled);
assert(element('run-page-label').textContent.includes('第 1 / 1 页'));
const empty=workbench.refreshRuns(true);requests[2].resolve(response({runs:[]}));await empty;
assert.deepEqual(visibleRuns(),[]);assert(element('run-list').innerHTML.includes('尚无'));
assert(element('run-page-prev').disabled);assert(element('run-page-next').disabled);
assert(requests.every(request=>!request.options.method||request.options.method==='GET'));
''')


def test_runs_merge_ordinary_reads_but_explicit_refresh_wins_out_of_order_and_retries():
    run_js(r'''
const old=workbench.refreshRuns(),duplicate=workbench.refreshRuns();
assert.equal(requests.length,1,'Ordinary overlapping reads share work');
const manual=fire('refresh-runs','click');await settle();
assert.equal(requests.length,2,'Explicit refresh must read newer server state');
requests[1].resolve(response({runs:[run('new')]}));await manual;
requests[0].resolve(response({runs:[run('old')]}));await Promise.all([old,duplicate]);
assert.deepEqual(visibleRuns(),['new'],'The old response cannot replace explicit latest state');
const failed=workbench.refreshRuns();requests[2].reject(new Error('fixture network failure'));
await assert.rejects(failed,/fixture network failure/);
const retry=workbench.refreshRuns();assert.equal(requests.length,4);
requests[3].resolve(response({runs:[run('recovered')]}));await retry;
assert.deepEqual(visibleRuns(),['recovered']);
''')


def test_tasks_merge_ordinary_reads_but_explicit_refresh_wins_and_failure_releases_pending():
    run_js(r'''
const old=workbench.refreshTasks(),duplicate=workbench.refreshTasks();assert.equal(requests.length,1);
const manual=fire('refresh-tasks','click');await settle();assert.equal(requests.length,2);
requests[1].resolve(response({tasks:[{job_id:'new',kind:'update',status:'cancelled'}]}));await manual;
requests[0].resolve(response({tasks:[{job_id:'old',kind:'update',status:'running'}]}));
await Promise.all([old,duplicate]);assert.equal(workbench.state.tasks[0].job_id,'new');
const failed=workbench.refreshTasks();requests[2].reject(new Error('fixture task network failure'));
await assert.rejects(failed,/fixture task network failure/);
const retry=workbench.refreshTasks();assert.equal(requests.length,4);
requests[3].resolve(response({tasks:[{job_id:'recovered',kind:'update',status:'succeeded'}]}));await retry;
assert.equal(workbench.state.tasks[0].job_id,'recovered');
''')


def test_cancel_reads_a_new_terminal_snapshot_even_if_old_poll_is_pending():
    run_js(r'''
const poll=workbench.refreshTasks();
const cancelled=workbench.cancelTask('tracked');assert.equal(requests.length,2);
assert.equal(requests[1].url,'/api/tasks/tracked/cancel');
assert.equal(requests[1].options.method,'POST');
requests[1].resolve(response({}));await settle();
assert.equal(requests.length,3,'Cancellation must not reuse its pre-mutation poll');
requests[2].resolve(response({tasks:[{job_id:'tracked',kind:'update',status:'cancelled'}]}));await cancelled;
requests[0].resolve(response({tasks:[{job_id:'tracked',kind:'update',status:'running'}]}));await poll;
assert.equal(workbench.state.tasks[0].status,'cancelled');
''')


def test_finished_tracked_task_refreshes_new_runs_instead_of_reusing_old_snapshot():
    run_js(r'''
location.hash='#data';workbench.state.tracked.update='finished-update';
const oldRuns=workbench.refreshRuns();
const finished=workbench.refreshTasks();
requests[1].resolve(response({tasks:[{job_id:'finished-update',kind:'update',status:'succeeded'}]}));
await settle();assert.equal(requests[2].url,'/api/data/status');
requests[2].resolve(response({available:false}));await finished;await settle();
assert.equal(requests.length,4,'A completed task needs a post-completion runs snapshot');
assert.equal(requests[3].url,'/api/runs?limit=30');
requests[3].resolve(response({runs:[run('new-completed-run')]}));await settle();
requests[0].resolve(response({runs:[run('old')]}));await oldRuns;
assert.deepEqual(visibleRuns(),['new-completed-run']);
''')
