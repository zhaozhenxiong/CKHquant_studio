"""The shipped panel requests compact status without weakening pinned reads."""
from my_strategy.tests.test_board_publication_ui import run_js


def test_current_panel_requests_compact_status_and_pins_both_followup_reads():
    run_js(r'''
(async()=>{
global.fetch=async path=>{
 calls.push(path);const url=new URL(path,'http://local');
 if(url.pathname==='/api/boards/status'){
  assert.equal(url.searchParams.get('compact'),'true');
  assert.equal(url.searchParams.has('generation_id'),false);
  return response(status('g2'));
 }
 if(url.pathname==='/api/light/status'||url.pathname==='/api/boards/jobs')return response({jobs:[]});
 assert.equal(url.searchParams.get('generation_id'),'g2');
 return response(url.pathname.endsWith('/rankings')?ranking('g2'):visual('g2'));
};
await ui.refresh();
assert.deepEqual(commits,[['status','g2'],['ranking','g2'],['visual','g2']]);
assert.equal(calls.filter(path=>new URL(path,'http://local').pathname==='/api/boards/status').length,1);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_selected_history_is_encoded_as_one_parameter_alongside_compact():
    run_js(r'''
(async()=>{
const selected='saved&compact=false?历史/+%';el('board-generation').value=selected;
global.fetch=async path=>{
 calls.push(path);const url=new URL(path,'http://local');
 if(url.pathname==='/api/boards/status'){
  assert.equal(url.searchParams.get('compact'),'true');
  assert.deepEqual(url.searchParams.getAll('generation_id'),[selected]);
  assert.equal([...url.searchParams].length,2);
  return response(status(selected));
 }
 if(url.pathname==='/api/light/status'||url.pathname==='/api/boards/jobs')return response({jobs:[]});
 assert.equal(url.searchParams.get('generation_id'),selected);
 return response(url.pathname.endsWith('/rankings')?ranking(selected):visual(selected));
};
await ui.refresh();
assert.deepEqual(commits,[['status',selected],['ranking',selected],['visual',selected]]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_delayed_compact_status_cannot_overwrite_a_newer_refresh():
    run_js(r'''
(async()=>{
const old=deferred();let n=0;
global.fetch=async path=>{
 calls.push(path);const url=new URL(path,'http://local');
 if(url.pathname==='/api/boards/status'){
  assert.equal(url.searchParams.get('compact'),'true');
  return ++n===1?old.promise:response(status('g2'));
 }
 if(url.pathname==='/api/light/status'||url.pathname==='/api/boards/jobs')return response({jobs:[]});
 const selected=url.searchParams.get('generation_id');
 return response(url.pathname.endsWith('/rankings')?ranking(selected):visual(selected));
};
const first=ui.refresh();await flush();await ui.refresh();
old.resolve(response(status('g1')));await first;
assert.deepEqual(commits,[['status','g2'],['ranking','g2'],['visual','g2']]);
assert.equal(calls.some(path=>new URL(path,'http://local').searchParams.get('generation_id')==='g1'),false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_compact_reply_for_wrong_history_is_rejected_before_any_view_commit():
    run_js(r'''
(async()=>{
el('board-generation').value='g1';
global.fetch=async path=>{calls.push(path);return response(status('g2'));};
await assert.rejects(ui.refresh(),/指定历史版本核验失败/);
assert.deepEqual(commits,[]);assert.equal(calls.length,1);
const url=new URL(calls[0],'http://local');
assert.equal(url.searchParams.get('compact'),'true');assert.equal(url.searchParams.get('generation_id'),'g1');
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
