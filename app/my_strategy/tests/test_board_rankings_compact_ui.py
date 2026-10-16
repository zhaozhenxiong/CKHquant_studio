"""Compact ranking responses retain selection and atomic UI commit guards."""
import pytest

from my_strategy.tests.test_board_publication_ui import run_js


def test_panel_requests_only_ranking_compaction_and_keeps_details_independent():
    run_js(r'''
(async()=>{
global.fetch=async path=>{
 calls.push(path);const url=new URL(path,'http://local');
 if(url.pathname==='/api/boards/status')return response(status('g2'));
 if(url.pathname==='/api/light/status'||url.pathname==='/api/boards/jobs')return response({jobs:[]});
 assert.equal(url.searchParams.get('generation_id'),'g2');
 if(url.pathname==='/api/boards/rankings'){
  assert.equal(url.searchParams.get('compact'),'true');return response(ranking('g2'));
 }
 assert.equal(url.searchParams.has('compact'),false);return response(visual('g2'));
};
await ui.refresh();assert.deepEqual(commits,[['status','g2'],['ranking','g2'],['visual','g2']]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


@pytest.mark.parametrize('field,wrong', [('kind', 'concept'), ('calculation_scope', 'wrong_scope')])
def test_compact_ranking_kind_and_scope_mismatch_never_commits_a_mixed_view(field, wrong):
    run_js(r'''
(async()=>{
global.fetch=async path=>{
 const url=new URL(path,'http://local');
 if(url.pathname==='/api/boards/status')return response(status('g2'));
 if(url.pathname==='/api/boards/rankings'){
  const value=ranking('g2');value.rankings[0]['''+repr(field)+''']='''+repr(wrong)+r''';return response(value);
 }
 return response(visual('g2'));
};
await assert.rejects(ui.refresh(),/板块发布版本核验失败/);assert.deepEqual(commits,[]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_compact_historical_rows_keep_their_original_scope_and_generation():
    run_js(r'''
(async()=>{
el('board-generation').value='g1';
global.fetch=async path=>{
 const url=new URL(path,'http://local');
 if(url.pathname==='/api/boards/status'){
  const value=status('g1');delete value.fluent.market.calculation_scope;return response(value);
 }
 if(url.pathname==='/api/light/status'||url.pathname==='/api/boards/jobs')return response({jobs:[]});
 assert.equal(url.searchParams.get('generation_id'),'g1');
 if(url.pathname==='/api/boards/rankings'){
  assert.equal(url.searchParams.get('compact'),'true');
  const value=ranking('g1');delete value.rankings[0].calculation_scope;return response(value);
 }
 return response(visual('g1'));
};
await ui.refresh();assert.deepEqual(commits,[['status','g1'],['ranking','g1'],['visual','g1']]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
