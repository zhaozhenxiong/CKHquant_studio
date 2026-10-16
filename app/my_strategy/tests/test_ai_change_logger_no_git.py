"""No-Git auditing must capture source changes without reading private trees."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def isolate_git_discovery(monkeypatch, directory):
    # ci_guard places --basetemp below the real checkout. An empty fixture
    # directory alone is therefore not a no-Git workspace: Git walks upwards.
    # Limit only this test's real Git subprocesses, retaining local .git discovery.
    for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE',
                 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(directory.parent.resolve()))


@pytest.fixture
def logger(tmp_path,monkeypatch):
    isolate_git_discovery(monkeypatch, tmp_path)
    source=Path(__file__).parents[1]/'scripts/ai_change_logger.py'
    spec=importlib.util.spec_from_file_location('isolated_change_logger',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.PROJECT_ROOT=tmp_path
    module.KB_ROOT=tmp_path/'my_strategy/knowledge_base'
    module.SYSTEM_ROOT=tmp_path/'my_strategy/knowledge_base_system'
    for name,relative in {'SESSION_JSON_ROOT':'sessions','PROMPT_ROOT':'prompts','DIFF_ROOT':'diffs'}.items():setattr(module,name,module.SYSTEM_ROOT/relative)
    module.CHANGE_ROOT=module.KB_ROOT/'04_AI_Changes';module.SESSION_NOTE_ROOT=module.KB_ROOT/'05_AI_Sessions';module.CHANGELOG=tmp_path/'AI_CHANGELOG.md'
    # Real Git executes within the fixture's explicit discovery boundary.
    original=module.run_process
    monkeypatch.setattr(module,'run_process',lambda args,**kwargs:original(args,cwd=tmp_path,**{k:v for k,v in kwargs.items() if k!='cwd'}))
    monkeypatch.setattr(module,'rebuild_obsidian_indexes',lambda:False)
    return module


def test_discovery_ceiling_blocks_ancestor_repo_without_disabling_real_git(tmp_path, monkeypatch):
    git = shutil.which('git')
    if git is None:
        pytest.skip('Git executable unavailable for repository discovery regression')
    outer = tmp_path / 'real-checkout'; outer.mkdir()
    isolate_git_discovery(monkeypatch, outer)
    def run(directory, *args):
        return subprocess.run([git, *args], cwd=directory, capture_output=True, text=True, check=False)
    assert run(outer, 'init', '--quiet').returncode == 0
    nested = outer / '.scratch' / 'fixture-app'; nested.mkdir(parents=True)
    found = run(nested, 'rev-parse', '--show-toplevel')
    assert found.returncode == 0 and Path(found.stdout.strip()).resolve() == outer.resolve()
    isolate_git_discovery(monkeypatch, nested)
    assert run(nested, 'rev-parse', '--show-toplevel').returncode != 0
    assert run(nested, 'init', '--quiet').returncode == 0
    found = run(nested, 'rev-parse', '--show-toplevel')
    assert found.returncode == 0 and Path(found.stdout.strip()).resolve() == nested.resolve()


def write(root,name,text):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path


def test_real_no_git_start_finish_records_added_modified_deleted_redacted_diff(logger):
    root=logger.PROJECT_ROOT
    changed=write(root,'my_strategy/services/example.py','value = 1\napi_key = "old secret with spaces"\n')
    deleted=write(root,'my_strategy/docs/retired.md','old page\n')
    session=logger.start_session(agent='test',task='source audit')
    assert session['git_available'] is False and session['baseline_commit']=='NO_GIT'
    baseline=json.loads(gzip.decompress((root/session['source_baseline']['snapshot_path']).read_bytes()))
    assert baseline['complete'] and 'old secret with spaces' not in json.dumps(baseline)
    changed.write_text('value = 2\napi_key = "new secret with spaces"\n')
    deleted.unlink();write(root,'my_strategy/storage/added.py','new = True\n')
    final=logger.finish_session(sid=session['session_id'],summary='done')
    assert final['change_capture']['status']=='available'
    assert {item['path']:item['change'] for item in final['change_capture']['changes']}=={'my_strategy/services/example.py':'modified','my_strategy/docs/retired.md':'deleted','my_strategy/storage/added.py':'added'}
    note=(root/final['note_path']).read_text()
    assert '-value = 1' in note and '+value = 2' in note and 'secret with spaces' not in note
    assert '3 files' in logger.CHANGELOG.read_text() and not (root/'.git').exists()


def test_whitelist_never_reads_business_env_private_inventory_or_symlink(logger,monkeypatch):
    root=logger.PROJECT_ROOT
    private=[write(root,name,'PRIVATE_SECRET') for name in ('.env','.env.local','inventory.json','my_strategy/data/hidden.py','my_strategy/artifacts/hidden.md','my_strategy/knowledge_base/hidden.md','my_strategy/knowledge_base_system/hidden.py','my_strategy/kronos_model/config.json','my_strategy/services/data/hidden.py')]
    write(root,'.env.example','TOKEN="example secret with spaces"\n')
    write(root,'my_strategy/services/good.py','x=1\n')
    link=root/'my_strategy/services/linked.py';link.symlink_to(private[0])
    directory=root/'my_strategy/services/linked_dir';directory.symlink_to(root/'my_strategy/data',target_is_directory=True)
    original=Path.read_bytes
    def guarded(path,*args,**kwargs):
        if path in private or path==link:pytest.fail('read a private or linked source: '+str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    snapshot=logger.source_snapshot()
    assert set(snapshot['files'])=={'.env.example','my_strategy/services/good.py'}
    assert 'example secret with spaces' not in json.dumps(snapshot)


def test_legacy_session_is_unavailable_and_external_diff_is_only_supplement(logger):
    session=logger.start_session(agent='test',task='legacy')
    session.pop('source_baseline');logger.save_session(session)
    external=write(logger.PROJECT_ROOT,'deployment.patch','+++ b/source.py\n+password="private value"\n')
    final=logger.finish_session(sid=session['session_id'],external_diff=external)
    assert final['change_capture']['status']=='unavailable' and final['diff_sha256']==''
    assert 'unknown files' in logger.CHANGELOG.read_text()
    assert '无法判断变化数量' in (logger.PROJECT_ROOT/final['note_path']).read_text()
    assert 'source_baseline' not in logger.load_session(session['session_id'])
    evidence=final['external_diff_evidence'];assert evidence['kind']=='supplemental_external_evidence'
    assert 'private value' not in gzip.decompress((logger.PROJECT_ROOT/evidence['artifact_path']).read_bytes()).decode()


def test_snapshot_corruption_or_missing_source_is_not_zero_changes(logger):
    source=write(logger.PROJECT_ROOT,'my_strategy/services/good.py','x=1\n')
    session=logger.start_session(agent='test',task='corrupt')
    snapshot=logger.PROJECT_ROOT/session['source_baseline']['snapshot_path'];snapshot.write_bytes(b'changed')
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='unavailable' and 'SHA mismatch' in final['change_capture']['reason']
    assert 'unknown files' in logger.CHANGELOG.read_text()


def test_source_read_failure_is_explicit_partial_not_false_complete(logger,monkeypatch):
    source=write(logger.PROJECT_ROOT,'my_strategy/services/good.py','x=1\n')
    session=logger.start_session(agent='test',task='partial')
    monkeypatch.setattr(logger,'SOURCE_MAX_BYTES',1)
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='partial' and final['change_capture']['errors']
    assert not final['changed_files'] and final['change_capture']['unverifiable_paths']==['my_strategy/services/good.py']
    assert 'unknown files' in logger.CHANGELOG.read_text()


def test_git_workflow_keeps_existing_capture_and_avoids_source_snapshot(logger,monkeypatch):
    monkeypatch.setattr(logger,'git_available',lambda:True)
    monkeypatch.setattr(logger,'git_status',lambda:' M original.py')
    monkeypatch.setattr(logger,'git_head',lambda:'commit')
    monkeypatch.setattr(logger,'git_diff',lambda:'-old\n+new\n')
    monkeypatch.setattr(logger,'git_diff_stat',lambda:'original.py | 2 +-')
    monkeypatch.setattr(logger,'save_source_baseline',lambda *_:pytest.fail('Git session used fallback'))
    session=logger.start_session(agent='test',task='git')
    final=logger.finish_session(sid=session['session_id'])
    assert final['changed_files']==['original.py'] and final['change_capture']['method']=='git'
    assert 'source_baseline' not in final


def test_source_replaced_with_symlink_is_unverifiable_and_never_read(logger):
    source=write(logger.PROJECT_ROOT,'my_strategy/services/good.py','x=1\n')
    private=write(logger.PROJECT_ROOT,'.env','PRIVATE_SECRET')
    session=logger.start_session(agent='test',task='replaced')
    source.unlink();source.symlink_to(private)
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='partial'
    assert final['change_capture']['unverifiable_paths']==['my_strategy/services/good.py']
    assert 'PRIVATE_SECRET' not in (logger.PROJECT_ROOT/final['note_path']).read_text()


@pytest.mark.parametrize('broken',[False,True])
def test_whole_package_link_never_reads_external_tree_or_claims_deletion(logger,tmp_path,broken):
    original=write(logger.PROJECT_ROOT,'my_strategy/services/good.py','x=1\n')
    session=logger.start_session(agent='test',task='ancestor link')
    package=logger.PROJECT_ROOT/'my_strategy'
    # Audit storage remains at its original location while the source package is linked.
    retired=logger.PROJECT_ROOT/'original-package';package.rename(retired)
    for name in ('KB_ROOT','SYSTEM_ROOT','SESSION_JSON_ROOT','PROMPT_ROOT','DIFF_ROOT','CHANGE_ROOT','SESSION_NOTE_ROOT'):
        old=getattr(logger,name);setattr(logger,name,retired/old.relative_to(package))
    # Keep the baseline's original audit-relative location valid through explicit relocation.
    current=logger.load_session(session['session_id'])
    for key in ('snapshot_path','manifest_path'):
        current['source_baseline'][key]=str(retired.relative_to(logger.PROJECT_ROOT)/Path(current['source_baseline'][key]).relative_to('my_strategy'))
    logger.save_session(current)
    outside=tmp_path.parent/(tmp_path.name+'-external')
    if not broken:write(outside,'services/private.py','TOP_PRIVATE_VALUE')
    package.symlink_to(outside,target_is_directory=True)
    captured=logger.source_snapshot()
    assert not captured['complete'] and 'TOP_PRIVATE_VALUE' not in json.dumps(captured)
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='partial'
    assert 'my_strategy/services/good.py' in final['change_capture']['unverifiable_paths']
    assert not final['change_capture']['changes']


def test_mutation_during_read_is_partial_with_no_mixed_source(logger,monkeypatch):
    source=write(logger.PROJECT_ROOT,'my_strategy/services/good.py','x=1\n')
    original=logger.os.fstat;calls=[]
    def fstat(fd):
        value=original(fd);calls.append(1)
        if len(calls)==1:source.write_text('x=999999\n')
        return value
    monkeypatch.setattr(logger.os,'fstat',fstat)
    snapshot=logger.source_snapshot()
    assert not snapshot['complete'] and not snapshot['files']
    assert 'changed during' in snapshot['errors'][0]['error']


def test_native_vendor_and_type_stub_changes_are_in_source_scope(logger):
    rust=write(logger.PROJECT_ROOT,'vendor/czsc/src/lib.rs','pub const VALUE: i32 = 1;\n')
    stub=write(logger.PROJECT_ROOT,'my_strategy/core/example.pyi','value: int\n')
    session=logger.start_session(agent='test',task='native source')
    rust.write_text('pub const VALUE: i32 = 2;\n');stub.write_text('value: float\n')
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='available'
    assert set(final['changed_files'])=={'vendor/czsc/src/lib.rs','my_strategy/core/example.pyi'}
