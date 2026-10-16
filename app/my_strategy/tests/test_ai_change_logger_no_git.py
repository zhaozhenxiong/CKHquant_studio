"""No-Git auditing must capture source changes without reading private trees."""
import gzip
import errno
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


def capture_diagnostic(value):
    return json.dumps({key:value.get(key) for key in ('source_baseline','change_capture','changed_files')},
                      ensure_ascii=False,indent=2)


def complete_baseline(logger,session,expected):
    baseline=json.loads(gzip.decompress((logger.PROJECT_ROOT/session['source_baseline']['snapshot_path']).read_bytes()))
    assert baseline['complete'],capture_diagnostic(session)
    assert set(baseline['files'])==set(expected),capture_diagnostic(session)
    return baseline


class StatView:
    """Override one observed field without hiding any real stat fields."""
    def __init__(self,value,**changes):
        self.value=value;self.changes=changes
    def __getattr__(self,name):
        return self.changes[name] if name in self.changes else getattr(self.value,name)


def test_real_no_git_start_finish_records_added_modified_deleted_redacted_diff(logger):
    root=logger.PROJECT_ROOT
    changed=write(root,'my_strategy/services/example.py','value = 1\napi_key = "old secret with spaces"\n')
    deleted=write(root,'my_strategy/docs/retired.md','old page\n')
    session=logger.start_session(agent='test',task='source audit')
    assert session['git_available'] is False and session['baseline_commit']=='NO_GIT'
    baseline=complete_baseline(logger,session,{'my_strategy/services/example.py','my_strategy/docs/retired.md'})
    assert 'old secret with spaces' not in json.dumps(baseline)
    changed.write_text('value = 2\napi_key = "new secret with spaces"\n')
    deleted.unlink();write(root,'my_strategy/storage/added.py','new = True\n')
    final=logger.finish_session(sid=session['session_id'],summary='done')
    assert final['change_capture']['status']=='available',capture_diagnostic(final)
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
    complete_baseline(logger,session,{'my_strategy/services/good.py'})
    monkeypatch.setattr(logger,'SOURCE_MAX_BYTES',1)
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='partial' and final['change_capture']['errors'],capture_diagnostic(final)
    assert not final['changed_files'] and final['change_capture']['unverifiable_paths']==['my_strategy/services/good.py'],capture_diagnostic(final)
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
    complete_baseline(logger,session,{'my_strategy/services/good.py'})
    source.unlink();source.symlink_to(private)
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='partial',capture_diagnostic(final)
    assert final['change_capture']['unverifiable_paths']==['my_strategy/services/good.py'],capture_diagnostic(final)
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
    complete_baseline(logger,session,{'vendor/czsc/src/lib.rs','my_strategy/core/example.pyi'})
    rust.write_text('pub const VALUE: i32 = 2;\n');stub.write_text('value: float\n')
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='available',capture_diagnostic(final)
    assert set(final['changed_files'])=={'vendor/czsc/src/lib.rs','my_strategy/core/example.pyi'}


def test_windows_path_creation_time_differs_from_fstat_change_time(logger,monkeypatch):
    source=write(logger.PROJECT_ROOT,'my_strategy/services/good.py','x=1\n')
    original=Path.stat
    def windows_path_stat(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        # CPython 3.12 Windows path stat reports creation time in st_ctime_ns;
        # fstat reports ChangeTime. They can differ on a stable edited file.
        return StatView(value,st_ctime_ns=value.st_ctime_ns-1_000_000) if path==source else value
    monkeypatch.setattr(Path,'stat',windows_path_stat)
    session=logger.start_session(agent='test',task='Windows timestamp semantics')
    complete_baseline(logger,session,{'my_strategy/services/good.py'})
    source.write_text('x=2\n')
    final=logger.finish_session(sid=session['session_id'])
    assert final['change_capture']['status']=='available',capture_diagnostic(final)
    assert final['changed_files']==['my_strategy/services/good.py']
    assert final['change_capture']['changes'][0]['change']=='modified'


@pytest.mark.parametrize('failure,descriptor_count',[
    (None,2),('read_ctime_changed',1),('reopen_denied',1),
    ('reopened_identity_changed',2),('link_before_reopen',1),('link_after_reopen',2),
    ('size_bound',1),('initial_stat_failed',1),('verification_stat_failed',2),
    ('reopened_ctime_changed',2),
])
def test_stable_read_keeps_identity_checks_and_closes_all_descriptors(logger,monkeypatch,failure,descriptor_count):
    source=write(logger.PROJECT_ROOT,'my_strategy/services/good.py','x=1\n')
    real_open,real_fstat,real_safe=logger.os.open,logger.os.fstat,logger.safe_source_path
    opened=[];stats=[]
    def tracked_open(path,*args,**kwargs):
        assert Path(path)==source
        if failure=='reopen_denied' and opened:raise PermissionError('fixture reopen denied')
        fd=real_open(path,*args,**kwargs);opened.append(fd);return fd
    def observed_stat(fd):
        value=real_fstat(fd);stats.append(value)
        if (failure=='initial_stat_failed' and len(stats)==1
                or failure=='verification_stat_failed' and len(stats)==3):
            raise OSError('fixture stat failed')
        # Simulate metadata-only mutation: common identity, size and mtime
        # stay unchanged, so discarding ctime would wrongly accept this read.
        if failure=='read_ctime_changed' and len(stats)==2:
            return StatView(value,st_ctime_ns=value.st_ctime_ns+1)
        if failure=='reopened_identity_changed' and len(stats)==3:
            return StatView(value,st_ino=value.st_ino+1)
        if failure=='reopened_ctime_changed' and len(stats)==3:
            return StatView(value,st_ctime_ns=value.st_ctime_ns+1)
        return value
    def observed_boundary(path):
        # Simulate an ancestor/leaf link appearing between the OS calls without
        # requiring Windows to unlink a file while its read handle is open.
        if failure=='link_before_reopen' and len(opened)==1:return False
        if failure=='link_after_reopen' and len(opened)==2:return False
        return real_safe(path)
    monkeypatch.setattr(logger.os,'open',tracked_open)
    monkeypatch.setattr(logger.os,'fstat',observed_stat)
    monkeypatch.setattr(logger,'safe_source_path',observed_boundary)
    if failure=='size_bound':monkeypatch.setattr(logger,'SOURCE_MAX_BYTES',1)
    if failure:
        with pytest.raises((ValueError,OSError)):
            logger.stable_source_bytes(source)
    else:
        assert logger.stable_source_bytes(source)==source.read_bytes()
    assert len(opened)==descriptor_count
    assert len(set(opened))==len(opened),'Original fd must remain open until path verification finishes'
    for fd in opened:
        with pytest.raises(OSError) as error:real_fstat(fd)
        assert error.value.errno==errno.EBADF
