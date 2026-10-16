"""Isolated retirement guard/transaction tests; never touch workspace data."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import pytest

from my_strategy.scripts import retire_beijing_data as retire


@pytest.fixture
def service_probe(monkeypatch):
    """Exercise both service-port checks without creating any real socket."""
    calls = []
    listening = set()
    class SocketFixture:
        def settimeout(self, timeout):
            assert timeout == .3
        def connect_ex(self, address):
            assert address[0] == '127.0.0.1'
            calls.append(address)
            return 0 if address[1] in listening else 61
        def close(self):
            pass
    # Replace only the script's module reference; no real socket is ever made.
    monkeypatch.setattr(retire, 'socket', SimpleNamespace(socket=SocketFixture))
    return calls, listening


@pytest.fixture
def raw_fixture(tmp_path, monkeypatch):
    project = tmp_path / 'app'; project.mkdir(); (project / 'AGENTS.md').write_text('fixture')
    data = project / 'my_strategy/data'
    expected = {'project': project.resolve(), 'raw': data / 'raw/khquant_raw.db', 'metadata': data / 'metadata',
                'results': data / 'processed/czsc/results.db', 'runs': project / 'my_strategy/artifacts/runs',
                'kronos': project / 'my_strategy/kronos_model'}
    monkeypatch.setattr(retire, '_configured_paths', lambda: expected)
    p = retire.paths(project)
    p['raw'].parent.mkdir(parents=True)
    with sqlite3.connect(p['raw']) as conn:
        conn.execute('CREATE TABLE securities(stock TEXT PRIMARY KEY,exchange TEXT,name TEXT)')
        conn.executemany('INSERT INTO securities VALUES(?,?,?)', [('920001.BJ','BJ','Beijing'),('000001.SZ','SZ','Shenzhen'),('900001.SH','SH','Shanghai B')])
        for table, col in retire.TARGETS.items():
            if table=='securities': continue
            conn.execute(f'CREATE TABLE "{table}"("{col}" TEXT,value TEXT)')
            conn.executemany(f'INSERT INTO "{table}" VALUES(?,?)', [('920001.BJ','old'),('000001.SZ','keep sz'),('900001.SH','keep sh'),(None,'keep null')])
        conn.execute('CREATE TABLE update_log(code TEXT,message TEXT)')
        conn.execute("INSERT INTO update_log VALUES('920001.BJ','immutable audit')")
    p['metadata'].mkdir(parents=True)
    (p['metadata'] / 'light_research').mkdir()
    assets={'schema':'task_cleanup_protection_v1','watchlist':{'count':2},'accounts':{'count':2},'research_objects':{},'pointers':{},'pins':{},'model_files':{},'errors':[]}
    monkeypatch.setattr(retire,'activity',lambda *a: [])
    monkeypatch.setattr(retire,'protected_assets',lambda *a: deepcopy(assets))
    acceptance={'final_runs':{m:'final-'+m for m in retire.METHODS},'references':{'fixture':{'sha256':'evidence'}}}
    monkeypatch.setattr(retire,'accepted_scans',lambda *a: deepcopy(acceptance))
    plan=retire.preview(p,state=tmp_path/'state.json',evidence=tmp_path,full_protection=True)
    return p,plan


def test_preview_no_database_changes_and_exact_bj_only(raw_fixture):
    p,plan=raw_fixture
    before=retire.file_identity(p['raw'])
    again=retire.preview(p,state=plan['state'],evidence=plan['scan_evidence'],full_protection=True)
    assert retire.file_identity(p['raw'])==before
    assert again['symbols']==['920001.BJ']
    assert set(again['candidates'])==set(retire.TARGETS)
    assert all(row['count']==1 for row in again['candidates'].values())
    assert again['protected_raw_rows']['update_log']['count']==1
    assert again['protected_raw_rows']['stock_daily_raw']['count']==3


def test_apply_deletes_only_previewed_bj_rows_and_preserves_audit_and_non_bj(raw_fixture,tmp_path):
    p,plan=raw_fixture
    result=retire.apply(p,plan,backup=tmp_path/'rollback.db',journal=tmp_path/'receipt.json')
    assert result['status']=='committed'
    assert set(result['deleted'])==set(retire.TARGETS)
    with retire.readonly(p['raw']) as conn:
        assert retire.symbols(conn)==[]
        assert retire.raw_rows(conn,plan['symbols'],protected=True)==plan['protected_raw_rows']
        assert list(conn.execute('SELECT code FROM update_log'))==[('920001.BJ',)]
    with retire.readonly(tmp_path/'rollback.db') as conn:
        assert retire.symbols(conn)==['920001.BJ']
    assert json.loads((tmp_path/'receipt.json').read_text())['status']=='committed'


@pytest.mark.parametrize('change',['target','preserved','audit'])
def test_input_changes_after_preview_fail_closed(raw_fixture,tmp_path,change):
    p,plan=raw_fixture
    with sqlite3.connect(p['raw']) as conn:
        if change=='target': conn.execute("UPDATE stock_daily_raw SET value='changed' WHERE stock='920001.BJ'")
        elif change=='preserved': conn.execute("UPDATE stock_daily_raw SET value='changed' WHERE stock='000001.SZ'")
        else: conn.execute("UPDATE update_log SET message='changed'")
    with pytest.raises(ValueError,match='changed after preview'):
        retire.apply(p,plan,backup=tmp_path/'backup.db',journal=tmp_path/'receipt.json')
    with retire.readonly(p['raw']) as conn: assert retire.symbols(conn)==['920001.BJ']
    assert not (tmp_path/'backup.db').exists()


def test_delete_trigger_touching_protected_stock_rolls_back(raw_fixture,tmp_path):
    p,_=raw_fixture
    with sqlite3.connect(p['raw']) as conn:
        conn.execute("CREATE TRIGGER bad_delete AFTER DELETE ON stock_daily_raw BEGIN UPDATE stock_daily_raw SET value='oops' WHERE stock='000001.SZ'; END")
    plan=retire.preview(p,state=tmp_path/'state.json',evidence=tmp_path,full_protection=True)
    with pytest.raises(ValueError,match='Protected raw row digest changed'):
        retire.apply(p,plan,backup=tmp_path/'backup.db',journal=tmp_path/'receipt.json')
    with retire.readonly(p['raw']) as conn:
        assert retire.symbols(conn)==['920001.BJ']
        assert conn.execute("SELECT value FROM stock_daily_raw WHERE stock='000001.SZ'").fetchone()[0]=='keep sz'
    assert json.loads((tmp_path/'receipt.json').read_text())['status']=='rolled_back'


@pytest.mark.parametrize('change',['plan','active','protected','acceptance'])
def test_guards_refuse_changed_or_active_inputs(raw_fixture,tmp_path,monkeypatch,change):
    p,plan=raw_fixture
    if change=='plan': plan['symbols']=['000001.SZ']
    elif change=='active': monkeypatch.setattr(retire,'activity',lambda *a:['active task'])
    elif change=='protected': monkeypatch.setattr(retire,'protected_assets',lambda *a:{'changed':True})
    else: monkeypatch.setattr(retire,'accepted_scans',lambda *a:{'changed':True})
    with pytest.raises(ValueError): retire.apply(p,plan,backup=tmp_path/'backup.db',journal=tmp_path/'receipt.json')
    assert not (tmp_path/'backup.db').exists()


def test_active_preview_cannot_be_applied(raw_fixture,tmp_path,monkeypatch):
    p,_=raw_fixture
    monkeypatch.setattr(retire,'activity',lambda *a:['running scan'])
    plan=retire.preview(p,state=tmp_path/'state.json',evidence=tmp_path,full_protection=True)
    assert plan['protected_raw_rows'] is None
    with pytest.raises(ValueError,match='blocked'): retire.apply(p,plan,backup=tmp_path/'backup.db',journal=tmp_path/'receipt.json')


def test_ambiguous_stock_exchange_refused(raw_fixture):
    p,_=raw_fixture
    with sqlite3.connect(p['raw']) as conn: conn.execute("UPDATE securities SET exchange='BJ' WHERE stock='900001.SH'")
    with retire.readonly(p['raw']) as conn:
        with pytest.raises(ValueError,match='Ambiguous'): retire.symbols(conn)


def test_activity_checks_task_state_leases_and_service(tmp_path,monkeypatch,service_probe):
    p={'metadata':tmp_path}
    monkeypatch.setattr(retire,'project_processes',lambda p:[])
    with sqlite3.connect(tmp_path/'czsc_tasks.db') as conn:
        conn.execute('CREATE TABLE czsc_tasks(status TEXT)');conn.execute("INSERT INTO czsc_tasks VALUES('running')")
    (tmp_path/'light_research').mkdir()
    with sqlite3.connect(tmp_path/'light_research/research.db') as conn:
        conn.execute('CREATE TABLE leases(expires_at REAL)');conn.execute('INSERT INTO leases VALUES(?)',(retire.time.time()+60,))
    calls, listening = service_probe
    port = 49123  # Pure fixture identity, never bound or connected.
    listening.add(port)
    problems=retire.activity(p,port)
    assert set(calls) == {('127.0.0.1', 8124), ('127.0.0.1', port)}
    assert len(calls) == 2
    assert any(f'listening on {port}' in problem for problem in problems)
    assert not any('listening on 8124' in problem for problem in problems)
    assert len(problems)>=3


def test_incomplete_or_unaccepted_controller_refused(tmp_path):
    state=tmp_path/'state.json';state.write_text('{"status":"running"}')
    with pytest.raises(ValueError,match='not completed'): retire.accepted_scans({},state,tmp_path)
    state.write_text('{"status":"completed"}')
    (tmp_path/'comparison.json').write_text('{"verification":{"passed":true},"trial":true}')
    with pytest.raises(ValueError,match='not accepted'): retire.accepted_scans({},state,tmp_path)


def test_empty_wal_is_included_in_physical_identity(raw_fixture):
    p,_=raw_fixture
    wal=Path(str(p['raw'])+'-wal');wal.write_bytes(b'')
    assert retire.raw_physical(p)['wal']=={'bytes':0,'sha256':retire.hashlib.sha256(b'').hexdigest()}


def test_official_path_resolver_must_match_project(tmp_path,monkeypatch):
    (tmp_path/'AGENTS.md').write_text('fixture')
    monkeypatch.setattr(retire,'_configured_paths',lambda:{'project':tmp_path/'wrong'})
    with pytest.raises(ValueError,match='Configured project'):retire.paths(tmp_path)


def test_without_rowid_digest_uses_primary_key_order(tmp_path):
    with sqlite3.connect(tmp_path/'fixture.db') as conn:
        conn.execute('CREATE TABLE sample(a TEXT,b INT,PRIMARY KEY(a,b)) WITHOUT ROWID')
        conn.executemany('INSERT INTO sample VALUES(?,?)',[('z',3),('a',2)])
        assert retire.row_digest(conn,'sample')['count']==2


def test_orphan_beijing_identifier_requires_review(raw_fixture,tmp_path):
    p,_=raw_fixture
    with sqlite3.connect(p['raw']) as conn:
        conn.execute("INSERT INTO stock_daily_normalized VALUES('920002.BJ','orphan')")
    with pytest.raises(ValueError,match='orphan Beijing'):
        retire.preview(p,state=tmp_path/'state',evidence=tmp_path,full_protection=True)


def test_foreign_key_protection_prevents_orphaning(raw_fixture,tmp_path):
    p,_=raw_fixture
    with sqlite3.connect(p['raw']) as conn:
        conn.execute('CREATE TABLE preserved_reference(stock TEXT REFERENCES securities(stock))')
        conn.execute("INSERT INTO preserved_reference VALUES('920001.BJ')")
    plan=retire.preview(p,state=tmp_path/'state',evidence=tmp_path,full_protection=True)
    with pytest.raises(sqlite3.IntegrityError):retire.apply(p,plan,backup=tmp_path/'backup.db',journal=tmp_path/'journal.json')
    with retire.readonly(p['raw']) as conn:assert retire.symbols(conn)==['920001.BJ']
    assert json.loads((tmp_path/'journal.json').read_text())['status']=='rolled_back'


def test_output_cannot_overwrite_business_or_audit_file(raw_fixture,tmp_path):
    p,_=raw_fixture
    with pytest.raises(ValueError):retire.audit_output(p,p['metadata']/'new.json')
    target=tmp_path/'prior.json';target.write_text('immutable')
    with pytest.raises(ValueError):retire.audit_output(p,target)
    assert target.read_text()=='immutable'


_REAL_ACCEPTANCE = retire.accepted_scans


def _acceptance_fixture(p, evidence):
    evidence.mkdir(); population=['000001.SZ']; common={'end':'2026-10-15'}
    state={'status':'completed','parameters':common,'methods':{}}
    raw={k:v['sha256'] for k,v in retire.raw_physical(p).items()}
    frozen={'population':population,'raw':raw,'calendar':{'sha':'calendar'},'code':{'sha':'code'},'prediction':{'sha':'prediction'}}
    coverage={'complete':True,'deferred':0,'requested':1,'evaluated':1,'success':0,'successful':0,'failed':1,'unavailable':0}
    comparison={'schema':'account_free_screening_comparison_v1','trial':False,'population':population,
        'verification':{'passed':True,'all_population_processed':True,'shared_frozen_identity_hashes':{}},
        'sources':{},'methods':{},'stocks':[{'symbol':population[0],'methods':{}}]}
    p['results'].parent.mkdir(parents=True)
    with sqlite3.connect(p['results']) as conn:
        conn.execute('CREATE TABLE czsc_runs(run_id TEXT,report_path TEXT,summary TEXT)')
        for method in retire.METHODS:
            run='final-'+method;state['methods'][method]={'final_run_id':run}
            report={'run_id':run,'analysis_method':method,'run_context':{'scope':'full'},'reference_only':True,'account_scope':'none',
                'population':population,'coverage':coverage,'frozen_identity':frozen,'method_context':common,
                'rows':[{'symbol':population[0],'method_evaluation':{'status':'failed','direction':'WAIT'}}]}
            path=p['runs']/run/'reports/czsc.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(report))
            conn.execute('INSERT INTO czsc_runs VALUES(?,?,?)',(run,str(path.relative_to(p['runs'])),json.dumps({'coverage':coverage})))
            comparison['sources'][method]={'run_id':run};comparison['methods'][method]={'run_id':run,'coverage':coverage}
            comparison['verification']['shared_frozen_identity_hashes'][method]={k:retire.digest(v) for k,v in frozen.items()}
            comparison['stocks'][0]['methods'][method]={'status':'failed','raw_direction':'WAIT','kronos':{},'score':None,'source_identity_sha256':None}
    snapshot=retire.protected_assets(p)
    snapshot['raw']={'database':{'sha256':raw['database'],'size_bytes':p['raw'].stat().st_size},'wal':None}
    for name,value in [('state.json',state),('comparison.json',comparison),('protection-after.comparison.json',{'passed':True}),
                       ('protection-before.json',snapshot),('protection-after.json',snapshot)]:
        (evidence/name).write_text(json.dumps(value))
    (evidence/'all_stock_comparison.csv').write_text('symbol\n000001.SZ\n')
    (evidence/'screening_comparison.md').write_text('Accepted comparison fixture')
    return state,comparison


def test_actual_acceptance_binds_reports_population_comparison_and_preservation(raw_fixture,tmp_path):
    p,_=raw_fixture;evidence=tmp_path/'evidence';_acceptance_fixture(p,evidence)
    result=_REAL_ACCEPTANCE(p,evidence/'state.json',evidence)
    assert len(result['final_runs'])==3 and len(result['references'])==10


@pytest.mark.parametrize('tamper',['pending','comparison_row','final_row','coverage','protection','run_binding','population'])
def test_actual_acceptance_rejects_tampered_or_pending_evidence(raw_fixture,tmp_path,tamper):
    p,_=raw_fixture;evidence=tmp_path/'evidence';state,comparison=_acceptance_fixture(p,evidence)
    if tamper=='pending':
        state['methods'][retire.METHODS[0]]['submission_intent']={'requested':True};(evidence/'state.json').write_text(json.dumps(state))
    elif tamper=='comparison_row':
        comparison['stocks'][0]['methods'][retire.METHODS[0]]['raw_direction']='BUY';(evidence/'comparison.json').write_text(json.dumps(comparison))
    elif tamper in {'final_row','coverage'}:
        path=p['runs']/('final-'+retire.METHODS[0])/'reports/czsc.json';report=json.loads(path.read_text())
        if tamper=='final_row':report['rows'][0]['method_evaluation']['status']='deferred'
        else:report['coverage']['evaluated']=0
        path.write_text(json.dumps(report))
    elif tamper=='protection':
        path=evidence/'protection-after.json';report=json.loads(path.read_text());report['accounts']['count']=99;path.write_text(json.dumps(report))
    elif tamper=='run_binding':
        comparison['sources'][retire.METHODS[0]]['run_id']='other';(evidence/'comparison.json').write_text(json.dumps(comparison))
    else:
        comparison['population']=['920001.BJ'];(evidence/'comparison.json').write_text(json.dumps(comparison))
    with pytest.raises(ValueError):_REAL_ACCEPTANCE(p,evidence/'state.json',evidence)


def test_unknown_task_state_and_training_lease_block(tmp_path,monkeypatch,service_probe):
    monkeypatch.setattr(retire,'project_processes',lambda p:[])
    with sqlite3.connect(tmp_path/'czsc_tasks.db') as conn:
        conn.execute('CREATE TABLE czsc_tasks(status TEXT)');conn.execute("INSERT INTO czsc_tasks VALUES('mystery')")
    (tmp_path/'light_research').mkdir()
    with sqlite3.connect(tmp_path/'light_research/research.db') as conn:conn.execute('CREATE TABLE leases(expires_at REAL)')
    with sqlite3.connect(tmp_path/'model_training.db') as conn:
        conn.execute('CREATE TABLE training_jobs(status TEXT)');conn.execute("INSERT INTO training_jobs VALUES('running')")
    with sqlite3.connect(tmp_path/'training_resource.db') as conn:
        conn.execute('CREATE TABLE leases(expires_at REAL)');conn.execute('INSERT INTO leases VALUES(?)',(retire.time.time()+60,))
    problems=retire.activity({'metadata':tmp_path},0)
    calls, _ = service_probe
    assert set(calls) == {('127.0.0.1', 8124), ('127.0.0.1', 0)}
    assert len(calls) == 2
    assert any('mystery' in x for x in problems)
    assert any('training_jobs' in x for x in problems)
    assert any('training_resource' in x for x in problems)


def test_independent_python_process_is_detected_without_service_port(raw_fixture,monkeypatch):
    import psutil
    from types import SimpleNamespace
    p,_=raw_fixture
    fake=SimpleNamespace(pid=123456,info={'name':'Python','cmdline':['python','-m','my_strategy.cli'],
        'cwd':str(p['project'])},open_files=lambda:[])
    monkeypatch.setattr(psutil,'process_iter',lambda fields:[fake])
    assert retire.project_processes(p)==['Project/protected-storage process still alive: pid=123456']


def test_training_transaction_lease_is_detected_without_project_cwd_or_args(raw_fixture,monkeypatch):
    import psutil
    from types import SimpleNamespace
    p,_=raw_fixture
    fake=SimpleNamespace(pid=789012,info={'name':'Python','cmdline':['python','-c','worker()'],'cwd':'/tmp'},
        open_files=lambda:[SimpleNamespace(path=str(p['metadata']/'training_resource.db'))])
    monkeypatch.setattr(psutil,'process_iter',lambda fields:[fake])
    assert retire.project_processes(p)==['Project/protected-storage process still alive: pid=789012']


def test_plan_is_bound_to_exact_cleanup_script(raw_fixture):
    _,plan=raw_fixture
    plan['script_sha256']='0'*64
    plan['plan_sha256']=retire.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    with pytest.raises(ValueError,match='different cleanup script'):retire.validate_plan(plan)


def _raw_identity_fixture():
    sha = 'a' * 64
    empty = retire.hashlib.sha256(b'').hexdigest()
    return ({'database': sha, 'wal': empty},
            {'database': {'bytes': 4096, 'sha256': sha}},
            {'database': {'size_bytes': 4096, 'sha256': sha}, 'wal': None})


@pytest.mark.parametrize('accepted_empty', [None, 'explicit'])
def test_empty_wal_close_records_original_and_actual_without_mutation(accepted_empty):
    expected, observed, accepted = _raw_identity_fixture()
    if accepted_empty:
        accepted['wal'] = {'size_bytes': 0, 'sha256': expected['wal']}
    original = deepcopy((expected, observed, accepted))
    result = retire.validate_raw_scan_identity(expected, observed, accepted)
    assert result['relation'] == 'empty_wal_removed_after_close'
    assert result['frozen_identity'] == expected
    assert result['observed_physical'] == observed
    assert (expected, observed, accepted) == original
    result['frozen_identity']['database'] = 'changed'
    result['observed_physical']['database']['bytes'] = -1
    assert (expected, observed, accepted) == original


@pytest.mark.parametrize('wal', [None, 'empty', 'nonempty'])
def test_exact_raw_identity_keeps_existing_acceptance(wal):
    expected, observed, accepted = _raw_identity_fixture()
    if wal is None:
        del expected['wal']
    else:
        size, sha = (0, expected['wal']) if wal == 'empty' else (32, 'b' * 64)
        expected['wal'] = sha
        observed['wal'] = {'bytes': size, 'sha256': sha}
        accepted['wal'] = {'size_bytes': size, 'sha256': sha}
    assert retire.validate_raw_scan_identity(expected, observed, accepted)['relation'] == 'exact'


@pytest.mark.parametrize('tamper', [
    'frozen_nonempty_wal', 'observed_nonempty_wal', 'observed_empty_wrong_sha',
    'observed_nonempty_empty_sha', 'changed_database', 'changed_accepted_database',
    'accepted_nonempty_wal', 'accepted_empty_wrong_sha', 'accepted_malformed_wal', 'missing_accepted_wal',
    'extra_expected_key', 'extra_observed_key', 'extra_physical_key',
    'invalid_frozen_sha', 'invalid_observed_sha', 'missing_expected_database',
    'missing_observed_database', 'new_empty_wal',
])
def test_empty_wal_compatibility_rejects_every_other_identity_difference(tamper):
    expected, observed, accepted = _raw_identity_fixture()
    if tamper == 'frozen_nonempty_wal': expected['wal'] = 'b' * 64
    elif tamper == 'observed_nonempty_wal': observed['wal'] = {'bytes': 32, 'sha256': 'b' * 64}
    elif tamper == 'observed_empty_wrong_sha': observed['wal'] = {'bytes': 0, 'sha256': 'b' * 64}
    elif tamper == 'observed_nonempty_empty_sha': observed['wal'] = {'bytes': 32, 'sha256': expected['wal']}
    elif tamper == 'changed_database': observed['database']['sha256'] = 'b' * 64
    elif tamper == 'changed_accepted_database': accepted['database']['sha256'] = 'b' * 64
    elif tamper == 'accepted_nonempty_wal': accepted['wal'] = {'size_bytes': 32, 'sha256': 'b' * 64}
    elif tamper == 'accepted_empty_wrong_sha': accepted['wal'] = {'size_bytes': 0, 'sha256': 'b' * 64}
    elif tamper == 'accepted_malformed_wal': accepted['wal'] = {}
    elif tamper == 'missing_accepted_wal': del accepted['wal']
    elif tamper == 'extra_expected_key': expected['journal'] = 'b' * 64
    elif tamper == 'extra_observed_key': observed['journal'] = {'bytes': 32, 'sha256': 'b' * 64}
    elif tamper == 'extra_physical_key': observed['database']['other'] = 1
    elif tamper == 'invalid_frozen_sha': expected['wal'] = 'empty'
    elif tamper == 'invalid_observed_sha': observed['database']['sha256'] = 'bad'
    elif tamper == 'missing_expected_database': del expected['database']
    elif tamper == 'missing_observed_database': del observed['database']
    else:
        observed['wal'] = {'bytes': 0, 'sha256': expected.pop('wal')}
    with pytest.raises(ValueError): retire.validate_raw_scan_identity(expected, observed, accepted)


def test_actual_acceptance_allows_only_closed_empty_wal_and_keeps_reports(raw_fixture, tmp_path):
    p, _ = raw_fixture
    wal = Path(str(p['raw']) + '-wal'); wal.write_bytes(b'')
    evidence = tmp_path / 'evidence'; _acceptance_fixture(p, evidence)
    paths = [*evidence.iterdir(), *p['runs'].glob('*/reports/czsc.json')]
    before = {str(path): retire.file_identity(path) for path in paths}
    wal.unlink()
    result = _REAL_ACCEPTANCE(p, evidence / 'state.json', evidence)
    assert result['raw_identity_validation']['relation'] == 'empty_wal_removed_after_close'
    assert result['raw_identity_validation']['frozen_identity']['wal'] == retire.hashlib.sha256(b'').hexdigest()
    assert 'wal' not in result['raw_identity_validation']['observed_physical']
    assert {str(path): retire.file_identity(path) for path in paths} == before


@pytest.mark.parametrize('change', ['reappear_empty', 'reappear_nonempty', 'remove_empty'])
def test_plan_freshness_still_rejects_wal_lifetime_changes(raw_fixture, tmp_path, monkeypatch, change):
    p, _ = raw_fixture
    wal = Path(str(p['raw']) + '-wal'); wal.write_bytes(b'')
    evidence = tmp_path / 'evidence'; _acceptance_fixture(p, evidence)
    if change != 'remove_empty': wal.unlink()
    monkeypatch.setattr(retire, 'accepted_scans', _REAL_ACCEPTANCE)
    plan = retire.preview(p, state=evidence / 'state.json', evidence=evidence, full_protection=True)
    assert plan['blockers'] == []
    if change == 'remove_empty': wal.unlink()
    else: wal.write_bytes(b'' if change == 'reappear_empty' else b'changed')
    with pytest.raises(ValueError, match='differs from accepted scans|changed after preview'):
        retire.apply(p, plan, backup=tmp_path / 'backup.db', journal=tmp_path / 'receipt.json')
    assert not (tmp_path / 'backup.db').exists()
    assert not (tmp_path / 'receipt.json').exists()
    with retire.readonly(p['raw']) as conn: assert retire.symbols(conn) == ['920001.BJ']


def test_closed_wal_acceptance_still_requires_current_protected_assets(raw_fixture, tmp_path, monkeypatch):
    p, _ = raw_fixture
    wal = Path(str(p['raw']) + '-wal'); wal.write_bytes(b'')
    evidence = tmp_path / 'evidence'; _acceptance_fixture(p, evidence); wal.unlink()
    monkeypatch.setattr(retire, 'protected_assets', lambda p: {'changed': True})
    with pytest.raises(ValueError, match='Protected models/accounts changed'):
        _REAL_ACCEPTANCE(p, evidence / 'state.json', evidence)


def test_real_sqlite_normal_close_removes_empty_wal_without_database_change(raw_fixture):
    p, _ = raw_fixture
    conn = sqlite3.connect(p['raw'])
    try:
        assert conn.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
        conn.execute("UPDATE stock_daily_raw SET value='wal fixture' WHERE stock='920001.BJ'")
        conn.commit()
        assert conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0
        before = retire.raw_physical(p)
        assert before['wal']['bytes'] == 0
    finally:
        conn.close()
    after = retire.raw_physical(p)
    assert 'wal' not in after and before['database'] == after['database']
    result = retire.validate_raw_scan_identity(
        {key: value['sha256'] for key, value in before.items()}, after,
        {'database': {'sha256': before['database']['sha256']}, 'wal': None})
    assert result['relation'] == 'empty_wal_removed_after_close'


def test_closed_empty_wal_full_fixture_preview_apply_preserves_protected_rows(raw_fixture, tmp_path, monkeypatch):
    p, _ = raw_fixture
    wal = Path(str(p['raw']) + '-wal'); wal.write_bytes(b'')
    evidence = tmp_path / 'evidence'; _acceptance_fixture(p, evidence); wal.unlink()
    monkeypatch.setattr(retire, 'accepted_scans', _REAL_ACCEPTANCE)
    plan = retire.preview(p, state=evidence / 'state.json', evidence=evidence, full_protection=True)
    assert plan['accepted_scans']['raw_identity_validation']['relation'] == 'empty_wal_removed_after_close'
    result = retire.apply(p, plan, backup=tmp_path / 'backup.db', journal=tmp_path / 'receipt.json')
    assert result['status'] == 'committed'
    with retire.readonly(p['raw']) as conn:
        assert retire.symbols(conn) == []
        assert retire.raw_rows(conn, plan['symbols'], protected=True) == plan['protected_raw_rows']
