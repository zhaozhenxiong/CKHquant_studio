import json
import sqlite3

import pytest

from my_strategy.scripts import retire_failed_training_wrappers as cleanup
from my_strategy.scripts.migrate_native_storage import atomic_json
from my_strategy.storage.research_store import ResearchStore


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, 'process_iter', lambda *_: [])
    meta=tmp_path/'my_strategy/data/metadata';meta.mkdir(parents=True)
    with sqlite3.connect(meta/'czsc_tasks.db') as c:
        c.execute('CREATE TABLE czsc_tasks(payload TEXT)')
        for i,run in enumerate(cleanup.RUN_IDS):
            c.execute('INSERT INTO czsc_tasks VALUES(?)',(json.dumps({'job_id':str(i),'run_id':run,'kind':'model_training','status':'failed','error':'test failure'}),))
            folder=tmp_path/'my_strategy/artifacts/runs'/run;folder.mkdir(parents=True)
            (folder/'metadata.json').write_text(json.dumps({'run_id':run}))
    results=tmp_path/'my_strategy/data/processed/czsc';results.mkdir(parents=True)
    with sqlite3.connect(results/'results.db') as c:c.execute('CREATE TABLE czsc_runs(run_id TEXT,summary TEXT,report_path TEXT)')
    return tmp_path


def test_dry_run_then_apply_exact_wrappers_preserves_receipt(workspace):
    audit=workspace/'audit'
    plan=cleanup.inspect(workspace)
    assert len(plan['candidates'])==4
    atomic_json(audit/'failed_wrapper_cleanup_dry_run.json',plan)
    done=cleanup.apply(workspace,audit)
    assert len(done['deleted_run_ids'])==4
    assert done['native_sources_deleted']==done['models_deleted']==done['raw_files_deleted']==0
    assert (audit/'failed_wrapper_cleanup_completed.json').exists()


def test_recovered_succeeded_same_run_is_never_deleted(workspace):
    with sqlite3.connect(workspace/'my_strategy/data/metadata/czsc_tasks.db') as c:
        c.execute('INSERT INTO czsc_tasks VALUES(?)',(json.dumps({'job_id':'recovered','run_id':cleanup.RUN_IDS[0],'kind':'model_training','status':'succeeded'}),))
    with pytest.raises(ValueError,match='recovered task'):cleanup.inspect(workspace)
    assert (workspace/'my_strategy/artifacts/runs'/cleanup.RUN_IDS[0]).exists()


def test_pinned_payload_reference_blocks_deletion(workspace):
    with ResearchStore(workspace/'my_strategy/data/processed/research') as store:
        store.put({'kind':'backtest'}, {'source':cleanup.RUN_IDS[0]},pin='account:one')
    with pytest.raises(ValueError,match='pinned_result'):cleanup.inspect(workspace)


def test_changed_files_after_preview_are_preserved(workspace):
    audit=workspace/'audit';atomic_json(audit/'failed_wrapper_cleanup_dry_run.json',cleanup.inspect(workspace))
    path=workspace/'my_strategy/artifacts/runs'/cleanup.RUN_IDS[0]/'metadata.json'
    path.write_text('changed')
    with pytest.raises(ValueError,match='changed since dry-run'):cleanup.apply(workspace,audit)
    assert path.exists()
