from copy import deepcopy
import json
import sqlite3

import pytest

from my_strategy.services.light_experiment_registry import (
    NAMESPACE, SETTING, merge_experiment_registry, persist_experiment_registry, load_experiment_input,
)
from my_strategy.services.light_qualification import content_hash
from my_strategy.services.light_score_v2 import _registry
from my_strategy.storage.research_store import ResearchStore, canonical


def large_identity(index=0):
    value={'sources':[{'symbol':'002436.SZ','snapshot':'frozen-'+str(index),
        'engine_manifest':{'source_code':('code-'+str(index)+'-')*6000},
        'raw_sha256':'a'*64,'population':['002436.SZ','600519.SH']}],
        'labels':{'horizon':5,'mature':True}}
    assert len(canonical(value).encode())>=35000
    return value


def entry(kind,index,identity=None):
    value={'kind':kind,'configuration':{'trial':index}}
    if identity is not None:value.update(variant='A',input_identity=identity)
    value['experiment_id']=content_hash(value)
    return value


def test_six_35kb_inputs_share_one_cas_object_and_keep_original_trial_ids(tmp_path):
    identity=large_identity();trials=[{'seed':i} for i in range(6)]
    original=[{'kind':'mlp','variant':'A','configuration':trial,'input_identity':identity} for trial in trials]
    ids=[content_hash(item) for item in original]
    registry=_registry(None,trials,'A',identity)
    assert len(registry['input_identities'])==1
    assert [e['experiment_id'] for e in registry['entries']]==ids
    with ResearchStore(tmp_path) as store:
        saved=persist_experiment_registry(store,registry)
        assert 'input_identities' not in saved
        assert len(canonical(saved).encode())<8192
        assert len(store.list(NAMESPACE))==1
        assert [e['experiment_id'] for e in saved['entries']]==ids
        assert all(load_experiment_input(store,e['input_identity_ref'])==identity for e in saved['entries'])
        # This was the real failure: adding cost policies to repeated MLP input
        # provenance overflowed settings even though the trial budget was valid.
        for stress in (1.,1.5,2.):
            saved=persist_experiment_registry(store,merge_experiment_registry(saved,[entry('policy',stress)]))
        assert len(saved['entries'])==9 and len(store.list(NAMESPACE))==1
        assert store.get_setting(SETTING)==saved
        same=persist_experiment_registry(store,_registry(saved,trials,'A',identity))
        assert same==saved
        with pytest.raises(ValueError,match='experiment budget'):
            _registry(saved,[{'seed':6}],'A',identity)
        assert store.get_setting(SETTING)==saved


def test_six_distinct_35kb_inputs_are_referenced_without_raising_settings_limit(tmp_path):
    entries=[entry('mlp',i,large_identity(i)) for i in range(6)]
    registry={'schema':'light_experiment_registry_v1','entries':entries}
    with ResearchStore(tmp_path) as store:
        saved=persist_experiment_registry(store,registry)
        assert len(store.list(NAMESPACE))==6 and len(canonical(saved).encode())<8192
        for original,compact in zip(entries,saved['entries']):
            assert compact['experiment_id']==original['experiment_id']
            assert load_experiment_input(store,compact['input_identity_ref'])==original['input_identity']
        with pytest.raises(ValueError,match='64 KiB'):
            store.set_setting('still-bounded','x'*65536)


def test_inline_migration_preserves_old_model_and_every_existing_registration(tmp_path):
    old={'schema':'light_experiment_registry_v1','entries':[entry('mlp',0,large_identity())],
        'round':'original-round'}
    original=deepcopy(old)
    with ResearchStore(tmp_path) as store:
        store.set_setting(SETTING,old)
        model=store.put({'model':'already-trained'},{'registry':old,'weights':[1,2,3]},pin='saved-model')
        migrated=persist_experiment_registry(store,old)
        assert old==original and migrated['round']=='original-round'
        assert [r['experiment_id'] for r in migrated['entries']]==[r['experiment_id'] for r in old['entries']]
        assert store.get_by_key(model['cache_key'])=={'registry':old,'weights':[1,2,3]}
        assert persist_experiment_registry(store,old)==migrated
        assert len(store.list(NAMESPACE))==1


def test_failed_pointer_publication_preserves_old_registry_and_protects_cas(tmp_path):
    now=[1.]
    old={'entries':[entry('mlp',0,large_identity())]}
    with ResearchStore(tmp_path,ttl_seconds=1,clock=lambda:now[0]) as store:
        store.set_setting(SETTING,old)
        store.connection.execute("CREATE TRIGGER fail_registry BEFORE UPDATE ON settings WHEN NEW.name='light_experiment_registry' BEGIN SELECT RAISE(ABORT,'simulated publication interruption'); END")
        with pytest.raises(sqlite3.IntegrityError,match='publication interruption'):
            persist_experiment_registry(store,old)
        assert store.get_setting(SETTING)==old
        assert len(store.list(NAMESPACE))==1 and store.list(NAMESPACE)[0]['pin_count']==1
        now[0]=100.
        store.gc(dry_run=False)
        assert len(store.list(NAMESPACE))==1
        store.connection.execute('DROP TRIGGER fail_registry')
        saved=persist_experiment_registry(store,old)
        assert load_experiment_input(store,saved['entries'][0]['input_identity_ref'])==old['entries'][0]['input_identity']


def test_missing_or_tampered_input_cannot_publish_a_successor(tmp_path):
    with ResearchStore(tmp_path) as store:
        saved=persist_experiment_registry(store,{'entries':[entry('mlp',0,large_identity())]})
        bad=deepcopy(saved);bad['entries'][0]['input_identity_ref']['cache_key']='f'*64
        with pytest.raises(ValueError,match='cache identity'):
            persist_experiment_registry(store,bad)
        assert store.get_setting(SETTING)==saved
        item=store.list(NAMESPACE)[0];path=store._path(item['sha256'])
        path.chmod(0o644);path.write_bytes(b'corrupt')
        with pytest.raises(ValueError,match='SHA/size'):
            persist_experiment_registry(store,merge_experiment_registry(saved,[entry('policy',1)]))
        assert store.get_setting(SETTING)==saved


def test_concurrent_append_during_cas_preparation_is_merged_without_lost_budget(tmp_path,monkeypatch):
    with ResearchStore(tmp_path) as store:
        old={'entries':[entry('mlp',0,large_identity())]}
        store.set_setting(SETTING,old)
        original_put=store.put;race=[False]
        def put(*args,**kwargs):
            result=original_put(*args,**kwargs)
            if not race[0]:
                race[0]=True
                # Simulate a second writer after our read and before publication.
                store.set_setting(SETTING,{'entries':old['entries']+[entry('policy','other-writer')]})
            return result
        monkeypatch.setattr(store,'put',put)
        saved=persist_experiment_registry(store,merge_experiment_registry(old,[entry('policy','this-writer')]))
        assert [e['experiment_id'] for e in saved['entries']]==[
            old['entries'][0]['experiment_id'],entry('policy','other-writer')['experiment_id'],entry('policy','this-writer')['experiment_id']]
        assert len(store.list(NAMESPACE))==1


def test_duplicate_ids_do_not_reset_existing_counts_and_changed_trial_is_rejected():
    old={'entries':[entry('mlp',i) for i in range(6)]+[entry('policy',i) for i in range(6)]+[entry('kronos',i) for i in range(2)]}
    assert len(merge_experiment_registry(old,[old['entries'][0]])['entries'])==14
    with pytest.raises(ValueError,match='experiment budget'):
        merge_experiment_registry(old,[entry('policy',6)])
    changed=deepcopy(old['entries'][0]);changed['configuration']['trial']='changed'
    with pytest.raises(ValueError,match='different evidence'):
        merge_experiment_registry(old,[changed])
    assert len(old['entries'])==14
