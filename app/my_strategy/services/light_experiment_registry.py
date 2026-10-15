"""Bounded trial ledger with shared, pinned input identities.

Trial IDs always describe the original complete experiment. Moving repeated
input provenance into CAS never creates a trial, resets a budget, or edits a
model. Input objects commit before the small registry setting is published.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import json

from my_strategy.services.light_qualification import content_hash
from my_strategy.storage.research_store import canonical

SCHEMA = 'light_experiment_registry_v1'
INPUT_SCHEMA = 'light_experiment_input_v1'
REF_SCHEMA = 'light_experiment_input_ref_v1'
NAMESPACE = 'experiment_input_identities'
SETTING = 'light_experiment_registry'
LIMITS = {'mlp': 6, 'kronos': 2, 'policy': 6}


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid experiment input SHA256')
    return value


def _normalized(value):
    result = deepcopy(value or {'schema': SCHEMA, 'entries': []})
    if result.get('schema', SCHEMA) != SCHEMA or not isinstance(result.get('entries'), list):
        raise ValueError('Invalid experiment registry')
    result['schema'] = SCHEMA
    inputs = result.pop('input_identities', {})
    if not isinstance(inputs, dict): raise ValueError('Invalid shared experiment inputs')
    for sha, identity in inputs.items():
        if content_hash(identity) != _digest(sha): raise ValueError('Shared experiment input identity mismatch')
    for item in result['entries']:
        if not isinstance(item, dict) or item.get('kind') not in LIMITS:
            raise ValueError('Unknown experiment kind')
        if not isinstance(item.get('experiment_id'), str) or not item['experiment_id']:
            raise ValueError('Experiment ID is required')
        if 'input_identity' in item:
            identity = item.pop('input_identity')
            sha = content_hash(identity)
            if 'input_identity_ref' in item and item['input_identity_ref'].get('sha256') != sha:
                raise ValueError('Inline and referenced experiment inputs disagree')
            inputs[sha] = identity
            item['input_identity_ref'] = {'schema': REF_SCHEMA, 'sha256': sha}
        ref = item.get('input_identity_ref')
        if ref is not None:
            if not isinstance(ref, dict) or ref.get('schema') != REF_SCHEMA:
                raise ValueError('Invalid experiment input reference')
            sha = _digest(ref.get('sha256'))
            if 'cache_key' in ref:
                expected = content_hash({'namespace':NAMESPACE,
                    'identity':{'schema':INPUT_SCHEMA,'input_identity_sha256':sha}})
                if ref['cache_key'] != expected: raise ValueError('Experiment input cache identity mismatch')
    if inputs: result['input_identities'] = inputs
    return result


def _entry_identity(item):
    value = deepcopy(item)
    if value.get('input_identity_ref'):
        value['input_identity_ref'].pop('cache_key', None)
    return canonical(value)


def _check_budget(entries):
    if len(entries) > 14 or any(sum(e['kind'] == kind for e in entries) > limit for kind, limit in LIMITS.items()):
        raise ValueError('Pre-registered experiment budget exceeded')


def merge_experiment_registry(existing, new_entries=()):
    """Share inline identities in memory; retain existing IDs/order/counts.

    The optional input_identities table is transport-only. The persistence helper
    replaces it with CAS references before writing the 64-KiB setting.
    """
    result = _normalized(existing)
    proposed = _normalized({'schema': SCHEMA, 'entries': list(new_entries)})
    inputs = {**result.pop('input_identities', {}), **proposed.pop('input_identities', {})}
    for item in proposed['entries']:
        known = [e for e in result['entries'] if e['experiment_id'] == item['experiment_id']]
        if known:
            if any(_entry_identity(e) != _entry_identity(item) for e in known):
                raise ValueError('Experiment ID already registered with different evidence')
        else:
            result['entries'].append(item)
    _check_budget(result['entries'])
    if inputs: result['input_identities'] = inputs
    return result


def preserves_experiment_trials(registry, original):
    """Compare trial facts across inline/shared/CAS layouts, including counts.

    A matching ID alone is insufficient: kind, parameters and the complete
    input's semantic SHA must also match the immutable model's registration.
    """
    current = _normalized(registry)
    previous = _normalized(original)
    current_counts = Counter(_entry_identity(e) for e in current['entries'])
    previous_counts = Counter(_entry_identity(e) for e in previous['entries'])
    return bool(previous_counts) and all(current_counts[key] >= count for key,count in previous_counts.items())


def _input_key(store, sha):
    identity = {'schema': INPUT_SCHEMA, 'input_identity_sha256': _digest(sha)}
    return identity, store.cache_key(identity, NAMESPACE)


def load_experiment_input(store, ref):
    """Read the complete shared source identity, checking CAS and semantic SHA."""
    if not isinstance(ref, dict) or ref.get('schema') != REF_SCHEMA:
        raise ValueError('Invalid experiment input reference')
    sha = _digest(ref.get('sha256'))
    _, expected = _input_key(store, sha)
    if ref.get('cache_key') != expected: raise ValueError('Experiment input cache identity mismatch')
    payload = store.get_by_key(expected)
    if (not isinstance(payload, dict) or payload.get('schema') != INPUT_SCHEMA
            or payload.get('input_identity_sha256') != sha
            or content_hash(payload.get('input_identity')) != sha):
        raise ValueError('Experiment input is missing or its source identity changed')
    return deepcopy(payload['input_identity'])


def _persist_inputs(store, registry, setting_name):
    result = _normalized(registry)
    inputs = result.pop('input_identities', {})
    written = set()
    for item in result['entries']:
        ref = item.get('input_identity_ref')
        if not ref: continue
        sha = ref['sha256']; identity, key = _input_key(store, sha)
        owner = 'experiment_registry:' + setting_name
        if sha in written:
            pass
        elif sha in inputs:
            payload = {'schema': INPUT_SCHEMA, 'input_identity_sha256': sha, 'input_identity': inputs[sha]}
            receipt = store.put(identity, payload, namespace=NAMESPACE, pin=owner)
            if receipt['cache_key'] != key: raise ValueError('Experiment input publication identity changed')
        else:
            # An old compact reference must still resolve before publishing any
            # successor. Never silently replace missing provenance with a hash.
            load_experiment_input(store, ref)
            store.pin(key, owner)
        written.add(sha)
        item['input_identity_ref'] = {'schema': REF_SCHEMA, 'sha256': sha, 'cache_key': key}
    return result


def persist_experiment_registry(store, registry, *, setting_name=SETTING):
    """Safely migrate inline registries and atomically append registered trials.

    Pinned input objects are written first. A compare-and-swap SQLite transaction
    then publishes the compact registry, preserving concurrent registrations.
    Failed publication leaves the previous setting and its source inputs intact.
    """
    if not isinstance(setting_name, str) or not setting_name or len(setting_name) > 200:
        raise ValueError('Invalid experiment registry setting name')
    proposed = _normalized(registry)
    _check_budget(proposed['entries'])
    for _ in range(8):
        before = store.get_setting(setting_name)
        current = _normalized(before)
        inputs = {**current.pop('input_identities', {}), **proposed.get('input_identities', {})}
        if inputs: current['input_identities'] = inputs
        combined = merge_experiment_registry(current, proposed['entries'])
        compact = _persist_inputs(store, combined, setting_name)
        text = canonical(compact)
        if len(text.encode()) > 65536:
            raise ValueError('Compact experiment registry still exceeds the 64 KiB setting limit')
        # This is the same bounded setting representation as ResearchStore;
        # direct SQL is needed to compare and publish inside one writer lock.
        with store._tx():
            row = store.connection.execute('SELECT value_json FROM settings WHERE name=?', (setting_name,)).fetchone()
            actual = json.loads(row[0]) if row else None
            if canonical(actual) != canonical(before): continue
            store.connection.execute('INSERT INTO settings VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at',
                (setting_name, text, store.clock()))
        return compact
    raise RuntimeError('Experiment registry changed concurrently; retry registration')
