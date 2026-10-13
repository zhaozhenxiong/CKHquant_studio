"""Account evidence preserves real targets while storing repeated facts once."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil

import numpy as np
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_account_artifacts import (
    ACCOUNT_GRAPH_SCHEMA, ACCOUNT_REFERENCE_SCHEMA,
    read_account_artifact, validate_account_artifact,
    write_account_artifact, write_account_artifacts,
)
from my_strategy.services.czsc_trend_support import _json


def account_evidence():
    trades = [{'date': f'2026-01-{i + 1:02}', 'action': 'BUY' if i % 2 == 0 else 'SELL',
               'shares': 1000, 'cash_flow': -10005. if i % 2 == 0 else 11095.,
               'fee': 5., 'causal_identity': 'trade-' + str(i)} for i in range(16)]
    outputs = {'family': 'trend_consensus', 'verified_frozen_contract': True,
               'checkpoint_sha256': 'a' * 64, 'p_up': .1, 'p_down': .8, 'p_neutral': .1,
               'ordered_features': ['ma' + str(i) for i in range(100)]}
    reference = {'ledger': trades, 'trades': trades, 'broker_fills': trades,
                 'account_snapshots': [{'ledger': trades[:min(i // 8 + 1, 16)],
                     'date': str(i), 'model_output': outputs, 'cash': 100000. - i}
                     for i in range(128)],
                 'terminal_state': {'ledger': trades, 'model_output': outputs}}
    clones = [{'ledger': trades, 'daily': [{'date': str(i), 'equity': 101000.} for i in range(20)],
               'outcome': {'net_return': .1089, 'label': 1., 'r_multiple': 3.,
                           'label_available': True, 'label_end': '2026-01-16'}}]
    return {'reference_account.json': reference, 'candidate_clones.json': clones,
            'holding_pairs.json': [{'forced_ledger': trades[-1:], 'outcome': clones[0]['outcome']}]}


def rewrite_graph(directory, envelope, graph):
    """Keep byte hashes valid to exercise structural validation separately."""
    content = json.dumps(graph, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode()
    compressed = gzip.compress(content, mtime=0)
    sha = hashlib.sha256(compressed).hexdigest()
    envelope.update(asset='account_evidence.' + sha + '.json.gz', asset_sha256=sha,
                    graph_sha256=hashlib.sha256(content).hexdigest(), graph_bytes=len(content))
    (directory / envelope['asset']).write_bytes(compressed)
    (directory / envelope['document']).write_text(json.dumps(envelope))


def test_shared_graph_removes_repetition_before_gzip_and_preserves_every_object(tmp_path):
    documents = account_evidence()
    metadata = write_account_artifacts(tmp_path, documents)
    stats = metadata['storage_stats']
    assert stats['schema'] == ACCOUNT_REFERENCE_SCHEMA
    assert stats['graph_schema'] == ACCOUNT_GRAPH_SCHEMA
    assert stats['graph_canonical_bytes'] < stats['expanded_canonical_bytes'] * .3
    assert stats['deduplicated_composite_occurrences'] > 100
    assert len([name for name in metadata['files'] if name.endswith('.gz')]) == 1
    for name, original in documents.items():
        path = tmp_path / name
        restored = read_account_artifact(path, metadata['files'][name]['sha256'])
        assert restored == original
        assert stable_hash(restored) == stable_hash(original)
        assert json.loads(path.read_text())['asset'] == stats['asset']
        assert validate_account_artifact(path)['expanded_semantics_verified'] is False


def test_references_expand_independently_as_a_normal_json_read(tmp_path):
    write_account_artifacts(tmp_path, account_evidence())
    restored = read_account_artifact(tmp_path / 'reference_account.json')
    restored['ledger'][0]['fee'] = 999.
    assert restored['trades'][0]['fee'] == 5.
    restored['account_snapshots'][0]['model_output']['p_down'] = .2
    assert restored['account_snapshots'][1]['model_output']['p_down'] == .8


def test_serialization_normalization_matches_former_json_writer(tmp_path):
    original = {'numeric': np.int64(7), 'positive': np.bool_(True),
                'masked': np.float64(np.nan), 'array': np.array([1., np.inf]),
                'tuple': (1, 2), 'minus_zero': -0.0, 'ref': {'ref': 1}}
    path = tmp_path / 'account.json'
    write_account_artifact(path, original)
    restored = read_account_artifact(path)
    assert restored == _json(original)
    assert stable_hash(restored) == stable_hash(_json(original))


def test_assets_are_deterministic_across_input_order_and_relocation(tmp_path):
    first = tmp_path / 'first'
    documents = account_evidence()
    one = write_account_artifacts(first, documents)
    second = tmp_path / 'second'
    two = write_account_artifacts(second, dict(reversed(list(documents.items()))))
    assert one['storage_stats']['asset'] == two['storage_stats']['asset']
    for name in one['files']:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    relocated = tmp_path / 'relocated'
    shutil.move(str(first), relocated)
    for name in documents:
        assert read_account_artifact(relocated / name) == documents[name]


def test_plain_json_is_rejected_except_explicit_archive_read(tmp_path):
    path = tmp_path / 'account.json'
    path.write_text(json.dumps(account_evidence()['reference_account.json']))
    with pytest.raises(ValueError, match='archive-only'):
        read_account_artifact(path)
    assert read_account_artifact(path, allow_plain=True)['ledger']


def test_unknown_schema_is_never_allowed_as_archive(tmp_path):
    path = tmp_path / 'account.json'
    path.write_text(json.dumps({'schema': 'unsupported'}))
    with pytest.raises(ValueError, match='unsupported'):
        read_account_artifact(path, allow_plain=True)


def test_document_reference_cannot_be_substituted_under_another_filename(tmp_path):
    write_account_artifacts(tmp_path, account_evidence())
    source = tmp_path / 'candidate_clones.json'
    target = tmp_path / 'reference_account.json'
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='does not match its filename'):
        validate_account_artifact(target)


@pytest.mark.parametrize('kind', ['reference', 'asset'])
def test_corrupted_stored_bytes_are_rejected(tmp_path, kind):
    metadata = write_account_artifacts(tmp_path, account_evidence())
    path = tmp_path / 'reference_account.json'
    if kind == 'reference':
        path.write_bytes(path.read_bytes() + b' ')
    else:
        asset = tmp_path / metadata['storage_stats']['asset']
        asset.write_bytes(asset.read_bytes()[:-1] + b'x')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        read_account_artifact(path, metadata['files'][path.name]['sha256'])


@pytest.mark.parametrize('asset', ['../outside.json.gz', '/tmp/outside.json.gz', r'..\outside.json.gz'])
def test_asset_traversal_is_rejected(tmp_path, asset):
    write_account_artifacts(tmp_path, account_evidence())
    path = tmp_path / 'reference_account.json'
    envelope = json.loads(path.read_text())
    envelope['asset'] = asset
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='contained relative filename'):
        validate_account_artifact(path)


def test_symlinked_asset_is_rejected(tmp_path):
    metadata = write_account_artifacts(tmp_path, account_evidence())
    asset = tmp_path / metadata['storage_stats']['asset']
    outside = tmp_path.parent / 'outside-account-evidence.gz'
    asset.replace(outside)
    asset.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        read_account_artifact(tmp_path / 'reference_account.json')


@pytest.mark.parametrize('damage', ['schema', 'reference', 'logical_size', 'semantic_hash'])
def test_graph_and_restored_identity_are_verified(tmp_path, damage):
    write_account_artifacts(tmp_path, account_evidence())
    path = tmp_path / 'reference_account.json'
    envelope = json.loads(path.read_text())
    graph = json.loads(gzip.decompress((tmp_path / envelope['asset']).read_bytes()))
    if damage == 'schema':
        graph['schema'] = 'unsupported'
    elif damage == 'reference':
        graph['objects'][0] = {'l': [{'ref': 0}]}
    elif damage == 'logical_size':
        envelope['expanded_bytes'] += 1
    else:
        envelope['semantic_sha256'] = '0' * 64
    rewrite_graph(tmp_path, envelope, graph)
    with pytest.raises(ValueError):
        read_account_artifact(path)


def test_graph_expansion_bomb_is_rejected_without_materialization(tmp_path):
    write_account_artifacts(tmp_path, account_evidence())
    path = tmp_path / 'reference_account.json'
    envelope = json.loads(path.read_text())
    nodes = [{'l': ['large']}]
    for index in range(31):
        nodes.append({'l': [{'ref': index}, {'ref': index}]})
    graph = {'schema': ACCOUNT_GRAPH_SCHEMA, 'objects': nodes,
             'documents': {path.name: {'ref': len(nodes) - 1}}}
    rewrite_graph(tmp_path, envelope, graph)
    with pytest.raises(ValueError, match='beyond supported size'):
        validate_account_artifact(path)
