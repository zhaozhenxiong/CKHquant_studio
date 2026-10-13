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


class _StorageApiFunction:
    def __init__(self, function):
        self.function, self.calls = function, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.function(*args)


def _storage_windows_api(monkeypatch, *, flags=0x80, allocation=4096, low=4096,
                         high=0, last_error=0, info_success=True):
    import ctypes
    import sys
    from types import SimpleNamespace
    errors = [last_error]

    def compressed(path, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32)).contents.value = high
        errors[0] = last_error
        return low

    def information(handle, info_class, pointer, size):
        assert handle == 12345 and info_class == 1 and size == 24
        pointer._obj.AllocationSize = allocation
        pointer._obj.EndOfFile = 5
        errors[0] = last_error
        return int(info_success)

    kernel = SimpleNamespace(GetFileAttributesW=_StorageApiFunction(lambda _: flags),
                             GetCompressedFileSizeW=_StorageApiFunction(compressed),
                             GetFileInformationByHandleEx=_StorageApiFunction(information))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: errors[0], raising=False)
    monkeypatch.setattr(ctypes, 'set_last_error', lambda value: errors.__setitem__(0, value), raising=False)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(get_osfhandle=lambda fd: 12345))
    return kernel


def _storage_platform(monkeypatch, name):
    import os
    from types import SimpleNamespace
    from my_strategy.services import czsc_account_artifacts as accounts
    # Keep the real global os.name/Path platform intact for cross-platform CI.
    monkeypatch.setattr(accounts, 'os', SimpleNamespace(name=name, fsync=os.fsync))
    return accounts


def test_windows_storage_regular_file_reports_allocation_not_logical_size(tmp_path, monkeypatch):
    from my_strategy.services import czsc_account_artifacts as accounts
    path = tmp_path / 'regular.json'
    path.write_bytes(b'abcde')
    kernel = _storage_windows_api(monkeypatch, allocation=4096, low=5)
    assert accounts._windows_allocated_bytes(path) == 4096
    assert len(kernel.GetFileInformationByHandleEx.calls) == 1
    assert not kernel.GetCompressedFileSizeW.calls


@pytest.mark.parametrize('flags', [0x800, 0x200, 0xA00])
def test_windows_storage_compressed_sparse_high_dword(tmp_path, monkeypatch, flags):
    from my_strategy.services import czsc_account_artifacts as accounts
    path = tmp_path / 'compressed.json'
    path.write_bytes(b'abcde')
    kernel = _storage_windows_api(monkeypatch, flags=flags, low=7, high=2)
    assert accounts._windows_allocated_bytes(path) == (2 << 32) | 7
    assert len(kernel.GetCompressedFileSizeW.calls) == 1
    assert not kernel.GetFileInformationByHandleEx.calls


def test_windows_storage_invalid_low_dword_zero_error_is_success(tmp_path, monkeypatch):
    from my_strategy.services import czsc_account_artifacts as accounts
    path = tmp_path / 'sparse.json'
    path.write_bytes(b'abcde')
    _storage_windows_api(monkeypatch, flags=0x200, low=0xFFFFFFFF, high=3, last_error=0)
    assert accounts._windows_allocated_bytes(path) == (3 << 32) | 0xFFFFFFFF


@pytest.mark.parametrize('flags,info_success,low,error', [(0xFFFFFFFF, True, 3, 5),
                                                       (0x800, True, 0xFFFFFFFF, 5),
                                                       (0x80, False, 3, 6)])
def test_windows_storage_failure_is_null_with_reason(tmp_path, monkeypatch, flags, info_success, low, error):
    path = tmp_path / 'failure.json'
    path.write_bytes(b'abcde')
    _storage_windows_api(monkeypatch, flags=flags, info_success=info_success, low=low, last_error=error)
    accounts = _storage_platform(monkeypatch, 'nt')
    stats = accounts._physical_storage_stats(tmp_path, {path.name: {}})
    assert stats['physical_bytes'] is None
    assert stats['physical_bytes_unavailable_reason'] == 'windows_allocation_query_unavailable_or_failed'
    assert stats['physical_bytes_error_code'] == error


def test_windows_storage_api_unavailable_keeps_written_semantic_assets(tmp_path, monkeypatch):
    import ctypes
    import sys
    from types import SimpleNamespace
    def unavailable(*args, **kwargs):
        raise AttributeError('unavailable')
    monkeypatch.setattr(ctypes, 'WinDLL', unavailable, raising=False)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace())
    _storage_platform(monkeypatch, 'nt')
    documents = account_evidence()
    result = write_account_artifacts(tmp_path, documents)
    assert result['storage_stats']['physical_bytes'] is None
    assert result['storage_stats']['physical_bytes_error_type'] == 'AttributeError'
    for name, original in documents.items():
        entry = result['files'][name]
        path = tmp_path / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        assert read_account_artifact(path, entry['sha256']) == original


def test_windows_storage_partial_allocation_sum_is_not_reported(tmp_path, monkeypatch):
    accounts = _storage_platform(monkeypatch, 'nt')
    calls = []
    def allocated(path):
        calls.append(path.name)
        if path.name == 'second':
            raise OSError(5, 'unavailable')
        return 8192
    monkeypatch.setattr(accounts, '_windows_allocated_bytes', allocated)
    stats = accounts._physical_storage_stats(tmp_path, {'first': {}, 'second': {}, 'third': {}})
    assert stats['physical_bytes'] is None and calls == ['first', 'second']


def test_posix_storage_preserves_stat_order_units_and_no_windows_query(monkeypatch):
    from types import SimpleNamespace
    accounts = _storage_platform(monkeypatch, 'posix')
    calls = []
    class PosixFile:
        def __init__(self, name):
            self.name = name
        def stat(self):
            calls.append(self.name)
            return SimpleNamespace(st_blocks={'first': 8, 'second': 16}[self.name], st_size=5)
    class Directory:
        def __truediv__(self, name):
            return PosixFile(name)
    def forbidden(_):
        raise AssertionError('POSIX must not query Windows')
    monkeypatch.setattr(accounts, '_windows_allocated_bytes', forbidden)
    assert accounts._physical_storage_stats(Directory(), {'first': {}, 'second': {}}) == {'physical_bytes': 24 * 512}
    assert calls == ['first', 'second']


def test_non_windows_unknown_allocated_blocks_explicit_null(monkeypatch):
    from types import SimpleNamespace
    accounts = _storage_platform(monkeypatch, 'posix')
    class UnknownDirectory:
        def __truediv__(self, name):
            return self
        def stat(self):
            return SimpleNamespace(st_size=123456)
    stats = accounts._physical_storage_stats(UnknownDirectory(), {'file': {}})
    assert stats['physical_bytes'] is None
    assert stats['physical_bytes_unavailable_reason'] == 'posix_allocated_blocks_unavailable'
