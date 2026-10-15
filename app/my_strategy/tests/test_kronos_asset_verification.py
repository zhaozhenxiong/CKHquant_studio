"""Pinned bytes remain authoritative when filesystem timestamps have low precision."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from my_strategy.services import kronos_assets as assets


@pytest.fixture(autouse=True)
def isolated_verification_cache(monkeypatch):
    monkeypatch.setattr(assets, '_VERIFIED', {})
    # Exercise the weight strategy with tiny fixtures, never workstation models.
    monkeypatch.setattr(assets, '_MEMO_MIN_BYTES', 1)
    monkeypatch.setattr(assets.urllib.request, 'urlopen',
                        lambda *args, **kwargs: pytest.fail('Verification attempted a download'))


def pin(tmp_path, raw=b'original pinned weight', name='model.safetensors', **extra):
    path = tmp_path / name
    path.write_bytes(raw)
    return path, {'path': name, 'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'repo': 'synthetic', 'revision': 'frozen', **extra}


@pytest.mark.parametrize('identity_kind', ['sha256', 'git_blob'])
def test_config_always_checks_bytes_when_all_observed_metadata_is_unchanged(tmp_path, monkeypatch, identity_kind):
    raw = b'{"value": 1}'
    path, entry = pin(tmp_path, raw, 'config.json')
    if identity_kind == 'git_blob':
        entry.pop('sha256')
        entry['git_blob'] = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    monkeypatch.setattr(assets, 'PINNED_ASSETS', (entry,))
    monkeypatch.setattr(assets, '_stat_identity', lambda _: ('unchanged metadata',))
    monkeypatch.setattr(assets, '_memo_version', lambda _: ('unchanged version',))
    assert assets.asset_status(tmp_path)['ready']
    assert not assets._VERIFIED
    path.write_bytes(b'{"value": 2}')
    rejected = assets.asset_status(tmp_path)
    assert not rejected['ready'] and rejected['errors']
    with pytest.raises(ValueError, match='SHA|configuration identity'):
        assets.prepare_assets(tmp_path)


def test_windows_weight_rehashes_even_with_identical_file_metadata(tmp_path, monkeypatch):
    raw = b'original pinned weight'
    path, entry = pin(tmp_path, raw)
    monkeypatch.setattr(assets, 'PINNED_ASSETS', (entry,))
    monkeypatch.setattr(assets, '_WINDOWS', True)
    monkeypatch.setattr(assets, '_stat_identity', lambda _: ('same Windows stat',))
    observed = []
    original = assets.hashlib.file_digest

    def digest(handle, algorithm):
        observed.append(handle.name)
        return original(handle, algorithm)

    monkeypatch.setattr(assets.hashlib, 'file_digest', digest)
    assert assets.asset_status(tmp_path)['ready']
    assert assets.asset_status(tmp_path)['ready']
    assert len(observed) == 2 and not assets._VERIFIED
    path.write_bytes(b'changed! pinned weight')
    assert path.stat().st_size == len(raw)
    result = assets.asset_status(tmp_path)
    assert not result['ready'] and 'SHA mismatch' in result['errors'][0]
    assert len(observed) == 3
    with pytest.raises(ValueError, match='SHA mismatch'):
        assets.prepare_assets(tmp_path)


def test_posix_weight_memo_is_bounded_and_binds_the_whole_pin(tmp_path, monkeypatch):
    path, entry = pin(tmp_path)
    monkeypatch.setattr(assets, '_WINDOWS', False)
    observed = []
    original = assets.hashlib.file_digest

    def digest(handle, algorithm):
        observed.append(handle.name)
        return original(handle, algorithm)

    monkeypatch.setattr(assets.hashlib, 'file_digest', digest)
    first = assets._verify(path, entry)
    assert assets._verify(path, entry) == first and len(observed) == 1
    assert assets._verify(path, {**entry, 'revision': 'new-pin'})['revision'] == 'new-pin'
    assert len(observed) == 2 and len(assets._VERIFIED) == 1
    with pytest.raises(ValueError, match='SHA mismatch'):
        assets._verify(path, {**entry, 'sha256': '0' * 64})
    assert len(observed) == 3
    for index in range(assets._MEMO_LIMIT + 2):
        directory = tmp_path / str(index)
        directory.mkdir()
        item, identity = pin(directory)
        assets._verify(item, identity)
    assert len(assets._VERIFIED) == assets._MEMO_LIMIT


def test_same_size_path_replacement_during_hash_cannot_publish_verification(tmp_path, monkeypatch):
    path, entry = pin(tmp_path)
    monkeypatch.setattr(assets, 'PINNED_ASSETS', (entry,))
    monkeypatch.setattr(assets, '_WINDOWS', True)
    original = assets.hashlib.file_digest

    def digest(handle, algorithm):
        result = original(handle, algorithm)
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return result

    monkeypatch.setattr(assets.hashlib, 'file_digest', digest)
    result = assets.asset_status(tmp_path)
    # If Windows denies replacement of the open handle, that OSError also fails
    # closed. A successful replacement must be rejected by the path/file ID.
    assert not result['ready'] and result['errors'] and not assets._VERIFIED


def test_config_digests_share_one_observed_byte_sequence(tmp_path, monkeypatch):
    raw = b'{"value": 1}'
    path, entry = pin(tmp_path, raw, 'config.json')
    entry['git_blob'] = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    original = Path.read_bytes

    def changed_read(current):
        if current == path:
            return b'{"value": 2}'
        return original(current)

    monkeypatch.setattr(Path, 'read_bytes', changed_read)
    with pytest.raises(ValueError, match='changed during verification'):
        assets._verify(path, entry)
