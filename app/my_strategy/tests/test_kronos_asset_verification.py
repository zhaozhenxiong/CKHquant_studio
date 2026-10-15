"""Pinned bytes remain authoritative when filesystem timestamps have low precision."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

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
    # This is a memo-contract unit test, including when run on Windows. Supply
    # an explicit POSIX physical version instead of assuming that forcing the
    # platform flag also gives Windows fstat/stat POSIX timestamp semantics.
    # Unmocked file observations are covered separately below.
    physical_version = {'value': 1}
    monkeypatch.setattr(assets, '_stat_identity',
                        lambda info: (1, 42, info.st_size, physical_version['value'],
                                      physical_version['value']))
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
    physical_version['value'] += 1
    assert assets._verify(path, entry) == first
    assert len(observed) == 4 and len(assets._VERIFIED) == 1
    for index in range(assets._MEMO_LIMIT + 2):
        directory = tmp_path / str(index)
        directory.mkdir()
        item, identity = pin(directory)
        assets._verify(item, identity)
    assert len(assets._VERIFIED) == assets._MEMO_LIMIT


def test_posix_memo_version_uses_every_physical_identity_field(monkeypatch):
    monkeypatch.setattr(assets, '_WINDOWS', False)
    fields = dict(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5)
    baseline = assets._memo_version(SimpleNamespace(**fields))
    assert baseline == tuple(fields.values())
    for name, value in fields.items():
        assert assets._memo_version(SimpleNamespace(**{**fields, name: value + 10})) != baseline
    monkeypatch.setattr(assets, '_WINDOWS', True)
    assert assets._memo_version(SimpleNamespace(**fields)) is None


def test_windows_full_sha_with_unmocked_fresh_file_observations(tmp_path, monkeypatch):
    # On Windows CI these are real Windows fstat/path-stat observations. Do not
    # mock them, sleep, retry, or suppress a changed-during-verification failure.
    monkeypatch.setattr(assets, '_WINDOWS', True)
    observed = []
    original = assets.hashlib.file_digest

    def digest(handle, algorithm):
        observed.append(handle.name)
        return original(handle, algorithm)

    monkeypatch.setattr(assets.hashlib, 'file_digest', digest)
    for index in range(assets._MEMO_LIMIT + 2):
        directory = tmp_path / str(index)
        directory.mkdir()
        path, entry = pin(directory)
        assert assets._verify(path, entry)['sha256'] == entry['sha256']
        assert assets._verify(path, entry)['sha256'] == entry['sha256']
        path.write_bytes(b'changed! pinned weight')
        with pytest.raises(ValueError, match='SHA mismatch'):
            assets._verify(path, entry)
    assert len(observed) == 3 * (assets._MEMO_LIMIT + 2)
    assert not assets._VERIFIED


@pytest.mark.parametrize('name,raw', [('config.json', b'{"value": 1}'),
                                     ('model.safetensors', b'original pinned weight')])
def test_windows_path_birthtime_differs_from_handle_change_time(tmp_path, monkeypatch, name, raw):
    path, entry = pin(tmp_path, raw, name)
    monkeypatch.setattr(assets, '_WINDOWS', True)
    original_stat = Path.stat

    def path_stat(current, *args, **kwargs):
        info = original_stat(current, *args, **kwargs)
        if current == path:
            # CPython 3.12.10 win32_xstat copies birthtime into path ctime;
            # _Py_fstat_noraise instead retains FILE_BASIC_INFO.ChangeTime.
            return SimpleNamespace(st_dev=info.st_dev, st_ino=info.st_ino,
                                   st_size=info.st_size, st_mtime_ns=info.st_mtime_ns,
                                   st_ctime_ns=info.st_ctime_ns - 100)
        return info

    monkeypatch.setattr(Path, 'stat', path_stat)
    with path.open('rb') as handle:
        assert assets._stat_identity(path.stat()) != assets._stat_identity(assets.os.fstat(handle.fileno()))
    assert assets._verify(path, entry)['sha256'] == entry['sha256']
    assert not assets._VERIFIED


@pytest.mark.parametrize('field', ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'])
@pytest.mark.parametrize('phase', ['before', 'after'])
def test_windows_reopened_path_still_rejects_every_changed_identity_field(tmp_path, monkeypatch, field, phase):
    path, entry = pin(tmp_path)
    monkeypatch.setattr(assets, '_WINDOWS', True)
    with path.open('rb') as handle:
        info = assets.os.fstat(handle.fileno())
    values = {key: getattr(info, key) for key in
              ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
    values[field] += 1
    changed = assets._stat_identity(SimpleNamespace(**values))
    unchanged = assets._stat_identity(info)
    observation = (changed, changed, False) if phase == 'before' else (unchanged, changed, False)
    monkeypatch.setattr(assets, '_path_observation', lambda *_: observation)
    with pytest.raises(ValueError, match='changed during verification'):
        assets._verify(path, entry)
    assert not assets._VERIFIED


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
    original = Path.open
    modes = []

    class ChangedRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self):
            self.handle.read()
            return b'{"value": 2}'

    def changed_open(current, mode='r', *args, **kwargs):
        handle = original(current, mode, *args, **kwargs)
        if current == path:
            modes.append(mode)
            if len(modes) == 2:
                return ChangedRead(handle)
        return handle

    monkeypatch.setattr(Path, 'open', changed_open)
    with pytest.raises(ValueError, match='changed during verification'):
        assets._verify(path, entry)
    assert modes == ['rb', 'rb']
