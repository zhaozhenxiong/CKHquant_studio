import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_strategy.scripts.release_scratch_cleanup import plan, execute, validate_target


@pytest.fixture
def scratch(tmp_path):
    root = tmp_path / '.release-work'
    for version in ('0.1.7', '0.1.8', '1.0.0'):
        (root / ('github-' + version) / 'install-env').mkdir(parents=True)
    return root


@pytest.mark.parametrize('target', ['../data', '/tmp', 'github-1.0.0/install-env',
                                  'github-0.1.8/install-env', 'github-0.1.7/public',
                                  'github-0.1.7/git'])
def test_only_obsolete_scratch_is_eligible(scratch, target):
    with pytest.raises(ValueError): validate_target(scratch, target)


def test_hardlinks_and_symlink_targets_are_preserved(scratch, tmp_path):
    directory = scratch / 'github-0.1.7/install-env'
    outside = tmp_path / 'protected'
    outside.write_bytes(b'never-delete')
    os.link(outside, directory / 'shared')
    (directory / 'external-link').symlink_to(outside)
    manifest = plan(scratch, ['github-0.1.7/install-env'], check_use=lambda _: [])
    assert manifest['summary']['all_hardlinks_selected_bytes'] == 0
    result = execute(manifest, check_use=lambda _: [])
    assert result['removed'] == ['github-0.1.7/install-env']
    assert outside.read_bytes() == b'never-delete'
    assert not directory.exists()


def test_sha_or_metadata_change_refuses_before_removing_bytes(scratch):
    file = scratch / 'github-0.1.7/install-env/test.py'
    file.write_text('before')
    manifest = plan(scratch, ['github-0.1.7/install-env'], check_use=lambda _: [])
    file.write_text('after!')
    with pytest.raises(RuntimeError, match='changed'):
        execute(manifest, check_use=lambda _: [])
    assert file.read_text() == 'after!'


def test_live_process_and_git_history_block_retirement(scratch):
    directory = scratch / 'github-0.1.7/install-env'
    with pytest.raises(RuntimeError, match='Active'):
        plan(scratch, ['github-0.1.7/install-env'], check_use=lambda _: ['used'])
    (directory / '.git').mkdir()
    with pytest.raises(ValueError, match='Git history'):
        plan(scratch, ['github-0.1.7/install-env'], check_use=lambda _: [])


def test_symlink_parent_cannot_escape_and_overlaps_fail(scratch, tmp_path):
    directory = scratch / 'github-0.1.7/install-env'
    (directory / 'nested').mkdir()
    with pytest.raises(ValueError, match='Overlapping'):
        plan(scratch, ['github-0.1.7/install-env', 'github-0.1.7/install-env/nested'], check_use=lambda _: [])
    (directory / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        validate_target(scratch, 'github-0.1.7/install-env/link')


def test_new_live_use_after_rename_restores_tree_and_records_recovery(scratch):
    directory = scratch / 'github-0.1.7/install-env'
    (directory/'keep').write_text('needed')
    manifest = plan(scratch, ['github-0.1.7/install-env'], check_use=lambda _: [])
    def check(paths):
        return ['new-owner'] if any(p.name.startswith('.retired-') for p in paths) else []
    with pytest.raises(RuntimeError, match='became active'):
        execute(manifest, check_use=check)
    assert (directory/'keep').read_text() == 'needed'
    journal = next(scratch.glob('.scratch-retirement-*.jsonl'))
    assert [json.loads(line)['stage'] for line in journal.read_text().splitlines()] == [
        'prepared', 'retired', 'restored', 'failed']


def test_missing_windows_block_counts_are_unknown_and_do_not_change_cleanup_guards(scratch, monkeypatch):
    directory = scratch / 'github-0.1.7/install-env'
    (directory / 'fixture').write_bytes(b'disposable')
    original = Path.lstat
    def windows_lstat(path):
        value = original(path)
        return SimpleNamespace(**{key: getattr(value, key) for key in (
            'st_dev', 'st_ino', 'st_size', 'st_nlink', 'st_mode', 'st_mtime_ns', 'st_ctime_ns')})
    monkeypatch.setattr(Path, 'lstat', windows_lstat)
    manifest = plan(scratch, ['github-0.1.7/install-env'], check_use=lambda _: [])
    assert manifest['summary']['allocated_bytes'] is None
    assert manifest['summary']['all_hardlinks_selected_bytes'] is None
    assert manifest['summary']['allocation_metrics_available'] is False
    assert manifest['summary']['allocation_unknown_inodes'] == 1
    with pytest.raises(RuntimeError, match='Active'):
        execute(manifest, check_use=lambda _: ['new owner'])
    assert directory.exists()
    result = execute(manifest, check_use=lambda _: [])
    assert result['removed'] == ['github-0.1.7/install-env'] and not directory.exists()


def test_missing_posix_active_use_verifier_still_refuses_cleanup(monkeypatch):
    from my_strategy.scripts import release_scratch_cleanup as cleanup
    monkeypatch.setattr(cleanup, 'os', SimpleNamespace())
    with pytest.raises(RuntimeError, match='verifier unavailable'):
        cleanup.active_uses([])


@pytest.mark.parametrize('packed_refs', [False, True])
def test_bare_git_history_inside_temporary_tree_is_never_selected(scratch, packed_refs):
    directory = scratch / 'github-0.1.7/install-env/saved-history.git'
    (directory/'objects').mkdir(parents=True)
    (directory/'HEAD').write_text('ref: refs/heads/main\n')
    if packed_refs:
        (directory/'packed-refs').write_text('# pack-refs\n')
    else:
        (directory/'refs').mkdir()
    with pytest.raises(ValueError, match='Bare Git history'):
        plan(scratch, ['github-0.1.7/install-env'], check_use=lambda _: [])
    assert (directory/'HEAD').read_text() == 'ref: refs/heads/main\n'
