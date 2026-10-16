"""Release jobs must verify and publish the same application version and bytes."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib
from zipfile import ZipFile

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / '.github/workflows'
pytestmark = pytest.mark.skipif(not WORKFLOWS.is_dir(), reason='Workflow contracts require the Git source checkout')


def workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding='utf-8'))


def steps(name, job):
    return workflow(name)['jobs'][job]['steps']


def application_version():
    return tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']


def test_application_wheel_and_isolated_version_assertions_match_project():
    expected = application_version()
    for name in ('windows-release-check.yml', 'portable-release.yml'):
        text = (WORKFLOWS / name).read_text(encoding='utf-8')
        versions = re.findall(r'khquant-(\d+\.\d+\.\d+)-py3-none-any\.whl', text)
        assert versions and set(versions) == {expected}
        isolated_versions = re.findall(r"importlib\.metadata\.version\('khquant'\) == '([^']+)'", text)
        assert isolated_versions and set(isolated_versions) == {expected}


def test_portable_final_wheel_is_built_from_combined_source_and_installed_before_smoke():
    actual = steps('portable-release.yml', 'archive')
    commands = [item.get('run', '') for item in actual]
    version = application_version()
    combined_source = f'dist/releases/{version}/KHQuant-{version}'
    build = next(i for i, command in enumerate(commands) if f'pip wheel {combined_source} ' in command)
    install = next(i for i, command in enumerate(commands) if 'pip install --no-deps --force-reinstall' in command)
    identity = next(i for i, command in enumerate(commands) if "'-I', '-c', probe" in command)
    smoke = next(i for i, command in enumerate(commands) if 'verify_installation.py --device cpu' in command)
    assert build < install < identity < smoke
    assert combined_source in commands[smoke]
    assert any('archive_info' in command and 'sha256' in command for command in commands)
    upload = next(item['with']['path'] for item in actual if item.get('uses') == 'actions/upload-artifact@v4')
    assert f'dist/releases/{version}/*.zip' in upload
    assert f'dist/releases/{version}/*.whl' in upload
    assert f'dist/releases/{version}/SHA256SUMS.txt' in upload


def test_checksum_extension_preserves_source_and_binds_all_uploaded_bytes(tmp_path, monkeypatch):
    actual = steps('portable-release.yml', 'archive')
    command = next(item['run'] for item in actual if item.get('name') == 'Extend checksums with verified application and installation assets')
    version = application_version()
    root = tmp_path / f'dist/releases/{version}'
    reports = root / 'platform-verification/installation-windows-latest'
    reports.mkdir(parents=True)
    archive = root / f'KHQuant-{version}.zip'
    archive.write_bytes(b'source archive bytes')
    assets = [(root / f'khquant-{version}-py3-none-any.whl', b'application wheel bytes'),
              (root / 'installation-combined-identity.json', b'{"version":"current"}'),
              (reports / 'installation-cpu.json', b'{"status":"passed"}')]
    for path, content in assets:
        path.write_bytes(content)
    original = f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n'
    checksum = root / 'SHA256SUMS.txt'
    checksum.write_text(original, encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    exec(compile(command, 'workflow-checksum-extension', 'exec'), {})
    result = checksum.read_text(encoding='utf-8')
    assert result.startswith(original)
    records = dict(line.split('  ', 1)[::-1] for line in result.splitlines())
    assert len(records) == len(assets) + 1
    for path, content in [(archive, archive.read_bytes()), *assets]:
        assert records[path.relative_to(root).as_posix()] == hashlib.sha256(content).hexdigest()


def test_native_upload_uses_fresh_build_directory_and_installer_selects_it():
    actual = steps('portable-release.yml', 'wheels')
    build = next(item['run'] for item in actual if 'build_czsc_runtime.py' in item.get('run', ''))
    install = next(item['run'] for item in actual if 'python install.py' in item.get('run', ''))
    assert '--out .scratch/native-wheel ' in build
    assert '--wheel-dir .scratch/native-wheel ' in install
    assert next(i for i, item in enumerate(actual) if item.get('name') == 'Align checkout native bytes with the verified fresh index') < next(
        i for i, item in enumerate(actual) if 'python install.py' in item.get('run', ''))
    upload = next(item['with'] for item in actual if item.get('with', {}).get('name') == 'wheels-${{ matrix.os }}')
    assert upload['path'] == '.scratch/native-wheel/'
    assert upload['include-hidden-files'] is True
    for name in ('windows-release-check.yml', 'portable-release.yml'):
        for job in workflow(name)['jobs'].values():
            for item in job['steps']:
                if item.get('uses') == 'actions/upload-artifact@v4' and '.scratch/' in item['with']['path']:
                    assert item['with']['include-hidden-files'] is True


@pytest.mark.parametrize('corrupt', [False, True])
def test_ci_native_alignment_copies_verified_new_bytes_without_touching_frozen_source(tmp_path, monkeypatch, corrupt):
    actual = steps('portable-release.yml', 'wheels')
    script = next(item['run'] for item in actual if item.get('name') == 'Align checkout native bytes with the verified fresh index')
    fresh = tmp_path / '.scratch/native-wheel'
    destination = tmp_path / 'app/vendor/wheels'
    fresh.mkdir(parents=True)
    destination.mkdir(parents=True)
    name = 'czsc-1.0.1-cp310-abi3-macosx_11_0_arm64.whl'
    new_bytes = b'verified current platform rebuild'
    old_bytes = b'previous physical native asset'
    (fresh / name).write_bytes(new_bytes)
    (destination / name).write_bytes(old_bytes)
    digest = hashlib.sha256(new_bytes).hexdigest()
    index = {'wheels': [{'filename': name, 'sha256': '0' * 64 if corrupt else digest,
                        'build': {'cargo_locked': True, 'source_algorithm_patches': False}}]}
    index_bytes = json.dumps(index).encode()
    (fresh / 'wheel-index.json').write_bytes(index_bytes)
    (destination / 'wheel-index.json').write_bytes(index_bytes)
    source_manifest = tmp_path / 'app/czsc-source-manifest.json'
    original_source = b'{"built_wheel_sha256":"original scientific identity", "files": []}\n'
    source_manifest.write_bytes(original_source)
    monkeypatch.chdir(tmp_path)
    if corrupt:
        with pytest.raises(AssertionError, match='Fresh native wheel checksum mismatch'):
            exec(compile(script, 'ci-fresh-native-alignment', 'exec'), {})
        assert (destination / name).read_bytes() == old_bytes
    else:
        exec(compile(script, 'ci-fresh-native-alignment', 'exec'), {})
        assert (destination / name).read_bytes() == new_bytes
        active = json.loads((destination / 'wheel-index.json').read_text())['wheels'][0]
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == active['sha256']
    assert source_manifest.read_bytes() == original_source
    assert (destination / 'wheel-index.json').read_bytes() == index_bytes


def test_windows_can_verify_exact_portable_asset_with_authenticated_artifact_download():
    actual = steps('windows-release-check.yml', 'windows')
    downloaded = next(item for item in actual if item.get('uses') == 'actions/download-artifact@v4')
    assert downloaded['with']['name'] == 'KHQuant-source-release'
    assert downloaded['with']['run-id'] == '${{ inputs.portable_run_id }}'
    assert downloaded['with']['github-token'] == '${{ github.token }}'
    assert workflow('windows-release-check.yml')['permissions']['actions'] == 'read'
    installed = next(item['run'] for item in actual if 'application_wheel_sha256' in item.get('run', ''))
    assert "direct['archive_info']['hashes']['sha256'] == wheel_sha" in installed
    assert "native_direct['archive_info']['hashes']['sha256'] == expected_native['sha256']" in installed


@pytest.mark.parametrize('failure', [None, 'wheel_sha', 'native_index'])
def test_selected_portable_windows_assets_bind_app_sha_zip_and_matching_native_before_install(tmp_path, monkeypatch, failure):
    actual = steps('windows-release-check.yml', 'windows')
    script = next(item['run'] for item in actual if item.get('name') == 'Prepare selected portable application and matching native runtime')
    assets = tmp_path / '.scratch/portable-release'
    assets.mkdir(parents=True)
    version = application_version()
    wheel = assets / f'khquant-{version}-py3-none-any.whl'
    index = {'wheels': [{'filename': 'czsc-1.0.1-cp310-abi3-win_amd64.whl', 'sha256': 'same-native-bytes'}]}
    wheel_index = {'wheels': []} if failure == 'native_index' else index
    with ZipFile(wheel, 'w') as zipped:
        zipped.writestr('my_strategy/_runtime/vendor/wheels/wheel-index.json', json.dumps(wheel_index))
    archive = assets / f'KHQuant-{version}.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr(f'KHQuant-{version}/release.json', json.dumps({'git_revision': 'exact-source-commit'}))
        zipped.writestr(f'KHQuant-{version}/app/vendor/wheels/wheel-index.json', json.dumps(index))
        zipped.writestr(f'KHQuant-{version}/install.py', '# fixture installer; never executed')
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (assets / 'SHA256SUMS.txt').write_text(
        f'{digest}  {wheel.name}\n{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n', encoding='utf-8')
    monkeypatch.setenv('APPLICATION_WHEEL_SHA256', '0' * 64 if failure == 'wheel_sha' else digest)
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'exact-source-commit\n')
    installs = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: installs.append(command))
    monkeypatch.chdir(tmp_path)
    if failure:
        with pytest.raises(AssertionError, match='SHA-256 mismatch' if failure == 'wheel_sha' else 'indexes differ'):
            exec(compile(script, 'windows-portable-asset-stage', 'exec'), {})
        assert not installs
        return
    exec(compile(script, 'windows-portable-asset-stage', 'exec'), {})
    assert (tmp_path / '.scratch/release-wheels' / wheel.name).read_bytes() == wheel.read_bytes()
    assert len(installs) == 1
    selected = installs[0]
    assert Path(selected[1]).parts[-3:] == ('extracted', f'KHQuant-{version}', 'install.py')
    assert Path(selected[selected.index('--wheel-dir') + 1]) == assets / f'extracted/KHQuant-{version}/app/vendor/wheels'
    assert Path(selected[selected.index('--venv-dir') + 1]) == tmp_path / '.venv-release'


def test_embedded_python_and_isolated_probes_compile():
    for name in ('windows-release-check.yml', 'portable-release.yml'):
        for job in workflow(name)['jobs'].values():
            for step in job['steps']:
                if step.get('shell') != 'python':
                    continue
                script = step['run']
                compile(script, name, 'exec')
                for node in ast.walk(ast.parse(script)):
                    if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                            and any(isinstance(target, ast.Name) and target.id == 'probe' for target in node.targets)):
                        compile(node.value.value, name + '-isolated-probe', 'exec')


@pytest.mark.parametrize('portable', [False, True])
def test_windows_final_probe_uses_selected_source_and_exact_installed_bytes(tmp_path, monkeypatch, portable):
    script = next(item['run'] for item in steps('windows-release-check.yml', 'windows')
                  if item.get('name') == 'Verify the selected installed release with its matching probe')
    wheel = tmp_path / '.scratch/release-wheels/khquant-1.0.1-py3-none-any.whl'
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b'verified final wheel')
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    source = tmp_path / '.scratch/portable-release/extracted/KHQuant-1.0.1' if portable else tmp_path
    verifier = source / 'app/verify_installation.py'
    verifier.parent.mkdir(parents=True)
    verifier.write_text('# frozen verifier', encoding='utf-8')
    monkeypatch.setenv('PORTABLE_RUN_ID', '123' if portable else '')
    monkeypatch.setenv('APPLICATION_WHEEL_SHA256', digest if portable else '')
    commands = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: commands.append((command, kwargs)))
    monkeypatch.chdir(tmp_path)
    exec(compile(script, 'windows-selected-probe', 'exec'), {})
    assert len(commands) == 1
    command, options = commands[0]
    assert command[1] == '-I' and Path(command[2]).resolve() == verifier
    assert command[command.index('--expected-wheel-sha256') + 1] == digest
    assert command[command.index('--expected-version') + 1] == application_version()
    assert options['check'] is True
