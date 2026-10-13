"""Release native assets must match their filename, bytes and frozen source."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.utils import parse_wheel_filename

from build_release import release_archive, wheel_index
from my_strategy.scripts.public_provenance import public_provenance


@pytest.fixture
def manifest():
    return {"package": "czsc", "version": "1.0.1",
            "files": [{"path": "engine.py", "sha256": hashlib.sha256(b"frozen algorithm").hexdigest()}]}


def indexed_wheel(directory: Path, manifest: dict, name: str = "czsc-1.0.1-cp312-cp312-macosx_14_0_arm64.whl",
                  content: bytes = b"fixture native wheel bytes") -> tuple[dict, dict]:
    directory.mkdir(parents=True)
    (directory / name).write_bytes(content)
    entry = {"filename": name, "sha256": hashlib.sha256(content).hexdigest(),
             "tags": sorted(str(tag) for tag in parse_wheel_filename(name)[3]),
             "build": {"cargo_locked": True, "source_algorithm_patches": False}}
    source_hash = hashlib.sha256("\n".join(f"{item['path']} {item['sha256']}" for item in manifest["files"]).encode()).hexdigest()
    index = {"schema_version": 1, "package": "czsc", "version": manifest["version"],
             "source_tree_sha256": source_hash, "wheels": [entry]}
    save_index(directory, index)
    return index, entry


def save_index(directory: Path, index: dict) -> None:
    (directory / "wheel-index.json").write_text(json.dumps(index), encoding="utf-8")


def test_merge_platform_assets_binds_names_tags_and_bytes(tmp_path, manifest):
    first, second = tmp_path / "mac", tmp_path / "linux"
    mac_index, mac = indexed_wheel(first, manifest)
    linux_index, linux = indexed_wheel(second, manifest, "czsc-1.0.1-cp312-cp312-manylinux_2_17_x86_64.whl")
    # A combined index can refer to an asset downloaded into another directory.
    mac_index["wheels"].append(deepcopy(linux))
    save_index(first, mac_index)
    merged, assets = wheel_index([first, second], manifest)
    assert set(assets) == {mac["filename"], linux["filename"]}
    assert len(merged["wheels"]) == 2
    for item in merged["wheels"]:
        assert hashlib.sha256(assets[item["filename"]].read_bytes()).hexdigest() == item["sha256"]
        assert set(item["tags"]) == {str(tag) for tag in parse_wheel_filename(item["filename"])[3]}


@pytest.mark.parametrize("name", ["other-1.0.1-cp312-cp312-win_amd64.whl", "czsc-9.0.0-cp312-cp312-win_amd64.whl"])
def test_reject_other_package_or_version_even_when_index_and_sha_are_valid(tmp_path, manifest, name):
    directory = tmp_path / "native"
    indexed_wheel(directory, manifest, name)
    with pytest.raises(ValueError, match="package/version mismatch"):
        wheel_index([directory], manifest)


def test_reject_index_tags_that_misrepresent_filename(tmp_path, manifest):
    directory = tmp_path / "native"
    index, entry = indexed_wheel(directory, manifest)
    entry["tags"] = ["cp312-cp312-win_amd64"]
    save_index(directory, index)
    with pytest.raises(ValueError, match="tags mismatch"):
        wheel_index([directory], manifest)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("schema_version", None),
                                         ("schema_version", True), ("source_tree_sha256", "another-source")])
def test_reject_unsupported_index_or_frozen_source_conflict(tmp_path, manifest, field, value):
    directory = tmp_path / "native"
    index, _entry = indexed_wheel(directory, manifest)
    index[field] = value
    save_index(directory, index)
    with pytest.raises(ValueError, match="source mismatch"):
        wheel_index([directory], manifest)


@pytest.mark.parametrize("prefix", ["../", "..\\", "directory/", "directory\\"])
def test_reject_both_platform_path_separators_before_resolving_assets(tmp_path, manifest, prefix):
    directory = tmp_path / "native"
    index, entry = indexed_wheel(directory, manifest)
    entry["filename"] = prefix + entry["filename"]
    save_index(directory, index)
    with pytest.raises(ValueError, match="Unsafe wheel filename"):
        wheel_index([directory], manifest)


def test_reject_changed_wheel_bytes_and_conflicting_same_name_builds(tmp_path, manifest):
    first, second = tmp_path / "one", tmp_path / "two"
    _index, entry = indexed_wheel(first, manifest)
    indexed_wheel(second, manifest, content=b"different native build")
    with pytest.raises(ValueError, match="Conflicting wheel builds"):
        wheel_index([first, second], manifest)
    (first / entry["filename"]).write_bytes(b"changed after index creation")
    with pytest.raises(ValueError, match="checksum mismatch"):
        wheel_index([first], manifest)


def test_existing_frozen_native_wheel_validates_without_rewriting_manifest():
    app = Path(__file__).resolve().parents[2]
    source_path = app / "czsc-source-manifest.json"
    original = source_path.read_bytes()
    source = json.loads(original)
    directory = app / "vendor/wheels"
    if not (directory / source["built_wheel_name"]).is_file():
        pytest.skip("native asset not present on this test host")
    merged, assets = wheel_index([directory], source)
    name = source["built_wheel_name"]
    assert name in assets
    # A locked rebuild has its own physical SHA in wheel-index.json; the
    # attachment manifest retains the original historical wheel SHA.
    indexed = next(entry for entry in merged["wheels"] if entry["filename"] == name)
    assert hashlib.sha256(assets[name].read_bytes()).hexdigest() == indexed["sha256"]
    assert merged["version"] == source["version"]
    assert source_path.read_bytes() == original


def test_release_archive_preserves_unix_script_modes_bytes_and_package_layout(tmp_path):
    package = tmp_path / "KHQuant-0.1.0"
    files = {"khquant-native.sh": b"#!/bin/sh\nprintf 'start\\n'\n",
             "macos/install.command": b"#!/bin/sh\nexec python3 install.py\n",
             "README.md": b"fixture documentation\r\n",
             "app/model.pt": b"\x00\xff\n"}
    for relative, content in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (package / "app/empty-state").mkdir()
    archive = release_archive(package)
    assert archive == tmp_path / "KHQuant-0.1.0.zip"
    with ZipFile(archive) as zipped:
        assert "KHQuant-0.1.0/" in zipped.namelist()
        assert "KHQuant-0.1.0/app/empty-state/" in zipped.namelist()
        assert all(name.startswith("KHQuant-0.1.0/") for name in zipped.namelist())
        for relative, content in files.items():
            name = f"KHQuant-0.1.0/{relative}"
            info = zipped.getinfo(name)
            expected_mode = 0o100755 if Path(relative).suffix in {".sh", ".command"} else 0o100644
            assert info.create_system == 3
            assert info.external_attr >> 16 == expected_mode
            assert zipped.read(name) == content
            assert (package / relative).read_bytes() == content


def test_public_build_metadata_redacts_only_host_paths_and_keeps_native_identity():
    original = {"files": [{"path": "engine.py", "sha256": "immutable-source"}], "built_wheel_sha256": "immutable-wheel",
        "source_tree_sha256": "immutable-tree", "build": {"cargo_locked": True, "source_algorithm_patches": False,
            "cargo_target_dir": "C:\\Users\\Private\\target", "rustflags": "-C opt-level=3 -Lnative=C:/Users/Private/lib -C dlltool=C:/Users/Private/bin/dlltool.exe",
            "release_profile_overrides": {"CARGO_PROFILE_RELEASE_LTO": "false"}},
        "wheels": [{"filename": "native.whl", "sha256": "immutable-native", "tags": ["cp312-cp312-win_amd64"],
            "build": {"cargo_target_dir": "/Users/Private/project/target", "rustflags": "-C debuginfo=0"}}]}
    preserved = deepcopy(original)
    result = public_provenance(original)
    assert original == preserved
    assert result["files"] == original["files"] and result["built_wheel_sha256"] == original["built_wheel_sha256"]
    assert result["source_tree_sha256"] == original["source_tree_sha256"]
    assert result["wheels"][0]["sha256"] == original["wheels"][0]["sha256"]
    assert result["build"]["cargo_locked"] and result["build"]["source_algorithm_patches"] is False
    assert result["build"]["release_profile_overrides"] == original["build"]["release_profile_overrides"]
    assert result["build"]["rustflags"] == "-C opt-level=3 -Lnative=<build-local-native-library-dir> -C dlltool=<build-local-dlltool>"
    assert result["wheels"][0]["build"]["rustflags"] == "-C debuginfo=0"
    assert "Users" not in json.dumps(result)
    assert public_provenance(result) == result


def test_release_assembly_is_relative_clean_and_retains_validation_document(tmp_path, monkeypatch, manifest):
    import build_release
    from my_strategy.scripts.package_project import verify_package
    repo, assets = tmp_path / "source", tmp_path / "native"
    app = repo / "app"
    app.mkdir(parents=True)
    for name, body in {"install.py": "# fixture installer", "RELEASE_VALIDATION.md": "# Current validation",
        "pyproject.toml": '[project]\nversion="0.1.4"\n', "runtime-manifest.json": '{"platform_verification": {}}'}.items():
        (repo / name).write_text(body, encoding="utf-8")
    (app / "AGENTS.md").write_text("fixture source checkout", encoding="utf-8")
    paths = app / "my_strategy/core/paths.py"
    paths.parent.mkdir(parents=True)
    paths.write_text("# fixture application path boundary", encoding="utf-8")
    native = app / "vendor/czsc"
    native.mkdir(parents=True)
    (native / "engine.py").write_bytes(b"frozen algorithm")
    manifest = deepcopy(manifest)
    manifest["build"] = {"cargo_target_dir": "/Users/Private/target", "rustflags": "-Lnative=C:/Users/Private/lib"}
    manifest_path = app / "czsc-source-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original = manifest_path.read_bytes()
    index, entry = indexed_wheel(assets, manifest)
    entry["build"].update(cargo_target_dir="C:\\Users\\Private\\target", rustflags="-C dlltool=C:/Users/Private/dlltool.exe")
    save_index(assets, index)
    old_assets = app / "vendor/wheels"
    old_assets.mkdir()
    (old_assets / "unselected-native.whl").write_bytes(b"not selected for release")
    for name in ("data/raw/private.db", "artifacts/runs/private/models/model.pt", "knowledge_base_system/evidence/private.json"):
        private = app / "my_strategy" / name
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_bytes(b"private fixture must be excluded")
    monkeypatch.setattr(build_release, "ROOT", repo)
    monkeypatch.setattr(build_release, "APP", app)
    release = build_release.build_release(tmp_path / "public", [assets])
    package = Path(release["package"])
    assert verify_package(package) == []
    assert (package / "RELEASE_VALIDATION.md").read_text() == "# Current validation"
    assert json.loads((package / "image-manifest.json").read_text())["source_project_root"] == "app"
    assert manifest_path.read_bytes() == original
    assert (package / "app/vendor/czsc/engine.py").read_bytes() == b"frozen algorithm"
    public_index = json.loads((package / "app/vendor/wheels/wheel-index.json").read_text())
    assert public_index["source_tree_sha256"] == index["source_tree_sha256"]
    assert public_index["wheels"][0]["sha256"] == entry["sha256"]
    assert (package / "app/vendor/wheels" / entry["filename"]).read_bytes() == (assets / entry["filename"]).read_bytes()
    assert not (package / "app/vendor/wheels/unselected-native.whl").exists()
    assert not list(package.rglob("*.db")) and not list(package.rglob("*.pt"))
    assert not (package / "app/my_strategy/knowledge_base_system").exists()
    for name in ("app/czsc-source-manifest.json", "app/vendor/wheels/wheel-index.json", "image-manifest.json"):
        assert "Users" not in (package / name).read_text()
    assert hashlib.sha256(Path(release["archive"]).read_bytes()).hexdigest() == release["archive_sha256"]
