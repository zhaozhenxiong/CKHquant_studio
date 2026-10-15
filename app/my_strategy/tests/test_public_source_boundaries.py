"""GitHub automatic source archives must not expose local business/audit state."""
from pathlib import Path, PurePosixPath
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_gitignore_covers_kronos_config_tokenizer_and_weights():
    if not (ROOT / '.git').exists():
        pytest.skip('Git ignore boundary check requires an actual source checkout')
    paths = ['app/my_strategy/kronos_model/config.json',
             'app/my_strategy/kronos_model/model.safetensors',
             'app/my_strategy/kronos_model/tokenizer/config.json',
             'app/my_strategy/kronos_model/tokenizer/model.safetensors',
             'app/kronos_model/config.json']
    # Git's stdin protocol is byte-oriented. Text-mode CRLF translation on
    # Windows would append a literal carriage return to every tested filename.
    result = subprocess.run(['git', 'check-ignore', '--no-index', '-z', '--stdin'], cwd=ROOT,
                            input=('\0'.join(paths)+'\0').encode('utf-8'),
                            capture_output=True, timeout=15, check=True)
    assert set(filter(None, result.stdout.decode('utf-8').split('\0'))) == set(paths)


def test_frozen_vendor_files_disable_checkout_newline_conversion():
    if not (ROOT / '.git').exists():
        pytest.skip('Git attributes check requires an actual source checkout')
    import json

    vendor = ROOT / 'app/vendor/kronos'
    manifest = json.loads((vendor / 'source_manifest.json').read_text(encoding='utf-8'))
    paths = ['app/vendor/kronos/' + item['path'] for item in manifest['files']]
    result = subprocess.run(['git', 'check-attr', '-z', '--stdin', 'text'], cwd=ROOT,
                            input=('\0'.join(paths)+'\0').encode('utf-8'),
                            capture_output=True, timeout=15, check=True)
    fields = result.stdout.decode('utf-8').split('\0')[:-1]
    assert list(zip(fields[::3], fields[1::3], fields[2::3])) == [
        (path, 'text', 'unset') for path in paths
    ]


def test_git_source_index_excludes_private_business_and_session_state():
    if not (ROOT / ".git").exists():
        pytest.skip("Git source boundary check requires an actual source checkout")
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                            text=True, timeout=15, check=True)
    private_prefixes = ("app/my_strategy/data/", "app/my_strategy/artifacts/",
                        "app/my_strategy/knowledge_base_system/",
                        "app/my_strategy/knowledge_base/",
                        "app/my_strategy/knowledge_base/05_AI_Sessions/")
    invalid = []
    for name in filter(None, result.stdout.split("\0")):
        path = PurePosixPath(name)
        if (name.startswith(private_prefixes) or name == "app/AI_CHANGELOG.md"
                or "kronos_model" in path.parts
                or path.name == ".env" or (path.name.startswith(".env.") and not path.name.endswith(".example"))
                or path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".pt", ".pth", ".joblib", ".safetensors", ".ckpt", ".pkl", ".npz"}):
            invalid.append(name)
    assert invalid == [], f"Private files would enter GitHub automatic source archives: {invalid}"
