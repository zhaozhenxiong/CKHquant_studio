"""One complete engine authentication feeds both per-call producer fields."""
import hashlib
import json
from pathlib import Path

import pytest

from my_strategy.adapters import czsc_adapter
from my_strategy.services import czsc_analysis, incremental_source as source
from my_strategy.services.czsc_native_evidence import NATIVE_SCHEMA_HASH


def original_identity():
    """Original producer construction, executed against the same actual files."""
    names = ['incremental_source.py', 'incremental_rolling.py', 'czsc_incremental_state.py', 'czsc_native_evidence.py',
        'czsc_market_features.py', 'czsc_market_profiles.py', 'czsc_price_volume_features.py', 'czsc_price_volume_events.py']
    root = Path(source.__file__).parent
    return {'code': {**{name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        'czsc_analysis.native_generation': czsc_analysis.native_generation_identity()},
        'input_contract': source.numerical_input_contract(), 'engine': czsc_adapter.engine_contract(),
        'schema': NATIVE_SCHEMA_HASH, 'config': czsc_analysis.strategy_config(), 'pandas': source.pd.__version__}


def test_producer_json_equals_original_construction_and_authenticates_engine_each_call(monkeypatch):
    expected = original_identity()
    calls = []
    authenticate = czsc_analysis.engine_contract
    def observed():
        result = authenticate()
        calls.append(result)
        return result
    monkeypatch.setattr(czsc_analysis, 'engine_contract', observed)
    # Keep the transitive function-source identity independent of this test's
    # spy, so the comparison isolates runtime engine construction.
    original_native_identity = expected['code']['czsc_analysis.native_generation']
    monkeypatch.setattr(czsc_analysis, 'native_generation_identity', lambda: original_native_identity)
    for _ in range(2):
        result = source.code_identity()
        assert result == expected
        assert result['engine'] == result['config']['engine_contract'] == calls[-1]
    assert len(calls) == 2


def test_changed_or_damaged_actual_engine_is_rechecked_without_a_cross_call_cache(tmp_path, monkeypatch):
    resources = tmp_path/'runtime'
    engine = resources/'vendor/czsc'
    engine.mkdir(parents=True)
    code = engine/'entry.py'
    code.write_text('verified engine fixture = 1\n')
    checksum = hashlib.sha256(code.read_bytes()).hexdigest()
    manifest = resources/'czsc-source-manifest.json'
    manifest.write_text(json.dumps({'package': 'czsc', 'version': 'fixture-1',
        'files': [{'path': 'entry.py', 'sha256': checksum}]}))
    monkeypatch.setattr(czsc_adapter, 'runtime_resource_root', lambda: resources)
    first = source.code_identity()
    assert first['engine'] == first['config']['engine_contract']
    code.write_text('damaged engine fixture = 2\n')
    with pytest.raises(RuntimeError, match='源码校验失败'):
        source.code_identity()
    with pytest.raises(RuntimeError, match='源码校验失败'):
        original_identity()
    checksum = hashlib.sha256(code.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'package': 'czsc', 'version': 'fixture-2',
        'files': [{'path': 'entry.py', 'sha256': checksum}]}))
    changed = source.code_identity()
    assert changed == original_identity()
    assert changed['engine'] != first['engine']
    assert changed['engine'] == changed['config']['engine_contract']
