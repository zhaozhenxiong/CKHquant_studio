"""Optional POSIX metrics never prevent portable audits or invent capacity."""
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_strategy.scripts.benchmark_research_store import allocation_metrics
from my_strategy.storage.native_payload_codec import encode_native_payload


@pytest.mark.parametrize('blocks,expected,unknown', [
    ([2, 3], 2560, 0), ([2, None], None, 1), ([None], None, 1), ([], 0, 0),
])
def test_disk_allocation_reports_missing_windows_fields_as_unknown(blocks, expected, unknown):
    paths = [SimpleNamespace(stat=lambda value=value: SimpleNamespace(
        st_size=99999, **({'st_blocks': value} if value is not None else {}))) for value in blocks]
    result = allocation_metrics(paths)
    assert result == {'allocated_object_bytes': expected, 'allocation_metrics_available': unknown == 0,
                      'allocation_unknown_objects': unknown}


def test_codec_benchmark_import_and_lossless_export_without_posix_resource(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / 'scripts/benchmark_native_payload_codec.py'
    original = builtins.__import__
    def windows_import(name, *args, **kwargs):
        if name == 'resource':
            raise ModuleNotFoundError("No module named 'resource'")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', windows_import)
    spec = importlib.util.spec_from_file_location('portable_codec_benchmark_fixture', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fixture = tmp_path / 'original.json.gz'
    content = encode_native_payload({'evaluations': [{'day': 1}, {'day': 2}]})
    fixture.write_bytes(content)
    result = module.run([fixture], tmp_path / 'copies')
    assert result['process_maxrss_platform_units'] is None
    assert result['process_maxrss_available'] is False
    assert result['process_maxrss_unavailable_reason'] == 'resource_module_unavailable'
    assert result['samples'][0]['equivalent'] and result['samples'][0]['original_unchanged']
    assert fixture.read_bytes() == content
    assert result['production_migration_performed'] is result['default_enabled'] is False


def test_codec_benchmark_retains_actual_resource_counter_when_supported(monkeypatch):
    from my_strategy.scripts import benchmark_native_payload_codec as codec
    seen = []
    def usage(kind):
        seen.append(kind); return SimpleNamespace(ru_maxrss=12345)
    monkeypatch.setattr(codec, 'resource', SimpleNamespace(RUSAGE_SELF=7, getrusage=usage))
    assert codec.resource_usage() == {'process_maxrss_platform_units': 12345,
        'process_maxrss_available': True, 'process_maxrss_unavailable_reason': None}
    assert seen == [7]
