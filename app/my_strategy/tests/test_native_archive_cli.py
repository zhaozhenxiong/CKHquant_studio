import json

import pytest

from my_strategy.scripts.archive_native_payload import create,restore
from my_strategy.storage.native_payload_codec import canonical,decode_native_payload,encode_native_payload


def test_archive_copy_manifest_and_independent_restore_preserve_original(tmp_path):
    source=tmp_path/'native.json.gz';payload={'evaluations':[{'day':n,'invalidated':list(range(n))} for n in range(50)]}
    original=encode_native_payload(payload);source.write_bytes(original)
    archive=tmp_path/'archive/native.delta.json.gz';result=create(source,archive)
    restored=restore(archive,result['manifest_path'],tmp_path/'restored/native.json.gz')
    assert source.read_bytes()==original and result['original_preserved']
    assert canonical(decode_native_payload((tmp_path/'restored/native.json.gz').read_bytes()))==canonical(payload)
    assert restored['logical_equivalent'] and not restored['original_modified'] and not restored['index_modified']


def test_archive_never_overwrites_existing_output_or_original(tmp_path):
    source=tmp_path/'original.gz';source.write_bytes(encode_native_payload({'evaluations':[{'x':1},{'x':2}]}))
    archive=tmp_path/'archive.gz';receipt=create(source,archive)
    with pytest.raises(FileExistsError):create(source,archive)
    with pytest.raises(ValueError,match='independent'):restore(archive,receipt['manifest_path'],source)
    output=tmp_path/'restored.gz';output.write_bytes(b'existing')
    with pytest.raises(FileExistsError):restore(archive,receipt['manifest_path'],output)
    assert output.read_bytes()==b'existing'


@pytest.mark.parametrize('part',['archive','logical','legacy'])
def test_changed_archive_or_manifest_refuses_restore(tmp_path,part):
    source=tmp_path/'original.gz';source.write_bytes(encode_native_payload({'evaluations':[{'x':1},{'x':2}]}))
    archive=tmp_path/'archive.gz';receipt=create(source,archive)
    manifest=tmp_path/'archive.gz.manifest.json'
    if part=='archive':archive.write_bytes(archive.read_bytes()+b'corrupt')
    else:
        record=json.loads(manifest.read_text());record['logical_sha256' if part=='logical' else 'restored_legacy_gzip_sha256']='0'*64
        manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='SHA'):restore(archive,manifest,tmp_path/'restore.gz')
    assert not (tmp_path/'restore.gz').exists()
