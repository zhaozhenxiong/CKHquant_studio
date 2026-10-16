from copy import deepcopy
import gzip
import json
import math

import pytest

from my_strategy.storage.native_payload_codec import FORMAT,canonical,encode_native_payload,decode_native_payload


def payload(n=300):
    return {'version':'fixture','config_hash':'c','schema_hash':'s','events':[{'id':'original','value':2}],
        'evaluations':[{'date':f'{i:04d}','structure':{'confirmed':i%3==0,'price':float(i),
            'invalidated_event_ids':['event'+str(k) for k in range(i//3)],'support_facts':[{'id':'a','value':1}]},
            'value':None if i%5==0 else i,'nested':{'optional':True} if i%7 else {}}
            for i in range(n)]}


def test_full_history_roundtrip_append_revision_calendar_boundary_and_restart(tmp_path):
    original=payload()
    encoded=encode_native_payload(original,codec=FORMAT)
    assert canonical(decode_native_payload(encoded))==canonical(original)
    path=tmp_path/'native.delta.json.gz';path.write_bytes(encoded)
    assert canonical(decode_native_payload(path.read_bytes()))==canonical(original)
    appended=payload(301)
    appended['evaluations'][-1]['date']='2026-11-02'  # month/week rollover is an ordinary exact value
    assert canonical(decode_native_payload(encode_native_payload(appended,codec=FORMAT)))==canonical(appended)
    revised=deepcopy(original);revised['evaluations'][12]['structure']['price']=.125
    assert canonical(decode_native_payload(encode_native_payload(revised,codec=FORMAT)))==canonical(revised)
    assert canonical(decode_native_payload(path.read_bytes()))==canonical(original)
    assert len(encoded)<len(encode_native_payload(original))


def test_scalar_types_negative_zero_and_original_envelope_like_keys_survive():
    original={'format':'user_format','static':{'unchanged':True},'streams':{},'evaluations':[
        {'x':1,'y':0.0},{'x':True,'y':-0.0},{'x':1.0,'y':None},{'x':'1','y':[]},{'x':None,'y':{}},
        {'x':False,'y':['a']},{'x':0,'y':[]} ]}
    result=decode_native_payload(encode_native_payload(original,codec=FORMAT,checkpoint_rows=2))
    assert canonical(result)==canonical(original)
    assert type(result['evaluations'][0]['x']) is int and type(result['evaluations'][1]['x']) is bool
    assert math.copysign(1,result['evaluations'][1]['y'])==-1


def test_reconstructed_rows_do_not_share_mutable_objects():
    original={'evaluations':[{'facts':[{'value':1}],'ids':[1]} for _ in range(3)]}
    result=decode_native_payload(encode_native_payload(original,codec=FORMAT))
    result['evaluations'][0]['facts'][0]['value']=2;result['evaluations'][0]['ids'].append(2)
    assert result['evaluations'][1]==original['evaluations'][1]


def test_legacy_gzip_bytes_are_read_without_schema_change():
    original=payload(3)
    blob=gzip.compress(json.dumps(original,ensure_ascii=False,indent=2).encode())
    assert decode_native_payload(blob)==original


@pytest.mark.parametrize('mutation',['hash','size','length','first','patch','interval'])
def test_corruption_or_invalid_operations_fail_closed(mutation):
    encoded=encode_native_payload(payload(3),codec=FORMAT)
    envelope=json.loads(gzip.decompress(encoded))
    if mutation=='hash':envelope['logical_sha256']='0'*64
    elif mutation=='size':envelope['logical_json_bytes']+=1
    elif mutation=='length':envelope['streams']['evaluations']['length']+=1
    elif mutation=='first':envelope['streams']['evaluations']['blocks'][0]['start']=1
    elif mutation=='patch':envelope['streams']['evaluations']['blocks'][0]['patches'][0]=['suffix',999,[]]
    elif mutation=='interval':envelope['checkpoint_rows']=0
    with pytest.raises(ValueError):decode_native_payload(gzip.compress(json.dumps(envelope).encode()))


def test_empty_streams_unicode_escaped_strings_and_nonfinite_rejection():
    original={'evaluations':[],'special':'中文\\n\"','none':None,'empty':{},'single':[{'id':1}]}
    assert decode_native_payload(encode_native_payload(original,codec=FORMAT))==original
    with pytest.raises(ValueError):encode_native_payload({'value':float('nan')},codec=FORMAT)
    with pytest.raises(ValueError):decode_native_payload(gzip.compress(b'{"x":NaN}'))
