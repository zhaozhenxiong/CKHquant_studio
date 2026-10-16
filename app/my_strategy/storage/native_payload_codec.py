"""Lossless native JSON record streams with independently decodable checkpoints.

This opt-in format never rewrites an existing object. Logical SHA binds the full
canonical legacy payload, not an approximation of event history. Normal gzip
JSON remains readable. A decoder returns independent mutable rows, preserving
legacy callers' mutation semantics as well as exact JSON scalar types.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json

FORMAT = 'native_lossless_delta_v1'


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _same(left,right):
    if type(left) is not type(right):return False
    if isinstance(left,float):return left.hex()==right.hex()
    if isinstance(left,dict):return left.keys()==right.keys() and all(_same(value,right[key]) for key,value in left.items())
    if isinstance(left,list):return len(left)==len(right) and all(_same(a,b) for a,b in zip(left,right))
    return left==right


def _patch(previous,current):
    if type(previous) is not type(current):return ['set',current]
    if isinstance(current,dict):
        removed=[key for key in previous if key not in current];changes=[]
        for key,value in current.items():
            change=['set',value] if key not in previous else _patch(previous[key],value)
            if change is not None:changes.append([key,change])
        return ['dict',removed,changes] if removed or changes else None
    if isinstance(current,list):
        common=0
        for left,right in zip(previous,current):
            if not _same(left,right):break
            common+=1
        if common==len(previous)==len(current):return None
        return ['suffix',common,current[common:]] if common else ['set',current]
    return None if _same(previous,current) else ['set',current]


def _apply(previous,patch):
    if patch is None:return deepcopy(previous)
    if not isinstance(patch,list) or not patch:raise ValueError('invalid native delta operation')
    operation=patch[0]
    if operation=='set' and len(patch)==2:return deepcopy(patch[1])
    if operation=='suffix' and len(patch)==3:
        if not isinstance(previous,list) or type(patch[1]) is not int or not 0<=patch[1]<=len(previous) or not isinstance(patch[2],list):
            raise ValueError('invalid native suffix operation')
        return deepcopy(previous[:patch[1]])+deepcopy(patch[2])
    if operation=='dict' and len(patch)==3 and isinstance(previous,dict):
        if not isinstance(patch[1],list) or not isinstance(patch[2],list):raise ValueError('invalid native dict operation')
        removed=set()
        for key in patch[1]:
            if not isinstance(key,str) or key not in previous or key in removed:raise ValueError('invalid native removed key')
            removed.add(key)
        changes={}
        for item in patch[2]:
            if not isinstance(item,list) or len(item)!=2 or not isinstance(item[0],str):raise ValueError('invalid native changed key')
            key,change=item
            if key in changes or key in removed:raise ValueError('duplicate native changed key')
            if key not in previous and not (isinstance(change,list) and change and change[0]=='set'):
                raise ValueError('native new key must carry a value')
            changes[key]=change
        # Copy each unchanged branch once; copying the whole previous tree and
        # then replacing changed subtrees multiplies full-history decode cost.
        result={key:(_apply(value,changes[key]) if key in changes else deepcopy(value))
            for key,value in previous.items() if key not in removed}
        for key,change in changes.items():
            if key not in previous:result[key]=_apply(None,change)
        return result
    raise ValueError('invalid native delta operation')


def encode_native_payload(payload, *, codec='legacy', checkpoint_rows=128, compression_level=1):
    if not isinstance(payload,dict):raise ValueError('native payload must be an object')
    if type(compression_level) is not int or not 1<=compression_level<=9:raise ValueError('gzip level must be 1..9')
    logical=canonical(payload)  # rejects NaN/Infinity and unsupported JSON values
    if codec=='legacy':return gzip.compress(logical,mtime=0,compresslevel=compression_level)
    if codec!=FORMAT:raise ValueError('unsupported native payload codec')
    if type(checkpoint_rows) is not int or not 1<=checkpoint_rows<=4096:
        raise ValueError('native checkpoint rows must be 1..4096')
    static,streams={},{}
    for key,value in payload.items():
        if not isinstance(value,list) or len(value)<2 or not all(isinstance(row,dict) for row in value):
            static[key]=value;continue
        blocks=[]
        for start in range(0,len(value),checkpoint_rows):
            rows=value[start:start+checkpoint_rows]
            blocks.append({'start':start,'first':rows[0],
                'patches':[_patch(previous,current) for previous,current in zip(rows,rows[1:])]})
        streams[key]={'length':len(value),'blocks':blocks}
    envelope={'format':FORMAT,'logical_sha256':hashlib.sha256(logical).hexdigest(),
        'logical_json_bytes':len(logical),'checkpoint_rows':checkpoint_rows,'static':static,'streams':streams}
    return gzip.compress(canonical(envelope),mtime=0,compresslevel=compression_level)


def decode_native_payload(blob):
    def invalid_constant(value):raise ValueError('nonfinite native JSON: '+value)
    value=json.loads(gzip.decompress(blob),parse_constant=invalid_constant)
    if not isinstance(value,dict):raise ValueError('native payload must be an object')
    if value.get('format')!=FORMAT:
        if str(value.get('format','')).startswith('native_lossless_delta_'):
            raise ValueError('unsupported native payload codec')
        return value
    required={'format','logical_sha256','logical_json_bytes','checkpoint_rows','static','streams'}
    if set(value)!=required or not isinstance(value['static'],dict) or not isinstance(value['streams'],dict):
        raise ValueError('native delta envelope schema mismatch')
    interval=value['checkpoint_rows']
    if type(interval) is not int or not 1<=interval<=4096:raise ValueError('invalid native checkpoint interval')
    result=deepcopy(value['static'])
    for key,stream in value['streams'].items():
        if key in result or not isinstance(stream,dict) or set(stream)!={'length','blocks'} or type(stream['length']) is not int or stream['length']<2 or not isinstance(stream['blocks'],list):
            raise ValueError('invalid native stream schema')
        rows=[]
        for block in stream['blocks']:
            if not isinstance(block,dict) or set(block)!={'start','first','patches'} or type(block['start']) is not int or block['start']!=len(rows) or not isinstance(block['first'],dict) or not isinstance(block['patches'],list) or len(block['patches'])>=interval:
                raise ValueError('invalid native checkpoint block')
            rows.append(deepcopy(block['first']))
            for patch in block['patches']:
                row=_apply(rows[-1],patch)
                if not isinstance(row,dict):raise ValueError('native stream row is not an object')
                rows.append(row)
        if len(rows)!=stream['length']:raise ValueError('native stream row count mismatch')
        result[key]=rows
    logical=canonical(result)
    if len(logical)!=value['logical_json_bytes'] or hashlib.sha256(logical).hexdigest()!=value['logical_sha256']:
        raise ValueError('native reconstructed logical SHA/size mismatch')
    return result
