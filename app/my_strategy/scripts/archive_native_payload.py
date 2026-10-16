"""Create/restore verified native delta archive COPIES; existing files are never replaced.

Restoration preserves the complete JSON value and exact scalar types. Canonical
legacy gzip has its own physical SHA; it must not replace a frozen source object.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from my_strategy.storage.native_payload_codec import FORMAT,canonical,encode_native_payload,decode_native_payload

MANIFEST='native_archive_copy_v1'


def sha(blob):return hashlib.sha256(blob).hexdigest()


def write_new(path,content):
    path=Path(path).expanduser().resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content);stream.flush();os.fsync(stream.fileno())
    return path


def create(source,archive,*,manifest=None):
    source=Path(source).expanduser()
    if source.is_symlink() or not source.is_file():raise ValueError('original native file is absent or unsafe')
    source=source.resolve();archive=Path(archive).expanduser().resolve()
    manifest=Path(manifest or str(archive)+'.manifest.json').expanduser().resolve()
    if len({source,archive,manifest})!=3:raise ValueError('source/archive/manifest paths must be distinct')
    if archive.exists() or manifest.exists():raise FileExistsError('archive or manifest already exists; never overwritten')
    stat=source.stat();original=source.read_bytes();payload=decode_native_payload(original)
    logical=canonical(payload);encoded=encode_native_payload(payload,codec=FORMAT)
    restored=encode_native_payload(decode_native_payload(encoded))
    if canonical(decode_native_payload(restored))!=logical:raise ValueError('archive reconstruction mismatch')
    after=source.stat()
    if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or sha(source.read_bytes())!=sha(original):
        raise ValueError('original changed during archive preparation')
    record={'schema':MANIFEST,'codec':FORMAT,'original_path':str(source),'archive_path':str(archive),
        'original_gzip_sha256':sha(original),'archive_sha256':sha(encoded),
        'logical_sha256':sha(logical),'logical_json_bytes':len(logical),
        'restored_legacy_gzip_sha256':sha(restored),'original_bytes':len(original),'archive_bytes':len(encoded),
        'original_preserved':True,'index_modified':False,'reconstruction':'complete canonical JSON; new physical gzip SHA',
        'source_stat':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'device':stat.st_dev,'inode':stat.st_ino}}
    write_new(archive,encoded);write_new(manifest,json.dumps(record,ensure_ascii=False,indent=2).encode()+b'\n')
    return {**record,'manifest_path':str(manifest)}


def restore(archive,manifest,output):
    archive=Path(archive).expanduser();manifest=Path(manifest).expanduser();output=Path(output).expanduser().resolve()
    if archive.is_symlink() or manifest.is_symlink():raise ValueError('archive/manifest symlinks are not accepted')
    record=json.loads(manifest.read_text())
    if record.get('schema')!=MANIFEST or record.get('codec')!=FORMAT:raise ValueError('native archive manifest schema mismatch')
    if output in {archive.resolve(),manifest.resolve(),Path(record['original_path']).resolve()}:
        raise ValueError('restore output must be a new independent path; frozen original identity is retained')
    if output.exists():raise FileExistsError('restore output exists; never overwritten')
    encoded=archive.read_bytes()
    if sha(encoded)!=record['archive_sha256'] or len(encoded)!=record['archive_bytes']:raise ValueError('archive SHA/size mismatch')
    payload=decode_native_payload(encoded);logical=canonical(payload)
    if sha(logical)!=record['logical_sha256'] or len(logical)!=record['logical_json_bytes']:raise ValueError('reconstructed logical SHA/size mismatch')
    restored=encode_native_payload(payload)
    if sha(restored)!=record['restored_legacy_gzip_sha256']:raise ValueError('reconstructed legacy gzip SHA mismatch')
    write_new(output,restored)
    return {'output_path':str(output),'output_sha256':sha(restored),'logical_sha256':sha(logical),
        'logical_equivalent':True,'original_modified':False,'index_modified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    make=commands.add_parser('create');make.add_argument('--source',type=Path,required=True);make.add_argument('--archive',type=Path,required=True);make.add_argument('--manifest',type=Path)
    recover=commands.add_parser('restore');recover.add_argument('--archive',type=Path,required=True);recover.add_argument('--manifest',type=Path,required=True);recover.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=create(args.source,args.archive,manifest=args.manifest) if args.action=='create' else restore(args.archive,args.manifest,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
