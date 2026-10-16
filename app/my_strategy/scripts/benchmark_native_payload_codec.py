"""Export and verify opt-in native delta copies; original files remain untouched."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
try:
    import resource
except ImportError:  # Windows has no POSIX resource module.
    resource = None

from my_strategy.storage.native_payload_codec import FORMAT,canonical,encode_native_payload,decode_native_payload


def resource_usage():
    available = resource is not None
    return {'process_maxrss_platform_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if available else None,
        'process_maxrss_available': available,
        'process_maxrss_unavailable_reason': None if available else 'resource_module_unavailable'}


def run(samples,output_dir):
    output_dir=Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for filename in samples:
        path=Path(filename).expanduser()
        if path.is_symlink() or not path.is_file():raise ValueError('native input is absent or a symlink')
        path=path.resolve()
        if path.is_relative_to(output_dir):raise ValueError('output directory must not contain original inputs')
        before=path.stat();blob=path.read_bytes();physical_sha=hashlib.sha256(blob).hexdigest()
        if len(path.name)==64 and physical_sha!=path.name:raise ValueError('source CAS SHA mismatch')
        started=time.perf_counter();payload=decode_native_payload(blob);legacy_decode=time.perf_counter()-started
        logical=canonical(payload);logical_sha=hashlib.sha256(logical).hexdigest()
        started=time.perf_counter();legacy=encode_native_payload(payload);legacy_encode=time.perf_counter()-started
        started=time.perf_counter();encoded=encode_native_payload(payload,codec=FORMAT);delta_encode=time.perf_counter()-started
        destination=output_dir/(physical_sha+'.delta.json.gz')
        if destination.exists() and destination.read_bytes()!=encoded:raise ValueError('different codec output already exists')
        destination.write_bytes(encoded)
        started=time.perf_counter();restored=decode_native_payload(destination.read_bytes());delta_decode=time.perf_counter()-started
        if canonical(restored)!=logical:raise ValueError('lossless reconstruction failed')
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or hashlib.sha256(path.read_bytes()).hexdigest()!=physical_sha:
            raise ValueError('original changed during codec audit')
        rows.append({'input_path':str(path),'input_sha256':physical_sha,'output_path':str(destination),
            'output_sha256':hashlib.sha256(encoded).hexdigest(),'logical_sha256':logical_sha,
            'original_bytes':len(blob),'canonical_legacy_level1_bytes':len(legacy),'delta_level1_bytes':len(encoded),
            'saved_vs_original_fraction':1-len(encoded)/len(blob),'saved_vs_canonical_legacy_fraction':1-len(encoded)/len(legacy),
            'rows':len(payload.get('evaluations',[])),'logical_json_bytes':len(logical),
            'legacy_decode_seconds':legacy_decode,'delta_decode_seconds':delta_decode,
            'canonical_legacy_encode_seconds':legacy_encode,'delta_encode_seconds':delta_encode,
            'equivalent':True,'original_unchanged':True})
        del payload,logical,legacy,encoded,restored,blob;gc.collect()
    return {'schema':'native_codec_benchmark_v1','codec':FORMAT,'gzip_level':1,'samples':rows,
        **resource_usage(),
        'default_enabled':False,'production_migration_performed':False,
        'scope':'explicit copy exports only; small real samples do not estimate full-store savings'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();report=run(args.sample,args.output_dir)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
