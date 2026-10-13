"""Read-only relocation evidence for frozen trend-reassessment artifacts.

No training code or frozen manifest is rewritten. An origin receipt binds the
original completion audit and byte inventory; verification reads only the
target tree. Runtime-only verification never certifies a complete state copy.
The legacy completion API and generic state exporter remain unsupported for
arbitrary reassessment-state relocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.migration_manifest import file_sha256

FAMILY = 'czsc_trend_consensus_v2'
ORIGIN_SCHEMA = 'khquant_trend_reassessment_origin_receipt_v1'
CERTIFICATE_SCHEMA = 'khquant_trend_reassessment_relocation_certificate_v1'
STOCK_ASSETS = frozenset({'entries.parquet','exits.parquet','episodes.parquet',
    'reference_account.json','candidate_clones.json','holding_pairs.json'})
SOURCE_GENERATORS = frozenset('services/'+name for name in [
    'czsc_trend_v2_source.py','czsc_research_features.py','czsc_research_profiles.py',
    'czsc_wyckoff_features.py','czsc_wyckoff_events.py','czsc_native_evidence_v4.py',
    'czsc_analysis.py','czsc_research.py','czsc_research_data.py']) | {
        'adapters/czsc_adapter.py','configs/czsc_strategy.json'}


def _id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}',value):
        raise ValueError('unsafe artifact run identifier')
    return value


def _digest(value):
    if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value):
        raise ValueError('invalid frozen SHA-256 identity')
    return value


def _root(value):
    root=Path(value).absolute()
    if '..' in root.parts or root.is_symlink():raise ValueError('unsafe or linked artifact root')
    return root


def _file(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('asset path escapes its declared root')
    path=root
    for part in relative.parts:
        path=path/part
        if path.is_symlink() or bool(getattr(path,'is_junction',lambda:False)()):
            raise ValueError('linked artifact asset is not permitted')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('missing or outside artifact asset: '+str(relative))
    return path


def _directory(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('directory outside declared run')
    path=root
    for part in relative.parts:
        path=path/part
        if path.is_symlink() or bool(getattr(path,'is_junction',lambda:False)()):raise ValueError('linked model directory')
    if not path.is_dir() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('missing model directory')
    return path


def _json(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise ValueError('expected an artifact JSON object')
    return value


def _relative_old(path,old_root):
    # Lexical mapping deliberately does not resolve/open the original root.
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('frozen absolute asset path is unsafe')
    try:relative=path.relative_to(old_root)
    except ValueError as error:raise ValueError('frozen asset outside approved old root') from error
    if not relative.parts:raise ValueError('empty mapped artifact path')
    _id(relative.parts[0])
    return relative.as_posix()


def _completion(receipt,report,report_sha):
    audit=receipt.get('completion_audit') or {}
    if (receipt.get('run_id')!=report['run_id'] or receipt.get('status')!='completed' or
            receipt.get('strategy_version')!=FAMILY or audit.get('read_only') is not True or
            audit.get('family')!=FAMILY or audit.get('model_population_DAG_verified') is not True or
            audit.get('report_sha256')!=report_sha):
        raise ValueError('original completion receipt does not bind this completed report')
    expected={'E0','E1',*['final_population/'+_id(row['fold']) for row in report['population_folds']]}
    recorded=audit.get('dataset_assets') or []
    if len(recorded)!=len(expected) or {row['stage'] for row in recorded}!=expected:
        raise ValueError('original completion receipt omits a dataset stage')
    return audit


def _models(root,run_id):
    from my_strategy.services import czsc_trend_v2_pipeline as pipeline
    model_root=root/_id(run_id)
    report_path=_file(root,run_id+'/reports/trend_consensus.json');report=_json(report_path)
    if report.get('run_id')!=run_id or report.get('status')!='completed' or report.get('strategy_version')!=FAMILY:
        raise ValueError('relocation requires a completed compatible V2 run')
    for bundle in [*report['trend_bundles'],*report['ma_exit_bundles'],*report['architecture_bundles']]:
        for record in bundle['heads'].values():_members(_directory(model_root,record['path']))
    if set(report['source_snapshot']['source_snapshot']['code_hashes'])!=SOURCE_GENERATORS:
        raise ValueError('source generator closure is incomplete')
    pipeline._verify_run(report,model_root)
    return report,report_path


def _cold_heads(root,report):
    """Deserialize every verified checkpoint, without issuing a forecast."""
    import joblib
    from my_strategy.services.czsc_trend_v2_model import load_trend_head
    from my_strategy.services.czsc_trend_v2_interval_model import HeadSession
    session=HeadSession();loaded=[]
    for bundle in [*report['trend_bundles'],*report['ma_exit_bundles'],*report['architecture_bundles']]:
        for head,record in bundle['heads'].items():
            path=_directory(root/report['run_id'],record['path'])
            if head=='trend_direction':
                manifest=load_trend_head(path);model=joblib.load(_file(path,manifest['checkpoint']))
                if list(model.classes_)!=[0,1,2]:raise ValueError('relocated trend checkpoint class drift')
            else:manifest,_=session._model(path)
            loaded.append({'head':head,'checkpoint':bundle['name'],'manifest_sha256':manifest['manifest_sha256']})
    return loaded


def _known_members(root,old_root,report,audit,*,complete,calendar_source=None,calendar_original_root=None):
    """Derive mandatory members from frozen reports, never from a loose glob."""
    known={};run_id=report['run_id'];model=root/run_id
    def add(relative,sha):
        relative=Path(relative).as_posix();sha=_digest(sha)
        if relative in known and known[relative]!=sha:raise ValueError('conflicting frozen member identity')
        known[relative]=sha
    audits={row['stage']:row for row in audit['dataset_assets']}
    stages=['E0','E1',*['final_population/'+row['fold'] for row in report['population_folds']]]
    manifests={row['stage']:row for row in report['dataset_manifests']}
    if len(manifests)!=len(stages) or set(manifests)!=set(stages):
        raise ValueError('frozen dataset manifest catalogue is incomplete')
    symbols=[row['symbol'] for row in report['dataset_records']]
    if not symbols or len(symbols)!=len(set(symbols)):raise ValueError('invalid frozen model cohort')
    for stage in stages:
        record=manifests[stage];relative=run_id+'/'+record['path']
        if record['path']!='dataset/'+stage+'/manifest.json':raise ValueError('dataset manifest stage path mismatch')
        path=_file(root,relative)
        if file_sha256(path)!=record['sha256'] or audits[stage]['manifest_sha256']!=record['sha256']:
            raise ValueError('dataset manifest differs from its original completion audit')
        add(relative,record['sha256']);manifest=_json(path);records=manifest.get('records') or []
        if (manifest.get('failures') or len(records)!=len(symbols) or
                {row['symbol'] for row in records}!=set(symbols) or audits[stage]['stocks']!=len(symbols)):
            raise ValueError('dataset cohort is incomplete')
        if set(manifest.get('pooled_files') or {})!={'entries','exits','episodes'}:
            raise ValueError('pooled training assets are incomplete')
        for stock in records:
            if not STOCK_ASSETS.issubset(stock.get('files',{})):raise ValueError('account/label assets are incomplete')
            if complete:
                stock_manifest=_json(_file(model,'dataset/'+stage+'/'+stock['symbol'].replace('.','_')+'/manifest.json'))
                if stock_manifest!=stock:raise ValueError('stock manifest differs from the frozen parent summary')
            for name,asset in stock['files'].items():
                if Path(name).name!=name:raise ValueError('unsafe account asset filename')
                relative=_relative_old(asset['path'],old_root)
                expected=run_id+'/dataset/'+stage+'/'+stock['symbol'].replace('.','_')+'/'+name
                if relative!=expected:raise ValueError('account asset escaped its frozen run/stage')
                if complete:add(relative,asset['sha256'])
        for name,asset in manifest['pooled_files'].items():
            relative=_relative_old(asset['path'],old_root)
            if relative!=run_id+'/dataset/'+stage+'/'+name+'.parquet':raise ValueError('pooled asset stage mismatch')
            if complete:add(relative,asset['sha256'])
    if not complete:return known,{run_id:root/run_id},{}
    source=report['source_snapshot'];native_id=_id(source['source_native_run_id'])
    native_path=_file(root,native_id+'/reports/native_v4.json')
    if file_sha256(native_path)!=source['source_report_sha256']:raise ValueError('native source report byte drift')
    add(native_id+'/reports/native_v4.json',source['source_report_sha256']);native=_json(native_path)
    if (native.get('run_id')!=native_id or native.get('status')!='completed' or native.get('strategy_version')!=FAMILY or
            native.get('raw_only_rebuild') is not True or native.get('reused_model_or_label') is not False or
            native.get('source_snapshot')!=source['source_snapshot']):raise ValueError('native source protocol mismatch')
    raw_id=_id(native['source_snapshot']['raw_source_run_id']);raw_path=_file(root,raw_id+'/reports/raw_source.json');raw=_json(raw_path)
    if (raw.get('run_id')!=raw_id or raw.get('status')!='completed' or raw.get('failures') or raw.get('strategy_version')!=FAMILY or
            raw.get('raw_only_rebuild') is not True or raw.get('reused_model_or_label') is not False or
            raw.get('snapshot_hash')!=stable_hash({key:value for key,value in raw.items() if key!='snapshot_hash'}) or
            raw['snapshot_hash']!=native['source_snapshot']['raw_snapshot_hash']):raise ValueError('raw source snapshot identity mismatch')
    raw_records=raw.get('raw_records') or [];native_records=native.get('dataset_records') or []
    raw_by_symbol={row['symbol']:row for row in raw_records};native_by_symbol={row['symbol']:row for row in native_records}
    if (not raw_records or len(raw_by_symbol)!=len(raw_records) or len(native_by_symbol)!=len(native_records) or
            set(raw_by_symbol)!=set(native_by_symbol) or set(raw_by_symbol)!=set(raw['selected_symbols']) or
            native['source_snapshot']['raw_input_hashes']!={symbol:row['raw_sha256'] for symbol,row in raw_by_symbol.items()}):
        raise ValueError('full raw/native source cohort is incomplete')
    for symbol,row in native_by_symbol.items():
        original=raw_by_symbol[symbol]
        if any(row.get(key)!=value for key,value in original.items()):raise ValueError('native record changed its frozen raw identity')
        if row.get('raw_only_rebuild') is not True or row.get('reused_model_or_label') is not False:
            raise ValueError('native record imports a model or label')
        attrs=row.get('raw_attrs') or {}
        if (not {'price_basis','legacy_fuyao_bars','unverified_trade_price_bars','volume_unit','amount_unit','data_version'}.issubset(attrs) or
                attrs['data_version']!=row.get('raw_series_hash')):raise ValueError('source raw observation identity is missing')
        for key,digest,identity,name in [('raw_path','raw_sha256',raw_id,'raw.parquet'),
                ('path','sha256',native_id,'features.parquet'),('native_path','native_sha256',native_id,'native.json')]:
            relative=_relative_old(row[key],old_root)
            sub='raw' if key=='raw_path' else 'native'
            if relative!=identity+'/data_snapshots/'+sub+'/'+symbol.replace('.','_')+'/'+name:
                raise ValueError('source asset escaped its declared source run')
            add(relative,row[digest])
    for row in report['dataset_records']:
        if native_by_symbol.get(row['symbol'])!=row:raise ValueError('model input record differs from its complete native source')
    calendar_id=_id(native['source_snapshot']['calendar_run_id'])
    if len({run_id,raw_id,native_id,calendar_id})!=4:raise ValueError('source dependency run identity collision')
    calendar_root=Path(calendar_source).absolute() if calendar_source else root/calendar_id
    calendar_old=_root(calendar_original_root or (calendar_root.parent if calendar_source else old_root))
    if calendar_root.name!=calendar_id or calendar_root.is_symlink():raise ValueError('calendar source must be a canonical matching run')
    calendar=_json(_file(calendar_root,'reports/calendar.json'));dates=calendar.get('dates') or []
    calendar_identity=calendar.get('run_id')
    if calendar_identity is None:
        calendar_identity=_json(_file(calendar_root,'metadata.json')).get('run_id')
    observed=native.get('calendar_dates') or []
    if (calendar_identity!=calendar_id or calendar.get('verified') is not True or not calendar.get('source') or dates!=sorted(set(dates)) or
            observed!=[day for day in dates if day<=native['data_end']] or not observed or
            observed!=raw.get('dates') or stable_hash(observed)!=native['source_snapshot']['calendar_hash'] or
            raw.get('calendar_run_id')!=calendar_id or raw.get('calendar_hash')!=stable_hash(observed)):
        raise ValueError('verified calendar/source binding mismatch')
    snapshots=_json(_file(model,'data_snapshots/calendar.json'))
    if snapshots.get('verified') is not True or snapshots.get('source_run_id')!=calendar_id or snapshots.get('dates')!=[d for d in observed if d<=report['data_end']]:
        raise ValueError('model frozen calendar mismatch')
    roots={run_id:model,raw_id:root/raw_id,native_id:root/native_id};visited=set();current_id=calendar_id
    current_root=calendar_root;current=calendar
    while True:
        if current_id in visited or current_id in {run_id,raw_id,native_id}:raise ValueError('calendar dependency cycle or identity collision')
        visited.add(current_id);roots[current_id]=current_root
        current_dates=current.get('dates') or []
        if current.get('verified') is not True or not current.get('source') or current_dates!=sorted(set(current_dates)):
            raise ValueError('invalid calendar provenance dependency')
        assets=list(current.get('source_artifacts') or [])
        if current.get('artifact'):
            assets.append({'path':current['artifact'],'sha256':current['sha256']})
        for asset in assets:
            relative=_relative_old(asset['path'],calendar_old)
            if not relative.startswith(current_id+'/'):raise ValueError('calendar asset escaped its own provenance run')
            add(relative,asset['sha256'])
        parent=current.get('parent_calendar')
        if not parent:break
        parent_id=_id(parent['run_id']);relative=_relative_old(parent['path'],calendar_old)
        if relative!=parent_id+'/reports/calendar.json':raise ValueError('calendar parent path mismatch')
        parent_root=(calendar_root.parent if calendar_source else root)/parent_id
        parent_path=_file(parent_root,'reports/calendar.json')
        if file_sha256(parent_path)!=parent['file_sha256']:raise ValueError('calendar parent report byte drift')
        add(relative,parent['file_sha256']);value=_json(parent_path)
        if stable_hash(value)!=parent['hash']:raise ValueError('calendar parent semantic identity drift')
        current_id,current_root,current=parent_id,parent_root,value
    return known,roots,{'raw_run_id':raw_id,'native_run_id':native_id,'calendar_run_id':calendar_id,
        'calendar_provenance_run_ids':sorted(visited),'calendar_original_runs_root':str(calendar_old),'source_symbols':len(raw_records)}


def _members(root):
    if root.is_symlink() or not root.is_dir():raise ValueError('missing or linked artifact run')
    result=[]
    for current,dirs,files in os.walk(root,followlinks=False):
        for name in [*dirs,*files]:
            path=Path(current)/name
            if path.is_symlink() or bool(getattr(path,'is_junction',lambda:False)()):raise ValueError('linked artifact member')
        result.extend(Path(current)/name for name in files)
    return sorted(result)


def _evidence_path(output,forbidden_roots):
    output=Path(output).absolute()
    resolved=output.resolve()
    if any(output.is_relative_to(root) or resolved.is_relative_to(root.resolve()) for root in forbidden_roots):
        raise ValueError('evidence must remain outside frozen artifact roots')
    if output.exists() or output.is_symlink():raise FileExistsError('refusing to overwrite existing evidence')
    return output


def _write_evidence(output,payload,forbidden_roots):
    output=_evidence_path(output,forbidden_roots)
    payload['evidence_hash']=stable_hash(payload)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return payload


def make_origin_receipt(*,original_runs_root,model_run_id,completion_audit,output,calendar_source=None):
    root=_root(original_runs_root);_evidence_path(output,[root]);run_id=_id(model_run_id);report,path=_models(root,run_id)
    receipt_path=Path(completion_audit);receipt=_json(receipt_path);audit=_completion(receipt,report,file_sha256(path))
    known,roots,source_info=_known_members(root,root,report,audit,complete=True,calendar_source=calendar_source)
    inventory=[]
    for identity,physical_root in sorted(roots.items()):
        for member in _members(physical_root):
            relative=identity+'/'+member.relative_to(physical_root).as_posix()
            inventory.append({'path':relative,'bytes':member.stat().st_size,
                'sha256':known.get(relative) or file_sha256(member),
                'hash_basis':'frozen_member_manifest' if relative in known else 'origin_extra_file_byte_snapshot'})
    if not set(known).issubset({row['path'] for row in inventory}):raise ValueError('origin tree is missing a mandatory frozen asset')
    payload={'schema':ORIGIN_SCHEMA,'recorded_at':datetime.now(timezone.utc).isoformat(),'model_run_id':run_id,
        'original_runs_root':str(root),'original_completion_receipt':receipt,'original_completion_receipt_sha256':file_sha256(receipt_path),
        'model_report_sha256':file_sha256(path),'protocol_freeze_hash':report['protocol_freeze']['freeze_hash'],
        'source_snapshot_hash':report['source_snapshot']['snapshot_hash'],'run_ids':sorted(roots),'source_dependencies':source_info,
        'inventory':inventory,'known_assets_rehashed_at_origin':False,
        'hash_basis_note':'Large members retain SHA anchored by frozen manifests and the original completion receipt. Every target byte is rehashed during verification; extra files are hashed here.',
        'tool_sha256':file_sha256(Path(__file__)),'publication_allowed':False}
    return _write_evidence(output,payload,[root])


def verify_relocation(*,original_runs_root,target_runs_root,model_run_id,output,origin_receipt=None,completion_audit=None,scope='complete'):
    old=_root(original_runs_root);target=_root(target_runs_root);run_id=_id(model_run_id)
    if old.resolve()==target.resolve() or old.resolve().is_relative_to(target.resolve()) or target.resolve().is_relative_to(old.resolve()):
        raise ValueError('relocation roots must be disjoint')
    _evidence_path(output,[old,target])
    if scope not in {'complete','runtime-only'}:raise ValueError('unknown relocation verification scope')
    if scope=='complete' and origin_receipt is None:raise ValueError('complete verification requires a bound origin receipt')
    evidence_path=Path(origin_receipt or completion_audit or '').absolute()
    if any(evidence_path.resolve().is_relative_to(root.resolve()) for root in (old,target)):
        raise ValueError('verification receipts must be external to artifact roots')
    origin=_json(Path(origin_receipt)) if origin_receipt else None
    if origin:
        if (origin.get('schema')!=ORIGIN_SCHEMA or origin.get('evidence_hash')!=stable_hash({k:v for k,v in origin.items() if k!='evidence_hash'}) or
                origin.get('model_run_id')!=run_id or origin.get('original_runs_root')!=str(old)):
            raise ValueError('origin receipt identity or root mapping drift')
        base_receipt=origin['original_completion_receipt']
    elif completion_audit:base_receipt=_json(Path(completion_audit))
    else:raise ValueError('runtime verification requires original completion evidence')
    accesses=[];denied=[];guard={'active':True};denied_roots=[old.resolve()]
    if origin:
        dependencies=origin.get('source_dependencies') or {}
        calendar_old=dependencies.get('calendar_original_runs_root')
        if calendar_old and _root(calendar_old).resolve()!=target.resolve():
            denied_roots.extend((_root(calendar_old)/identity).resolve() for identity in dependencies.get('calendar_provenance_run_ids',[]))
    def hook(event,args):
        if event!='open' or not guard['active'] or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();accesses.append(str(path))
        if any(path.is_relative_to(root) for root in denied_roots):
            denied.append(str(path));raise PermissionError('verification refused an original-root file open')
    sys.addaudithook(hook)
    try:
        report,report_path=_models(target,run_id);report_sha=file_sha256(report_path)
        audit=_completion(base_receipt,report,report_sha)
        if origin and (origin['model_report_sha256']!=report_sha or origin['protocol_freeze_hash']!=report['protocol_freeze']['freeze_hash'] or
                origin['source_snapshot_hash']!=report['source_snapshot']['snapshot_hash']):
            raise ValueError('origin receipt/report/source binding mismatch')
        cold_heads=_cold_heads(target,report)
        known,roots,source_info=_known_members(target,old,report,audit,complete=scope=='complete',
            calendar_original_root=(origin or {}).get('source_dependencies',{}).get('calendar_original_runs_root'))
        records=[]
        if scope=='complete':
            if (origin['model_report_sha256']!=report_sha or origin['protocol_freeze_hash']!=report['protocol_freeze']['freeze_hash'] or
                    origin['source_snapshot_hash']!=report['source_snapshot']['snapshot_hash'] or origin['run_ids']!=sorted(roots) or
                    origin['source_dependencies']!=source_info):raise ValueError('origin receipt/report/source binding mismatch')
            expected={row['path']:row for row in origin['inventory']}
            if len(expected)!=len(origin['inventory']) or not set(known).issubset(expected):raise ValueError('origin inventory omits mandatory assets')
            actual={identity+'/'+member.relative_to(physical).as_posix() for identity,physical in roots.items() for member in _members(physical)}
            if actual!=set(expected):raise ValueError('target inventory has missing or unexpected artifact members')
            for relative,record in sorted(expected.items()):
                if relative.split('/')[0] not in roots:raise ValueError('inventory run is outside approved closure')
                path=_file(target,relative);expected_sha=_digest(record['sha256'])
                if relative in known and known[relative]!=expected_sha:raise ValueError('origin inventory contradicts frozen member SHA')
                before=path.stat();digest=file_sha256(path);after=path.stat()
                if before.st_size!=record['bytes'] or digest!=expected_sha or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise ValueError('target frozen byte drift: '+relative)
                records.append({'path':relative,'bytes':before.st_size,'sha256':digest})
        else:
            for relative,digest in sorted(known.items()):
                path=_file(target,relative)
                if file_sha256(path)!=digest:raise ValueError('runtime catalogue byte drift')
                records.append({'path':relative,'bytes':path.stat().st_size,'sha256':digest})
    finally:guard['active']=False
    from my_strategy.services import czsc_trend_v2_pipeline as pipeline
    payload={'schema':CERTIFICATE_SCHEMA,'verified_at':datetime.now(timezone.utc).isoformat(),
        'scope':scope,'complete_state_verified':scope=='complete','model_run_id':run_id,
        'original_runs_root':str(old),'target_runs_root':str(target),'old_root_access_denied':True,'old_root_open_attempts':denied,
        'denied_original_roots':[str(path) for path in denied_roots],
        'opened_paths':sorted(set(accesses)),'verified_assets':records,'verified_asset_count':len(records),
        'verified_bytes':sum(row['bytes'] for row in records),'loaded_model_heads':len(cold_heads),'cold_deserialized_heads':cold_heads,
        'loaded_generating_code_root':str(pipeline.MY_STRATEGY_ROOT),'generating_code_sha256':report['protocol_freeze']['code_hashes'],
        'source_generator_sha256':report['source_snapshot']['source_snapshot']['code_hashes'],
        'origin_receipt_sha256':file_sha256(Path(origin_receipt)) if origin_receipt else None,
        'completion_receipt_report_sha256':report_sha,'protocol_freeze_hash':report['protocol_freeze']['freeze_hash'],
        'source_snapshot_hash':report['source_snapshot']['snapshot_hash'],'source_dependencies':source_info,
        'required_recent_heads_ready':report['required_recent_heads_ready'],'publication_allowed':report['publication_allowed'],
        'model_gate':report['model_gate'],'legacy_finalize_relocation_supported':False,'generic_state_export_certified':False,
        'limitations':['The legacy finalize_existing_run(target) remains incompatible with old absolute asset paths.',
            'Runtime-only scope does not certify copied labels or source inputs; complete scope requires every origin-inventory member.',
            'Trusted original completion/origin receipts are required; this is local integrity evidence, not a signature or production qualification.'],
        'tool_sha256':file_sha256(Path(__file__))}
    return _write_evidence(output,payload,[old,target])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    origin=commands.add_parser('origin',help='Bind existing original completion evidence and frozen source inventory')
    origin.add_argument('--original-runs-root',type=Path,required=True);origin.add_argument('--model-run-id',required=True)
    origin.add_argument('--completion-audit',type=Path,required=True);origin.add_argument('--calendar-source',type=Path)
    origin.add_argument('--output',type=Path,required=True)
    verify=commands.add_parser('verify',help='Read only target assets and write an external, immutable certificate')
    verify.add_argument('--original-runs-root',type=Path,required=True);verify.add_argument('--target-runs-root',type=Path,required=True)
    verify.add_argument('--model-run-id',required=True);verify.add_argument('--scope',choices=['complete','runtime-only'],default='complete')
    evidence=verify.add_mutually_exclusive_group(required=True)
    evidence.add_argument('--origin-receipt',type=Path);evidence.add_argument('--completion-audit',type=Path)
    verify.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args(argv));command=args.pop('command')
    try:result=make_origin_receipt(**args) if command=='origin' else verify_relocation(**args)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'verified':False,'error':str(error)},ensure_ascii=False));return 1
    print(json.dumps({'verified':True,'schema':result['schema'],'scope':result.get('scope','origin'),
        'complete_state_verified':result.get('complete_state_verified',False),'evidence_hash':result['evidence_hash']},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
