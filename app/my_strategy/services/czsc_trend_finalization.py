"""Read-only completion audit for independently rebuilt reassessment models.

Unlike the legacy recovery adapter, this never reconstructs an interrupted
model report or imports a previous family. A complete current DAG must already exist.
"""
from pathlib import Path
import json

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.services.czsc_trend_support import _file_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID


def audit_stage_assets(root, relative, *, expected_symbols, expected_population=None):
    root=Path(root).resolve();directory=(root/'dataset'/relative).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('dataset audit path outside model run')
    manifest=json.loads((directory/'manifest.json').read_text())
    records=manifest.get('records') or []
    if manifest.get('failures') or {row['symbol'] for row in records}!=set(expected_symbols) or len(records)!=len(expected_symbols):
        raise ValueError('incomplete reassessment dataset cohort')
    if expected_population is not None and manifest.get('reference_population')!=expected_population:
        raise ValueError('dataset reference population differs from its frozen L2 dependency')
    for record in records:
        required={'entries.parquet','exits.parquet','episodes.parquet','reference_account.json',
            'candidate_clones.json','holding_pairs.json'}
        if not required.issubset(record.get('files',{})):
            raise ValueError('incomplete reassessment account/label assets')
        for asset in record.get('files',{}).values():
            path=Path(asset['path']).resolve()
            if not path.is_relative_to(directory) or _file_hash(path)!=asset['sha256']:
                raise ValueError('stored reassessment account/label asset drift')
    if set(manifest.get('pooled_files') or {})!={'entries','exits','episodes'}:
        raise ValueError('incomplete pooled reassessment training data')
    for asset in manifest['pooled_files'].values():
        path=Path(asset['path']).resolve()
        if not path.is_relative_to(directory) or _file_hash(path)!=asset['sha256']:
            raise ValueError('pooled reassessment training data drift')
    return {'stage':relative,'stocks':len(records),'files':sum(len(row.get('files',{})) for row in records),
        'manifest_sha256':_file_hash(directory/'manifest.json')}


def finalize_existing_run(run_id):
    """Verify a completed current run; incomplete runs require a fresh frozen run."""
    root=(ARTIFACT_RUNS_ROOT/str(run_id)).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):
        raise ValueError('invalid reassessment completion root')
    target=root/'reports/trend_consensus.json'
    if not target.is_file():
        raise ValueError('current completion cannot recover legacy/interrupted labels or model populations; rebuild a fresh run')
    from my_strategy.services.czsc_trend_pipeline import read_run
    verified,report=read_run(run_id)
    if verified!=root or report.get('strategy_version')!=STRATEGY_ID:
        raise ValueError('foreign completion report cannot serve reassessment strategy')
    symbols=[record['symbol'] for record in report['dataset_records']]
    audits=[audit_stage_assets(root,stage,expected_symbols=symbols) for stage in ('E0','E1')]
    for population in report['population_folds']:
        audits.append(audit_stage_assets(root,'final_population/'+population['fold'],
            expected_symbols=symbols,expected_population=population['reference_population']))
    return {**report,'completion_audit':{'read_only':True,'family':STRATEGY_ID,
        'model_population_DAG_verified':True,'dataset_assets':audits,'report_sha256':_file_hash(target)}}
