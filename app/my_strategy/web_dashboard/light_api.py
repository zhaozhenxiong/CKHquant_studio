"""Local, bounded prediction and paper-account API."""
from datetime import date,datetime,timezone
from typing import Literal

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from my_strategy.web_dashboard.api import clean_symbol, submit_compute

router=APIRouter(prefix='/api/light')

class PredictionSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    symbol:str
    end:date|None=None
    lookback:Literal[128,256]=128
    horizon:Literal[1,5,10]=5
    samples:Literal[8,32,64]=8
    seed:int=Field(default=42,ge=0,le=2147483647)
    device:Literal['auto','cpu','mps','cuda']='auto'
    temperature:float=Field(default=.8,ge=.1,le=2)
    top_p:float=Field(default=.9,gt=0,le=1)
    _symbol=field_validator('symbol')(clean_symbol)

class SimulationSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    symbols:list[str]=Field(min_length=1,max_length=10)
    start:date=date(2026,1,1)
    end:date
    initial_cash:float=Field(default=1000000,ge=10000,le=100000000,allow_inf_nan=False)
    lookback:Literal[128,256]=128
    horizon:Literal[1,5,10]=5
    samples:Literal[8,32]=8
    seed:int=Field(default=42,ge=0,le=2147483647)
    device:Literal['auto','cpu','mps','cuda']='auto'
    temperature:float=Field(default=.8,ge=.1,le=2)
    top_p:float=Field(default=.9,gt=0,le=1)
    max_predictions:int=Field(default=60,ge=1,le=200)
    budget_seconds:int=Field(default=300,ge=10,le=1800)
    save:bool=False
    score_model_id:str|None=Field(default=None,max_length=160)
    score_model_type:Literal['mlp','l2']='mlp'
    research_mode:Literal['historical_candidate','paper']='historical_candidate'
    candidate_sessions:Literal[1,3]=3
    fusion_mode:Literal['evidence','hard_filters']='evidence'
    execution_mode:Literal['historical','forward']='historical'
    cost_stress:Literal[1.0,1.5,2.0]=1.0
    qualification_event:dict|None=None
    corporate_actions:list[dict]|None=Field(default=None,max_length=1000)
    @field_validator('symbols')
    @classmethod
    def stocks(cls,v):return list(dict.fromkeys(clean_symbol(s) for s in v))
    @model_validator(mode='after')
    def dates(self):
        if self.start>=self.end:raise ValueError('回测结束必须晚于开始')
        return self

class ScoreSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    symbols:list[str]=Field(default=['002436.SZ','600519.SH'],min_length=1,max_length=20)
    end:date=date(2025,12,31)
    model_type:Literal['mlp','l2']='mlp'
    variant:Literal['A','B']='A'
    device:Literal['cpu','auto','mps','cuda']='cpu'
    budget_seconds:int=Field(default=600,ge=10,le=600)
    select:Literal[False]=False
    observation_manifest:dict|None=None
    _stocks=field_validator('symbols')(SimulationSpec.stocks.__func__)

class AccountSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    enabled:bool=True
    shadow_research:bool=False

class ObservationSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    symbols:list[str]|None=Field(default=None,min_length=1,max_length=20)
    start:date|None=None
    end:date|None=None
    manifest_id:str|None=Field(default=None,max_length=64)
    max_calls:int=Field(default=200,ge=1,le=200)
    budget_seconds:int=Field(default=300,ge=1,le=300)
    device:Literal['auto','cpu','mps','cuda']='auto'
    samples:Literal[8,32]=32
    lookback:Literal[128,256]=128
    horizon:Literal[1,5,10]=5
    @field_validator('symbols')
    @classmethod
    def stocks(cls,v):return list(dict.fromkeys(clean_symbol(s) for s in v)) if v else v
    @model_validator(mode='after')
    def boundary(self):
        if not self.manifest_id and (not self.symbols or not self.start or not self.end or self.start>=self.end):
            raise ValueError('新观察人口需股票池及有效日期范围')
        return self

class DailySpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    required_price_date:date|None=None

class TuneSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    symbols:list[str]=Field(default=['002436.SZ','600519.SH'],min_length=1,max_length=5)
    validation_dates:list[date]=Field(default=[date(2025,9,1),date(2025,9,15)],min_length=1,max_length=5)
    test_dates:list[date]=Field(default=[date(2025,10,9),date(2025,10,20)],min_length=1,max_length=5)
    device:Literal['auto','mps','cpu','cuda']='auto'
    budget_seconds:int=Field(default=300,ge=10,le=600)
    _stocks=field_validator('symbols')(SimulationSpec.stocks.__func__)


def _recent_job_summaries(light):
    """Read bounded terminal history without restoring any executable job."""
    from contextlib import nullcontext
    terminal={'succeeded','partial','failed','blocked','cancelled'}
    merged={};recorded={};store=light.store
    if hasattr(store,'connection') and hasattr(store,'_lock'):
        now=store.clock()
        with store._lock:
            rows=store.connection.execute('''SELECT r.cache_key,r.created_at FROM results r
                JOIN objects o USING(sha256) WHERE r.namespace='jobs' AND o.state='active'
                AND (r.expires_at>? OR EXISTS(SELECT 1 FROM pins p WHERE p.cache_key=r.cache_key)
                    OR EXISTS(SELECT 1 FROM leases l WHERE l.cache_key=r.cache_key AND l.expires_at>?))
                ORDER BY r.created_at DESC,r.cache_key LIMIT 30''',(now,now)).fetchall()
        for row in rows:
            try:
                saved=store.get_by_key(row['cache_key'])
                if (not isinstance(saved,dict) or saved.get('status') not in terminal
                        or not isinstance(saved.get('job_id'),str)
                        or store.cache_key({'job_id':saved['job_id']},'jobs')!=row['cache_key']):continue
            except (OSError,ValueError,KeyError,TypeError,EOFError):continue
            merged[saved['job_id']]={k:v for k,v in saved.items() if k!='result'}
            recorded[saved['job_id']]=row['created_at']
    # In-memory observations take precedence even if they have not finished.
    with getattr(light,'lock',None) or nullcontext():
        for key,value in light.jobs.items():
            summary={k:v for k,v in value.items() if k!='result'}
            summary.setdefault('job_id',key)
            merged[summary['job_id']]=summary
    def observed_at(item):
        for name in ('finished_at','updated_at','created_at'):
            value=item.get(name)
            if isinstance(value,(int,float)) and not isinstance(value,bool):return float(value)
            if isinstance(value,str):
                try:
                    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
                    if parsed.tzinfo is None:parsed=parsed.replace(tzinfo=timezone.utc)
                    return parsed.timestamp()
                except (ValueError,OverflowError):pass
        return recorded.get(item['job_id'],0)
    return sorted(merged.values(),key=lambda item:(observed_at(item),item['job_id']),reverse=True)[:30]


@router.get('/status')
def status(request:Request):
    from my_strategy.services.kronos_prediction import asset_status
    from my_strategy.services.light_research import VERSION
    light=request.app.state.light
    model=light.current_score()
    return {'version':VERSION,'assets':asset_status(),
        'score_model':model,'score_candidates':light.list_score_models(),
        'jobs':_recent_job_summaries(light),
        'defaults':{'daily_mode':'lightweight','automatic_training':False,
                    'candidate_limit':20,'preview_samples':8,'cache_ttl_days':7,'cache_budget_gib':8},
        'cancelled_tasks_resumed':False}

@router.post('/predict',status_code=202)
def predict(spec:PredictionSpec,request:Request):
    return submit_compute(request,'light','prediction',spec.model_dump(mode='json',exclude_none=True))

@router.post('/simulate',status_code=202)
def simulate(spec:SimulationSpec,request:Request):
    return submit_compute(request,'light','simulation',spec.model_dump(mode='json'))

@router.post('/score/train',status_code=202)
def train_score(spec:ScoreSpec,request:Request):
    return submit_compute(request,'light','score_training',spec.model_dump(mode='json'))

@router.post('/score/observations',status_code=202)
def score_observations(spec:ObservationSpec,request:Request):
    return submit_compute(request,'light','score_observations',spec.model_dump(mode='json',exclude_none=True))

def _observation_manifest_view(value,manifest_id,*,details=False):
    """Expose resumable evidence, excluding storage/host location metadata."""
    import re
    from my_strategy.core.run_context import stable_hash
    def public(item):
        if isinstance(item,dict):
            return {key:public(child) for key,child in item.items()
                if key not in {'path','root','object_path','storage_path'}
                and not key.endswith(('_path','_root'))}
        if isinstance(item,list):return [public(child) for child in item]
        if isinstance(item,str):
            return re.sub(r"(?:/(?:Users|home|private|tmp|Volumes|var|opt)/[^\s\"'<>]+|(?<!\w)[A-Za-z]:[\\/][^\s\"'<>]+)",'<local-path>',item)
        return item
    keys=('schema','model_independent','source_policy_id','frozen_at','symbols','start','end',
          'parameters','total_calls','batch_count','coverage','requested','status','reason')
    result=public({key:value[key] for key in keys if key in value})
    result.update(manifest_id=manifest_id,source_population_sha256=stable_hash(value.get('population_identity')),
        saved_manifest_sha256=stable_hash(value),automatic_resume=False)
    if details:
        result['observations']=public(value.get('observations',[]))
        result['attempts']=public(value.get('attempts',[]))
    return result

@router.get('/score/observations')
def saved_score_observations(request:Request):
    from my_strategy.core.run_context import stable_hash
    light=request.app.state.light
    catalog=light.store.get_setting('checkpoint_catalog:score_observation_manifests',[]) or []
    rows=[];seen=set()
    for item in reversed(catalog[-30:]):
        reference=item.get('reference') or {}
        value=light.store.get_by_key(reference.get('cache_key')) if reference.get('cache_key') else None
        if not value or value.get('schema')!='score_observation_manifest_v1':continue
        manifest_id=stable_hash(value['population_identity'])
        if manifest_id in seen:continue
        seen.add(manifest_id)
        rows.append(_observation_manifest_view(value,manifest_id))
    return {'manifests':rows,'automatic_resume':False}

@router.get('/score/observations/{manifest_id}')
def saved_score_observation(manifest_id:str,request:Request):
    import re
    if re.fullmatch('[0-9a-f]{64}',manifest_id) is None:raise HTTPException(404,'冻结观察清单不存在或已过期')
    light=request.app.state.light
    value=light._read_checkpoint({'manifest_id':manifest_id},'score_observation_manifests')
    if not value or value.get('schema')!='score_observation_manifest_v1':
        raise HTTPException(404,'冻结观察清单不存在或已过期')
    return _observation_manifest_view(value,manifest_id,details=True)

@router.get('/score/models/{model_id}')
def score_model(model_id:str,request:Request):
    try:
        model=request.app.state.light.score_model(model_id)
        artifact=model.get('artifact') or {}
        from my_strategy.services.light_qualification import evaluate_qualification
        qualification=evaluate_qualification(artifact) if artifact.get('schema')=='light_score_v2' else artifact.get('qualification')
        return {**{k:v for k,v in model.items() if k!='artifact'},
                'metrics':artifact.get('metrics'), 'population':artifact.get('population'),
                'qualification':qualification,'model_version':artifact.get('model_version')}
    except (KeyError,ValueError) as exc:raise HTTPException(404,str(exc))

@router.post('/tune',status_code=202)
def tune(spec:TuneSpec,request:Request):
    return submit_compute(request,'light','tuning',spec.model_dump(mode='json'))

@router.get('/jobs/{job_id}')
def job(job_id:str,request:Request):
    try: value=request.app.state.light.get_job(job_id)
    except KeyError: raise HTTPException(404,'任务不存在或已过留存期')
    if value.get('kind')!='incremental' or isinstance(value.get('result'),dict):return value
    from my_strategy.services.incremental_sync import get_plan
    plan_id=(value.get('spec') or {}).get('plan_id')
    if not plan_id:
        return {**value,'result_availability':'not_recorded','result_unavailable_reason':'任务未记录增量计划身份'}
    try:
        plan=get_plan(store=request.app.state.light.store,plan_id=plan_id)
        result=plan.get('result')
        if isinstance(result,dict) and result.get('plan_id')==plan_id:
            return {**value,'result':result,'result_availability':'available','result_source':'incremental_plan'}
        return {**value,'result_availability':'not_recorded','result_unavailable_reason':'增量计划未记录可读取的执行结果'}
    except KeyError:
        return {**value,'result_availability':'expired_or_missing','result_unavailable_reason':'增量计划不存在或已过留存期；原始终态摘要保留，结果计数未记录'}
    except (OSError,ValueError,TypeError,EOFError):
        return {**value,'result_availability':'verification_failed','result_unavailable_reason':'增量计划结果完整性核验失败，未读取结果计数'}

@router.post('/jobs/{job_id}/cancel')
def cancel(job_id:str,request:Request):
    try: return request.app.state.light.cancel(job_id)
    except KeyError: raise HTTPException(404,'任务不存在或已过留存期')


def _derived_cost_report(value,store):
    """Join only three exact saved CAS identities; never rewrite research history."""
    from copy import deepcopy
    import hashlib
    import json
    from my_strategy.services.light_experiment_report import POLICIES,build_experiment_report
    from my_strategy.storage.research_store import canonical

    identity=value.get('identity') or {}
    required={'version','sources','calendar','start','end','cash','prediction','model','score',
              'score_model_type','max_predictions','budget_seconds','configs','corporate_actions',
              'code','execution_mode','qualification_event'}
    configs=identity.get('configs') or {}
    if (value.get('status')!='completed' or identity.get('kind')!='agent-simulation-v1'
            or not required.issubset(identity) or set(configs)!=set(POLICIES)
            or any(not isinstance(config,dict) or 'cost_stress' not in config for config in configs.values())
            or any(key in identity for key in ('partial_observation_scope','partial_stage','complete_through'))):
        return value
    if store.cache_key(identity,'simulations')!=value.get('result_id'):
        return value

    expected={}
    for scale in ('1.0','1.5','2.0'):
        candidate=deepcopy(identity)
        for config in candidate['configs'].values():config['cost_stress']=float(scale)
        expected[scale]=(store.cache_key(candidate,'simulations'),candidate)
    # Point lookup only: no historical enumeration, best-result selection, or unbounded reads.
    keys=tuple(key for key,_ in expected.values())
    with store._lock:
        rows=store.connection.execute('''SELECT r.cache_key,r.identity_json,r.sha256,o.size_bytes
            FROM results r JOIN objects o USING(sha256)
            WHERE r.namespace='simulations' AND o.state='active' AND r.cache_key IN (?,?,?)
            AND EXISTS(SELECT 1 FROM pins p WHERE p.cache_key=r.cache_key)''',keys).fetchall()
    indexed={row['cache_key']:dict(row) for row in rows}
    sources={};matched={};issues={}
    for scale,(key,candidate_identity) in expected.items():
        row=indexed.get(key)
        if row is None:
            issues[scale]='matching_saved_result_missing';continue
        try:
            if json.loads(row['identity_json'])!=candidate_identity:
                raise ValueError('indexed identity mismatch')
            candidate=store.get_by_key(key)  # verifies compressed bytes SHA and size
            if (not isinstance(candidate,dict) or candidate.get('result_id')!=key
                    or candidate.get('identity')!=candidate_identity):
                raise ValueError('payload identity mismatch')
            if candidate.get('status')!='completed':
                issues[scale]='matching_result_not_completed';continue
        except (OSError,ValueError,KeyError,TypeError,EOFError):
            issues[scale]='matching_result_verification_failed';continue
        matched[scale]=candidate
        sources[scale]={'result_id':key,'cache_key':key,'namespace':'simulations',
            'physical_sha256':row['sha256'],'compressed_bytes':row['size_bytes'],
            'identity_sha256':hashlib.sha256(canonical(candidate_identity).encode()).hexdigest(),
            'stored_report_id':(candidate.get('experiment_report') or {}).get('report_id')}

    # The requested result also has to be saved and verified; previews keep their stored report.
    if value['result_id'] not in {item['result_id'] for item in sources.values()}:
        return value
    report=build_experiment_report(value,cost_stress_results=matched)
    derivation={'schema':'saved_cost_report_view_v1','persisted':False,
        'stored_report_id':(value.get('experiment_report') or {}).get('report_id'),
        'report_id':report['report_id'],'source_results':sources,'issues':issues,
        'matching':'exact_complete_identity_except_configs_cost_stress',
        'expected_result_ids':{scale:key for scale,(key,_) in expected.items()},
        'candidate_limit':3,'verified_results':len(matched),'qualification_granted':False}
    derivation['derivation_id']=hashlib.sha256(canonical(derivation).encode()).hexdigest()
    return {**value,'experiment_report':report,'experiment_report_derivation':derivation}


@router.get('/results/{result_id}')
def result(result_id:str,request:Request):
    try:
        light=request.app.state.light
        return _derived_cost_report(light.result(result_id),light.store)
    except KeyError: raise HTTPException(404,'结果不存在或预览已过期')

@router.get('/saved')
def saved(request:Request):return {'results':request.app.state.light.store.list(namespace='simulations',limit=50)}

@router.post('/accounts/{result_id}')
def account(result_id:str,spec:AccountSpec,request:Request):
    try: return request.app.state.light.enable_account(result_id,spec.enabled,shadow_research=spec.shadow_research)
    except KeyError: raise HTTPException(404,'结果不存在或预览已过期')
    except ValueError as exc:raise HTTPException(409,str(exc))

@router.get('/accounts')
def accounts(request:Request):
    return {'accounts':request.app.state.light.store.get_setting('enabled_accounts',{}),
            'last_daily':request.app.state.light.store.get_setting('last_daily')}

@router.post('/daily',status_code=202)
def daily(spec:DailySpec,request:Request):
    return submit_compute(request,'light','daily',spec.model_dump(mode='json',exclude_none=True))
