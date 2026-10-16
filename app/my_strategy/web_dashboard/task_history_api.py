"""Unified task history and exact, previewed maintenance plans."""
from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from my_strategy.services.task_maintenance import TaskMaintenance, MaintenanceConflict

router = APIRouter(prefix='/api/task-history')


class TaskRef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    backend: Literal['legacy', 'light']
    job_id: str = Field(min_length=1, max_length=160, pattern=r'^[\w.-]+$')


class DeleteSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tasks: list[TaskRef] = Field(min_length=1, max_length=1000)


class PreviewSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tasks: list[TaskRef] = Field(default_factory=list, max_length=1000)
    include_artifacts: bool = True


class ExecuteSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    plan_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    plan_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')


def service(request):
    if not hasattr(request.app.state, 'task_maintenance'):
        request.app.state.task_maintenance = TaskMaintenance(request.app.state.tasks, request.app.state.light)
    return request.app.state.task_maintenance


@router.get('')
def history(request: Request, status: str = 'all', kind: str = 'all',
            backend: Literal['all', 'legacy', 'light'] = 'all',
            page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100)):
    try:
        return service(request).list(status=status, kind=kind, backend=backend, page=page, page_size=page_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get('/overview')
def overview(request: Request, status: str = 'terminal', kind: str = 'all',
             backend: Literal['all','legacy','light'] = 'all',
             page: int = Query(default=1,ge=1), page_size: int = Query(default=20,ge=1,le=100)):
    try:
        return service(request).overview(status=status,kind=kind,backend=backend,page=page,page_size=page_size)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@router.post('/delete')
def delete(spec: DeleteSpec, request: Request):
    with request.app.state.compute_submit_lock:
        return service(request).delete_records([r.model_dump() for r in spec.tasks])


@router.post('/cleanup/preview')
def preview(spec: PreviewSpec, request: Request):
    with request.app.state.compute_submit_lock:
        return service(request).preview([r.model_dump() for r in spec.tasks], include_artifacts=spec.include_artifacts)


@router.post('/cleanup/execute')
def execute(spec: ExecuteSpec, request: Request):
    try:
        with request.app.state.compute_submit_lock:
            return service(request).execute(spec.plan_id, spec.plan_sha256)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='清理计划不存在') from exc


@router.post('/cleanup/recover')
def recover(spec: ExecuteSpec, request: Request):
    try:
        with request.app.state.compute_submit_lock:
            return service(request).recover(spec.plan_id, spec.plan_sha256)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
