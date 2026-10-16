"""Execution-only resource selection and measured forecast device accounting."""
from __future__ import annotations

import os
from copy import deepcopy


def source_workers(requested=None, *, population=None):
    """Reserve coordinator capacity; never fan one stock out more than once."""
    if requested is None:
        workers = min(8, max(1, (os.cpu_count() or 2) - 2))
    else:
        if isinstance(requested, bool) or int(requested) != requested or not 1 <= requested <= 12:
            raise ValueError('CPU preparation workers must be an integer from 1 to 12')
        workers = int(requested)
    return min(workers, max(1, population)) if population is not None else workers


class PredictionExecution:
    """Count only tensor-backed, synchronized calls made by this invocation."""
    def __init__(self):
        self.value = dict(fresh_calls=0, reused_calls=0, execution_verified_calls=0,
            actual_devices={}, cached_provenance_devices={})

    def add(self, forecast):
        resource = (forecast or {}).get('resource') or {}
        execution = resource.get('execution') or {}
        device = execution.get('actual_device')
        if (forecast or {}).get('reused') is True:
            self.value['reused_calls'] += 1
            if isinstance(device, str) and device:
                values = self.value['cached_provenance_devices']
                values[device] = values.get(device, 0) + 1
            return
        self.value['fresh_calls'] += 1
        if not (execution.get('schema') == 'kronos_execution_v1'
                and execution.get('fresh_inference') is True
                and execution.get('tensor_devices_verified') is True
                and execution.get('synchronization_completed') is True
                and execution.get('inference_calls') == 1 and device in {'cpu', 'mps'} | {
                    f'cuda:{index}' for index in range(256)}):
            return
        groups = [execution.get(key) for key in
            ('model_parameter_devices', 'tokenizer_parameter_devices', 'input_tensor_devices')]
        if not all(isinstance(group, list) and group and set(group) == {device} for group in groups):
            return
        self.value['execution_verified_calls'] += 1
        values = self.value['actual_devices']
        values[device] = values.get(device, 0) + 1

    def report(self):
        return deepcopy(self.value)
