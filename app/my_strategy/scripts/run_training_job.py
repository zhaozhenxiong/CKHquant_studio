"""Own a persisted training job without restarting a busy dashboard server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import signal
import time


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', value):
        raise argparse.ArgumentTypeError('invalid task or run identifier')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-id', required=True, type=identifier)
    parser.add_argument('--parameters-file', type=Path)
    parser.add_argument('--resume-training-run-id', type=identifier)
    parser.add_argument('--resume-from-job-id', type=identifier)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args(argv)
    from my_strategy.web_dashboard.tasks import TaskManager, ACTIVE
    manager = TaskManager(workers=1)
    spec = {'plan_id': args.plan_id}
    if args.parameters_file:
        spec['parameters'] = json.loads(args.parameters_file.read_text())
    for key in ('resume_training_run_id', 'resume_from_job_id'):
        if getattr(args, key): spec[key] = getattr(args, key)
    # Validate before enqueueing; execution repeats the causal source checks
    # under the normal resource lease and claims this plan exactly once.
    manager.training.validate_plan(args.plan_id, spec.get('parameters'))
    job = manager.submit('model_training', spec)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    print(json.dumps({'event': 'training_job', **job}, ensure_ascii=False), flush=True)
    previous = {}
    def cancel(_number, _frame): manager.cancel(job['job_id'])
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, cancel)
    try:
        last = None
        while True:
            current = manager.get(job['job_id'])
            marker = (current['status'], current.get('progress_detail'))
            if marker != last:
                print(json.dumps({'event': 'training_job_progress', 'job_id': job['job_id'],
                    'status': current['status'], 'progress_detail': current.get('progress_detail'),
                    'run_id': current.get('run_id'), 'error': current.get('error')}, ensure_ascii=False), flush=True)
                last = marker
            if current['status'] not in ACTIVE:
                if args.receipt: args.receipt.write_text(json.dumps(current, ensure_ascii=False, indent=2))
                return 0 if current['status'] == 'succeeded' else 130 if current['status'] == 'cancelled' else 2
            time.sleep(1)
    finally:
        for number, handler in previous.items(): signal.signal(number, handler)
        manager.close()


if __name__ == '__main__': raise SystemExit(main())
