#!/usr/bin/env python3
"""Verify and continue one interrupted trend training run's frozen stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run-id', '--resume-training-run-id', dest='training_run_id', required=True)
    parser.add_argument('--verify-only', action='store_true', help='Read-only frozen asset verification; do not start training')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], help='Must match the original fitted head device')
    parser.add_argument('--cpu-workers', type=int, help='Must match the original frozen parameter')
    parser.add_argument('--source-native-run-id')
    parser.add_argument('--end')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--stock-batch-size', type=int)
    parser.add_argument('--cohort-file', type=Path)
    parser.add_argument('--calendar-run-id')
    args = parser.parse_args()
    from my_strategy.services.czsc_trend_resume import inspect_resume, run_resume
    request = dict(source_native_run_id=args.source_native_run_id, end=args.end, device=args.device,
                   cpu_workers=args.cpu_workers, stock_batch_size=args.stock_batch_size,
                   calendar_run_id=args.calendar_run_id,
                   config=json.loads(args.config.read_text()) if args.config else None,
                   cohort=json.loads(args.cohort_file.read_text()) if args.cohort_file else None)
    if args.verify_only:
        result = inspect_resume(args.training_run_id, **request)
    else:
        report = run_resume(args.training_run_id, **request)
        result = dict(run_id=report['run_id'], status=report['status'], resume_provenance=report['resume_provenance'],
                      publication_allowed=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
