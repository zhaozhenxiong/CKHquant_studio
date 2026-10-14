"""Resume any checkpointed stage of an accelerated frozen training run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(resume_function=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run-id', '--resume-training-run-id', dest='run_id', required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--cache-path', type=Path)
    parser.add_argument('--end')
    parser.add_argument('--source-native-run-id')
    parser.add_argument('--cpu-workers', type=int)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--stock-batch-size', type=int)
    parser.add_argument('--calendar-run-id')
    parser.add_argument('--cohort-file', type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if resume_function is None:
        from my_strategy.services.czsc_training_acceleration import resume_fast_training
        resume_function = resume_fast_training
    result = resume_function(args.run_id, verify_only=args.verify_only, cache_path=args.cache_path,
        end=args.end, source_native_run_id=args.source_native_run_id, cpu_workers=args.cpu_workers,
        device=args.device, stock_batch_size=args.stock_batch_size, calendar_run_id=args.calendar_run_id,
        cohort=json.loads(args.cohort_file.read_text()) if args.cohort_file else None,
        config=json.loads(args.config.read_text()) if args.config else None)
    print(json.dumps({'run_id': result['run_id'], 'status': result.get('status'),
                      'publication_allowed': False, 'native_rebuilt': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
