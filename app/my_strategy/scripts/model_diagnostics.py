"""Run frozen-model sample diagnostics in a separately cancellable process."""
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-id",required=True)
    parser.add_argument("--cpu-workers",type=int,default=2)
    args=parser.parse_args()
    from my_strategy.services.czsc_high_return_sample_diagnostics import run_diagnostics
    def progress(fold,current,total):
        print("DIAGNOSTIC_PROGRESS "+json.dumps(dict(stage="diagnostics",message=fold,fold=fold,
            scope="current_diagnostic_fold_stocks",completed=current,total=total)),flush=True)
    print("STAGE diagnostics",flush=True)
    report=run_diagnostics(args.model_run_id,cpu_workers=args.cpu_workers,progress=progress)
    print("DIAGNOSTIC_RUN_ID "+report["run_id"],flush=True)


if __name__=="__main__":
    main()
