#!/usr/bin/env python3
"""Run separate immutable training-pattern / existing-head diagnostics."""
import argparse
import json

from my_strategy.services.czsc_operation_pattern_diagnostics import run_operation_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--cutoff", default="2026-06-30")
    parser.add_argument("--no-probabilities", action="store_true")
    args = parser.parse_args()
    def progress(stage, completed, requested):
        print(json.dumps({"stage": stage, "completed": completed, "requested": requested}), flush=True)
    result = run_operation_diagnostics(args.model_run_id, cutoff=args.cutoff,
        include_probabilities=not args.no_probabilities, progress=progress)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"], "coverage": result["coverage"],
        "pending_model_heads": len(result["pending_model_heads"]),
        "publication_allowed": result["publication_allowed"]}), flush=True)


if __name__ == "__main__":
    main()
