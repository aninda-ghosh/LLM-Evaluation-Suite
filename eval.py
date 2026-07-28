#!/usr/bin/env python3
"""
CLI Evaluation Driver for LLM Benchmark Harness.
Delegates core execution to `eval_core` and benchmark scoring to `evaluators`.

Usage:
    python3 eval.py --model gemma-4-e2b --benchmark mmlu --batch_size 1 --debug
    python3 eval.py --model qwen3.5-2b --benchmark all --output results.json
"""

import sys
import argparse
import json
from pathlib import Path

# Core Infrastructure & Re-exports for Backward Compatibility with run_suite.py
from eval_core import (
    WORKSPACE_ROOT,
    resolve_model_path,
    load_models_config,
    discover_available_datasets,
    run_benchmark_eval,
    clear_gpu_vram
)

__all__ = [
    "resolve_model_path",
    "load_models_config",
    "discover_available_datasets",
    "run_benchmark_eval",
    "clear_gpu_vram",
]


def main():
    """CLI Argument Parser and Driver."""
    parser = argparse.ArgumentParser(description="Fully Dynamic LLM Benchmark Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Model key, alias, local folder path, or HF repo ID")
    parser.add_argument("--benchmark", type=str, default="all", help="Benchmark dataset name, json path, or 'all'")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for parallel inference (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to evaluate per benchmark (e.g. --limit 50)")
    parser.add_argument("--debug", action="store_true", help="Print complete prompt, think block, raw response, extracted answer, and score reasoning")
    parser.add_argument("--log_file", type=str, default=None, help="Output JSONL log path for per-sample logging")
    parser.add_argument("--output", type=str, default="results.json", help="Output results summary JSON path")

    args = parser.parse_args()

    model_target = resolve_model_path(args.model)
    print(f"Target Model: {model_target}")

    if args.benchmark == "all":
        benchmarks = discover_available_datasets()
        if not benchmarks:
            print("❌ No dataset JSON files found in ./data/. Run download_datasets.py first.")
            sys.exit(1)
    else:
        benchmarks = [args.benchmark]

    results = {}
    print(f"\n==================================================")
    print(f"EVALUATING MODEL: {args.model} ({model_target})")
    print(f"BENCHMARKS     : {', '.join(benchmarks).upper()}")
    print(f"BATCH SIZE     : {args.batch_size}")
    if args.limit:
        print(f"LIMIT          : {args.limit} samples/benchmark")
    print(f"DEBUG MODE     : {'ENABLED' if args.debug else 'DISABLED'}")
    print(f"==================================================")

    log_path = Path(args.log_file) if args.log_file else None

    for b_name in benchmarks:
        b_res = run_benchmark_eval(
            model_target,
            b_name,
            batch_size=args.batch_size,
            max_samples=args.limit,
            debug=args.debug,
            log_file=log_path
        )
        if b_res:
            results[b_name] = b_res

    if results:
        out_path = WORKSPACE_ROOT / args.output
        report = {
            "model_input": args.model,
            "resolved_model": model_target,
            "batch_size": args.batch_size,
            "results": results
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Evaluation complete. Saved summary report to '{out_path}'.\n")

    clear_gpu_vram()


if __name__ == "__main__":
    main()
