#!/usr/bin/env python3
"""
Stage 1 Orchestrator: Multi-GPU Raw Inference.
Dispatches single evaluation jobs across GPUs dynamically.
"""

import sys
import argparse
import subprocess
import time
import yaml
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def get_available_gpu_ids() -> list[int]:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return [0]


def run_single_job(gpu_id: int, model_key: str, benchmark_name: str, batch_size: int, limit: int, debug: bool, log_path: Path) -> tuple:
    cmd = [
        sys.executable, "eval.py",
        "--model", model_key,
        "--benchmark", benchmark_name,
        "--gpu", str(gpu_id),
        "--batch_size", str(batch_size),
        "--log_file", str(log_path)
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if debug:
        cmd.append("--debug")

    print(f"[LAUNCH GPU {gpu_id}] Model: '{model_key}' | Benchmark: '{benchmark_name.upper()}'")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()

    if proc.returncode == 0:
        return model_key, benchmark_name, gpu_id, True, None
    err_msg = stderr[-200:] if stderr else "Process failed"
    return model_key, benchmark_name, gpu_id, False, err_msg


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Multi-GPU Raw Inference Orchestrator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Override inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Override sample limit")
    parser.add_argument("--preflight", nargs="?", const=4, type=int, help="Run preflight pass with N samples")
    parser.add_argument("--debug", action="store_true", help="Enable debug print outputs")
    args = parser.parse_args()

    config_path = WORKSPACE_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models_to_run = cfg.get("models", [])
    datasets_to_run = cfg.get("datasets", [])
    opts = cfg.get("options", {})

    batch_size = args.batch_size or opts.get("batch_size", 1)
    limit = args.preflight if args.preflight is not None else (args.limit or opts.get("limit", None))
    debug = True if args.preflight is not None else (args.debug or opts.get("debug", False))

    gpu_ids = get_available_gpu_ids()
    num_gpus = max(len(gpu_ids), 1)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = WORKSPACE_ROOT / opts.get("output_dir", "outputs") / f"run_{timestamp_str}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_queue = [(m, b) for m in models_to_run for b in datasets_to_run]
    total_jobs = len(job_queue)

    print(f"\n==================================================")
    print(f"STAGE 1: MULTI-GPU RAW INFERENCE ({total_jobs} JOBS ON {num_gpus} GPUs)")
    print(f"Output Dir: {run_dir}")
    print(f"==================================================\n")

    available_gpus = list(gpu_ids)
    pending_jobs = {}
    completed_jobs = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        while available_gpus and job_queue:
            gpu_id = available_gpus.pop(0)
            m_key, b_name = job_queue.pop(0)
            log_path = logs_dir / f"{m_key}_{b_name}.jsonl"
            fut = executor.submit(run_single_job, gpu_id, m_key, b_name, batch_size, limit, debug, log_path)
            pending_jobs[fut] = (m_key, b_name, gpu_id)

        while pending_jobs:
            done_futures, _ = wait(list(pending_jobs.keys()), return_when=FIRST_COMPLETED)
            for fut in done_futures:
                m_key, b_name, freed_gpu = pending_jobs.pop(fut)
                _, _, _, ok, err = fut.result()
                completed_jobs += 1

                status = "[OK]" if ok else "[ERROR]"
                print(f"{status} Job {completed_jobs}/{total_jobs} on GPU {freed_gpu}: '{m_key}' @ '{b_name}'")
                if not ok:
                    print(f"   Reason: {err}")

                available_gpus.append(freed_gpu)
                if job_queue:
                    next_gpu = available_gpus.pop(0)
                    next_m, next_b = job_queue.pop(0)
                    next_log = logs_dir / f"{next_m}_{next_b}.jsonl"
                    next_fut = executor.submit(run_single_job, next_gpu, next_m, next_b, batch_size, limit, debug, next_log)
                    pending_jobs[next_fut] = (next_m, next_b, next_gpu)

    elapsed = round(time.time() - t0, 2)
    print(f"\n[STAGE 1 COMPLETE] {completed_jobs} jobs finished in {elapsed}s.")
    print(f"\nTo run Stage 2 scoring and generate reports, execute:")
    print(f"  python3 score_suite.py --run_dir {run_dir}\n")


if __name__ == "__main__":
    main()
