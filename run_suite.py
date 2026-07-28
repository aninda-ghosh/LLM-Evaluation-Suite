#!/usr/bin/env python3
"""
Automated Multi-GPU LLM Benchmark Suite Orchestrator with JSONL Per-Sample Logging.

Reads evaluation matrix and GPU settings from `config.yaml` and `models.json`.
Dispatches tasks dynamically across available GPUs and stores per-sample JSONL logs
inside `outputs/run_YYYYMMDD_HHMMSS/logs/<model>_<benchmark>.jsonl`.

Usage:
    python3 run_suite.py
    python3 run_suite.py --limit 10
"""

import os
import sys
import signal
import argparse
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# Workspace Root Setup
WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from eval import (
    resolve_model_path,
    load_models_config,
    discover_available_datasets
)

ACTIVE_SUBPROCESSES = []


def purge_stale_eval_processes():
    """Kills any orphaned background eval.py processes from previous runs to free GPU VRAM."""
    current_pid = os.getpid()
    try:
        res = subprocess.run(["pgrep", "-f", "eval.py"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            pids = res.stdout.strip().split()
            for pid_str in pids:
                pid = int(pid_str)
                if pid != current_pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        print(f"[CLEANUP] Purged stale eval process (PID {pid}) to free VRAM.")
                    except OSError:
                        pass
    except Exception:
        pass


def register_active_process(proc):
    """Registers a subprocess for emergency termination on exit."""
    ACTIVE_SUBPROCESSES.append(proc)


def unregister_active_process(proc):
    """Unregisters a completed subprocess."""
    if proc in ACTIVE_SUBPROCESSES:
        ACTIVE_SUBPROCESSES.remove(proc)


def emergency_cleanup(signum=None, frame=None):
    """Kills all active child subprocesses cleanly upon Ctrl+C / SIGINT."""
    print("\n[INTERRUPT] SIGINT / Ctrl+C detected. Terminating all running evaluation processes...")
    for proc in list(ACTIVE_SUBPROCESSES):
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    sys.exit(130)


signal.signal(signal.SIGINT, emergency_cleanup)
signal.signal(signal.SIGTERM, emergency_cleanup)


def get_available_gpu_ids() -> list:
    """Detects available CUDA GPU IDs on the system."""
    try:
        import torch
        if torch.cuda.is_available():
            cnt = torch.cuda.device_count()
            if cnt > 0:
                return list(range(cnt))
    except Exception:
        pass
    return [0]


def load_yaml_config(config_path: Path) -> dict:
    """Loads YAML configuration file safely with PyYAML or fallback."""
    if not config_path.exists():
        print(f"[ERROR] Config file '{config_path}' not found.")
        sys.exit(1)

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        cfg = {"models": [], "datasets": [], "options": {}}
        current_section = None
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if line_str.endswith(":"):
                    current_section = line_str[:-1]
                elif line_str.startswith("- "):
                    val = line_str[2:].strip()
                    if current_section in cfg:
                        cfg[current_section].append(val)
                elif ":" in line_str and current_section == "options":
                    k, v = line_str.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if v in ["null", "None"]:
                        cfg["options"][k] = None
                    elif v.isdigit():
                        cfg["options"][k] = int(v)
                    elif v.lower() in ["true", "false"]:
                        cfg["options"][k] = (v.lower() == "true")
                    else:
                        cfg["options"][k] = v
        return cfg


def resolve_gpu_configuration(gpu_cfg) -> list:
    """Resolves GPU configuration from config.yaml option (auto, int, or list)."""
    system_gpus = get_available_gpu_ids()

    if gpu_cfg is None or gpu_cfg == "auto" or str(gpu_cfg).lower() == "auto":
        return system_gpus

    if isinstance(gpu_cfg, int):
        return list(range(min(gpu_cfg, len(system_gpus))))

    if isinstance(gpu_cfg, list):
        return [int(g) for g in gpu_cfg if int(g) in system_gpus]

    return system_gpus


def resolve_model_batch_size(model_key: str, global_batch_size: int, config_overrides: dict) -> int:
    """Resolves model-specific batch size from config.yaml or models.json."""
    if config_overrides and model_key in config_overrides:
        return config_overrides[model_key]

    models_cfg = load_models_config()
    if model_key in models_cfg and "batch_size" in models_cfg[model_key]:
        return models_cfg[model_key]["batch_size"]

    return global_batch_size


def run_eval_task(gpu_id: int, model_key: str, benchmark_name: str, batch_size: int, limit: int, debug: bool, log_file_path: Path) -> tuple:
    """Runs a single (model, benchmark) evaluation subprocess pinned strictly to a specific GPU with JSONL logging."""
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    tmp_out = WORKSPACE_ROOT / ".cache" / f"eval_{model_key}_{benchmark_name}_gpu{gpu_id}.json"

    cmd = [
        sys.executable, "eval.py",
        "--model", model_key,
        "--benchmark", benchmark_name,
        "--batch_size", str(batch_size),
        "--output", str(tmp_out),
        "--log_file", str(log_file_path)
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if debug:
        cmd.append("--debug")

    print(f"[LAUNCH] [GPU {gpu_id}] Model: {model_key} | Benchmark: {benchmark_name.upper()} | Log: {log_file_path.name}")

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    register_active_process(proc)

    try:
        stdout, stderr = proc.communicate()
    except Exception:
        proc.kill()
        unregister_active_process(proc)
        return model_key, benchmark_name, gpu_id, None, "Process interrupted"

    unregister_active_process(proc)

    if proc.returncode == 0 and tmp_out.exists():
        try:
            with open(tmp_out, "r") as f:
                data = json.load(f)
            tmp_out.unlink(missing_ok=True)
            b_data = data.get("results", {}).get(benchmark_name)
            return model_key, benchmark_name, gpu_id, b_data, None
        except Exception as e:
            return model_key, benchmark_name, gpu_id, None, str(e)
    else:
        err_msg = stderr[-300:] if stderr else "Subprocess failed"
        return model_key, benchmark_name, gpu_id, None, err_msg


def update_readme_leaderboard(suite_summary: dict):
    """Updates the model leaderboard table in README.md with latest suite results."""
    readme_path = WORKSPACE_ROOT / "README.md"
    if not readme_path.exists():
        return

    content = readme_path.read_text(encoding="utf-8")
    table_lines = [
        "| Model Key | Model Repository | Engine | MMLU | GSM8K | HellaSwag | TruthfulQA | Avg Latency (ms) | Avg Speed (tok/s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    models_cfg = load_models_config()

    for m_key, m_data in suite_summary.get("models", {}).items():
        repo = models_cfg.get(m_key, {}).get("hf_repo", m_key)
        res = m_data.get("results", {})

        mmlu_acc = f"{res['mmlu']['accuracy_pct']}%" if "mmlu" in res else "-"
        gsm_acc = f"{res['gsm8k']['accuracy_pct']}%" if "gsm8k" in res else "-"
        hs_acc = f"{res['hellaswag']['accuracy_pct']}%" if "hellaswag" in res else "-"
        tqa_acc = f"{res['truthfulqa']['accuracy_pct']}%" if "truthfulqa" in res else "-"

        latencies = [b["avg_sample_latency_ms"] for b in res.values() if isinstance(b, dict) and "avg_sample_latency_ms" in b]
        speeds = [b["avg_aggregate_tps"] for b in res.values() if isinstance(b, dict) and "avg_aggregate_tps" in b]

        avg_lat = f"{sum(latencies)/len(latencies):.1f}" if latencies else "-"
        avg_spd = f"{sum(speeds)/len(speeds):.1f}" if speeds else "-"

        table_lines.append(f"| `{m_key}` | `{repo}` | HF | {mmlu_acc} | {gsm_acc} | {hs_acc} | {tqa_acc} | {avg_lat} | {avg_spd} |")

    new_table_str = "\n".join(table_lines)

    if "| Model Key | Model Repository | Engine |" in content:
        start_idx = content.find("| Model Key | Model Repository | Engine |")
        end_idx = content.find("---", start_idx)
        if end_idx != -1:
            updated_content = content[:start_idx] + new_table_str + "\n\n" + content[end_idx:]
            readme_path.write_text(updated_content, encoding="utf-8")
            print("[INFO] Updated README.md leaderboard table.")


def generate_markdown_report(suite_summary: dict, report_path: Path):
    """Generates a detailed Markdown report with per-benchmark model comparison tables."""
    timestamp = suite_summary.get("timestamp")
    models_dict = suite_summary.get("models", {})

    # Collect all unique benchmarks evaluated across models
    benchmarks_set = set()
    for m_data in models_dict.values():
        benchmarks_set.update(m_data.get("results", {}).keys())

    # Preferred benchmark display order
    preferred_order = ["gsm8k", "mmlu", "hellaswag", "truthfulqa"]
    benchmarks_list = [b for b in preferred_order if b in benchmarks_set]
    for b in sorted(benchmarks_set):
        if b not in benchmarks_list:
            benchmarks_list.append(b)

    lines = [
        f"# Multi-GPU LLM Benchmark Evaluation Suite Report",
        f"**Run Timestamp:** `{timestamp}`  ",
        f"**GPUs Utilized:** `{suite_summary.get('gpus_used')}`  ",
        f"**Batch Size Default:** `{suite_summary.get('options', {}).get('batch_size')}`  ",
        f"**Sample Limit:** `{suite_summary.get('options', {}).get('limit') or 'Full Dataset'}`  \n",
        "---",
        "## Overall Cross-Benchmark Comparison Matrix\n",
        "| Model Key | GSM8K | MMLU | HellaSwag | TruthfulQA | Avg Latency (ms) | Aggregate Speed (tok/s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for m_key, m_data in models_dict.items():
        res = m_data.get("results", {})
        gsm = f"{res['gsm8k']['accuracy_pct']}%" if "gsm8k" in res else "-"
        mmlu = f"{res['mmlu']['accuracy_pct']}%" if "mmlu" in res else "-"
        hs = f"{res['hellaswag']['accuracy_pct']}%" if "hellaswag" in res else "-"
        tqa = f"{res['truthfulqa']['accuracy_pct']}%" if "truthfulqa" in res else "-"

        lats = [b["avg_sample_latency_ms"] for b in res.values() if isinstance(b, dict) and "avg_sample_latency_ms" in b]
        spds = [b["avg_aggregate_tps"] for b in res.values() if isinstance(b, dict) and "avg_aggregate_tps" in b]

        avg_lat = f"{sum(lats)/len(lats):.2f}ms" if lats else "-"
        avg_spd = f"{sum(spds)/len(spds):.2f} tok/s" if spds else "-"

        lines.append(f"| `{m_key}` | {gsm} | {mmlu} | {hs} | {tqa} | {avg_lat} | {avg_spd} |")

    lines.append("\n---")
    lines.append("## Per-Benchmark Model Comparisons\n")

    for b_name in benchmarks_list:
        lines.append(f"### {b_name.upper()} Benchmark Comparison\n")
        lines.append("| Model Key | Accuracy (%) | Passed / Total | Avg Latency (ms) | Aggregate Speed (tok/s) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")

        b_rows = []
        for m_key, m_data in models_dict.items():
            b_res = m_data.get("results", {}).get(b_name)
            if isinstance(b_res, dict):
                b_rows.append((m_key, b_res))

        # Sort models by accuracy descending for this benchmark
        b_rows.sort(key=lambda x: x[1]['accuracy_pct'], reverse=True)

        for m_key, b_res in b_rows:
            lines.append(
                f"| `{m_key}` | **{b_res['accuracy_pct']}%** | {b_res['passed']} / {b_res['total']} | "
                f"{b_res['avg_sample_latency_ms']}ms | {b_res['avg_aggregate_tps']} tok/s |"
            )
        lines.append("")

    lines.append("---\n*Report auto-generated by `run_suite.py`*")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Automated LLM Benchmark Suite Orchestrator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    parser.add_argument("--limit", type=int, default=None, help="Override sample limit option (e.g. --limit 10)")
    parser.add_argument("--preflight", nargs="?", const=10, type=int, help="Run a quick preflight verification pass with N samples (default: 10) per model/benchmark")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging per sample")

    args = parser.parse_args()

    purge_stale_eval_processes()

    config_path = WORKSPACE_ROOT / args.config
    cfg = load_yaml_config(config_path)

    models_to_run = cfg.get("models", [])
    datasets_to_run = cfg.get("datasets", [])
    opts = cfg.get("options", {})

    global_batch_size = opts.get("batch_size", 1)
    batch_overrides = opts.get("batch_size_overrides", {})
    
    if args.preflight is not None:
        limit = args.preflight
    elif args.limit is not None:
        limit = args.limit
    else:
        limit = opts.get("limit", None)

    debug = args.debug or opts.get("debug", False)
    base_out_dir = WORKSPACE_ROOT / opts.get("output_dir", "outputs")

    gpu_ids = resolve_gpu_configuration(opts.get("gpus", "auto"))

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_out_dir / f"run_{timestamp_str}"
    logs_dir = run_output_dir / "logs"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mode_label = f"PREFLIGHT MODE ({limit} samples/benchmark)" if args.preflight is not None else (f"{limit} samples/benchmark" if limit else "FULL DATASET")

    print(f"\n==================================================")
    print(f"STARTING MULTI-GPU BENCHMARK SUITE")
    print(f"Timestamp   : {timestamp_str}")
    print(f"Output Dir  : {run_output_dir}")
    print(f"Logs Dir    : {logs_dir}")
    print(f"GPUs Active : {gpu_ids} ({len(gpu_ids)} GPUs)")
    print(f"Models      : {', '.join(models_to_run)}")
    print(f"Datasets    : {', '.join(datasets_to_run)}")
    print(f"Batch Size  : Default: {global_batch_size}")
    print(f"Run Mode    : {mode_label}")
    print(f"==================================================\n")

    task_queue = []
    for m_key in models_to_run:
        for b_name in datasets_to_run:
            m_batch_size = resolve_model_batch_size(m_key, global_batch_size, batch_overrides)
            log_path = logs_dir / f"{m_key}_{b_name}.jsonl"
            task_queue.append((m_key, b_name, m_batch_size, log_path))

    suite_summary = {
        "timestamp": timestamp_str,
        "gpus_used": gpu_ids,
        "options": {
            "batch_size": global_batch_size,
            "limit": limit,
            "preflight": args.preflight is not None,
            "debug": debug
        },
        "models": {m: {"resolved_path": resolve_model_path(m), "results": {}} for m in models_to_run}
    }

    start_suite_time = time.time()
    num_workers = max(len(gpu_ids), 1)

    available_gpus = list(gpu_ids)
    pending_futures = {}

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while available_gpus and task_queue:
                gpu_to_use = available_gpus.pop(0)
                m_key, b_name, m_batch_size, log_path = task_queue.pop(0)
                fut = executor.submit(
                    run_eval_task,
                    gpu_id=gpu_to_use,
                    model_key=m_key,
                    benchmark_name=b_name,
                    batch_size=m_batch_size,
                    limit=limit,
                    debug=debug,
                    log_file_path=log_path
                )
                pending_futures[fut] = (m_key, b_name, gpu_to_use)

            while pending_futures:
                done_futures, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)

                for fut in done_futures:
                    m_key, b_name, gpu_used = pending_futures.pop(fut)
                    m_res_key, b_res_name, freed_gpu, b_data, err = fut.result()

                    if err:
                        print(f"[ERROR] [GPU {freed_gpu}] Failed: {m_key} on {b_name}: {err}")
                    elif b_data:
                        print(f"[SUCCESS] [GPU {freed_gpu}] Completed: {m_key} on {b_name.upper()} | Accuracy: {b_data['accuracy_pct']}%")
                        suite_summary["models"][m_key]["results"][b_name] = b_data

                    available_gpus.append(freed_gpu)

                    if task_queue:
                        next_gpu = available_gpus.pop(0)
                        next_m_key, next_b_name, next_m_batch_size, next_log_path = task_queue.pop(0)
                        next_fut = executor.submit(
                            run_eval_task,
                            gpu_id=next_gpu,
                            model_key=next_m_key,
                            benchmark_name=next_b_name,
                            batch_size=next_m_batch_size,
                            limit=limit,
                            debug=debug,
                            log_file_path=next_log_path
                        )
                        pending_futures[next_fut] = (next_m_key, next_b_name, next_gpu)

    except KeyboardInterrupt:
        emergency_cleanup()

    total_suite_sec = round(time.time() - start_suite_time, 2)
    suite_summary["total_suite_duration_sec"] = total_suite_sec

    summary_json_path = run_output_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    report_md_path = run_output_dir / "report.md"
    generate_markdown_report(suite_summary, report_md_path)

    update_readme_leaderboard(suite_summary)

    print(f"\n==================================================")
    print(f"[SUCCESS] MULTI-GPU BENCHMARK SUITE COMPLETE IN {total_suite_sec} seconds")
    print(f"Results stored in: {run_output_dir}")
    print(f"  - JSON Summary : {summary_json_path}")
    print(f"  - Report MD    : {report_md_path}")
    print(f"  - JSONL Logs   : {logs_dir}")
    print(f"==================================================\n")


if __name__ == "__main__":
    main()
