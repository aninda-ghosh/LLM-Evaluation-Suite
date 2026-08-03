#!/usr/bin/env python3
"""
Stage 2: Response Extraction, Evaluation & Report Generator.
Reads raw JSONL outputs from Stage 1, extracts answers and scores model generations
using benchmark-tailored evaluators (`evaluators/`), creates consolidated reports,
and generates model comparison charts per benchmark.
"""

import os
import sys
import json
import csv
import argparse
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluators import get_evaluator, _EVALUATORS

CSV_FIELDS = [
    "run_timestamp",
    "model",
    "benchmark",
    "accuracy_pct",
    "passed",
    "failed",
    "did_not_finish",
    "total",
    "avg_latency_ms",
    "avg_tps",
]


def parse_log_filename(stem: str) -> tuple[str, str]:
    """Parses model_key and benchmark_name from log filename (e.g. gemma-3-1b-it_mmlu_pro.jsonl)."""
    known = sorted(_EVALUATORS.keys(), key=len, reverse=True)
    for bname in known:
        suffix = f"_{bname}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], bname
    parts = stem.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (stem, stem)


def get_expected_benchmark_total(b_name: str) -> int:
    """Returns total expected dataset items for benchmark from data/<b_name>.json if available."""
    data_file = WORKSPACE_ROOT / "data" / f"{b_name}.json"
    if data_file.exists():
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                return len(json.load(f))
        except Exception:
            pass
    return 0


def score_jsonl_log(log_file: Path) -> tuple[dict, list[dict]]:
    """Reads raw JSONL file, extracts & scores responses with benchmark-tailored evaluator."""
    model_key, b_name = parse_log_filename(log_file.stem)
    evaluator = get_evaluator(b_name)

    enriched_entries = []
    passed_cnt, failed_cnt, did_not_finish_cnt = 0, 0, 0
    latencies, speeds = [], []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line.strip())
            dataset_item = entry.get("dataset_item", {})
            model_output = entry.get("model_output", "")

            # Benchmark-tailored extraction & scoring
            think_out = evaluator.extract_think(model_output) if hasattr(evaluator, "extract_think") else "N/A"
            clean_out = evaluator.extract_answer(model_output, dataset_item) if hasattr(evaluator, "extract_answer") else model_output
            expected_ans = evaluator.get_expected_answer(dataset_item) if hasattr(evaluator, "get_expected_answer") else str(dataset_item.get("answer", ""))
            is_pass, pred_val, score_reason = evaluator.score_item(dataset_item, model_output)

            # Categorize status: passed (True), failed (False), or did_not_finish
            is_empty = not model_output.strip()
            is_truncated = entry.get("finish_reason") == "length" or entry.get("truncated", False)
            no_answer = not pred_val or str(pred_val).strip() == ""

            if is_pass:
                status = "passed"
                passed_cnt += 1
            elif is_empty or (no_answer and is_truncated):
                status = "did_not_finish"
                did_not_finish_cnt += 1
            else:
                status = "failed"
                failed_cnt += 1

            latencies.append(entry.get("sample_latency_ms", 0.0))
            speeds.append(entry.get("aggregate_tps", 0.0))

            entry.update({
                "expected_answer": expected_ans,
                "think_reasoning": think_out,
                "extracted_answer": clean_out,
                "predicted_answer": pred_val,
                "status": status,
                "passed": is_pass,
                "score_reason": score_reason
            })
            enriched_entries.append(entry)

    with open(log_file, "w", encoding="utf-8") as f:
        for entry in enriched_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Account for un-evaluated / missing items if generation stopped early
    evaluated_cnt = len(enriched_entries)
    expected_dataset_total = get_expected_benchmark_total(b_name)
    total_benchmark_items = max(expected_dataset_total, evaluated_cnt)

    missing_items = max(total_benchmark_items - evaluated_cnt, 0)
    did_not_finish_cnt += missing_items

    accuracy = round((passed_cnt / total_benchmark_items * 100.0), 2) if total_benchmark_items > 0 else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else 0.0

    metrics = {
        "model": model_key,
        "benchmark": b_name,
        "accuracy_pct": accuracy,
        "passed": passed_cnt,
        "failed": failed_cnt,
        "did_not_finish": did_not_finish_cnt,
        "total": total_benchmark_items,
        "avg_sample_latency_ms": avg_latency,
        "avg_aggregate_tps": avg_speed
    }
    return metrics, enriched_entries


def generate_benchmark_charts(suite_summary: dict, run_dir: Path):
    """Generates PNG comparison charts for each benchmark dataset."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARNING] matplotlib not installed. Skipping chart generation.")
        return

    models_dict = suite_summary.get("models", {})
    benchmarks = {}

    for m_key, m_data in models_dict.items():
        for b_name, b_res in m_data.get("results", {}).items():
            if b_name not in benchmarks:
                benchmarks[b_name] = []
            benchmarks[b_name].append({
                "model": m_key,
                "accuracy_pct": b_res.get("accuracy_pct", 0.0),
                "passed": b_res.get("passed", 0),
                "failed": b_res.get("failed", 0),
                "did_not_finish": b_res.get("did_not_finish", 0),
                "total": b_res.get("total", 0),
                "avg_latency_ms": b_res.get("avg_sample_latency_ms", 0.0),
                "avg_tps": b_res.get("avg_aggregate_tps", 0.0)
            })

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")

    for b_name, b_models in benchmarks.items():
        # Sort models by accuracy descending
        b_models.sort(key=lambda x: x["accuracy_pct"], reverse=True)

        model_names = [m["model"] for m in b_models]
        passed_counts = [m["passed"] for m in b_models]
        failed_counts = [m["failed"] for m in b_models]
        dnf_counts = [m["did_not_finish"] for m in b_models]
        accuracies = [m["accuracy_pct"] for m in b_models]

        fig, ax = plt.subplots(figsize=(10, max(5, len(model_names) * 0.6)))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fa')

        y_pos = np.arange(len(model_names))
        height = 0.55

        ax.barh(y_pos, passed_counts, height, label="Passed (True)", color="#27ae60", edgecolor="none")
        ax.barh(y_pos, failed_counts, height, left=passed_counts, label="Failed (False)", color="#e74c3c", edgecolor="none")

        left_dnf = [p + f for p, f in zip(passed_counts, failed_counts)]
        ax.barh(y_pos, dnf_counts, height, left=left_dnf, label="Didn't Finish", color="#f39c12", edgecolor="none")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=11, fontweight="bold", color="#2c3e50")
        ax.invert_yaxis()  # top accuracy at top
        ax.set_xlabel("Sample Count", fontsize=12, fontweight="bold", labelpad=10, color="#2c3e50")
        ax.set_title(f"Benchmark Comparison: {b_name.upper()}", fontsize=14, fontweight="bold", pad=35, color="#2c3e50")

        max_total = max([left_dnf[i] + dnf_counts[i] for i in range(len(model_names))]) if model_names else 100
        ax.set_xlim(0, max_total * 1.14)

        for i, acc in enumerate(accuracies):
            total_len = left_dnf[i] + dnf_counts[i]
            offset = max(max_total * 0.015, 1)
            ax.text(total_len + offset, i, f"{acc:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold", color="#1b6ec2")

        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=10.5, labelcolor="#2c3e50")
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cccccc")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(colors='#2c3e50', labelsize=10)

        plt.tight_layout()
        chart_path = charts_dir / f"{b_name}_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

    print(f"[CHARTS] Saved benchmark comparison charts to '{charts_dir}'.")


def write_reports(suite_summary: dict, run_dir: Path):
    """Generates report.md and separate per-benchmark CSV results from suite summary."""
    timestamp = suite_summary.get("timestamp", "")
    models_dict = suite_summary.get("models", {})
    rows = []

    for m_key, m_data in models_dict.items():
        for b_name, b_res in m_data.get("results", {}).items():
            if isinstance(b_res, dict):
                rows.append({
                    "run_timestamp": timestamp,
                    "model": m_key,
                    "benchmark": b_name,
                    "accuracy_pct": b_res.get("accuracy_pct"),
                    "passed": b_res.get("passed", 0),
                    "failed": b_res.get("failed", 0),
                    "did_not_finish": b_res.get("did_not_finish", 0),
                    "total": b_res.get("total", 0),
                    "avg_latency_ms": b_res.get("avg_sample_latency_ms"),
                    "avg_tps": b_res.get("avg_aggregate_tps"),
                })

    # Group rows by benchmark for per-benchmark CSV writing & reporting
    benchmarks_data = {}
    for r in rows:
        b = r["benchmark"]
        if b not in benchmarks_data:
            benchmarks_data[b] = []
        benchmarks_data[b].append(r)

    # Remove single combined results.csv if it exists
    old_csv = run_dir / "results.csv"
    if old_csv.exists():
        old_csv.unlink()

    # Write separate CSV file for each benchmark comparing all models
    for b_name, b_rows in sorted(benchmarks_data.items()):
        b_rows.sort(key=lambda x: x["accuracy_pct"], reverse=True)

        per_bench_csv = run_dir / f"results_{b_name}.csv"
        with open(per_bench_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(b_rows)

    # Generate per-benchmark comparison charts
    generate_benchmark_charts(suite_summary, run_dir)

    lines = [
        "# Evaluation Suite Report",
        f"**Timestamp:** `{timestamp}`\n",
        "---\n"
    ]

    for b_name, b_rows in sorted(benchmarks_data.items()):
        b_rows.sort(key=lambda x: x["accuracy_pct"], reverse=True)
        chart_rel_path = f"charts/{b_name}_comparison.png"

        lines.append(f"## {b_name.upper()} Benchmark\n")
        if (run_dir / chart_rel_path).exists():
            lines.append(f"![{b_name.upper()} Model Comparison]({chart_rel_path})\n")

        lines.append("| Model | Accuracy (%) | Passed (True) | Failed (False) | Didn't Finish | Total | Avg Latency | Speed |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in b_rows:
            lines.append(f"| `{r['model']}` | **{r['accuracy_pct']}%** | {r['passed']} | {r['failed']} | {r['did_not_finish']} | {r['total']} | {r['avg_latency_ms']}ms | {r['avg_tps']} tok/s |")
        lines.append("\n---\n")

    report_content = "\n".join(lines)
    (run_dir / "report.md").write_text(report_content, encoding="utf-8")

    # Sync latest report and chart assets to docs/ for Git documentation
    docs_dir = WORKSPACE_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_charts_dir = docs_dir / "charts"
    docs_charts_dir.mkdir(parents=True, exist_ok=True)

    run_charts_dir = run_dir / "charts"
    if run_charts_dir.exists():
        for chart_file in run_charts_dir.glob("*.png"):
            shutil.copy2(chart_file, docs_charts_dir / chart_file.name)

    (docs_dir / "report.md").write_text(report_content, encoding="utf-8")
    print(f"[DOCS] Synced latest report.md and charts to '{docs_dir}'.")


def process_run_directory(run_dir: Path) -> dict:
    logs_dir = run_dir / "logs"
    log_files = sorted(logs_dir.glob("*.jsonl")) if logs_dir.exists() else []
    if not log_files:
        print(f"[ERROR] No JSONL log files in '{logs_dir}'.")
        sys.exit(1)

    print(f"\n[STAGE 2] Scoring raw logs in '{run_dir}' ({len(log_files)} tasks)...")
    suite_summary = {"timestamp": run_dir.name.replace("run_", ""), "models": {}}

    for log_file in log_files:
        metrics, _ = score_jsonl_log(log_file)
        m_key, b_name = metrics["model"], metrics["benchmark"]
        if m_key not in suite_summary["models"]:
            suite_summary["models"][m_key] = {"results": {}}
        suite_summary["models"][m_key]["results"][b_name] = metrics
        print(f"  [SCORE] {m_key:<22} @ {b_name.upper():<10} | Accuracy: {metrics['accuracy_pct']}% | Passed: {metrics['passed']} | Failed: {metrics['failed']} | Didn't Finish: {metrics['did_not_finish']} | Total: {metrics['total']}")

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    write_reports(suite_summary, run_dir)
    print(f"[STAGE 2 COMPLETE] Reports saved under '{run_dir}'.\n")
    return suite_summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Response Extraction & Scoring Engine")
    parser.add_argument("--run_dir", type=str, help="Path to run output directory")
    parser.add_argument("--latest", action="store_true", help="Score the latest run in outputs/")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if not run_dir or args.latest:
        base_dir = WORKSPACE_ROOT / "outputs"
        runs = sorted([d for d in base_dir.glob("run_*") if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True) if base_dir.exists() else []
        if runs:
            run_dir = runs[0]

    if not run_dir or not run_dir.exists():
        print("[ERROR] No valid run directory found.")
        sys.exit(1)

    process_run_directory(run_dir)


if __name__ == "__main__":
    main()
