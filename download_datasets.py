#!/usr/bin/env python3
"""
Standalone Hugging Face Benchmark Dataset Downloader.

Downloads raw dataset repositories from Hugging Face into `./data/raw/<benchmark_name>/`
using Hugging Face Hub snapshot download AND exports structured JSON files to `./data/<benchmark_name>.json`.

Supported Benchmarks:
    - mmlu        : Hugging Face 'cais/mmlu'
    - mmlu_pro    : Hugging Face 'TIGER-Lab/MMLU-Pro' (up to 10 options per question)
    - gsm8k       : Hugging Face 'openai/gsm8k'
    - hellaswag   : Hugging Face 'Rowan/hellaswag'
    - truthfulqa  : Hugging Face 'domenicrosati/TruthfulQA' / 'truthful_qa'

Usage:
    python3 download_datasets.py --benchmark mmlu
    python3 download_datasets.py --benchmark mmlu_pro
    python3 download_datasets.py --benchmark gsm8k
    python3 download_datasets.py --benchmark hellaswag
    python3 download_datasets.py --benchmark truthfulqa
    python3 download_datasets.py --all --limit 50
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Workspace directory setup
WORKSPACE_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = WORKSPACE_ROOT / ".cache"
LOCAL_TMP_DIR = LOCAL_CACHE_DIR / "tmp"
LOCAL_DATA_DIR = WORKSPACE_ROOT / "data"
LOCAL_RAW_DATA_DIR = LOCAL_DATA_DIR / "raw"

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(LOCAL_CACHE_DIR / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
os.environ["TMPDIR"] = str(LOCAL_TMP_DIR)
os.environ["TEMP"] = str(LOCAL_TMP_DIR)
os.environ["TMP"] = str(LOCAL_TMP_DIR)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
# Disable the Xet/CAS transfer backend: it can fail parquet reconstruction with
# HTTP 416 (Requested Range Not Satisfiable), notably on Windows. Forces plain HTTPS.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def get_hf_token() -> str | None:
    """Retrieves Hugging Face authentication token from env or local token cache."""
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")

    ws_token = LOCAL_CACHE_DIR / "huggingface" / "token"
    if ws_token.exists():
        return ws_token.read_text().strip()

    user_token = Path.home() / ".cache" / "huggingface" / "token"
    if user_token.exists():
        return user_token.read_text().strip()

    return None


def download_raw_hf_repo(repo_id: str, dest_dir: Path) -> bool:
    """Downloads raw dataset repository directly from Hugging Face Hub using snapshot_download."""
    token = get_hf_token()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading raw HF dataset repository '{repo_id}' to '{dest_dir}'...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
            token=token
        )
        print(f"[SUCCESS] Downloaded raw repository '{repo_id}' to '{dest_dir}'.")
        return True
    except Exception as e:
        print(f"[WARNING] Raw HF snapshot download failed for '{repo_id}': {e}")
        return False


def fetch_mmlu_dataset(limit: int = None) -> tuple[list, str]:
    """Fetches raw MMLU dataset from Hugging Face ('cais/mmlu')."""
    repo_id = "cais/mmlu"
    raw_dest = LOCAL_RAW_DATA_DIR / "mmlu"
    download_raw_hf_repo(repo_id, raw_dest)

    print("[INFO] Processing MMLU items for JSON output...")
    try:
        from datasets import load_dataset
        token = get_hf_token()
        ds = load_dataset(
            repo_id,
            "all",
            split="test",
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface" / "datasets"),
            token=token
        )
        items = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            items.append({
                "id": f"mmlu_{i}",
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"]
            })
        return items, repo_id
    except Exception as e:
        print(f"[ERROR] MMLU processing failed: {e}")
        return [], repo_id


def fetch_mmlu_pro_dataset(limit: int = None) -> tuple[list, str]:
    """Fetches raw MMLU-Pro dataset from Hugging Face ('TIGER-Lab/MMLU-Pro').

    MMLU-Pro has up to 10 options per question (padded with 'N/A'). Trailing
    'N/A' pads are dropped; `answer` is the integer index of the correct option,
    consistent with the MMLU schema. Note: scoring MMLU-Pro requires an evaluator
    that supports more than 4 choices (A-J), not the existing 4-way mmlu evaluator.
    """
    repo_id = "TIGER-Lab/MMLU-Pro"
    raw_dest = LOCAL_RAW_DATA_DIR / "mmlu_pro"
    download_raw_hf_repo(repo_id, raw_dest)

    print("[INFO] Processing MMLU-Pro items for JSON output...")
    try:
        from datasets import load_dataset
        token = get_hf_token()
        ds = load_dataset(
            repo_id,
            split="test",
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface" / "datasets"),
            token=token
        )
        items = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            options = [o for o in item.get("options", []) if o != "N/A"]
            items.append({
                "id": f"mmlu_pro_{i}",
                "question": item.get("question", ""),
                "choices": options,
                "answer": item.get("answer_index", 0),
                "category": item.get("category", "")
            })
        return items, repo_id
    except Exception as e:
        print(f"[ERROR] MMLU-Pro processing failed: {e}")
        return [], repo_id


def fetch_gsm8k_dataset(limit: int = None) -> tuple[list, str]:
    """Fetches raw GSM8K dataset from Hugging Face ('openai/gsm8k')."""
    repo_id = "openai/gsm8k"
    raw_dest = LOCAL_RAW_DATA_DIR / "gsm8k"
    download_raw_hf_repo(repo_id, raw_dest)

    print("[INFO] Processing GSM8K items for JSON output...")
    try:
        from datasets import load_dataset
        token = get_hf_token()
        cache_p = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
        try:
            ds = load_dataset(repo_id, "main", split="test", cache_dir=cache_p, token=token)
        except Exception:
            ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_p, trust_remote_code=True, token=token)

        items = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            items.append({
                "id": f"gsm8k_{i}",
                "question": item["question"],
                "answer": item["answer"]
            })
        return items, repo_id
    except Exception as e:
        print(f"[ERROR] GSM8K processing failed: {e}")
        return [], repo_id


def fetch_hellaswag_dataset(limit: int = None) -> tuple[list, str]:
    """Fetches raw HellaSwag dataset from Hugging Face ('Rowan/hellaswag')."""
    repo_id = "Rowan/hellaswag"
    raw_dest = LOCAL_RAW_DATA_DIR / "hellaswag"
    download_raw_hf_repo(repo_id, raw_dest)

    print("[INFO] Processing HellaSwag items for JSON output...")
    try:
        from datasets import load_dataset
        token = get_hf_token()
        cache_p = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
        try:
            ds = load_dataset(repo_id, split="validation", cache_dir=cache_p, token=token)
        except Exception:
            ds = load_dataset("hellaswag", split="validation", cache_dir=cache_p, trust_remote_code=True, token=token)

        items = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            items.append({
                "id": f"hellaswag_{i}",
                "ctx": item.get("ctx", ""),
                "endings": item.get("endings", []),
                "label": item.get("label", 0)
            })
        return items, repo_id
    except Exception as e:
        print(f"[ERROR] HellaSwag processing failed: {e}")
        return [], repo_id


def fetch_truthfulqa_dataset(limit: int = None) -> tuple[list, str]:
    """Fetches raw TruthfulQA dataset from Hugging Face ('truthfulqa/truthful_qa')."""
    repo_id = "truthfulqa/truthful_qa"
    raw_dest = LOCAL_RAW_DATA_DIR / "truthfulqa"
    download_raw_hf_repo(repo_id, raw_dest)

    print("[INFO] Processing TruthfulQA items for JSON output...")
    try:
        from datasets import load_dataset
        token = get_hf_token()
        cache_p = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
        try:
            ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", cache_dir=cache_p, token=token)
        except Exception:
            ds = load_dataset("domenicrosati/TruthfulQA", split="train", cache_dir=cache_p, token=token)

        items = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            mc1 = item.get("mc1_targets", {})
            q_val = item.get("question") or item.get("Question") or ""
            items.append({
                "id": f"truthfulqa_{i}",
                "question": q_val,
                "mc1_targets": mc1
            })
        return items, repo_id
    except Exception as e:
        print(f"[ERROR] TruthfulQA processing failed: {e}")
        return [], repo_id


def download_benchmark(benchmark_name: str, limit: int = None) -> bool:
    """Downloads raw HF repository into ./data/raw/<benchmark>/ AND exports JSON to ./data/<benchmark>.json."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fetchers = {
        "mmlu": fetch_mmlu_dataset,
        "mmlu_pro": fetch_mmlu_pro_dataset,
        "gsm8k": fetch_gsm8k_dataset,
        "hellaswag": fetch_hellaswag_dataset,
        "truthfulqa": fetch_truthfulqa_dataset
    }

    if benchmark_name not in fetchers:
        print(f"[ERROR] Unknown benchmark '{benchmark_name}'. Available: {list(fetchers.keys())}")
        return False

    items, repo_id = fetchers[benchmark_name](limit=limit)
    if not items:
        print(f"[ERROR] Failed to fetch dataset items for '{benchmark_name}'.")
        return False

    out_file = LOCAL_DATA_DIR / f"{benchmark_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    raw_dir = LOCAL_RAW_DATA_DIR / benchmark_name
    print(f"[SUCCESS] Downloaded raw repository '{repo_id}' -> '{raw_dir}'")
    print(f"[SUCCESS] Exported JSON dataset ({len(items)} items) -> '{out_file}'.\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Raw Hugging Face Benchmark Dataset Downloader")
    parser.add_argument("--benchmark", type=str, choices=["mmlu", "mmlu_pro", "gsm8k", "hellaswag", "truthfulqa"], help="Benchmark dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all benchmark datasets")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (e.g. --limit 50). Omit for full dataset.")

    args = parser.parse_args()

    benchmarks_list = ["mmlu", "mmlu_pro", "gsm8k", "hellaswag", "truthfulqa"]

    if args.all:
        print(f"\n==================================================")
        print(f"DOWNLOADING BENCHMARK DATASETS (RAW REPOS + JSON)")
        print(f"Target Directory: {LOCAL_DATA_DIR}")
        print(f"Raw Repos Dir   : {LOCAL_RAW_DATA_DIR}")
        print(f"Local Cache Dir : {LOCAL_CACHE_DIR}")
        print(f"==================================================\n")
        for b_name in benchmarks_list:
            download_benchmark(b_name, limit=args.limit)
    elif args.benchmark:
        download_benchmark(args.benchmark, limit=args.limit)
    else:
        print("Specify a benchmark dataset to download:")
        print("  python3 download_datasets.py --benchmark mmlu")
        print("  python3 download_datasets.py --benchmark mmlu_pro")
        print("  python3 download_datasets.py --benchmark gsm8k")
        print("  python3 download_datasets.py --benchmark hellaswag")
        print("  python3 download_datasets.py --benchmark truthfulqa")
        print("  python3 download_datasets.py --all --limit 50")


if __name__ == "__main__":
    main()
