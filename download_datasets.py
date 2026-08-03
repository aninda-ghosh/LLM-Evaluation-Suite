#!/usr/bin/env python3
"""
Hugging Face Benchmark Dataset Downloader.

Fetches benchmark datasets (mmlu, mmlu_pro, gsm8k, hellaswag, truthfulqa)
from Hugging Face and exports JSON datasets into `./data/<benchmark>.json`.

Usage:
    python3 download_datasets.py --benchmark mmlu
    python3 download_datasets.py --all --limit 50
"""

import os
import argparse
import json
from pathlib import Path

# Workspace Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = WORKSPACE_ROOT / ".cache"
LOCAL_DATA_DIR = WORKSPACE_ROOT / "data"
LOCAL_RAW_DIR = LOCAL_DATA_DIR / "raw"

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Environment overrides
os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def get_hf_token() -> str | None:
    """Retrieves Hugging Face authentication token."""
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    for p in [LOCAL_CACHE_DIR / "huggingface" / "token", Path.home() / ".cache" / "huggingface" / "token"]:
        if p.exists():
            return p.read_text().strip()
    return None


def download_raw_repo(repo_id: str, dest_dir: Path):
    """Downloads raw HF dataset repo using snapshot_download."""
    token = get_hf_token()
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
            token=token
        )
    except Exception as e:
        print(f"[WARNING] Snapshot download warning for '{repo_id}': {e}")


def fetch_mmlu(limit: int = None) -> tuple[list, str]:
    repo_id = "cais/mmlu"
    download_raw_repo(repo_id, LOCAL_RAW_DIR / "mmlu")
    from datasets import load_dataset
    ds = load_dataset(repo_id, "all", split="test", cache_dir=str(LOCAL_CACHE_DIR / "huggingface" / "datasets"), token=get_hf_token())
    items = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        items.append({"id": f"mmlu_{i}", "question": item["question"], "choices": item["choices"], "answer": item["answer"]})
    return items, repo_id


def fetch_mmlu_pro(limit: int = None) -> tuple[list, str]:
    repo_id = "TIGER-Lab/MMLU-Pro"
    download_raw_repo(repo_id, LOCAL_RAW_DIR / "mmlu_pro")
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="test", cache_dir=str(LOCAL_CACHE_DIR / "huggingface" / "datasets"), token=get_hf_token())
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


def fetch_gsm8k(limit: int = None) -> tuple[list, str]:
    repo_id = "openai/gsm8k"
    download_raw_repo(repo_id, LOCAL_RAW_DIR / "gsm8k")
    from datasets import load_dataset
    cache_p = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
    try:
        ds = load_dataset(repo_id, "main", split="test", cache_dir=cache_p, token=get_hf_token())
    except Exception:
        ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_p, trust_remote_code=True, token=get_hf_token())

    items = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        items.append({"id": f"gsm8k_{i}", "question": item["question"], "answer": item["answer"]})
    return items, repo_id


def fetch_hellaswag(limit: int = None) -> tuple[list, str]:
    repo_id = "Rowan/hellaswag"
    download_raw_repo(repo_id, LOCAL_RAW_DIR / "hellaswag")
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="validation", cache_dir=str(LOCAL_CACHE_DIR / "huggingface" / "datasets"), token=get_hf_token())
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


def fetch_truthfulqa(limit: int = None) -> tuple[list, str]:
    repo_id = "truthfulqa/truthful_qa"
    download_raw_repo(repo_id, LOCAL_RAW_DIR / "truthfulqa")
    from datasets import load_dataset
    cache_p = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
    try:
        ds = load_dataset(repo_id, "multiple_choice", split="validation", cache_dir=cache_p, token=get_hf_token())
    except Exception:
        ds = load_dataset("domenicrosati/TruthfulQA", split="train", cache_dir=cache_p, token=get_hf_token())

    items = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        mc1 = item.get("mc1_targets", {})
        q_val = item.get("question") or item.get("Question") or ""
        items.append({"id": f"truthfulqa_{i}", "question": q_val, "mc1_targets": mc1})
    return items, repo_id


FETCHERS = {
    "mmlu": fetch_mmlu,
    "mmlu_pro": fetch_mmlu_pro,
    "gsm8k": fetch_gsm8k,
    "hellaswag": fetch_hellaswag,
    "truthfulqa": fetch_truthfulqa,
}


def download_dataset(benchmark_name: str, limit: int = None) -> bool:
    """Downloads benchmark dataset and writes ./data/<benchmark>.json."""
    if benchmark_name not in FETCHERS:
        print(f"[ERROR] Unknown benchmark '{benchmark_name}'. Available: {list(FETCHERS.keys())}")
        return False

    print(f"[INFO] Downloading dataset '{benchmark_name}'...")
    try:
        items, repo_id = FETCHERS[benchmark_name](limit=limit)
        out_file = LOCAL_DATA_DIR / f"{benchmark_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        print(f"[SUCCESS] Saved {len(items)} items to '{out_file}'.\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed downloading '{benchmark_name}': {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Benchmark Dataset Downloader")
    parser.add_argument("--benchmark", type=str, choices=list(FETCHERS.keys()), help="Benchmark dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all benchmark datasets")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit (e.g. --limit 50)")

    args = parser.parse_args()

    if args.all:
        for b_name in FETCHERS.keys():
            download_dataset(b_name, limit=args.limit)
    elif args.benchmark:
        download_dataset(args.benchmark, limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
