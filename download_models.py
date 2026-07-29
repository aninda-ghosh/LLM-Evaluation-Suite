#!/usr/bin/env python3
"""
Dynamic Model Downloader.

Downloads model weights into local workspace root `./models/<model_key>`.
Supports both pre-configured models in `models.json` AND dynamic arbitrary Hugging Face
repository IDs passed via CLI (e.g. `python3 download_models.py --model meta-llama/Llama-3.2-1B-Instruct`).

Usage:
    python3 download_models.py --model gemma-4-e2b
    python3 download_models.py --model Qwen/Qwen2.5-1.5B-Instruct
    python3 download_models.py --all
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path

# Workspace directory setup
WORKSPACE_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = WORKSPACE_ROOT / ".cache"
LOCAL_TMP_DIR = LOCAL_CACHE_DIR / "tmp"
LOCAL_MODELS_DIR = WORKSPACE_ROOT / "models"

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Disable Xet/CAS backend for the in-process snapshot_download path too
# (can fail with HTTP 416 during reconstruction, especially on Windows).
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def get_hf_token():
    """Retrieves Hugging Face authentication token from env or token files."""
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")

    ws_token = LOCAL_CACHE_DIR / "huggingface" / "token"
    if ws_token.exists():
        return ws_token.read_text().strip()

    user_token = Path.home() / ".cache" / "huggingface" / "token"
    if user_token.exists():
        return user_token.read_text().strip()

    return None


def get_subprocess_env():
    """Builds environment for subprocess with token and local cache paths."""
    env = os.environ.copy()
    env["HF_HOME"] = str(LOCAL_CACHE_DIR / "huggingface")
    env["TRANSFORMERS_CACHE"] = str(LOCAL_CACHE_DIR / "huggingface" / "hub")
    env["HF_DATASETS_CACHE"] = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
    env["TMPDIR"] = str(LOCAL_TMP_DIR)
    env["TEMP"] = str(LOCAL_TMP_DIR)
    env["TMP"] = str(LOCAL_TMP_DIR)
    env["HF_XET_HIGH_PERFORMANCE"] = "0"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Disable Xet/CAS backend (can fail with HTTP 416 during reconstruction, esp. on Windows).
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    token = get_hf_token()
    if token:
        env["HF_TOKEN"] = token

    return env


def load_models_config() -> dict:
    """Loads configured models from models.json."""
    p = WORKSPACE_ROOT / "models.json"
    if p.exists():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def download_model(model_input: str) -> bool:
    """
    Dynamically downloads model repository into `./models/`.
    Checks `models.json` for aliases/repo mappings, or downloads any raw Hugging Face repo ID directly.
    """
    models_cfg = load_models_config()
    m_input_lower = model_input.lower().strip()

    repo_id = model_input
    dest_key = model_input

    for k, info in models_cfg.items():
        if k.lower() == m_input_lower:
            repo_id = info.get("hf_repo", model_input)
            dest_key = k
            break
        aliases = [a.lower() for a in info.get("aliases", [])]
        if m_input_lower in aliases:
            repo_id = info.get("hf_repo", model_input)
            dest_key = k
            break

    if "/" in dest_key:
        dest_key = dest_key.split("/")[-1]

    dest_dir = LOCAL_MODELS_DIR / dest_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Target Model Repo : {repo_id}")
    print(f"[INFO] Destination Dir   : {dest_dir}")

    token = get_hf_token()
    try:
        from huggingface_hub import snapshot_download
        print("[INFO] Downloading model via huggingface_hub API...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
            token=token
        )
        print(f"[SUCCESS] Downloaded '{repo_id}' to '{dest_dir}'.\n")
        return True
    except Exception as e:
        print(f"[WARNING] Python snapshot_download fallback: {e}")

    cmd = ["hf", "download", repo_id, "--local-dir", str(dest_dir)]
    print(f"[INFO] Running CLI: {' '.join(cmd)}")
    sub_env = get_subprocess_env()

    try:
        res = subprocess.run(cmd, env=sub_env, check=True)
        if res.returncode == 0:
            print(f"[SUCCESS] Downloaded '{repo_id}' to '{dest_dir}'.\n")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[ERROR] Downloading {repo_id} failed: {e}\n")
        return False

    return False


def main():
    parser = argparse.ArgumentParser(description="Dynamic Model Downloader")
    parser.add_argument("--model", type=str, help="Model key, alias, or raw Hugging Face repo ID (e.g. --model Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--all", action="store_true", help="Download all pre-configured models in models.json")

    args = parser.parse_args()
    models_cfg = load_models_config()

    if args.all:
        print(f"\n==================================================")
        print(f"DOWNLOADING CONFIGURATED MODELS ({len(models_cfg)} models)")
        print(f"==================================================\n")
        for m_key in models_cfg.keys():
            download_model(m_key)
    elif args.model:
        download_model(args.model)
    else:
        print("Specify a model to download:")
        print("  python3 download_models.py --model gemma-4-e2b")
        print("  python3 download_models.py --model Qwen/Qwen2.5-1.5B-Instruct")
        print("  python3 download_models.py --all")


if __name__ == "__main__":
    main()
