#!/usr/bin/env python3
"""
Hugging Face Model Downloader.

Downloads model weights into `./models/<model_key>` for models configured
in `models.json` or arbitrary Hugging Face repository IDs.

Usage:
    python3 download_models.py --model gemma-4-e2b
    python3 download_models.py --all
"""

import os
import argparse
import json
from pathlib import Path

# Workspace Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = WORKSPACE_ROOT / ".cache"
LOCAL_MODELS_DIR = WORKSPACE_ROOT / "models"

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

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


def load_models_config() -> dict:
    """Loads model definitions from models.json."""
    config_path = WORKSPACE_ROOT / "models.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def download_model(model_input: str) -> bool:
    """Downloads model weights into `./models/<dest_key>`."""
    models_cfg = load_models_config()
    m_lower = model_input.lower().strip()

    repo_id = model_input
    dest_key = model_input

    for key, info in models_cfg.items():
        if key.lower() == m_lower:
            repo_id = info.get("hf_repo", model_input)
            dest_key = key
            break

    if "/" in dest_key:
        dest_key = dest_key.split("/")[-1]

    dest_dir = LOCAL_MODELS_DIR / dest_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Target Repo: {repo_id}")
    print(f"[INFO] Target Dir : {dest_dir}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
            token=get_hf_token()
        )
        print(f"[SUCCESS] Downloaded '{repo_id}' to '{dest_dir}'.\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed downloading '{repo_id}': {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Model Weight Downloader")
    parser.add_argument("--model", type=str, help="Model key, alias, or HF repo ID")
    parser.add_argument("--all", action="store_true", help="Download all models in models.json")

    args = parser.parse_args()
    models_cfg = load_models_config()

    if args.all:
        for m_key in models_cfg.keys():
            download_model(m_key)
    elif args.model:
        download_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
