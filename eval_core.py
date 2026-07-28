"""
================================================================================
Core Engine for LLM Benchmark Evaluation Platform
================================================================================

Provides shared infrastructure for environment configuration, model path resolution,
Hugging Face model loading, batched PyTorch inference, memory cleanup, and dataset orchestration.
"""

import os
import sys
import json
import re
import time
import gc
from pathlib import Path

# Workspace Root & Environment Setup
WORKSPACE_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = WORKSPACE_ROOT / ".cache"
LOCAL_TMP_DIR = LOCAL_CACHE_DIR / "tmp"
LOCAL_MODELS_DIR = WORKSPACE_ROOT / "models"

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# PyTorch CUDA VRAM Optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(LOCAL_CACHE_DIR / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(LOCAL_CACHE_DIR / "huggingface" / "datasets")
os.environ["TMPDIR"] = str(LOCAL_TMP_DIR)
os.environ["TEMP"] = str(LOCAL_TMP_DIR)
os.environ["TMP"] = str(LOCAL_TMP_DIR)

from evaluators import get_evaluator


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


def load_models_config() -> dict:
    """Loads model configuration mapping from models.json if present."""
    p = WORKSPACE_ROOT / "models.json"
    if p.exists():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def resolve_model_path(model_input: str) -> str:
    """Resolves model key, alias, local directory, or HF repo ID."""
    m_input_lower = model_input.lower().strip()

    direct_p = Path(model_input)
    if direct_p.exists() and (direct_p.is_dir() or direct_p.is_file()):
        return str(direct_p)

    local_p = LOCAL_MODELS_DIR / model_input
    if local_p.exists() and any(local_p.iterdir()):
        return str(local_p)

    cfg = load_models_config()
    target_key = None
    target_info = None

    for k, info in cfg.items():
        if k.lower() == m_input_lower:
            target_key = k
            target_info = info
            break
        aliases = [a.lower() for a in info.get("aliases", [])]
        if m_input_lower in aliases:
            target_key = k
            target_info = info
            break

    if target_key and target_info:
        loc = LOCAL_MODELS_DIR / target_key
        if loc.exists() and any(loc.iterdir()):
            return str(loc)
        loc_inp = LOCAL_MODELS_DIR / model_input
        if loc_inp.exists() and any(loc_inp.iterdir()):
            return str(loc_inp)
        return target_info.get("hf_repo", model_input)

    if LOCAL_MODELS_DIR.exists():
        for sub in LOCAL_MODELS_DIR.iterdir():
            if sub.is_dir() and sub.name.lower() == m_input_lower:
                return str(sub)

    return model_input


def discover_available_datasets() -> list:
    """Scans ./data/ directory for available benchmark JSON files."""
    data_dir = WORKSPACE_ROOT / "data"
    if not data_dir.exists():
        return []
    return [f.stem for f in sorted(data_dir.glob("*.json"))]


_MODEL_CACHE = {}


def clear_gpu_vram():
    """Frees PyTorch CUDA memory and triggers Python garbage collection."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def load_hf_model_and_tokenizer(model_path_or_repo: str):
    """Loads and caches PyTorch model and tokenizer pinned to target GPU/CPU."""
    global _MODEL_CACHE
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] 'transformers' and 'torch' packages are required.")
        sys.exit(1)

    hf_token = get_hf_token()

    if model_path_or_repo not in _MODEL_CACHE:
        clear_gpu_vram()
        print(f"[INFO] Loading model from '{model_path_or_repo}'...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_repo,
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
            trust_remote_code=True,
            token=hf_token
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo,
            torch_dtype=dtype,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
            trust_remote_code=True,
            token=hf_token
        )
        if not torch.cuda.is_available():
            model = model.to("cpu")

        _MODEL_CACHE[model_path_or_repo] = (model, tokenizer, device)

    return _MODEL_CACHE[model_path_or_repo]


def run_hf_batch_inference(model_path_or_repo: str, prompts: list, max_new_tokens: int = 512):
    """Executes batched LLM text generation."""
    import torch

    model, tokenizer, device = load_hf_model_and_tokenizer(model_path_or_repo)

    formatted_prompts = []
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                messages = [{"role": "user", "content": prompt}]
                f_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                f_p = prompt
        else:
            f_p = prompt
        formatted_prompts.append(f_p)

    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    batch_cnt = len(prompts)

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[ERROR] CUDA Out of Memory Error encountered: {e}")
        print("[INFO] Try reducing --batch_size (e.g., --batch_size 1).")
        clear_gpu_vram()
        sys.exit(1)

    t1 = time.perf_counter()

    batch_wall_sec = max(t1 - t0, 0.001)
    total_gen_tokens = sum([max(len(outputs[i]) - input_len, 0) for i in range(batch_cnt)])
    aggregate_tps = total_gen_tokens / batch_wall_sec
    sample_latency_ms = (batch_wall_sec / max(batch_cnt, 1)) * 1000.0

    batch_results = []
    for i in range(batch_cnt):
        full_text = tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
        prompt_text = tokenizer.decode(input_ids[i], skip_special_tokens=True).strip()

        if full_text.startswith(prompt_text):
            gen_text = full_text[len(prompt_text):].strip()
        else:
            gen_tokens = outputs[i][input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        if not gen_text:
            gen_text = full_text.replace(prompt_text, "").strip()

        batch_results.append({
            "text": gen_text,
            "sample_latency_ms": round(sample_latency_ms, 2),
            "batch_wall_sec": round(batch_wall_sec, 3),
            "aggregate_tps": round(aggregate_tps, 2)
        })

    return batch_results


def extract_think_part(text: str) -> str:
    """Extracts internal reasoning / thought blocks (<think> or Gemma 4 channel format)."""
    if "<|channel>thought" in text:
        match = re.search(r"<\|channel\|>thought\s*(.*?)(?:<channel\|>|$)", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    if "<think>" in text and "</think>" in text:
        match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    elif "<think>" in text:
        return text.split("<think>")[-1].strip()
    return "N/A (Direct Answer without think tag)"


def extract_final_answer(text: str) -> str:
    """Strips internal thinking blocks to leave only the final output answer."""
    clean = text
    if "<|channel>thought" in clean:
        clean = re.sub(r"<\|channel\|>thought.*?(?:<channel\|>|$)", "", clean, flags=re.DOTALL).strip()
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    elif "<think>" in clean:
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).replace("<think>", "").strip()
    return clean.strip() if clean.strip() else text.strip()


def run_benchmark_eval(model_target: str, benchmark_name: str, batch_size: int = 1, max_samples: int = None, debug: bool = False, log_file: Path = None):
    """Executes evaluation for a dataset JSON file with per-sample JSONL logging."""
    data_file = WORKSPACE_ROOT / "data" / f"{benchmark_name}.json"
    if not data_file.exists():
        data_file = Path(benchmark_name)

    if not data_file.exists():
        print(f"[ERROR] Dataset file '{benchmark_name}' not found in ./data/. Run download_datasets.py first.")
        return None

    with open(data_file, "r") as f:
        items = json.load(f)

    if max_samples and max_samples < len(items):
        items = items[:max_samples]

    evaluator = get_evaluator(benchmark_name)

    print(f"\n==================================================")
    print(f"BENCHMARK RUNNER: {benchmark_name.upper()} ({len(items)} samples | Batch Size: {batch_size})")
    if log_file:
        print(f"[INFO] Sample Log File: {log_file}")
        log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"==================================================")

    correct = 0
    sample_latencies, aggregate_speeds = [], []
    total_items = len(items)

    log_handle = open(log_file, "w", encoding="utf-8") if log_file else None

    try:
        for i in range(0, total_items, batch_size):
            batch_items = items[i : i + batch_size]
            prompts = [evaluator.format_prompt(item) for item in batch_items]

            results = run_hf_batch_inference(model_target, prompts)

            for j, res in enumerate(results):
                idx = i + j
                item = batch_items[j]
                prompt = prompts[j]

                clean_out = extract_final_answer(res["text"])
                think_out = extract_think_part(res["text"])

                is_pass, pred_val, score_reason = evaluator.score_item(item, clean_out)

                if is_pass:
                    correct += 1

                sample_latencies.append(res["sample_latency_ms"])
                aggregate_speeds.append(res["aggregate_tps"])

                status_str = "PASS" if is_pass else "FAIL"

                if log_handle:
                    entry = {
                        "sample_index": idx + 1,
                        "prompt": prompt,
                        "raw_item": item,
                        "think_reasoning": think_out,
                        "raw_response": res["text"],
                        "extracted_answer": clean_out,
                        "passed": is_pass,
                        "score_reason": score_reason,
                        "sample_latency_ms": res["sample_latency_ms"],
                        "aggregate_tps": res["aggregate_tps"]
                    }
                    log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    log_handle.flush()

                if debug:
                    print(f"\n[Sample {idx+1}/{total_items}] [{status_str}] | Latency: {res['sample_latency_ms']}ms | Speed: {res['aggregate_tps']} tok/s")
                    print(f"  [PROMPT]          :\n{prompt}\n")
                    print(f"  [THINK REASONING] :\n{think_out}\n")
                    print(f"  [RAW RESPONSE]    :\n{res['text']}\n")
                    print(f"  [EXTRACTED ANSWER]:\n{clean_out}\n")
                    print(f"  [SCORE REASON]    : {score_reason}")
                    print("=" * 60)
                else:
                    print(f"[{idx+1}/{total_items}] [{status_str}] | Latency: {res['sample_latency_ms']}ms | Speed: {res['aggregate_tps']} tok/s | Reason: {score_reason}")
    finally:
        if log_handle:
            log_handle.close()

    accuracy = (correct / total_items * 100.0) if total_items else 0.0
    avg_latency = sum(sample_latencies) / len(sample_latencies) if sample_latencies else 0.0
    avg_speed = sum(aggregate_speeds) / len(aggregate_speeds) if aggregate_speeds else 0.0

    print(f"\n--------------------------------------------------")
    print(f"SUMMARY [{benchmark_name.upper()}]: Accuracy: {accuracy:.2f}% ({correct}/{total_items}) | Avg Sample Latency: {avg_latency:.1f}ms | Avg System Speed: {avg_speed:.1f} tok/s")
    print(f"--------------------------------------------------\n")

    clear_gpu_vram()

    return {
        "accuracy_pct": round(accuracy, 2),
        "passed": correct,
        "total": total_items,
        "avg_sample_latency_ms": round(avg_latency, 2),
        "avg_aggregate_tps": round(avg_speed, 2)
    }
