#!/usr/bin/env python3
"""
Stage 1: Pinned GPU Model Inference Script.
Executes batch inference for 1 model on 1 benchmark dataset on an assigned GPU.
Outputs raw generation results to a JSONL log file.
"""

import os
import sys
import json
import time
import gc
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = WORKSPACE_ROOT / ".cache"
LOCAL_MODELS_DIR = WORKSPACE_ROOT / "models"
LOCAL_DATA_DIR = WORKSPACE_ROOT / "data"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR / "huggingface")

from evaluators import get_evaluator


def get_hf_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    for p in [LOCAL_CACHE_DIR / "huggingface" / "token", Path.home() / ".cache" / "huggingface" / "token"]:
        if p.exists():
            return p.read_text().strip()
    return None


def resolve_model_path(model_input: str) -> str:
    """Resolves model key or directory to usable local path or HF repo ID."""
    p = Path(model_input)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)

    local_p = LOCAL_MODELS_DIR / model_input
    if local_p.is_dir() and (local_p / "config.json").exists():
        return str(local_p)

    config_path = WORKSPACE_ROOT / "models.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                for key, info in cfg.items():
                    if key.lower() == model_input.lower().strip():
                        target_p = LOCAL_MODELS_DIR / key
                        if target_p.is_dir() and (target_p / "config.json").exists():
                            return str(target_p)
                        return info.get("hf_repo", model_input)
        except Exception:
            pass

    return model_input


def clear_gpu_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def load_model_and_tokenizer(model_target: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    clear_gpu_vram()
    token = get_hf_token()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] Loading '{model_target}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_target,
        cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
        trust_remote_code=True,
        token=token
    )
    # Left padding is required for decoder-only batched generation so new tokens append to the right
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_target,
        torch_dtype=dtype,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        cache_dir=str(LOCAL_CACHE_DIR / "huggingface"),
        trust_remote_code=True,
        token=token
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    return model, tokenizer, device


def run_batch_inference(model, tokenizer, device, prompts: list[str], max_new_tokens: int = 512) -> list[dict]:
    import torch

    formatted_prompts = []
    for p in prompts:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                formatted_prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                try:
                    formatted_prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
                except Exception:
                    formatted_prompts.append(p)
            except Exception:
                formatted_prompts.append(p)
        else:
            formatted_prompts.append(p)

    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]
    batch_cnt = len(prompts)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    batch_sec = max(time.perf_counter() - t0, 0.001)

    gen_token_cnt = sum([max(len(outputs[i]) - input_len, 0) for i in range(batch_cnt)])
    aggregate_tps = round(gen_token_cnt / batch_sec, 2)
    sample_latency_ms = round((batch_sec / max(batch_cnt, 1)) * 1000.0, 2)

    results = []
    for i in range(batch_cnt):
        full_text = tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
        prompt_text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True).strip()

        gen_text = full_text[len(prompt_text):].strip() if full_text.startswith(prompt_text) else tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip()
        if not gen_text:
            gen_text = full_text.replace(prompt_text, "").strip()

        results.append({
            "text": gen_text,
            "sample_latency_ms": sample_latency_ms,
            "aggregate_tps": aggregate_tps
        })

    return results


def run_single_job(model_input: str, benchmark_name: str, batch_size: int = 1, limit: int = None, debug: bool = False, log_file: Path = None):
    data_file = LOCAL_DATA_DIR / f"{benchmark_name}.json"
    if not data_file.exists():
        print(f"[ERROR] Dataset '{data_file}' not found.")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    if limit:
        items = items[:limit]

    model_target = resolve_model_path(model_input)
    evaluator = get_evaluator(benchmark_name)
    total_items = len(items)

    print(f"[STAGE 1] Model: '{model_input}' | Benchmark: '{benchmark_name.upper()}' | Samples: {total_items}")

    model, tokenizer, device = load_model_and_tokenizer(model_target)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "w", encoding="utf-8") if log_file else None

    try:
        for i in range(0, total_items, batch_size):
            batch_items = items[i : i + batch_size]
            prompts = [evaluator.format_prompt(item) for item in batch_items]
            results = run_batch_inference(model, tokenizer, device, prompts, 2048)

            for j, res in enumerate(results):
                idx = i + j
                entry = {
                    "sample_index": idx + 1,
                    "prompt": prompts[j],
                    "dataset_item": batch_items[j],
                    "model_output": res["text"],
                    "sample_latency_ms": res["sample_latency_ms"],
                    "aggregate_tps": res["aggregate_tps"]
                }
                if log_handle:
                    log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    log_handle.flush()

                if debug:
                    print(f"[{idx+1}/{total_items}] Latency: {res['sample_latency_ms']}ms | Speed: {res['aggregate_tps']} tok/s")
                    print(f"PROMPT:\n{prompts[j]}\nRESPONSE:\n{res['text']}\n" + "-"*50)
                else:
                    print(f"[{idx+1}/{total_items}] Latency: {res['sample_latency_ms']}ms | Speed: {res['aggregate_tps']} tok/s")
    finally:
        if log_handle:
            log_handle.close()
        clear_gpu_vram()

    print(f"[SUCCESS] Completed raw generation for '{model_input}' @ '{benchmark_name}'.")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Pinned GPU Inference Job")
    parser.add_argument("--model", type=str, required=True, help="Model key, alias, or path")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark dataset name")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit sample count")
    parser.add_argument("--debug", action="store_true", help="Print debug outputs")
    parser.add_argument("--log_file", type=str, default=None, help="Output JSONL log path")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_single_job(
        model_input=args.model,
        benchmark_name=args.benchmark,
        batch_size=args.batch_size,
        limit=args.limit,
        debug=args.debug,
        log_file=Path(args.log_file) if args.log_file else None
    )


if __name__ == "__main__":
    main()
