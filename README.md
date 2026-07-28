# LLM Evaluation Suite

A Python evaluation platform for benchmarking Large Language Models (LLMs) across standardized datasets (MMLU, GSM8K, HellaSwag, TruthfulQA) with PyTorch/Transformers inference on multi-GPU systems.

---

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Hugging Face Authentication
Authenticate to download gated models and dataset repositories:
```bash
hf auth login
```
Or set the environment variable:
```bash
export HF_TOKEN="your_hf_token"
```

### 3. Download Benchmark Datasets
Download raw Hugging Face dataset repositories into `./data/raw/` and export JSON files into `./data/`:
```bash
# Download all datasets (MMLU, GSM8K, HellaSwag, TruthfulQA)
python3 download_datasets.py --all

# Download test samples (50 items per dataset)
python3 download_datasets.py --all --limit 50
```

### 4. Download Model Weights
Download model weights into `./models/`:
```bash
# Download a pre-configured model key
python3 download_models.py --model gemma-4-e2b

# Download any Hugging Face repository ID
python3 download_models.py --model Qwen/Qwen2.5-1.5B-Instruct

# Download all configured models in models.json
python3 download_models.py --all
```

---

## Running Evaluations

### Single Benchmark Evaluation
```bash
# Evaluate on MMLU
python3 eval.py --model gemma-4-e2b --benchmark mmlu --batch_size 1 --debug

# Evaluate on GSM8K
python3 eval.py --model gemma-4-e2b --benchmark gsm8k --batch_size 1

# Evaluate on HellaSwag
python3 eval.py --model gemma-4-e2b --benchmark hellaswag --batch_size 1

# Evaluate on TruthfulQA
python3 eval.py --model gemma-4-e2b --benchmark truthfulqa --batch_size 1
```

### Multi-GPU Suite Execution
```bash
python3 run_suite.py
```

---

## Benchmark Datasets

| Dataset | Source Repository | Domain | Metric |
| :--- | :--- | :--- | :--- |
| **MMLU** | `cais/mmlu` | General Knowledge | Choice Accuracy (%) |
| **GSM8K** | `openai/gsm8k` | Math & Reasoning | Numerical Accuracy (%) |
| **HellaSwag** | `Rowan/hellaswag` | Commonsense Reasoning | Choice Accuracy (%) |
| **TruthfulQA** | `truthful_qa` | Truthfulness | Choice Accuracy (%) |

---

## Benchmark Results

> Full-dataset evaluation run on **2x GPU** (batch size `16`).  

| Model | GSM8K | MMLU | HellaSwag | TruthfulQA | Avg Latency | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2.5-3b` | **82.94%** | **62.95%** | **76.29%** | **56.43%** | 1933.77 ms | 28.94 tok/s |
| `gemma-4-e2b` | 80.82% | 56.96% | 56.46% | 52.75% | 3093.76 ms | 34.40 tok/s |
| `qwen3.5-2b` | 69.07% | 58.00% | 62.93% | 39.90% | 1687.21 ms | 39.36 tok/s |
| `gemma-2-2b` | 59.89% | 50.29% | 59.84% | 42.96% | 2330.26 ms | 43.83 tok/s |
| `qwen2.5-0.5b` | 43.97% | 35.94% | 33.71% | 30.72% | 714.29 ms | **122.57 tok/s** |

**Key Takeaways:**
- `qwen2.5-3b` leads on accuracy across all four benchmarks.
- `qwen2.5-0.5b` is the fastest at **122.57 tok/s** with the lowest latency (**714 ms**), making it the best fit for latency-sensitive on-device workloads.
- `qwen3.5-2b` offers the best accuracy-to-speed balance among the 2B-class models.

---

## CLI Reference

| Flag | Description | Example |
| :--- | :--- | :--- |
| `--model` | Model key, alias, local path, or HF repo ID | `--model gemma-4-e2b` |
| `--benchmark` | Benchmark dataset (`mmlu`, `gsm8k`, `hellaswag`, `truthfulqa`, `all`) | `--benchmark gsm8k` |
| `--batch_size` | Parallel sequences per batch | `--batch_size 1` |
| `--limit` | Maximum samples to evaluate | `--limit 50` |
| `--debug` | Print prompt, model response, and score analysis | `--debug` |
| `--output` | Summary JSON output path | `--output results.json` |

---

## Batch Size, Latency & Throughput Metrics

The evaluation engine calculates performance metrics per batch as follows:

$$\text{Aggregate Speed (tok/s)} = \frac{\text{Total Generated Tokens Across All Batch Items}}{\text{Batch Wall-Clock Time (sec)}}$$

$$\text{Average Latency per Sample (ms)} = \left(\frac{\text{Batch Wall-Clock Time (sec)}}{\text{Batch Size}}\right) \times 1000$$

### Performance Effects of Increasing Batch Size ($B$)

* **Aggregate Speed (`tok/s`)**: **Increases (Higher Throughput)**
  * **GPU Parallelism**: Matrix multiplications across multiple sequences execute concurrently. Model weights loaded into VRAM are reused across $B$ prompts per step, maximizing GPU compute utilization.
* **Average Latency per Sample (`ms`)**: **Decreases (Lower Latency per Sample)**
  * **Amortization**: Total batch wall-clock time increases slightly, but dividing that time by $B$ results in a lower average per-sample latency compared to sequential processing.
* **Trade-offs**:
  * **VRAM Usage**: Larger batch sizes consume more CUDA memory for KV-cache and activation tensors (risk of `CUDA OutOfMemoryError`).
  * **Padding Overhead**: Uneven prompt lengths in a batch introduce padding tokens.