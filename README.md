# LLM Evaluation Suite

A clean, modular Python suite for evaluating Large Language Models (LLMs) across standardized benchmark datasets (**MMLU**, **MMLU-Pro**, **GSM8K**, **HellaSwag**, **TruthfulQA**) using PyTorch / Transformers on multi-GPU systems.

---

## Quickstart Guide

### 1. Installation & Authentication
```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face (for gated models/datasets)
export HF_TOKEN="your_hf_token"
```

### 2. Download Datasets & Models
```bash
# Download benchmark datasets into ./data/
python download_datasets.py --all

# Download model weights into ./models/
python download_models.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### 3. Run Two-Stage Pipeline

```bash
# Stage 1: Run raw GPU model inference across active GPUs
python3 run_suite.py

# Stage 2: Extract answers, score generations, and create reports
python3 score_suite.py --latest
# (or target a specific folder: python3 score_suite.py --run_dir outputs/run_20260730_120000)
```

---

## Two-Stage Architecture

```text
  [Dataset JSON]
        |
        v
+-----------------+   Raw JSONL   +-----------------+
|     STAGE 1     |-------------->|     STAGE 2     |---> [report.md]
| GPU Inference   |               | Extract & Score |---> [results.csv]
+-----------------+               +-----------------+---> [summary.json]
```

1. Stage 1 - Raw GPU Generation (`eval.py` / `run_suite.py`):
   - Loads model onto a pinned GPU.
   - Formats prompts via evaluator classes (`evaluator.format_prompt`).
   - Runs batched inference and writes outputs (`prompt`, `dataset_item`, `model_output`, latency, throughput) to a JSONL log file.

2. Stage 2 - Extract, Evaluate & Score (`score_suite.py`):
   - Reads raw JSONL entries from Stage 1.
   - Uses benchmark-specific evaluator classes (`BaseEvaluator` subclasses) to extract answers (`extract_answer`) and think blocks (`extract_think`).
   - Scores extracted answers against gold standards (`score_item`) and generates consolidated reports.

---

## Supported Benchmark Datasets

| Dataset | Metric | Class Evaluator | Description |
| :--- | :--- | :--- | :--- |
| **GSM8K** | Numerical Accuracy (%) | `GSM8KEvaluator` | Grade school math word problems. |
| **MMLU** | Choice Accuracy (%) | `MMLUEvaluator` | 4-choice general knowledge test. |
| **MMLU-Pro** | Choice Accuracy (%) | `MMLUProEvaluator` | Advanced multi-domain test (up to 10 choices). |
| **HellaSwag** | Choice Accuracy (%) | `HellaSwagEvaluator` | Commonsense context completion. |
| **TruthfulQA** | Choice Accuracy (%) | `TruthfulQAEvaluator` | Truthfulness & misconception test. |

---

## Core Scripts & CLI Options

* **`eval.py`**: Single-model evaluation job on a single pinned GPU.
  ```bash
  python3 eval.py --model Qwen2.5-0.5B-Instruct --benchmark gsm8k --gpu 0 --batch_size 1
  ```

* **`run_suite.py`**: Multi-GPU Stage 1 raw inference orchestrator.
  ```bash
  python3 run_suite.py              # Full Stage 1 run
  python3 run_suite.py --preflight 4  # Quick test with 4 samples per benchmark
  ```

* **`score_suite.py`**: Stage 2 scoring engine.
  ```bash
  python3 score_suite.py --latest
  python3 score_suite.py --run_dir outputs/run_20260730_120000
  ```

---

## Output Files

Each run creates an output directory under `outputs/run_<timestamp>/`:

- **`report.md`**: Human-readable markdown comparison table of model accuracies.
- **`results.csv`**: Tidy CSV containing accuracy, latency, and token speed metrics.
- **`summary.json`**: Machine-readable JSON summary of the entire run.
- **`logs/`**: Raw and scored `.jsonl` task files containing per-sample outputs.

---

## Detailed Documentation

- **[Batched Inference & Padding Guide](docs/batched_inference.md)**: Comprehensive explanation of GPU batching, left-padding vs. right-padding, ASCII diagrams, and throughput metrics.
- **[Scoring Methodology Guide](docs/scoring.md)**: Detailed breakdown of evaluation algorithms per benchmark dataset.