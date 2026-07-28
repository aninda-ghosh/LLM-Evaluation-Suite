# Changelog

All notable changes to the LLM Benchmark Evaluation Suite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-28

### Added
- **Core Evaluation Engine (`eval_core.py`)**:
  - Batched PyTorch inference engine with dynamic device allocation and VRAM optimizations (`PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`).
  - Robust Hugging Face authentication handling via environment variables (`HF_TOKEN`) and local cache files.
  - Automatic model path resolution with alias lookup from `models.json`.

- **Multi-GPU Suite Execution Framework (`run_suite.py`)**:
  - Parallel process orchestration across auto-detected or user-configured GPU IDs.
  - Automatic generation of formatted markdown evaluation reports (`report.md`) featuring cross-benchmark matrix tables, accuracy metrics, average latency (ms), and token throughput (`tok/s`).
  - Automated execution logging and JSON result artifact persistence under `outputs/`.

- **Modular Benchmark Evaluators (`evaluators/`)**:
  - `gsm8k_eval.py`: Numerical extraction and step-by-step chain-of-thought math reasoning evaluation.
  - `mmlu_eval.py`: Multiple-choice knowledge evaluation across diverse domain subjects.
  - `hellaswag_eval.py`: Commonsense sentence completion and choice log-likelihood scoring.
  - `truthfulqa_eval.py`: Truthfulness evaluation across multiple-choice questions.

- **Single Benchmark CLI (`eval.py`)**:
  - Standalone evaluation script for single model and single benchmark execution with batch size options and detailed `--debug` mode output.

- **Dataset & Model Download Utilities**:
  - `download_datasets.py`: Automated fetching and formatting of raw Hugging Face datasets into local `./data/` directory with `--limit` test sample support.
  - `download_models.py`: Hugging Face repository weight downloader supporting pre-configured alias mappings (`qwen2.5-0.5b`, `qwen3.5-2b`, `qwen2.5-3b`, `gemma-2-2b`, `gemma-4-e2b`).

- **Configuration & Interactive Workspace**:
  - `config.yaml`: Global suite configuration specifying target models, evaluation datasets, GPU auto-detection, default batch size, and output directories.
  - `models.json`: Structured registry mapping model keys to Hugging Face repos, descriptions, default batch sizes, and aliases.
  - `test_eval_raw.ipynb`: Interactive Jupyter notebook for inspecting raw evaluation responses and performing step-by-step debugging.
  - `.gitignore`: Comprehensive ignore rules for PyTorch model weights (`models/`), dataset caches (`data/`), build caches (`.cache/`), and run outputs (`outputs/`).
