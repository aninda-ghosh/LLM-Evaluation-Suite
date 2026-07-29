# Changelog

All notable changes to the LLM Benchmark Evaluation Suite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.2] - 2026-07-28

### Added
- **More benchmark models (`models.json`)**: registered instruct models up to 4B across four families — Qwen 2.5 (`qwen2.5-1.5b`), Qwen 3.5 (`qwen3.5-4b`), Gemma 3 (`gemma-3-1b`, `gemma-3-4b`), and Llama 3.2 (`llama-3.2-1b`, `llama-3.2-3b`). Llama and Gemma repos are gated and require Hugging Face access approval.
- **MMLU-Pro dataset (`download_datasets.py`)**: added `--benchmark mmlu_pro` fetching `TIGER-Lab/MMLU-Pro` (test split, up to 10 options per question); included in `--all`. Note: scoring requires a >4-choice (A–J) evaluator, not the existing 4-way MMLU evaluator.

### Fixed
- **Hugging Face Xet download failures**: disabled the Xet/CAS transfer backend (`HF_HUB_DISABLE_XET=1`) in both download scripts to avoid `HTTP 416 Requested Range Not Satisfiable` parquet reconstruction errors (seen on Windows), and silenced the symlink warning (`HF_HUB_DISABLE_SYMLINKS_WARNING=1`).

---

## [1.0.1] - 2026-07-28

### Fixed
- **TruthfulQA option ordering (`evaluators/truthfulqa_eval.py`)**: Raw MC1 data always lists the correct choice first, so the correct answer landed in option **"A"** for all 817 items. Scores therefore reflected a model's positional bias toward "A" rather than truthfulness. Options are now shuffled with a deterministic per-item permutation (seeded by item id via `md5`), shared between `format_prompt` and `score_item` so prompt and grading stay consistent across calls and runs. **TruthfulQA must be re-run to produce valid results; GSM8K, MMLU, and HellaSwag are unaffected.**

### Documentation
- Added the step-by-step scoring algorithm to each evaluator's `score_item` docstring (`gsm8k_eval.py`, `mmlu_eval.py`, `hellaswag_eval.py`, `truthfulqa_eval.py`).
- Added `docs/scoring.md` documenting per-benchmark scoring methodology, report aggregation, and known extraction limitations.

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
