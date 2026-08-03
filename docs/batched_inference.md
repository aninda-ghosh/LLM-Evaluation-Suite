# Batched Inference & Padding Guide

This document explains how batched inference is performed across causal language models (LLMs) in this suite, detailing **tensor alignment**, **left-padding**, and **attention masking**.

---

## 1. Why Left Padding is Required for Generation

In decoder-only autoregressive LLMs (such as LLaMA, Qwen, Gemma), **new tokens are generated sequentially at the right end** of the token sequence.

When processing a batch of prompts with varying token lengths, padding tokens (`<pad>`) are inserted to make all tensor rows equal in length.

### ASCII Diagram: Left Padding vs. Right Padding

#### Left Padding (`padding_side = "left"`) -- CORRECT FOR GENERATION
Padding tokens are prepended at the **beginning** of shorter prompts so that all prompts finish at the exact same rightmost column index:

```text
Batch Item 1 (Short):  [<pad>  <pad>  <pad>  What   is    2     +     2   ?] [Token 1] [Token 2] ...
Batch Item 2 (Long) :  [  In   math,  what   is    the   sum   of    2   ?] [Token 1] [Token 2] ...
                       +------------------- Input Tensor -------------------+ +------ Generation ------+
                                                                           ^
                                                            All prompts end at index N
                                                            Generation begins cleanly!
```

- **Attention Mask**: `[0, 0, 0, 1, 1, 1, 1, 1, 1]` (ignores `<pad>` tokens).
- **Result**: Generation appends new tokens directly to the end of all prompts simultaneously across GPU threads.

---

#### Right Padding (`padding_side = "right"`) -- INCORRECT FOR GENERATION
Padding tokens are appended at the **end** of shorter prompts:

```text
Batch Item 1 (Short):  [What   is    2     +     2    ?   <pad> <pad> <pad>] ??? (Tries to generate after <pad>)
Batch Item 2 (Long) :  [  In   math,  what   is    the   sum   of    2   ?] [Token 1] [Token 2] ...
                       +------------------- Input Tensor -------------------+
```

- **Result**: For shorter prompts, `model.generate()` attempts to generate tokens *after* the padding tokens or corrupts the positional embeddings, leading to corrupted outputs or premature EOS stops.

---

## 2. Batched Tensor Memory Layout

During Stage 1 (`eval.py`), batch sequence prompts are tokenized and formatted into PyTorch tensors:

```text
Input IDs Matrix (Batch Size = 2, Length = 8):

Row 0:  [ PAD_ID, PAD_ID, PAD_ID,  tok_0,  tok_1,  tok_2,  tok_3,  tok_4 ]
Row 1:  [  tok_0,  tok_1,  tok_2,  tok_3,  tok_4,  tok_5,  tok_6,  tok_7 ]

Attention Mask Matrix:

Row 0:  [      0,      0,      0,      1,      1,      1,      1,      1 ]
Row 1:  [      1,      1,      1,      1,      1,      1,      1,      1 ]
```

---

## 3. Performance Metrics Calculation

The batched generation loop calculates two primary performance metrics per batch:

### 1. Aggregate Speed (Throughput)
- **Formula**: `Aggregate Speed (tok/s) = Total Generated Tokens / Batch Wall-Clock Time (sec)`

### 2. Average Latency per Sample
- **Formula**: `Average Latency per Sample (ms) = (Batch Wall-Clock Time (sec) / Batch Size) * 1000`

---

## 4. Key Takeaways

1. **`tokenizer.padding_side = "left"`** is set on model initialization in `eval.py`.
2. **`tokenizer.pad_token = tokenizer.eos_token`** ensures a valid padding token ID is available if the tokenizer lacks a default pad token.
3. Increasing `batch_size` improves **throughput (tok/s)** by reusing model weights loaded in GPU VRAM across multiple sequence rows.
