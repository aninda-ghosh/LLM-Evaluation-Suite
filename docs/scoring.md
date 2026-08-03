# Scoring Methodology

How each benchmark evaluator converts a raw model response into a pass/fail, and
how those per-item results roll up into the report. Every evaluator exposes two
functions:

- `format_prompt(item)`  -  builds the prompt shown to the model.
- `score_item(item, response)`  -  returns `(is_pass, extracted_answer, reason)`.

Accuracy for a model/benchmark is simply `passed / total`.

---

## GSM8K (math, exact numeric match)

**Prompt:** the question followed by `Solve step-by-step and state the final numerical answer after ####.`

**Scoring steps:**

1. Read the gold answer from `item["answer"]`, split on `####`, take the text
   after it, strip whitespace, and remove commas -> `expected`.
2. Extract the model's number:
   a. Look for `#### <number>` in the response; if found, use that number.
   b. Otherwise, strip commas and take the **last** number found in the response.
3. If both a predicted and expected number exist, pass when
   `abs(pred - expected) < 1e-3` (float comparison, so `18`, `18.0`, and `18.00`
   all match). A non-numeric value counts as fail.
4. If no number can be extracted, fail with "No numeric answer found".

**Notes:** matching is purely numeric  -  units, `$`, and surrounding text are
ignored. The `1e-3` tolerance absorbs formatting/precision differences.

---

## MMLU (4-way multiple choice)

**Prompt:** the question, choices rendered as `A) ... B) ... C) ... D) ...`, and
`Answer with the letter (A, B, C, or D) only.`

**Scoring steps:**

1. Map the gold index in `item["answer"]` (an integer 0-3) to a letter:
   `0->A, 1->B, 2->C, 3->D` -> `expected`.
2. Extract the model's letter:
   a. Look for a phrase like `answer is X` / `option: X` / `choice X`
      (case-insensitive, `X` in A-D).
   b. Otherwise take the **last** standalone A-D letter in the response.
   c. If neither matches, use the first character of the response.
3. Pass when `pred == expected`.

---

## MMLU-Pro (up to 10-way multiple choice, A-J)

**Prompt:** the question, choices rendered as `A) ... B) ... ...`, and
`Answer with the letter (A-X) only.` (where X is the last option letter, up to J).

**Scoring steps:**

1. Map the gold index in `item["answer"]` (an integer 0-9) to a letter:
   `0->A, 1->B, ..., 9->J` -> `expected`.
2. Extract the model's letter:
   a. Look for a phrase like `answer is X` / `option: X` / `choice X`
      (case-insensitive, `X` in valid option letters A-J).
   b. Otherwise take the **last** standalone valid option letter in the response.
   c. If neither matches, use the first character of the response.
3. Pass when `pred == expected`.

---

## HellaSwag (4-way multiple choice, sentence completion)

**Prompt:** the context, four candidate endings as `A) ... B) ... C) ... D) ...`, and
`Answer with the letter (A, B, C, or D) only.`

**Scoring steps:**

1. Map `item["label"]` (index 0-3) to a letter -> `expected`.
2. Extract the model's letter using the same logic as MMLU (phrase match ->
   last A-D letter -> first character fallback).
3. Pass when `pred == expected`.

---

## TruthfulQA (multiple choice, MC1  -  options shuffled)

**Important:** the raw TruthfulQA MC1 data always lists the correct choice first.
Presenting options in raw order made the answer always "A", so the score measured
a model's positional bias rather than truthfulness. The evaluator now shuffles the
options with a **deterministic per-item permutation seeded by the item id**
(`md5(id)`), so `format_prompt` and `score_item` produce the same order across
separate calls and across runs. Item option counts vary (2-13), so the correct
letter can be beyond D.

**Prompt:** the question, the **shuffled** choices as `A) ... B) ... ...`, and
`Answer with the letter of the correct choice only.`

**Scoring steps:**

1. Find the correct choice  -  the index in `mc1_targets["labels"]` whose value is
   `1` (fallback to index 0).
2. Rebuild the deterministic shuffle (seeded by item id) and locate where the
   correct choice landed -> `expected` letter.
3. Extract the model's letter:
   a. Look for a phrase like `answer is X` / `option: X` / `choice X`
      (case-insensitive, `X` any A-Z).
   b. Otherwise take the **last** standalone A-Z letter in the response.
   c. If neither matches, use the first character of the response.
4. Pass when `pred == expected`.

---

## Aggregation into the report

For each model/benchmark the suite records:

- **Accuracy (%)** = `passed / total`.
- **Passed / Total** counts.
- **Avg latency (ms)** = mean of per-sample `sample_latency_ms`.
- **Aggregate speed (tok/s)** = mean of per-sample `aggregate_tps`.

The overall cross-benchmark matrix averages latency and tok/s **across the four
benchmarks unweighted**  -  each benchmark counts equally regardless of how many
samples it has. Accuracy columns in the matrix are the same per-benchmark values.

---

## Known limitations

- **Choice extraction is regex/heuristic, not logprob-based.** It works well when
  models reply with a bare letter; on verbose responses the "last letter" fallback
  can misread the intended answer.
- **GSM8K takes the last number** when there is no `####` marker, which can pick up
  an intermediate value if the model doesn't follow the format.
