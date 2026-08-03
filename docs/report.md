# Evaluation Suite Report
**Timestamp:** `20260730_154428`

---

## GSM8K Benchmark

![GSM8K Model Comparison](charts/gsm8k_comparison.png)

| Model | Accuracy (%) | Passed (True) | Failed (False) | Didn't Finish | Total | Avg Latency | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3.5-4B` | **92.27%** | 1217 | 102 | 0 | 1319 | 3274.41ms | 134.15 tok/s |
| `gemma-4-E2B-it` | **91.05%** | 1201 | 118 | 0 | 1319 | 1642.71ms | 311.0 tok/s |
| `Qwen2.5-3B-Instruct` | **83.24%** | 1098 | 221 | 0 | 1319 | 1613.85ms | 221.37 tok/s |
| `gemma-3-4b-it` | **82.87%** | 1093 | 226 | 0 | 1319 | 2967.72ms | 174.69 tok/s |
| `Qwen3.5-2B` | **74.53%** | 983 | 336 | 0 | 1319 | 1638.91ms | 294.48 tok/s |
| `Qwen2.5-1.5B-Instruct` | **67.4%** | 889 | 430 | 0 | 1319 | 1100.07ms | 397.91 tok/s |
| `gemma-2-2b-it` | **60.88%** | 803 | 516 | 0 | 1319 | 1081.98ms | 267.55 tok/s |
| `gemma-3-1b-it` | **48.14%** | 635 | 684 | 0 | 1319 | 2013.99ms | 549.43 tok/s |
| `Qwen2.5-0.5B-Instruct` | **43.44%** | 573 | 746 | 0 | 1319 | 722.91ms | 807.58 tok/s |

---

## HELLASWAG Benchmark

![HELLASWAG Model Comparison](charts/hellaswag_comparison.png)

| Model | Accuracy (%) | Passed (True) | Failed (False) | Didn't Finish | Total | Avg Latency | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3.5-4B` | **84.08%** | 8443 | 1599 | 0 | 10042 | 180.97ms | 23.34 tok/s |
| `Qwen2.5-3B-Instruct` | **76.27%** | 7659 | 2383 | 0 | 10042 | 78.97ms | 27.03 tok/s |
| `Qwen2.5-1.5B-Instruct` | **63.13%** | 6340 | 3702 | 0 | 10042 | 56.52ms | 129.78 tok/s |
| `Qwen3.5-2B` | **62.85%** | 6311 | 3731 | 0 | 10042 | 75.92ms | 55.41 tok/s |
| `gemma-3-4b-it` | **60.66%** | 6091 | 3951 | 0 | 10042 | 101.37ms | 31.62 tok/s |
| `gemma-2-2b-it` | **58.48%** | 5873 | 4169 | 0 | 10042 | 726.04ms | 237.37 tok/s |
| `gemma-4-E2B-it` | **56.5%** | 5674 | 4368 | 0 | 10042 | 128.35ms | 51.15 tok/s |
| `gemma-3-1b-it` | **32.77%** | 3291 | 6751 | 0 | 10042 | 40.82ms | 121.13 tok/s |
| `Qwen2.5-0.5B-Instruct` | **30.93%** | 3106 | 6936 | 0 | 10042 | 78.42ms | 566.05 tok/s |

---

## MMLU Benchmark

![MMLU Model Comparison](charts/mmlu_comparison.png)

| Model | Accuracy (%) | Passed (True) | Failed (False) | Didn't Finish | Total | Avg Latency | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3.5-4B` | **72.38%** | 10164 | 3878 | 0 | 14042 | 157.73ms | 33.27 tok/s |
| `Qwen2.5-3B-Instruct` | **62.99%** | 8845 | 5197 | 0 | 14042 | 63.34ms | 38.94 tok/s |
| `gemma-4-E2B-it` | **60.76%** | 8532 | 5510 | 0 | 14042 | 1743.48ms | 225.11 tok/s |
| `Qwen3.5-2B` | **58.15%** | 8166 | 5876 | 0 | 14042 | 120.82ms | 80.41 tok/s |
| `gemma-3-4b-it` | **57.86%** | 8125 | 5917 | 0 | 14042 | 246.9ms | 48.04 tok/s |
| `Qwen2.5-1.5B-Instruct` | **57.76%** | 8111 | 5931 | 0 | 14042 | 75.8ms | 158.8 tok/s |
| `gemma-2-2b-it` | **47.02%** | 6602 | 7440 | 0 | 14042 | 989.03ms | 216.41 tok/s |
| `Qwen2.5-0.5B-Instruct` | **34.55%** | 4852 | 9190 | 0 | 14042 | 278.78ms | 726.99 tok/s |
| `gemma-3-1b-it` | **29.67%** | 4166 | 9876 | 0 | 14042 | 775.68ms | 500.3 tok/s |

---

## MMLU_PRO Benchmark

![MMLU_PRO Model Comparison](charts/mmlu_pro_comparison.png)

| Model | Accuracy (%) | Passed (True) | Failed (False) | Didn't Finish | Total | Avg Latency | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemma-4-E2B-it` | **50.4%** | 6064 | 5720 | 248 | 12032 | 4428.78ms | 282.77 tok/s |
| `Qwen3.5-4B` | **48.06%** | 5783 | 6001 | 248 | 12032 | 5707.5ms | 62.31 tok/s |
| `Qwen2.5-3B-Instruct` | **35.72%** | 4298 | 7734 | 0 | 12032 | 122.5ms | 29.97 tok/s |
| `Qwen3.5-2B` | **28.73%** | 3457 | 8575 | 0 | 12032 | 2190.8ms | 118.9 tok/s |
| `gemma-3-4b-it` | **28.19%** | 3392 | 7104 | 1536 | 12032 | 5476.64ms | 98.04 tok/s |
| `Qwen2.5-1.5B-Instruct` | **25.89%** | 3115 | 8917 | 0 | 12032 | 887.86ms | 256.8 tok/s |
| `gemma-2-2b-it` | **14.44%** | 1737 | 10295 | 0 | 12032 | 2614.0ms | 215.2 tok/s |
| `Qwen2.5-0.5B-Instruct` | **13.45%** | 1618 | 10414 | 0 | 12032 | 949.98ms | 623.34 tok/s |
| `gemma-3-1b-it` | **10.61%** | 1277 | 10755 | 0 | 12032 | 2282.01ms | 535.04 tok/s |

---

## TRUTHFULQA Benchmark

![TRUTHFULQA Model Comparison](charts/truthfulqa_comparison.png)

| Model | Accuracy (%) | Passed (True) | Failed (False) | Didn't Finish | Total | Avg Latency | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3.5-4B` | **69.77%** | 570 | 247 | 0 | 817 | 119.59ms | 34.1 tok/s |
| `gemma-4-E2B-it` | **54.1%** | 442 | 375 | 0 | 817 | 1141.11ms | 267.45 tok/s |
| `Qwen2.5-3B-Instruct` | **53.24%** | 435 | 382 | 0 | 817 | 49.15ms | 43.67 tok/s |
| `gemma-3-4b-it` | **48.1%** | 393 | 424 | 0 | 817 | 66.03ms | 51.7 tok/s |
| `Qwen3.5-2B` | **44.55%** | 364 | 453 | 0 | 817 | 51.37ms | 80.19 tok/s |
| `Qwen2.5-1.5B-Instruct` | **35.86%** | 293 | 524 | 0 | 817 | 67.21ms | 323.13 tok/s |
| `gemma-2-2b-it` | **30.23%** | 247 | 570 | 0 | 817 | 918.7ms | 263.56 tok/s |
| `gemma-3-1b-it` | **19.22%** | 157 | 660 | 0 | 817 | 127.87ms | 281.59 tok/s |
| `Qwen2.5-0.5B-Instruct` | **18.97%** | 155 | 662 | 0 | 817 | 63.82ms | 758.42 tok/s |

---
