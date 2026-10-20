# TriCore V5 Experimental Benchmark Report

- **Compute Device:** `CUDA`
- **Operator:** TorchScript JIT-fused TriCore scan (mathematically equivalent to V4)
- **Track 1:** 2-Hop Chained Associative Recall with adversarial distractor gap; 500 optimizer steps, batch 64, lr 3e-3, seed 42; 6 chains, 4 queries, 16 training noise tokens.
- **Track 2:** Contextual Code LM; 300 optimizer steps, batch 32, lr 3e-3, seed 42; 6 variables, 6 NOP statements, sequence length 82.
- **Open-source packaging:** `ready` (MIT LICENSE, .gitignore, pyproject.toml)

## 1. JIT Fusion and Track 1 Ablation

### 1.1 JIT timing versus V4

| Measurement | Time | Note |
| :--- | :---: | :--- |
| V4 Full, Python per-token loop | 136.0 s | reference run |
| V5 Full, TorchScript fused scan | 125.3 s | timed 500 steps |
| V5 / V4 speed ratio | 1.08x | >1 means faster; <1 means slower |
| V5 first-call warm-up | 0.5 s | excluded from timed training |

The comparison uses the previously recorded V4 Full time of 136.0 s. The warm-up call is excluded from the timed V5 run; timed training includes data generation, forward, backward, and optimizer steps.

### 1.2 Five-way ablation

| Architecture Variant | Parameters | Final Val Loss | In-Domain 2-Hop Recall (%) | 4x Noise-Gap Extrapolation (%) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TriCore Full (3-in-1) | 207,503 | 0.0010 | **100.00%** | **100.00%** | 125.3 |
| TriCore (No Engine I: Surprise Gate) | 207,503 | 0.4066 | **91.88%** | **84.14%** | 97.9 |
| TriCore (No Engine II: Fast-Weight) | 207,503 | 3.0008 | **5.78%** | **4.14%** | 29.2 |
| TriCore (No Engine III: Adaptive Loop) | 207,503 | 1.8660 | **16.48%** | **16.56%** | 59.7 |
| Vanilla Transformer (2-Layer) | 543,744 | 2.3131 | **16.72%** | **15.23%** | 15.9 |

## 2. Track 2: Contextual Code Language Modeling

| Model | Parameters | Validation Loss | Validation PPL | Dereference Acc (%) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TriCore Full (1-Layer) | 207,503 | 0.5914 | 1.81 | **21.09%** | 83.4 |
| Vanilla Transformer (2-Layer) | 543,744 | 0.6753 | 1.96 | **18.75%** | 2.9 |

> Track 2 chance-level dereference accuracy is **16.67%** (6 equally likely literals). The metric is measured at the return alias position. The task is dense next-token prediction over a synthetic imperative grammar; it is not human code.

## 3. Inference State Cache Scaling (Per Sample)

| Sequence Length | Transformer KV Cache (KB) | TriCore O(1) State (KB) | Memory Reduction Factor |
| :---: | :---: | :---: | :---: |
| 128 | 256.00 KB | 17.00 KB | **15.1x smaller** |
| 512 | 1024.00 KB | 17.00 KB | **60.2x smaller** |
| 2,048 | 4096.00 KB | 17.00 KB | **240.9x smaller** |
| 8,192 | 16384.00 KB | 17.00 KB | **963.8x smaller** |
| 32,768 | 65536.00 KB | 17.00 KB | **3855.1x smaller** |

**State accounting.** The single-core TriCore figure is `(D + H*d_head^2 + H*d_head) * 4 / 1024` KiB per sample at FP32, where `D=128`, `H=4`, and `d_head=32`. It includes `h`, `F_mat`, and the causal-shift key cache `prev_k`, and is independent of sequence length. The Transformer figure is `n_layers * 2 * L * D * 4 / 1024` KiB with `n_layers=2`.

**Metric note.** Track 1 recall is measured only at query positions where the target is not `-100`. The 4x extrapolation multiplies the adversarial distractor gap (16 -> 64 tokens), not the total sequence length. Track 1 uses fixed evaluation batches generated with seed 2026/2027; Track 2 uses fixed evaluation batches generated with seed 3030. There is no fixed real-world held-out set.

**Scope.** These are single-seed synthetic experiments. They do not establish natural-language ability, tuned-Transformer parity, multi-seed confidence intervals, or production throughput.

