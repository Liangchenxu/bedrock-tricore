# TriCore V7 Experimental Benchmark Report

- **Compute Device:** `CUDA`
- **Operator:** TorchScript JIT recurrent scan + vectorized parallel multi-hop Engine III + native `init_state` / `forward_step` streaming
- **Track 1:** 2-Hop Chained Associative Recall with adversarial distractor gap; 500 optimizer steps, batch 64, lr 3e-3, seed 42; 6 chains, 4 queries, 16 training noise tokens.
- **Track 2:** Causal-bytecode Contextual Code LM; 450 optimizer steps, batch 64, lr 3e-3, seed 42; 6 variables, 4 NOP statements, sequence length 80; joint loss `0.25 * dense_loss + deref_loss`.
- **V7 Streaming:** `forward_step` equivalence on 2 x 64 tokens; 4,096-step latency/memory check; 17.00 KiB persistent state.
- **Track 3A:** 3-Hop Chained Recall; 450 steps, batch 64, lr 3e-3, seed 42; 4 chains, 4 queries, 12 training distractors; `max_loops=1` vs `max_loops=2` vs 2-layer Transformer.
- **Track 3B:** Real Python source held-out LM; 300 steps, batch 32, lr 3e-3, seed 42; ASCII byte vocabulary 128; first 85% train, final 15% strict held-out.
- **Open-source packaging:** `ready` (MIT LICENSE, .gitignore, pyproject.toml)

## 1. JIT Fusion and Track 1 Ablation

### 1.1 JIT timing versus V4 and V5

| Measurement | Time | Note |
| :--- | :---: | :--- |
| V4 Full, Python per-token loop | 136.0 s | reference run |
| V5 Full, TorchScript fused scan | 125.3 s | reference run |
| V6 Full, vectorized multi-hop scan | 80.0 s | timed 500 steps |
| V6 / V4 speed ratio | 1.70x | >1 means faster; <1 means slower |
| V6 / V5 speed ratio | 1.57x | >1 means faster; <1 means slower |
| V6 first-call warm-up | 0.4 s | excluded from timed training |

The comparison uses the previously recorded V4 Full time of 136.0 s and V5 Full time of 125.3 s. The warm-up call is excluded from the timed V6 run; timed training includes data generation, forward, backward, and optimizer steps.

### 1.2 Five-way ablation

| Architecture Variant | Parameters | Final Val Loss | In-Domain 2-Hop Recall (%) | 4x Noise-Gap Extrapolation (%) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TriCore Full (3-in-1) | 207,503 | 0.0010 | **100.00%** | **100.00%** | 80.0 |
| TriCore (No Engine I: Surprise Gate) | 207,503 | 3.0005 | **5.55%** | **4.14%** | 62.3 |
| TriCore (No Engine II: Fast-Weight) | 207,503 | 3.0008 | **5.78%** | **3.98%** | 36.1 |
| TriCore (No Engine III: Adaptive Loop) | 207,503 | 1.8660 | **16.48%** | **17.50%** | 63.4 |
| Vanilla Transformer (2-Layer) | 543,744 | 2.3131 | **16.72%** | **14.92%** | 18.6 |

## 2. Track 2: Contextual Code Language Modeling

| Model | Parameters | Validation Loss | Validation PPL | Dereference Acc (%) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TriCore Full (1-Layer) | 207,503 | 1.1529 | 3.17 | **100.00%** | 78.5 |
| Vanilla Transformer (2-Layer) | 543,744 | 1.2686 | 3.56 | **19.63%** | 19.5 |

> Track 2 chance-level dereference accuracy is **16.67%** (6 equally likely literals). The metric is measured at the return alias position. The task is dense next-token prediction over a synthetic imperative grammar; it is not human code.

<!-- V7_APPENDIX_START -->
## 3. V7 Native O(1) Streaming Inference

`TriCoreLM.forward(idx)` remains the training path.  The new `init_state` / `forward_step` API carries only `(h, F_mat, prev_k)` between tokens; it does not materialize the training-only `F_seq` tensor and does not retain a KV cache.

### 3.1 Streaming equivalence

| Check | Result |
| :--- | :---: |
| Device | `CUDA` |
| Max absolute logit difference (`forward` vs `forward_step`) | `5.960e-07` |
| Max absolute surprise difference | `1.192e-07` |
| Streaming latency, B=1, 4,096 steps | `2.160 ms/token` |

The equivalence check uses `eval()` mode, a 2 x 64 token batch, and `torch.allclose` with `atol=1e-5`.  The recurrent transition is causally shifted exactly as in `tricore_fused_scan`.

### 3.2 Measured persistent state memory

| Sequence Length | Streamed State Bytes | Per Sample |
| :---: | :---: | :---: |
| 128 | 17,408 | 17.00 KiB |
| 512 | 17,408 | 17.00 KiB |
| 2,048 | 17,408 | 17.00 KiB |
| 4,096 | 17,408 | 17.00 KiB |

The measured state is exactly `D + H*d_head^2 + H*d_head` FP32 values per sample with `D=128`, `H=4`, `d_head=32`: **17.00 KiB / sample, independent of sequence length**.  The training path still materializes `F_seq` for vectorized multi-hop updates; only the persistent inference state is O(1).

## 4. Track 3A: 3-Hop Chained Reasoning

Protocol: 450 optimizer steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`, seed 42.  Each sample contains four shuffled 3-hop chains, 12 distractor tokens, and four queries.  The target is emitted at the query key, so a correct answer requires `k_q -> m1_q -> m2_q -> v_q`.  In-domain evaluation uses a 12-token gap; the 4x test uses a 48-token gap.  Chance is 25% (four values).

| Model | Parameters | In-Domain Val Loss | In-Domain 3-Hop Acc | 4x Gap Extrapolation Acc | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TriCore Full (1-Layer, max_loops=2) | 207,503 | 0.0010 | **100.00%** | **100.00%** | 66.4 s |
| TriCore (max_loops=1, 2-Hop limit) | 203,278 | 1.8395 | **25.39%** | **23.91%** | 62.8 s |
| Vanilla Transformer (2-Layer) | 297,984 | 2.2188 | **23.83%** | **23.44%** | 16.7 s |

`max_loops=2` adds **+74.61 percentage points** over the `max_loops=1` two-hop limit on the in-domain test.  The `max_loops=1` model has only the initial fast-weight read plus one refinement, so it cannot complete the third hop; the Full model's second refinement is the third-hop read.

## 5. Track 3B: Real Python Source Code Held-Out LM

Protocol: 300 optimizer steps, batch 32, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`, seed 42.  The corpus is the concatenation of `tricore/model.py`, `tricore/tasks.py`, and `run_experiment.py`, encoded as ASCII bytes (`vocab_size=128`).  The first 85% is training data; the final 15% is a strict held-out suffix.  Training windows never cross the split.  The Transformer uses a short position table (`max_seq_len=96`) and `d_model=144` to match the TriCore parameter budget instead of a 2,048-position table.

| Model | Parameters | Held-Out Loss | Held-Out PPL | Next-Byte Acc | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TriCore Full (1-Layer) | 223,887 | 2.9240 | 18.616 | **41.47%** | 57.7 s |
| Vanilla Transformer (1-Layer, param-matched) | 218,448 | 2.9667 | 19.428 | **28.15%** | 1.5 s |

This is a small single-seed real-source experiment, not a human-code benchmark.  The held-out suffix is unseen source text, but the corpus is small and comes from this repository.

## 6. V7 Scope

- Streaming equivalence is checked with the default Full flags and the fixed 1-layer, `d_model=128`, `d_fast=32` configuration.
- `F_seq` is still created during training to vectorize Engine III; the O(1) claim applies to persistent autoregressive state.
- Track 3A uses fixed synthetic chains with four values per sample; Track 3B uses a 15% source-file suffix.  Both are single-seed.
- No multi-seed confidence interval, production throughput, natural-language, or tuned-Transformer parity claim is made.

<!-- V7_APPENDIX_END -->

## 7. Inference State Cache Scaling (Per Sample)

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
