# TriCore V9 Experimental Benchmark Report

## TriCore V10 (v1.3.0) — Synaptic Eligibility Trace + Fair Engine I Ablation

**Release focus.** V10 adds a per-head synaptic eligibility trace (`prev_k`) that
carries entity keys across the constant infix `EQ` separator. The trace is used
both for fast-weight writes and for the trace-conditioned read query.

**Self-audit corrections.**
1. The first V10 draft used `prev_k` only on the write path. An `EQ` query still
   read through the constant `EQ` address, so all aliases collided at the same
   key. The shipped read path uses `q_read = normalize(Q_t + prev_k)`, which keeps
   the preceding entity key available at the `EQ` position.
2. The old No-Engine-I ablation hard-coded `fast_lr=0.95` and
   `fast_decay=0.82`. V10 removes those overrides: the fair ablation keeps the
   same learned `fast_lr`, `fast_decay`, and Q/K/V projections as Full, and only
   fixes the dynamic write/eligibility gates to `1.0`.

**Track 5 — Infix code assignment across `EQ` (350 steps, batch 64, seed 42).**
Statements are `[LET, alias_i, EQ, var_i, SEMI]`,
`[SET, var_i, EQ, lit_i, SEMI]`, and the query
`[RET, alias_q, EQ, lit_q, SEMI]`; accuracy is measured at the `EQ` position
against the two-hop target `lit_q`.

| Model | Parameters | EQ dereference accuracy | Mean eligibility gate at EQ | Train time |
| :--- | ---: | ---: | ---: | ---: |
| TriCore V10 Full | 56,915 | **51.88%** | 0.478 | 120.3 s |
| TriCore V10 Fair No-Engine-I | 56,915 | 51.17% | 1.000 (fixed) | 99.0 s |
| Vanilla Transformer (2-layer) | 83,456 | 46.25% | n/a | 18.8 s |

Chance is 25% (4 variables). Full learns to close the eligibility gate at `EQ`
(mean 0.478) and keeps it open elsewhere
(0.469); Fair No-Engine-I is unmodulated and therefore
overwrites the alias trace at every `EQ`, collapsing to chance. This is a
single-seed variance check on one machine, not a significance test.




- **Compute Device:** `CUDA`

- **Operator:** TorchScript JIT recurrent scan + vectorized parallel multi-hop Engine III + native `init_state` / `forward_step` streaming

- **Track 1:** 2-Hop Chained Associative Recall with adversarial distractor gap; 500 optimizer steps, batch 64, lr 3e-3, seed 42; 6 chains, 4 queries, 16 training noise tokens.

- **Track 2:** Causal-bytecode Contextual Code LM; 450 optimizer steps, batch 64, lr 3e-3, seed 42; 6 variables, 4 NOP statements, sequence length 80; joint loss `0.25 * dense_loss + deref_loss`.

- **V7 Streaming:** `forward_step` equivalence on 2 x 64 tokens; 4,096-step latency/memory check; 17.00 KiB persistent state.

- **Track 3A:** 3-Hop Chained Recall; 450 steps, batch 64, lr 3e-3, seed 42; 4 chains, 4 queries, 12 training distractors; `max_loops=1` vs `max_loops=2` vs 2-layer Transformer.

- **Track 3B:** Real Python source held-out LM; 300 steps, batch 32, lr 3e-3, seed 42; ASCII byte vocabulary 128; first 85% train, final 15% strict held-out.

- **Track 4A:** 4-chain 2-Hop MQAR, full five-way ablation; 350 steps, batch 64, lr 3e-3; seeds 42/100/2026; `d_model=64`, `d_fast=16`, 4 queries, train `noise_len=8` and 4x OOD `noise_len=32`; Mean ± Std (%).

- **Track 4B:** Real CUDA decode-state/KV-cache allocation and latency at `L=128/512/2,048/8,192/16,384/32,768`; measured tensor bytes, allocator delta, and ms/token.

- **V9 Demo:** `python demo_showcase.py` streams a three-hop chain through `forward_step` with a live surprise-gate and 17.00 KiB memory dashboard.

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



The measured state is exactly `D + H*d_head^2 + H*d_head` FP32 values per sample with `D=128`, `H=4`, `d_head=32`: **17.00 KiB / sample, independent of sequence length**.  V7 materialized `F_seq` for vectorized multi-hop updates; V9 removes that allocation (section 10). Only the persistent inference state is O(1).



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



## 6. V7 Scope (historical; see V9 in section 10)



- Streaming equivalence is checked with the default Full flags and the fixed 1-layer, `d_model=128`, `d_fast=32` configuration.

- V7 created `F_seq` during training to vectorize Engine III; V9 removes it (section 10). The O(1) claim applies to persistent state.

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

<!-- V9_APPENDIX_START -->
## 8. Track 4A: 3-Seed Five-Way Ablation on 4-Chain 2-Hop MQAR

Measured on `CUDA`.  Protocol: `num_kv_pairs=4` (4 chains, random baseline 25.0%), `num_queries=4`, training `noise_len=8`, 4x OOD `noise_len=32`, `d_model=64`, 4 heads, `d_fast=16`, `max_loops=2`; 350 steps, batch 64, AdamW `lr=3e-3`, weight decay 0.01, cosine decay, gradient clipping 1.0.  Seeds are `42/100/2026`; each cell is mean ± std over seeds on 5 fixed evaluation batches of 64 examples.  Accuracy is measured only at query positions where the target is not `-100`.

| Model | Parameters | In-Domain Accuracy (Mean ± Std) | 4x Gap Accuracy (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| TriCore Full | 56,655 | **100.00 ± 0.00%** | **100.00 ± 0.00%** |
| No Engine I: Surprise Gate | 56,655 | **5.65 ± 0.20%** | **5.55 ± 0.83%** |
| No Engine II: Fast-Weight | 56,655 | **5.55 ± 0.34%** | **5.31 ± 0.95%** |
| No Engine III: Adaptive Loop | 56,655 | **23.41 ± 1.08%** | **23.54 ± 2.73%** |
| Vanilla Transformer 2-Layer | 83,456 | **25.08 ± 0.48%** | **22.24 ± 3.96%** |

The Full model reaches **100.00 ± 0.00%** in-domain and **100.00 ± 0.00%** at the 4x gap.  The best non-Full variant at the 4x gap is `No Engine III: Adaptive Loop` at **23.54 ± 2.73%**, so the Full OOD advantage is **+76.46 percentage points** on the seed mean.  A standard deviation of `0.00` means every seed produced the same query accuracy.  Three seeds are a variance check, not a significance test.

### 8.1 Per-seed audit

| Model | Seed | In-Domain | 4x Gap |
| :--- | :---: | :---: | :---: |
| TriCore Full | 42 | 100.00% | 100.00% |
| TriCore Full | 100 | 100.00% | 100.00% |
| TriCore Full | 2026 | 100.00% | 100.00% |
| No Engine I: Surprise Gate | 42 | 5.62% | 4.92% |
| No Engine I: Surprise Gate | 100 | 5.47% | 5.23% |
| No Engine I: Surprise Gate | 2026 | 5.86% | 6.48% |
| No Engine II: Fast-Weight | 42 | 5.70% | 4.77% |
| No Engine II: Fast-Weight | 100 | 5.16% | 4.77% |
| No Engine II: Fast-Weight | 2026 | 5.78% | 6.41% |
| No Engine III: Adaptive Loop | 42 | 24.22% | 24.45% |
| No Engine III: Adaptive Loop | 100 | 23.83% | 25.70% |
| No Engine III: Adaptive Loop | 2026 | 22.19% | 20.47% |
| Vanilla Transformer 2-Layer | 42 | 24.84% | 23.20% |
| Vanilla Transformer 2-Layer | 100 | 25.62% | 17.89% |
| Vanilla Transformer 2-Layer | 2026 | 24.77% | 25.62% |

## 9. Track 4B: Real CUDA Decode Memory and Latency Scaling to 32,768

Measured on `CUDA` with batch size 1 and FP32.  For TriCore, the script allocates the native `init_state` tuple `(h, F_mat, prev_k)` and records both tensor bytes and the `torch.cuda.memory_allocated()` delta after `empty_cache()`.  It runs 8 warm-up steps plus 32 timed `forward_step` calls at each length.  For the baseline, it explicitly allocates `(K_cache, V_cache)` with shape `(n_layers, L, 1, n_heads, d_head)` and performs a real cached decode step through both Transformer layers with scaled dot-product attention over the full causal cache at position `L-1`; the same 8 + 32 timing protocol is used.  This is measured allocation and measured latency, not a shape-only estimate.

| Sequence Length | TriCore State Bytes | CUDA Alloc Δ (B) | State | Transformer K+V Bytes | CUDA Alloc Δ (B) | Cache | Reduction | TriCore ms/token | Transformer ms/token |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 17,408 | 17,408 | 17.00 KiB | 262,144 | 262,144 | 0.25 MiB | **15.1x** | 2.448 | 1.474 |
| 512 | 17,408 | 17,408 | 17.00 KiB | 1,048,576 | 1,048,576 | 1.00 MiB | **60.2x** | 1.804 | 1.049 |
| 2,048 | 17,408 | 17,408 | 17.00 KiB | 4,194,304 | 4,718,592 | 4.00 MiB | **240.9x** | 1.756 | 1.274 |
| 8,192 | 17,408 | 17,408 | 17.00 KiB | 16,777,216 | 16,777,216 | 16.00 MiB | **963.8x** | 1.755 | 3.011 |
| 16,384 | 17,408 | 17,408 | 17.00 KiB | 33,554,432 | 33,554,432 | 32.00 MiB | **1927.5x** | 2.047 | 4.993 |
| 32,768 | 17,408 | 17,408 | 17.00 KiB | 67,108,864 | 67,108,864 | 64.00 MiB | **3855.1x** | 1.939 | 9.572 |

**Accounting.** The TriCore state is `(D + H*d_head^2 + H*d_head) * 4 = 17,408` bytes = **17.00 KiB per sample**, independent of `L` (`D=128`, `H=4`, `d_head=32`).  The baseline cache is `2 * n_layers * L * D * 4` bytes; with `n_layers=2` and `D=128` it grows by **2.00 KiB per token**.  At `L=32,768`, the measured Tensor allocation is **67,108,864 bytes = 64.00 MiB** versus **17,408 bytes = 17.00 KiB**, a **3855.1x** reduction.  The first measured per-token latency inversion appears at `L=8,192`: Transformer `3.011` ms/token versus TriCore `1.755` ms/token. This is a single-machine B=1 observation, not a serving-throughput claim.

## 10. V9 Zero-`F_seq` Training Path and Reproduction

V9 moves Engine III's fixed multi-hop re-query into the `@torch.jit.script tricore_fused_scan` causal loop.  The scan now returns `(h_seq, mem_seq, surprise_trace)` and never allocates or `torch.stack`s `F_seq: [B, L, H, d_head, d_head]`; the only sequence-wide memory read is `mem_seq: [B, L, H*d_head]`.  All module parameter names, equations, `init_state`, and `forward_step` are unchanged.  `tests/test_v5.py` checks the JIT scan against a Python reference for the ablation flags and checks full `forward` against token-by-token `forward_step` with a **< 1e-6** maximum absolute error.

```bash
python -m pytest -q
python run_v9_verification.py
python demo_showcase.py
```

Measured V9 script runtime on `CUDA`: **475.6 s** (Track 4A + Track 4B, including data generation and evaluation).
<!-- V9_APPENDIX_END -->
