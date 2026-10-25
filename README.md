# TriCore V13
<!-- V13_RELEASE_SECTION -->
## V13 (v1.6.0) — Static Long-Term Bindings, Four-Round Hot Mutation, and the Decay-Interference Dilemma

**Release focus.** V13 replaces V12's single overwrite with a stateful mutation task that cannot be solved by lowering a global Hebbian decay. Every sequence contains four shuffled declarations, two **static** variables assigned once at the beginning, two **hot** variables overwritten for **four cyclic rounds**, six adversarial NOPs, and four alias returns: 24 five-token statements, **120 tokens**, four supervised queries, and a four-way chance level of **25.0%**.

1. **Static long-term retention.** Static bindings must survive every later mutation round, NOP, and query. High fast-weight decay preserves them; low decay erases them.
2. **Multi-round hot mutation.** The same hot key is overwritten four times. Hebbian accumulation retains old values; Delta's `V_t - F_{t-1}K_{t-1}` erase term removes the addressed old value before writing the new one.
3. **Decay-Interference Dilemma.** Hebbian has one global decay for both requirements: high decay preserves static bindings but accumulates all four hot values, while low decay preserves the newest hot value but loses the early static bindings. Delta keeps decay high and erases pointwise at each key.
4. **No token-ID prior.** All three TriCore arms use `autonomous_syntax=True` with `separator_token=None`, `syntax_tokens=None`, and `reset_trace_tokens=None`. A hot variable occurs six times (one LET, four SETs, one NOP), so V13 raises `autonomous_entity_max_freq` from 4 to 6; the V12 default remains 4. The rule is still frequency-only and permutation-invariant.
5. **Measured three-seed comparison.** Delta, Hebbian, Fair No-Engine-I, and a vanilla two-layer Transformer share batches, optimizer budget, and 400 steps.

**Protocol.** Seeds `42/100/2026`; 400 steps; batch 64; CUDA; `d_model=64`, `d_fast=16`, `max_loops=2`; 10 evaluation batches of 64. Overall/static/hot accuracy and hot-only stale-value confusion are reported. The Transformer receives no token-role prior; Fair keeps Delta and the learned write path but disables the surprise gate.

| Variant | Params | Overall query acc. | Static acc. | Hot acc. | Hot stale confusion | Val loss |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| **TriCore V13 Delta Full (Zero-Prior)** | 61,851 | 99.30 ± 0.67% | 98.70 ± 1.16% | 99.90 ± 0.18% | 0.08 ± 0.14% | 0.03 ± 0.03 |
| TriCore V13 Hebbian Full (No Delta Erasure) | 61,851 | 46.88 ± 17.15% | 48.05 ± 25.58% | 45.70 ± 8.74% | 27.40 ± 1.24% | 1.18 ± 0.31 |
| TriCore V13 Fair No-Engine-I | 61,851 | 37.89 ± 0.27% | 31.17 ± 0.75% | 44.61 ± 0.36% | 23.64 ± 0.44% | 1.38 ± 0.02 |
| Vanilla Transformer (2-layer) | 83,456 | 45.57 ± 0.29% | 45.91 ± 0.16% | 45.23 ± 0.56% | 19.00 ± 0.35% | 1.09 ± 0.01 |

Per-seed overall accuracy: Delta `99.84/99.49/98.55%`; Hebbian `36.99/66.68/36.95%`; Fair `38.20/37.70/37.77%`; Transformer `45.27/45.59/45.86%`.

**Exact two-tailed Welch tests.** Delta vs Hebbian overall: `t=5.290`, `df=2.006`, **`p=0.0337`**; static: `t=3.426`, `df=2.008`, **`p=0.0752`**; hot: `t=10.734`, `df=2.002`, **`p=0.0085`**; stale-value confusion: `t=-37.849`, `df=2.049`, **`p=0.0006`**. Delta vs Fair overall: `t=147.664`, `df=2.655`, **`p=2.84e-06`**. Delta vs Transformer overall: `t=127.819`, `df=2.746`, **`p=2.89e-06`**. Hebbian vs Fair overall: `t=0.907`, `df=2.001`, **`p=0.4600`**. At the conventional 0.05 level the overall, hot, and stale contrasts reject the null; the static-only contrast is marginal (`p=0.0752`) and is not claimed as significant with three seeds.

**Runtime.** Mean train time per run: Delta Full 258 s, Hebbian Full 225 s, Fair No-Engine-I 191 s, Vanilla Transformer 7 s. Total V13 suite runtime on the reference RTX 3060: **1857.9 s**. No V4-V12 table or result was removed.

**Scope.** Three observations per arm on one closed synthetic grammar are not a population-level claim; Welch values are exact arithmetic on those observations and assume approximate normality. The 100% ceiling is task-specific and does not establish general reasoning. The zero-prior path is a short-sequence frequency prior whose entity bound is explicitly task-scaled for V13.


## V12 (v1.5.0) — Zero-Prior Autonomous Syntax, Delta-Rule Mutation, and 1,048,576-Token CUDA Profiling

**Release focus.** V12 removes the final token-ID prior from the infix task and adds a controlled Delta-rule test:

1. **Zero-prior syntax discovery.** With `autonomous_syntax=True` and `separator_token=None`, `syntax_tokens=None`, `reset_trace_tokens=None`, the LM labels structural tokens from **within-sequence frequency** and **preceding-token frequency** alone. No vocabulary ID is passed. A random bijection of all 64 token IDs leaves the EQ/SEMI masks bit-identical.
2. **Correct eligibility direction.** A closed eligibility gate now reads the frozen preceding key (`q_read = normalize(e q_t + (1-e) prev_k)`) and also stops `prev_k` from advancing. At `alias EQ var`, the next value is therefore written at the alias key: `F(alias)=var`. This is the shifted binding that makes the two-hop `alias -> var -> lit` dereference possible.
3. **Delta rule and Track 7.** `enable_delta_rule` selects the erase term `V_t - F_{t-1}K_{t-1}`; `False` uses the identical decay/learning-rate schedule with a plain Hebbian outer product. Track 7 overwrites a live variable and tests whether the model returns the **new** literal rather than the stale one.
4. **1M-token CUDA profile.** Real `torch.cuda.memory_allocated()` deltas and real cached-decoder latency are measured for both models at `L=128..1,048,576`, not estimated from shapes.
5. **2x3 apex figure.** `results/tricore_v12_apex.png` and the master `results/tricore_vs_transformer.png` summarize zero-prior Track 5, Track 7, memory, and latency.

**Track 5 protocol (seeds 42/100/2026, 350 steps, batch 64, CUDA).** No token-ID masks. Four variables, four queries, four NOPs; the target is read at the EQ position; chance is **25.0%**.

| Variant | Parameters | Val loss (mean ± std) | EQ two-hop acc (mean ± std) | Eligibility gate `EQ` / entity | Mean train time |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V12 Zero-Prior Full** | 61,851 | 0.00237 ± 0.00021 | **100.00 ± 0.00%** | 0.009 / 0.909 | 171.7 s |
| TriCore V12 Fair No-Engine-I | 61,851 | 1.4731 ± 0.0185 | 25.13 ± 0.67% | 1.0000 / 1.0000 | 141.9 s |
| Vanilla Transformer (2-layer) | 83,456 | 1.4397 ± 0.0075 | 25.01 ± 0.63% | n/a / n/a | 13.9 s |

**Interpretation.** The zero-prior Full arm is at **100.00%** on all three seeds while closing the eligibility gate at EQ (mean **0.009**) and leaving entity positions open (**0.909**). The fair No-Engine-I ablation never closes the gate and stays at four-way chance; the untuned Transformer also stays at chance. **Welch two-tailed tests:** Full vs Fair accuracy `p = 2.70×10⁻⁵` (t = 192.52, df = 2.00), Full vs Transformer accuracy `p = 2.34×10⁻⁵` (t = 206.87, df = 2.00), Full vs Fair loss `p = 5.25×10⁻⁵` (t = −137.88, df = 2.00), and Full vs Transformer loss `p = 8.81×10⁻⁶` (t = −333.87, df = 2.00).

**Track 7 protocol (seeds 42/100/2026, 400 steps, batch 64, CUDA).** Variables are declared, assigned an old literal, overwritten with a new literal, distracted by a NOP, and queried through an alias. The target at the query EQ is the **new** literal; chance is **25.0%**. A Hebbian memory accumulates old and new values; the Delta rule erases the old value before writing.

| Variant | Params | New-value accuracy (mean ± std) | Stale-value confusion (mean ± std) | Val loss (mean ± std) | Mean train time |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V12 Delta Full** | 61,851 | **100.00 ± 0.00%** | **0.00 ± 0.00%** | **0.00227 ± 0.00012** | 175.0 s |
| TriCore V12 Hebbian Full | 61,851 | 73.71 ± 42.22% | 1.06 ± 0.95% | 0.6164 ± 0.9452 | 167.2 s |
| TriCore V12 Fair No-Engine-I Delta | 61,851 | 25.12 ± 0.41% | 0.00 ± 0.00% | 1.7284 ± 0.0234 | 141.5 s |
| Vanilla Transformer (2-layer) | 83,456 | 24.28 ± 0.41% | 0.00 ± 0.00% | 1.4840 ± 0.0674 | 3.3 s |

Per-seed new-value accuracy: Delta **100.00 / 100.00 / 100.00%**; Hebbian **98.67 / 97.50 / 24.96%**; Fair No-Engine-I Delta **24.65 / 25.39 / 25.31%**; Transformer **24.22 / 24.73 / 23.91%**. Delta is stable on all three seeds and never returns the stale literal. Hebbian is near-perfect on seeds 42 and 100 but collapses to chance on seed 2026 (24.96%, validation loss 1.7077), so its three-seed mean (73.71 ± 42.22%) is not a robust effect. **Exact two-tailed Welch tests:** Delta vs Fair new value `t=318.03`, `df=2.00`, **`p=9.89e-06`**; Delta vs Transformer `t=316.76`, `df=2.00`, **`p=9.97e-06`**; Delta vs Hebbian `t=1.078`, `df=2.00`, **`p=0.394`** (not significant), and Delta vs Hebbian stale value `p=0.194`. The defensible claim is that the dynamic-write engine is necessary (Delta, and Hebbian when it trains, are far above the 25.0% chance floor) while Delta is the variant that stayed stable on every seed. With three observations per arm this does **not** establish a population-level Delta > Hebbian advantage.

**Track 4B protocol (B=1, FP32, RTX 3060).** The baseline is a real two-layer cached decoder: K/V are allocated, written, and attended over the full causal cache. TriCore runs the native `forward_step` state path. Timed/warm-up steps are 32/8 below 65K, 16/4 at 65K–131K, and 8/2 at 262K–1M. `Alloc` is the measured allocator delta after `empty_cache()`, not a formula.

| Length | TriCore state (B) | Transformer K+V cache (B) | Cache (binary) | Reduction | TriCore ms/token | Transformer ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 17,408 | 262,144 | 256.00 KiB | 15.1× | 2.596 | 1.277 |
| 512 | 17,408 | 1,048,576 | 1.00 MiB | 60.2× | 3.091 | 1.408 |
| 2,048 | 17,408 | 4,194,304 | 4.00 MiB | 240.9× | 3.249 | 1.675 |
| 8,192 | 17,408 | 16,777,216 | 16.00 MiB | 963.8× | 3.151 | 3.339 |
| 16,384 | 17,408 | 33,554,432 | 32.00 MiB | 1,927.5× | 2.893 | 5.311 |
| 32,768 | 17,408 | 67,108,864 | 64.00 MiB | 3,855.1× | 2.784 | 10.519 |
| 65,536 | 17,408 | 134,217,728 | 128.00 MiB | 7,710.1× | 3.221 | 18.257 |
| 131,072 | 17,408 | 268,435,456 | 256.00 MiB | 15,420.2× | 3.245 | 35.382 |
| 262,144 | 17,408 | 536,870,912 | 512.00 MiB | 30,840.5× | 3.039 | 71.015 |
| 524,288 | 17,408 | 1,073,741,824 | 1.00 GiB | 61,680.9× | 2.879 | 137.640 |
| **1,048,576** | **17,408** | **2,147,483,648** | **2.00 GiB** | **123,361.9×** | **3.794** | **274.275** |

TriCore state is exactly **17,408 B = 17.00 KiB per sample at every measured length**. At `L=1,048,576` the real cached Transformer allocates **2,147,483,648 B (2.00 GiB)** of K/V, or **123,361.9×** the TriCore state, and decodes at **274.275 ms/token** versus **3.794 ms/token** for TriCore (**72.30× slower**). The measured latency crossover is at `L=8,192` (3.339 vs 3.151 ms/token); below it the small cached Transformer is faster. Timed/warm-up steps are 32/8 through 32,768, 16/4 at 65,536–131,072, and 8/2 at 262,144–1,048,576. These are allocator deltas and timings on one RTX 3060 at batch 1, not serving throughput.

> **Scope.** Track 5 and Track 7 are three-seed Welch checks with three observations per arm, not population inference. Track 4B is a single-machine B=1 measurement, not serving throughput. Autonomous discovery is a short-sequence role prior for regular closed syntax, not a general parser; it is disabled in the 1M profile. The 100% scores are on closed synthetic grammars and do not establish general reasoning.

**V12 runtime:** **493.8 s** for the measured V12 run (Track 5 three seeds + Track 7 four arms × three seeds + Track 4B eleven lengths), on the RTX 3060 reference machine.


## V11 (v1.4.0) — Four-Hop Chained Recall, Three-Seed Welch Tests, and 128K CUDA Profiling

**Release focus.** V11 is an evaluation-and-operator release built on the V10 synaptic core, with one additive delimiter prior:

1. **Track 6 — single-forward four-hop recall.** A shuffled four-tier fact layout is followed by a 12-token adversarial distractor gap and four `[key, 0]` queries. The target requires `k_q -> m1_q -> m2_q -> m3_q -> v_q` in one forward pass. V11 adds `reset_trace_tokens=(0,)`: at every statement delimiter the per-head eligibility key `prev_k` is zeroed, so each fact triple begins with a clean entity trace. The boolean `reset_mask` is threaded through both `tricore_fused_scan` and `forward_step`; a unit test checks full-vs-streaming parity under the reset path at `atol=1e-4`.
2. **Track 5 — three-seed significance.** The infix-`EQ` two-hop dereference is repeated over seeds `42/100/2026` and reported with Welch's unequal-variance two-sample t-test (implemented with the regularized incomplete Beta function; no SciPy dependency).
3. **Track 4B Extended — real CUDA profiling to 131,072 tokens.** Measured `torch.cuda.memory_allocated()` deltas and cached-decode latency for both models, not formula estimates.
4. **Six-panel master figure.** `results/tricore_v11_masterpiece.png` is a 2x3 conference-style summary of depth scaling, Track 5, Track 4A, memory, latency, and code modeling.

**Track 6 protocol (single seed 42, 500 steps, batch 64, CUDA).** Four chains and four queries; sequence length 68; four shuffled fact tiers of `[key, value, 0]` triples; 12 distractor tokens; 4x OOD gap uses 48 distractors (sequence length 104). Chance is **25.0%**.

| Track 6 variant | Parameters | In-domain | 4x-OOD | Query router `[h1,h2,h3,h4]` | Train time |
| :--- | ---: | ---: | ---: | :--- | ---: |
| **TriCore Full (1-layer, `max_loops=3`)** | 236,436 | **100.00%** | **100.00%** | 0.005 / 0.004 / 0.003 / **0.988** | 214.2 s |
| TriCore (1-layer, `max_loops=2`, 3-hop limit) | 232,211 | 24.95% | 24.76% | 0.002 / **0.997** / 0.001 / 0.000 | 184.0 s |
| TriCore (1-layer, `max_loops=1`, 2-hop limit) | 227,986 | 25.73% | 26.27% | 0.019 / **0.981** / 0.000 / 0.000 | 148.8 s |
| Vanilla Transformer (2-layer, `d_model=128`, 4 heads) | 306,176 | 24.41% | 25.78% | n/a | 4.0 s |

**Interpretation.** Only the `max_loops=3` model crosses four edges in one forward pass: it reaches **100.00% in-domain and 100.00% at the 4x gap**, with the router placing **98.8%** of its query-position mass on hop 4. The `max_loops=2` and `max_loops=1` models collapse onto their last available slot (0.997 on slot 2, 0.981 on slot 1) but remain at chance on the task, which is the expected depth-limit signature. The Transformer is also at chance at this budget. This is a **single-seed controlled comparison**, not a significance test.

**Track 5 protocol (seeds 42/100/2026, 350 steps, batch 64, CUDA).** Declaration `[LET, alias_i, EQ, var_i, SEMI]`, assignment `[SET, var_i, EQ, lit_i, SEMI]`, adversarial `[NOP, var_i, EQ, SEMI, SEMI]`, query `[RET, alias_q, EQ, SEMI, SEMI]`; the two-hop target is read at `EQ`. The target is one of 16 literals, with 25.0% four-way chance after the four SET literals are identified.

| Architecture variant | Parameters | Val loss (mean ± std) | Infix `EQ` 2-hop acc (mean ± std) | Eligibility gate at `EQ` / entity | Mean train time |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V11 Full (eligibility trace)** | 60,819 | **0.00218 ± 0.00022** | **100.00 ± 0.00%** | 0.0032 / 0.9054 | 156.3 s |
| TriCore V11 Fair No-Engine-I (unmodulated gates) | 60,819 | 1.4682 ± 0.0186 | 24.80 ± 1.12% | 1.0000 / 1.0000 | 129.0 s |
| Vanilla Transformer (2-layer) | 83,456 | 1.4397 ± 0.0075 | 25.01 ± 0.63% | n/a / n/a | 14.8 s |

**Welch tests (two-tailed, Welch-Satterthwaite df).** Full vs Fair on accuracy: `t=116.649`, `df=2.000`, **`p=7.35e-05`**. Full vs Transformer on accuracy: `t=206.869`, `df=2.000`, **`p=2.34e-05`**. Full vs Fair on validation loss: **`p=5.34e-05`**; Full vs Transformer on loss: **`p=8.81e-06`**. With only three seeds these are exact arithmetic on three observations per arm, not a population-level claim; they assume approximate normality and equal-sized groups.

**Track 4B Extended protocol (single machine, batch 1, FP32).** The Transformer baseline is a real cached 2-layer decode: K/V tensors are allocated, written, and attended over the full causal cache at every length. `Alloc` is the measured `torch.cuda.memory_allocated()` delta after `empty_cache()`. Latency is 8 warm-up + 32 timed steps below 65K, and 4 + 16 steps at 65K/128K.

| Length | TriCore state | Transformer K+V cache | Reduction | TriCore ms/token | Transformer ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 17,408 B | 262,144 B | 15.1x | 2.971 | 1.415 |
| 512 | 17,408 B | 1,048,576 B | 60.2x | 2.571 | 1.387 |
| 2,048 | 17,408 B | 4,194,304 B | 240.9x | 2.866 | 1.549 |
| 8,192 | 17,408 B | 16,777,216 B | 963.8x | 2.922 | 3.865 |
| 16,384 | 17,408 B | 33,554,432 B | 1,927.5x | 3.408 | 5.421 |
| 32,768 | 17,408 B | 67,108,864 B | 3,855.1x | 2.767 | 9.735 |
| 65,536 | 17,408 B | 134,217,728 B | 7,710.1x | 2.719 | 18.275 |
| **131,072** | **17,408 B** | **268,435,456 B (256.00 MiB)** | **15,420.2x** | **2.737** | **35.548** |

**Interpretation.** The TriCore state is **17,408 bytes = 17.00 KiB per sample** at every length. At 128K the Transformer cache is **256.00 MiB**, a **15,420.2x** reduction in measured tensor bytes. TriCore latency stays in a narrow **2.57-3.41 ms/token** band while the Transformer grows from **1.39 to 35.55 ms/token**; the first measured crossover is at `L=8,192`, and at `L=131,072` TriCore is **12.99x** faster per token. This is a measured B=1 memory/latency result on one RTX 3060, not serving throughput.

**Scope.** Track 5 is a three-seed Welch check. Track 6 and Track 4B are single-seed/single-machine controlled comparisons. The Track 6 TriCore variants receive a constant token-role prior (`separator_token=0`, `syntax_tokens=(0,)`, `reset_trace_tokens=(0,)`); this is an explicit input mask, not a learned embedding, and it is applied identically to all three depth budgets. The 100% Track 6 score is on a closed synthetic pointer-chain task and does not establish general reasoning.

## V10 (v1.3.0) — Synaptic Eligibility Trace + 100% Fair Engine I Ablation

**Release focus.** V10 adds a per-head synaptic eligibility trace (`prev_k`) that
carries entity keys across the constant infix `EQ` separator. The trace is used by
the fast-weight write path and by the trace-conditioned read query
`q_read = normalize(elig_gate * q_t + (1 - elig_gate) * prev_k)`.

**Self-audit corrections.**
1. The earlier draft used `prev_k` only on the write path; an `EQ` read still used the
   constant `EQ` address, so aliases collided. The shipped read path latches the
   preceding entity key at `EQ` and queries with it.
2. The old No-Engine-I ablation hard-coded `fast_lr=0.95` and `fast_decay=0.82`.
   The fair ablation keeps the same learned `fast_lr`, `fast_decay`, and Q/K/V
   projections as Full, and fixes only the dynamic write/eligibility gates to `1.0`.
3. The old Track 5 query accidentally wrote the answer `lit_q` into the query input.
   The query is now `[RET, alias_q, EQ, SEMI, SEMI]`; the target is read at the `EQ`
   position (`pos + 2`) and never appears in the query.

**Track 5 protocol (350 steps, batch 64, seed 42, CUDA).**
- Declaration: `[LET, alias_i, EQ, var_i, SEMI]`
- Assignment: `[SET, var_i, EQ, lit_i, SEMI]`
- Adversarial distractor: `[NOP, var_i, EQ, SEMI, SEMI]`
- Query: `[RET, alias_q, EQ, SEMI, SEMI]`; the two-hop target `lit_q` is evaluated at `EQ`.
- `EQ` closes both the eligibility gate and the write gate; `SEMI/LET/SET/RET/NOP`
  close only the write gate, so statement boundaries refresh `prev_k` without
  overwriting entity memory.
- The target is one of 16 literal candidates. Uniform 16-way chance is **6.25%**;
  after identifying the four SET literals, the four-way chance is **25.0%**.

| Architecture variant | Parameters | Val loss | Infix `EQ` 2-hop deref acc (%) | Mean eligibility gate at `EQ` / entity | Train time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **TriCore V10 Full (with eligibility trace)** | 60,819 | 0.0021 | **100.00%** | 0.0036 / 0.9011 | 166.8 |
| **TriCore V10 Fair No-Engine-I (unmodulated gates)** | 60,819 | 1.4884 | **24.18%** | 1.0000 / 1.0000 | 137.6 |
| **Vanilla Transformer (2-layer)** | 83,456 | 1.4349 | **25.47%** | n/a / n/a | 15.4 |

**Interpretation.** Full reaches **100.00%** and closes the eligibility
gate at `EQ` (mean 0.0036) while leaving entity positions open
(mean 0.9011). The fair No-Engine-I ablation reaches
**24.18%**; without a gate it overwrites the alias trace at
every `EQ`. The attention baseline remains far below Full on this small training budget; this is a single-seed observation, not a claim about all Transformers. The run is one seed and one machine, so this is a
variance check, not a significance test.

---

**A JIT-recurrent, attention-free sequence operator with zero-prior autonomous syntax discovery, zero-`F_seq` training, native O(1) streaming inference, a 17 KiB constant-size state, vectorized multi-hop synaptic chaining, Delta-rule live overwrite, and causal bytecode / real-source code tasks.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TorchScript](https://img.shields.io/badge/TorchScript-JIT%20fused-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/jit.html)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen)](#8-verification)
[![V12](https://img.shields.io/badge/V12-zero--prior%20%2B%201M%20CUDA-blue)](#v12-v150--zero-prior-autonomous-syntax-delta-rule-mutation-and-1048576-token-cuda-profiling)

> **Status:** experimental research reference; **V12** adds zero-prior autonomous syntax discovery, a selectable Delta rule, stateful live-overwrite evaluation, and real CUDA profiling to `L=1,048,576`. The V10 synaptic eligibility trace remains the operator core, the V9 zero-`F_seq` training path remains intact, and all V6/V7/V8/V9/V10/V11 records are preserved. Read [Limitations](#9-limitations) before generalizing.

> **TL;DR (Track 1, 2-Hop MQAR).** Full TriCore reaches **100.00% in-domain recall** and **100.00% recall under a 4x longer adversarial distractor gap** with a **17.00 KiB O(1) state**. Removing Engine I gives **5.55% / 4.14%**; removing Engine II gives **5.78% / 3.98%**; removing Engine III gives **16.48% / 17.50%**. A 2-layer Transformer reaches **16.72% / 14.92%**, which is chance at this budget. The V6 Full run takes **80.0 s** for 500 steps versus **136.0 s** for V4 (**1.70x** faster) and **125.3 s** for V5 (**1.57x** faster), excluding a 0.4 s warm-up. In the 4-chain V9 3-seed check, Full reaches **100.00 ± 0.00%** at the 4x gap versus **22.24 ± 3.96%** for the Transformer, **23.54 ± 2.73%** without Engine III, and **5.31 ± 0.95%** without Engine II; measured decode state is **17,408 B** at every length versus a 64.00 MiB KV cache at `L=32,768` (**3,855.1x**).

> **TL;DR (Track 2, Causal Bytecode LM).** TriCore (1 layer) reaches validation **PPL 3.17** and **100.00% dereference accuracy** on the synthetic `alias -> var -> lit` return task; the 2-layer Transformer reaches **PPL 3.56** and **19.63%**. The 6-way chance level is **16.67%**. The near-perfect score is on a closed synthetic bytecode grammar, not evidence of general code understanding.

> **TL;DR (V7 Streaming).** `TriCoreLM.forward_step` matches the full `forward` pass with **max absolute logit error 5.96e-07** and **max surprise error 1.19e-07** on CUDA. Persistent state is exactly **17,408 bytes = 17.00 KiB per sample** at `L=128, 512, 2,048, 4,096`; measured streaming latency is **2.160 ms/token**. V9 removes the old training-only `F_seq` materialization entirely.

> **TL;DR (Track 3A, 3-Hop).** Full `max_loops=2` reaches **100.00% in-domain** and **100.00% at the 4x gap**. The `max_loops=1` two-hop limit falls to **25.39% / 23.91%**, and the 2-layer Transformer to **23.83% / 23.44%**; the Full advantage over `max_loops=1` is **+74.61 percentage points**.

> **TL;DR (Track 3B, Real Python Source).** On the strict 15% held-out source suffix, one-layer TriCore reaches **loss 2.9240, PPL 18.616, next-byte accuracy 41.47%**; the parameter-matched one-layer Transformer reaches **2.9667, 19.428, 28.15%**. TriCore takes **57.7 s** versus **1.5 s**, so this is a quality/budget observation, not a throughput claim.

> **中文速览.** V7 保留 V6 全部公式、权重命名和 V6 双赛道数据，新增原生 `init_state` / `forward_step` 流式接口：完整序列与逐 token 解码的最大 logit 误差 **5.96e-07**，状态恒为 **17,408 bytes = 17.00 KiB / 样本**，4,096 步解码实测 **2.160 ms/token**。Track 3A 中 Full `max_loops=2` 达到 **100.00% 域内 / 100.00% 四倍间隔外推**，而 `max_loops=1` 仅 **25.39% / 23.91%**、2 层 Transformer 为 **23.83% / 23.44%**，证明第二次 refinement 真正走通第三跳。Track 3B 在真实仓库 Python 源码的严格 15% held-out 后缀上，TriCore 为 **PPL 18.616 / 41.47% next-byte acc**，参数匹配 Transformer 为 **19.428 / 28.15%**；TriCore 训练 **57.7 s**、Transformer **1.5 s**，因此不是吞吐优势。结论仍限单 seed 小规模任务，不代表自然语言或通用代码能力。

---

## Contents

- [1. The problem](#1-the-problem)
- [2. Architecture](#2-architecture)
- [3. What changed in V7-V12](#3-what-changed-in-v7-v12)
- [4. Quick start](#4-quick-start)
- [5. Benchmark](#5-benchmark)
- [6. Repository layout](#6-repository-layout)
- [7. Experimental protocol](#7-experimental-protocol)
- [8. Verification](#8-verification)
- [9. Limitations](#9-limitations)
- [10. Roadmap](#10-roadmap)
- [11. Citation](#11-citation)
- [12. Contributing](#12-contributing)

---

## 1. The problem

Attention is a powerful sequence mixer, but it carries two structural costs:

1. **Quadratic sequence mixing.** Self-attention compares every token with every other token, so compute and activation memory grow as `O(L^2)`.
2. **Linear inference cache.** Every layer stores keys and values for every past token. The KV cache grows as `O(L)` and dominates long-context decoding.

Associative recall adds a third requirement. A single fast-weight lookup can learn `key -> value`; it cannot follow `key -> bridge -> value` unless the model can use the first retrieved item as the query for a second retrieval. Stacking another physical layer is one answer. TriCore tests another:

> **Can one constant-state core perform multiple associative reads by looping the retrieved latent back through the same fast weights?**

V6 asks two practical questions that V5 could not answer:

> **Can the multi-hop refinement be vectorized across all timesteps instead of re-running a serial per-token loop? And can Track 2 force a causal pointer dereference instead of leaking the answer through a surface token?**

The answers are deliberately narrow. On a controlled 2-hop task, a causal shifted delta-rule memory plus a latent re-query loop solves the task with an O(1) inference state. V6 removes the per-token hop loop by batching the refinement over all positions, but the state recurrence is still serial, the implementation is still a compiled graph of tensor operations rather than a custom CUDA kernel, and it remains much slower than a 2-layer Transformer on these short sequences. The second track is a synthetic bytecode grammar, not human code.

### Design goals

| Goal | V7/V8/V9 mechanism | Evidence |
| :--- | :--- | :--- |
| No attention over positions | Only the hop router uses softmax, over `max_loops + 1` slots | Five-way Track 1 table |
| O(1) inference state | `h`, `F_mat`, `prev_k` | 17.00 KiB per sample; 3,855.1x smaller than a 2-layer KV cache at 32K |
| Native streaming decode | `init_state` / `forward_step`, no `F_seq` in the persistent state | `5.96e-07` max logit error; `2.160 ms/token` |
| Multi-hop with one core | Latent re-query of the same fast weights, vectorized over all positions | Full versus No-Engine-III ablation |
| Selective writes | Learned surprise-gated write protection | Full versus No-Engine-I at this seed |
| Zero-prior syntax roles | Within-sequence frequency + causal adjacency; no token IDs | Track 5/7 permutation-invariant mask test |
| Live variable overwrite | Delta-rule erase versus Hebbian accumulation | V12 Track 7 |
| Faster multi-hop | JIT state loop + batched 5-D refinement | V6 `80.0 s` vs V5 `125.3 s` vs V4 `136.0 s` |
| Second task | Causal bytecode LM with alias dereference at `pos + 1` | `PPL 3.17`; `100.00%` dereference vs `16.67%` chance |
| Open-source packaging | MIT license, `pyproject.toml`, `.gitignore`, tests | [Repository layout](#6-repository-layout) |

---

## 2. Architecture

TriCore is a residual sequence block with three independently ablatable engines. `D=128`, `H=4`, and `d_head=32` are the reference settings. There is no positional embedding and no attention over sequence positions.

### 2.1 Causal input path

```math
u_t = \operatorname{LN}(x_t), \qquad \tilde{u}_t = \operatorname{SiLU}(W_u u_t + b_u)
```

```math
a_t = \operatorname{normalize}(W_a u_t)
```

```math
Q_t = \operatorname{normalize}(a_t + 0.15 W_q u_t), \quad
K_t = \operatorname{normalize}(a_t + 0.15 W_k u_t), \quad
V_t = a_t + 0.15 W_v u_t
```

```math
w_t^{\text{hop}} = \operatorname{softmax}(W_r u_t)
```

`Q/K/V` have shape `[B, L, H, d_head]`; the hop router has shape `[B, L, L_max+1]`. All L2 normalization is over the head dimension. The shared normalized address `a_t` keeps a retrieved bridge compatible with the next query.

### 2.2 Engine I — surprise-gated state collapse and write protection

Initial state: `h_0 = 0`, shape `[B, D]`.

```math
e_t = |\tilde{u}_t - h_{t-1}|, \qquad g_t = \sigma(W_s u_t + e_t)
```

```math
s_t = \frac{1}{D}\sum_j g_{t,j}, \qquad
\omega_t = \sigma(W_g u_t + s_t - 0.5)
```

```math
\eta_{\text{eff},t} = \eta \odot \omega_t
```

```math
h_t = \lambda \odot (1 - 0.5 g_t) \odot h_{t-1} + g_t \odot \tilde{u}_t
```

`lambda` is `sigmoid(base_decay_logit)` with shape `[D]`; `eta` is `sigmoid(fast_lr_logit)` with shape `[1, H, 1, 1]`. The write gate is per head and shared across that head's matrix. When `enable_surprise_gate=False`, the block uses `g_t=0.5`, `s_t=0.5`, `eta_eff=1.0`, `decay=1.0`, and no learned write gate.

### 2.3 Engine II — causal shifted fast weights

Initial state: `F_0 = 0`, shape `[B, H, d_head, d_head]`; `prev_k = 0`, shape `[B, H, d_head, 1]`.

For `t > 0`:

```math
F_t = \lambda_f \odot F_{t-1}
+ \eta_{\text{eff},t} \odot (V_t - F_{t-1}K_{t-1})K_{t-1}^{\top}
```

The read happens after the write:

```math
r_t^{(0)} = F_t Q_t
```

At `t=0`, no write occurs and the first read is zero. At the end of each step, `prev_k = K_t`, so the next write binds the previous key to the current value. The write is causal and shifted: in a `[k_i, m_i, 0]` block it binds `k_i -> m_i`; in a `[m_i, v_i, 0]` block it binds `m_i -> v_i`.

### 2.4 Engine III — latent multi-hop synaptic equilibrium loop

```math
r^{(0)} = r_t^{(0)}, \qquad r_{\text{accum}} = w_{t,0}^{\text{hop}} r^{(0)}
```

For `i = 0, \ldots, L_{\max}-1`:

```math
q^{(i+1)} = \operatorname{normalize}\left(r^{(i)} + H_i r^{(i)}\right)
```

```math
r^{(i+1)} = F_t q^{(i+1)}, \qquad
r_{\text{accum}} \leftarrow r_{\text{accum}} + w_{t,i+1}^{\text{hop}} r^{(i+1)}
```

`r^(i)` is `[B, H, d_head, 1]`; `H_i` is `[1, H, d_head, d_head]`; normalization is L2 over `dim=-2`. With `max_loops=2`, the initial read is hop 1, the first re-query is hop 2, and the second re-query is an extra latent refinement. The loop count is fixed; the router changes weights, not depth.

### 2.5 Readout

```math
z_t = h_t + W_o \operatorname{vec}(r_t^{\text{out}})
```

```math
z_t \leftarrow z_t + s_t \odot \operatorname{MLP}(\operatorname{LN}(z_t)), \qquad
y_t = x_t + \operatorname{LN}(z_t)
```

When `enable_adaptive_loop=False`, `r_t^out = r_t^(0)` and the refinement MLP is skipped. When `enable_fast_weight=False`, Engine II is skipped, `z=h_seq`, and Engine III has no memory to read. All modules are constructed regardless of flags, so all TriCore variants report the same parameter count.

### 2.6 JIT-recurrent state and vectorized multi-hop pass

V6 splits the operator into two regions. The causal O(1) state evolution stays in a scripted loop because it is genuinely sequential. The multi-hop equilibrium pass is vectorized across every timestep:

```python
@torch.jit.script
def tricore_fused_scan(
    u_feat, u_surp_bias, w_gate_logits, Q, K, V,
    base_decay, fast_decay, fast_lr,
    enable_surprise_gate, enable_fast_weight,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

The recurrent state itself is `h [B,D]`, `F_mat [B,H,d_head,d_head]`, and `prev_k [B,H,d_head,1]`. V9 fuses Engine III's fixed re-query into the same scripted causal loop, so the scan no longer returns or stacks per-position `F` snapshots. The training scan returns:

- `h_seq: [B, L, D]`
- `mem_seq: [B, L, H*d_head]` — the routed multi-hop memory read (hop-1 only when `enable_adaptive_loop=False`; zeros when `enable_fast_weight=False`)
- `surprise_trace: [B, L]`

The persistent inference state remains O(1) in `L`, and no KV-style cache is carried across decoding steps.

`TriCoreBlock` then loops only over the fixed `max_loops` budget, not over tokens. Each refinement step normalizes the previous read, projects it into query space, and applies `torch.matmul(F_seq, q_next)` to all `L` positions at once. The Python-level loop count is therefore `max_loops`, while the work inside each hop is a batched 5-D matrix product.

The JIT path is mathematically equivalent to the V4/V5 Python reference. `tests/test_v5.py` compares the scripted function against an explicit Python reference for all-engines, each single-engine ablation, and all-disabled, asserting `allclose` on all four outputs.

This is **not** a custom CUDA kernel and it is **not** a parallel prefix scan. The recurrent state update remains serial; only the multi-hop refinement is vectorized across sequence positions. It still pays kernel-launch and small-matrix costs, so the measured speedup is a runtime cleanup rather than a hardware-optimal scan.

---

## 3. What changed in V7-V13

V13 is additive to V12. It adds the static-plus-four-round hot mutation generator (`num_rounds=4`), an `autonomous_entity_max_freq` discovery bound for repeated hot keys (default 4, V13 uses 6), and a resumable three-seed Delta/Hebbian/Fair/Transformer benchmark with static, hot, and stale-value metrics. The V12 zero-prior default and all V4-V12 equations are preserved.

V12 is additive to V11. It adds zero-prior autonomous syntax discovery (`autonomous_syntax=True`), an explicit `enable_delta_rule` toggle, Track 7 stateful live-overwrite evaluation, real CUDA profiling to `L=1,048,576`, and a new 2x3 apex figure. No token-ID separator, syntax, or reset masks are passed in the V12 runs. The V10/V11 equations and all recurrent-state shapes are preserved; the only new parameters are the small gate-role lookup/bypass path, which is not part of the 17,408-byte inference state.

V11 is additive to V10. It introduces the optional `reset_trace_tokens` delimiter prior (no new parameters), Track 6 four-hop single-forward chaining, a three-seed Track 5 with exact Welch tests, real CUDA profiling to `L=131,072`, and a 2x3 master figure. The V10/V9 equations and all parameter names are preserved.

V10 is additive to V9. It adds the per-head synaptic eligibility trace `prev_k`, the trace-conditioned read query, the two-layer token-role gate networks, and the 100% fair Engine I ablation. Parameter names from V6-V9 are preserved.

V9 is strictly additive to V8 at the API level and removes a training-time allocation: Engine III now runs inside `tricore_fused_scan`, so the scan returns `(h_seq, mem_seq, surprise_trace)` instead of `(h_seq, r1_seq, F_seq, surprise_trace)`. The V6/V7 report rows and all parameter names are preserved. V9 also upgrades Track 4A to a 4-chain, five-way, 3-seed ablation and extends real CUDA profiling to `L=32,768`. The V9 runner is `run_v9_verification.py`; the old name `run_v8_verification.py` remains as a compatibility alias.

V8 is strictly additive to V7; the V6/V7 runtime changes below remain in force. V8 adds `run_v8_verification.py` (3-seed Track 4A + measured CUDA Track 4B), `demo_showcase.py` (one-command streaming demo), and the `1.1.0` package metadata. It does not change the model equations or the V6/V7 report rows.

V7 is strictly additive to V6; the V6 runtime changes below remain in force.

| Area | V5 | V6 |
| :--- | :--- | :--- |
| Recurrence | One serial TorchScript loop for state evolution and every hop | JIT loop for the O(1) causal state, then a fully batched multi-hop pass over all `L` positions |
| Engine III | Fixed re-query inside the scripted causal scan | The current `F_mat` is re-queried per timestep; only `mem_seq: [B, L, H*d_head]` is returned, never `F_seq` snapshots |
| Track 2 task | `EQ`-based statements that collided in the key/value hash | Causal bytecode `[LET, alias, var, SEMI]`, `[ALIAS, var, lit, SEMI]`, `[NOP, var, SEMI, SEMI]`, `[RET, alias, lit, SEMI]` |
| Dereference target | Predicted at the literal position | Predicted at `pos + 1` (the return alias position), the first causal position where the pointer value is required |
| Public math | V4 equations | Identical equations, state shapes, and parameter count |
| Evidence | Two synthetic tracks | Two synthetic tracks with V6 timings and causal-pointer accuracy |
| Tests | JIT equivalence on three outputs | Four-output equivalence (`h`, hop-1 read, `F` snapshots, surprise) plus causal bytecode backward |

The model equations, parameter count (`207,503`), and O(1) inference-state accounting are unchanged. V6 is a runtime and evidence upgrade, not a new operator.

### 3.1 V7 additions

| Area | V7 |
| :--- | :--- |
| Streaming API | `TriCoreBlock.init_state` / `forward_step` and `TriCoreLM.init_state` / `forward_step`; one causal token per call |
| Persistent decode state | `(h, F_mat, prev_k)` only; `D + H*d_head^2 + H*d_head = 17,408` FP32 bytes/sample |
| Streaming equivalence | `eval()` full-sequence logits and per-token step logits match with max absolute error `5.96e-07` on CUDA |
| Track 3A | Three-hop chained recall; `max_loops=1` is limited to two hops, `max_loops=2` performs the third re-query |
| Track 3B | Real repository Python sources, ASCII byte vocabulary 128, first 85% train / final 15% strict held-out |
| Tests | `tests/test_v7_streaming.py`: streaming equivalence, constant 17 KiB state, 3-hop and real-corpus gradients |

`forward_step` is exact for the default Full flags. V7 left `tricore_fused_scan` unchanged; V9 subsequently fused Engine III into that scan, so the training forward no longer materializes `F_seq`. The O(1) claim is about persistent inference state, not training activation memory.

---

## 4. Quick start

```bash
# From the repository root
python -m pip install -e .

# Or install only the runtime dependencies
python -m pip install -r requirements.txt

# Run the full dual-track experiment (regenerates report + plot)
python run_experiment.py

# Quick smoke run
python run_experiment.py --steps 2 --code-steps 2

# V12 apex (zero-prior Track 5 + Track 7 Delta/Hebbian + 1M-token CUDA)
python run_v12_apex.py

# Quick V12 smoke (writes only the requested metrics file; skips plotting)
python run_v12_apex.py --steps5 5 --steps7 10 --seeds 42 --max-length 512 --skip-plot --metrics _smoke_v12.json

# V11 flagship (Track 5 three-seed Welch + Track 6 four-hop + 128K CUDA)
python run_v11_flagship.py

# Quick V11 smoke (writes only the requested metrics file; skips plotting)
python run_v11_flagship.py --steps5 5 --steps6 20 --eval-batches6 1 --max-length 512 --skip-plot --metrics _smoke_v11.json

# V7 streaming + Track 3 only (does not rerun V6 Track 1/2)
python run_v7_benchmark.py

# Tests
python -m pytest -q
```

`run_experiment.py` writes:

- `results/BENCHMARK_REPORT.md` — generated tables and protocol notes
- `results/tricore_vs_transformer.png` — 2x2 panel: Track 1 learning curves, Track 1 ablation bars, Track 2 PPL, Track 2 dereference accuracy

The device is selected automatically: CUDA if available, otherwise CPU. The script uses AdamW, cosine decay, gradient clipping at 1.0, and a fixed evaluation set. Default seeds are 42 for training and 2026/2027 (Track 1) / 3030 (Track 2) for evaluation batches.

```python
import torch
from tricore import (
    TriCoreBlock, TriCoreLM, BaselineTransformerLM,
    generate_associative_recall_batch, generate_contextual_code_lm_batch,
    generate_3hop_recall_batch, load_real_python_corpus_batches,
)

x, y = generate_associative_recall_batch(batch_size=4, device="cpu")
model = TriCoreLM(vocab_size=64, d_model=128, n_layers=1, d_fast=32, max_loops=2)
logits, surprise = model(x)          # [4, 60, 64], [4, 60]

code_x, code_y, deref_mask = generate_contextual_code_lm_batch(
    batch_size=4, num_vars=6, num_noise=6, device="cpu"
)
code_logits, code_surprise = model(code_x)  # [4, 82, 64], [4, 82]

# V7 native O(1) streaming decode (one token at a time)
device = next(model.parameters()).device
states = model.init_state(1, device)
for t in range(8):
    logits_t, surprise_t, states = model.forward_step(
        torch.tensor([1], device=device), states, t
    )
# logits_t: [1, 64]; state tensors stay 17,408 bytes at FP32.
```

---

## 5. Benchmark

The V6 tables below are generated by `python run_experiment.py`; the V7 sections are generated by `python run_v7_benchmark.py`; the V9 Track 4 sections are generated by `python run_v9_verification.py`; the V12 zero-prior, Delta/Hebbian, and 1M-token sections come from `python run_v12_apex.py`. The V6/V7 sections are single-seed; V9 Track 4A and the V11/V12 three-seed checks report mean ± std, and every Track 4B table is a direct CUDA measurement. The V12 top section is the current release; older sections are preserved below it.

### 5.1 V6 timing versus V4 and V5

| Measurement | Time | Note |
| :--- | :---: | :--- |
| V4 Full, Python per-token loop | 136.0 s | reference run |
| V5 Full, TorchScript fused scan | 125.3 s | reference run |
| V6 Full, vectorized multi-hop scan | **80.0 s** | timed 500 steps |
| V6 / V4 speed ratio | **1.70x** | >1 means faster; <1 means slower |
| V6 / V5 speed ratio | **1.57x** | >1 means faster; <1 means slower |
| V6 first-call warm-up | 0.4 s | excluded from timed training |

Timed training includes data generation, forward, backward, and optimizer steps. Warm-up is excluded. The ratio is a single-seed wall-clock measurement on one machine, not a throughput guarantee.

### 5.2 Track 1: 2-Hop Chained Associative Recall

500 steps, batch 64, `lr=3e-3`, seed 42; 6 chains, 4 queries, 16 training noise tokens. Fixed evaluation batches use seeds 2026/2027. The 4x extrapolation multiplies the distractor gap from 16 to 64 tokens, not the total sequence length.

| Architecture variant | Parameters | Final val loss | In-domain 2-hop recall | 4x noise-gap extrapolation | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (3-in-1)** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 80.0 s |
| TriCore No Engine I: Surprise Gate | 207,503 | 3.0005 | 5.55% | 4.14% | 62.3 s |
| TriCore No Engine II: Fast-Weight | 207,503 | 3.0008 | 5.78% | 3.98% | 36.1 s |
| TriCore No Engine III: Adaptive Loop | 207,503 | 1.8660 | 16.48% | 17.50% | 63.4 s |
| Vanilla Transformer (2-layer) | 543,744 | 2.3131 | 16.72% | 14.92% | 18.6 s |

### 5.3 Track 1 reading

1. **Full TriCore solves the task.** It reaches `100.00%` in-domain and `100.00%` at the 4x gap; validation loss is `0.0010`.
2. **Engine II is the associative memory.** Without fast weights, recall is `5.78% / 3.98%`, below the 6-way chance level of `16.67%`.
3. **Engine III is the second hop.** Without the latent re-query, recall is `16.48% / 17.50%`, essentially chance.
4. **Engine I is necessary in this V6 run.** Removing the write gate gives `5.55% / 4.14%`, a collapse to chance. This differs from the V5 record (`91.88% / 84.14%`) even though the equations are unchanged; the vectorized hop path changes the optimization trajectory, and this remains a single-seed result. Do not treat either run as a universal statement about Engine I.
5. **The Transformer is at chance.** `16.72% / 14.92%` at this budget; it is not tuned, parameter-matched, or compute-matched. This is not evidence of an attention ceiling.
6. **Vectorizing the hop loop helps materially.** V6 Full is `80.0 s` versus `136.0 s` for V4 (`1.70x`) and `125.3 s` for V5 (`1.57x`). The Transformer is still much faster (`18.6 s`) because TriCore executes a sequential state recurrence and many small matrix products.

### 5.4 Track 2: Causal Bytecode LM

450 steps, batch 64, `lr=3e-3`, seed 42; 6 variables, 4 NOP statements, sequence length 80. Training uses the joint loss `0.25 * dense_loss + deref_loss`; evaluation reports dense cross-entropy/PPL and dereference accuracy at the return-alias position `pos + 1`, where the next token is the literal and the input token is only the alias. Fixed evaluation batches use seed 3030.

| Model | Parameters | Validation loss | Validation PPL | Dereference accuracy | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (1-layer)** | 207,503 | 1.1529 | **3.17** | **100.00%** | 78.5 s |
| Vanilla Transformer (2-layer) | 543,744 | 1.2686 | 3.56 | 19.63% | 19.5 s |

The task is a synthetic imperative grammar. Each sample contains, in random statement order:

```text
[LET,   alias_i, var_i, SEMI]
[ALIAS, var_i,   lit_i, SEMI]
[NOP,   var_i,   SEMI,  SEMI]  × 4
...
[RET,   alias_i, lit_i, SEMI]
```

Variable, alias, and literal namespaces are disjoint. The `LET` statement binds `alias_i -> var_i`; the `ALIAS` statement binds `var_i -> lit_i`; the return exposes only `alias_i` and places the target `lit_i` at `pos + 1`. Predicting it therefore requires resolving `alias_i -> var_i -> lit_i` rather than copying a surface token. There are 6 equally likely literals, so chance is `16.67%`.

**Reading.** TriCore reaches `PPL 3.17` and `100.00%` dereference accuracy; the Transformer reaches `3.56` and `19.63%`, barely above the 6-way chance level of `16.67%`. The 100% result is on a closed, deterministic synthetic grammar with a fixed statement inventory and no held-out templates, so it demonstrates that the model can learn the intended two-level pointer lookup; it does **not** establish general code understanding, compositional generalization, or a Transformer-beating result on real code. The run is single-seed and the task is small.

### 5.5 Cache scaling

Per sample, FP32, `D=128`, `H=4`, `d_head=32`, Transformer `n_layers=2`.

| Sequence length | Transformer KV cache | TriCore O(1) state | Reduction |
| :---: | :---: | :---: | :---: |
| 128 | 256.00 KB | 17.00 KB | 15.1x |
| 512 | 1,024.00 KB | 17.00 KB | 60.2x |
| 2,048 | 4,096.00 KB | 17.00 KB | 240.9x |
| 8,192 | 16,384.00 KB | 17.00 KB | 963.8x |
| 16,384 | 32,768.00 KB | 17.00 KB | 1,927.5x |
| 32,768 | 65,536.00 KB | 17.00 KB | **3,855.1x** |

`C_TriCore = (D + H*d_head^2 + H*d_head) * 4 / 1024` KiB includes `h`, `F_mat`, and `prev_k` and is independent of `L`. The Transformer figure is `n_layers * 2 * L * D * 4 / 1024` KiB. This table is shape accounting; V9 Track 4B measures the actual CUDA allocation and decode latency at all six lengths (section 5.10), including the crossover at `L=8,192`.

### 5.6 V7 streaming equivalence and memory

Measured on CUDA with a one-layer Full model (`d_model=128`, `d_fast=32`, `max_loops=2`). The equivalence batch is 2 x 64 tokens; memory is measured at batch size 1.

| Check | Result |
| :--- | :---: |
| Max absolute logit difference (`forward` vs `forward_step`) | **5.960e-07** |
| Max absolute surprise difference | **1.192e-07** |
| Streaming latency, batch 1, 4,096 steps | **2.160 ms/token** |
| Persistent state per sample, FP32 | **17,408 bytes = 17.00 KiB** |

| Sequence length | Streamed state bytes | Per sample |
| :---: | :---: | :---: |
| 128 | 17,408 | 17.00 KiB |
| 512 | 17,408 | 17.00 KiB |
| 2,048 | 17,408 | 17.00 KiB |
| 4,096 | 17,408 | 17.00 KiB |

`forward_step` does not materialize `F_seq`, and the V9 training scan no longer does either. This table measures persistent decode state, not training activation memory.

### 5.7 Track 3A: 3-Hop Chained Reasoning

450 steps, batch 64, `lr=3e-3`, seed 42; 4 chains, 4 queries, 12 distractor tokens. In-domain evaluation uses a 12-token gap and the 4x test uses a 48-token gap. Chance is 25%.

| Model | Parameters | In-domain val loss | In-domain 3-hop acc | 4x gap extrapolation | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (1-layer, `max_loops=2`)** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 66.4 s |
| TriCore (`max_loops=1`, 2-hop limit) | 203,278 | 1.8395 | 25.39% | 23.91% | 62.8 s |
| Vanilla Transformer (2-layer) | 297,984 | 2.2188 | 23.83% | 23.44% | 16.7 s |

**Reading.** The second refinement is doing the third hop: removing it drops in-domain accuracy from `100.00%` to `25.39%` (chance), a `74.61` percentage-point gap. The Transformer is at chance at this budget. This is a synthetic 3-hop task, not a general reasoning benchmark.

### 5.8 Track 3B: Real Python Source Held-Out LM

300 steps, batch 32, `lr=3e-3`, seed 42. The corpus is the actual concatenation of `tricore/model.py`, `tricore/tasks.py`, and `run_experiment.py`, encoded as ASCII bytes (`vocab_size=128`). The first 85% is training data; the final 15% is a strict held-out suffix. Training windows never cross the split.

| Model | Parameters | Held-out loss | Held-out PPL | Next-byte acc | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (1-layer)** | 223,887 | **2.9240** | **18.616** | **41.47%** | 57.7 s |
| Vanilla Transformer (1-layer, param-matched) | 218,448 | 2.9667 | 19.428 | 28.15% | 1.5 s |

**Reading.** TriCore has lower held-out loss/PPL and higher next-byte accuracy at a nearly matched parameter count (`223,887` vs `218,448`; the Transformer uses `d_model=144` and a short `max_seq_len=96` position table). It is also about `38x` slower in this small training loop, so this is a quality/budget observation, not a throughput claim. The corpus is a small repository suffix, not a human-code benchmark.

---

### 5.9 Track 4A: V9 4-Chain Five-Way 3-Seed Verification

`run_v9_verification.py` reruns a 4-chain 2-Hop MQAR under seeds `42`, `100`, and `2026`: **350 steps, batch 64, AdamW, `lr=3e-3`**, weight decay `0.01`, cosine decay, gradient clipping `1.0`; `d_model=64`, `d_fast=16`, 4 heads, `max_loops=2`, 4 chains, 4 queries, 8 training noise tokens, `vocab_size=64` (sequence length 40). The 4x OOD test uses `noise_len=32` (sequence length 64). Each cell is **mean ± std over the three seeds**, measured on 5 fixed evaluation batches of 64 examples; random baseline is 25.0%.

| Architecture variant | Parameters | In-domain recall | 4x-gap recall | Mean train time |
| :--- | :---: | :---: | :---: | :---: |
| **TriCore Full (`max_loops=2`)** | 56,655 | **100.00 ± 0.00%** | **100.00 ± 0.00%** | 59.60 s |
| TriCore No Engine I: Surprise Gate | 56,655 | 5.65 ± 0.20% | 5.55 ± 0.83% | 48.52 s |
| TriCore No Engine II: Fast-Weight | 56,655 | 5.55 ± 0.34% | 5.31 ± 0.95% | 9.39 s |
| TriCore No Engine III: Adaptive Loop | 56,655 | 23.41 ± 1.08% | 23.54 ± 2.73% | 26.02 s |
| Vanilla Transformer (2-layer) | 83,456 | 25.08 ± 0.48% | 22.24 ± 3.96% | 2.01 s |

**Reading.** Only Full saturates both evaluations. Every ablation is at or below the `25.0%` four-way chance level: No-I `5.65% / 5.55%`, No-II `5.55% / 5.31%`, No-III `23.41% / 23.54%`, and the untuned Transformer `25.08% / 22.24%`. On the 4x gap, Full is `+94.45` points over No-I, `+94.69` over No-II, `+76.46` over No-III, and `+77.76` over the Transformer. Full has zero seed variance; every non-Full variant has non-zero variance. Three seeds are a variance check, not a significance test, and the compact task is not interchangeable with the V6/V7 reference runs.

### 5.10 Track 4B: Real CUDA Decode-State and KV-Cache Profiling to 32K

Measured on CUDA, batch size 1, FP32. TriCore allocates the native `init_state` tuple `(h, F_mat, prev_k)`; the baseline explicitly allocates `(K_cache, V_cache)` shaped `(n_layers, L, 1, n_heads, d_head)` and runs a real cached Transformer decode step (scaled dot-product attention over the full causal cache). Both use 8 warm-up + 32 timed steps; `CUDA Alloc Δ` is the `torch.cuda.memory_allocated()` delta after `empty_cache()`.

| Sequence length | TriCore state bytes | CUDA Alloc Δ | Transformer K+V bytes | CUDA Alloc Δ | Reduction | TriCore ms/token | Transformer ms/token |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 17,408 | 17,408 | 262,144 | 262,144 | **15.1x** | 2.448 | 1.474 |
| 512 | 17,408 | 17,408 | 1,048,576 | 1,048,576 | **60.2x** | 1.804 | 1.049 |
| 2,048 | 17,408 | 17,408 | 4,194,304 | 4,718,592 | **240.9x** | 1.756 | 1.274 |
| 8,192 | 17,408 | 17,408 | 16,777,216 | 16,777,216 | **963.8x** | 1.755 | 3.011 |
| 16,384 | 17,408 | 17,408 | 33,554,432 | 33,554,432 | **1,927.5x** | 2.047 | 4.993 |
| 32,768 | 17,408 | 17,408 | 67,108,864 | 67,108,864 | **3,855.1x** | 1.939 | 9.572 |

**Accounting and scope.** `(D + H*d_head^2 + H*d_head) * 4 = 17,408` bytes = **17.00 KiB per sample** for `D=128`, `H=4`, `d_head=32`, independent of `L`. The baseline cache is `2 * n_layers * L * D * 4`, i.e. **2.00 KiB per token**, reaching **64.00 MiB at `L=32,768`**. This is measured GPU allocation, not a formula. The `L=2,048` allocator delta is `4,718,592 B` (4.50 MiB) while the tensor bytes are `4,194,304 B` (4.00 MiB); the reduction column uses tensor bytes. Decode latency crosses over at `L=8,192` (TriCore `1.755` vs Transformer `3.011 ms/token`), and the Transformer reaches `9.572` vs `1.939 ms/token` at `L=32,768`. At `128/512/2,048` the Transformer is faster (`1.474/1.049/1.274` vs `2.448/1.804/1.756`). This is a single-machine B=1 result, not a serving-throughput benchmark.

### 5.11 V9 reproduction and demo

```bash
python run_v9_verification.py
python demo_showcase.py
```

`demo_showcase.py` streams a three-hop chain (`Key_A -> Bridge_1 -> Bridge_2 -> Target_Z`) through `TriCoreLM.init_state` / `forward_step`, prints the per-token surprise-gate trace, and shows the flat 17.00 KiB dashboard. It needs no checkpoint and falls back to CPU if CUDA is unavailable.

The measured V9 script runtime on the reference CUDA machine is **475.6 s** (Track 4A `474.2 s`, Track 4B `1.4 s`, including data generation and evaluation). The five-way, three-seed protocol trains 15 models, so it is intentionally much longer than the earlier one-seed runs; it is reported rather than hidden.

---

## 6. Repository layout

```text
tricore-arch/
├── tricore/
│   ├── __init__.py          # TriCoreBlock, TriCoreLM, BaselineTransformerLM,
│   │                        # both task generators
│   ├── model.py             # tricore_fused_scan, TriCoreBlock, TriCoreLM, baseline
│   └── tasks.py             # 2-Hop MQAR + Contextual Code LM generators
├── tests/
│   ├── test_sanity.py       # forward/backward/shape/gradient-flow
│   ├── test_v5.py           # JIT-vs-Python equivalence + code-task backward
│   └── test_v7_streaming.py # streaming equivalence, 17 KiB state, Track 3
├── results/
│   ├── BENCHMARK_REPORT.md  # V11 Track 6/5/4B + V6/V7/V9/V10 tables
│   ├── V9_METRICS.json      # historical machine-readable V9 metrics
│   ├── V10_METRICS.json     # machine-readable V10 Track 5 metrics
│   ├── V11_METRICS.json     # V11 Track 5/6/4B metrics and Welch tests
│   ├── V12_METRICS.json     # V12 zero-prior Track 5/7 + 1M-token CUDA metrics
│   ├── tricore_vs_transformer.png
│   ├── tricore_v11_masterpiece.png
│   └── tricore_v12_apex.png
├── run_experiment.py        # V6 dual-track training, ablation, cache, report, plot
├── run_v7_benchmark.py      # V7 streaming + Track 3; preserves V6 report tables
├── run_v9_verification.py   # V9 five-way 3-seed Track 4A + 32K CUDA Track 4B
├── run_v11_flagship.py      # V11 Track 5/6/4B + six-panel figure
├── run_v12_apex.py          # V12 zero-prior Track 5/7 + 1M-token CUDA profile
├── run_v8_verification.py   # compatibility alias -> run_v9_verification
├── demo_showcase.py         # one-command streaming demo (no checkpoint)
├── requirements.txt         # torch, numpy, matplotlib
├── pyproject.toml           # installable tricore-arch 1.6.0, MIT
├── LICENSE                  # MIT
├── .gitignore               # Python/cache/venv; keeps report and PNGs
├── SPEC_V2.md               # V9 architecture + contracts
└── README.md
```

### 6.1 Installable package

`pyproject.toml` declares `tricore-arch==1.6.0`, Python `>=3.10`, and the runtime dependencies `torch>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`. The package name remains `tricore-arch`; the import package is `tricore`.

```bash
python -m pip install -e .
python -c "import tricore; print(tricore.__file__)"
```

### 6.2 Generated artifacts

`results/BENCHMARK_REPORT.md` is the source of truth for the latest numbers; this README and `SPEC_V2.md` summarize it. `run_v12_apex.py` writes `results/V12_METRICS.json`, `results/tricore_v12_apex.png`, and the master `results/tricore_vs_transformer.png`. `run_v11_flagship.py` remains the historical V11 runner and writes `results/V11_METRICS.json` plus the V11 six-panel figure. Do not hand-edit a generated results file and then claim it came from a run; rerun the scripts.

---

## 7. Experimental protocol

### 7.1 Track 1 — 2-Hop Chained Associative Recall

- 6 chains, 4 queries, 16 training noise tokens; sequence length `6*3 + 6*3 + 16 + 4*2 = 60`.
- 500 optimizer steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`.
- Training seed `42`; fixed evaluation batches are generated with seeds `2026` (in-domain) and `2027` (4x gap), 10 batches of 32 per evaluation.
- The 4x evaluation multiplies the distractor gap from `16` to `64` tokens; the total sequence length becomes `108`, not `240`.
- Recall is computed only at query positions whose target is not `-100`.
- Full, No Engine I, No Engine II, No Engine III, and a 2-layer Transformer are trained from scratch with the same protocol.

### 7.2 Track 2 — Causal Bytecode LM

- 6 variables, 4 NOP statements; sequence length `(6*2 + 4 + 4)*4 = 80`.
- 450 optimizer steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`.
- Training seed `42`; fixed evaluation batches use seed `3030`, 8 batches of 64.
- Joint objective `0.25 * dense_loss + deref_loss`; dense cross-entropy is computed over all non-`-100` positions and dereference loss only on the return-alias mask.
- Dereference accuracy is measured at the single return-alias position, where the target is one of 6 literals (`16.67%` chance).
- TriCore uses one layer (`d_model=128`, `d_fast=32`, `max_loops=2`); the baseline uses two Transformer layers.

### 7.3 Track 3A — 3-Hop Chained Recall

- 4 chains, 4 queries, 12 training distractor tokens; sequence length `4*3*3 + 12 + 4*2 = 56`.
- 450 steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`, seed 42.
- Three variants: Full one-layer `max_loops=2`, one-layer `max_loops=1` (only two hops), and a 2-layer Transformer.
- Fixed in-domain evaluation uses `noise_len=12` and seed 5101; the 4x gap uses `noise_len=48` and seed 5102. Accuracy is measured only at the query key positions.

### 7.4 Track 3B — Real Python Source Held-Out LM

- 300 steps, batch 32, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`, seed 42.
- Corpus: concatenated `tricore/model.py`, `tricore/tasks.py`, and `run_experiment.py`, encoded as ASCII bytes; `vocab_size=128`.
- First 85% is training data, final 15% is a strict held-out suffix; training windows cannot cross the split.
- Models: one-layer TriCore (`d_model=128`, `d_fast=32`, `max_loops=2`) and a one-layer Transformer with `d_model=144` and a short `max_seq_len=96` position table to match the parameter budget.

### 7.5 Timing

- V6 timing includes data generation, forward, backward, and optimizer steps for 500 Full steps.
- The first scripted call is a warm-up and is excluded from the timed loop.
- V4's 136.0 s and V5's 125.3 s are previously recorded reference runs, not simultaneous measurements.
- Timings are single-machine wall-clock values and vary with GPU, driver, PyTorch version, and thermal state.

### 7.6 What is not controlled

V9 Track 4A is a three-seed variance check. V11/V12 Track 5 and V12 Track 7 add exact two-tailed Welch tests but still have only three seeds per arm, so they are not population-level inference. V11 Track 6 and all Track 4B profiles are single-seed/single-machine controlled comparisons. V12 autonomous discovery assumes short, regular statements with a repeated terminal delimiter and low-frequency entity tokens; it is not a general parser and is disabled in the 1M-token profile. There is no hyperparameter search for the Transformer and no FLOP matching. Track 1/2 have no real held-out corpus. Track 3B has a real-source held-out suffix, but the corpus is only this small repository and is not a human-code benchmark. The V7 and V11 latency numbers are one machine's per-token loop, not a production serving benchmark. Treat the numbers as existence demonstrations and pipeline checks.

### 7.7 V11 Track 6 / Track 5 / Track 4B

- **Track 6:** 4 chains, 4 queries, 4 fact tiers, 12 distractors; sequence length 68; 4x OOD gap 48 distractors; 500 steps, batch 64, seed 42; `d_model=128`, `d_fast=32`; depth budgets `max_loops=3/2/1`; fixed eval seeds 6101/6102, 8 batches of 64.
- **Track 5:** 350 steps, batch 64, seeds 42/100/2026; 4 variables, 4 queries, 4 NOPs; identical architecture variants and evaluation batches as V10; exact two-tailed Welch t-tests on accuracy and validation loss.
- **Track 4B:** real CUDA allocation and cached decode at `L=128, 512, 2,048, 8,192, 16,384, 32,768, 65,536, 131,072`; B=1, FP32; 8 + 32 timed steps below 65K, 4 + 16 above; RTX 3060.

### 7.8 V12 Track 5 / Track 7 / Track 4B

- **Track 5:** 350 steps, batch 64, seeds 42/100/2026; `d_model=64`, `d_fast=16`, four variables, four queries, four NOPs; `autonomous_syntax=True`, all token-role arguments `None`; 25.0% chance.
- **Track 7:** 400 steps, batch 64, seeds 42/100/2026; four variables, four queries, four NOPs; shuffled LET, initial SET, live SET overwrite, NOP, and RET statements; target is the new literal at the query EQ; 25.0% chance. Delta Full, Hebbian Full, fair No-Engine-I Delta, and a 2-layer Transformer share batch/evaluation data.
- **Track 4B:** actual CUDA allocation and cached decode at `L=128..1,048,576`; B=1, FP32; RTX 3060; 1M uses 2 warm-ups + 8 timed steps. TriCore uses the same 17,408-byte state at every length; autonomous discovery is disabled because it is a short-sequence role prior.

---

## 8. Verification

### 8.1 JIT equivalence

`tests/test_v5.py` builds random tensors for `h`, `F`, `prev_k`, and projections, then runs `tricore_fused_scan` and an explicit Python reference over the four combinations of its two scan flags:

- `enable_surprise_gate=True, enable_fast_weight=True`
- `enable_surprise_gate=False, enable_fast_weight=True` (No Engine I)
- `enable_surprise_gate=True, enable_fast_weight=False` (No Engine II)
- both flags `False`

Engine III is fused inside the scripted scan and is covered by the `TriCoreLM` forward/backward test plus the JIT/reference test; the causal bytecode task has its own shape, mask, and backward test.

All three outputs (`h_seq`, `mem_seq`, `surprise_trace`) must satisfy `torch.allclose(..., atol=1e-6, rtol=1e-5)`. The test also checks that full `forward` matches token-by-token `forward_step` with `< 1e-6` maximum absolute error, plus the causal bytecode shapes, the dereference mask at `pos + 1`, and a backward pass with finite gradients.

Run:

```bash
python -m pytest -q
python tests/test_v5.py
python tests/test_v7_streaming.py
```

### 8.2 V7 streaming and Track 3 tests

`tests/test_v7_streaming.py` checks six properties:

1. For a one-layer Full model in `eval()` mode, stacking `forward_step` logits over `t=0..L-1` matches `forward(idx)` under `torch.allclose(atol=1e-5)`; surprise traces are checked as well.
2. The total bytes in `states` are exactly `batch_size * 17,408` at both `t=16` and `t=512`, with no growth from `F_seq`.
3. The 3-hop generator, real Python corpus loader, dense backward pass, and gradient finiteness are exercised.
4. The V11 4-hop generator is checked for shape `(4, 68)`, 16 valid targets, finite gradients, and full-vs-`forward_step` parity with the `reset_trace_tokens=(0,)` path (`atol=1e-4`).
5. The V12 parity test exercises `enable_delta_rule` in both states and confirms that the Delta erase term changes the output.
6. The V12 zero-prior discovery test verifies the exact Track 5 and Track 7 EQ/SEMI masks, then repeats the discovery after a random bijection of all 64 vocabulary IDs and requires bit-identical masks.

The suite currently reports **11 passing tests**: the sanity test, four V5 JIT/bytecode/eligibility tests, four V7/V11 streaming and recall tests, the V12 Delta-rule parity test, and the V12 zero-prior mask-discovery test (including an arbitrary vocabulary permutation).

### 8.3 Full run

The V12 reference run completed on CUDA and regenerated:

- `results/BENCHMARK_REPORT.md`
- `results/tricore_vs_transformer.png` (master path)
- `results/tricore_v12_apex.png`
- `results/V12_METRICS.json`

The report contains the V12 zero-prior Track 5, Track 7 Delta/Hebbian, and 1M-token Track 4B sections, with the complete V11/V10/V9/V7/V6 record preserved below. If the code and this README disagree, `results/BENCHMARK_REPORT.md`, `results/V12_METRICS.json`, and the code win.

### 8.4 Smoke run

```bash
python run_experiment.py --steps 2 --code-steps 2
python run_v12_apex.py --steps5 5 --steps7 10 --seeds 42 --max-length 512 --skip-plot --metrics _smoke_v12.json
python run_v11_flagship.py --steps5 5 --steps6 20 --eval-batches6 1 --max-length 512 --skip-plot --metrics _smoke_v11.json
python run_v7_benchmark.py  # streaming + Track 3; preserves V6 tables
```

This is a pipeline check, not a result. The V6 smoke compiles the scripted scan, runs both task generators, exercises all five Track 1 variants and both Track 2 models, writes the report, and saves the plot. The V7 script adds the streaming equivalence/memory check and the two Track 3 runs; it is a full benchmark, not a smoke run.

---

## 9. Limitations

- **Mixed seed coverage.** The V6/V7 Track 1/2/3 tables are still single-seed. V9 Track 4A is a 3-seed variance check, V11/V12 Track 5 and V12 Track 7 are 3-seed Welch checks, and V11 Track 6 / every Track 4B profile are single-seed/single-machine controlled comparisons. The V11/V12 tests have only three observations per arm, so the p-values are exact arithmetic on a tiny sample, not population inference. The V6 No-Engine-I result (`5.55% / 4.14%`) differs substantially from the V5 record (`91.88% / 84.14%`); the equations are the same, but the vectorized hop path changes the optimization trajectory and fixed evaluation batches are used.
- **V12 Track 7 is not a clean Delta win.** Delta is stable at 100.00% on all three seeds, but Hebbian collapses on seed 2026 (24.96%), so the Delta-vs-Hebbian Welch test is **not significant** (`p=0.394`). The supported V12 claim is that a dynamic-write engine is necessary and that Delta remained stable; it is not a population-level demonstration that Delta beats Hebbian.
- **Streaming scope.** `forward_step` is an exact one-token decoding path, and the V9 training forward no longer materializes `F_seq`; it returns the fused `mem_seq` read. The 17 KiB O(1) claim applies to persistent inference state, not training activation memory or throughput.
- **Track 3 and Track 6 scope.** Track 3A is a synthetic 3-hop chain task with only four chains and four queries. V11 Track 6 is a synthetic 4-hop pointer chain with four chains and four queries. Track 3B is a small held-out suffix from this repository, not a human-code benchmark; its next-byte accuracy and PPL do not establish general code understanding.
- **Synthetic tracks.** Track 1 is a controlled associative-recall construction. Track 2 is a tiny imperative grammar with disjoint namespaces. Track 3A and Track 6 are four-chain synthetic 3-hop/4-hop pointer tasks. None is natural language or a general reasoning benchmark.
- **Track 2 is a closed synthetic grammar.** V6 reaches `100.00%` dereference on a deterministic bytecode task with fixed namespaces and no held-out templates. That shows the causal two-level lookup is learnable, not that the model understands real code. The PPL gap over the Transformer (`3.17` vs `3.56`) is also single-seed.
- **The multi-hop pass is not a parallel scan.** The state recurrence is still serial. V6 vectorizes only the fixed-hop refinement over `L`; it still launches many small CUDA kernels. The measured speedup is a runtime improvement on one machine, not a hardware-optimal kernel.
- **Transformer baselines are not fully matched.** The Track 1 baseline is untuned and much larger (`543,744` vs `207,503`); Track 3B is parameter-matched (`218,448` vs `223,887`) but not compute-matched, and TriCore trains about 38x slower there.
- **Training memory is no longer dominated by `F_seq`.** V9 fuses Engine III into the scripted causal scan and never allocates or stacks `F_seq: [B, L, H, d_head, d_head]`. Training still stores sequence activations (`h_seq`, `mem_seq`, and autograd buffers), so it is not activation-O(1); the removed tensor was the large `L * H * d_head^2` snapshot.
- **Long-context decode is measured to 1,048,576 tokens.** V12 Track 4B measures real CUDA allocation and latency through `L=1,048,576`; at 1M the measured KV cache is **2,147,483,648 B (2.00 GiB)** versus **17,408 B (17.00 KiB)**, a **123,361.9x** reduction. Latency first inverts at `L=8,192`; at 1M it is 274.275 vs 3.794 ms/token. This is a single-machine B=1 result, not a serving benchmark.
- **No natural-language evidence.** Track 3A tests exactly three hops, Track 6 tests exactly four hops, and Track 3B tests next-byte prediction on a small repository suffix. None establishes general reasoning or natural-language ability.
- **Task-specific ablations.** Engine III was redundant on the V3 single-hop task and is necessary here; Engine I's necessity depends on the noise pattern and seed. Conclusions are conditional.
- **No production hardening.** No checkpoint format, distributed training, mixed precision, quantization, serving API, or security review.

---

## 10. Roadmap

1. **Multi-seed evidence.** V9 Track 4A reports mean ± std for every variant and V11/V12 Track 5 plus V12 Track 7 add exact three-seed Welch tests. The next step is more seeds and repetition at the full V6/V7 sizes, not just a small-sample check.
2. **More hops.** V11 reaches four hops in one forward pass; test 5-8 hop chains, more bridge tokens, and variable `max_loops` to see whether the router uses or downweights extra re-queries.
3. **Stronger adversarial writes.** Deletion, overwrite, repeated distractors, and higher noise density.
4. **Long-context streaming.** V11 measures real allocation and latency to 131,072 tokens; extend the same measured protocol to 1M and compare against a production KV-cache serving stack.
5. **Kernel-level efficiency.** Replace the scripted state loop with a fused/chunked scan or a compiled CUDA/Triton kernel; compare against the V6 JIT plus vectorized-hop baseline.
6. **Matched baselines.** Tune the Transformer and match non-position parameters, tokens, and compute.
7. **Harder code task.** More variables, longer schedules, held-out program templates, and a real code corpus.
8. **Natural-language pretraining.** Test whether surprise-gated writes and latent re-query transfer beyond synthetic tasks.

---

## 11. Citation

```bibtex
@misc{tricore2026v11,
  title        = {TriCore V11: Four-Hop Latent Chaining, Three-Seed Welch Tests,
                  and 128K Real CUDA Memory/Latency Profiling},
  author       = {The TriCore Authors},
  year         = {2026},
  howpublished = {Open-source reference implementation},
  note         = {Version 11; MIT license; V6/V7/V9/V10 records preserved}
}
```

---

## 12. Contributing

Issues and pull requests are welcome. Preserve these invariants:

1. **No attention over tokens.** No softmax may mix sequence positions. The only softmax is the hop router over `max_loops + 1` slots.
2. **O(1) state.** `h`, `F_mat`, and `prev_k` shapes must be independent of `L`.
3. **Causal shifted write.** Engine II binds `K_{t-1}` to `V_t`; the read uses `Q_t` after the write.
4. **Delta rule.** The write is `(V_t - F_{t-1}K_{t-1})K_{t-1}^T`, not a plain Hebbian outer product.
5. **Per-head matrices.** `F_mat` is `[B, H, d_head, d_head]`.
6. **Canonical address.** Q/K/V retain the shared normalized address plus residual projections.
7. **Write protection.** Engine I retains a learned write gate; `enable_surprise_gate=False` is the unconditional-delta ablation.
8. **Latent re-query.** Engine III maps the previous read back into query space and re-queries the same `F_mat`; disabling it removes the loop and the refinement MLP.
9. **Independent ablations.** Flags gate compute without changing parameter registration.
10. **JIT equivalence.** Any change to `tricore_fused_scan` must pass `tests/test_v5.py` against the Python reference on all four returned tensors.
11. **Fixed loop count.** Do not describe Engine III as adaptive computation time; `max_loops` is a constant.
12. **Vectorized refinement.** Engine III must re-query `F_seq` with one batched 5-D `matmul` per fixed hop; the Python loop is over `max_loops`, not `L`. The state recurrence remains serial.
13. **Causal dereference.** Track 2 must mark the return alias position `pos + 1` in `deref_mask`, never the literal position, and `LET`/`ALIAS` must not place the answer beside the query alias.
14. **Reproducible artifacts.** `python run_experiment.py` must regenerate the V6 report/plot, and `python run_v7_benchmark.py` must append the V7 streaming/Track 3 sections without deleting the V6 tables. State accounting must include `prev_k`.

---

<p align="center">
  <i>TriCore V7 — keep a constant state, stream one token at a time, vectorize the latent loop, and force the pointer to be dereferenced.</i>
</p>
