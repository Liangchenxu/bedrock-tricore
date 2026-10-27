# TriCore V13 — Architecture Specification and Benchmark Record

> **Current package release:** `bedrock-tricore` v1.6.3. This release adds native O(1) autoregressive `generate()`, `inspect_telemetry()`, and a 220-DPI architecture blueprint. The V13 architecture and benchmark record below is unchanged and preserved.
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


<!-- V12_SPEC_SECTION -->
## V12 (v1.5.0) — Zero-Prior Autonomous Syntax, Delta-Rule Mutation, and 1,048,576-Token CUDA Profiling

**Release focus.** V12 removes the last token-ID prior from Track 5 and adds a controlled state-mutation test:

1. **Autonomous syntax discovery.** `TriCoreLM(..., autonomous_syntax=True)` is constructed with `separator_token=None`, `syntax_tokens=None`, and `reset_trace_tokens=None`. The model discovers a structural set, the terminal delimiter, and infix connectors from the current sequence's token frequencies and causal predecessor frequencies. No vocabulary ID, learned role table, or recurrent state is used for discovery.
2. **Structural eligibility direction.** A closed eligibility gate at an infix connector reads the frozen preceding key instead of the connector address, and prevents `prev_k` from advancing. The write gate remains closed at connectors and delimiters. In the two-hop layout this yields the shifted binding `F(alias) = var`, which is exactly the address later queried by `[RET, alias, EQ, ...]`.
3. **Selectable Delta rule.** `enable_delta_rule=True` uses `F_t = decay*F_{t-1} + lr*(V_t - F_{t-1}K_{t-1})K_{t-1}^T`; `False` uses the same `decay` and `lr` with the plain Hebbian outer product `F_t = decay*F_{t-1} + lr*V_t K_{t-1}^T`. The switch changes no parameter shape and no state size.
4. **Track 7 live overwrite.** A variable is assigned an old literal, overwritten with a new literal, separated by a NOP, and queried through an alias. The target is the new literal. This isolates old-value erasure rather than static binding.
5. **Track 4B to one million tokens.** Real allocator deltas and real cached-decoder latency are measured at `L=128..1,048,576` for a constant 17,408-byte TriCore state and a growing two-layer K/V cache.

**Discovery rule (no token IDs, no parameters, no recurrence).** For a batch `x` of shape `[B, L]`, compute token frequencies and causal predecessor frequencies inside each sequence. A token is **structural** when `freq >= max(6, L // 10)`; the **terminal delimiter** is the structural token at the last position; an **infix connector** is structural, is not the delimiter, and has a predecessor frequency `<= 4`. These are pure equality and frequency tests, so any permutation of the 64 vocabulary IDs produces bit-identical connector/delimiter masks. The rule is `O(B*L^2)` and is disabled for the 1M profile.

**Track 5 result (seeds 42/100/2026; 350 steps; batch 64; CUDA).**

| Variant | Parameters | Val loss (mean ± std) | EQ two-hop accuracy (mean ± std) | Eligibility gate `EQ` / entity | Mean train time |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V12 Zero-Prior Full** | 61,851 | **0.00237 ± 0.00021** | **100.00 ± 0.00%** | 0.009 / 0.909 | 171.7 s |
| TriCore V12 Fair No-Engine-I | 61,851 | 1.4731 ± 0.0185 | 25.13 ± 0.67% | 1.0000 / 1.0000 | 141.9 s |
| Vanilla Transformer (2-layer) | 83,456 | 1.4397 ± 0.0075 | 25.01 ± 0.63% | n/a / n/a | 13.9 s |

Exact two-tailed Welch results: Full vs Fair accuracy `t=192.524`, `df=2.000`, **`p=2.70e-05`**; Full vs Transformer accuracy `t=206.869`, `df=2.000`, **`p=2.34e-05`**; Full vs Fair loss `t=-137.883`, `df=2.001`, **`p=5.25e-05`**; Full vs Transformer loss `t=-333.867`, `df=2.003`, **`p=8.81e-06`**. Three seeds per arm: this is exact arithmetic on three observations, not a population claim.

**Track 7 protocol (seeds 42/100/2026; 400 steps; batch 64; CUDA).**

Each sequence declares a variable, assigns an old literal, overwrites it with a new literal, inserts a NOP, and queries through an alias. Supervision at the query `EQ` is the **new** literal; four-way chance is 25.0%. Both TriCore arms use the same surprise-gated write path; the only change is `enable_delta_rule=True` (erase term) versus `False` (plain Hebbian outer product). Fair No-Engine-I keeps the Delta rule but sets `enable_surprise_gate=False`. The Transformer receives no token-role prior. Seeds `42/100/2026`, 400 steps, batch 64, CUDA.

| Variant | Parameters | New-value accuracy (mean ± std) | Stale-value confusion (mean ± std) | Val loss (mean ± std) | Mean train time |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V12 Delta Full** | 61,851 | **100.00 ± 0.00%** | **0.00 ± 0.00%** | **0.00227 ± 0.00012** | 175.0 s |
| TriCore V12 Hebbian Full | 61,851 | 73.71 ± 42.22% | 1.06 ± 0.95% | 0.6164 ± 0.9452 | 167.2 s |
| TriCore V12 Fair No-Engine-I Delta | 61,851 | 25.12 ± 0.41% | 0.00 ± 0.00% | 1.7284 ± 0.0234 | 141.5 s |
| Vanilla Transformer (2-layer) | 83,456 | 24.28 ± 0.41% | 0.00 ± 0.00% | 1.4840 ± 0.0674 | 3.3 s |

Per-seed new-value accuracy is Delta `100.00/100.00/100.00%`, Hebbian `98.67/97.50/24.96%`, Fair `24.65/25.39/25.31%`, and Transformer `24.22/24.73/23.91%`. Delta is stable on all three seeds and never returns the stale literal. Hebbian succeeds on seeds 42 and 100 but collapses to chance on seed 2026 (24.96%, validation loss 1.7077), so its mean is not a robust effect. Exact two-tailed Welch results: Delta vs Fair new value `t=318.03`, `df=2.00`, **`p=9.89e-06`**; Delta vs Transformer `t=316.76`, `df=2.00`, **`p=9.97e-06`**; Delta vs Hebbian `t=1.078`, `df=2.00`, **`p=0.394`** (not significant), and Delta vs Hebbian stale value `p=0.194`. The supported conclusion is that a dynamic-write engine is required and that Delta remained stable on every seed; this three-observation experiment does not establish a population-level Delta > Hebbian advantage.

**Track 4B protocol (batch 1, FP32, RTX 3060).** The baseline is a real two-layer cached decoder: K/V tensors are allocated, written, and attended over the full causal cache. TriCore runs the native `forward_step` state path. Below 65K the measurement uses 8 warm-up + 32 timed steps; 65K–131K uses 4 + 16; 262K–1M uses 2 + 8. `Alloc` is the measured `torch.cuda.memory_allocated()` delta after `empty_cache()`.

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

TriCore state remains exactly **17,408 B (17.00 KiB) per sample** at every length. At `L=1,048,576` the measured cached Transformer K/V allocation is **2,147,483,648 B (2.00 GiB)**, a **123,361.9×** tensor-byte reduction, and decode latency is **274.275 ms/token** versus **3.794 ms/token** for TriCore (**72.30×**). The measured crossover is at `L=8,192`; below it the small cached Transformer is faster. The 1M row uses 2 warm-up + 8 timed steps because of cache size. This is a single RTX 3060, batch 1, FP32 measurement, not a serving-throughput claim.

**Scope.** Track 5 and Track 7 are three-seed checks with three observations per arm, not population inference. Track 4B is a single-machine B=1 measurement, not serving throughput. Autonomous discovery is a short-sequence role prior for regular closed syntax, not a general parser. The 100% scores are on closed synthetic grammars and do not establish general reasoning.

**V12 runtime:** **493.8 s** for Track 5 (three seeds), Track 7 (four arms × three seeds), and Track 4B (eleven lengths) on the RTX 3060 reference machine.



## V11 (v1.4.0) — Four-Hop Chained Recall, Three-Seed Welch Tests, and 128K CUDA Profiling

**Release focus.** V11 does not change the V10 equations or parameter names. It adds:

1. **A delimiter reset prior for deep chaining.** `TriCoreLM(..., reset_trace_tokens=(0,))` builds a boolean `reset_mask` and zeros the per-head key trace after the scan update at those tokens: `prev_k <- prev_k * (1 - reset_t)`. The mask is threaded through `tricore_fused_scan`, `TriCoreBlock.forward`, `TriCoreLM.forward`, `TriCoreBlock.forward_step`, and `TriCoreLM.forward_step`. The full-sequence and streaming paths therefore apply the reset at the same position. The write/read equations and Q/K/V projections are unchanged; the reset only clears `prev_k` after the delimiter update, and `fast_weight=False` bypasses it because no `prev_k` exists.
2. **Track 6**, a four-hop single-forward pointer-chain task with four shuffled fact tiers, a 12-token distractor gap, and four query keys. The 4x OOD variant uses 48 distractors.
3. **Track 5 over three seeds** with exact two-tailed Welch tests implemented from the regularized incomplete Beta function.
4. **Track 4B Extended** through `L=131,072` with measured CUDA allocator deltas and real cached-decoder latency.
5. **A 2x3 master figure** at both `results/tricore_vs_transformer.png` and `results/tricore_v11_masterpiece.png`.

**Track 6 result (single seed 42; 500 steps; batch 64).** The reset prior is essential for this layout: without it, `prev_k` leaks the last distractor bridge into the query read. With `reset_trace_tokens=(0,)`, Full reaches **100.00% in-domain / 100.00% 4x-OOD** with 236,436 parameters and routes **0.988** of query mass to hop 4. The 3-hop and 2-hop budgets (232,211 and 227,986 parameters) route **0.997** and **0.981** to their last slot but remain at chance (24.95/24.76% and 25.73/26.27%). A 2-layer Transformer (306,176 parameters) is also at chance (24.41/25.78%). The TriCore budgets receive the same explicit `separator_token=0`, `syntax_tokens=(0,)`, and `reset_trace_tokens=(0,)`; the Transformer does not receive that prior. This is a single-seed controlled comparison.

**Track 5 result (seeds 42/100/2026; 350 steps; batch 64).**

| Variant | Parameters | Val loss | Accuracy | Gate `EQ` / entity |
| :--- | ---: | ---: | ---: | ---: |
| **TriCore V11 Full** | 60,819 | **0.00218 ± 0.00022** | **100.00 ± 0.00%** | 0.0032 / 0.9054 |
| TriCore V11 Fair No-Engine-I | 60,819 | 1.4682 ± 0.0186 | 24.80 ± 1.12% | 1.0000 / 1.0000 |
| Vanilla Transformer (2-layer) | 83,456 | 1.4397 ± 0.0075 | 25.01 ± 0.63% | n/a / n/a |

Two-tailed Welch: Full vs Fair accuracy `t=116.649`, `df=2.000`, **`p=7.35e-05`**; Full vs Transformer accuracy `t=206.869`, `p=2.34e-05`; Full vs Fair loss `p=5.34e-05`; Full vs Transformer loss `p=8.81e-06`. The test is exact arithmetic on three observations per arm, not a population claim.

**Track 4B result.** The persistent TriCore state is `(D + H*d_head^2 + H*d_head) * 4 = 17,408` bytes at every length. The real 2-layer K+V cache grows to **268,435,456 B = 256.00 MiB** at `L=131,072`, a **15,420.2x** tensor-byte reduction. TriCore stays between **2.57 and 3.41 ms/token**; the Transformer grows from **1.39 to 35.55 ms/token**, crossing at `L=8,192` and reaching **12.99x** slower at 128K. These are B=1 measurements on one RTX 3060.

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

**Project:** `bedrock-tricore` 1.6.0
**Operator:** attention-free, constant-size recurrent state, causal shifted fast-weight binding, fused latent multi-hop re-query, zero-prior autonomous syntax discovery  
**Runtime:** `@torch.jit.script tricore_fused_scan` for the O(1) state and zero-`F_seq` multi-hop training, plus native `init_state` / `forward_step` per-token decoding  
**Status:** research reference; V12 zero-prior Track 5, Delta-vs-Hebbian Track 7, and 1,048,576-token CUDA profiling; V11 Track 6 four-hop chaining + three-seed Welch tests + 128K CUDA profiling, V10 Track 5, V9 zero-`F_seq`/4-chain verification, and V6/V7 records preserved  
**Executable source of truth:** `tricore/model.py`  
**Benchmark source of truth:** `results/BENCHMARK_REPORT.md`

> **Document provenance.** This specification describes V12; V11/V10/V9/V7/V6 records are retained. The equations, shapes, flags, state accounting, autonomous syntax masks, reset mask, and streaming API mirror `tricore/model.py`, `tricore/tasks.py`, `run_experiment.py`, `run_v9_verification.py`, `run_v11_flagship.py`, and `run_v12_apex.py`. V12 adds no trainable parameters: `autonomous_syntax` derives token roles from within-sequence frequency, and `enable_delta_rule` selects the erase term without changing shapes or state. V11 adds no trainable parameters: `reset_trace_tokens` is a constant token-role mask. V9 remains additive to V6/V7 at the API level and removes the training-only `F_seq` snapshot tensor by fusing Engine III into the JIT scan. The benchmark numbers are artifacts generated on the reference machine. If this document and the code disagree, the code wins and this document must be corrected.

---

## 1. Scope

TriCore V12 is a sequence operator with three independently ablatable mechanisms plus a zero-prior syntax-role discovery path:

1. **Engine I — Surprise-Gated State Collapse and Write Protection:** a gated recurrent vector `h_t`, plus a learned per-head write gate on the fast-weight learning rate.
2. **Engine II — Subspace-Aligned Causal Shifted Fast Weights:** a per-head delta-rule matrix `F_t` that binds `K_{t-1}` to `V_t`, with a shared canonical address for Q/K/V.
3. **Engine III — Vectorized Latent Multi-Hop Synaptic Equilibrium Loop:** fixed-step re-queries of per-position `F_t` snapshots, with a learned per-token hop router and a surprise-scaled refinement MLP.

V6 added two engineering/evaluation contributions without changing the equations:

- A TorchScript-JIT recurrent state scan that returns `h`, the routed multi-hop memory read, and the surprise trace. V9 fuses Engine III into this same causal loop, so the training path does not allocate or stack the old `F_seq` snapshot tensor.
- A corrected Track 2, **Causal Bytecode LM**, whose `[LET, alias, var, SEMI]`, `[ALIAS, var, lit, SEMI]`, `[NOP, var, SEMI, SEMI]`, and `[RET, alias, lit, SEMI]` format forces a two-level pointer dereference at the return alias position `pos + 1`.

V7 adds three contributions while keeping all V6 equations and parameter names:

- A native streaming API: `TriCoreBlock.init_state` / `TriCoreBlock.forward_step` and the matching `TriCoreLM` methods.  The persistent tuple is exactly `(h, F_mat, prev_k)`, is independent of `L`, and does not materialize `F_seq`.
- **Track 3A**, a 3-hop chained-recall task with shuffled one-, two-, and three-hop facts plus an adversarial distractor gap; `max_loops=1` is limited to two hops, while `max_loops=2` performs the third re-query.
- **Track 3B**, a byte-level held-out LM over the actual repository Python sources (`tricore/model.py`, `tricore/tasks.py`, `run_experiment.py`), with the final 15% of the stream reserved as an unseen validation suffix.

V9 adds one runtime change and two evaluations:

- **Zero-`F_seq` training:** Engine III's fixed re-query is inside `tricore_fused_scan`; it uses the current `F_mat` at each timestep and returns `(h_seq, mem_seq, surprise_trace)`. The only sequence-wide memory read is `mem_seq: [B, L, H*d_head]`; `F_seq: [B, L, H, d_head, d_head]` is never allocated.
- **Track 4A:** a 4-chain 2-Hop MQAR with four queries, training `noise_len=8`, 4x OOD `noise_len=32`, `d_model=64`, `d_fast=16`, 350 steps, and three seeds; all five models are reported as mean ± std.
- **Track 4B:** real CUDA allocation and cached-decoding latency at `L=128..32,768`. V11 extends the same measured protocol to `L=131,072`.

V12 adds zero-prior role discovery and a selectable Delta rule:

- `TriCoreLM(..., autonomous_syntax=True)` with `separator_token=None`, `syntax_tokens=None`, and `reset_trace_tokens=None` discovers infix connectors, write-protected structural tokens, and the terminal delimiter from within-sequence frequency and causal adjacency. It uses no vocabulary IDs and adds no recurrent state.
- `enable_delta_rule=True` selects `F_t = decay*F_{t-1} + lr*(V_t - F_{t-1}K_{t-1})K_{t-1}^T`; `False` selects the Hebbian `F_t = decay*F_{t-1} + lr*V_t K_{t-1}^T` with identical decay/learning-rate terms.
- **Track 7** measures stateful live variable overwrite, where a Delta erase is compared with Hebbian accumulation and a fair No-Engine-I ablation.
- **Track 4B Extended** profiles real CUDA state/cache allocation and cached decode through `L=1,048,576`.

V11 adds one operator prior and three evaluations:

- **Delimiter reset:** `reset_trace_tokens=(0,)` zeroes `prev_k` after the update at every delimiter, in both the fused scan and `forward_step`. No parameter shapes change.
- **Track 6:** a four-hop single-forward pointer chain with four shuffled fact tiers, four queries, and a 4x distractor gap; depth budgets `max_loops=3/2/1` plus a 2-layer Transformer.
- **Track 5 Welch check:** the V10 infix-`EQ` task repeated over seeds 42/100/2026, with exact two-tailed Welch tests on accuracy and loss.
- **Track 4B Extended:** real CUDA state/KV allocation and cached latency through 128K.

The specification has six non-goals:

- It does **not** claim natural-language or reasoning capability.
- It does **not** claim superiority over a tuned Transformer.
- It does **not** claim that every component is necessary for every task or seed.
- It does **not** claim adaptive computation time; `max_loops` is a fixed hyperparameter.
- It does **not** claim that the JIT scan is a custom CUDA kernel or a production-throughput solution.
- It does **not** claim that the recurrent state update is parallel; only the multi-hop refinement is vectorized across `L`.

The evidence is three single-seed V6/V7 evaluation tracks — a 500-step five-variant 2-hop associative recall study, a 450-step two-model causal bytecode LM study, and a V7 study with streaming equivalence/memory checks, a 450-step 3-hop comparison, and a 300-step real-source held-out LM comparison — plus V9's 4-chain, five-way, 3-seed 2-Hop MQAR variance check, V11's three-seed Track 5 Welch check, V11's single-seed Track 6 four-hop controlled comparison, and measured CUDA decode-state/KV-cache profiling through 128K.

---

## 2. Repository contract

```text
bedrock-tricore/
├── tricore/
│   ├── __init__.py          # block/LM/baseline exports and all task loaders
│   ├── model.py             # tricore_fused_scan, TriCoreBlock, TriCoreLM,
│   │                        # BaselineTransformerLM, init_state/forward_step
│   └── tasks.py             # 2-hop MQAR, Causal Bytecode LM, 3-hop/4-hop recall,
│                            # stateful mutation, real Python source byte stream
├── tests/
│   ├── test_sanity.py       # forward/backward/shape/gradient-flow
│   ├── test_v5.py           # four-output JIT equivalence + causal bytecode backward
│   └── test_v7_streaming.py # streaming, 17 KiB state, 3-hop/4-hop, V12 zero-prior masks
├── results/
│   ├── BENCHMARK_REPORT.md  # V12 top sections + preserved V6/V7/V9/V10/V11 records
│   ├── tricore_vs_transformer.png
│   ├── tricore_v11_masterpiece.png
│   ├── tricore_v12_apex.png
│   └── V12_METRICS.json
├── run_experiment.py        # V6 dual-track training, ablation, cache, report, plot
├── run_v7_benchmark.py      # V7 streaming/Track 3 benchmark; does not rerun V6
├── run_v9_verification.py   # V9 five-way 3-seed Track 4A + 32K Track 4B
├── run_v11_flagship.py      # V11 Track 5/6/4B + six-panel figure
├── run_v12_apex.py          # V12 zero-prior Track 5/7 + 1M-token CUDA apex
├── requirements.txt
├── pyproject.toml           # installable package, MIT
├── LICENSE                  # MIT
├── .gitignore
├── SPEC_V2.md               # this document (file name kept for compatibility)
└── README.md
```

### 2.1 Dependencies

```text
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

`pyproject.toml` declares `bedrock-tricore==1.6.0`, Python `>=3.10`, and the same runtime dependencies. `pip install -e .` installs the `tricore` import package.

### 2.2 Public API

```python
from tricore import (
    TriCoreBlock, TriCoreLM, BaselineTransformerLM,
    generate_associative_recall_batch, generate_contextual_code_lm_batch,
    generate_3hop_recall_batch, generate_4hop_recall_batch,
    load_real_python_corpus_batches,
)
```

`TriCoreBlock.forward(...)` returns `(y, surprise_trace)`:

- `y: [B, L, D]`
- `surprise_trace: [B, L]`

`TriCoreLM.forward(idx)` returns:

- `logits: [B, L, vocab_size]`
- `mean_surprise: [B, L]`, the mean of per-layer surprise traces

`BaselineTransformerLM.forward(idx)` returns `(logits, None)`.

`TriCoreLM(...)` accepts `reset_trace_tokens: tuple | None = None`. When set, it builds a boolean `[B, L]` mask and zeros `prev_k` after the delimiter update in both the fused scan and `forward_step`. The option adds no parameters and is `None` for all V6-V10 tasks.

`generate_associative_recall_batch(...)` returns `(inputs, targets)`, both `[B, L]`.

`generate_contextual_code_lm_batch(...)` returns `(inputs, targets, deref_mask)`:

- `inputs: [B, L]`, `targets: [B, L]` (`-100` at the final position)
- `deref_mask: [B, L]` bool, exactly one `True` per sample at the return-alias position

---

## 3. Model contract

### 3.1 JIT scan signature

```python
@torch.jit.script
def tricore_fused_scan(
    u_feat: torch.Tensor,        # [B, L, D]
    u_surp_bias: torch.Tensor,   # [B, L, D]
    w_gate_logits: torch.Tensor, # [B, L, H, 1, 1]
    elig_gate_logits: torch.Tensor, # [B, L, H, 1, 1]
    hop_weights: torch.Tensor,   # [B, L, max_loops+1]
    Q: torch.Tensor,             # [B, L, H, d_head]
    K: torch.Tensor,             # [B, L, H, d_head]
    V: torch.Tensor,             # [B, L, H, d_head]
    base_decay: torch.Tensor,    # [D]
    fast_decay: torch.Tensor,    # [1, H, 1, 1]
    fast_lr: torch.Tensor,       # [1, H, 1, 1]
    hop_proj: torch.Tensor,      # [max_loops, 1, 1, H, d_head, d_head]
    enable_surprise_gate: bool,
    enable_fast_weight: bool,
    enable_adaptive_loop: bool,
    max_loops: int,
    reset_mask: Optional[torch.Tensor] = None,  # [B, L] bool or None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Returns `(h_seq, mem_seq, surprise_trace)` with shapes:

- `h_seq: [B, L, D]`
- `mem_seq: [B, L, H*d_head]`; the routed multi-hop read. When `enable_adaptive_loop=False` it is the hop-1 read; when `enable_fast_weight=False` it is zeros.
- `surprise_trace: [B, L]`

The function allocates only O(1) state per sample:

- `h: [B, D]`
- `F_mat: [B, H, d_head, d_head]`
- `prev_k: [B, H, d_head, 1]`

and accumulates per-position outputs in lists. It never materializes a `[B, L, H, d_head, d_head]` fast-weight cache.

### 3.2 Causal input path

```math
u_t = \mathrm{LN}(x_t), \qquad \tilde{u}_t = \mathrm{SiLU}(W_u u_t + b_u)
```

```math
a_t = \mathrm{normalize}(W_a u_t)
```

```math
Q_t = \mathrm{normalize}(a_t + 0.15 W_q u_t), \quad
K_t = \mathrm{normalize}(a_t + 0.15 W_k u_t), \quad
V_t = a_t + 0.15 W_v u_t
```

```math
w_t^{\text{hop}} = \mathrm{softmax}(W_r u_t) \in \mathbb{R}^{L_{\max}+1}
```

All L2 normalization is over the head dimension (`dim=-1`). There is no positional embedding and no attention over sequence positions.

### 3.3 Engine I

Initial state `h_0 = 0`, shape `[B, D]`:

```math
e_t = |\tilde{u}_t - h_{t-1}|, \qquad g_t = \sigma(W_s u_t + e_t)
```

```math
s_t = \frac{1}{D}\sum_j g_{t,j}, \qquad
\omega_t = \sigma(W_g u_t + s_t - 0.5)
```

```math
\eta_{\text{eff},t} = \eta \odot \omega_t, \qquad
h_t = \lambda \odot (1 - 0.5 g_t) \odot h_{t-1} + g_t \odot \tilde{u}_t
```

`lambda = sigmoid(base_decay_logit)` has shape `[D]`; `eta = sigmoid(fast_lr_logit)` has shape `[1, H, 1, 1]`. The write gate is per head and shared across the head's matrix.

When `enable_surprise_gate=False`:

```math
g_t = 0.5, \quad s_t = 0.5, \quad \eta_{\text{eff},t} = 1.0, \quad \lambda_f = 1.0
```

and no learned write gate is applied. This is the No-Engine-I ablation.

### 3.4 Engine II

Initial state `F_0 = 0`, shape `[B, H, d_head, d_head]`; `prev_k = 0`, shape `[B, H, d_head, 1]`. For `t > 0`:

```math
F_t = \lambda_f \odot F_{t-1}
+ \eta_{\text{eff},t} \odot (V_t - F_{t-1}K_{t-1})K_{t-1}^{\top}
```

The read happens after the write:

```math
r_t^{(0)} = F_t Q_t
```

At `t=0`, no write occurs and `r_0^(0)=0`. At the end of each step, `prev_k = K_t`. The write is causal and shifted: at `m_i` in `[k_i, m_i, 0]`, it binds `k_i -> m_i`; at `v_i` in `[m_i, v_i, 0]`, it binds `m_i -> v_i`.

### 3.5 Engine III

```math
r^{(0)} = r_t^{(0)}, \qquad r_{\text{accum}} = w_{t,0}^{\text{hop}} r^{(0)}
```

For `i = 0, \ldots, L_{\max}-1`:

```math
q^{(i+1)} = \mathrm{normalize}\left(r^{(i)} + H_i r^{(i)}\right)
```

```math
r^{(i+1)} = F_t q^{(i+1)}, \qquad
r_{\text{accum}} \leftarrow r_{\text{accum}} + w_{t,i+1}^{\text{hop}} r^{(i+1)}
```

```math
r_t^{\text{out}} = r_{\text{accum}}
```

`r^(i)` is `[B, H, d_head, 1]`; `H_i` is `[1, H, d_head, d_head]`; normalization is L2 over `dim=-2`. With `max_loops=2`, the initial read is hop 1, `step=0` is hop 2, and `step=1` is an extra latent refinement. The loop count is fixed.

V9 fuses these equations into the causal scripted scan: each timestep uses its current `F_mat`, and the scan returns only the fused `mem_seq: [B, L, H*d_head]`. It never allocates or stacks `F_seq: [B, L, H, d_head, d_head]`. The state recurrence and the fixed-hop refinement remain causal and serial in `L`; the Python loop is over `max_loops`.

When `enable_adaptive_loop=False`, `r_t^out = r_t^(0)` and the refinement MLP is skipped.

### 3.6 Readout

```math
z_t = h_t + W_o \mathrm{vec}(r_t^{\text{out}})
```

```math
z_t \leftarrow z_t + s_t \odot \mathrm{MLP}(\mathrm{LN}(z_t)), \qquad
y_t = x_t + \mathrm{LN}(z_t)
```

### 3.7 Ablation semantics

| Flag | Effect |
| :--- | :--- |
| `enable_surprise_gate=False` | `g_t=0.5`, `s_t=0.5`, `eta_eff=1.0`, `decay=1.0`; no learned write gate |
| `enable_fast_weight=False` | Engine II skipped; `z=h_seq`; Engine III has no memory; refinement MLP still runs if adaptive loop is on |
| `enable_adaptive_loop=False` | Engine III skipped; `recalled=r_hop`; no re-query and no refinement MLP |
| All `True` | Full 3-in-1 block |

Flags gate compute only. All modules are constructed regardless of flags, so every TriCore variant reports the same parameter count.

### 3.8 `TriCoreLM` and `BaselineTransformerLM`

```python
TriCoreLM(
    vocab_size: int,
    d_model: int = 128,
    n_layers: int = 1,
    d_fast: int = 32,
    max_loops: int = 2,
    enable_surprise_gate: bool = True,
    enable_fast_weight: bool = True,
    enable_adaptive_loop: bool = True,
    separator_token: int | None = None,
    syntax_tokens: tuple | None = None,
    reset_trace_tokens: tuple | None = None,
    enable_delta_rule: bool = True,
    autonomous_syntax: bool = False,
    autonomous_entity_max_freq: int = 4,
)
```

When `autonomous_syntax=True` and all three token-role arguments are `None`,
`TriCoreLM` discovers the roles from the sequence itself: tokens occurring at
least `max(6, L // 10)` times are structural; the structural token at the last
position is the delimiter (`reset_mask`); any other structural token whose
predecessor occurs at most `autonomous_entity_max_freq` times is an infix
connector (eligibility gate closed).  The write gate closes on every structural
token.  The discovery is purely a function of equality/frequency, so arbitrary
vocabulary permutations produce identical masks.  It adds no parameters and no
recurrent state.  The bound defaults to `4` (V12); V13 sets it to `6` because
four repeated overwrites plus one declaration and one NOP make each hot
variable occur six times.

`d_fast` is passed to `TriCoreBlock` as `d_head`; `n_heads` is fixed at 4. The embedding has no positional table. `final_norm` and a bias-free `lm_head` produce logits. The returned surprise is the mean across layers.

```python
BaselineTransformerLM(
    vocab_size: int,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 2048,
)
```

The baseline uses learned absolute positional embeddings, pre-norm `TransformerEncoderLayer`, GELU, FFN width `2*d_model`, dropout 0, and a causal mask. It has two physical layers and no latent loop.

### 3.9 V7 native streaming API

V7 adds two methods to `TriCoreBlock` and matching wrappers on `TriCoreLM` without changing `forward`:

```python
h, F_mat, prev_k = block.init_state(batch_size, device, dtype=torch.float32)
# h:    [B, D]
# F_mat:[B, H, d_head, d_head]
# prev_k:[B, H, d_head, 1]

out_t, surprise_t, new_state = block.forward_step(
    x_t, state, step_idx,
    enable_surprise_gate=True,
    enable_fast_weight=True,
    enable_adaptive_loop=True,
)
logits_t, surprise_t, new_states = lm.forward_step(idx_t, states, step_idx)
```

`TriCoreBlock.forward_step` performs the same causal-shifted Engine I update, Engine II delta-rule write/read, fixed-hop Engine III refinement, and readout as `forward` for one timestep. `step_idx == 0` skips the fast-weight write exactly as the scan does. `TriCoreLM.init_state` returns one block state per layer; a single block state is also accepted for a one-layer model. Calling `forward_step` for `t=0..L-1` and stacking the logits must match `forward(idx)`; V9 checks both logits and surprise with `< 1e-6` maximum absolute error.

The streaming state excludes the old training-only `F_seq` tensor entirely. `forward_step` has O(1) persistent inference state, and V9's training scan no longer materializes `F_seq`; it returns the fused `mem_seq` read instead.

### 3.10 V12 zero-prior syntax discovery

With `autonomous_syntax=True` and all token-role arguments `None`, the LM builds three boolean masks from `idx` alone:

```text
freq[b,t]        = sum_u (idx[b,t] == idx[b,u])
structural       = freq >= max(6, L // 10)
delimiter        = structural & (idx == idx[:, -1:])
infix_connector  = structural & ~delimiter & (freq_at_previous_position <= 4)
```

`write_sep_mask = structural` closes the write gate at every repeated syntax token. `elig_sep_mask = infix_connector` closes the eligibility gate only at the infix connector. `reset_mask = delimiter` clears `prev_k` after the delimiter update. All three are computed once per full forward and passed to every layer.

The eligibility gate has two coupled effects:

```math
q_t^{\text{read}} = \mathrm{normalize}\left(e_t q_t + (1-e_t)\,\text{prev\_k}_t\right)
```

```math
\text{prev\_k}_{t+1} = \mathrm{normalize}\left((1-e_t)\,\text{prev\_k}_t + e_t K_t\right)
```

Thus `e_t \approx 0` at `alias EQ var` reads the alias trace and freezes `prev_k`; at the following `var` the value is written at the alias key, producing the shifted binding `F(alias)=var`. This is why the direction of the gate matters: an open-at-EQ convention would write `F(EQ)=var` and could not support `alias -> var -> lit`.

The write rule remains

```math
F_t = \lambda F_{t-1} + \eta_{\text{eff}}\,(V_t - F_{t-1}K_{t-1})\,K_{t-1}^{T}
```

when `enable_delta_rule=True`, and drops the erase term when it is `False`. The masks add no parameters and no recurrent state. Discovery is O(B L^2) in token-equality comparisons for a short closed-syntax sequence; it is disabled for the 1M-token CUDA profile. The streaming `forward_step` API operates on one token and therefore has no sequence-frequency context; autonomous discovery is defined for the full-sequence training path.

---

## 4. State, parameter, and cache accounting

### 4.1 Persistent state

For `D=128`, `H=4`, `d_head=32`, `n_layers=1`:

| State | Shape per core | Floats per core | Bytes per core (FP32) |
| :--- | :---: | :---: | :---: |
| `h` | `[B, 128]` | 128 | 512 |
| `F_mat` | `[B, 4, 32, 32]` | 4,096 | 16,384 |
| `prev_k` | `[B, 4, 32, 1]` | 128 | 512 |
| **Strict total** | | **4,352** | **17,408** |

```math
C_{\text{TriCore}} = \frac{(D + H d_{\text{head}}^2 + H d_{\text{head}}) \cdot 4}{1024}\ \text{KiB} = 17.00\ \text{KiB}
```

The persistent inference state is exactly these tensors; it is O(1) in `L`. V9's `forward_step` carries only `h`, `F_mat`, and `prev_k`. The training scan uses the same O(1) state recurrence and fuses Engine III per timestep, so it no longer allocates `F_seq: [B, L, H, d_head, d_head]`; training still stores `h_seq`, `mem_seq`, and autograd buffers, so it is not activation-O(1). The measured V7 streaming state is exactly 17,408 bytes per sample at FP32 and stays constant from `L=128` to `L=4096`.

### 4.2 Parameters

| Model | Parameters |
| :--- | :---: |
| TriCore Full, one layer, `d_model=128`, `d_fast=32` | 207,503 |
| Transformer, two layers, `d_model=128`, FFN `2*d_model` | 543,744 |

The TriCore count is identical across all four flag combinations because flags gate compute, not parameter registration.

### 4.3 KV-cache comparison

```math
C_{\text{Transformer}} = \frac{n_{\text{layers}} \cdot 2 \cdot L \cdot D \cdot 4}{1024}\ \text{KiB}
```

At `L=32,768`, `n_layers=2`, `D=128`, the Transformer cache is `65,536 KiB`; the ratio is `65,536 / 17.0 = 3,855.1x`. This is an arithmetic state comparison, not a measured decode benchmark.

---

## 5. Task specifications

### 5.1 Track 1 — 2-Hop Chained Associative Recall

For `n` chains, `q` queries, and `noise_len` distractor tokens:

```text
Hop 1: [k_1, m_1, 0] ... [k_n, m_n, 0]         3n tokens
Hop 2: [m_1, v_1, 0] ... [m_n, v_n, 0]         3n tokens
Gap:   [m_i, 0] / [k_i, 0] pairs              noise_len tokens
Query: [k_q, 0] ...                            2q tokens
```

The two hop blocks are independently shuffled. The gap contains unpaired bridge and key tokens; under an ungated exact delta, these erase parts of the chain. At each query position, the target is the value two hops away; all other targets are `-100`.

Reference settings: `n=6`, `q=4`, `noise_len=16`, `vocab_size=64`; sequence length `60`. The 4x extrapolation uses `noise_len=64`; sequence length `108`.

### 5.2 Track 2 — Causal Bytecode LM

Token layout:

| ID | Meaning |
| :---: | :--- |
| 0 | `SEMI` |
| 1 .. 18 | alias tokens |
| 19 .. 36 | variable pointer tokens |
| 37 .. 54 | literal value tokens |
| 55 .. 59 | reserved |
| 60 | `LET` |
| 61 | `ALIAS` |
| 62 | `NOP` |
| 63 | `RET` |

The generator draws `n` distinct tokens from each of the three disjoint tiers. For `n=6`, a sample uses 6 alias tokens from `1..18`, 6 variable tokens from `19..36`, and 6 literals from `37..54`. The control tokens are `60..63`, so no data token can collide with a statement keyword.

Each sample contains, in random statement order:

```text
[LET,   alias_i, var_i, SEMI]   # alias_i -> var_i
[ALIAS, var_i,   lit_i, SEMI]   # var_i   -> lit_i
[NOP,   var_i,   SEMI,  SEMI]  (m times)
```

and terminates with:

```text
[RET, alias_i, lit_i, SEMI]
```

Sequence length is `(n*2 + m + q) * 4`; for `n=6`, `m=4`, `q=4`, it is `80`. Targets are shifted next-token targets with `-100` at the final position. `deref_mask` is true at `pos + 1` of each return statement, where the input is `alias_i` and the target is `lit_i`. Because the return uses an alias and the namespaces are disjoint, a correct prediction must resolve `alias_i -> var_i -> lit_i`, not copy a surface token. Chance is `1/n` (`16.67%` for `n=6`).

### 5.3 Track 3A — 3-Hop Chained Recall

For `n` chains, `q` queries, and an even `noise_len`:

```text
[k_i,  m1_i, 0] * n     # shuffled hop-1 facts
[m1_i, m2_i, 0] * n    # shuffled hop-2 facts
[m2_i, v_i,  0] * n    # shuffled hop-3 facts
[m1_i, 0, m2_i, 0, ...]  # noise_len adversarial distractor tokens
[k_q, 0] * q             # queries; target v_q at the k_q position
```

The four token tiers are disjoint. The reference settings are `n=4`, `q=4`, `noise_len=12`, `vocab_size=64`; sequence length is `4*3*3 + 12 + 4*2 = 56`. The 4x evaluation uses `noise_len=48`, sequence length `92`. `max_loops=1` performs the initial read plus one refinement (two hops); `max_loops=2` performs the initial read plus two refinements (three hops). Chance is `1/n = 25%`.

### 5.4 Track 3B — Real Python Source Held-Out LM

`load_real_python_corpus_batches` concatenates the actual bytes of `tricore/model.py`, `tricore/tasks.py`, and `run_experiment.py`, replaces non-ASCII bytes with `?`, and builds an ASCII byte vocabulary (`0..127`). The first 85% is training data; the final 15% is a strict held-out suffix. Training windows satisfy `start + seq_len + 1 <= split`, and validation windows start at or after `split`. Reference settings: `batch_size=32`, `seq_len=96`, 300 steps.

### 5.5 Track 7 — Stateful live variable overwrite

Each sequence contains five-token statements with shuffled ordering:

```text
[LET, alias_i, EQ, var_i,     SEMI]   # alias_i -> var_i
[SET, var_i,   EQ, lit_old_i, SEMI]   # var_i   -> old literal
[SET, var_i,   EQ, lit_new_i, SEMI]   # live overwrite
[NOP, var_i,   EQ, SEMI,      SEMI]   # adversarial no-op
[RET, alias_q, EQ, SEMI,      SEMI]   # target at EQ = lit_new_q
```

The query target is the **new** literal at the EQ position. A Hebbian memory accumulates `old + new` and confuses the stale value; the Delta rule erases the old value with `V_t - F_{t-1}K_{t-1}` before writing. Reference settings: `batch_size=64`, `num_vars=4`, `num_queries=4`, `num_nops=4`, sequence length 100, 400 steps, seeds 42/100/2026, 25.0% chance.

---

## 6. Training and evaluation protocol

### 6.1 Track 1

- 500 optimizer steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`.
- Training seed `42`; fixed evaluation batches use seeds `2026` (in-domain) and `2027` (4x gap), 10 batches of 32.
- Variants: Full, No Engine I, No Engine II, No Engine III, 2-layer Transformer.
- Recall is the mean over query positions whose target is not `-100`.

### 6.2 Track 2

- 450 optimizer steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`.
- Training seed `42`; fixed evaluation batches use seed `3030`, 8 batches of 64.
- Models: TriCore one layer (`d_model=128`, `d_fast=32`, `max_loops=2`) and 2-layer Transformer.
- Objective: `0.25 * dense_loss + deref_loss`, where dense loss is next-token cross-entropy over all positions and deref loss is cross-entropy on the return-alias mask.
- Metrics: dense next-token cross-entropy, PPL `exp(loss)`, and dereference accuracy at the return-alias position.

### 6.3 Track 3A

- 450 optimizer steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`, seed `42`.
- Models: one-layer TriCore Full (`max_loops=2`, 207,503 parameters), one-layer TriCore (`max_loops=1`, two hops), and a 2-layer Transformer.
- Fixed evaluation: 5 batches of 64 for in-domain (`noise_len=12`, seed 5101) and 5 batches of 64 for the 4x gap (`noise_len=48`, seed 5102).
- Metric: query-position accuracy only, where `target != -100`.

### 6.4 Track 3B

- 300 optimizer steps, batch 32, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`, seed `42`.
- Models: one-layer TriCore (`d_model=128`, `d_fast=32`, `max_loops=2`) and a parameter-matched one-layer Transformer (`d_model=144`, `max_seq_len=96`).
- Fixed held-out suffix: final 15% of the concatenated source stream; non-overlapping 96-token windows.
- Metrics: mean next-byte cross-entropy, PPL `exp(loss)`, and next-byte accuracy.

### 6.5 Timing

The V6 Full timing includes data generation, forward, backward, and optimizer steps for 500 steps. A first scripted call warms up the JIT and is excluded from the timed loop. The V4 reference time is 136.0 s from the prior Python-loop run and the V5 reference is 125.3 s; neither is a simultaneous measurement. Timings are single-machine wall-clock values. V7 streaming latency is measured over 4,096 one-token steps at batch size 1 and is not a serving benchmark.

### 6.6 Evaluation limits

There is no multi-seed confidence interval for the full V6/V7 Track 1/2/3 reference runs, no hyperparameter search for the Transformer, and no FLOP matching. V9 adds a 4-chain, five-way, 3-seed mean ± std check for Track 4A (`d_model=64`, 4 chains, 4 queries); V11 adds a three-seed Track 5 Welch check. Neither repowers the full-size V6/V7 runs, and the three-observation Welch p-values are not population inference. Track 1/2 have no real held-out set. Track 3B has a real-source held-out suffix but the corpus is only this small repository. Track 2, Track 3A, and Track 6 are synthetic; Track 3B is not a human-code benchmark.

---

## 7. Latest benchmark record

**Generated by:** `python run_experiment.py`; V9/V11/V12 append via `run_v9_verification.py`, `run_v11_flagship.py`, and `run_v12_apex.py`  
**Report:** `results/BENCHMARK_REPORT.md`  
**Plots:** `results/tricore_vs_transformer.png`, `results/tricore_v12_apex.png`, `results/tricore_v11_masterpiece.png`  
**Device:** CUDA

### 7.1 V6 timing versus V4 and V5

| Measurement | Time | Note |
| :--- | :---: | :--- |
| V4 Full, Python per-token loop | 136.0 s | reference run |
| V5 Full, TorchScript fused scan | 125.3 s | reference run |
| V6 Full, vectorized multi-hop scan | **80.0 s** | timed 500 steps |
| V6 / V4 speed ratio | **1.70x** | >1 faster, <1 slower |
| V6 / V5 speed ratio | **1.57x** | >1 faster, <1 slower |
| V6 first-call warm-up | 0.4 s | excluded |

### 7.2 Track 1: five-way ablation

| Architecture variant | Parameters | Final val loss | In-domain 2-hop recall | 4x noise-gap extrapolation | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (3-in-1)** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 80.0 s |
| TriCore No Engine I: Surprise Gate | 207,503 | 3.0005 | 5.55% | 4.14% | 62.3 s |
| TriCore No Engine II: Fast-Weight | 207,503 | 3.0008 | 5.78% | 3.98% | 36.1 s |
| TriCore No Engine III: Adaptive Loop | 207,503 | 1.8660 | 16.48% | 17.50% | 63.4 s |
| Vanilla Transformer (2-layer) | 543,744 | 2.3131 | 16.72% | 14.92% | 18.6 s |

**Interpretation.** Full solves the task. Engine II is the associative memory; without it, recall is below chance. Engine III is the second hop; without it, recall is chance. Engine I now collapses to `5.55% / 4.14%` when removed, unlike the V5 record (`91.88% / 84.14%`); the equations are unchanged, but the vectorized hop path changes the optimization trajectory, and the result remains single-seed. The untuned Transformer is at chance; this is not an attention ceiling.

### 7.3 Track 2: Causal Bytecode LM

| Model | Parameters | Validation loss | Validation PPL | Dereference accuracy | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (1-layer)** | 207,503 | 1.1529 | **3.17** | **100.00%** | 78.5 s |
| Vanilla Transformer (2-layer) | 543,744 | 1.2686 | 3.56 | 19.63% | 19.5 s |

**Interpretation.** TriCore reaches `PPL 3.17` and `100.00%` dereference accuracy; the Transformer reaches `3.56` and `19.63%` against a 6-way chance level of `16.67%`. The 100% result is on a closed deterministic bytecode grammar with no held-out templates. It shows that the causal `alias -> var -> lit` lookup is learnable, not that the model understands real code. This is a single-seed synthetic result.

### 7.4 Cache scaling

| Sequence length | Transformer KV cache | TriCore O(1) state | Reduction |
| :---: | :---: | :---: | :---: |
| 128 | 256.00 KB | 17.00 KB | 15.1x |
| 512 | 1,024.00 KB | 17.00 KB | 60.2x |
| 2,048 | 4,096.00 KB | 17.00 KB | 240.9x |
| 8,192 | 16,384.00 KB | 17.00 KB | 963.8x |
| 16,384 | 32,768.00 KB | 17.00 KB | 1,927.5x |
| 32,768 | 65,536.00 KB | 17.00 KB | **3,855.1x** |

### 7.5 V7 streaming

| Check | Result |
| :--- | :---: |
| Max absolute logit difference (`forward` vs `forward_step`) | **5.960e-07** |
| Max absolute surprise difference | **1.192e-07** |
| Streaming latency, batch 1, 4,096 steps | **2.160 ms/token** |
| Persistent state, `L=128/512/2,048/4,096` | **17,408 bytes = 17.00 KiB** |

`forward_step` carries only `(h, F_mat, prev_k)`; it does not create or grow `F_seq`. The latency is a single-machine per-token loop, not a serving throughput result.

### 7.6 Track 3A: 3-hop chained reasoning

| Model | Parameters | In-domain val loss | In-domain 3-hop acc | 4x gap extrapolation | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (1-layer, `max_loops=2`)** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 66.4 s |
| TriCore (`max_loops=1`, 2-hop limit) | 203,278 | 1.8395 | 25.39% | 23.91% | 62.8 s |
| Vanilla Transformer (2-layer) | 297,984 | 2.2188 | 23.83% | 23.44% | 16.7 s |

Chance is 25% (four values). The second refinement is required to complete `k_q -> m1_q -> m2_q -> v_q`; removing it drops in-domain accuracy from `100.00%` to `25.39%` (`-74.61` points).

### 7.7 Track 3B: real Python source held-out LM

| Model | Parameters | Held-out loss | Held-out PPL | Next-byte acc | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (1-layer)** | 223,887 | **2.9240** | **18.616** | **41.47%** | 57.7 s |
| Vanilla Transformer (1-layer, param-matched) | 218,448 | 2.9667 | 19.428 | 28.15% | 1.5 s |

The corpus is the ASCII byte stream of this repository's `tricore/model.py`, `tricore/tasks.py`, and `run_experiment.py`; the final 15% is an unseen suffix. The Transformer is parameter-matched with a short position table, but TriCore is about 38x slower in this small training loop. This is a single-seed quality/budget observation, not a throughput or human-code claim.

### 7.8 Track 4A: V9 4-chain five-way 3-seed verification

`run_v9_verification.py` runs a 4-chain 2-Hop MQAR under seeds `42/100/2026`: 350 steps, batch 64, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, clipping `1.0`; `d_model=64`, `d_fast=16`, 4 heads, `max_loops=2`, 4 chains, 4 queries, training `noise_len=8`, `vocab_size=64` (sequence length 40). The 4x OOD test uses `noise_len=32` (sequence length 64). Metrics are mean ± std over seeds on 5 fixed evaluation batches of 64 examples; random baseline is 25.0%.

| Variant | Parameters | In-domain recall | 4x-gap recall | Mean train time |
| :--- | :---: | :---: | :---: | :---: |
| **TriCore Full (`max_loops=2`)** | 56,655 | **100.00 ± 0.00%** | **100.00 ± 0.00%** | 59.60 s |
| TriCore No Engine I: Surprise Gate | 56,655 | 5.65 ± 0.20% | 5.55 ± 0.83% | 48.52 s |
| TriCore No Engine II: Fast-Weight | 56,655 | 5.55 ± 0.34% | 5.31 ± 0.95% | 9.39 s |
| TriCore No Engine III: Adaptive Loop | 56,655 | 23.41 ± 1.08% | 23.54 ± 2.73% | 26.02 s |
| Vanilla Transformer (2-layer) | 83,456 | 25.08 ± 0.48% | 22.24 ± 3.96% | 2.01 s |

Only Full saturates both evaluations. All ablations are at or below the `25.0%` four-way chance level. On the 4x gap, Full is `+94.45` points over No Engine I, `+94.69` over No Engine II, `+76.46` over No Engine III, and `+77.76` over the untuned Transformer on the seed mean. Full has zero seed variance; all non-Full variants have non-zero variance. With three seeds this is a variance check, not a significance test, and the compact configuration is not a replacement for the V6/V7 reference runs.

### 7.9 Track 4B: measured CUDA decode state and KV cache to 32K

Measured on CUDA, batch size 1, FP32, 8 warm-up + 32 timed steps. `CUDA Alloc Δ` is the `torch.cuda.memory_allocated()` delta after `empty_cache()`. The baseline performs a real cached 2-layer Transformer decode with scaled dot-product attention over the full causal cache; it is not a formula estimate.

| Length | TriCore state bytes | CUDA Alloc Δ | Transformer K+V bytes | CUDA Alloc Δ | Reduction | TriCore ms/token | Transformer ms/token |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 17,408 | 17,408 | 262,144 | 262,144 | **15.1x** | 2.448 | 1.474 |
| 512 | 17,408 | 17,408 | 1,048,576 | 1,048,576 | **60.2x** | 1.804 | 1.049 |
| 2,048 | 17,408 | 17,408 | 4,194,304 | 4,718,592 | **240.9x** | 1.756 | 1.274 |
| 8,192 | 17,408 | 17,408 | 16,777,216 | 16,777,216 | **963.8x** | 1.755 | 3.011 |
| 16,384 | 17,408 | 17,408 | 33,554,432 | 33,554,432 | **1,927.5x** | 2.047 | 4.993 |
| 32,768 | 17,408 | 17,408 | 67,108,864 | 67,108,864 | **3,855.1x** | 1.939 | 9.572 |

The TriCore state is `(D + H*d_head^2 + H*d_head) * 4 = 17,408` bytes = **17.00 KiB per sample** for `D=128`, `H=4`, `d_head=32`, independent of `L`. The baseline cache is `2 * n_layers * L * D * 4`, i.e. 2.00 KiB per token, reaching **64.00 MiB at `L=32,768`**. The `L=2,048` allocator delta is `4,718,592 B` (4.50 MiB) while the tensor bytes are `4,194,304 B` (4.00 MiB); the reduction column uses tensor bytes. The measured Transformer is faster at short lengths, but the per-token latency inverts at `L=8,192` (`3.011` vs TriCore `1.755` ms/token) and reaches `9.572` vs `1.939` ms/token at `L=32,768`. This is a measured memory and latency result for one machine at B=1, not a serving-throughput claim.

### 7.10 V11 Track 6, three-seed Track 5, and 128K Track 4B

**Track 6 (single seed 42, 500 steps, batch 64).** Four shuffled fact tiers, 12 distractors, four queries, 25.0% chance; 4x OOD uses 48 distractors. Track 6 TriCore variants use `separator_token=0`, `syntax_tokens=(0,)`, `reset_trace_tokens=(0,)`. The Transformer does not receive that prior.

| Variant | Params | In-domain | 4x OOD | Query router `[h1,h2,h3,h4]` | Train time |
| :--- | ---: | ---: | ---: | :--- | ---: |
| **TriCore Full (`max_loops=3`)** | 236,436 | **100.00%** | **100.00%** | 0.005 / 0.004 / 0.003 / **0.988** | 214.2 s |
| TriCore (`max_loops=2`, 3-hop limit) | 232,211 | 24.95% | 24.76% | 0.002 / **0.997** / 0.001 / 0.000 | 184.0 s |
| TriCore (`max_loops=1`, 2-hop limit) | 227,986 | 25.73% | 26.27% | 0.019 / **0.981** / 0.000 / 0.000 | 148.8 s |
| Vanilla Transformer (2-layer) | 306,176 | 24.41% | 25.78% | n/a | 4.0 s |

**Track 5 (seeds 42/100/2026, 350 steps, batch 64).**

| Variant | Params | Val loss | Accuracy | Gate `EQ` / entity | Mean train time |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V11 Full** | 60,819 | **0.00218 ± 0.00022** | **100.00 ± 0.00%** | 0.0032 / 0.9054 | 156.3 s |
| TriCore V11 Fair No-Engine-I | 60,819 | 1.4682 ± 0.0186 | 24.80 ± 1.12% | 1.0000 / 1.0000 | 129.0 s |
| Vanilla Transformer (2-layer) | 83,456 | 1.4397 ± 0.0075 | 25.01 ± 0.63% | n/a / n/a | 14.8 s |

Welch two-tailed: Full vs Fair accuracy `t=116.649`, `df=2.000`, `p=7.35e-05`; Full vs Transformer accuracy `t=206.869`, `p=2.34e-05`; Full vs Fair loss `p=5.34e-05`; Full vs Transformer loss `p=8.81e-06`.

**Track 4B Extended.** Measured B=1 CUDA state/KV allocation and cached-decode latency on an RTX 3060. The state stays at 17,408 B; the Transformer cache reaches 268,435,456 B (256.00 MiB) at `L=131,072`, a 15,420.2x tensor-byte reduction. TriCore latency stays in 2.57-3.41 ms/token versus 1.39-35.55 ms/token for the Transformer; the crossover is at `L=8,192`. Full V11 suite runtime was **1,556.8 s**.

---

## 8. Verification record

### 8.1 JIT equivalence

`tests/test_v5.py` compares `tricore_fused_scan` to a Python reference for the four combinations of its two scan flags:

1. `enable_surprise_gate=True, enable_fast_weight=True`
2. `enable_surprise_gate=False, enable_fast_weight=True` (No Engine I)
3. `enable_surprise_gate=True, enable_fast_weight=False` (No Engine II)
4. both flags `False`

Engine III is fused inside the scripted scan and is exercised through `TriCoreLM`; the causal bytecode task has separate shape, mask, and backward checks.

All three outputs (`h_seq`, `mem_seq`, `surprise_trace`) must satisfy `torch.allclose(atol=1e-6, rtol=1e-5)`. The test also checks that full `forward` matches token-by-token `forward_step` with `< 1e-6` maximum absolute error, plus the causal bytecode shapes, the dereference mask at `pos + 1`, and backward gradients.

### 8.2 Sanity

`tests/test_sanity.py` checks forward/backward, output shapes, and gradient flow for `TriCoreLM`.

### 8.3 V7 streaming and Track 3

`tests/test_v7_streaming.py` checks:

1. `TriCoreLM.eval()` full-sequence logits match stacked `forward_step` logits under `torch.allclose(atol=1e-5)`; surprise traces are also compared.
2. The persistent state is exactly `batch_size * 17,408` bytes at `t=16` and `t=512`.
3. `generate_3hop_recall_batch` shapes/targets and the real Python corpus loader support a finite gradient backward pass.
4. `generate_4hop_recall_batch` produces shape `(4, 68)` with 16 valid targets; the V11 model uses `reset_trace_tokens=(0,)`; finite gradients and full-vs-`forward_step` parity are checked at `atol=1e-4`.

The complete suite currently reports **12 passing tests**. `run_v7_benchmark.py` regenerates only the V7 streaming and Track 3 sections; it preserves the V6 Track 1/Track 2 tables in `results/BENCHMARK_REPORT.md`. V12 adds the Delta-rule on/off parity test and a zero-prior mask test that compares the exact Track 5/Track 7 EQ/SEMI masks before and after a random bijection of all 64 vocabulary IDs. v1.6.3 adds the native `generate()` shape/state test and the `inspect_telemetry()` key/shape contract test.

V9 adds no new parameter names. `run_v9_verification.py` exercises the zero-`F_seq` training path under three seeds and writes `results/V9_METRICS.json`; `demo_showcase.py` exercises `init_state` / `forward_step` end to end. Track 4A is a 4-chain, five-way, 3-seed rerun, and Track 4B is a real CUDA allocation/latency measurement through `L=32,768`, both reported in sections 7.8-7.9. V11 adds no trainable parameters, writes `results/V11_METRICS.json`, and extends the measured protocol to 128K; its tests and numbers are in sections 7.10 and 8.3.

### 8.4 Full run

The V6 reference run completed on CUDA and regenerated the report and plot. `run_v7_benchmark.py` then appended the V7 streaming and Track 3 sections in place, preserving the V6 tables. `run_v11_flagship.py` writes the V11 Track 5/6/4B metrics and replaces the six-panel master figure; the full V11 run took 1,556.8 s. The report is the source of truth for all numbers in this document. The legacy V6 plot is a 2x2 panel:

- Track 1 learning curves
- Track 1 ablation bars
- Track 2 validation PPL
- Track 2 dereference accuracy

V12's 2x3 master figure (`results/tricore_v12_apex.png`, also written to `results/tricore_vs_transformer.png`) adds zero-prior Track 5, Track 7 Delta/Hebbian, memory to 1M, latency to 1M, and code modeling. The V11 figure (`results/tricore_v11_masterpiece.png`) remains as the historical four-hop/128K summary.

### 8.5 Commands

```bash
python -m pytest -q
python tests/test_v5.py
python tests/test_v7_streaming.py
python run_experiment.py
python run_v7_benchmark.py
python run_v9_verification.py
python run_v11_flagship.py
python run_v12_apex.py
python demo_showcase.py
python run_experiment.py --steps 2 --code-steps 2  # smoke
```

---

## 9. Invariants and open work

### 9.1 Invariants

1. **No attention over tokens.** No softmax may mix sequence positions. The only softmax is the hop router over `max_loops + 1` slots.
2. **O(1) state.** `h`, `F_mat`, and `prev_k` shapes must be independent of `L`.
3. **Causal shifted write.** Engine II binds `K_{t-1}` to `V_t`; the read uses `Q_t` after the write.
4. **Selectable fast-weight rule.** `enable_delta_rule=True` writes `(V_t - F_{t-1}K_{t-1})K_{t-1}^T`; `False` uses `V_t K_{t-1}^T` with the same decay and learning rate. Both keep `F_mat` as `[B, H, d_head, d_head]` and do not change the 17,408-byte state.
5. **Per-head matrices.** `F_mat` is `[B, H, d_head, d_head]`.
6. **Canonical address.** Q/K/V retain the shared normalized address plus residual projections.
7. **Write protection.** Engine I retains a learned write gate; `enable_surprise_gate=False` is the unconditional-delta ablation.
8. **Latent re-query.** Engine III maps the previous read back into query space and re-queries the same `F_mat`.
9. **Independent ablations.** Flags gate compute without changing parameter registration.
10. **JIT equivalence.** Any change to `tricore_fused_scan` must pass `tests/test_v5.py` on all three returned tensors and keep `forward` versus `forward_step` below `1e-6` maximum absolute error.
11. **Fixed loop count.** `max_loops` is a constant; do not describe Engine III as adaptive computation time.
12. **Zero `F_seq`.** Engine III must re-query the current `F_mat` inside the causal scan and return only `mem_seq`; no training path may allocate or stack `F_seq: [B, L, H, d_head, d_head]`.
13. **Causal dereference.** In Track 2, `deref_mask` must mark the return alias position `pos + 1`, never the literal position, and `LET`/`ALIAS` must not place the answer next to the query alias.
14. **Reproducible artifacts.** `python run_experiment.py` must regenerate the report and plot, and state accounting must include `prev_k`.

### 9.2 Open work

- More seeds and a significance test for the full-size V6/V7 tracks (V9 does 4-chain Track 4A; V11 Track 5 does three-seed Welch on the smaller infix task).
- 5-8 hop chains and variable `max_loops`; V11 demonstrates four hops in one forward pass.
- Stronger adversarial writes and higher noise density.
- Streaming latency and memory beyond the measured 1M range and against a production serving stack.
- A fused CUDA/Triton scan to replace the TorchScript loop.
- Tuned, parameter-matched Transformer baselines at matched compute as well as parameters.
- A larger real-code corpus with held-out files/repositories for Track 3B.
- Natural-language pretraining beyond synthetic recall.

---

<p align="center">
  <i>TriCore V9 — zero-F_seq training, constant state, surprise-gated writes, causal delta memory, a fused latent loop, and native one-token streaming.</i>
</p>