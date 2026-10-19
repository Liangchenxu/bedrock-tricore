# TriCore V3

**An attention-free sequence operator with a constant-size recurrent state, causal shifted fast-weight binding, and surprise-scaled refinement.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Ready-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/sanity%20test-passing-brightgreen)](#verification)

> **Status:** experimental research reference, not a production language model. Every number below comes from a single-seed, 600-step synthetic MQAR run. Read [Limitations](#limitations) before generalizing.

> **TL;DR.** On the controlled Multi-Query Associative Recall (MQAR) benchmark, Full TriCore reaches **100.00% in-domain recall** and **99.45% recall under a 4x longer distractor gap**, with a **33 KB reported O(1) state** instead of a KV cache that grows with context. Removing fast weights collapses recall to **3.28% / 3.83%**, identifying causal shifted key-value binding as the decisive mechanism. Removing the adaptive loop changes nothing measurable on this task. The implementation is slower than the Transformer baseline because its recurrence is a Python per-token loop.

> **中文速览.** TriCore V3 是一个无注意力、推理状态为 O(1) 的序列算子。在 600 步、单 seed 的受控 MQAR 任务上，Full 版本达到 100.00% 域内召回与 99.45% 的 4× 噪声间隔外推；去掉快权重后跌至 3.28% / 3.83%，说明“因果移位键值绑定”是核心。Engine III 在该任务上没有可测增益，不应被宣传为精度来源。按报告口径，O(1) 状态为 33 KB，而 Transformer 在 32K 上下文下 KV cache 为 65,536 KB（约 1985.9×）；若计入实现中的 `prev_k`，真实常驻状态约 34 KB。TriCore 当前 Python 逐 token 实现比 Transformer 慢，且结果不外推到自然语言或大规模训练。

---

## Contents

- [1. The problem](#1-the-problem)
- [2. Architecture](#2-architecture)
- [3. What changed in V3](#3-what-changed-in-v3)
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

A third limitation is less discussed: the weights of a trained Transformer are static at inference time. It can attend to context, but it cannot perform a local, associative write into its own state without another forward pass.

TriCore explores a different trade-off:

> **Can a sequence model keep a constant-size state, write key-value associations into fast weights, and spend refinement compute only where prediction error is high?**

The answer validated here is deliberately narrow. On MQAR, a causal shifted fast-weight memory solves the associative recall task with an O(1) state. This README does **not** claim that the same result transfers to natural language, reasoning, or scale. It documents the operator, the exact equations, the ablation, and the boundary of the evidence.

### Design goals

| Goal | TriCore V3 mechanism | Evidence in this repository |
| :--- | :--- | :--- |
| No attention | Causal convolution + recurrent state + fast weights | `tricore/model.py` contains no softmax attention |
| O(1) inference state | `h_t` and `F_t` have shapes independent of `L` | State-cache table below |
| Associative memory | Causal `K_{t-1} -> V_t` delta-rule writes | No-Fast-Weight ablation collapses to 3.28% |
| Test-time adaptation | Local delta-rule update, no inference-time backprop | Engine II equations |
| Surprise-aware compute | Prediction-error gate scales state updates and refinement | Engine I / III equations |
| Independent ablations | `enable_fast_weight`, `enable_adaptive_loop` | Four-way benchmark |

---

## 2. Architecture

**Notation.** `x_t ∈ R^D` is the input at position `t`; `LN` is layer normalization; `σ` is the sigmoid; `⊙` is elementwise multiplication; `normalize(·)` is L2 normalization over the head dimension. The implementation uses `D = 128`, `H = 4` heads, `d_head = 32`, `L_max = 2`, and two layers.

### 2.1 Causal input path

The block first computes a local, causal feature stream. The depthwise convolution has kernel size 3, left padding 2, and the output is truncated to length `L`:

```math
u_t = \operatorname{LN}(x_t), \qquad
c_t = \operatorname{SiLU}\!\left(b + \sum_{j=0}^{2} w_j \odot u_{t-j}\right),
\qquad
\tilde{u}_t = W_u c_t + b_u .
```

`c_t` depends only on `u_{t-2}, u_{t-1}, u_t`. There is no future leakage. Q, K, V, and `ũ` are projected for all positions in parallel before the recurrence; only the state update itself is sequential.

### 2.2 Engine I — surprise-gated state collapse

Engine I compresses the local feature stream into a persistent vector state `h_t ∈ R^D`:

```math
e_t = \left| \tilde{u}_t - h_{t-1} \right|,
```

```math
g_t = \sigma\!\left(W_s e_t + b_s\right),
```

```math
h_t = \lambda \odot \left(1 - \tfrac{1}{2} g_t\right) \odot h_{t-1}
      + g_t \odot \tilde{u}_t,
\qquad
\lambda = \sigma(\lambda_{\text{logit}}) \in (0,1)^D .
```

The scalar surprise signal is the mean gate response:

```math
s_t = \frac{1}{D}\sum_{j=1}^{D} g_{t,j} \in (0,1).
```

`e_t` is an absolute prediction error against the compressed state. `g_t` is a learned gate, not a calibrated entropy estimate; `s_t` is therefore a **surprise proxy**, not information-theoretic surprise. A larger `g_t` writes more of the current feature and decays the old state more strongly. The `1 - 0.5 g_t` factor keeps the decay in `(0.5, 1)`.

**State:** `h_t` is `D` floats per layer, independent of sequence length.

### 2.3 Engine II — causal shifted multi-head fast weights

This is the V3 core. A conventional fast-weight update at time `t` would bind the current key `K_t` to the current value `V_t`. MQAR pairs are adjacent: a key appears at `t-1`, its value at `t`. V3 therefore writes the **previous token's key** into the **current token's value**:

```math
Q_t = \operatorname{normalize}\!\left(W_q u_t\right), \qquad
K_t = \operatorname{normalize}\!\left(W_k u_t\right), \qquad
V_t = W_v c_t,
```

with `Q_t, K_t, V_t ∈ R^{H × d_head}`. For each head, the fast-weight matrix is `F_t ∈ R^{d_head × d_head}`, initialized as `F_0 = 0`. The shifted delta-rule write is

```math
F_t =
\lambda_f \odot F_{t-1}
+ \eta_{\text{eff},t} \odot
\left(V_t - F_{t-1} K_{t-1}\right) K_{t-1}^{\top},
\qquad t > 0,
```

```math
\eta_{\text{eff},t} = \eta \odot \left(0.3 + 0.7\, s_t\right),
\qquad
\eta = 0.9\,\sigma(\eta_{\text{logit}}),
\qquad
\lambda_f = \sigma(\lambda_{f,\text{logit}}).
```

The read uses the **current** query and the **updated** matrix:

```math
r_t = F_t Q_t, \qquad
z_t = h_t + W_o \operatorname{vec}(r_t).
```

At `t = 0` there is no previous key, so no write occurs; the read returns zero. This is why the first token has no associative memory. The implementation keeps `prev_k ∈ R^{H × d_head}` as the only additional recurrent state.

Why the shift matters: in a next-token MQAR stream, `(K_{t-1}, V_t)` is an adjacent key-value pair. Binding it at `t`, then querying with a later occurrence of the key, retrieves the value. Binding `(K_t, V_t)` instead stores self-pairs and cannot solve the task — this was the central V2 bottleneck. Q/K L2 normalization makes retrieval sharp; V is left unnormalized so values retain magnitude. The learning rate is surprise-modulated, but the write is a local delta rule, not gradient descent at inference time.

**State:** `F_t` is `H · d_head² = 4096` floats per layer, plus `prev_k` of `H · d_head = 128` floats. Both are independent of `L`.

### 2.4 Engine III — surprise-scaled equilibrium refinement

Engine III refines the memory output with a shared MLP. It runs a **fixed** number of steps, `L_max = 2`. Surprise changes the step size, not the number of steps:

```math
z_t^{(0)} = z_t,
```

```math
z_t^{(i+1)} = z_t^{(i)}
+ \frac{s_t}{i+1}\,
\operatorname{MLP}\!\left(\operatorname{LN}\!\left(z_t^{(i)}\right)\right),
\qquad i = 0, \ldots, L_{\max}-1.
```

The loop is vectorized across all positions `t`; only `L_max` sequential MLP calls occur per block, not `L` calls. The block output is

```math
y_t = x_t + \operatorname{LN}\!\left(z_t^{(L_{\max})}\right).
```

Engine III has no persistent state of its own. It is an optional refinement path, and the ablation below shows that it is **not** responsible for MQAR accuracy at this budget.

### 2.5 One-block summary

```text
x
├─ LN ── causal depthwise conv (k=3) ── SiLU ── W_u ── ũ
├─ Engine I:  e_t = |ũ_t - h_{t-1}|;  g_t = σ(W_s e_t)
│             h_t = λ ⊙ (1 - 0.5 g_t) ⊙ h_{t-1} + g_t ⊙ ũ_t
│             s_t = mean(g_t)
├─ Engine II: Q,K,V = projections of u / c;  Q,K are L2-normalized
│             F_t = λ_f ⊙ F_{t-1} + η_eff ⊙ (V_t - F_{t-1}K_{t-1})K_{t-1}ᵀ
│             r_t = F_t Q_t;  z_t = h_t + W_o r_t
├─ Engine III: z ← z + (s_t / (i+1)) · MLP(LN(z)),  i = 0..L_max-1
└─ y = x + LN(z)
```

---

## 3. What changed in V3

V3 is a targeted repair of the first benchmark's mathematical bottleneck, not a cosmetic refactor:

| Change | V2 behavior | V3 behavior | Why it matters |
| :--- | :--- | :--- | :--- |
| Fast-weight binding | `K_t -> V_t` self-binding | `K_{t-1} -> V_t` causal shifted binding | MQAR keys precede values; self-pairs cannot be retrieved by a later query |
| Head structure | Single fast-weight matrix | `H = 4` independent `d_head × d_head` matrices | Multiple associative subspaces; retrieval is less interference-prone |
| Projection placement | Linear layers inside the token loop | Q/K/V, `ũ`, and conv computed in parallel across `L` | Fewer Python-level operations; same math |
| Recall normalization | LayerNorm on recalled memory | No LayerNorm on the raw read | Avoids amplifying near-zero memory noise at early steps |
| Engine III placement | Per-token sequential MLP | Vectorized across `L`, fixed `L_max` loop | GPU-friendly refinement; surprise scales step size, not depth |

The core loop remains deliberately readable PyTorch. It has **no attention**, no softmax over sequence positions, and no explicit positional embedding. The only *sequential* work is the per-token recurrence; all projections and the Engine III refinement are parallel across `L`. The persistent state shapes do not depend on `L`.

---

## 4. Quick start

```bash
# 1. Install the three runtime dependencies
python -m pip install -r requirements.txt

# 2. Gradient, shape, and O(1) state sanity check
python tests/test_sanity.py

# 3. Full four-way benchmark (600 steps, batch 64)
python run_experiment.py
```

The experiment writes:

- `results/BENCHMARK_REPORT.md` — machine-generated tables;
- `results/tricore_vs_transformer.png` — loss, recall, and cache-scaling plots.

A CUDA device is used when available. The CPU fallback is functionally identical but slower. The reference environment is Python 3.13.5, PyTorch 2.7.1+cu118, NumPy 1.26.4, and Matplotlib 3.10.9.

Minimal API:

```python
import torch
from tricore import TriCoreLM, BaselineTransformerLM, generate_associative_recall_batch

model = TriCoreLM(
    vocab_size=64, d_model=128, n_layers=2, d_fast=32, max_loops=2,
    enable_fast_weight=True, enable_adaptive_loop=True,
)
x, y = generate_associative_recall_batch(batch_size=4, num_kv_pairs=8, num_queries=4)
logits, surprise = model(x)          # [B, L, 64], [B, L]
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, 64), y.view(-1), ignore_index=-100
)
loss.backward()
```

`enable_fast_weight=False` and `enable_adaptive_loop=False` are the two ablation switches. They gate compute, not parameter registration; all TriCore variants therefore report the same trainable parameter count.

---

## 5. Benchmark

### 5.1 MQAR accuracy and ablation

Task: Multi-Query Associative Recall with 8 key-value pairs, 4 queries, and 16 distractor tokens (40-token training sequences). The extrapolation column uses a **4× longer distractor gap** (64 noise tokens; 88-token sequences), not 4× total sequence length.

| Architecture variant | Parameters | Final val loss | In-domain recall | 4× noise-gap extrapolation | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (3-in-1)** | 348,432 | **0.0012** | **100.00%** | **99.45%** | 129.5 s |
| TriCore No Fast-Weight | 348,432 | 3.4330 | 3.28% | 3.83% | 60.6 s |
| TriCore No Adaptive Loop | 348,432 | **0.0012** | **100.00%** | **99.84%** | 126.0 s |
| Vanilla Transformer | 543,744 | 2.8272 | 13.12% | 13.05% | 16.4 s |

**Reading the table honestly.**

- **Fast weights are the decisive engine.** Removing them drops in-domain recall from 100.00% to 3.28% and extrapolation from 99.45% to 3.83%, while validation loss rises from 0.0012 to 3.4330. The task is solved by causal associative binding, not by the recurrent state alone.
- **Engine III is not responsible for this result.** Full and No-Adaptive-Loop both reach 100.00% in-domain and 0.0012 loss. Their extrapolation difference (99.45% vs 99.84%) is a single-seed fluctuation, not evidence that removing the loop helps. Engine III costs about 3.5 s (2.7%) in this run and has no measurable accuracy benefit on MQAR. It remains an independently ablatable research component, not a claimed accuracy driver.
- **The Transformer baseline is not a tuned ceiling.** It uses the same 600-step budget, but no learning-rate or architecture search. Its 13.12% / 13.05% result should be read as “this small, short-budget baseline did not solve MQAR,” not as a general statement about attention.
- **Parameter counts are not matched.** The Transformer's 543,744 parameters include a 262,144-parameter learned positional embedding (`2048 × 128`); without it, the baseline has 281,600 parameters, fewer than TriCore's 348,432. TriCore has no positional embedding table.
- **TriCore is slower in wall-clock time here.** Its recurrence is a Python per-token loop (129.5 s vs 16.4 s). The current value proposition is constant-size state, not throughput.

### 5.2 Inference cache scaling

The report uses per-sample FP32 state accounting. For TriCore with `D = 128`, `H = 4`, `d_head = 32`, and 2 layers:

```math
C_{\text{TriCore}} =
n_{\text{layers}}\left(D + H d_{\text{head}}^2\right) \times 4\text{ bytes}
= 2\left(128 + 4 \cdot 32^2\right) \times 4
= 33{,}792\text{ bytes} \approx 33.0\text{ KiB}.
```

The Transformer KV cache is

```math
C_{\text{KV}}(L) = n_{\text{layers}} \cdot 2 \cdot L \cdot D \cdot 4\text{ bytes}.
```

| Sequence length | Transformer KV cache | TriCore O(1) state | Reduction |
| :---: | :---: | :---: | :---: |
| 128 | 256.00 KB | 33.00 KB | **7.8× smaller** |
| 512 | 1,024.00 KB | 33.00 KB | **31.0× smaller** |
| 2,048 | 4,096.00 KB | 33.00 KB | **124.1× smaller** |
| 8,192 | 16,384.00 KB | 33.00 KB | **496.5× smaller** |
| 32,768 | 65,536.00 KB | 33.00 KB | **1,985.9× smaller** |

**Accounting note.** The benchmark formula counts `h` and `F`. The implementation also carries `prev_k` (`H · d_head = 128` floats per layer, 1 KiB total for two layers), so the strict persistent state footprint is about **34 KiB**. It remains constant in `L`. The table and plot use the report's 33.00 KB convention so that the figure and the generated markdown agree.

The O(1) claim is about **state size**, not total memory. Activations during training, framework allocator overhead, and temporary tensors are not included.

### 5.3 Plot

![TriCore V3 versus Vanilla Transformer: validation loss, recall accuracy, and cache scaling](results/tricore_vs_transformer.png)

The left and middle panels show validation loss and recall over 600 steps. The right panel is the log-log cache comparison: the Transformer KV cache grows linearly with `L`, while TriCore's state stays flat.

---

## 6. Repository layout

```text
tricore-arch/
├── tricore/
│   ├── __init__.py          # Public exports
│   ├── model.py             # TriCoreBlock, TriCoreLM, BaselineTransformerLM
│   └── tasks.py             # MQAR / associative-recall generator
├── tests/
│   └── test_sanity.py       # Forward, backward, shape, and gradient-flow check
├── results/
│   ├── BENCHMARK_REPORT.md
│   └── tricore_vs_transformer.png
├── run_experiment.py        # Four-way training, ablation, and cache profiler
├── requirements.txt
├── SPEC.md                  # Archived V1 blueprint (superseded by SPEC_V2.md)
├── SPEC_V2.md               # V3 architecture contract and benchmark record
└── README.md
```

The reference notebook-free implementation is intentionally small: `tricore/model.py` is the executable source of truth. `run_experiment.py` regenerates both result artifacts from scratch.

---

## 7. Experimental protocol

All four variants are trained from scratch with the same budget and seed. There is no per-variant hyperparameter search.

| Item | Value |
| :--- | :--- |
| Task | MQAR with 8 key-value pairs, 4 queries, 16 distractor tokens |
| Training sequence length | 40 tokens |
| Vocabulary | 64 tokens (keys, values, and one noise token) |
| Loss | Cross-entropy at query positions only (predict the value token that follows; `ignore_index=-100` elsewhere) |
| Steps / batch | 600 / 64 |
| Optimizer | AdamW, `lr=3e-3`, `weight_decay=1e-2` |
| Schedule | Cosine annealing over 600 steps |
| Gradient clipping | Global norm 1.0 |
| Seed | 42 for every variant |
| TriCore size | `d_model=128`, `n_layers=2`, `H=4`, `d_head=32`, `max_loops=2` |
| Transformer baseline | 2 layers, 4 heads, `d_model=128`, FFN 256, GELU, pre-norm, causal mask, learned positional embeddings (`max_seq_len=2048`) |
| In-domain evaluation | 5 batches of 32, `noise_len=16` |
| Extrapolation evaluation | 10 batches of 32, `noise_len=64` (4× noise gap) |
| Device | CUDA if available; CPU fallback |
| Reference environment | Python 3.13.5, PyTorch 2.7.1+cu118, NumPy 1.26.4, Matplotlib 3.10.9 |

The data generator constructs each batch on CPU and transfers it to the accelerator once per batch. This avoids thousands of per-element CUDA assignments; the generated token/target semantics are unchanged.

**Reproducibility boundary.** This is one seed and one 600-step budget. The full and no-adaptive runs both show a sharp transition between step 50 and step 100, consistent with a grokking-like phase change, but a single run cannot establish its stability. The next step is multi-seed reporting with mean and standard deviation.

---

## 8. Verification

The sanity test instantiates a small TriCore model, runs a forward pass, checks logits and surprise shapes, computes cross-entropy, and backpropagates through every parameter tensor:

```text
[PASS] Sanity check succeeded on cuda. Loss: 4.1582, Checked 50 parameter tensors.
```

The exact loss varies because the sanity test does not fix a seed. All Python files compile:

```text
COMPILE_OK
```

Reproduce with:

```bash
python tests/test_sanity.py
python -m py_compile tricore/model.py tricore/tasks.py tricore/__init__.py tests/test_sanity.py run_experiment.py
```

The sanity check validates gradient flow, not convergence. Convergence evidence is the benchmark table above.

---

## 9. Limitations

TriCore V3 is a research reference. The following caveats are part of the result, not footnotes to hide:

- **Synthetic task only.** MQAR measures associative recall under controlled noise. It is not natural-language modeling, reasoning, or instruction following.
- **Single seed, single budget.** All numbers come from seed 42 and 600 steps. No confidence intervals, no multi-seed mean ± std, and no convergence study. The 100.00% result should not be treated as a stable population estimate.
- **The extrapolation is a 4× noise gap, not 4× total length.** Training sequences are 40 tokens; extrapolation sequences are 88 tokens with a 64-token distractor gap. This tests memory persistence across distractors, not 32K-token streaming.
- **The Transformer baseline is small and untuned.** It uses the same step budget but no architecture or learning-rate search. Its low score is evidence about this baseline, not a general ceiling for attention.
- **Parameter and compute counts are not matched.** The Transformer has more total parameters because of its positional embedding table, but fewer non-position parameters. TriCore is also much slower in wall-clock time in this Python implementation.
- **Engine III has no measurable benefit here.** Full and No-Adaptive-Loop are indistinguishable on in-domain accuracy and loss; the small extrapolation difference is single-seed noise. Do not cite Engine III as the source of the MQAR result.
- **The O(1) cache is an inference state-size claim.** It excludes training activations (which are O(L) for backprop), allocator overhead, temporary tensors, and the `prev_k` term in the headline 33 KB figure. The strict persistent inference state is about 34 KiB.
- **No kernel optimization.** The per-token Python loop is intentionally transparent. It is not a throughput-optimized implementation, and the wall-clock comparison is not hardware-optimal for TriCore.
- **No scale claim.** These experiments do not establish parity with a fully trained Transformer at language-model scale. They validate an operator, its gradients, its ablation behavior, and its cache scaling.

---

## 10. Roadmap

The current benchmark is deliberately a starting point. The highest-value next steps are:

1. **Multi-seed evidence.** Run at least three seeds and report mean ± std for loss, recall, extrapolation, and wall-clock time. The present single-seed 100.00% should be treated as an existence result.
2. **Harder associative tasks.** Increase the number of key-value pairs, query count, vocabulary, and distractor length; test whether Engine III becomes useful when one delta-rule write is insufficient.
3. **Long-context streaming evaluation.** Measure decode latency and memory at 32K–128K tokens, where the O(1) state is expected to matter most, and compare against a real KV-cache implementation rather than a formula.
4. **Kernel-level efficiency.** Replace the Python per-token loop with a fused/chunked scan or a compiled kernel. The present wall-clock comparison is not hardware-optimal for TriCore.
5. **Matched baselines.** Tune the Transformer baseline and match non-position parameters, training tokens, and compute. A fair comparison needs more than an identical step count.
6. **Natural-language pretraining.** Test whether causal shifted fast weights and surprise-gated state collapse provide any transfer advantage beyond synthetic associative recall.

## 11. Citation

If you use TriCore in your research, please cite the architecture specification:

```bibtex
@misc{tricore2026,
  title        = {TriCore: Surprise-Gated State Collapse, Causal Shifted Fast-Weight
                  Plasticity, and Surprise-Scaled Equilibrium Refinement},
  author       = {The TriCore Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/tricore-arch/tricore}},
  note         = {Open-source reference implementation, V3}
}
```

## 12. Contributing

Issues and pull requests are welcome. Please preserve the following invariants:

1. `TriCoreBlock` must remain attention-free and must keep an inference state whose shape is independent of sequence length.
2. The causal shift in Engine II must remain: the write binds `K_{t-1}` to `V_t`, and the read uses `Q_t` after the write.
3. Every engine must remain independently ablatable through `enable_fast_weight` and `enable_adaptive_loop`.
4. Numerical changes to the equations require a gradient-stability test and a regenerated benchmark report.
5. Benchmark claims must be reproducible with `python run_experiment.py`, and must state the seed, budget, task, and parameter counts.
6. Do not describe Engine III as dynamic depth. It is a fixed `L_max` loop; surprise scales the step size.

---

<p align="center">
  <i>TriCore V3 — remember in constant state, write what surprises you.</i>
</p>
