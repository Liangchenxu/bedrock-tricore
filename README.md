# TriCore V6

**A JIT-recurrent, attention-free sequence operator with a 17 KiB constant-size inference state, vectorized multi-hop synaptic chaining, and a causal bytecode pointer task.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TorchScript](https://img.shields.io/badge/TorchScript-JIT%20fused-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/jit.html)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-sanity%20%2B%20V6%20equivalence-brightgreen)](#8-verification)

> **Status:** experimental research reference, not a production language model. V6 keeps the V4/V5 equations, separates the O(1) recurrent state from a fully vectorized multi-hop pass, and fixes Track 2 to a causal bytecode format with the dereference target at `pos + 1`. All reported numbers are single-seed runs. Read [Limitations](#9-limitations) before generalizing.

> **TL;DR (Track 1, 2-Hop MQAR).** Full TriCore reaches **100.00% in-domain recall** and **100.00% recall under a 4x longer adversarial distractor gap** with a **17.00 KiB O(1) state**. Removing Engine I gives **5.55% / 4.14%**; removing Engine II gives **5.78% / 3.98%**; removing Engine III gives **16.48% / 17.50%**. A 2-layer Transformer reaches **16.72% / 14.92%**, which is chance at this budget. The V6 Full run takes **80.0 s** for 500 steps versus **136.0 s** for V4 (**1.70x** faster) and **125.3 s** for V5 (**1.57x** faster), excluding a 0.4 s warm-up.

> **TL;DR (Track 2, Causal Bytecode LM).** TriCore (1 layer) reaches validation **PPL 3.17** and **100.00% dereference accuracy** on the synthetic `alias -> var -> lit` return task; the 2-layer Transformer reaches **PPL 3.56** and **19.63%**. The 6-way chance level is **16.67%**. The near-perfect score is on a closed synthetic bytecode grammar, not evidence of general code understanding.

> **中文速览.** TriCore V6 保持 V4/V5 的全部数学公式，把 O(1) 状态演化保留在 `@torch.jit.script` 循环中，并把多跳重查询改为跨全部时间步的批量 5-D `matmul`；第二赛道改为因果字节码，解引用目标位于返回别名的 `pos + 1`。Track 1 中 Full 达到 **100.00% 域内召回 / 100.00% 四倍噪声间隔外推**，单核推理状态恒为 **17.00 KB**；去掉 Engine I 为 5.55%/4.14%，去掉 Engine II 为 5.78%/3.98%，去掉 Engine III 为 16.48%/17.50%，2 层 Transformer 为 16.72%/14.92%。Full 训练 **80.0 s**，对比 V4 Python 循环 **136.0 s** 加速 **1.70×**，对比 V5 **125.3 s** 加速 **1.57×**。Track 2 中 TriCore 验证 PPL **3.17**、解引用准确率 **100.00%**；Transformer 为 **3.56 / 19.63%**，6 类随机基线为 16.67%。结论仅限单 seed 合成任务，不代表自然语言或真实代码能力。

---

## Contents

- [1. The problem](#1-the-problem)
- [2. Architecture](#2-architecture)
- [3. What changed in V6](#3-what-changed-in-v6)
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

| Goal | V6 mechanism | Evidence |
| :--- | :--- | :--- |
| No attention over positions | Only the hop router uses softmax, over `max_loops + 1` slots | Five-way Track 1 table |
| O(1) inference state | `h`, `F_mat`, `prev_k` | 17.00 KiB per sample; 3,855.1x smaller than a 2-layer KV cache at 32K |
| Multi-hop with one core | Latent re-query of the same fast weights, vectorized over all positions | Full versus No-Engine-III ablation |
| Selective writes | Learned surprise-gated write protection | Full versus No-Engine-I at this seed |
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

The recurrent state itself is `h [B,D]`, `F_mat [B,H,d_head,d_head]`, and `prev_k [B,H,d_head,1]`. To vectorize the refinement, the scripted scan also returns per-position `F` snapshots as a training-time `F_seq [B,L,H,d_head,d_head]` tensor. That tensor is not a persistent inference cache: the deployed inference state remains O(1) in `L`, and no KV-style cache is carried across decoding steps. The scripted loop is compiled once and returns:

- `h_seq: [B, L, D]`
- `r1_seq: [B, L, H, d_head, 1]` — hop-1 reads (zeros when `enable_fast_weight=False`)
- `F_seq: [B, L, H, d_head, d_head]` — per-position fast-weight snapshots
- `surprise_trace: [B, L]`

`TriCoreBlock` then loops only over the fixed `max_loops` budget, not over tokens. Each refinement step normalizes the previous read, projects it into query space, and applies `torch.matmul(F_seq, q_next)` to all `L` positions at once. The Python-level loop count is therefore `max_loops`, while the work inside each hop is a batched 5-D matrix product.

The JIT path is mathematically equivalent to the V4/V5 Python reference. `tests/test_v5.py` compares the scripted function against an explicit Python reference for all-engines, each single-engine ablation, and all-disabled, asserting `allclose` on all four outputs.

This is **not** a custom CUDA kernel and it is **not** a parallel prefix scan. The recurrent state update remains serial; only the multi-hop refinement is vectorized across sequence positions. It still pays kernel-launch and small-matrix costs, so the measured speedup is a runtime cleanup rather than a hardware-optimal scan.

---

## 3. What changed in V6

| Area | V5 | V6 |
| :--- | :--- | :--- |
| Recurrence | One serial TorchScript loop for state evolution and every hop | JIT loop for the O(1) causal state, then a fully batched multi-hop pass over all `L` positions |
| Engine III | Per-token re-query loop inside the scripted scan | `F_seq: [B, L, H, d_head, d_head]` snapshots re-queried with batched 5-D `matmul`; Python loops only over fixed hops |
| Track 2 task | `EQ`-based statements that collided in the key/value hash | Causal bytecode `[LET, alias, var, SEMI]`, `[ALIAS, var, lit, SEMI]`, `[NOP, var, SEMI, SEMI]`, `[RET, alias, lit, SEMI]` |
| Dereference target | Predicted at the literal position | Predicted at `pos + 1` (the return alias position), the first causal position where the pointer value is required |
| Public math | V4 equations | Identical equations, state shapes, and parameter count |
| Evidence | Two synthetic tracks | Two synthetic tracks with V6 timings and causal-pointer accuracy |
| Tests | JIT equivalence on three outputs | Four-output equivalence (`h`, hop-1 read, `F` snapshots, surprise) plus causal bytecode backward |

The model equations, parameter count (`207,503`), and O(1) inference-state accounting are unchanged. V6 is a runtime and evidence upgrade, not a new operator.

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

# Tests
python -m pytest -q
```

`run_experiment.py` writes:

- `results/BENCHMARK_REPORT.md` — generated tables and protocol notes
- `results/tricore_vs_transformer.png` — 2x2 panel: Track 1 learning curves, Track 1 ablation bars, Track 2 PPL, Track 2 dereference accuracy

The device is selected automatically: CUDA if available, otherwise CPU. The script uses AdamW, cosine decay, gradient clipping at 1.0, and a fixed evaluation set. Default seeds are 42 for training and 2026/2027 (Track 1) / 3030 (Track 2) for evaluation batches.

```python
from tricore import (
    TriCoreBlock, TriCoreLM, BaselineTransformerLM,
    generate_associative_recall_batch, generate_contextual_code_lm_batch,
)

x, y = generate_associative_recall_batch(batch_size=4, device="cpu")
model = TriCoreLM(vocab_size=64, d_model=128, n_layers=1, d_fast=32, max_loops=2)
logits, surprise = model(x)          # [4, 60, 64], [4, 60]

code_x, code_y, deref_mask = generate_contextual_code_lm_batch(
    batch_size=4, num_vars=6, num_noise=6, device="cpu"
)
code_logits, code_surprise = model(code_x)  # [4, 82, 64], [4, 82]
```

---

## 5. Benchmark

All numbers below are generated by `python run_experiment.py` on the reference machine and copied from `results/BENCHMARK_REPORT.md`. They are single-seed results.

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
| 32,768 | 65,536.00 KB | 17.00 KB | **3,855.1x** |

`C_TriCore = (D + H*d_head^2 + H*d_head) * 4 / 1024` KiB includes `h`, `F_mat`, and `prev_k` and is independent of `L`. The Transformer figure is `n_layers * 2 * L * D * 4 / 1024` KiB. This is an inference-state accounting comparison, not a measured decode benchmark.

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
│   └── test_v5.py           # JIT-vs-Python equivalence + code-task backward
├── results/
│   ├── BENCHMARK_REPORT.md  # generated by run_experiment.py
│   └── tricore_vs_transformer.png
├── run_experiment.py        # dual-track training, ablation, cache, report, plot
├── requirements.txt         # torch, numpy, matplotlib
├── pyproject.toml           # installable tricore-arch 0.6.0, MIT
├── LICENSE                  # MIT
├── .gitignore               # Python/cache/venv; keeps report and PNGs
├── SPEC_V2.md               # V6 architecture + contracts
└── README.md
```

### 6.1 Installable package

`pyproject.toml` declares `tricore-arch==0.6.0`, Python `>=3.10`, and the runtime dependencies `torch>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`. The package name remains `tricore-arch`; the import package is `tricore`.

```bash
python -m pip install -e .
python -c "import tricore; print(tricore.__file__)"
```

### 6.2 Generated artifacts

`results/BENCHMARK_REPORT.md` is the machine-generated source of truth for the latest numbers; this README and `SPEC_V2.md` summarize it. The plot is a 2x2 panel. Do not hand-edit the generated report and then claim it came from a run; rerun the script.

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

### 7.3 Timing

- V6 timing includes data generation, forward, backward, and optimizer steps for 500 Full steps.
- The first scripted call is a warm-up and is excluded from the timed loop.
- V4's 136.0 s and V5's 125.3 s are previously recorded reference runs, not simultaneous measurements.
- Timings are single-machine wall-clock values and vary with GPU, driver, PyTorch version, and thermal state.

### 7.4 What is not controlled

No multi-seed confidence interval, no hyperparameter search for the Transformer, no parameter- or FLOP-matching, no real held-out corpus, and no decode-latency benchmark. The Track 2 task is synthetic and small. Treat the numbers as an existence demonstration and a pipeline check.

---

## 8. Verification

### 8.1 JIT equivalence

`tests/test_v5.py` builds random tensors for `h`, `F`, `prev_k`, and projections, then runs `tricore_fused_scan` and an explicit Python reference over the four combinations of its two scan flags:

- `enable_surprise_gate=True, enable_fast_weight=True`
- `enable_surprise_gate=False, enable_fast_weight=True` (No Engine I)
- `enable_surprise_gate=True, enable_fast_weight=False` (No Engine II)
- both flags `False`

Engine III is outside the scripted scan and is covered by the `TriCoreLM` forward/backward test; the causal bytecode task has its own shape, mask, and backward test.

All four outputs (`h_seq`, `r1_seq`, `F_seq`, `surprise_trace`) must satisfy `torch.allclose(..., atol=1e-6, rtol=1e-5)`. The test also checks the causal bytecode shapes, the dereference mask at `pos + 1`, and a backward pass with finite gradients.

Run:

```bash
python -m pytest -q
python tests/test_v5.py
```

### 8.2 Full run

The reference run completed on CUDA and regenerated:

- `results/BENCHMARK_REPORT.md`
- `results/tricore_vs_transformer.png`

The report contains the JIT timing, the five-way Track 1 table, the Track 2 table, and the cache table. If the code and this README disagree, `results/BENCHMARK_REPORT.md` and the code win.

### 8.3 Smoke run

```bash
python run_experiment.py --steps 2 --code-steps 2
```

This is a pipeline check, not a result. It compiles the scripted scan, runs both task generators, exercises all five Track 1 variants and both Track 2 models, writes the report, and saves the plot.

---

## 9. Limitations

- **Single seed.** Every number is one run. No confidence intervals, no seed variance, no significance test. The V6 No-Engine-I result (`5.55% / 4.14%`) differs substantially from the V5 record (`91.88% / 84.14%`); the equations are the same, but the vectorized hop path changes the optimization trajectory and fixed evaluation batches are used.
- **Synthetic tasks only.** Track 1 is a controlled associative-recall construction. Track 2 is a tiny imperative grammar with disjoint namespaces. Neither is natural language or real code.
- **Track 2 is a closed synthetic grammar.** V6 reaches `100.00%` dereference on a deterministic bytecode task with fixed namespaces and no held-out templates. That shows the causal two-level lookup is learnable, not that the model understands real code. The PPL gap over the Transformer (`3.17` vs `3.56`) is also single-seed.
- **The V6 pass is not a parallel scan.** The state recurrence is still serial. V6 vectorizes only the fixed-hop refinement over `L`; it still launches many small CUDA kernels. The measured speedup is a runtime improvement on one machine, not a hardware-optimal kernel.
- **Transformer baseline is untuned.** It is not parameter-matched (`543,744` vs `207,503`), not compute-matched, and not tuned. Its chance-level Track 1 result is not an attention ceiling.
- **Training memory is not O(1).** V6 materializes `F_seq: [B, L, H, d_head, d_head]` for the vectorized hop pass. The 17.00 KiB O(1) claim applies to persistent inference state, not training activations; `F_seq` scales with `L`.
- **No long-context decode measurement.** The 3,855.1x cache reduction at 32K is arithmetic from state shapes, not a measured serving benchmark.
- **No 3-5 hop or natural-language evidence.** The router and memory are only tested on 2-hop synthetic chains and the 80-token causal bytecode grammar.
- **Task-specific ablations.** Engine III was redundant on the V3 single-hop task and is necessary here; Engine I's necessity depends on the noise pattern and seed. Conclusions are conditional.
- **No production hardening.** No checkpoint format, distributed training, mixed precision, quantization, serving API, or security review.

---

## 10. Roadmap

1. **Multi-seed evidence.** At least three seeds with mean ± std for loss, recall, extrapolation, and wall-clock time.
2. **More hops.** 3-5 hop chains, more bridge tokens, and variable `max_loops`; measure whether the router uses or downweights extra re-queries.
3. **Stronger adversarial writes.** Deletion, overwrite, repeated distractors, and higher noise density.
4. **Long-context streaming.** Measure decode latency and memory at 32K-128K tokens against a real KV-cache implementation.
5. **Kernel-level efficiency.** Replace the scripted state loop with a fused/chunked scan or a compiled CUDA/Triton kernel; compare against the V6 JIT plus vectorized-hop baseline.
6. **Matched baselines.** Tune the Transformer and match non-position parameters, tokens, and compute.
7. **Harder code task.** More variables, longer schedules, held-out program templates, and a real code corpus.
8. **Natural-language pretraining.** Test whether surprise-gated writes and latent re-query transfer beyond synthetic tasks.

---

## 11. Citation

```bibtex
@misc{tricore2026v6,
  title        = {TriCore V6: Vectorized Latent Multi-Hop Chaining,
                  Causal Fast Weights, and a Causal Bytecode Pointer Task},
  author       = {The TriCore Authors},
  year         = {2026},
  howpublished = {Open-source reference implementation},
  note         = {Version 6; MIT license; single-seed synthetic benchmarks}
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
14. **Reproducible artifacts.** `python run_experiment.py` must regenerate the report and plot, and state accounting must include `prev_k`.

---

<p align="center">
  <i>TriCore V6 — keep a constant state, protect what matters, vectorize the latent loop, and force the pointer to be dereferenced.</i>
</p>
