# TriCore V5

**A JIT-fused, attention-free sequence operator with a 17 KiB constant-size inference state, latent multi-hop synaptic chaining, and a dual-track evidence package.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TorchScript](https://img.shields.io/badge/TorchScript-JIT%20fused-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/jit.html)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-sanity%20%2B%20JIT%20equivalence-brightgreen)](#8-verification)

> **Status:** experimental research reference, not a production language model. V5 keeps the V4 equations and adds a TorchScript-JIT recurrent kernel plus a second synthetic track. All reported numbers are single-seed runs. Read [Limitations](#9-limitations) before generalizing.

> **TL;DR (Track 1, 2-Hop MQAR).** Full TriCore reaches **100.00% in-domain recall** and **100.00% recall under a 4x longer adversarial distractor gap** with a **17.00 KiB O(1) state**. Removing Engine I gives **91.88% / 84.14%**; removing Engine II gives **5.78% / 4.14%**; removing Engine III gives **16.48% / 16.56%**. A 2-layer Transformer reaches **16.72% / 15.23%**, which is chance at this budget. The JIT-fused Full run takes **125.3 s** for 500 steps versus **136.0 s** for the V4 Python loop (**1.08x** faster), excluding a 0.5 s warm-up.

> **TL;DR (Track 2, Contextual Code LM).** TriCore (1 layer) reaches validation **PPL 1.81** and **21.09% dereference accuracy**; the 2-layer Transformer reaches **PPL 1.96** and **18.75%**. The 6-way chance level is **16.67%**. This is a small edge on a synthetic grammar, not evidence of general code understanding.

> **中文速览.** TriCore V5 保持 V4 的全部数学公式，把逐 token 的 Python 循环替换为 `@torch.jit.script` 融合扫描，并新增“上下文代码语言建模”第二赛道。Track 1 中 Full 达到 **100.00% 域内召回 / 100.00% 四倍噪声间隔外推**，单核推理状态恒为 **17.00 KB**；去掉 Engine I 为 91.88%/84.14%，去掉 Engine II 为 5.78%/4.14%，去掉 Engine III 为 16.48%/16.56%，2 层 Transformer 为 16.72%/15.23%。Full 训练 **125.3 s**，对比 V4 Python 循环 **136.0 s**，JIT 加速 **1.08×**。Track 2 中 TriCore 验证 PPL **1.81**、解引用准确率 **21.09%**；Transformer 为 **1.96 / 18.75%**，6 类随机基线为 16.67%。结论仅限单 seed 合成任务，不代表自然语言或真实代码能力。

---

## Contents

- [1. The problem](#1-the-problem)
- [2. Architecture](#2-architecture)
- [3. What changed in V5](#3-what-changed-in-v5)
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

V5 also asks a practical question that V4 could not answer:

> **Does replacing the Python per-token recurrence with a TorchScript-JIT scan remove enough host overhead to make the operator competitive in wall-clock training?**

The answers are deliberately narrow. On a controlled 2-hop task, a causal shifted delta-rule memory plus a latent re-query loop solves the task with an O(1) inference state. The JIT scan is **1.08x** faster than the V4 Python loop on the reference machine, but it is still a compiled loop of tensor operations, not a custom CUDA kernel, and it remains much slower than a 2-layer Transformer on these short sequences. The second track is a synthetic code-like grammar, not human code.

### Design goals

| Goal | V5 mechanism | Evidence |
| :--- | :--- | :--- |
| No attention over positions | Only the hop router uses softmax, over `max_loops + 1` slots | Five-way Track 1 table |
| O(1) inference state | `h`, `F_mat`, `prev_k` | 17.00 KiB per sample; 3,855.1x smaller than a 2-layer KV cache at 32K |
| Multi-hop with one core | Latent re-query of the same fast weights | Full 100% versus 16.48% without Engine III |
| Selective writes | Learned surprise-gated write protection | Full 100% versus 91.88% without Engine I at this seed |
| Faster recurrence | TorchScript-JIT fused scan | 125.3 s versus 136.0 s (V4) |
| Second task | Contextual Code LM with alias dereference | PPL 1.81, deref 21.09% (chance 16.67%) |
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

### 2.6 TorchScript-JIT fused scan

V5 moves the entire recurrent scan into one scripted function:

```python
@torch.jit.script
def tricore_fused_scan(
    u_feat, u_surp_bias, w_gate_logits, hop_weights, Q, K, V,
    base_decay, fast_decay, fast_lr, hop_proj,
    enable_surprise_gate, enable_fast_weight, enable_adaptive_loop, max_loops,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

The function owns the exact state `h [B,D]`, `F_mat [B,H,d_head,d_head]`, and `prev_k [B,H,d_head,1]`; it never materializes a `[B,L,...]` fast-weight cache. The per-token loop is compiled once, removes Python dispatch from the hot path, and returns:

- `h_seq: [B, L, D]`
- `mem_seq: [B, L, H*d_head]` (zeros when `enable_fast_weight=False`)
- `surprise_trace: [B, L]`

`TriCoreBlock` precomputes projections and calls the scan with keyword arguments. The JIT path is mathematically equivalent to the V4 Python loop. `tests/test_v5.py` compares the scripted function against an explicit Python reference for all-engines, each single-engine ablation, and all-disabled, asserting `allclose` on all three outputs.

This is **not** a custom CUDA kernel. It is a compiled graph of PyTorch tensor operations, so it still pays kernel-launch and small-matrix costs. It is a runtime cleanup, not a hardware-optimal scan.

---

## 3. What changed in V5

| Area | V4 | V5 |
| :--- | :--- | :--- |
| Recurrence | Python per-token loop in `forward` | Single `@torch.jit.script tricore_fused_scan` |
| Public math | Reference equations | Identical equations and state shapes |
| Tasks | 2-Hop MQAR only | 2-Hop MQAR + Contextual Code LM |
| Evidence | One 500-step track | Two tracks with separate protocols |
| Speed measurement | Not isolated | V4 136.0 s vs V5 125.3 s Full, 1.08x |
| Tests | Shape/gradient sanity | Plus JIT-vs-Python equivalence and code-task backward |
| Packaging | README/spec only | MIT `LICENSE`, `.gitignore`, `pyproject.toml` |
| Report | Single-track report | `results/BENCHMARK_REPORT.md` with JIT, both tracks, cache |

The model equations, parameter count (`207,503`), and O(1) state accounting are unchanged. V5 is a runtime and evidence upgrade, not a new operator.

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

### 5.1 JIT timing versus V4

| Measurement | Time | Note |
| :--- | :---: | :--- |
| V4 Full, Python per-token loop | 136.0 s | reference run |
| V5 Full, TorchScript fused scan | 125.3 s | timed 500 steps |
| V5 / V4 speed ratio | **1.08x** | >1 means faster; <1 means slower |
| V5 first-call warm-up | 0.5 s | excluded from timed training |

Timed training includes data generation, forward, backward, and optimizer steps. Warm-up is excluded. The ratio is a single-seed wall-clock measurement on one machine, not a throughput guarantee.

### 5.2 Track 1: 2-Hop Chained Associative Recall

500 steps, batch 64, `lr=3e-3`, seed 42; 6 chains, 4 queries, 16 training noise tokens. Fixed evaluation batches use seeds 2026/2027. The 4x extrapolation multiplies the distractor gap from 16 to 64 tokens, not the total sequence length.

| Architecture variant | Parameters | Final val loss | In-domain 2-hop recall | 4x noise-gap extrapolation | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (3-in-1)** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 125.3 s |
| TriCore No Engine I: Surprise Gate | 207,503 | 0.4066 | 91.88% | 84.14% | 97.9 s |
| TriCore No Engine II: Fast-Weight | 207,503 | 3.0008 | 5.78% | 4.14% | 29.2 s |
| TriCore No Engine III: Adaptive Loop | 207,503 | 1.8660 | 16.48% | 16.56% | 59.7 s |
| Vanilla Transformer (2-layer) | 543,744 | 2.3131 | 16.72% | 15.23% | 15.9 s |

### 5.3 Track 1 reading

1. **Full TriCore solves the task.** It reaches `100.00%` in-domain and `100.00%` at the 4x gap; validation loss is `0.0010`.
2. **Engine II is the associative memory.** Without fast weights, recall is `5.78% / 4.14%`, below the 6-way chance level of `16.67%`.
3. **Engine III is the second hop.** Without the latent re-query, recall is `16.48% / 16.56%`, essentially chance.
4. **Engine I contributes but is not the whole story at this seed.** Removing the write gate gives `91.88% / 84.14%`, a drop of `8.12 / 15.86` points from Full, but not the collapse seen in the V4 record. The result is seed- and protocol-sensitive; the V4 report's `63.44% / 51.41%` is not reproduced.
5. **The Transformer is at chance.** `16.72% / 15.23%` at this budget; it is not tuned, parameter-matched, or compute-matched. This is not evidence of an attention ceiling.
6. **JIT helps modestly.** Full is `125.3 s` versus `136.0 s` for the V4 Python loop, a `1.08x` speedup. The Transformer is still much faster (`15.9 s`) because TriCore executes a sequential recurrence and many small matrix products.

### 5.4 Track 2: Contextual Code LM

300 steps, batch 32, `lr=3e-3`, seed 42; 6 variables, 6 NOP statements, sequence length 82. Dense next-token cross-entropy; dereference accuracy is measured only at the return-alias position. Fixed evaluation batches use seed 3030.

| Model | Parameters | Validation loss | Validation PPL | Dereference accuracy | Train time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full (1-layer)** | 207,503 | 0.5914 | **1.81** | **21.09%** | 83.4 s |
| Vanilla Transformer (2-layer) | 543,744 | 0.6753 | 1.96 | 18.75% | 2.9 s |

The task is a synthetic imperative grammar. Each sample contains, in random statement order:

```text
[LET,   var_i,   EQ, lit_i,  SEMI]
[ALIAS, alias_i, EQ, var_i,  SEMI]
[NOP,   NOISE,   SEMI]
...
[RET,   alias_i, lit_i,  SEMI]
```

Variable, alias, and literal namespaces are disjoint. The return uses an alias, so predicting `lit_i` at the alias position requires resolving `alias_i -> var_i -> lit_i` rather than copying a surface token. There are 6 equally likely literals, so chance is `16.67%`.

**Reading.** TriCore has a small edge on both PPL and dereference accuracy, but `21.09%` is only `4.42` points above chance and the run is single-seed. This track validates the pipeline and shows a non-degenerate signal; it does **not** establish general code understanding, compositional generalization, or a Transformer-beating result. The Transformer is also near chance. A longer schedule, more variables, or real code would be a different experiment.

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
├── pyproject.toml           # installable tricore-arch 0.5.0, MIT
├── LICENSE                  # MIT
├── .gitignore               # Python/cache/venv; keeps report and PNGs
├── SPEC_V2.md               # V5 architecture + contracts
└── README.md
```

### 6.1 Installable package

`pyproject.toml` declares `tricore-arch==0.5.0`, Python `>=3.10`, and the runtime dependencies `torch>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`. The package name remains `tricore-arch`; the import package is `tricore`.

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

### 7.2 Track 2 — Contextual Code LM

- 6 variables, 6 NOP statements; sequence length `6*(2*5) + 6*3 + 4 = 82`.
- 300 optimizer steps, batch 32, AdamW, `lr=3e-3`, weight decay `0.01`, cosine decay, gradient clipping `1.0`.
- Training seed `42`; fixed evaluation batches use seed `3030`, 8 batches of 32.
- Dense next-token cross-entropy over all non-`-100` positions.
- Dereference accuracy is measured at the single return-alias position, where the target is one of 6 literals (`16.67%` chance).
- TriCore uses one layer (`d_model=128`, `d_fast=32`, `max_loops=2`); the baseline uses two Transformer layers.

### 7.3 Timing

- V5 timing includes data generation, forward, backward, and optimizer steps for 500 Full steps.
- The first scripted call is a warm-up and is excluded; the reported warm-up is 0.5 s.
- V4's 136.0 s is the previously recorded Python-loop Full run, not a simultaneous measurement.
- Timings are single-machine wall-clock values and vary with GPU, driver, PyTorch version, and thermal state.

### 7.4 What is not controlled

No multi-seed confidence interval, no hyperparameter search for the Transformer, no parameter- or FLOP-matching, no real held-out corpus, and no decode-latency benchmark. The Track 2 task is synthetic and small. Treat the numbers as an existence demonstration and a pipeline check.

---

## 8. Verification

### 8.1 JIT equivalence

`tests/test_v5.py` builds random tensors for `h`, `F`, `prev_k`, projections, and hop parameters, then runs `tricore_fused_scan` and an explicit Python reference over five flag combinations:

- all three engines enabled
- No Engine I
- No Engine II
- No Engine III
- all engines disabled

All three outputs (`h_seq`, `mem_seq`, `surprise_trace`) must satisfy `torch.allclose(..., atol=1e-6, rtol=1e-5)`. The test also checks the code-task shapes, the dereference mask, and a backward pass with finite gradients.

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

- **Single seed.** Every number is one run. No confidence intervals, no seed variance, no significance test. The V5 No-Engine-I result (`91.88% / 84.14%`) differs substantially from the V4 record (`63.44% / 51.41%`); the equations are the same, but optimization and fixed evaluation batches differ.
- **Synthetic tasks only.** Track 1 is a controlled associative-recall construction. Track 2 is a tiny imperative grammar with disjoint namespaces. Neither is natural language or real code.
- **Near-chance Track 2.** `21.09%` dereference is only `4.42` points above chance. PPL is lower than the baseline, but the task and budget are small. Do not describe this as code understanding.
- **JIT is not a kernel.** The speed ratio is `1.08x` on one machine. TorchScript removes Python dispatch but still launches many small CUDA kernels; a custom fused/chunked kernel is future work.
- **Transformer baseline is untuned.** It is not parameter-matched (`543,744` vs `207,503`), not compute-matched, and not tuned. Its chance-level Track 1 result is not an attention ceiling.
- **No long-context decode measurement.** The 3,855.1x cache reduction at 32K is arithmetic from state shapes, not a measured serving benchmark.
- **No 3-5 hop or natural-language evidence.** The router and memory are only tested on 2-hop synthetic chains and the 82-token code grammar.
- **Task-specific ablations.** Engine III was redundant on the V3 single-hop task and is necessary here; Engine I's necessity depends on the noise pattern and seed. Conclusions are conditional.
- **No production hardening.** No checkpoint format, distributed training, mixed precision, quantization, serving API, or security review.

---

## 10. Roadmap

1. **Multi-seed evidence.** At least three seeds with mean ± std for loss, recall, extrapolation, and wall-clock time.
2. **More hops.** 3-5 hop chains, more bridge tokens, and variable `max_loops`; measure whether the router uses or downweights extra re-queries.
3. **Stronger adversarial writes.** Deletion, overwrite, repeated distractors, and higher noise density.
4. **Long-context streaming.** Measure decode latency and memory at 32K-128K tokens against a real KV-cache implementation.
5. **Kernel-level efficiency.** Replace the scripted loop with a fused/chunked scan or a compiled CUDA/Triton kernel; compare against the V5 JIT baseline.
6. **Matched baselines.** Tune the Transformer and match non-position parameters, tokens, and compute.
7. **Harder code task.** More variables, longer schedules, held-out program templates, and a real code corpus.
8. **Natural-language pretraining.** Test whether surprise-gated writes and latent re-query transfer beyond synthetic tasks.

---

## 11. Citation

```bibtex
@misc{tricore2026v5,
  title        = {TriCore V5: JIT-Fused Surprise-Gated Write Protection,
                  Causal Fast Weights, and Latent Multi-Hop Synaptic Chaining},
  author       = {The TriCore Authors},
  year         = {2026},
  howpublished = {Open-source reference implementation},
  note         = {Version 5; MIT license; single-seed synthetic benchmarks}
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
10. **JIT equivalence.** Any change to `tricore_fused_scan` must pass `tests/test_v5.py` against the Python reference.
11. **Fixed loop count.** Do not describe Engine III as adaptive computation time; `max_loops` is a constant.
12. **Reproducible artifacts.** `python run_experiment.py` must regenerate the report and plot, and state accounting must include `prev_k`.

---

<p align="center">
  <i>TriCore V5 — keep a constant state, protect what matters, compile the loop, and chase the latent pointer twice.</i>
</p>
