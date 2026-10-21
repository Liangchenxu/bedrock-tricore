
## V10 (v1.3.0) Track 5: Infix Code Assignment with `=` (`EQ`) & 100% Fair Engine I Ablation

- **Self-Audit Fix 1 (Infix `EQ` Bridging):** Restores `[LET, alias_i, EQ, var_i, SEMI]` and `[SET, var_i, EQ, lit_i, SEMI]` with an explicit `=` (`EQ`) token between keys and values. Synaptic Eligibility Trace (`prev_k = normalize((1 - elig_gate) * prev_k + elig_gate * k_t)`) latches entity keys across low-surprise syntax connectors.

- **Self-Audit Fix 2 (100% Fair No-Engine-I Ablation):** Removed all artificial decay penalties (`0.82`) and projection handicaps. `Fair No-Engine-I` shares the exact same learned `fast_decay`, `fast_lr`, and residual projections as `Full`, only fixing `write_gate = 1.0` and `elig_gate = 1.0`.


| Architecture Variant | Parameters | Val Loss | Infix `EQ` 2-Hop Deref Acc (%) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **TriCore V10 Full (With Eligibility Trace)** | 60,819 | 1.7359 | **42.24%** | 103.5s |
| **TriCore V10 Fair No-Engine-I (Unmodulated Gate)** | 60,819 | 1.6614 | **45.80%** | 93.8s |
| **Vanilla Transformer (2-Layer)** | 206,336 | 1.6690 | **32.03%** | 14.4s |

---
# TriCore V10 — Architecture Specification and Benchmark Record

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


**Project:** `tricore-arch` 1.2.0  
**Operator:** attention-free, constant-size recurrent state, causal shifted fast-weight binding, fused latent multi-hop re-query  
**Runtime:** `@torch.jit.script tricore_fused_scan` for the O(1) state and zero-`F_seq` multi-hop training, plus native `init_state` / `forward_step` per-token decoding  
**Status:** research reference; V6/V7 records plus a 4-chain, five-way, 3-seed V9 verification and 32K CUDA profiling  
**Executable source of truth:** `tricore/model.py`  
**Benchmark source of truth:** `results/BENCHMARK_REPORT.md`

> **Document provenance.** This specification describes V9. The equations, shapes, flags, state accounting, and streaming API mirror `tricore/model.py`, `tricore/tasks.py`, `run_experiment.py`, and `run_v9_verification.py`. V9 is additive to V6/V7 at the API level, preserves every parameter name, and removes the training-only `F_seq` snapshot tensor by fusing Engine III into the JIT scan. The benchmark numbers are artifacts generated on the reference machine. If this document and the code disagree, the code wins and this document must be corrected.

---

## 1. Scope

TriCore V9 is a sequence operator with three independently ablatable mechanisms:

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
- **Track 4B:** real CUDA allocation and cached-decoding latency at `L=128..32,768`.

The specification has six non-goals:

- It does **not** claim natural-language or reasoning capability.
- It does **not** claim superiority over a tuned Transformer.
- It does **not** claim that every component is necessary for every task or seed.
- It does **not** claim adaptive computation time; `max_loops` is a fixed hyperparameter.
- It does **not** claim that the JIT scan is a custom CUDA kernel or a production-throughput solution.
- It does **not** claim that the recurrent state update is parallel; only the multi-hop refinement is vectorized across `L`.

The evidence is three single-seed V6/V7 evaluation tracks — a 500-step five-variant 2-hop associative recall study, a 450-step two-model causal bytecode LM study, and a V7 study with streaming equivalence/memory checks, a 450-step 3-hop comparison, and a 300-step real-source held-out LM comparison — plus V9's 4-chain, five-way, 3-seed 2-Hop MQAR variance check and measured CUDA decode-state/KV-cache profiling through 32K.

---

## 2. Repository contract

```text
tricore-arch/
├── tricore/
│   ├── __init__.py          # block/LM/baseline exports and all task loaders
│   ├── model.py             # tricore_fused_scan, TriCoreBlock, TriCoreLM,
│   │                        # BaselineTransformerLM, init_state/forward_step
│   └── tasks.py             # 2-hop MQAR, Causal Bytecode LM, 3-hop recall,
│                            # real Python source held-out byte stream
├── tests/
│   ├── test_sanity.py       # forward/backward/shape/gradient-flow
│   ├── test_v5.py           # four-output JIT equivalence + causal bytecode backward
│   └── test_v7_streaming.py # streaming equivalence, 17 KiB state, Track 3 gradients
├── results/
│   ├── BENCHMARK_REPORT.md  # V6 tracks preserved + V7 streaming/Track 3
│   └── tricore_vs_transformer.png
├── run_experiment.py        # V6 dual-track training, ablation, cache, report, plot
├── run_v7_benchmark.py      # V7 streaming/Track 3 benchmark; does not rerun V6
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

`pyproject.toml` declares `tricore-arch==1.2.0`, Python `>=3.10`, and the same runtime dependencies. `pip install -e .` installs the `tricore` import package.

### 2.2 Public API

```python
from tricore import (
    TriCoreBlock, TriCoreLM, BaselineTransformerLM,
    generate_associative_recall_batch, generate_contextual_code_lm_batch,
    generate_3hop_recall_batch, load_real_python_corpus_batches,
)
```

`TriCoreBlock.forward(...)` returns `(y, surprise_trace)`:

- `y: [B, L, D]`
- `surprise_trace: [B, L]`

`TriCoreLM.forward(idx)` returns:

- `logits: [B, L, vocab_size]`
- `mean_surprise: [B, L]`, the mean of per-layer surprise traces

`BaselineTransformerLM.forward(idx)` returns `(logits, None)`.

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
    Q: torch.Tensor,             # [B, L, H, d_head]
    K: torch.Tensor,             # [B, L, H, d_head]
    V: torch.Tensor,             # [B, L, H, d_head]
    base_decay: torch.Tensor,    # [D]
    fast_decay: torch.Tensor,    # [1, H, 1, 1]
    fast_lr: torch.Tensor,       # [1, H, 1, 1]
    hop_proj: torch.Tensor,      # [max_loops, 1, H, d_head, d_head]
    enable_surprise_gate: bool,
    enable_fast_weight: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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
w_t^{\text{hop}} = \operatorname{softmax}(W_r u_t) \in \mathbb{R}^{L_{\max}+1}
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
q^{(i+1)} = \operatorname{normalize}\left(r^{(i)} + H_i r^{(i)}\right)
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
z_t = h_t + W_o \operatorname{vec}(r_t^{\text{out}})
```

```math
z_t \leftarrow z_t + s_t \odot \operatorname{MLP}(\operatorname{LN}(z_t)), \qquad
y_t = x_t + \operatorname{LN}(z_t)
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
)
```

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

There is no multi-seed confidence interval for the full V6/V7 Track 1/2/3 reference runs, no hyperparameter search for the Transformer, and no FLOP matching. V9 adds a 4-chain, five-way, 3-seed mean ± std check for Track 4A (`d_model=64`, 4 chains, 4 queries); it is a variance check, not a significance test, and it does not repower the full-size runs. Track 1/2 have no real held-out set. Track 3B has a real-source held-out suffix but the corpus is only this small repository. Track 2 and Track 3A are synthetic; Track 3B is not a human-code benchmark.

---

## 7. Latest benchmark record

**Generated by:** `python run_experiment.py`  
**Report:** `results/BENCHMARK_REPORT.md`  
**Plot:** `results/tricore_vs_transformer.png`  
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

The complete suite currently reports 7 passing tests. `run_v7_benchmark.py` regenerates only the V7 streaming and Track 3 sections; it preserves the V6 Track 1/Track 2 tables in `results/BENCHMARK_REPORT.md`.

V9 adds no new parameter names. `run_v9_verification.py` exercises the zero-`F_seq` training path under three seeds and writes `results/V9_METRICS.json`; `demo_showcase.py` exercises `init_state` / `forward_step` end to end. Track 4A is a 4-chain, five-way, 3-seed rerun, and Track 4B is a real CUDA allocation/latency measurement through `L=32,768`, both reported in sections 7.8-7.9.

### 8.4 Full run

The V6 reference run completed on CUDA and regenerated the report and plot. `run_v7_benchmark.py` then appended the V7 streaming and Track 3 sections in place, preserving the V6 tables. The report is the source of truth for all numbers in this document. The generated V6 plot is a 2x2 panel:

- Track 1 learning curves
- Track 1 ablation bars
- Track 2 validation PPL
- Track 2 dereference accuracy

### 8.5 Commands

```bash
python -m pytest -q
python tests/test_v5.py
python tests/test_v7_streaming.py
python run_experiment.py
python run_v7_benchmark.py
python run_v9_verification.py
python demo_showcase.py
python run_experiment.py --steps 2 --code-steps 2  # smoke
```

---

## 9. Invariants and open work

### 9.1 Invariants

1. **No attention over tokens.** No softmax may mix sequence positions. The only softmax is the hop router over `max_loops + 1` slots.
2. **O(1) state.** `h`, `F_mat`, and `prev_k` shapes must be independent of `L`.
3. **Causal shifted write.** Engine II binds `K_{t-1}` to `V_t`; the read uses `Q_t` after the write.
4. **Delta rule.** The write is `(V_t - F_{t-1}K_{t-1})K_{t-1}^T`.
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

- 3-seed mean ± std for the full-size V6/V7 tracks (V9 already does the 4-chain 2-Hop MQAR) and a significance test.
- 4-6 hop chains and variable `max_loops`.
- Stronger adversarial writes and higher noise density.
- Streaming latency and memory beyond the measured 32K range (128K-1M) and against a production serving stack.
- A fused CUDA/Triton scan to replace the TorchScript loop.
- Tuned, parameter-matched Transformer baselines at matched compute as well as parameters.
- A larger real-code corpus with held-out files/repositories for Track 3B.
- Natural-language pretraining beyond synthetic recall.

---

<p align="center">
  <i>TriCore V9 — zero-F_seq training, constant state, surprise-gated writes, causal delta memory, a fused latent loop, and native one-token streaming.</i>
</p>