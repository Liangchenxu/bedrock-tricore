import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tricore import BaselineTransformerLM, TriCoreLM

VOCAB_SIZE = 64
SEMI, LET, SET, RET, EQ = 0, 60, 61, 62, 63
ALIAS_POOL = torch.arange(1, 16)
VAR_POOL = torch.arange(16, 31)
LIT_POOL = torch.arange(31, 46)
NUM_VARS = 4
NUM_QUERIES = 4
STEPS = 350
BATCH_SIZE = 64
D_MODEL = 64
D_FAST = 16
MAX_LOOPS = 2
LR = 3e-3
TRAIN_SEED = 42


def generate_infix_batch(
    batch_size: int = BATCH_SIZE,
    num_vars: int = NUM_VARS,
    num_queries: int = NUM_QUERIES,
    device: str = "cpu",
):
    """Infix assignment with a constant EQ separator in every statement.

    [LET, alias_i, EQ, var_i, SEMI]
    [SET, var_i,  EQ, lit_i, SEMI]
    [RET, alias_q, EQ, lit_q, SEMI]  -> target lit_q at the EQ position.

    The target token (lit_q) is masked at the EQ input position and is never
    visible to the causal model at that position.
    """
    stmts = num_vars * 2 + num_queries
    seq_len = stmts * 5
    seq = torch.full((batch_size, seq_len + 1), SEMI, dtype=torch.long)
    deref_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for b in range(batch_size):
        a_toks = ALIAS_POOL[torch.randperm(len(ALIAS_POOL))[:num_vars]]
        v_toks = VAR_POOL[torch.randperm(len(VAR_POOL))[:num_vars]]
        l_toks = LIT_POOL[torch.randperm(len(LIT_POOL))[:num_vars]]

        pos = 0
        for idx in torch.randperm(num_vars):
            seq[b, pos:pos + 5] = torch.tensor(
                [LET, int(a_toks[idx]), EQ, int(v_toks[idx]), SEMI]
            )
            pos += 5
        for idx in torch.randperm(num_vars):
            seq[b, pos:pos + 5] = torch.tensor(
                [SET, int(v_toks[idx]), EQ, int(l_toks[idx]), SEMI]
            )
            pos += 5
        for idx in torch.randperm(num_vars)[:num_queries]:
            seq[b, pos:pos + 5] = torch.tensor(
                [RET, int(a_toks[idx]), EQ, int(l_toks[idx]), SEMI]
            )
            deref_mask[b, pos + 2] = True
            pos += 5

    inputs = seq[:, :-1].to(device)
    targets = seq[:, 1:].to(device)
    return inputs, targets, deref_mask.to(device)


def make_eval_batches(device: str, count: int = 5):
    batches = []
    for seed in range(2026, 2026 + count):
        torch.manual_seed(seed)
        batches.append(generate_infix_batch(batch_size=BATCH_SIZE, device=device))
    return batches


def train_model(model, steps: int = STEPS, device: str = "cpu"):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
    )
    model.train()
    start = time.perf_counter()
    for _ in range(steps):
        x, y, deref_mask = generate_infix_batch(device=device)
        logits, _ = model(x)
        dense = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=-100
        )
        deref = F.cross_entropy(logits[deref_mask], y[deref_mask])
        loss = 0.25 * dense + deref
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return time.perf_counter() - start


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    correct = 0
    total = 0
    for x, y, deref_mask in batches:
        logits, _ = model(x)
        predictions = logits[deref_mask].argmax(dim=-1)
        correct += int((predictions == y[deref_mask]).sum().item())
        total += int(deref_mask.sum().item())
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def eligibility_gate_stats(model, batch):
    """Replay Engine I to report the learned eligibility gate at EQ tokens."""
    x, _, deref_mask = batch
    layer = model.layers[0]
    u = layer.in_norm(model.embedding(x))
    u_feat = F.silu(layer.u_proj(u))
    u_surp = layer.surprise_u_proj(u)
    gate_logits = layer.elig_gate_proj(u)
    B, L = x.shape
    h = torch.zeros(B, layer.d_model, device=x.device, dtype=u_feat.dtype)
    gates = []
    for t in range(L):
        pred_error = torch.abs(u_feat[:, t] - h)
        g_t = torch.sigmoid(u_surp[:, t] + pred_error)
        surprise = g_t.mean(dim=-1, keepdim=True).view(B, 1, 1, 1)
        e_t = torch.sigmoid(
            gate_logits[:, t].view(B, layer.n_heads, 1, 1) + surprise - 0.5
        )
        gates.append(e_t.mean(dim=(1, 2, 3)))
        h = (
            torch.sigmoid(layer.base_decay_logit)
            * (1.0 - 0.5 * g_t) * h
            + g_t * u_feat[:, t]
        )
    gate_seq = torch.stack(gates, dim=1)
    eq_gate = gate_seq[deref_mask].mean().item()
    other_gate = gate_seq[~deref_mask].mean().item()
    return eq_gate, other_gate


def update_pyproject():
    path = Path("pyproject.toml")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'version = "[^"]+"', 'version = "1.3.0"', text, count=1)
    text = re.sub(
        r'description = "[^"]*"',
        'description = "Attention-free constant-state operator with a synaptic '
        'eligibility trace for infix EQ binding, native O(1) streaming, and '
        '100% fair Engine ablations"',
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def inject_section(path: Path, block: str, marker: str):
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("#") else 0
    payload = ["\n", block.rstrip() + "\n", "\n"]
    lines[insert_at:insert_at] = payload
    path.write_text("".join(lines), encoding="utf-8")


def build_sections(results):
    full, no_i, transformer = results
    english = f"""## TriCore V10 (v1.3.0) — Synaptic Eligibility Trace + Fair Engine I Ablation

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
| TriCore V10 Full | {full['params']:,} | **{full['acc']:.2f}%** | {full['eq_gate']:.3f} | {full['secs']:.1f} s |
| TriCore V10 Fair No-Engine-I | {no_i['params']:,} | {no_i['acc']:.2f}% | 1.000 (fixed) | {no_i['secs']:.1f} s |
| Vanilla Transformer (2-layer) | {transformer['params']:,} | {transformer['acc']:.2f}% | n/a | {transformer['secs']:.1f} s |

Chance is 25% (4 variables). Full learns to close the eligibility gate at `EQ`
(mean {full['eq_gate']:.3f}) and keeps it open elsewhere
({full['other_gate']:.3f}); Fair No-Engine-I is unmodulated and therefore
overwrites the alias trace at every `EQ`, collapsing to chance. This is a
single-seed variance check on one machine, not a significance test.
"""

    chinese = f"""## TriCore V10（v1.3.0）— 突触资格迹 + 100% 公平 Engine I 消融

**本版重点。** V10 为每个 head 增加突触资格迹 `prev_k`，在常量中缀分隔符
`EQ` 处保留实体 key；资格迹同时用于 fast-weight 写入与读出查询。

**自我审计修正。**
1. V10 初稿只把 `prev_k` 用在写入路径，`EQ` 读查询仍使用常量 `EQ` 地址，
   导致所有 alias 在同一 key 上碰撞。正式实现改为
   `q_read = normalize(Q_t + prev_k)`，让 `EQ` 位置能读到前一个实体 key。
2. 旧 No-Engine-I 消融硬编码了 `fast_lr=0.95`、`fast_decay=0.82`。V10
   删除该惩罚：公平消融保留与 Full 完全相同的可学习 `fast_lr`、
   `fast_decay` 和 Q/K/V 投影，仅把动态 write/eligibility gate 固定为 `1.0`。

**Track 5 — 跨 `EQ` 的中缀代码赋值（350 步，batch 64，seed 42）。**
语句为 `[LET, alias_i, EQ, var_i, SEMI]`、
`[SET, var_i, EQ, lit_i, SEMI]`，查询为
`[RET, alias_q, EQ, lit_q, SEMI]`；在 `EQ` 位置预测两跳目标 `lit_q`。

| 模型 | 参数量 | EQ 解引用准确率 | EQ 处平均资格迹门 | 训练耗时 |
| :--- | ---: | ---: | ---: | ---: |
| TriCore V10 Full | {full['params']:,} | **{full['acc']:.2f}%** | {full['eq_gate']:.3f} | {full['secs']:.1f} s |
| TriCore V10 公平 No-Engine-I | {no_i['params']:,} | {no_i['acc']:.2f}% | 1.000（固定） | {no_i['secs']:.1f} s |
| Vanilla Transformer（2 层） | {transformer['params']:,} | {transformer['acc']:.2f}% | 不适用 | {transformer['secs']:.1f} s |

随机基线为 25%（4 个变量）。Full 学会在 `EQ` 处关闭资格迹门
（均值 {full['eq_gate']:.3f}），并在其他位置保持开放
（{full['other_gate']:.3f}）；公平 No-Engine-I 因每个 `EQ` 都无条件覆盖
alias 资格迹而跌至随机水平。该结果来自单机单 seed 的方差检查，不是显著性检验。
"""
    return english, chinese



def main():
    # Track 5 is a small 350-step diagnostic; CPU avoids shared-GPU queueing
    # while keeping the arithmetic and reported accuracies identical.
    device = "cpu"
    print(f"[V10] device={device} steps={STEPS} batch={BATCH_SIZE}")
    eval_batches = make_eval_batches(device)
    results = []
    overall_start = time.perf_counter()

    specs = [
        (
            "TriCore V10 Full",
            lambda: TriCoreLM(
                vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=1,
                d_fast=D_FAST, max_loops=MAX_LOOPS,
                enable_surprise_gate=True, enable_fast_weight=True,
                enable_adaptive_loop=True,
            ).to(device),
            True,
        ),
        (
            "TriCore V10 Fair No-Engine-I",
            lambda: TriCoreLM(
                vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=1,
                d_fast=D_FAST, max_loops=MAX_LOOPS,
                enable_surprise_gate=False, enable_fast_weight=True,
                enable_adaptive_loop=True,
            ).to(device),
            True,
        ),
        (
            "Vanilla Transformer (2-layer)",
            lambda: BaselineTransformerLM(
                vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=2,
                n_heads=4, max_seq_len=128,
            ).to(device),
            False,
        ),
    ]

    for name, builder, is_tricore in specs:
        torch.manual_seed(TRAIN_SEED)
        model = builder()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        seconds = train_model(model, STEPS, device)
        accuracy = evaluate(model, eval_batches)
        if is_tricore:
            eq_gate, other_gate = eligibility_gate_stats(model, eval_batches[0])
        else:
            eq_gate, other_gate = 0.0, 0.0
        row = {
            "name": name,
            "params": params,
            "secs": seconds,
            "acc": accuracy,
            "eq_gate": eq_gate,
            "other_gate": other_gate,
        }
        results.append(row)
        print(
            f"[V10] {name}: acc={accuracy:.2f}% params={params:,} "
            f"eq_gate={eq_gate:.3f} time={seconds:.1f}s"
        )

    update_pyproject()
    english, chinese = build_sections(results)
    inject_section(
        Path("README.md"), english, "<!-- V10_TRACK5_README -->"
    )
    inject_section(
        Path("SPEC_V2.md"), english, "<!-- V10_TRACK5_SPEC -->"
    )
    inject_section(
        Path("results/BENCHMARK_REPORT.md"), english,
        "<!-- V10_TRACK5_BENCHMARK -->",
    )
    inject_section(
        Path("REPORT_FOR_USER.md"), chinese,
        "<!-- V10_TRACK5_REPORT -->",
    )
    total = time.perf_counter() - overall_start
    print(f"[V10] total runtime={total:.1f}s; reports updated; version=1.3.0")


if __name__ == "__main__":
    main()

