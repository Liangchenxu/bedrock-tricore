import glob
import json
import os
import sys
import time

# Prevent deprecated pynvml shim from blocking during torch import on Windows
sys.modules.setdefault("pynvml", None)

import torch
import torch.nn.functional as F

from tricore import BaselineTransformerLM, TriCoreLM


def generate_infix_eq_batch(
    batch_size: int = 64,
    num_vars: int = 4,
    num_queries: int = 4,
    num_nops: int = 4,
    vocab_size: int = 64,
    device: str = "cpu",
):
    """Track 5: Infix code assignment with explicit '=' (EQ) connector & adversarial NOP gap.

    Statements (5 tokens each):
      1. Declaration: [LET, alias_i, EQ, var_i, SEMI]   (binds alias_i -> var_i across EQ)
      2. Assignment:  [SET, var_i,   EQ, lit_i, SEMI]   (binds var_i   -> lit_i across EQ)
      3. Distractor:  [NOP, var_i,   EQ, SEMI,  SEMI]   (adversarial infix no-op; overwrites var_i if ungated!)
      4. Return Eval: [RET, alias_q, EQ, SEMI,  SEMI]   (predicts two-hop target lit_q at the EQ position pos + 2!)

    Note: Target literal lit_q is NEVER placed in `inputs` during RET queries, eliminating
    any across-query elimination shortcut and keeping chance accuracy strictly at 1 / num_vars (25.0%).
    """
    SEMI, EQ, LET, SET, RET, NOP = 0, 59, 60, 61, 62, 63
    tier = 16
    alias_pool = torch.arange(1, 1 + tier)
    var_pool = torch.arange(1 + tier, 1 + 2 * tier)
    lit_pool = torch.arange(1 + 2 * tier, 1 + 3 * tier)

    total_stmts = num_vars * 2 + num_nops + num_queries
    seq_len = total_stmts * 5

    inputs = torch.full((batch_size, seq_len), SEMI, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        a_t = alias_pool[torch.randperm(tier)[:num_vars]]
        v_t = var_pool[torch.randperm(tier)[:num_vars]]
        l_t = lit_pool[torch.randperm(tier)[:num_vars]]

        pos = 0
        # 1. Shuffled LET statements: alias_i = var_i ;
        for i in torch.randperm(num_vars):
            inputs[b, pos : pos + 5] = torch.tensor([LET, a_t[i], EQ, v_t[i], SEMI])
            pos += 5

        # 2. Shuffled SET statements: var_i = lit_i ;
        for i in torch.randperm(num_vars):
            inputs[b, pos : pos + 5] = torch.tensor([SET, v_t[i], EQ, l_t[i], SEMI])
            pos += 5

        # 3. Adversarial NOP statements: NOP var_i = ; ;
        for n_i in range(num_nops):
            v_dist = v_t[n_i % num_vars]
            inputs[b, pos : pos + 5] = torch.tensor([NOP, v_dist, EQ, SEMI, SEMI])
            pos += 5

        # 4. Shuffled RET queries: RET alias_q = ; ; (predict lit_q at pos + 2, i.e., at EQ)
        for i in torch.randperm(num_vars)[:num_queries]:
            inputs[b, pos : pos + 5] = torch.tensor([RET, a_t[i], EQ, SEMI, SEMI])
            targets[b, pos + 2] = l_t[i]
            pos += 5

    return inputs.to(device), targets.to(device)


@torch.no_grad()
def compute_eq_elig_gate_stats(model: TriCoreLM, device: str) -> tuple[float, float]:
    """Measure mean eligibility gate activation at EQ tokens vs entity tokens."""
    if not isinstance(model, TriCoreLM):
        return float("nan"), float("nan")
    if not model.enable_surprise_gate:
        return 1.0, 1.0
    model.eval()
    torch.manual_seed(5099)
    x, _ = generate_infix_eq_batch(batch_size=64, num_vars=4, num_queries=4, device=device)
    emb = model.embedding(x)
    sep_mask = model._build_separator_mask(x)
    block = model.layers[0]
    B, L, D = emb.shape
    H = block.n_heads
    u = block.in_norm(emb)
    u_feat = F.silu(block.u_proj(u))
    u_surp_bias = block.surprise_u_proj(u)
    elig_logits = block.elig_gate_proj(u).view(B, L, H, 1, 1)
    if sep_mask is not None:
        elig_sep_mask, _ = sep_mask
        elig_logits = elig_logits - elig_sep_mask.to(elig_logits.dtype).view(B, L, 1, 1, 1) * 8.0

    # Run causal recurrence to get exact surprise_scalar per step
    h = torch.zeros(B, D, device=emb.device, dtype=emb.dtype)
    base_decay = torch.sigmoid(block.base_decay_logit)
    elig_gates = []
    for t in range(L):
        u_t = u_feat[:, t, :]
        pred_error = torch.abs(u_t - h)
        g_t = torch.sigmoid(u_surp_bias[:, t, :] + pred_error)
        s_4d = g_t.mean(dim=-1, keepdim=True).view(B, 1, 1, 1)
        e_gate = torch.sigmoid(elig_logits[:, t] + s_4d - 0.5)
        elig_gates.append(e_gate.mean(dim=1).squeeze(-1).squeeze(-1))
        h = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_t
    elig_seq = torch.stack(elig_gates, dim=1)  # [B, L]
    eq_mask = (x == 59)
    entity_mask = (x >= 1) & (x <= 48)
    mean_eq = elig_seq[eq_mask].mean().item()
    mean_entity = elig_seq[entity_mask].mean().item()
    model.train()
    return mean_eq, mean_entity


def eval_infix(model, device, batches=10):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for i in range(batches):
            torch.manual_seed(5000 + i)
            x, y = generate_infix_eq_batch(
                batch_size=64, num_vars=4, num_queries=4, device=device
            )
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, 64), y.view(-1), ignore_index=-100
            )
            loss_sum += loss.item()
            mask = y != -100
            correct += (logits.argmax(dim=-1)[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return loss_sum / batches, (correct / max(total, 1)) * 100.0


def train_and_eval(name, builder, device, steps=350):
    torch.manual_seed(42)
    model = builder().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = generate_infix_eq_batch(
            batch_size=64, num_vars=4, num_queries=4, device=device
        )
        opt.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"[{name}] Step {step:03d}/{steps} | train_loss={loss.item():.4f}", flush=True)
    val_loss, val_acc = eval_infix(model, device)
    elapsed = time.time() - t0
    eq_gate, ent_gate = compute_eq_elig_gate_stats(model, device)
    print(
        f"DONE {name}: Acc={val_acc:.2f}% | ValLoss={val_loss:.4f} | "
        f"EligGate(EQ/Entity)={eq_gate:.4f}/{ent_gate:.4f} | Time={elapsed:.1f}s",
        flush=True,
    )
    return params, val_loss, val_acc, eq_gate, ent_gate, elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = 350
    SEMI, EQ, LET, SET, RET, NOP = 0, 59, 60, 61, 62, 63
    syntax_toks = (SEMI, LET, SET, RET, NOP)

    print(f"=== Running TriCore V10 Track 5 Benchmark on {device.upper()} ===")
    configs = {
        "TriCore V10 Full (With Eligibility Trace)": lambda: TriCoreLM(
            vocab_size=64,
            d_model=64,
            n_layers=1,
            d_fast=16,
            max_loops=2,
            enable_surprise_gate=True,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=EQ,
            syntax_tokens=syntax_toks,
        ),
        "TriCore V10 Fair No-Engine-I (Unmodulated Gate)": lambda: TriCoreLM(
            vocab_size=64,
            d_model=64,
            n_layers=1,
            d_fast=16,
            max_loops=2,
            enable_surprise_gate=False,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=EQ,
            syntax_tokens=syntax_toks,
        ),
        "Vanilla Transformer (2-Layer)": lambda: BaselineTransformerLM(
            vocab_size=64, d_model=64, n_layers=2, n_heads=4, max_seq_len=128
        ),
    }

    results = {}
    for name, builder in configs.items():
        results[name] = train_and_eval(name, builder, device, steps)
        with open("_v10_progress", "w", encoding="utf-8") as handle:
            handle.write(f"DONE {name}: {results[name]}\n")

    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", encoding="utf-8") as handle:
            text = handle.read()
        text = text.replace('version = "1.2.0"', 'version = "1.3.0"')
        with open("pyproject.toml", "w", encoding="utf-8") as handle:
            handle.write(text)

    metrics = {
        name: {
            "parameters": params,
            "val_loss": val_loss,
            "accuracy_pct": val_acc,
            "mean_elig_gate_eq": eq_gate,
            "mean_elig_gate_entity": ent_gate,
            "train_time_s": elapsed,
        }
        for name, (params, val_loss, val_acc, eq_gate, ent_gate, elapsed) in results.items()
    }
    os.makedirs("results", exist_ok=True)
    with open("results/V10_METRICS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": "1.3.0",
                "track": 5,
                "steps": steps,
                "batch_size": 64,
                "seed": 42,
                "device": device,
                "chance_pct": 25.0,
                "results": metrics,
            },
            handle,
            indent=2,
        )

    print("V10_BENCHMARK_COMPLETED_SUCCESSFULLY", flush=True)
    for name, (params, val_loss, val_acc, eq_gate, ent_gate, elapsed) in results.items():
        print(
            f"RESULT | {name} | params={params:,} | val_loss={val_loss:.4f} | "
            f"acc={val_acc:.2f}% | elig_eq={eq_gate:.4f} | elig_entity={ent_gate:.4f} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
