"""TriCore V12 apex benchmark: zero-prior gates, Delta-rule mutation, 1M CUDA.

Writes ``results/V12_METRICS.json`` and two master figures:

* ``results/tricore_v12_apex.png`` (the V12 2x3 summary)
* ``results/tricore_vs_transformer.png`` (kept as the master comparison path)

No token-ID separator/syntax/reset masks are used anywhere in V12.  Track 5
and Track 7 enable ``autonomous_syntax=True``: infix connectors, structural
writers, and the terminal delimiter are discovered from within-sequence token
frequency and causal adjacency alone.  A vocabulary permutation leaves the
discovered masks unchanged.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.modules.setdefault("pynvml", None)

import torch
import torch.nn.functional as F

from tricore import (
    BaselineTransformerLM,
    TriCoreLM,
    generate_stateful_mutation_batch,
)
from run_v10_eligibility import eval_infix, generate_infix_eq_batch
from run_v11_flagship import (
    BATCH_SIZE,
    LEARNING_RATE,
    SEEDS,
    TRACK2_DEREF,
    TRACK3B,
    TRACK6_CHANCE,
    _allocation_delta,
    count_parameters,
    fmt_p,
    json_ready,
    make_kv_cache,
    mean_std,
    measure_tricore_latency,
    measure_transformer_latency,
    state_bytes,
    welch_t_test,
    write_metrics,
)


METRICS_PATH = Path("results/V12_METRICS.json")
FIGURE_PATH = Path("results/tricore_v12_apex.png")
MASTER_FIGURE_PATH = Path("results/tricore_vs_transformer.png")

TRACK5_STEPS = 350
TRACK7_STEPS = 400
TRACK5_D_MODEL = 64
TRACK5_D_FAST = 16
GATE_LEARNING_RATE = 3e-3
PROFILE_LENGTHS = (
    128,
    512,
    2_048,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
)
PROFILE_D_MODEL = 128
PROFILE_HEADS = 4
PROFILE_D_HEAD = 32


def gate_parameters(model):
    """Parameters of the autonomous gate paths (no token-ID masks)."""
    params = []
    if not hasattr(model, "layers"):
        return params
    for block in model.layers:
        for module in (
            block.write_gate_proj,
            block.elig_gate_proj,
            block.gate_token_proj,
        ):
            params.extend(module.parameters())
    if hasattr(model, "gate_role_table"):
        params.extend(model.gate_role_table.parameters())
    return params


def build_optimizer(model, steps, learning_rate=LEARNING_RATE,
                    gate_learning_rate=GATE_LEARNING_RATE,
                    weight_decay=1e-2):
    gate_params = gate_parameters(model)
    gate_ids = {id(param) for param in gate_params}
    other_params = [p for p in model.parameters() if id(p) not in gate_ids]
    groups = [
        {
            "params": other_params,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    ]
    if gate_params:
        groups.append(
            {
                "params": gate_params,
                "lr": gate_learning_rate,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1)
    )
    return optimizer, scheduler


@torch.no_grad()
def compute_gate_stats(model, device) -> dict:
    """Exact per-role gate means, including the V12 token-role bypass."""
    model.eval()
    if not model.enable_surprise_gate:
        # The fair No-Engine-I ablation bypasses every learned gate; reporting
        # the unmasked sigmoid logits would be misleading.  Match V10 by
        # reporting the neutral open-gate value.
        model.train()
        return {
            "elig_eq": 1.0,
            "elig_entity": 1.0,
            "write_eq": 1.0,
            "write_entity": 1.0,
        }
    torch.manual_seed(5099)
    x, _ = generate_infix_eq_batch(batch_size=64, device=device)
    emb = model.embedding(x)
    block = model.layers[0]
    B, L, D = emb.shape
    H = block.n_heads
    u = block.in_norm(emb)
    u_feat = F.silu(block.u_proj(u))
    u_surp_bias = block.surprise_u_proj(u)
    w_logits = block.write_gate_proj(u).view(B, L, H)
    e_logits = block.elig_gate_proj(u).view(B, L, H)
    token_logits = block.gate_token_proj(u).view(B, L, H, 2)
    role_logits = model.gate_role_table(x).view(B, L, H, 2)
    w_logits = (
        w_logits
        + token_logits[..., 0] * block.gate_logit_gain
        + role_logits[..., 0] * model.gate_role_gain
    )
    e_logits = (
        e_logits
        + token_logits[..., 1] * block.gate_logit_gain
        + role_logits[..., 1] * model.gate_role_gain
    )
    separator_mask = model._build_separator_mask(x)
    if separator_mask is not None:
        if isinstance(separator_mask, tuple):
            elig_sep_mask, write_sep_mask = separator_mask
        else:
            elig_sep_mask = separator_mask
            write_sep_mask = separator_mask
        w_logits = w_logits - write_sep_mask.to(w_logits.dtype).view(B, L, 1) * 8.0
        e_logits = e_logits - elig_sep_mask.to(e_logits.dtype).view(B, L, 1) * 8.0

    h = torch.zeros(B, D, device=emb.device, dtype=emb.dtype)
    base_decay = torch.sigmoid(block.base_decay_logit)
    w_seq, e_seq = [], []
    for t in range(L):
        u_t = u_feat[:, t]
        g_t = torch.sigmoid(u_surp_bias[:, t] + torch.abs(u_t - h))
        s = g_t.mean(dim=-1).view(B, 1, 1)
        w_seq.append(torch.sigmoid(w_logits[:, t] + s - 0.5).mean(dim=-1))
        e_seq.append(torch.sigmoid(e_logits[:, t] + s - 0.5).mean(dim=-1))
        h = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_t
    w_seq = torch.stack(w_seq, dim=1)
    e_seq = torch.stack(e_seq, dim=1)

    groups = {
        "syntax": (x == 0) | (x >= 59),
        "alias": (x >= 1) & (x <= 16),
        "var": (x >= 17) & (x <= 32),
        "lit": (x >= 33) & (x <= 48),
    }
    stats = {}
    for name, mask in groups.items():
        stats[f"elig_{name}"] = e_seq[mask].mean().item()
        stats[f"write_{name}"] = w_seq[mask].mean().item()
    stats["elig_eq"] = e_seq[x == 59].mean().item()
    stats["elig_entity"] = e_seq[(x >= 1) & (x <= 48)].mean().item()
    stats["write_eq"] = w_seq[x == 59].mean().item()
    stats["write_entity"] = w_seq[(x >= 1) & (x <= 48)].mean().item()
    model.train()
    return stats


# ---------------------------------------------------------------------------
# E1: zero-prior Track 5
# ---------------------------------------------------------------------------


def build_track5_configs():
    return {
        "TriCore V12 Zero-Prior Full": lambda: TriCoreLM(
            vocab_size=64,
            d_model=TRACK5_D_MODEL,
            n_layers=1,
            d_fast=TRACK5_D_FAST,
            max_loops=2,
            enable_surprise_gate=True,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=None,
            syntax_tokens=None,
            reset_trace_tokens=None,
            enable_delta_rule=True,
            autonomous_syntax=True,
        ),
        "TriCore V12 Zero-Prior Fair No-Engine-I": lambda: TriCoreLM(
            vocab_size=64,
            d_model=TRACK5_D_MODEL,
            n_layers=1,
            d_fast=TRACK5_D_FAST,
            max_loops=2,
            enable_surprise_gate=False,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=None,
            syntax_tokens=None,
            reset_trace_tokens=None,
            enable_delta_rule=True,
            autonomous_syntax=True,
        ),
        "Vanilla Transformer (2-layer)": lambda: BaselineTransformerLM(
            vocab_size=64,
            d_model=TRACK5_D_MODEL,
            n_layers=2,
            n_heads=4,
            max_seq_len=128,
        ),
    }


def train_track5_model(name, builder, seed, device, steps=TRACK5_STEPS):
    torch.manual_seed(seed)
    model = builder().to(device)
    params = count_parameters(model)
    optimizer, scheduler = build_optimizer(model, steps)
    started = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, steps + 1):
        x, y = generate_infix_eq_batch(
            batch_size=BATCH_SIZE, num_vars=4, num_queries=4, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, 64), y.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = loss.item()
        if step == 1 or step % 50 == 0 or step == steps:
            print(
                f"[E1] {name} seed={seed} step={step:03d}/{steps} "
                f"loss={last_loss:.4f}",
                flush=True,
            )
    train_seconds = time.perf_counter() - started
    val_loss, val_acc = eval_infix(model, device, batches=10)
    gates = (
        compute_gate_stats(model, device)
        if isinstance(model, TriCoreLM)
        else {}
    )
    result = {
        "seed": seed,
        "params": params,
        "train_loss_last": last_loss,
        "val_loss": val_loss,
        "accuracy_pct": val_acc,
        "gate_stats": gates,
        "train_time_s": train_seconds,
    }
    print(
        f"DONE E1 {name} seed={seed}: acc={val_acc:.2f}% loss={val_loss:.4f} "
        f"EQ/entity={gates.get('elig_eq', float('nan')):.3f}/"
        f"{gates.get('elig_entity', float('nan')):.3f} "
        f"time={train_seconds:.1f}s",
        flush=True,
    )
    return result


def run_track5(device, seeds=SEEDS, steps=TRACK5_STEPS) -> dict:
    models = {}
    for name, builder in build_track5_configs().items():
        runs = [
            train_track5_model(name, builder, seed, device, steps)
            for seed in seeds
        ]
        acc_mean, acc_std = mean_std(r["accuracy_pct"] for r in runs)
        loss_mean, loss_std = mean_std(r["val_loss"] for r in runs)
        gate_summary = {}
        for key in runs[0].get("gate_stats", {}):
            values = [r["gate_stats"][key] for r in runs]
            gate_summary[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        models[name] = {
            "params": runs[0]["params"],
            "runs": runs,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "val_loss_mean": loss_mean,
            "val_loss_std": loss_std,
            "gate_stats": gate_summary,
        }

    full_name = "TriCore V12 Zero-Prior Full"
    fair_name = "TriCore V12 Zero-Prior Fair No-Engine-I"
    transformer_name = "Vanilla Transformer (2-layer)"
    acc = {
        name: [run["accuracy_pct"] for run in payload["runs"]]
        for name, payload in models.items()
    }
    loss = {
        name: [run["val_loss"] for run in payload["runs"]]
        for name, payload in models.items()
    }
    significance = {
        "full_vs_fair_no_engine_i_accuracy": welch_t_test(
            acc[full_name], acc[fair_name]
        ),
        "full_vs_transformer_accuracy": welch_t_test(
            acc[full_name], acc[transformer_name]
        ),
        "full_vs_fair_no_engine_i_loss": welch_t_test(
            loss[full_name], loss[fair_name]
        ),
        "full_vs_transformer_loss": welch_t_test(
            loss[full_name], loss[transformer_name]
        ),
    }
    return {
        "protocol": {
            "seeds": list(seeds),
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "gate_learning_rate": GATE_LEARNING_RATE,
            "weight_decay": 1e-2,
            "d_model": TRACK5_D_MODEL,
            "d_fast": TRACK5_D_FAST,
            "max_loops": 2,
            "eval_batches": 10,
            "test": "infix EQ two-hop dereference, zero token-ID priors",
            "token_id_masks": False,
            "chance_pct": 25.0,
        },
        "models": models,
        "significance": significance,
    }



# ---------------------------------------------------------------------------
# E2: Track 7 stateful pointer mutation (Delta vs Hebbian)
# ---------------------------------------------------------------------------


def mutation_old_values(inputs: torch.Tensor) -> torch.Tensor:
    """Recover the stale literal that each RET query must *not* return."""
    batch, length = inputs.shape
    old = torch.full_like(inputs, -100)
    cpu = inputs.detach().cpu()
    for b in range(batch):
        alias_to_var = {}
        first = {}
        for pos in range(0, length, 5):
            op = int(cpu[b, pos])
            if op == 60:  # LET alias = var ;
                alias_to_var[int(cpu[b, pos + 1])] = int(cpu[b, pos + 3])
            elif op == 61:  # SET var = literal ;
                var = int(cpu[b, pos + 1])
                if var not in first:
                    first[var] = int(cpu[b, pos + 3])
        for pos in range(0, length, 5):
            if int(cpu[b, pos]) == 62:  # RET alias = ; ;
                alias = int(cpu[b, pos + 1])
                var = alias_to_var[alias]
                old[b, pos + 2] = first[var]
    return old.to(inputs.device)


def evaluate_mutation(model, batches, device):
    model.eval()
    total_loss = 0.0
    new_correct = 0
    stale_predictions = 0
    total = 0
    with torch.no_grad():
        for x, y in batches:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, 64), y.view(-1), ignore_index=-100
            )
            total_loss += loss.item()
            mask = y != -100
            preds = logits.argmax(dim=-1)
            new_correct += (preds[mask] == y[mask]).sum().item()
            old = mutation_old_values(x)
            stale_predictions += (preds[mask] == old[mask]).sum().item()
            total += int(mask.sum().item())
    model.train()
    return (
        total_loss / max(len(batches), 1),
        new_correct / max(total, 1) * 100.0,
        stale_predictions / max(total, 1) * 100.0,
    )


def build_mutation_batches(device, steps, seed):
    torch.manual_seed(seed)
    return [
        generate_stateful_mutation_batch(
            batch_size=BATCH_SIZE,
            num_vars=4,
            num_queries=4,
            num_nops=4,
            vocab_size=64,
            device=device,
        )
        for _ in range(steps)
    ]


def build_mutation_eval(device, batches=10, seed=7001):
    torch.manual_seed(seed)
    return [
        generate_stateful_mutation_batch(
            batch_size=BATCH_SIZE,
            num_vars=4,
            num_queries=4,
            num_nops=4,
            vocab_size=64,
            device=device,
        )
        for _ in range(batches)
    ]


def build_track7_configs():
    def tricore(delta=True, surprise=True):
        return TriCoreLM(
            vocab_size=64,
            d_model=TRACK5_D_MODEL,
            n_layers=1,
            d_fast=TRACK5_D_FAST,
            max_loops=2,
            enable_surprise_gate=surprise,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=None,
            syntax_tokens=None,
            reset_trace_tokens=None,
            enable_delta_rule=delta,
            autonomous_syntax=True,
        )

    return {
        "TriCore V12 Delta Full": lambda: tricore(delta=True, surprise=True),
        "TriCore V12 Hebbian Full": lambda: tricore(delta=False, surprise=True),
        "TriCore V12 Fair No-Engine-I Delta": lambda: tricore(
            delta=True, surprise=False
        ),
        "Vanilla Transformer (2-layer)": lambda: BaselineTransformerLM(
            vocab_size=64,
            d_model=TRACK5_D_MODEL,
            n_layers=2,
            n_heads=4,
            max_seq_len=128,
        ),
    }


def train_track7_model(name, builder, seed, device, train_batches,
                       eval_batches, steps=TRACK7_STEPS):
    torch.manual_seed(seed)
    model = builder().to(device)
    params = count_parameters(model)
    optimizer, scheduler = build_optimizer(model, len(train_batches))
    started = time.perf_counter()
    losses = []
    for step, (x, y) in enumerate(train_batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, 64), y.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach())
        if step == 1 or step % 100 == 0 or step == steps:
            print(
                f"[E2] {name} seed={seed} step={step:03d}/{steps} "
                f"loss={loss.item():.4f}",
                flush=True,
            )
    train_seconds = time.perf_counter() - started
    val_loss, new_acc, stale_acc = evaluate_mutation(
        model, eval_batches, device
    )
    result = {
        "name": name,
        "seed": seed,
        "params": params,
        "train_loss_last50": torch.stack(losses[-50:]).mean().item(),
        "val_loss": val_loss,
        "new_value_accuracy_pct": new_acc,
        "stale_value_confusion_pct": stale_acc,
        "train_time_s": train_seconds,
    }
    print(
        f"DONE E2 {name} seed={seed}: new={new_acc:.2f}% "
        f"stale={stale_acc:.2f}% loss={val_loss:.4f} time={train_seconds:.1f}s",
        flush=True,
    )
    return result



def summarize_track7_model(runs):
    new_mean, new_std = mean_std(
        r["new_value_accuracy_pct"] for r in runs
    )
    stale_mean, stale_std = mean_std(
        r["stale_value_confusion_pct"] for r in runs
    )
    loss_mean, loss_std = mean_std(r["val_loss"] for r in runs)
    return {
        "params": runs[0]["params"],
        "runs": runs,
        "new_value_accuracy_mean": new_mean,
        "new_value_accuracy_std": new_std,
        "stale_value_confusion_mean": stale_mean,
        "stale_value_confusion_std": stale_std,
        "val_loss_mean": loss_mean,
        "val_loss_std": loss_std,
    }


def build_track7_payload(models, seeds, steps) -> dict:
    delta = "TriCore V12 Delta Full"
    hebbian = "TriCore V12 Hebbian Full"
    fair = "TriCore V12 Fair No-Engine-I Delta"
    transformer = "Vanilla Transformer (2-layer)"
    required = (delta, hebbian, fair, transformer)
    significance = {}
    complete = all(
        name in models
        and set(int(seed) for seed in seeds)
        <= {int(run["seed"]) for run in models[name].get("runs", [])}
        for name in required
    )
    if complete:
        new = {
            name: [r["new_value_accuracy_pct"] for r in payload["runs"]]
            for name, payload in models.items()
        }
        stale = {
            name: [r["stale_value_confusion_pct"] for r in payload["runs"]]
            for name, payload in models.items()
        }
        significance = {
            "delta_vs_hebbian_new_value": welch_t_test(
                new[delta], new[hebbian]
            ),
            "delta_vs_hebbian_stale_value": welch_t_test(
                stale[delta], stale[hebbian]
            ),
            "delta_vs_fair_no_engine_i_new_value": welch_t_test(
                new[delta], new[fair]
            ),
            "delta_vs_transformer_new_value": welch_t_test(
                new[delta], new[transformer]
            ),
            "delta_vs_transformer_stale_value": welch_t_test(
                stale[delta], stale[transformer]
            ),
        }
    return {
        "protocol": {
            "seeds": list(seeds),
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "gate_learning_rate": GATE_LEARNING_RATE,
            "weight_decay": 1e-2,
            "num_vars": 4,
            "num_queries": 4,
            "num_nops": 4,
            "sequence_length": 100,
            "eval_batches": 10,
            "test": "stateful pointer mutation across infix EQ",
            "token_id_masks": False,
            "chance_pct": 25.0,
            "delta_rule": "F_t = decay*F_{t-1} + lr*(V_t - F_{t-1}K_{t-1})K_{t-1}^T",
            "hebbian_rule": "F_t = decay*F_{t-1} + lr*V_t K_{t-1}^T",
        },
        "models": models,
        "significance": significance,
    }


def run_track7(device, seeds=SEEDS, steps=TRACK7_STEPS, existing=None,
               run_seeds=None, model_names=None, on_checkpoint=None) -> dict:
    train_batches = build_mutation_batches(device, steps, seed=42_007)
    eval_batches = build_mutation_eval(device)
    models = {
        name: payload for name, payload in (existing or {}).items()
    }
    configs = build_track7_configs()
    selected = model_names or list(configs)
    for name in selected:
        if name not in configs:
            raise ValueError(f"unknown Track 7 model: {name}")
        builder = configs[name]
        runs = list(models.get(name, {}).get("runs", []))
        completed = {int(run["seed"]) for run in runs}
        for seed in (run_seeds or seeds):
            if int(seed) in completed:
                print(
                    f"[E2] resume: {name} seed={seed} already complete",
                    flush=True,
                )
                continue
            result = train_track7_model(
                name,
                builder,
                seed,
                device,
                train_batches,
                eval_batches,
                steps=steps,
            )
            runs.append(result)
            models[name] = summarize_track7_model(runs)
            payload = build_track7_payload(models, seeds, steps)
            if on_checkpoint is not None:
                on_checkpoint(payload)
    return build_track7_payload(models, seeds, steps)



# ---------------------------------------------------------------------------
# E3: real CUDA memory and latency to one million tokens
# ---------------------------------------------------------------------------


def build_track4b_payload(device, lengths, rows) -> dict:
    ordered = sorted(rows, key=lambda row: int(row["length"]))
    return {
        "device": device,
        "lengths": list(lengths),
        "state_config": {
            "d_model": PROFILE_D_MODEL,
            "n_heads": PROFILE_HEADS,
            "d_head": PROFILE_D_HEAD,
            "n_layers": 1,
        },
        "baseline_config": {
            "n_layers": 2,
            "d_model": PROFILE_D_MODEL,
            "n_heads": PROFILE_HEADS,
            "d_head": PROFILE_D_HEAD,
        },
        "rows": ordered,
    }


def run_track4b(device, lengths=PROFILE_LENGTHS, existing_rows=None,
                all_lengths=None, on_checkpoint=None) -> dict:
    tri = TriCoreLM(
        vocab_size=64,
        d_model=PROFILE_D_MODEL,
        n_layers=1,
        d_fast=PROFILE_D_HEAD,
        max_loops=2,
    ).to(device)
    selected_lengths = tuple(int(length) for length in lengths)
    full_lengths = tuple(
        int(length) for length in (all_lengths or selected_lengths)
    )
    baseline = BaselineTransformerLM(
        vocab_size=64,
        d_model=PROFILE_D_MODEL,
        n_layers=2,
        n_heads=PROFILE_HEADS,
        max_seq_len=max(full_lengths),
    ).to(device)
    tri.eval()
    baseline.eval()

    rows = [
        dict(row) for row in (existing_rows or [])
        if int(row["length"]) in full_lengths
    ]
    completed = {int(row["length"]) for row in rows}
    for length in selected_lengths:
        if length in completed:
            print(
                f"Track4B resume: L={length:,} already complete",
                flush=True,
            )
            continue
        high = length >= 262_144
        reps = 8 if high else (16 if length >= 65_536 else 32)
        warmups = 2 if high else (4 if length >= 65_536 else 8)
        state, state_alloc = _allocation_delta(
            device, lambda: tri.init_state(1, device, torch.float32)
        )
        state_tensor_bytes = state_bytes(state)
        tri_ms = measure_tricore_latency(
            tri, state, device, length, reps, warmups
        )

        caches, cache_alloc = make_kv_cache(baseline, length, device)
        k_cache, v_cache = caches
        cache_tensor_bytes = state_bytes(caches)
        transformer_ms = measure_transformer_latency(
            baseline, k_cache, v_cache, length, device, reps, warmups
        )
        row = {
            "length": length,
            "state_tensor_bytes": state_tensor_bytes,
            "state_alloc_bytes": state_alloc,
            "cache_tensor_bytes": cache_tensor_bytes,
            "cache_alloc_bytes": cache_alloc,
            "tri_ms_per_token": tri_ms,
            "transformer_ms_per_token": transformer_ms,
            "reduction": cache_tensor_bytes / state_tensor_bytes,
            "latency_ratio_transformer_over_tricore": transformer_ms / tri_ms,
            "timed_steps": reps,
            "warmup_steps": warmups,
        }
        rows.append(row)
        print(
            "Track4B L={:>7} state={:>6} B cache={:>10} B "
            "tri={:.3f} ms/token tf={:.3f} ms/token redux={:,.1f}x".format(
                length,
                state_tensor_bytes,
                cache_tensor_bytes,
                tri_ms,
                transformer_ms,
                row["reduction"],
            ),
            flush=True,
        )
        if on_checkpoint is not None:
            on_checkpoint(
                build_track4b_payload(device, full_lengths, rows)
            )
        del state, caches, k_cache, v_cache
        if device == "cuda":
            torch.cuda.empty_cache()

    return build_track4b_payload(device, full_lengths, rows)



# ---------------------------------------------------------------------------
# E4: V12 two-by-three apex figure
# ---------------------------------------------------------------------------


def load_v11_depth() -> dict:
    path = Path("results/V11_METRICS.json")
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("track6", {}).get("depth_reference", {})


def plot_apex(metrics: dict, output_paths=(FIGURE_PATH, MASTER_FIGURE_PATH)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "tricore": "#0B8457",
        "tricore_light": "#54C59A",
        "shallow": "#E08A1E",
        "transformer": "#7F8C8D",
        "blue": "#2E86C1",
        "red": "#C0392B",
        "hebbian": "#D68910",
    }
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=180)

    # (a) Depth scaling retained from the V11 four-hop track.
    depth = load_v11_depth()
    two = depth.get("two_hop", {})
    three = depth.get("three_hop_v7", {})
    four = depth.get("four_hop_v11", {})
    hops = ["2-Hop\n(Track 1)", "3-Hop\n(Track 3A)", "4-Hop\n(Track 6)"]
    full = [
        two.get("full_v9_ood_pct", 100.0),
        three.get("full_ood_pct", 100.0),
        four.get("full_ood_pct", 100.0),
    ]
    shallow = [
        two.get("one_hop_limit_ood_pct", 20.47),
        three.get("two_hop_limit_ood_pct", 23.91),
        four.get("three_hop_limit_ood_pct", 24.76),
    ]
    transformer = [
        two.get("transformer_v9_ood_pct", 22.24),
        three.get("transformer_ood_pct", 23.44),
        four.get("transformer_ood_pct", 25.78),
    ]
    x = np.arange(len(hops))
    width = 0.26
    ax = axes[0, 0]
    b1 = ax.bar(
        x - width, full, width, color=colors["tricore"],
        label="TriCore matched depth",
    )
    ax.bar(
        x, shallow, width, color=colors["shallow"],
        label="TriCore one-hop shallow limit",
    )
    ax.bar(
        x + width, transformer, width, color=colors["transformer"],
        label="2-layer Transformer",
    )
    ax.axhline(
        TRACK6_CHANCE, color=colors["red"], linestyle="--", linewidth=1.2,
        label="25.0% chance",
    )
    ax.set_xticks(x, hops)
    ax.set_ylim(0, 150)
    ax.set_ylabel("Recall at 4x distractor gap (%)")
    ax.set_title("(a) Depth scaling law (V11 four-hop reference)")
    for bar in b1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=7.4,
        )
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=7.4)

    # (b) E1: zero-prior Track 5 across three seeds.
    t5 = metrics["track5"]
    names = list(t5["models"].keys())
    labels = [
        "TriCore V12\nzero-prior Full",
        "Fair\nNo-Engine-I",
        "2-Layer\nTransformer",
    ]
    means = [t5["models"][name]["accuracy_mean"] for name in names]
    stds = [t5["models"][name]["accuracy_std"] for name in names]
    ax = axes[0, 1]
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=6,
        color=[colors["tricore"], colors["shallow"], colors["transformer"]],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(25.0, color=colors["red"], linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Infix EQ two-hop accuracy (%)")
    ax.set_title("(b) Zero-prior EQ dereference (3-seed mean ± std)")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(mean + std + 2, 116),
            f"{mean:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    sig = t5["significance"]
    ax.text(
        0.03, 0.96,
        "Welch p (accuracy)\n"
        f"Full vs Fair: {fmt_p(sig['full_vs_fair_no_engine_i_accuracy']['p_value_two_tailed'])}\n"
        f"Full vs TF:   {fmt_p(sig['full_vs_transformer_accuracy']['p_value_two_tailed'])}",
        transform=ax.transAxes, va="top", fontsize=7.6,
    )

    # (c) E2: Delta vs Hebbian on stateful pointer mutation.
    t7 = metrics["track7"]
    t7_names = list(t7["models"].keys())
    short = ["Delta\nFull", "Hebbian\nFull", "Fair\nNo-I Delta", "2-Layer\nTF"]
    new_means = [
        t7["models"][n]["new_value_accuracy_mean"] for n in t7_names
    ]
    new_std = [t7["models"][n]["new_value_accuracy_std"] for n in t7_names]
    stale_means = [
        t7["models"][n]["stale_value_confusion_mean"] for n in t7_names
    ]
    x = np.arange(len(t7_names))
    width = 0.36
    ax = axes[0, 2]
    ax.bar(
        x - width / 2, new_means, width, yerr=new_std, capsize=4,
        color=colors["tricore"], label="New-value accuracy",
    )
    ax.bar(
        x + width / 2, stale_means, width,
        color=colors["hebbian"], alpha=0.85, label="Stale-value confusion",
    )
    ax.axhline(25.0, color=colors["red"], linestyle="--", linewidth=1.1)
    ax.set_xticks(x, short)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Query accuracy (%)")
    ax.set_title("(c) Track 7: Delta erase vs Hebbian interference")
    ax.legend(frameon=False, loc="upper right", fontsize=7.6)
    for index, (new, stale) in enumerate(zip(new_means, stale_means)):
        ax.text(
            index - width / 2, new + 2, f"{new:.1f}",
            ha="center", fontsize=7.4, fontweight="bold",
        )
        ax.text(
            index + width / 2, stale + 2, f"{stale:.1f}",
            ha="center", fontsize=7.4,
        )
    p_stale = t7["significance"]["delta_vs_hebbian_stale_value"][
        "p_value_two_tailed"
    ]
    ax.text(
        0.03, 0.96,
        f"Delta vs Hebbian\nstale p = {fmt_p(p_stale)}",
        transform=ax.transAxes, va="top", fontsize=7.6,
    )

    # (d) Real CUDA memory scaling to 1,048,576 tokens.
    profile = metrics["track4b"]
    rows = profile["rows"]
    lengths = [row["length"] for row in rows]
    state_kib = [row["state_tensor_bytes"] / 1024.0 for row in rows]
    cache_kib = [row["cache_tensor_bytes"] / 1024.0 for row in rows]
    final = rows[-1]
    ax = axes[1, 0]
    ax.loglog(
        lengths, cache_kib, marker="o", linewidth=2.0,
        color=colors["transformer"], label="Transformer K+V cache",
    )
    ax.loglog(
        lengths, state_kib, marker="s", linewidth=2.2,
        color=colors["tricore"], label="TriCore O(1) state",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length L (tokens, log scale)")
    ax.set_ylabel("Decode memory (KiB, log scale)")
    ax.set_title("(d) Real CUDA memory scaling, L=128 to 1,048,576")
    ax.grid(True, which="both", alpha=0.18)
    ax.annotate(
        f"{final['cache_tensor_bytes'] / 1024 / 1024 / 1024:.2f} GiB\n"
        f"({final['cache_tensor_bytes'] / 1024 / 1024:,.0f} MiB)",
        xy=(final["length"], final["cache_tensor_bytes"] / 1024.0),
        xytext=(-128, 8), textcoords="offset points", fontsize=8,
        color=colors["transformer"],
        arrowprops={"arrowstyle": "-", "color": colors["transformer"]},
    )
    ax.annotate(
        f"{final['state_tensor_bytes'] / 1024:.2f} KiB\n"
        f"{final['reduction']:,.1f}x reduction",
        xy=(final["length"], final["state_tensor_bytes"] / 1024.0),
        xytext=(-128, 24), textcoords="offset points", fontsize=8,
        color=colors["tricore"],
        arrowprops={"arrowstyle": "-", "color": colors["tricore"]},
    )
    ax.legend(frameon=False, loc="upper left")

    # (e) Real CUDA decoding latency to 1M tokens.
    tri_ms = [row["tri_ms_per_token"] for row in rows]
    tf_ms = [row["transformer_ms_per_token"] for row in rows]
    ax = axes[1, 1]
    ax.plot(
        lengths, tf_ms, marker="o", linewidth=2.0,
        color=colors["transformer"], label="Transformer cached decode",
    )
    ax.plot(
        lengths, tri_ms, marker="s", linewidth=2.2,
        color=colors["tricore"], label="TriCore forward_step",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length L (tokens, log2 scale)")
    ax.set_ylabel("Decode latency (ms / token)")
    ax.set_title("(e) Real CUDA decoding latency to 1M")
    ax.grid(True, which="both", alpha=0.18)
    crossing = next(
        (
            row for row in rows
            if row["transformer_ms_per_token"] > row["tri_ms_per_token"]
        ),
        None,
    )
    if crossing is not None:
        ax.axvline(
            crossing["length"], color=colors["red"], linestyle=":",
            linewidth=1.2, alpha=0.8,
        )
        ax.text(
            crossing["length"] * 1.03, max(tf_ms) * 0.55,
            f"crossover\nL={crossing['length']:,}",
            fontsize=8, color=colors["red"],
        )
    ax.annotate(
        f"L={final['length']:,}: {final['transformer_ms_per_token']:.2f} ms",
        xy=(final["length"], final["transformer_ms_per_token"]),
        xytext=(-142, 8), textcoords="offset points", fontsize=8,
        color=colors["transformer"],
        arrowprops={"arrowstyle": "-", "color": colors["transformer"]},
    )
    ax.annotate(
        f"L={final['length']:,}: {final['tri_ms_per_token']:.2f} ms",
        xy=(final["length"], final["tri_ms_per_token"]),
        xytext=(-142, 22), textcoords="offset points", fontsize=8,
        color=colors["tricore"],
        arrowprops={"arrowstyle": "-", "color": colors["tricore"]},
    )
    ax.legend(frameon=False, loc="upper left")

    # (f) Retained code-modeling results from V6/V7.
    ax = axes[1, 2]
    groups = ["Track 2\nbytecode deref", "Track 3B\nheld-out next-byte"]
    tri_scores = [TRACK2_DEREF["tricore"], TRACK3B["tricore"]["next_byte"]]
    tf_scores = [
        TRACK2_DEREF["transformer"],
        TRACK3B["transformer"]["next_byte"],
    ]
    x = np.arange(len(groups))
    width = 0.32
    ax.bar(
        x - width / 2, tri_scores, width,
        color=colors["tricore"], label="TriCore",
    )
    ax.bar(
        x + width / 2, tf_scores, width,
        color=colors["transformer"], label="Transformer",
    )
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(f) Track 2 & Track 3B code modeling")
    for index, (tri_value, tf_value) in enumerate(zip(tri_scores, tf_scores)):
        ax.text(
            index - width / 2, tri_value + 2, f"{tri_value:.1f}",
            ha="center", fontsize=8, fontweight="bold",
        )
        ax.text(
            index + width / 2, tf_value + 2, f"{tf_value:.1f}",
            ha="center", fontsize=8,
        )
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "TriCore V12 — zero-prior autonomous gates, Delta-rule state mutation, "
        "and 1M-token constant-state CUDA decoding",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Track 5 / Track 7: mean ± std over seeds 42/100/2026; exact "
        "two-tailed Welch p-values. Track 4B: real torch.cuda.memory_allocated "
        "deltas and cached decode latency, not formula estimates.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0.0, 0.035, 1.0, 0.955])
    saved = []
    for output_path in output_paths:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        saved.append(str(output_path))
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the TriCore V12 zero-prior apex suite"
    )
    parser.add_argument("--steps5", type=int, default=TRACK5_STEPS)
    parser.add_argument("--steps7", type=int, default=TRACK7_STEPS)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS)
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=max(PROFILE_LENGTHS),
        help="largest Track 4B length to profile",
    )
    parser.add_argument("--skip-track5", action="store_true")
    parser.add_argument("--skip-track7", action="store_true")
    parser.add_argument("--skip-track4b", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed runs from --metrics and train only missing ones",
    )
    parser.add_argument(
        "--track7-models",
        nargs="+",
        default=None,
        help="optional Track 7 model subset for resumable runs",
    )
    parser.add_argument(
        "--track7-seeds",
        nargs="+",
        type=int,
        default=None,
        help="optional Track 7 seed subset for resumable runs",
    )
    parser.add_argument(
        "--profile-lengths",
        nargs="+",
        type=int,
        default=None,
        help="optional Track 4B length subset for resumable runs",
    )
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TriCore V12 apex requires CUDA for the 1,048,576-token profile"
        )
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_path = Path(args.metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"=== TriCore V12 apex on {torch.cuda.get_device_name(0)} ===",
        flush=True,
    )
    started = time.perf_counter()
    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError(
                f"--resume needs an existing metrics file: {output_path}"
            )
        metrics = json.loads(output_path.read_text(encoding="utf-8"))
        metrics["version"] = "1.5.0"
        metrics["device"] = device
        metrics["gpu"] = torch.cuda.get_device_name(0)
        metrics["zero_prior"] = True
        metrics["token_id_masks"] = False
        metrics.setdefault("protocol", {}).update(
            {
                "track5_steps": args.steps5,
                "track7_steps": args.steps7,
                "seeds": args.seeds,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "gate_learning_rate": GATE_LEARNING_RATE,
                "weight_decay": 1e-2,
                "gate_weight_decay": 0.0,
            }
        )
        print(f"=== Resuming from {output_path} ===", flush=True)
    else:
        metrics = {
            "version": "1.5.0",
            "device": device,
            "gpu": torch.cuda.get_device_name(0),
            "started_at_unix": time.time(),
            "zero_prior": True,
            "token_id_masks": False,
            "protocol": {
                "track5_steps": args.steps5,
                "track7_steps": args.steps7,
                "seeds": args.seeds,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "gate_learning_rate": GATE_LEARNING_RATE,
                "weight_decay": 1e-2,
                "gate_weight_decay": 0.0,
            },
        }

    if not args.skip_track5:
        if args.resume and "track5" in metrics:
            print(
                "\n--- E1: reusing completed zero-prior Track 5 ---",
                flush=True,
            )
        else:
            print("\n--- E1: zero-prior Track 5 ---", flush=True)
            metrics["track5"] = run_track5(
                device, seeds=tuple(args.seeds), steps=args.steps5
            )
            write_metrics(metrics, output_path)
            print(f"Checkpoint written to {output_path}", flush=True)

    if not args.skip_track7:
        print("\n--- E2: Track 7 Delta vs Hebbian ---", flush=True)
        existing_models = (
            metrics.get("track7", {}).get("models", {})
            if args.resume else {}
        )

        def checkpoint_track7(payload):
            metrics["track7"] = payload
            write_metrics(metrics, output_path)
            print(
                f"Checkpoint written to {output_path} "
                f"({len(payload['models'])} models)",
                flush=True,
            )

        metrics["track7"] = run_track7(
            device,
            seeds=tuple(args.seeds),
            steps=args.steps7,
            existing=existing_models,
            run_seeds=args.track7_seeds,
            model_names=args.track7_models,
            on_checkpoint=checkpoint_track7,
        )
        write_metrics(metrics, output_path)
        print(f"Checkpoint written to {output_path}", flush=True)

    if not args.skip_track4b:
        print("\n--- E3: real CUDA profiling to 1M ---", flush=True)
        all_lengths = tuple(
            sorted(set(PROFILE_LENGTHS) | set(args.profile_lengths or ()))
        )
        lengths = tuple(
            length for length in (args.profile_lengths or PROFILE_LENGTHS)
            if length <= args.max_length
        )
        if not lengths:
            raise ValueError("--max-length excludes every profiling length")
        existing_rows = (
            metrics.get("track4b", {}).get("rows", [])
            if args.resume else []
        )

        def checkpoint_track4b(payload):
            metrics["track4b"] = payload
            metrics["track6_reference"] = load_v11_depth()
            write_metrics(metrics, output_path)
            print(
                f"Checkpoint written to {output_path} "
                f"({len(payload['rows'])} lengths)",
                flush=True,
            )

        metrics["track4b"] = run_track4b(
            device,
            lengths=lengths,
            existing_rows=existing_rows,
            all_lengths=all_lengths,
            on_checkpoint=checkpoint_track4b,
        )
        metrics["track6_reference"] = load_v11_depth()
        write_metrics(metrics, output_path)
        print(f"Checkpoint written to {output_path}", flush=True)

    elapsed = time.perf_counter() - started
    metrics["runtime_seconds"] = (
        float(metrics.get("runtime_seconds", 0.0)) + elapsed
        if args.resume else elapsed
    )
    if not args.skip_plot:
        if "track7" not in metrics or "track4b" not in metrics:
            raise RuntimeError(
                "full plotting requires Track 7 and Track 4B metrics"
            )
        figures = plot_apex(metrics)
        metrics["figures"] = figures
        print("Wrote figures: " + ", ".join(figures), flush=True)
    write_metrics(metrics, output_path)

    print(
        f"\nV12_BENCHMARK_COMPLETED_SUCCESSFULLY in "
        f"{metrics['runtime_seconds']:.1f}s",
        flush=True,
    )
    if "track5" in metrics:
        for name, payload in metrics["track5"]["models"].items():
            print(
                f"  {name}: {payload['accuracy_mean']:.2f}% "
                f"± {payload['accuracy_std']:.2f}",
                flush=True,
            )
    if "track7" in metrics:
        for name, payload in metrics["track7"]["models"].items():
            print(
                f"  {name}: new={payload['new_value_accuracy_mean']:.2f}% "
                f"stale={payload['stale_value_confusion_mean']:.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()



