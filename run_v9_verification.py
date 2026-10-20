"""TriCore V9 verification: zero-F_seq training, 5-way 3-seed ablation, 32K CUDA.

The script preserves every existing V6/V7 section in
``results/BENCHMARK_REPORT.md`` and replaces only its own V8/V9 appendix
block.  It writes ``results/V9_METRICS.json``.
"""

import json
import os
import re
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricore import (
    BaselineTransformerLM,
    TriCoreLM,
    generate_associative_recall_batch,
)


REPORT_PATH = Path("results/BENCHMARK_REPORT.md")
METRICS_PATH = Path("results/V9_METRICS.json")
V8_START = "<!-- V8_APPENDIX_START -->"
V8_END = "<!-- V8_APPENDIX_END -->"
V9_START = "<!-- V9_APPENDIX_START -->"
V9_END = "<!-- V9_APPENDIX_END -->"

SEEDS = (42, 100, 2026)
STEPS = 350
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
VOCAB_SIZE = 64
D_MODEL = 64
D_FAST = 16
N_CHAINS = 4
N_QUERIES = 4
NOISE_LEN = 8
OOD_MULTIPLIER = 4
EVAL_BATCHES = 5
LENGTHS = (128, 512, 2048, 8192, 16384, 32768)


def sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def iter_state_tensors(states):
    if torch.is_tensor(states):
        yield states
        return
    for item in states:
        yield from iter_state_tensors(item)


def state_bytes(states) -> int:
    return sum(t.numel() * t.element_size() for t in iter_state_tensors(states))


def evaluate_recall(model: nn.Module, eval_batches) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in eval_batches:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item()
            mask = y != -100
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return total_loss / max(len(eval_batches), 1), (
        correct / max(total, 1)
    ) * 100.0


def build_train_batches(seed: int, device: str):
    torch.manual_seed(seed + 10000)
    return [
        generate_associative_recall_batch(
            batch_size=BATCH_SIZE,
            num_kv_pairs=N_CHAINS,
            num_queries=N_QUERIES,
            vocab_size=VOCAB_SIZE,
            noise_len=NOISE_LEN,
            device=device,
        )
        for _ in range(STEPS)
    ]


def build_eval_sets(device: str):
    torch.manual_seed(8101)
    in_domain = [
        generate_associative_recall_batch(
            batch_size=BATCH_SIZE,
            num_kv_pairs=N_CHAINS,
            num_queries=N_QUERIES,
            vocab_size=VOCAB_SIZE,
            noise_len=NOISE_LEN,
            device=device,
        )
        for _ in range(EVAL_BATCHES)
    ]
    torch.manual_seed(8102)
    extrapolation = [
        generate_associative_recall_batch(
            batch_size=BATCH_SIZE,
            num_kv_pairs=N_CHAINS,
            num_queries=N_QUERIES,
            vocab_size=VOCAB_SIZE,
            noise_len=NOISE_LEN * OOD_MULTIPLIER,
            device=device,
        )
        for _ in range(EVAL_BATCHES)
    ]
    return in_domain, extrapolation


def _tricore_builder(**flags):
    return lambda: TriCoreLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=1,
        d_fast=D_FAST,
        max_loops=2,
        **flags,
    )


def build_track4a_configs():
    return {
        "TriCore Full": _tricore_builder(),
        "No Engine I: Surprise Gate": _tricore_builder(
            enable_surprise_gate=False
        ),
        "No Engine II: Fast-Weight": _tricore_builder(
            enable_fast_weight=False
        ),
        "No Engine III: Adaptive Loop": _tricore_builder(
            enable_adaptive_loop=False
        ),
        "Vanilla Transformer 2-Layer": lambda: BaselineTransformerLM(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=2,
            n_heads=4,
            max_seq_len=128,
        ),
    }


def train_track4a_model(name, builder, seed, train_batches, eval_sets, device):
    torch.manual_seed(seed)
    model = builder().to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STEPS
    )

    warm_x, warm_y = train_batches[0]
    logits, _ = model(warm_x)
    loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE), warm_y.view(-1), ignore_index=-100
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)

    losses = []
    sync_device(device)
    start = time.perf_counter()
    for x, y in train_batches:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach())
    sync_device(device)
    train_time = time.perf_counter() - start
    train_loss = torch.stack(losses[-50:]).mean().item()

    in_loss, in_acc = evaluate_recall(model, eval_sets[0])
    ood_loss, ood_acc = evaluate_recall(model, eval_sets[1])
    result = {
        "name": name,
        "seed": seed,
        "params": params,
        "train_loss": train_loss,
        "in_loss": in_loss,
        "in_acc": in_acc,
        "ood_loss": ood_loss,
        "ood_acc": ood_acc,
        "train_time": train_time,
    }
    print(
        "Track4A {:<34} seed={:>4} params={:>7} in={:6.2f}% ood={:6.2f}% "
        "time={:5.1f}s".format(
            name, seed, params, in_acc, ood_acc, train_time
        ),
        flush=True,
    )
    return result


def run_track4a(device: str):
    configs = build_track4a_configs()
    eval_sets = build_eval_sets(device)
    results = []
    for seed in SEEDS:
        train_batches = build_train_batches(seed, device)
        print(f"Track4A seed {seed}: {len(train_batches)} pre-generated batches")
        for name, builder in configs.items():
            results.append(
                train_track4a_model(
                    name, builder, seed, train_batches, eval_sets, device
                )
            )
        del train_batches

    summary = []
    for name in configs:
        rows = [r for r in results if r["name"] == name]
        summary.append(
            {
                "name": name,
                "params": rows[0]["params"],
                "in_mean": statistics.mean(r["in_acc"] for r in rows),
                "in_std": statistics.stdev(r["in_acc"] for r in rows),
                "ood_mean": statistics.mean(r["ood_acc"] for r in rows),
                "ood_std": statistics.stdev(r["ood_acc"] for r in rows),
                "train_mean": statistics.mean(r["train_time"] for r in rows),
                "seeds": [r["seed"] for r in rows],
                "per_seed": [
                    {
                        "seed": r["seed"],
                        "in_acc": r["in_acc"],
                        "ood_acc": r["ood_acc"],
                    }
                    for r in rows
                ],
            }
        )
    return {
        "protocol": {
            "num_kv_pairs": N_CHAINS,
            "num_queries": N_QUERIES,
            "noise_len": NOISE_LEN,
            "ood_noise_len": NOISE_LEN * OOD_MULTIPLIER,
            "d_model": D_MODEL,
            "d_fast": D_FAST,
            "max_loops": 2,
            "chance_level": 100.0 / N_CHAINS,
        },
        "summary": summary,
        "runs": results,
    }


def _allocation_delta(device: str, factory):
    if device == "cuda":
        sync_device(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
    else:
        before = 0
    obj = factory()
    if device == "cuda":
        torch.cuda.synchronize()
        measured = torch.cuda.memory_allocated() - before
    else:
        measured = 0
    return obj, measured


def make_kv_cache(model: BaselineTransformerLM, length: int, device: str):
    n_layers = len(model.encoder.layers)
    n_heads = model.encoder.layers[0].self_attn.num_heads
    d_head = model.encoder.layers[0].self_attn.head_dim

    def allocate():
        shape = (n_layers, length, 1, n_heads, d_head)
        return (
            torch.zeros(shape, device=device, dtype=torch.float32),
            torch.zeros(shape, device=device, dtype=torch.float32),
        )

    return _allocation_delta(device, allocate)


def cached_transformer_step(model, idx_t, pos, k_cache, v_cache):
    """One real cached decode step through the 2-layer Transformer."""
    if idx_t.dim() == 2:
        idx_t = idx_t[:, 0]
    batch = idx_t.shape[0]
    positions = torch.tensor([pos], device=idx_t.device)
    x = model.embedding(idx_t).unsqueeze(1)
    x = x + model.pos_embedding(positions).unsqueeze(0)

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        n_heads = attn.num_heads
        d_head = attn.head_dim
        d_model = n_heads * d_head
        normed = layer.norm1(x)
        qkv = F.linear(normed, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, 1, n_heads, d_head).transpose(1, 2)
        k = k.reshape(batch, 1, n_heads, d_head).transpose(1, 2)
        v = v.reshape(batch, 1, n_heads, d_head).transpose(1, 2)

        k_cache[layer_idx, pos] = k[:, :, 0, :]
        v_cache[layer_idx, pos] = v[:, :, 0, :]
        keys = k_cache[layer_idx, : pos + 1].permute(1, 2, 0, 3)
        values = v_cache[layer_idx, : pos + 1].permute(1, 2, 0, 3)

        attended = F.scaled_dot_product_attention(
            q, keys, values, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, 1, d_model)
        x = x + attn.out_proj(attended)
        ff = layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        x = x + ff
    return model.lm_head(model.final_norm(x))


def measure_tricore_latency(
    model, state, device: str, length: int, reps: int = 32
):
    idx_t = torch.tensor([7], device=device, dtype=torch.long)
    base_step = max(length - reps - 8, 0)
    with torch.no_grad():
        for step in range(8):
            _, _, state = model.forward_step(
                idx_t, state, base_step + step
            )
        sync_device(device)
        start = time.perf_counter()
        for step in range(reps):
            _, _, state = model.forward_step(
                idx_t, state, base_step + 8 + step
            )
        sync_device(device)
    return (time.perf_counter() - start) * 1000.0 / reps


def measure_transformer_latency(
    model, k_cache, v_cache, length: int, device: str, reps: int = 32
):
    idx_t = torch.tensor([7], device=device, dtype=torch.long)
    pos = length - 1
    with torch.no_grad():
        for _ in range(8):
            cached_transformer_step(model, idx_t, pos, k_cache, v_cache)
        sync_device(device)
        start = time.perf_counter()
        for _ in range(reps):
            cached_transformer_step(model, idx_t, pos, k_cache, v_cache)
        sync_device(device)
    return (time.perf_counter() - start) * 1000.0 / reps


def run_track4b(device: str):
    tri = TriCoreLM(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_layers=1,
        d_fast=32,
        max_loops=2,
    ).to(device)
    baseline = BaselineTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=max(LENGTHS),
    ).to(device)
    tri.eval()
    baseline.eval()

    rows = []
    for length in LENGTHS:
        state, state_alloc = _allocation_delta(
            device, lambda: tri.init_state(1, device, torch.float32)
        )
        state_tensor_bytes = state_bytes(state)
        tri_ms = measure_tricore_latency(tri, state, device, length)

        caches, cache_alloc = make_kv_cache(baseline, length, device)
        k_cache, v_cache = caches
        cache_tensor_bytes = state_bytes(caches)
        tf_ms = measure_transformer_latency(
            baseline, k_cache, v_cache, length, device
        )
        rows.append(
            {
                "length": length,
                "state_tensor_bytes": state_tensor_bytes,
                "state_alloc_bytes": state_alloc,
                "cache_tensor_bytes": cache_tensor_bytes,
                "cache_alloc_bytes": cache_alloc,
                "tri_ms_per_token": tri_ms,
                "transformer_ms_per_token": tf_ms,
                "reduction": cache_tensor_bytes / state_tensor_bytes,
            }
        )
        print(
            "Track4B L={:>5} state={:>6} B cache={:>9} B "
            "tri={:.3f} ms/token tf={:.3f} ms/token".format(
                length,
                state_tensor_bytes,
                cache_tensor_bytes,
                tri_ms,
                tf_ms,
            ),
            flush=True,
        )
        del state, caches, k_cache, v_cache
        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "device": device,
        "lengths": list(LENGTHS),
        "state_config": {"d_model": 128, "n_heads": 4, "d_head": 32},
        "baseline_config": {
            "n_layers": 2,
            "d_model": 128,
            "n_heads": 4,
            "d_head": 32,
        },
        "rows": rows,
    }


def format_mean_std(mean: float, std: float) -> str:
    return f"**{mean:.2f} ± {std:.2f}%**"


def build_v9_sections(track4a, track4b, runtime: float, device: str) -> str:
    summary = track4a["summary"]
    full = summary[0]
    best_other = max(summary[1:], key=lambda item: item["ood_mean"])
    gain = full["ood_mean"] - best_other["ood_mean"]
    max_row = max(track4b["rows"], key=lambda row: row["length"])
    crossover = next(
        (
            row["length"]
            for row in track4b["rows"]
            if row["transformer_ms_per_token"] > row["tri_ms_per_token"]
        ),
        None,
    )
    if crossover is None:
        latency_sentence = (
            "Across the measured range the cached Transformer remains faster "
            "per token than TriCore. V9 therefore reports a measured memory "
            "advantage, not a measured latency win."
        )
    else:
        crossover_row = next(
            row for row in track4b["rows"] if row["length"] == crossover
        )
        latency_sentence = (
            "The first measured per-token latency inversion appears at "
            f"`L={crossover:,}`: Transformer "
            f"`{crossover_row['transformer_ms_per_token']:.3f}` ms/token "
            "versus TriCore "
            f"`{crossover_row['tri_ms_per_token']:.3f}` ms/token. This is a "
            "single-machine B=1 observation, not a serving-throughput claim."
        )

    track4a_rows = "\n".join(
        "| {name} | {params:,} | {in_cell} | {ood_cell} |".format(
            name=item["name"],
            params=item["params"],
            in_cell=format_mean_std(item["in_mean"], item["in_std"]),
            ood_cell=format_mean_std(item["ood_mean"], item["ood_std"]),
        )
        for item in summary
    )
    per_seed_rows = "\n".join(
        "| {name} | {seed} | {in_acc:.2f}% | {ood_acc:.2f}% |".format(
            name=item["name"], **row
        )
        for item in summary
        for row in item["per_seed"]
    )
    track4b_rows = "\n".join(
        "| {length:,} | {state_tensor_bytes:,} | {state_alloc} | "
        "{state_kib:.2f} KiB | {cache_tensor_bytes:,} | {cache_alloc} | "
        "{cache_mib:.2f} MiB | **{reduction:.1f}x** | "
        "{tri_ms_per_token:.3f} | {transformer_ms_per_token:.3f} |".format(
            length=row["length"],
            state_tensor_bytes=row["state_tensor_bytes"],
            state_alloc=(
                f"{row['state_alloc_bytes']:,}"
                if row["state_alloc_bytes"]
                else "n/a"
            ),
            state_kib=row["state_tensor_bytes"] / 1024.0,
            cache_tensor_bytes=row["cache_tensor_bytes"],
            cache_alloc=(
                f"{row['cache_alloc_bytes']:,}"
                if row["cache_alloc_bytes"]
                else "n/a"
            ),
            cache_mib=row["cache_tensor_bytes"] / (1024.0 * 1024.0),
            reduction=row["reduction"],
            tri_ms_per_token=row["tri_ms_per_token"],
            transformer_ms_per_token=row["transformer_ms_per_token"],
        )
        for row in track4b["rows"]
    )
    return _build_v9_lines(
        track4a, track4b, summary, full, best_other, gain,
        max_row, track4a_rows, per_seed_rows, track4b_rows,
        latency_sentence, runtime, device,
    )


def _build_v9_lines(
    track4a, track4b, summary, full, best_other, gain, max_row,
    track4a_rows, per_seed_rows, track4b_rows, latency_sentence,
    runtime, device,
):
    lines = [
        V9_START,
        "## 8. Track 4A: 3-Seed Five-Way Ablation on 4-Chain 2-Hop MQAR",
        "",
        f"Measured on `{device.upper()}`.  Protocol: `num_kv_pairs=4` "
        "(4 chains, random baseline 25.0%), `num_queries=4`, training "
        "`noise_len=8`, 4x OOD `noise_len=32`, `d_model=64`, 4 heads, "
        "`d_fast=16`, `max_loops=2`; 350 steps, batch 64, AdamW "
        "`lr=3e-3`, weight decay 0.01, cosine decay, gradient clipping 1.0.  "
        "Seeds are `42/100/2026`; each cell is mean ± std over seeds on 5 "
        "fixed evaluation batches of 64 examples.  Accuracy is measured only "
        "at query positions where the target is not `-100`.",
        "",
        "| Model | Parameters | In-Domain Accuracy (Mean ± Std) | 4x Gap Accuracy (Mean ± Std) |",
        "| :--- | :---: | :---: | :---: |",
        track4a_rows,
        "",
        f"The Full model reaches **{full['in_mean']:.2f} ± {full['in_std']:.2f}%** "
        f"in-domain and **{full['ood_mean']:.2f} ± {full['ood_std']:.2f}%** at "
        "the 4x gap.  The best non-Full variant at the 4x gap is "
        f"`{best_other['name']}` at **{best_other['ood_mean']:.2f} ± "
        f"{best_other['ood_std']:.2f}%**, so the Full OOD advantage is "
        f"**{gain:+.2f} percentage points** on the seed mean.  A standard "
        "deviation of `0.00` means every seed produced the same query accuracy.  "
        "Three seeds are a variance check, not a significance test.",
        "",
        "### 8.1 Per-seed audit",
        "",
        "| Model | Seed | In-Domain | 4x Gap |",
        "| :--- | :---: | :---: | :---: |",
        per_seed_rows,
        "",
        "## 9. Track 4B: Real CUDA Decode Memory and Latency Scaling to 32,768",
        "",
        f"Measured on `{track4b['device'].upper()}` with batch size 1 and FP32.  "
        "For TriCore, the script allocates the native `init_state` tuple "
        "`(h, F_mat, prev_k)` and records both tensor bytes and the "
        "`torch.cuda.memory_allocated()` delta after `empty_cache()`.  It runs "
        "8 warm-up steps plus 32 timed `forward_step` calls at each length.  "
        "For the baseline, it explicitly allocates `(K_cache, V_cache)` with "
        "shape `(n_layers, L, 1, n_heads, d_head)` and performs a real cached "
        "decode step through both Transformer layers with scaled dot-product "
        "attention over the full causal cache at position `L-1`; the same "
        "8 + 32 timing protocol is used.  This is measured allocation and "
        "measured latency, not a shape-only estimate.",
        "",
        "| Sequence Length | TriCore State Bytes | CUDA Alloc Δ (B) | State | Transformer K+V Bytes | CUDA Alloc Δ (B) | Cache | Reduction | TriCore ms/token | Transformer ms/token |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        track4b_rows,
        "",
        "**Accounting.** The TriCore state is "
        "`(D + H*d_head^2 + H*d_head) * 4 = 17,408` bytes = **17.00 KiB per "
        "sample**, independent of `L` (`D=128`, `H=4`, `d_head=32`).  The "
        "baseline cache is `2 * n_layers * L * D * 4` bytes; with `n_layers=2` "
        "and `D=128` it grows by **2.00 KiB per token**.  At "
        f"`L={max_row['length']:,}`, the measured Tensor allocation is "
        f"**{max_row['cache_tensor_bytes']:,} bytes = "
        f"{max_row['cache_tensor_bytes'] / (1024.0 * 1024.0):.2f} MiB** versus "
        f"**{max_row['state_tensor_bytes']:,} bytes = "
        f"{max_row['state_tensor_bytes'] / 1024.0:.2f} KiB**, a "
        f"**{max_row['reduction']:.1f}x** reduction.  "
        + latency_sentence,
        "",
        "## 10. V9 Zero-`F_seq` Training Path and Reproduction",
        "",
        "V9 moves Engine III's fixed multi-hop re-query into the "
        "`@torch.jit.script tricore_fused_scan` causal loop.  The scan now "
        "returns `(h_seq, mem_seq, surprise_trace)` and never allocates or "
        "`torch.stack`s `F_seq: [B, L, H, d_head, d_head]`; the only "
        "sequence-wide memory read is `mem_seq: [B, L, H*d_head]`.  All module "
        "parameter names, equations, `init_state`, and `forward_step` are "
        "unchanged.  `tests/test_v5.py` checks the JIT scan against a Python "
        "reference for the ablation flags and checks full `forward` against "
        "token-by-token `forward_step` with a **< 1e-6** maximum absolute "
        "error.",
        "",
        "```bash",
        "python -m pytest -q",
        "python run_v9_verification.py",
        "python demo_showcase.py",
        "```",
        "",
        f"Measured V9 script runtime on `{device.upper()}`: **{runtime:.1f} s** "
        "(Track 4A + Track 4B, including data generation and evaluation).",
        V9_END,
    ]
    return "\n".join(lines)


def write_report(track4a, track4b, runtime: float, device: str) -> None:
    existing = REPORT_PATH.read_text(encoding="utf-8")
    # V9 supersedes the V7 training-path statement; keep the V6/V7 numbers but
    # make the historical scope explicit.
    if "## 6. V7 Scope (historical" not in existing:
        existing = existing.replace(
            "## 6. V7 Scope",
            "## 6. V7 Scope (historical; see V9 in section 10)",
            1,
        )
    existing = existing.replace(
        "The training path still materializes `F_seq` for vectorized multi-hop "
        "updates; only the persistent inference state is O(1).",
        "V7 materialized `F_seq` for vectorized multi-hop updates; V9 removes "
        "that allocation (section 10). Only the persistent inference state is O(1).",
    )
    existing = existing.replace(
        "- `F_seq` is still created during training to vectorize Engine III; "
        "the O(1) claim applies to persistent autoregressive state.",
        "- V7 created `F_seq` during training to vectorize Engine III; V9 "
        "removes it (section 10). The O(1) claim applies to persistent state.",
    )
    for start, end in ((V8_START, V8_END), (V9_START, V9_END)):
        pattern = re.compile(
            r"\n*" + re.escape(start) + r".*?" + re.escape(end) + r"\n*",
            flags=re.DOTALL,
        )
        existing = pattern.sub("\n\n", existing)
    existing = existing.rstrip()
    existing = re.sub(
        r"^# TriCore V\d+ Experimental Benchmark Report$",
        "# TriCore V9 Experimental Benchmark Report",
        existing,
        count=1,
        flags=re.MULTILINE,
    )

    new_track4a = (
        "- **Track 4A:** 4-chain 2-Hop MQAR, full five-way ablation; 350 "
        "steps, batch 64, lr 3e-3; seeds 42/100/2026; `d_model=64`, "
        "`d_fast=16`, 4 queries, train `noise_len=8` and 4x OOD "
        "`noise_len=32`; Mean ± Std (%)."
    )
    new_track4b = (
        "- **Track 4B:** Real CUDA decode-state/KV-cache allocation and "
        "latency at `L=128/512/2,048/8,192/16,384/32,768`; measured tensor "
        "bytes, allocator delta, and ms/token."
    )
    new_demo = (
        "- **V9 Demo:** `python demo_showcase.py` streams a three-hop chain "
        "through `forward_step` with a live surprise-gate and 17.00 KiB "
        "memory dashboard."
    )

    def upsert(label, replacement):
        nonlocal existing
        pattern = re.compile(
            r"^- \*\*" + re.escape(label) + r":\*\*.*$",
            flags=re.MULTILINE,
        )
        if pattern.search(existing):
            existing = pattern.sub(replacement, existing, count=1)
        else:
            existing = re.sub(
                r"(^- \*\*Track 3B:\*\*.*$)",
                r"\1\n" + replacement,
                existing,
                count=1,
                flags=re.MULTILINE,
            )

    upsert("Track 4A", new_track4a)
    upsert("Track 4B", new_track4b)
    if re.search(r"^- \*\*(?:V8|V9) Demo:\*\*", existing, flags=re.MULTILINE):
        existing = re.sub(
            r"^- \*\*(?:V8|V9) Demo:\*\*.*$",
            new_demo,
            existing,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        existing = re.sub(
            r"(^- \*\*Track 4B:\*\*.*$)",
            r"\1\n" + new_demo,
            existing,
            count=1,
            flags=re.MULTILINE,
        )

    block = build_v9_sections(track4a, track4b, runtime, device)
    REPORT_PATH.write_text(
        existing.rstrip() + "\n\n" + block + "\n", encoding="utf-8"
    )


def main():
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TriCore V9 verification on {device.upper()} ===", flush=True)
    start = time.perf_counter()

    torch.manual_seed(42)
    track4a = run_track4a(device)
    track4a_time = time.perf_counter() - start
    print(f"Track4A complete in {track4a_time:.1f}s", flush=True)

    mark = time.perf_counter()
    track4b = run_track4b(device)
    track4b_time = time.perf_counter() - mark
    runtime = time.perf_counter() - start
    print(f"Track4B complete in {track4b_time:.1f}s", flush=True)

    write_report(track4a, track4b, runtime, device)
    metrics = {
        "version": "1.2.0",
        "device": device,
        "runtime_seconds": runtime,
        "track4a_seconds": track4a_time,
        "track4b_seconds": track4b_time,
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "d_model": D_MODEL,
            "d_fast": D_FAST,
            "n_chains": N_CHAINS,
            "n_queries": N_QUERIES,
            "noise_len": NOISE_LEN,
            "ood_multiplier": OOD_MULTIPLIER,
            "ood_noise_len": NOISE_LEN * OOD_MULTIPLIER,
            "eval_batches": EVAL_BATCHES,
            "lengths": list(LENGTHS),
        },
        "track4a": track4a,
        "track4b": track4b,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {REPORT_PATH} and {METRICS_PATH}", flush=True)
    print(f"Total V9 verification runtime: {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        Path("results/V9_ERROR.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise





