import argparse
import math
import os
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tricore import (
    BaselineTransformerLM,
    TriCoreLM,
    generate_associative_recall_batch,
    generate_contextual_code_lm_batch,
)


V4_FULL_TRAIN_TIME = 136.0
V5_FULL_TRAIN_TIME = 125.3
CODE_VOCAB_SIZE = 64
CODE_NUM_VARS = 6
CODE_NUM_QUERIES = 4
CODE_NUM_NOPS = 4


def sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mqar_eval_batches(
    vocab_size: int,
    device: str,
    noise_len: int,
    batches: int,
    seed: int,
):
    torch.manual_seed(seed)
    result = []
    for _ in range(batches):
        result.append(
            generate_associative_recall_batch(
                batch_size=32,
                num_kv_pairs=6,
                num_queries=4,
                vocab_size=vocab_size,
                noise_len=noise_len,
                device=device,
            )
        )
    return result


def build_mqar_eval_sets(vocab_size: int, device: str, batches: int):
    return {
        16: build_mqar_eval_batches(vocab_size, device, 16, batches, seed=2026),
        64: build_mqar_eval_batches(vocab_size, device, 64, batches, seed=2027),
    }


def evaluate_mqar(model: nn.Module, eval_batches, vocab_size: int):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in eval_batches:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            total_loss += loss.item()
            mask = y != -100
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return total_loss / max(len(eval_batches), 1), (correct / max(total, 1)) * 100.0


def train_mqar_config(
    name: str,
    builder,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    eval_sets,
    seed: int,
):
    torch.manual_seed(seed)
    model = builder().to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))

    warmup_x, warmup_y = generate_associative_recall_batch(
        batch_size=batch_size, num_kv_pairs=6, num_queries=4,
        vocab_size=64, noise_len=16, device=device,
    )
    sync_device(device)
    warmup_start = time.perf_counter()
    warmup_logits, _ = model(warmup_x)
    warmup_loss = F.cross_entropy(
        warmup_logits.view(-1, 64), warmup_y.view(-1), ignore_index=-100
    )
    warmup_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    sync_device(device)
    warmup_time = time.perf_counter() - warmup_start

    loss_curve = []
    sync_device(device)
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = generate_associative_recall_batch(
            batch_size=batch_size, num_kv_pairs=6, num_queries=4,
            vocab_size=64, noise_len=16, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0 or step == steps:
            loss_curve.append((step, loss.item()))
    sync_device(device)
    train_time = time.perf_counter() - start_time

    val_loss, val_acc = evaluate_mqar(model, eval_sets[16], 64)
    extrap_loss, extrap_acc = evaluate_mqar(model, eval_sets[64], 64)

    result = {
        "name": name,
        "params": params,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "extrap_loss": extrap_loss,
        "extrap_acc": extrap_acc,
        "train_time": train_time,
        "warmup_time": warmup_time,
        "loss_curve": loss_curve,
    }
    print(
        f"[Track 1] {name}: loss={val_loss:.4f}, acc={val_acc:.2f}%, "
        f"4x={extrap_acc:.2f}%, time={train_time:.1f}s, warmup={warmup_time:.1f}s"
    )
    return result


def build_track1_configs(vocab_size: int):
    return {
        "TriCore Full (3-in-1)": lambda: TriCoreLM(
            vocab_size=vocab_size, d_model=128, n_layers=1, d_fast=32, max_loops=2,
            enable_surprise_gate=True, enable_fast_weight=True, enable_adaptive_loop=True,
        ),
        "TriCore (No Engine I: Surprise Gate)": lambda: TriCoreLM(
            vocab_size=vocab_size, d_model=128, n_layers=1, d_fast=32, max_loops=2,
            enable_surprise_gate=False, enable_fast_weight=True, enable_adaptive_loop=True,
        ),
        "TriCore (No Engine II: Fast-Weight)": lambda: TriCoreLM(
            vocab_size=vocab_size, d_model=128, n_layers=1, d_fast=32, max_loops=2,
            enable_surprise_gate=True, enable_fast_weight=False, enable_adaptive_loop=True,
        ),
        "TriCore (No Engine III: Adaptive Loop)": lambda: TriCoreLM(
            vocab_size=vocab_size, d_model=128, n_layers=1, d_fast=32, max_loops=2,
            enable_surprise_gate=True, enable_fast_weight=True, enable_adaptive_loop=False,
        ),
        "Vanilla Transformer (2-Layer)": lambda: BaselineTransformerLM(
            vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4,
        ),
    }


def code_sequence_length() -> int:
    return (CODE_NUM_VARS * 2 + CODE_NUM_NOPS + CODE_NUM_QUERIES) * 4


def build_code_eval_batches(device: str, batch_size: int, batches: int, seed: int):
    torch.manual_seed(seed)
    result = []
    for _ in range(batches):
        result.append(
            generate_contextual_code_lm_batch(
                batch_size=batch_size,
                num_vars=CODE_NUM_VARS,
                num_queries=CODE_NUM_QUERIES,
                num_nops=CODE_NUM_NOPS,
                vocab_size=CODE_VOCAB_SIZE,
                device=device,
            )
        )
    return result


def evaluate_code_lm(model: nn.Module, eval_batches):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, deref_mask in eval_batches:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, CODE_VOCAB_SIZE), y.view(-1), ignore_index=-100
            )
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds[deref_mask] == y[deref_mask]).sum().item()
            total += deref_mask.sum().item()
    model.train()
    mean_loss = total_loss / max(len(eval_batches), 1)
    ppl = math.exp(mean_loss)
    deref_acc = (correct / max(total, 1)) * 100.0
    return mean_loss, ppl, deref_acc


def train_code_lm_config(
    name: str,
    builder,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    eval_batches,
    seed: int,
):
    torch.manual_seed(seed)
    model = builder().to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))

    warmup_x, warmup_y, warmup_mask = generate_contextual_code_lm_batch(
        batch_size=batch_size, num_vars=CODE_NUM_VARS,
        num_queries=CODE_NUM_QUERIES, num_nops=CODE_NUM_NOPS,
        vocab_size=CODE_VOCAB_SIZE, device=device,
    )
    sync_device(device)
    warmup_start = time.perf_counter()
    warmup_logits, _ = model(warmup_x)
    warmup_dense = F.cross_entropy(
        warmup_logits.view(-1, CODE_VOCAB_SIZE), warmup_y.view(-1), ignore_index=-100
    )
    warmup_deref = F.cross_entropy(warmup_logits[warmup_mask], warmup_y[warmup_mask])
    warmup_loss = 0.25 * warmup_dense + warmup_deref
    warmup_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    sync_device(device)
    warmup_time = time.perf_counter() - warmup_start

    sync_device(device)
    start_time = time.perf_counter()
    for _ in range(1, steps + 1):
        x, y, deref_mask = generate_contextual_code_lm_batch(
            batch_size=batch_size, num_vars=CODE_NUM_VARS,
            num_queries=CODE_NUM_QUERIES, num_nops=CODE_NUM_NOPS,
            vocab_size=CODE_VOCAB_SIZE, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        dense_loss = F.cross_entropy(
            logits.view(-1, CODE_VOCAB_SIZE), y.view(-1), ignore_index=-100
        )
        deref_loss = F.cross_entropy(logits[deref_mask], y[deref_mask])
        loss = 0.25 * dense_loss + deref_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    sync_device(device)
    train_time = time.perf_counter() - start_time

    val_loss, ppl, deref_acc = evaluate_code_lm(model, eval_batches)
    result = {
        "name": name,
        "params": params,
        "val_loss": val_loss,
        "ppl": ppl,
        "deref_acc": deref_acc,
        "train_time": train_time,
        "warmup_time": warmup_time,
        "last_dense_loss": dense_loss.item(),
        "last_deref_loss": deref_loss.item(),
    }
    print(
        f"[Track 2] {name}: val_loss={val_loss:.4f}, ppl={ppl:.2f}, "
        f"deref={deref_acc:.2f}%, time={train_time:.1f}s, warmup={warmup_time:.1f}s"
    )
    return result


def build_track2_configs():
    return {
        "TriCore Full (1-Layer)": lambda: TriCoreLM(
            vocab_size=CODE_VOCAB_SIZE, d_model=128, n_layers=1, d_fast=32, max_loops=2,
            enable_surprise_gate=True, enable_fast_weight=True, enable_adaptive_loop=True,
        ),
        "Vanilla Transformer (2-Layer)": lambda: BaselineTransformerLM(
            vocab_size=CODE_VOCAB_SIZE, d_model=128, n_layers=2, n_heads=4,
        ),
    }


def cache_accounting():
    d_model = 128
    n_heads = 4
    d_head = 32
    n_layers = 2
    tricore_kb = (d_model + n_heads * d_head * d_head + n_heads * d_head) * 4 / 1024
    seq_lengths = [128, 512, 2048, 8192, 32768]
    transformer_kb = [
        n_layers * 2 * seq_len * d_model * 4 / 1024 for seq_len in seq_lengths
    ]
    return seq_lengths, transformer_kb, [tricore_kb for _ in seq_lengths]


def plot_results(track1, track2, cache_rows, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    names = [
        item["name"]
        .replace("TriCore ", "")
        .replace("Vanilla ", "")
        .replace(" (3-in-1)", "")
        for item in track1
    ]
    xs = list(range(len(names)))
    width = 0.38
    axes[0, 0].bar(
        [x - width / 2 for x in xs], [item["val_acc"] for item in track1],
        width=width, label="In-domain (gap=16)", color="steelblue",
    )
    axes[0, 0].bar(
        [x + width / 2 for x in xs], [item["extrap_acc"] for item in track1],
        width=width, label="4x extrapolation (gap=64)", color="darkorange",
    )
    axes[0, 0].set_xticks(xs)
    axes[0, 0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0, 0].set_ylim(0, 108)
    axes[0, 0].set_ylabel("Recall Accuracy (%)")
    axes[0, 0].set_title("Track 1: 2-Hop MQAR Ablation")
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    seq_lengths, transformer_kb, tricore_kb = cache_rows
    axes[0, 1].plot(
        seq_lengths, transformer_kb, marker="o",
        label="Transformer KV Cache O(L)", linewidth=2, color="crimson",
    )
    axes[0, 1].plot(
        seq_lengths, tricore_kb, marker="s",
        label="TriCore State O(1)", linewidth=2, color="forestgreen",
    )
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log", base=10)
    axes[0, 1].set_title("Inference Cache Scaling (Per Sample)")
    axes[0, 1].set_xlabel("Sequence Length (L)")
    axes[0, 1].set_ylabel("Cache Memory (KB)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, which="both", alpha=0.3)

    code_names = [item["name"].replace("TriCore ", "") for item in track2]
    code_xs = list(range(len(code_names)))
    ppl_values = [max(item["ppl"], 1e-9) for item in track2]
    bars = axes[1, 0].bar(code_xs, ppl_values, color=["forestgreen", "crimson"])
    axes[1, 0].set_xticks(code_xs)
    axes[1, 0].set_xticklabels(code_names, fontsize=9)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Validation Perplexity (log scale)")
    axes[1, 0].set_title("Track 2: Contextual Code LM")
    for bar, item in zip(bars, track2):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08,
            f"{item['ppl']:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    deref_values = [item["deref_acc"] for item in track2]
    bars = axes[1, 1].bar(code_xs, deref_values, color=["forestgreen", "crimson"])
    axes[1, 1].axhline(
        100.0 / CODE_NUM_VARS, color="gray", linestyle="--",
        label=f"Chance = {100.0 / CODE_NUM_VARS:.2f}%",
    )
    axes[1, 1].set_xticks(code_xs)
    axes[1, 1].set_xticklabels(code_names, fontsize=9)
    axes[1, 1].set_ylim(0, 108)
    axes[1, 1].set_ylabel("Dereference Accuracy (%)")
    axes[1, 1].set_title("Track 2: Alias Dereference Accuracy")
    for bar, value in zip(bars, deref_values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.2f}%",
            ha="center", va="bottom", fontsize=9,
        )
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("TriCore V6: Vectorized Multi-Hop Scan, Dual-Track Benchmark", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved benchmark chart to {path}")


def write_report(track1, track2, cache_rows, path: str, device: str) -> None:
    full = track1[0]
    full_time = max(full["train_time"], 1e-9)
    v4_speedup = V4_FULL_TRAIN_TIME / full_time
    v5_speedup = V5_FULL_TRAIN_TIME / full_time
    seq_lengths, transformer_kb, tricore_kb = cache_rows
    code_len = code_sequence_length()
    chance = 100.0 / CODE_NUM_VARS
    packaging = all(os.path.exists(p) for p in ("LICENSE", ".gitignore", "pyproject.toml"))

    lines = [
        "# TriCore V6 Experimental Benchmark Report\n",
        f"- **Compute Device:** `{device.upper()}`",
        "- **Operator:** TorchScript JIT recurrent scan + vectorized parallel multi-hop Engine III",
        "- **Track 1:** 2-Hop Chained Associative Recall with adversarial distractor gap; "
        "500 optimizer steps, batch 64, lr 3e-3, seed 42; 6 chains, 4 queries, 16 training noise tokens.",
        f"- **Track 2:** Causal-bytecode Contextual Code LM; 450 optimizer steps, batch 64, lr 3e-3, seed 42; "
        f"{CODE_NUM_VARS} variables, {CODE_NUM_NOPS} NOP statements, sequence length {code_len}; "
        "joint loss `0.25 * dense_loss + deref_loss`.",
        f"- **Open-source packaging:** `{'ready' if packaging else 'pending'}` "
        "(MIT LICENSE, .gitignore, pyproject.toml)\n",
        "## 1. JIT Fusion and Track 1 Ablation\n",
        "### 1.1 JIT timing versus V4 and V5\n",
        "| Measurement | Time | Note |",
        "| :--- | :---: | :--- |",
        f"| V4 Full, Python per-token loop | {V4_FULL_TRAIN_TIME:.1f} s | reference run |",
        f"| V5 Full, TorchScript fused scan | {V5_FULL_TRAIN_TIME:.1f} s | reference run |",
        f"| V6 Full, vectorized multi-hop scan | {full['train_time']:.1f} s | timed 500 steps |",
        f"| V6 / V4 speed ratio | {v4_speedup:.2f}x | >1 means faster; <1 means slower |",
        f"| V6 / V5 speed ratio | {v5_speedup:.2f}x | >1 means faster; <1 means slower |",
        f"| V6 first-call warm-up | {full['warmup_time']:.1f} s | excluded from timed training |",
        "",
        "The comparison uses the previously recorded V4 Full time of 136.0 s and V5 Full time of 125.3 s. "
        "The warm-up call is excluded from the timed V6 run; timed training includes data generation, "
        "forward, backward, and optimizer steps.\n",
        "### 1.2 Five-way ablation\n",
        "| Architecture Variant | Parameters | Final Val Loss | In-Domain 2-Hop Recall (%) | 4x Noise-Gap Extrapolation (%) | Train Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for item in track1:
        lines.append(
            f"| {item['name']} | {item['params']:,} | {item['val_loss']:.4f} | "
            f"**{item['val_acc']:.2f}%** | **{item['extrap_acc']:.2f}%** | {item['train_time']:.1f} |"
        )

    lines.extend([
        "",
        "## 2. Track 2: Contextual Code Language Modeling\n",
        "| Model | Parameters | Validation Loss | Validation PPL | Dereference Acc (%) | Train Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for item in track2:
        lines.append(
            f"| {item['name']} | {item['params']:,} | {item['val_loss']:.4f} | "
            f"{item['ppl']:.2f} | **{item['deref_acc']:.2f}%** | {item['train_time']:.1f} |"
        )

    lines.extend([
        f"\n> Track 2 chance-level dereference accuracy is **{chance:.2f}%** "
        f"({CODE_NUM_VARS} equally likely literals). The metric is measured at the return alias position. "
        "The task is dense next-token prediction over a synthetic imperative grammar; it is not human code.\n",
        "## 3. Inference State Cache Scaling (Per Sample)\n",
        "| Sequence Length | Transformer KV Cache (KB) | TriCore O(1) State (KB) | Memory Reduction Factor |",
        "| :---: | :---: | :---: | :---: |",
    ])
    for seq_len, tf_kb, tc_kb in zip(seq_lengths, transformer_kb, tricore_kb):
        lines.append(
            f"| {seq_len:,} | {tf_kb:.2f} KB | {tc_kb:.2f} KB | **{tf_kb / tc_kb:.1f}x smaller** |"
        )

    lines.extend([
        "\n**State accounting.** The single-core TriCore figure is "
        "`(D + H*d_head^2 + H*d_head) * 4 / 1024` KiB per sample at FP32, "
        "where `D=128`, `H=4`, and `d_head=32`. It includes `h`, `F_mat`, and the causal-shift key cache "
        "`prev_k`, and is independent of sequence length. The Transformer figure is "
        "`n_layers * 2 * L * D * 4 / 1024` KiB with `n_layers=2`.\n",
        "**Metric note.** Track 1 recall is measured only at query positions where the target is not `-100`. "
        "The 4x extrapolation multiplies the adversarial distractor gap (16 -> 64 tokens), not the total "
        "sequence length. Track 1 uses fixed evaluation batches generated with seed 2026/2027; Track 2 uses "
        "fixed evaluation batches generated with seed 3030. There is no fixed real-world held-out set.\n",
        "**Scope.** These are single-seed synthetic experiments. They do not establish natural-language "
        "ability, tuned-Transformer parity, multi-seed confidence intervals, or production throughput.\n",
    ])

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(f"Saved detailed report to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="TriCore V6 dual-track benchmark")
    parser.add_argument("--steps", type=int, default=500, help="Track 1 optimizer steps")
    parser.add_argument("--code-steps", type=int, default=450, help="Track 2 optimizer steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Track 1 batch size")
    parser.add_argument("--code-batch-size", type=int, default=64, help="Track 2 batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Track 1 learning rate")
    parser.add_argument("--code-lr", type=float, default=3e-3, help="Track 2 learning rate")
    parser.add_argument("--eval-batches", type=int, default=10, help="Track 1 eval batches")
    parser.add_argument("--code-eval-batches", type=int, default=8, help="Track 2 eval batches")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    parser.add_argument("--report", default="results/BENCHMARK_REPORT.md")
    parser.add_argument("--plot", default="results/tricore_vs_transformer.png")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    if args.device is None or args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"=== TriCore V6 dual-track benchmark on {device.upper()} ===")

    eval_sets = build_mqar_eval_sets(64, device, args.eval_batches)
    code_eval = build_code_eval_batches(
        device, args.code_batch_size, args.code_eval_batches, seed=3030
    )
    cache_rows = cache_accounting()

    track1 = []
    for name, builder in build_track1_configs(64).items():
        track1.append(
            train_mqar_config(
                name, builder, device, args.steps, args.batch_size, args.lr,
                eval_sets, args.seed,
            )
        )

    track2 = []
    for name, builder in build_track2_configs().items():
        track2.append(
            train_code_lm_config(
                name, builder, device, args.code_steps, args.code_batch_size,
                args.code_lr, code_eval, args.seed,
            )
        )

    plot_results(track1, track2, cache_rows, args.plot)
    write_report(track1, track2, cache_rows, args.report, device)
    print("V6 benchmark complete.")


if __name__ == "__main__":
    main()
