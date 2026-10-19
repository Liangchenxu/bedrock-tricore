import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tricore import TriCoreLM, BaselineTransformerLM, generate_associative_recall_batch


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model: nn.Module, vocab_size: int, device: str, noise_len: int = 16, batches: int = 10):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = generate_associative_recall_batch(
                batch_size=32, num_kv_pairs=8, num_queries=4, vocab_size=vocab_size, noise_len=noise_len, device=device
            )
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            total_loss += loss.item()

            mask = y != -100
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return total_loss / batches, (correct / max(total, 1)) * 100.0


def train_and_benchmark():
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    vocab_size = 64
    steps = 600
    batch_size = 64
    lr = 3e-3

    configs = {
        "TriCore (Full 3-in-1)": lambda: TriCoreLM(
            vocab_size=vocab_size, d_model=128, n_layers=2, d_fast=32, max_loops=2,
            enable_fast_weight=True, enable_adaptive_loop=True
        ),
        "TriCore (No Fast-Weight)": lambda: TriCoreLM(
            vocab_size=vocab_size, d_model=128, n_layers=2, d_fast=32, max_loops=2,
            enable_fast_weight=False, enable_adaptive_loop=True
        ),
        "TriCore (No Adaptive Loop)": lambda: TriCoreLM(
            vocab_size=vocab_size, d_model=128, n_layers=2, d_fast=32, max_loops=2,
            enable_fast_weight=True, enable_adaptive_loop=False
        ),
        "Vanilla Transformer": lambda: BaselineTransformerLM(
            vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4
        ),
    }

    history = {}
    summary_stats = {}

    print(f"=== Starting TriCore Architecture Benchmark on {device.upper()} ===")

    for name, builder in configs.items():
        torch.manual_seed(42)
        model = builder().to(device)
        params = count_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        losses = []
        accs = []
        start_time = time.time()

        for step in range(1, steps + 1):
            x, y = generate_associative_recall_batch(
                batch_size=batch_size, num_kv_pairs=8, num_queries=4, vocab_size=vocab_size, noise_len=16, device=device
            )
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0 or step == 1:
                val_loss, val_acc = evaluate_model(model, vocab_size, device, noise_len=16, batches=5)
                losses.append((step, val_loss))
                accs.append((step, val_acc))
                print(f"[{name}] Step {step:03d}/{steps} | Val Loss: {val_loss:.4f} | Recall Acc: {val_acc:.1f}%")

        elapsed = time.time() - start_time
        # Zero-shot Length Extrapolation Test (Noise length 4x longer than training!)
        _, extrap_acc = evaluate_model(model, vocab_size, device, noise_len=64, batches=10)

        history[name] = {"losses": losses, "accs": accs}
        summary_stats[name] = {
            "params": params,
            "final_loss": losses[-1][1],
            "final_acc": accs[-1][1],
            "extrap_acc": extrap_acc,
            "train_time": elapsed,
        }

    # --- Theoretical State Cache Footprint Comparison (KB per sample at FP32) ---
    seq_lengths = [128, 512, 2048, 8192, 32768]
    d_model, n_layers, d_fast = 128, 2, 32
    tricore_state_kb = [
        (n_layers * (d_model + 4 * d_fast * d_fast) * 4) / 1024.0 for _ in seq_lengths
    ]
    transformer_kv_kb = [
        (n_layers * 2 * L * d_model * 4) / 1024.0 for L in seq_lengths
    ]

    # --- Plot Results ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, data in history.items():
        xs = [pt[0] for pt in data["losses"]]
        ys = [pt[1] for pt in data["losses"]]
        axes[0].plot(xs, ys, label=name, linewidth=2)
    axes[0].set_title("MQAR Validation Loss")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, data in history.items():
        xs = [pt[0] for pt in data["accs"]]
        ys = [pt[1] for pt in data["accs"]]
        axes[1].plot(xs, ys, label=name, linewidth=2)
    axes[1].set_title("Associative Recall Accuracy (%)")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(seq_lengths, transformer_kv_kb, marker="o", label="Transformer KV Cache O(L)", linewidth=2, color="crimson")
    axes[2].plot(seq_lengths, tricore_state_kb, marker="s", label="TriCore State + FastWeight O(1)", linewidth=2, color="forestgreen")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log", base=10)
    axes[2].set_title("Inference Memory Cache Scaling (Log-Log)")
    axes[2].set_xlabel("Sequence Length (L)")
    axes[2].set_ylabel("Cache Memory per Sample (KB)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/tricore_vs_transformer.png", dpi=160)
    print("\nSaved benchmark chart to results/tricore_vs_transformer.png")

    # --- Write Markdown Report ---
    report_lines = [
        "# TriCore Experimental Benchmark Report\n",
        f"- **Compute Device:** `{device.upper()}`\n",
        "- **Task:** Multi-Query Associative Recall (MQAR) with Distractor Noise & 4x Length Extrapolation\n\n",
        "## 1. Accuracy & Ablation Results\n",
        "| Architecture Variant | Parameters | Final Val Loss | In-Domain Recall Acc (%) | 4x Length Extrapolation Acc (%) | Train Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, st in summary_stats.items():
        report_lines.append(
            f"| **{name}** | {st['params']:,} | {st['final_loss']:.4f} | **{st['final_acc']:.2f}%** | **{st['extrap_acc']:.2f}%** | {st['train_time']:.1f}s |"
        )

    report_lines.extend([
        "\n## 2. Inference State Cache Scaling (Per Sample)\n",
        "| Sequence Length | Transformer KV Cache (KB) | TriCore O(1) State (KB) | Memory Reduction Factor |",
        "| :---: | :---: | :---: | :---: |",
    ])
    for L, tf_kb, tc_kb in zip(seq_lengths, transformer_kv_kb, tricore_state_kb):
        ratio = tf_kb / tc_kb
        report_lines.append(f"| {L:,} | {tf_kb:.2f} KB | {tc_kb:.2f} KB | **{ratio:.1f}x smaller** |")

    with open("results/BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("Saved detailed report to results/BENCHMARK_REPORT.md")


if __name__ == "__main__":
    train_and_benchmark()
