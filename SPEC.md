> **Archived V1 blueprint.** This document predates TriCore V3. The executable V3 contract is `SPEC_V2.md` and `tricore/model.py`.

Project Specification: TriCore Architecture (tricore-arch)1. Agent Execution Rules (CRITICAL FOR CLINE / DEEPSEEK)You are the Automated Engineering Executor for the TriCore open-source foundational architecture project.DO NOT simplify, stub out, or alter the mathematical equations or tensor operations inside tricore/model.py. Every line of the TriCoreBlock has been mathematically calibrated for gradient stability.EXECUTE STEP-BY-STEP:Create the exact directory structure and files specified below.Set up or verify the Python environment (torch, matplotlib, numpy).Run the unit test script tests/test_sanity.py via terminal and verify gradient flow and $O(1)$ state shapes.Run the full benchmark suite run_experiment.py via terminal and wait for completion.Inspect the generated results/BENCHMARK_REPORT.md and results/tricore_vs_transformer.png. If any environment or device error occurs, fix the infrastructure issue WITHOUT weakening the model architecture.2. Repository Directory StructureCreate the following exact file tree:Plaintexttricore-arch/
├── tricore/
│   ├── __init__.py
│   ├── model.py          # Core TriCore operator, TriCoreLM, and BaselineTransformerLM
│   └── tasks.py          # Synthetic Associative Recall (MQAR) & Long-Range Extrapolation data generator
├── tests/
│   └── test_sanity.py    # Unit tests for forward pass, backward gradients, and O(1) state memory
├── results/              # Auto-generated plots and markdown experiment reports
├── run_experiment.py     # Automated training, 4-way ablation study, and VRAM/latency profiler
├── requirements.txt
└── README.md             # Open-source documentation with mathematical formulation and benchmark results
3. File Contents to Generate ExactlyFile 1: requirements.txtPlaintexttorch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
File 2: tricore/__init__.pyPythonfrom .model import TriCoreBlock, TriCoreLM, BaselineTransformerLM
from .tasks import generate_associative_recall_batch

__all__ = [
    "TriCoreBlock",
    "TriCoreLM",
    "BaselineTransformerLM",
    "generate_associative_recall_batch",
]
File 3: tricore/model.pyPythonimport math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriCoreBlock(nn.Module):
    """
    TriCore: A Unified Sub-Quadratic Operator combining:
      1. Engine I: Surprise-Gated State Collapse (Entropy-driven linear state compression)
      2. Engine II: Forward Fast-Weight Synaptic Plasticity (Delta-rule test-time weight adaptation)
      3. Engine III: Adaptive Recurrent Equilibrium Loop (Dynamic compute allocation per token)
    """
    def __init__(self, d_model: int = 128, d_fast: int = 32, max_loops: int = 3, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_fast = d_fast
        self.max_loops = max_loops

        self.in_norm = nn.LayerNorm(d_model)

        # Causal 1D Short Convolution to capture local token transitions (e.g., Key -> Value pairs)
        self.short_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_model,
        )

        # --- Engine I: Surprise-Gated State Collapse ---
        self.state_predictor = nn.Linear(d_model, d_model)
        self.surprise_proj = nn.Linear(d_model * 2, d_model)
        self.base_decay_logit = nn.Parameter(torch.linspace(1.5, 3.0, d_model))

        # --- Engine II: Forward Fast-Weight Plasticity (Delta Rule) ---
        self.proj_q = nn.Linear(d_model, d_fast, bias=False)
        self.proj_k = nn.Linear(d_model, d_fast, bias=False)
        self.proj_v = nn.Linear(d_model, d_fast, bias=False)
        self.proj_out = nn.Linear(d_fast, d_model, bias=False)
        self.fast_lr_logit = nn.Parameter(torch.zeros(1, d_fast, 1))
        self.fast_decay_logit = nn.Parameter(torch.ones(1, d_fast, 1) * 2.5)
        self.fast_norm = nn.LayerNorm(d_fast)

        # --- Engine III: Adaptive Nonlinear Equilibrium Loop ---
        self.think_norm = nn.LayerNorm(d_model)
        self.think_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
    ):
        """
        Args:
            x: Input tensor of shape [Batch, Seq_Len, d_model]
        Returns:
            out: Output tensor of shape [Batch, Seq_Len, d_model]
            surprise_trace: Tensor of shape [Batch, Seq_Len] tracking mean surprise
        """
        B, L, D = x.shape
        device = x.device

         residual = x
        u = self.in_norm(x)

        # Causal short convolution for local temporal context
        u_conv = self.short_conv(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        # Pre-compute Fast-Weight Q, K, V projections
        # Key shifted interaction: K blends previous token context via u_conv, V captures current target
        Q = F.normalize(self.proj_q(u), p=2, dim=-1)
        K = F.normalize(self.proj_k(u_conv), p=2, dim=-1)
        V = self.proj_v(u_conv)

        base_decay = torch.sigmoid(self.base_decay_logit)           # (D,)
        fast_decay = torch.sigmoid(self.fast_decay_logit)           # (1, d_fast, 1)
        fast_lr = torch.sigmoid(self.fast_lr_logit) * 0.8           # Bounded in (0, 0.8) for stability

        # Persistent O(1) states across time
        h = torch.zeros(B, D, device=device, dtype=x.dtype)
        F_mat = torch.zeros(B, self.d_fast, self.d_fast, device=device, dtype=x.dtype)

        outputs = []
        surprise_list = []

        for t in range(L):
            u_t = u_conv[:, t, :]  # [B, D]

            # 1. Engine I: Surprise-Gated State Collapse
            pred_u = self.state_predictor(h)
            pred_error = u_t - pred_u
            g_t = torch.sigmoid(self.surprise_proj(torch.cat([u_t, torch.abs(pred_error)], dim=-1)))

            # Dynamic interpolation: low surprise decays smoothly, high surprise injects new feature
            h = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_t
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)  # [B, 1]
            surprise_list.append(surprise_scalar)

            # 2. Engine II: Forward Synaptic Fast-Weight Update (Delta Rule)
            if enable_fast_weight:
                k_t = K[:, t, :].unsqueeze(-1)  # [B, d_fast, 1]
                v_t = V[:, t, :].unsqueeze(-1)  # [B, d_fast, 1]
                q_t = Q[:, t, :].unsqueeze(-1)  # [B, d_fast, 1]

                # Recall first using current query (allows causal retrieval before overwriting)
                recalled_mem = torch.bmm(F_mat, q_t).squeeze(-1)  # [B, d_fast]
                recalled_mem = self.fast_norm(recalled_mem)

                # Synaptic prediction error & Delta update gated by surprise
                v_pred = torch.bmm(F_mat, k_t)
                synapse_err = v_t - v_pred
                effective_lr = fast_lr * (0.2 + 0.8 * surprise_scalar.unsqueeze(-1))

                F_mat = fast_decay * F_mat + torch.bmm(effective_lr * synapse_err, k_t.transpose(1, 2))
                z = h + self.proj_out(recalled_mem)
            else:
                z = h

            # 3. Engine III: Surprise-Driven Adaptive Equilibrium Loop
            if enable_adaptive_loop:
                for step in range(self.max_loops):
                    step_weight = surprise_scalar / float(step + 1)
                    z = z + step_weight * self.think_mlp(self.think_norm(z))

            outputs.append(z.unsqueeze(1))

        out = torch.cat(outputs, dim=1)
        out = residual + self.out_norm(out)
        surprise_trace = torch.cat(surprise_list, dim=1)
        return out, surprise_trace


class TriCoreLM(nn.Module):
    """Language / Sequence Model built exclusively on TriCore blocks (Zero Attention, No Positional Embeddings needed)."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_fast: int = 32,
        max_loops: int = 2,
        enable_fast_weight: bool = True,
        enable_adaptive_loop: bool = True,
    ):
        super().__init__()
        self.enable_fast_weight = enable_fast_weight
        self.enable_adaptive_loop = enable_adaptive_loop
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TriCoreBlock(d_model=d_model, d_fast=d_fast, max_loops=max_loops)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        x = self.embedding(idx)
        surprises = []
        for layer in self.layers:
            x, s = layer(
                x,
                enable_fast_weight=self.enable_fast_weight,
                enable_adaptive_loop=self.enable_adaptive_loop,
            )
            surprises.append(s)
        logits = self.lm_head(self.final_norm(x))
        mean_surprise = torch.stack(surprises, dim=0).mean(dim=0)
        return logits, mean_surprise


class BaselineTransformerLM(nn.Module):
    """Standard Causal Transformer Baseline with matched parameter scale for scientific benchmarking."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, max_seq_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, L = idx.shape
        positions = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.embedding(idx) + self.pos_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=idx.device)
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        logits = self.lm_head(self.final_norm(x))
        return logits, None
File 4: tricore/tasks.pyPythonimport torch


def generate_associative_recall_batch(
    batch_size: int = 32,
    num_kv_pairs: int = 8,
    num_queries: int = 4,
    vocab_size: int = 64,
    noise_len: int = 16,
    device: str = "cpu",
):
    """
    Multi-Query Associative Recall (MQAR) & Noise Resistance Task.
    Format:
      [k_1, v_1, k_2, v_2, ..., k_n, v_n] + [noise tokens...] + [q_1, ans_1, q_2, ans_2...]
    Only the answer positions after query keys contribute to CrossEntropy loss (other positions = -100).
    """
    half_vocab = (vocab_size - 2) // 2
    key_tokens = torch.arange(1, 1 + half_vocab, device=device)
    val_tokens = torch.arange(1 + half_vocab, 1 + 2 * half_vocab, device=device)
    noise_token_id = 0

    seq_len = num_kv_pairs * 2 + noise_len + num_queries * 2
    inputs = torch.full((batch_size, seq_len), noise_token_id, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        perm_k = key_tokens[torch.randperm(half_vocab, device=device)[:num_kv_pairs]]
        perm_v = val_tokens[torch.randperm(half_vocab, device=device)[:num_kv_pairs]]

        # Populate Key-Value context pairs
        for i in range(num_kv_pairs):
            inputs[b, i * 2] = perm_k[i]
            inputs[b, i * 2 + 1] = perm_v[i]

        # Insert distracting noise gap in the middle
        offset = num_kv_pairs * 2 + noise_len

        # Select query keys from the presented pairs
        q_indices = torch.randperm(num_kv_pairs, device=device)[:num_queries]
        for j, idx in enumerate(q_indices):
            q_pos = offset + j * 2
            inputs[b, q_pos] = perm_k[idx]
            if q_pos + 1 < seq_len:
                inputs[b, q_pos + 1] = perm_v[idx]
            # The model sees perm_k[idx] at q_pos and must predict perm_v[idx] at q_pos!
            targets[b, q_pos] = perm_v[idx]

    return inputs, targets
File 5: tests/test_sanity.pyPythonimport sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tricore import TriCoreLM, BaselineTransformerLM, generate_associative_recall_batch


def test_tricore_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TriCoreLM(vocab_size=64, d_model=64, n_layers=2, d_fast=16, max_loops=2).to(device)
    x, y = generate_associative_recall_batch(batch_size=4, num_kv_pairs=6, num_queries=3, vocab_size=64, device=device)

    logits, surprise = model(x)
    assert logits.shape == (4, x.shape[1], 64), f"Unexpected logits shape: {logits.shape}"
    assert surprise.shape == (4, x.shape[1]), f"Unexpected surprise shape: {surprise.shape}"

    loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(not torch.isnan(torch.tensor(g)) for g in grad_norms), "NaN gradient detected!"
    print(f"[PASS] Sanity check succeeded on {device}. Loss: {loss.item():.4f}, Checked {len(grad_norms)} parameter tensors.")


if __name__ == "__main__":
    test_tricore_forward_backward()
File 6: run_experiment.pyPythonimport os
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
    steps = 300
    batch_size = 32
    lr = 2e-3

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

            if step % 15 == 0 or step == 1:
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
        (n_layers * (d_model + d_fast * d_fast) * 4) / 1024.0 for _ in seq_lengths
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