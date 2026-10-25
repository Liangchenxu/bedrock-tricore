"""TriCore V13 definitive Track 7: static retention vs multi-round mutation.

The V12 mutation task wrote every variable exactly once before the query.  A
plain Hebbian memory could lower its global decay and forget the first value,
which is why V12's ``Delta vs Hebbian`` comparison was not significant
(``p=0.394``).  V13 removes that escape hatch:

* two variables are assigned once at the beginning and must survive every later
  mutation round and NOP ("static long-term bindings");
* two variables are overwritten for four cyclic rounds from the same active
  literal pool ("hot mutated bindings").

A high global decay preserves the static bindings but makes the four hot values
accumulate in a Hebbian ``F_mat``.  A low decay removes the hot history but also
erases the early static values.  The Delta rule keeps decay high and erases one
addressed key at a time with ``V_t - F_{t-1} K_{t-1}``.

The script is resumable: model/optimizer/scheduler state, step count and loss
history are checkpointed after every ``--save-every`` steps.  ``--chunk-steps``
caps one foreground invocation (use 50 to stay near the 30-second command
budget); repeat the same command until each run prints its final ``DONE`` line.
No background process and no progress polling are used.
"""

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.modules.setdefault("pynvml", None)

import torch
import torch.nn.functional as F

from tricore import BaselineTransformerLM, TriCoreLM, generate_stateful_mutation_batch
from run_v11_flagship import (
    BATCH_SIZE,
    LEARNING_RATE,
    SEEDS,
    count_parameters,
    fmt_p,
    json_ready,
    mean_std,
    welch_t_test,
    write_metrics,
)
from run_v12_apex import GATE_LEARNING_RATE, build_optimizer


METRICS_PATH = Path("results/V13_METRICS.json")
CHECKPOINT_DIR = Path("results/.v13_checkpoints")
FIGURE_PATH = Path("results/tricore_v13_definitive.png")
MASTER_FIGURE_PATH = Path("results/tricore_vs_transformer.png")
STEPS = 400
EVAL_BATCHES = 10
NUM_VARS = 4
NUM_QUERIES = 4
NUM_NOPS = 6
NUM_ROUNDS = 4
VOCAB_SIZE = 64
MAX_SEQ_LEN = 128
TRAIN_DATA_SEED = 42_007
EVAL_SEED = 7_001
CHANCE = 25.0

DELTA = "TriCore V13 Delta Full (Zero-Prior)"
HEBBIAN = "TriCore V13 Hebbian Full (No Delta Erasure)"
FAIR = "TriCore V13 Fair No-Engine-I"
TRANSFORMER = "Vanilla Transformer (2-layer)"
MODEL_ORDER = (DELTA, HEBBIAN, FAIR, TRANSFORMER)
MODEL_ALIASES = {
    "delta": DELTA,
    "hebbian": HEBBIAN,
    "fair": FAIR,
    "transformer": TRANSFORMER,
}
MODEL_SLUGS = {
    DELTA: "delta",
    HEBBIAN: "hebbian",
    FAIR: "fair",
    TRANSFORMER: "transformer",
}
COLORS = {
    DELTA: "#0B8457",
    HEBBIAN: "#D68910",
    FAIR: "#2E86C1",
    TRANSFORMER: "#7F8C8D",
}


def build_configs():
    """Build fresh, identically initialized model factories."""

    def tricore(delta=True, surprise=True):
        return TriCoreLM(
            vocab_size=VOCAB_SIZE,
            d_model=64,
            n_layers=1,
            d_fast=16,
            max_loops=2,
            enable_surprise_gate=surprise,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=None,
            syntax_tokens=None,
            reset_trace_tokens=None,
            enable_delta_rule=delta,
            autonomous_syntax=True,
            # Four overwrites put each hot variable at six total occurrences
            # (one LET + four SET + one NOP), so the V12 entity-frequency bound
            # of four would reject exactly the addresses this task must bind.
            autonomous_entity_max_freq=6,
        )

    return {
        DELTA: lambda: tricore(delta=True, surprise=True),
        HEBBIAN: lambda: tricore(delta=False, surprise=True),
        FAIR: lambda: tricore(delta=True, surprise=False),
        TRANSFORMER: lambda: BaselineTransformerLM(
            vocab_size=VOCAB_SIZE,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=MAX_SEQ_LEN,
        ),
    }


def build_train_batch(step: int, device: str):
    """Generate the exact same batch for a step from any resumed process."""
    state = torch.random.get_rng_state()
    # Seed only the CPU generator used by the data loader.  Seeding and then
    # saving/restoring every CUDA generator on each step synchronizes the GPU
    # and made batch construction cost more than the training update itself.
    torch.random.manual_seed(TRAIN_DATA_SEED + int(step))
    inputs, targets = generate_stateful_mutation_batch(
        batch_size=BATCH_SIZE,
        num_vars=NUM_VARS,
        num_queries=NUM_QUERIES,
        num_nops=NUM_NOPS,
        num_rounds=NUM_ROUNDS,
        vocab_size=VOCAB_SIZE,
        device="cpu",
    )
    torch.random.set_rng_state(state)
    return inputs.to(device), targets.to(device)


def build_eval_batches(device: str, count: int = EVAL_BATCHES):
    torch.manual_seed(EVAL_SEED)
    batches = [
        generate_stateful_mutation_batch(
            batch_size=BATCH_SIZE,
            num_vars=NUM_VARS,
            num_queries=NUM_QUERIES,
            num_nops=NUM_NOPS,
            num_rounds=NUM_ROUNDS,
            vocab_size=VOCAB_SIZE,
            device="cpu",
        )
        for _ in range(count)
    ]
    return [(inputs.to(device), targets.to(device)) for inputs, targets in batches]


def mutation_query_metadata(inputs: torch.Tensor):
    """Return query kind (1=static, 2=hot) and a valid stale literal per query.

    A static variable has exactly one assignment and therefore has no stale
    value to confuse.  A hot variable has several assignments; its last
    assignment is the target, and the first earlier value that differs from the
    target is a genuine stale competitor.
    """
    batch, length = inputs.shape
    kind = torch.zeros_like(inputs)
    stale = torch.full_like(inputs, -100)
    cpu = inputs.detach().cpu()
    for b in range(batch):
        alias_to_var = {}
        values = {}
        for pos in range(0, length, 5):
            op = int(cpu[b, pos])
            if op == 60:  # LET alias = var ;
                alias_to_var[int(cpu[b, pos + 1])] = int(cpu[b, pos + 3])
            elif op == 61:  # SET var = literal ;
                values.setdefault(int(cpu[b, pos + 1]), []).append(
                    int(cpu[b, pos + 3])
                )
        for pos in range(0, length, 5):
            if int(cpu[b, pos]) != 62:  # RET alias = ; ;
                continue
            alias = int(cpu[b, pos + 1])
            var = alias_to_var.get(alias)
            assigned = values.get(var, [])
            if len(assigned) == 1:
                kind[b, pos + 2] = 1
            elif len(assigned) > 1:
                kind[b, pos + 2] = 2
                final = assigned[-1]
                old = next(
                    (value for value in assigned[:-1] if value != final), -100
                )
                stale[b, pos + 2] = old
    return kind.to(inputs.device), stale.to(inputs.device)


def evaluate_mutation(model, batches, query_kind, stale_values):
    """Measure overall/static/hot accuracy and hot-only stale confusion."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    static_correct = 0
    static_total = 0
    hot_correct = 0
    hot_total = 0
    stale_correct = 0
    stale_total = 0
    with torch.no_grad():
        for (x, y), kind, stale in zip(
            batches, query_kind, stale_values
        ):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100
            )
            total_loss += loss.item()
            mask = y != -100
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == y[mask]).sum().item()
            total += int(mask.sum().item())

            static_mask = mask & (kind == 1)
            hot_mask = mask & (kind == 2)
            static_correct += (
                preds[static_mask] == y[static_mask]
            ).sum().item()
            static_total += int(static_mask.sum().item())
            hot_correct += (preds[hot_mask] == y[hot_mask]).sum().item()
            hot_total += int(hot_mask.sum().item())

            stale_mask = mask & (stale != -100)
            stale_correct += (
                preds[stale_mask] == stale[stale_mask]
            ).sum().item()
            stale_total += int(stale_mask.sum().item())
    model.train()
    return {
        "val_loss": total_loss / max(len(batches), 1),
        "new_value_accuracy_pct": correct / max(total, 1) * 100.0,
        "static_accuracy_pct": static_correct / max(static_total, 1) * 100.0,
        "hot_accuracy_pct": hot_correct / max(hot_total, 1) * 100.0,
        "stale_value_confusion_pct": stale_correct
        / max(stale_total, 1)
        * 100.0,
        "query_count": total,
        "static_query_count": static_total,
        "hot_query_count": hot_total,
        "stale_query_count": stale_total,
    }


def checkpoint_path(name: str, seed: int, directory=CHECKPOINT_DIR) -> Path:
    directory = Path(directory)
    return directory / f"{MODEL_SLUGS[name]}_seed{int(seed)}.pt"


def load_checkpoint(path: Path, device: str):
    if not path.is_file():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def protocol_matches(checkpoint: dict, steps: int) -> bool:
    protocol = checkpoint.get("protocol", {})
    return (
        int(protocol.get("steps", -1)) == int(steps)
        and int(protocol.get("num_rounds", -1)) == NUM_ROUNDS
        and int(protocol.get("num_nops", -1)) == NUM_NOPS
        and int(protocol.get("batch_size", -1)) == BATCH_SIZE
    )


def train_seed_chunk(name, builder, seed, device, args):
    path = checkpoint_path(name, seed, args.checkpoint_dir)
    checkpoint = load_checkpoint(path, device)
    if checkpoint is not None:
        if not protocol_matches(checkpoint, args.steps):
            raise RuntimeError(
                f"checkpoint protocol mismatch for {name} seed={seed}: {path}"
            )
        if checkpoint.get("complete") and checkpoint.get("final"):
            final = checkpoint["final"]
            print(
                f"RESUME V13 {name} seed={seed}: already complete "
                f"new={final['new_value_accuracy_pct']:.2f}%",
                flush=True,
            )
            return final

    torch.manual_seed(int(seed))
    model = builder().to(device)
    params = count_parameters(model)
    optimizer, scheduler = build_optimizer(
        model, args.steps, learning_rate=args.lr
    )
    start_step = 0
    losses = []
    elapsed = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_step = int(checkpoint["step"])
        losses = [float(value) for value in checkpoint.get("loss_history", [])]
        elapsed = float(checkpoint.get("train_seconds", 0.0))

    chunk_steps = int(args.chunk_steps)
    end_step = (
        args.steps
        if chunk_steps <= 0
        else min(args.steps, start_step + chunk_steps)
    )
    model.train()
    started = time.perf_counter()
    for step in range(start_step, end_step):
        inputs, targets = build_train_batch(step, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            targets.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if (step + 1) % max(int(args.save_every), 1) == 0:
            save_checkpoint(
                path,
                {
                    "model": name,
                    "seed": int(seed),
                    "step": step + 1,
                    "steps": int(args.steps),
                    "params": params,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss_history": losses,
                    "train_seconds": elapsed + time.perf_counter() - started,
                    "complete": False,
                    "protocol": {
                        "steps": int(args.steps),
                        "batch_size": BATCH_SIZE,
                        "num_vars": NUM_VARS,
                        "num_queries": NUM_QUERIES,
                        "num_nops": NUM_NOPS,
                        "num_rounds": NUM_ROUNDS,
                    },
                },
            )
    elapsed += time.perf_counter() - started
    completed = end_step >= args.steps

    final = None
    if completed:
        eval_batches = build_eval_batches(device, args.eval_batches)
        metadata = [mutation_query_metadata(x) for x, _ in eval_batches]
        eval_payload = evaluate_mutation(
            model,
            eval_batches,
            [item[0] for item in metadata],
            [item[1] for item in metadata],
        )
        final = {
            "name": name,
            "seed": int(seed),
            "params": params,
            "steps": int(args.steps),
            **eval_payload,
            "train_loss_last50": statistics.mean(losses[-50:]),
            "train_time_s": elapsed,
            "loss_history": losses,
        }

    checkpoint_payload = {
        "model": name,
        "seed": int(seed),
        "step": end_step,
        "steps": int(args.steps),
        "params": params,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss_history": losses,
        "train_seconds": elapsed,
        "complete": completed,
        "final": final,
        "protocol": {
            "steps": int(args.steps),
            "batch_size": BATCH_SIZE,
            "num_vars": NUM_VARS,
            "num_queries": NUM_QUERIES,
            "num_nops": NUM_NOPS,
            "num_rounds": NUM_ROUNDS,
        },
    }
    save_checkpoint(path, checkpoint_payload)

    if completed:
        print(
            f"DONE V13 {name} seed={seed}: "
            f"new={final['new_value_accuracy_pct']:.2f}% "
            f"static={final['static_accuracy_pct']:.2f}% "
            f"hot={final['hot_accuracy_pct']:.2f}% "
            f"stale={final['stale_value_confusion_pct']:.2f}% "
            f"loss={final['val_loss']:.4f} time={elapsed:.1f}s",
            flush=True,
        )
    else:
        print(
            f"CHUNK V13 {name} seed={seed}: step={end_step}/{args.steps} "
            f"train_seconds={elapsed:.1f}",
            flush=True,
        )
    return final
def summarize_model(runs, steps):
    ordered = sorted(runs, key=lambda run: int(run["seed"]))
    new_mean, new_std = mean_std(
        run["new_value_accuracy_pct"] for run in ordered
    )
    static_mean, static_std = mean_std(
        run["static_accuracy_pct"] for run in ordered
    )
    hot_mean, hot_std = mean_std(run["hot_accuracy_pct"] for run in ordered)
    stale_mean, stale_std = mean_std(
        run["stale_value_confusion_pct"] for run in ordered
    )
    loss_mean, loss_std = mean_std(run["val_loss"] for run in ordered)
    curves = [
        run["loss_history"]
        for run in ordered
        if len(run.get("loss_history", [])) == int(steps)
    ]
    curve_mean = []
    curve_std = []
    if curves:
        for index in range(int(steps)):
            mean, std = mean_std(curve[index] for curve in curves)
            curve_mean.append(mean)
            curve_std.append(std)
    return {
        "params": ordered[0]["params"],
        "runs": ordered,
        "new_value_accuracy_mean": new_mean,
        "new_value_accuracy_std": new_std,
        "static_accuracy_mean": static_mean,
        "static_accuracy_std": static_std,
        "hot_accuracy_mean": hot_mean,
        "hot_accuracy_std": hot_std,
        "stale_value_confusion_mean": stale_mean,
        "stale_value_confusion_std": stale_std,
        "val_loss_mean": loss_mean,
        "val_loss_std": loss_std,
        "loss_curve_mean": curve_mean,
        "loss_curve_std": curve_std,
    }


def collect_completed(steps, seeds, directory=CHECKPOINT_DIR):
    models = {}
    status = {}
    for name in MODEL_ORDER:
        runs = []
        for seed in seeds:
            path = checkpoint_path(name, seed, directory)
            checkpoint = load_checkpoint(path, "cpu")
            key = f"{MODEL_SLUGS[name]}_seed{seed}"
            if checkpoint is not None:
                status[key] = {
                    "step": int(checkpoint.get("step", 0)),
                    "complete": bool(checkpoint.get("complete", False)),
                }
                if checkpoint.get("complete") and checkpoint.get("final"):
                    runs.append(checkpoint["final"])
        if runs:
            models[name] = summarize_model(runs, steps)
    return models, status

def build_payload(models, status, seeds, steps):
    required = all(
        name in models
        and set(int(seed) for seed in seeds)
        <= {int(run["seed"]) for run in models[name]["runs"]}
        for name in MODEL_ORDER
    )
    significance = {}
    if required:
        def values(name, key):
            return [run[key] for run in models[name]["runs"]]

        significance = {
            "delta_vs_hebbian_new_value": welch_t_test(
                values(DELTA, "new_value_accuracy_pct"),
                values(HEBBIAN, "new_value_accuracy_pct"),
            ),
            "delta_vs_hebbian_static": welch_t_test(
                values(DELTA, "static_accuracy_pct"),
                values(HEBBIAN, "static_accuracy_pct"),
            ),
            "delta_vs_hebbian_hot": welch_t_test(
                values(DELTA, "hot_accuracy_pct"),
                values(HEBBIAN, "hot_accuracy_pct"),
            ),
            "delta_vs_hebbian_stale_value": welch_t_test(
                values(DELTA, "stale_value_confusion_pct"),
                values(HEBBIAN, "stale_value_confusion_pct"),
            ),
            "delta_vs_fair_no_engine_i_new_value": welch_t_test(
                values(DELTA, "new_value_accuracy_pct"),
                values(FAIR, "new_value_accuracy_pct"),
            ),
            "delta_vs_transformer_new_value": welch_t_test(
                values(DELTA, "new_value_accuracy_pct"),
                values(TRANSFORMER, "new_value_accuracy_pct"),
            ),
            "hebbian_vs_fair_no_engine_i_new_value": welch_t_test(
                values(HEBBIAN, "new_value_accuracy_pct"),
                values(FAIR, "new_value_accuracy_pct"),
            ),
        }
    statements = (
        NUM_VARS
        + NUM_VARS // 2
        + (NUM_VARS - NUM_VARS // 2) * NUM_ROUNDS
        + NUM_NOPS
        + NUM_QUERIES
    )
    return {
        "version": "1.6.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "zero_prior": True,
        "token_id_masks": False,
        "complete": required,
        "protocol": {
            "seeds": list(seeds),
            "steps": int(steps),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "gate_learning_rate": GATE_LEARNING_RATE,
            "weight_decay": 1e-2,
            "d_model": 64,
            "d_fast": 16,
            "max_loops": 2,
            "num_vars": NUM_VARS,
            "num_queries": NUM_QUERIES,
            "num_nops": NUM_NOPS,
            "num_rounds": NUM_ROUNDS,
            "n_static": NUM_VARS // 2,
            "n_hot": NUM_VARS - NUM_VARS // 2,
            "num_statements": statements,
            "sequence_length": statements * 5,
            "autonomous_entity_max_freq": 6,
            "eval_batches": EVAL_BATCHES,
            "chance_pct": CHANCE,
            "task": "static long-term retention + 4-round hot mutation",
            "train_data_seed": TRAIN_DATA_SEED,
            "eval_seed": EVAL_SEED,
            "delta_rule": (
                "F_t = decay*F_{t-1} + lr*(V_t - F_{t-1}K_{t-1})K_{t-1}^T"
            ),
            "hebbian_rule": "F_t = decay*F_{t-1} + lr*V_t K_{t-1}^T",
        },
        "models": models,
        "significance": significance,
        "checkpoint_status": status,
    }


def load_metrics(path: Path):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("version") == "1.6.0" else None

SHORT_LABELS = {
    DELTA: "Delta\nFull",
    HEBBIAN: "Hebbian\n(no erase)",
    FAIR: "Fair\nNo-I Delta",
    TRANSFORMER: "2-Layer\nTF",
}


def plot_v13(metrics, output_paths=(FIGURE_PATH, MASTER_FIGURE_PATH)):
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
    models = metrics["models"]
    order = [name for name in MODEL_ORDER if name in models]
    short = [SHORT_LABELS[name] for name in order]
    colors = [COLORS[name] for name in order]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=180)

    # (a) Overall new-value accuracy.
    means = [models[name]["new_value_accuracy_mean"] for name in order]
    stds = [models[name]["new_value_accuracy_std"] for name in order]
    ax = axes[0, 0]
    bars = ax.bar(
        short,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(CHANCE, color="#C0392B", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Overall query accuracy (%)")
    ax.set_title("(a) V13 overall accuracy (3-seed mean ± std)")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(mean + std + 2, 108),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
        )
    sig = metrics.get("significance", {})
    if "delta_vs_hebbian_new_value" in sig:
        p_value = sig["delta_vs_hebbian_new_value"]["p_value_two_tailed"]
        ax.text(
            0.03,
            0.96,
            f"Delta vs Hebbian\nWelch p = {fmt_p(p_value)}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.8,
        )

    # (b) The decay-interference dilemma: static retention vs hot mutation.
    static_means = [models[name]["static_accuracy_mean"] for name in order]
    static_stds = [models[name]["static_accuracy_std"] for name in order]
    hot_means = [models[name]["hot_accuracy_mean"] for name in order]
    hot_stds = [models[name]["hot_accuracy_std"] for name in order]
    x = np.arange(len(order))
    width = 0.36
    ax = axes[0, 1]
    ax.bar(
        x - width / 2,
        static_means,
        width,
        yerr=static_stds,
        capsize=4,
        color="#1F618D",
        label="Static (assigned once)",
    )
    ax.bar(
        x + width / 2,
        hot_means,
        width,
        yerr=hot_stds,
        capsize=4,
        color="#D68910",
        label="Hot (4x overwritten)",
    )
    ax.axhline(CHANCE, color="#C0392B", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, short)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Static retention vs hot-value erasure")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=7.5)

    # (c) Per-seed accuracy makes a seed collapse visible.
    ax = axes[0, 2]
    seed_values = [int(seed) for seed in metrics["protocol"]["seeds"]]
    for name in order:
        by_seed = {
            int(run["seed"]): run["new_value_accuracy_pct"]
            for run in models[name]["runs"]
        }
        y = [by_seed.get(seed, np.nan) for seed in seed_values]
        ax.plot(
            seed_values,
            y,
            marker="o",
            linewidth=1.8,
            color=COLORS[name],
            label=SHORT_LABELS[name].replace("\n", " "),
        )
    ax.axhline(CHANCE, color="#C0392B", linestyle="--", linewidth=1.1)
    ax.set_xticks(seed_values)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("(c) Per-seed stability")
    ax.legend(frameon=False, loc="lower left", fontsize=7.0)

    # (d) Hot-only stale-value confusion.
    stale_means = [
        models[name]["stale_value_confusion_mean"] for name in order
    ]
    stale_stds = [models[name]["stale_value_confusion_std"] for name in order]
    ax = axes[1, 0]
    bars = ax.bar(
        short,
        stale_means,
        yerr=stale_stds,
        capsize=5,
        color="#D68910",
        alpha=0.9,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Stale-value confusion (%)")
    ax.set_title("(d) Hot stale-value confusion (lower is better)")
    for bar, mean in zip(bars, stale_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(mean + 1.5, 96),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )
    if "delta_vs_hebbian_stale_value" in sig:
        p_value = sig["delta_vs_hebbian_stale_value"][
            "p_value_two_tailed"
        ]
        ax.text(
            0.03,
            0.96,
            f"Delta vs Hebbian\nstale p = {fmt_p(p_value)}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.8,
        )

    # (e) Validation loss on a log scale.
    losses = [models[name]["val_loss_mean"] for name in order]
    loss_stds = [models[name]["val_loss_std"] for name in order]
    ax = axes[1, 1]
    bars = ax.bar(
        short,
        losses,
        yerr=loss_stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Validation cross-entropy (log)")
    ax.set_title("(e) Validation loss")
    for bar, mean in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean * 1.08,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    # (f) Training curves, mean ± std across seeds.
    ax = axes[1, 2]
    for name in order:
        mean_curve = models[name].get("loss_curve_mean", [])
        std_curve = models[name].get("loss_curve_std", [])
        if not mean_curve:
            continue
        steps_axis = np.arange(1, len(mean_curve) + 1)
        mean_array = np.asarray(mean_curve, dtype=float)
        std_array = np.asarray(std_curve, dtype=float)
        ax.plot(
            steps_axis,
            mean_array,
            linewidth=1.8,
            color=COLORS[name],
            label=SHORT_LABELS[name].replace("\n", " "),
        )
        ax.fill_between(
            steps_axis,
            np.maximum(mean_array - std_array, 1e-8),
            mean_array + std_array,
            color=COLORS[name],
            alpha=0.13,
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training cross-entropy (log)")
    ax.set_title("(f) V13 training dynamics")
    ax.legend(frameon=False, loc="upper right", fontsize=7.0)

    fig.suptitle(
        "TriCore V13 — Static long-term retention + 4-round hot mutation "
        "(400 steps, seeds 42/100/2026)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "V13 isolates the Decay-Interference Dilemma: Hebbian decay must either "
        "forget early static bindings or accumulate all four hot values. "
        "Delta keeps decay high and erases each overwritten key pointwise. "
        "Bars are mean ± sample std over three seeds.",
        ha="center",
        fontsize=8.4,
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

def resolve_models(values):
    if not values:
        return list(MODEL_ORDER)
    selected = []
    for value in values:
        if value in MODEL_ORDER:
            name = value
        else:
            name = MODEL_ALIASES.get(value.lower())
        if name is None or name not in MODEL_ORDER:
            raise ValueError(
                f"unknown model {value!r}; use delta, hebbian, fair, "
                "transformer, or a full model name"
            )
        if name not in selected:
            selected.append(name)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the resumable TriCore V13 Track 7 definitive suite"
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=50,
        help="max training steps in one foreground invocation; 0 runs all",
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics)
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.reset:
        if metrics_path.exists():
            metrics_path.unlink()
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        print(f"RESET V13: removed {metrics_path} and {checkpoint_dir}", flush=True)

    if args.plot_only:
        metrics = load_metrics(metrics_path)
        if metrics is None:
            raise FileNotFoundError(metrics_path)
        if not metrics.get("complete"):
            raise RuntimeError("--plot-only requires a complete V13 metrics file")
        figures = plot_v13(metrics)
        metrics["figures"] = figures
        write_metrics(metrics, metrics_path)
        print("Wrote figures: " + ", ".join(figures), flush=True)
        return

    if args.device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("TriCore V13 Track 7 requires CUDA")
        device = "cuda"
    else:
        device = args.device
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f"=== V13 on {torch.cuda.get_device_name(0)} ===", flush=True)

    seeds = tuple(int(seed) for seed in args.seeds)
    selected = resolve_models(args.models)
    configs = build_configs()
    metrics = load_metrics(metrics_path) or {
        "version": "1.6.0",
        "started_at_unix": time.time(),
        "runtime_seconds": 0.0,
    }
    previous_runtime = float(metrics.get("runtime_seconds", 0.0))
    command_started = time.perf_counter()

    for name in selected:
        for seed in seeds:
            train_seed_chunk(name, configs[name], seed, device, args)
            models, status = collect_completed(
                args.steps, seeds, checkpoint_dir
            )
            payload = build_payload(models, status, seeds, args.steps)
            payload["started_at_unix"] = metrics.get(
                "started_at_unix", time.time()
            )
            payload["runtime_seconds"] = previous_runtime + (
                time.perf_counter() - command_started
            )
            write_metrics(payload, metrics_path)
            metrics = payload

    metrics["runtime_seconds"] = previous_runtime + (
        time.perf_counter() - command_started
    )
    if metrics.get("complete") and not args.no_plot:
        figures = plot_v13(metrics)
        metrics["figures"] = figures
        print("Wrote figures: " + ", ".join(figures), flush=True)
    write_metrics(metrics, metrics_path)

    if metrics.get("complete"):
        for name in MODEL_ORDER:
            model = metrics["models"][name]
            print(
                f"SUMMARY {name}: new={model['new_value_accuracy_mean']:.2f}% "
                f"static={model['static_accuracy_mean']:.2f}% "
                f"hot={model['hot_accuracy_mean']:.2f}% "
                f"stale={model['stale_value_confusion_mean']:.2f}%",
                flush=True,
            )
    else:
        print(
            "V13 incomplete: rerun the same command to continue from "
            f"{metrics_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

