"""TriCore V11 flagship benchmark.

Runs four independent measurements and writes ``results/V11_METRICS.json``:

* Track 5: three-seed infix-``EQ`` dereference with Welch's t-tests.
* Track 6: single-layer 2/3/4-hop depth scaling plus a hop-router audit.
* Track 4B Extended: real CUDA decode state / KV-cache allocation and latency
  from ``L=128`` through ``L=131,072``.
* A 2x3 conference-style figure saved to both master-figure paths.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# The deprecated pynvml shim can block torch import on Windows.
sys.modules.setdefault("pynvml", None)

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricore import (
    BaselineTransformerLM,
    TriCoreLM,
    generate_4hop_recall_batch,
    generate_associative_recall_batch,
)
from tricore.tasks import generate_3hop_recall_batch
from run_v10_eligibility import (
    compute_eq_elig_gate_stats,
    eval_infix,
    generate_infix_eq_batch,
)


METRICS_PATH = Path("results/V11_METRICS.json")
FIGURE_PATH = Path("results/tricore_vs_transformer.png")
MASTERPIECE_PATH = Path("results/tricore_v11_masterpiece.png")
SEEDS = (42, 100, 2026)
TRACK5_STEPS = 350
TRACK6_STEPS = 500
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-2
TRACK5_D_MODEL = 64
TRACK5_D_FAST = 16
TRACK6_D_MODEL = 128
TRACK6_D_FAST = 32
TRACK6_CHAINS = 4
TRACK6_QUERIES = 4
TRACK6_VOCAB = 96
TRACK6_NOISE = 12
TRACK6_OOD_NOISE = 48
TRACK6_EVAL_BATCHES = 8
TRACK6_EVAL_SEED_IN = 6101
TRACK6_EVAL_SEED_OOD = 6102
TRACK6_CHANCE = 25.0
PROFILE_LENGTHS = (128, 512, 2048, 8192, 16384, 32768, 65536, 131072)
PROFILE_D_MODEL = 128
PROFILE_HEADS = 4
PROFILE_D_HEAD = 32


# ---------------------------------------------------------------------------
# Module A: exact Welch two-sample t-test (no SciPy dependency)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function."""
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` with NR's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_stat: float, df: float) -> float:
    """Exact two-tailed Student-t tail probability via the Beta identity."""
    if math.isnan(t_stat) or math.isnan(df):
        return float("nan")
    if math.isinf(t_stat):
        return 0.0
    if df <= 0.0:
        return float("nan")
    x = df / (df + t_stat * t_stat)
    return regularized_incomplete_beta(df / 2.0, 0.5, x)


def _mean_std(values):
    values = [float(value) for value in values]
    n = len(values)
    mean = sum(values) / n
    sample_std = statistics.stdev(values) if n > 1 else 0.0
    population_std = statistics.pstdev(values) if n > 1 else 0.0
    return mean, sample_std, population_std


def welch_t_test(xs, ys) -> dict:
    """Welch's unequal-variance two-sample t-test with Welch-Satterthwaite df.

    Both sample (``ddof=1``) and population (``ddof=0``) standard deviations are
    reported.  The p-value is evaluated with the regularized incomplete Beta
    function, so no SciPy dependency is required.
    """
    xs = [float(value) for value in xs]
    ys = [float(value) for value in ys]
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Welch's t-test requires at least two values per group")

    mean_x, sample_std_x, population_std_x = _mean_std(xs)
    mean_y, sample_std_y, population_std_y = _mean_std(ys)
    nx, ny = len(xs), len(ys)
    var_x = sample_std_x ** 2
    var_y = sample_std_y ** 2
    se_sq = var_x / nx + var_y / ny
    mean_diff = mean_x - mean_y

    if se_sq == 0.0:
        if mean_diff == 0.0:
            t_stat, df, p_value = 0.0, float(nx + ny - 2), 1.0
        else:
            t_stat, df, p_value = float("inf"), float(nx + ny - 2), 0.0
    else:
        t_stat = mean_diff / math.sqrt(se_sq)
        numerator = se_sq ** 2
        denominator = (var_x / nx) ** 2 / (nx - 1) + (var_y / ny) ** 2 / (ny - 1)
        df = numerator / denominator
        p_value = student_t_two_sided_p(t_stat, df)

    return {
        "n_x": nx,
        "n_y": ny,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "sample_std_x": sample_std_x,
        "sample_std_y": sample_std_y,
        "population_std_x": population_std_x,
        "population_std_y": population_std_y,
        "mean_difference": mean_diff,
        "t_statistic": t_stat,
        "degrees_of_freedom": df,
        "p_value_two_tailed": p_value,
    }


def fmt_p(p_value) -> str:
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return "n/a"
    if p_value == 0.0:
        return "<1e-300"
    if p_value < 1.0e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def mean_std(values) -> tuple[float, float]:
    values = [float(value) for value in values]
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def write_metrics(metrics: dict, path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Module B: Track 5, three seeds across infix EQ
# ---------------------------------------------------------------------------


def build_track5_configs():
    semi, eq, let_, set_, ret, nop = 0, 59, 60, 61, 62, 63
    syntax_tokens = (semi, let_, set_, ret, nop)
    return {
        "TriCore V11 Full (with eligibility trace)": lambda: TriCoreLM(
            vocab_size=64,
            d_model=TRACK5_D_MODEL,
            n_layers=1,
            d_fast=TRACK5_D_FAST,
            max_loops=2,
            enable_surprise_gate=True,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=eq,
            syntax_tokens=syntax_tokens,
        ),
        "TriCore V11 Fair No-Engine-I (unmodulated gates)": lambda: TriCoreLM(
            vocab_size=64,
            d_model=TRACK5_D_MODEL,
            n_layers=1,
            d_fast=TRACK5_D_FAST,
            max_loops=2,
            enable_surprise_gate=False,
            enable_fast_weight=True,
            enable_adaptive_loop=True,
            separator_token=eq,
            syntax_tokens=syntax_tokens,
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
    )
    sync_device(device)
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
                f"[Track5] {name} seed={seed} step={step:03d}/{steps} "
                f"train_loss={last_loss:.4f}",
                flush=True,
            )
    sync_device(device)
    train_seconds = time.perf_counter() - started
    val_loss, val_acc = eval_infix(model, device, batches=10)
    eq_gate, entity_gate = compute_eq_elig_gate_stats(model, device)
    result = {
        "seed": seed,
        "params": params,
        "train_loss_last": last_loss,
        "val_loss": val_loss,
        "accuracy_pct": val_acc,
        "mean_elig_gate_eq": None if math.isnan(eq_gate) else eq_gate,
        "mean_elig_gate_entity": None if math.isnan(entity_gate) else entity_gate,
        "train_time_s": train_seconds,
    }
    print(
        f"DONE Track5 {name} seed={seed}: acc={val_acc:.2f}% "
        f"loss={val_loss:.4f} gates={eq_gate:.4f}/{entity_gate:.4f} "
        f"time={train_seconds:.1f}s",
        flush=True,
    )
    return result




def run_track5(device: str, seeds=SEEDS, steps=TRACK5_STEPS) -> dict:
    configs = build_track5_configs()
    models = {}
    for name, builder in configs.items():
        runs = [
            train_track5_model(name, builder, seed, device, steps)
            for seed in seeds
        ]
        acc_mean, acc_std = mean_std(run["accuracy_pct"] for run in runs)
        loss_mean, loss_std = mean_std(run["val_loss"] for run in runs)
        gate_eq = [
            r["mean_elig_gate_eq"]
            for r in runs
            if r["mean_elig_gate_eq"] is not None
        ]
        gate_entity = [
            r["mean_elig_gate_entity"]
            for r in runs
            if r["mean_elig_gate_entity"] is not None
        ]
        models[name] = {
            "params": runs[0]["params"],
            "runs": runs,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "val_loss_mean": loss_mean,
            "val_loss_std": loss_std,
            "elig_gate_eq_mean": (
                statistics.mean(gate_eq) if gate_eq else None
            ),
            "elig_gate_eq_std": (
                statistics.stdev(gate_eq)
                if len(gate_eq) > 1
                else (0.0 if gate_eq else None)
            ),
            "elig_gate_entity_mean": (
                statistics.mean(gate_entity) if gate_entity else None
            ),
            "elig_gate_entity_std": (
                statistics.stdev(gate_entity)
                if len(gate_entity) > 1
                else (0.0 if gate_entity else None)
            ),
        }

    full_name = "TriCore V11 Full (with eligibility trace)"
    fair_name = "TriCore V11 Fair No-Engine-I (unmodulated gates)"
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
            "weight_decay": WEIGHT_DECAY,
            "d_model": TRACK5_D_MODEL,
            "d_fast": TRACK5_D_FAST,
            "max_loops": 2,
            "eval_batches": 10,
            "test": "infix EQ two-hop dereference",
            "chance_pct": 25.0,
            "chance_note": (
                "25.0% is the four-way chance after the four SET literals are "
                "identified; uniform chance over all 16 literals is 6.25%."
            ),
        },
        "models": models,
        "significance": significance,
    }


# ---------------------------------------------------------------------------
# Module C: Track 6, four-hop single-step chain and depth scaling
# ---------------------------------------------------------------------------


def evaluate_recall(model, eval_batches, vocab_size: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in eval_batches:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), ignore_index=-100
            )
            total_loss += loss.item()
            mask = y != -100
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return (
        total_loss / max(len(eval_batches), 1),
        (correct / max(total, 1)) * 100.0,
    )


def build_4hop_train_batches(device: str, steps: int):
    torch.manual_seed(42_000)
    return [
        generate_4hop_recall_batch(
            batch_size=BATCH_SIZE,
            num_chains=TRACK6_CHAINS,
            num_queries=TRACK6_QUERIES,
            vocab_size=TRACK6_VOCAB,
            noise_len=TRACK6_NOISE,
            device=device,
        )
        for _ in range(steps)
    ]


def build_4hop_eval_sets(device: str, eval_batches: int = TRACK6_EVAL_BATCHES):
    torch.manual_seed(TRACK6_EVAL_SEED_IN)
    in_domain = [
        generate_4hop_recall_batch(
            batch_size=BATCH_SIZE,
            num_chains=TRACK6_CHAINS,
            num_queries=TRACK6_QUERIES,
            vocab_size=TRACK6_VOCAB,
            noise_len=TRACK6_NOISE,
            device=device,
        )
        for _ in range(eval_batches)
    ]
    torch.manual_seed(TRACK6_EVAL_SEED_OOD)
    extrapolation = [
        generate_4hop_recall_batch(
            batch_size=BATCH_SIZE,
            num_chains=TRACK6_CHAINS,
            num_queries=TRACK6_QUERIES,
            vocab_size=TRACK6_VOCAB,
            noise_len=TRACK6_OOD_NOISE,
            device=device,
        )
        for _ in range(eval_batches)
    ]
    return in_domain, extrapolation


def build_track6_configs():
    def tricore(max_loops: int):
        return TriCoreLM(
            vocab_size=TRACK6_VOCAB,
            d_model=TRACK6_D_MODEL,
            n_layers=1,
            d_fast=TRACK6_D_FAST,
            max_loops=max_loops,
            separator_token=0,
            syntax_tokens=(0,),
            reset_trace_tokens=(0,),
        )

    return {
        "TriCore Full (1-Layer, max_loops=3, 4-Hop capable)": lambda: tricore(3),
        "TriCore (1-Layer, max_loops=2, 3-Hop limit)": lambda: tricore(2),
        "TriCore (1-Layer, max_loops=1, 2-Hop limit)": lambda: tricore(1),
        "Vanilla Transformer (2-Layer, d_model=128, 4 heads)": lambda: BaselineTransformerLM(
            vocab_size=TRACK6_VOCAB,
            d_model=TRACK6_D_MODEL,
            n_layers=2,
            n_heads=4,
            max_seq_len=128,
        ),
    }


@torch.no_grad()
def hop_router_stats(model, eval_batches) -> list[float]:
    """Mean softmax routing weights on the four query-key positions."""
    if not isinstance(model, TriCoreLM):
        return [None] * 4
    block = model.layers[0]
    total = torch.zeros(block.max_loops + 1, dtype=torch.float64, device=next(model.parameters()).device)
    count = 0
    model.eval()
    for x, y in eval_batches:
        u = block.in_norm(model.embedding(x))
        weights = F.softmax(block.hop_router(u), dim=-1)
        mask = y != -100
        total += weights[mask].double().sum(dim=0)
        count += int(mask.sum().item())
    model.train()
    if count == 0:
        return [None] * (block.max_loops + 1)
    values = (total / count).cpu().tolist()
    while len(values) < 4:
        values.append(0.0)
    return values[:4]


def train_track6_model(name, builder, device, train_batches, eval_sets, seed=42):
    torch.manual_seed(seed)
    model = builder().to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_batches)
    )

    warm_x, warm_y = train_batches[0]
    warm_logits, _ = model(warm_x)
    warm_loss = F.cross_entropy(
        warm_logits.view(-1, TRACK6_VOCAB),
        warm_y.view(-1),
        ignore_index=-100,
    )
    optimizer.zero_grad(set_to_none=True)
    warm_loss.backward()
    optimizer.zero_grad(set_to_none=True)

    losses = []
    sync_device(device)
    started = time.perf_counter()
    for step, (x, y) in enumerate(train_batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, TRACK6_VOCAB), y.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach())
        if step == 1 or step % 100 == 0:
            print(
                f"[Track6] {name} step={step:03d}/{len(train_batches)} "
                f"train_loss={loss.item():.4f}",
                flush=True,
            )
    sync_device(device)
    train_seconds = time.perf_counter() - started
    train_loss = torch.stack(losses[-50:]).mean().item()
    in_loss, in_acc = evaluate_recall(model, eval_sets[0], TRACK6_VOCAB)
    ood_loss, ood_acc = evaluate_recall(model, eval_sets[1], TRACK6_VOCAB)
    router = (
        hop_router_stats(model, eval_sets[0])
        if isinstance(model, TriCoreLM)
        else [None] * 4
    )
    result = {
        "name": name,
        "seed": seed,
        "params": params,
        "train_loss_last50": train_loss,
        "in_loss": in_loss,
        "in_accuracy_pct": in_acc,
        "ood_loss": ood_loss,
        "ood_accuracy_pct": ood_acc,
        "train_time_s": train_seconds,
        "hop_router_query_weights": router,
        "state_bytes_per_sample": (
            state_bytes(model.init_state(1, device, torch.float32))
            if isinstance(model, TriCoreLM)
            else None
        ),
    }
    print(
        f"DONE Track6 {name}: in={in_acc:.2f}% ood={ood_acc:.2f}% "
        f"router={router} time={train_seconds:.1f}s",
        flush=True,
    )
    return model, result


def load_v9_track4a() -> dict:
    path = Path("results/V9_METRICS.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("track4a", {})


def track4a_welch(track4a: dict) -> dict:
    summaries = {row["name"]: row for row in track4a.get("summary", [])}
    if not summaries:
        return {}
    full = summaries.get("TriCore Full")
    if not full:
        return {}
    full_in = [row["in_acc"] for row in full["per_seed"]]
    full_ood = [row["ood_acc"] for row in full["per_seed"]]
    result = {}
    for name, row in summaries.items():
        if name == "TriCore Full":
            continue
        result[name] = {
            "in_domain": welch_t_test(
                full_in, [item["in_acc"] for item in row["per_seed"]]
            ),
            "ood_4x": welch_t_test(
                full_ood, [item["ood_acc"] for item in row["per_seed"]]
            ),
        }
    return result


def run_two_hop_depth_reference(device: str, steps: int = 350) -> dict:
    """Train a one-hop readout (max_loops=0) on the V9 2-hop protocol."""
    torch.manual_seed(43_000)
    train_batches = [
        generate_associative_recall_batch(
            batch_size=BATCH_SIZE,
            num_kv_pairs=4,
            num_queries=4,
            vocab_size=64,
            noise_len=8,
            device=device,
        )
        for _ in range(steps)
    ]
    torch.manual_seed(8101)
    in_domain = [
        generate_associative_recall_batch(
            batch_size=BATCH_SIZE,
            num_kv_pairs=4,
            num_queries=4,
            vocab_size=64,
            noise_len=8,
            device=device,
        )
        for _ in range(5)
    ]
    torch.manual_seed(8102)
    ood = [
        generate_associative_recall_batch(
            batch_size=BATCH_SIZE,
            num_kv_pairs=4,
            num_queries=4,
            vocab_size=64,
            noise_len=32,
            device=device,
        )
        for _ in range(5)
    ]
    torch.manual_seed(42)
    model = TriCoreLM(
        vocab_size=64,
        d_model=64,
        n_layers=1,
        d_fast=16,
        max_loops=0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
    )
    started = time.perf_counter()
    for x, y in train_batches:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, 64), y.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    sync_device(device)
    in_loss, in_acc = evaluate_recall(model, in_domain, 64)
    ood_loss, ood_acc = evaluate_recall(model, ood, 64)
    result = {
        "name": "TriCore 1-Hop readout (max_loops=0)",
        "params": count_parameters(model),
        "in_accuracy_pct": in_acc,
        "ood_accuracy_pct": ood_acc,
        "in_loss": in_loss,
        "ood_loss": ood_loss,
        "train_time_s": time.perf_counter() - started,
    }
    print(
        f"DONE 2-hop shallow reference: in={in_acc:.2f}% ood={ood_acc:.2f}%",
        flush=True,
    )
    return result



def run_track6(device: str, steps: int = TRACK6_STEPS, eval_batches: int = 8) -> dict:
    train_batches = build_4hop_train_batches(device, steps)
    eval_sets = build_4hop_eval_sets(device, eval_batches)
    runs = {}
    models = {}
    for name, builder in build_track6_configs().items():
        model, result = train_track6_model(
            name, builder, device, train_batches, eval_sets
        )
        models[name] = model
        runs[name] = result

    full_name = "TriCore Full (1-Layer, max_loops=3, 4-Hop capable)"
    shallow_name = "TriCore (1-Layer, max_loops=2, 3-Hop limit)"
    transformer_name = "Vanilla Transformer (2-Layer, d_model=128, 4 heads)"
    router = runs[full_name]["hop_router_query_weights"]

    v9_track4a = load_v9_track4a()
    two_hop_reference = run_two_hop_depth_reference(
        device, steps=min(350, steps)
    )
    del models
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "protocol": {
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW + cosine annealing + grad clip 1.0",
            "seed": 42,
            "num_chains": TRACK6_CHAINS,
            "num_queries": TRACK6_QUERIES,
            "vocab_size": TRACK6_VOCAB,
            "noise_len": TRACK6_NOISE,
            "sequence_len": 68,
            "ood_noise_len": TRACK6_OOD_NOISE,
            "ood_sequence_len": 104,
            "eval_seed_in_domain": TRACK6_EVAL_SEED_IN,
            "eval_seed_ood": TRACK6_EVAL_SEED_OOD,
            "eval_batches": eval_batches,
            "chance_pct": TRACK6_CHANCE,
            "task": "single-step k -> m1 -> m2 -> m3 -> v pointer chain",
        },
        "models": runs,
        "hop_router": {
            "query_position_softmax_mean": router,
            "slots": ["hop_1", "hop_2", "hop_3", "hop_4"],
        },
        "significance_note": (
            "Track 6 is a single seed; no significance test is claimed."
        ),
        "full_vs_transformer_4hop_ood": {
            "full_pct": runs[full_name]["ood_accuracy_pct"],
            "transformer_pct": runs[transformer_name]["ood_accuracy_pct"],
            "note": "single seed; controlled comparison only",
        },
        "depth_reference": {
            "two_hop": {
                "full_v9_ood_pct": next(
                    (
                        row["ood_mean"]
                        for row in v9_track4a.get("summary", [])
                        if row["name"] == "TriCore Full"
                    ),
                    None,
                ),
                "one_hop_limit_ood_pct": two_hop_reference["ood_accuracy_pct"],
                "transformer_v9_ood_pct": next(
                    (
                        row["ood_mean"]
                        for row in v9_track4a.get("summary", [])
                        if row["name"] == "Vanilla Transformer 2-Layer"
                    ),
                    None,
                ),
            },
            "three_hop_v7": {
                "full_ood_pct": 100.0,
                "two_hop_limit_ood_pct": 23.91,
                "transformer_ood_pct": 23.44,
            },
            "four_hop_v11": {
                "full_ood_pct": runs[full_name]["ood_accuracy_pct"],
                "three_hop_limit_ood_pct": runs[shallow_name]["ood_accuracy_pct"],
                "transformer_ood_pct": runs[transformer_name]["ood_accuracy_pct"],
            },
            "two_hop_shallow_probe": two_hop_reference,
        },
        "track4a_welch_vs_full": track4a_welch(v9_track4a),
    }


# ---------------------------------------------------------------------------
# Module D: Track 4B Extended, real CUDA state and KV-cache profiling
# ---------------------------------------------------------------------------


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


def measure_tricore_latency(model, state, device, length, reps, warmups):
    idx_t = torch.tensor([7], device=device, dtype=torch.long)
    base_step = max(length - reps - warmups - 8, 0)
    with torch.no_grad():
        for step in range(warmups):
            _, _, state = model.forward_step(
                idx_t, state, base_step + step
            )
        sync_device(device)
        started = time.perf_counter()
        for step in range(reps):
            _, _, state = model.forward_step(
                idx_t, state, base_step + warmups + step
            )
        sync_device(device)
    return (time.perf_counter() - started) * 1000.0 / reps


def measure_transformer_latency(
    model, k_cache, v_cache, length, device, reps, warmups
):
    idx_t = torch.tensor([7], device=device, dtype=torch.long)
    pos = length - 1
    with torch.no_grad():
        for _ in range(warmups):
            cached_transformer_step(model, idx_t, pos, k_cache, v_cache)
        sync_device(device)
        started = time.perf_counter()
        for _ in range(reps):
            cached_transformer_step(model, idx_t, pos, k_cache, v_cache)
        sync_device(device)
    return (time.perf_counter() - started) * 1000.0 / reps


def run_track4b(device: str, lengths=PROFILE_LENGTHS) -> dict:
    tri = TriCoreLM(
        vocab_size=64,
        d_model=PROFILE_D_MODEL,
        n_layers=1,
        d_fast=PROFILE_D_HEAD,
        max_loops=2,
    ).to(device)
    baseline = BaselineTransformerLM(
        vocab_size=64,
        d_model=PROFILE_D_MODEL,
        n_layers=2,
        n_heads=PROFILE_HEADS,
        max_seq_len=max(lengths),
    ).to(device)
    tri.eval()
    baseline.eval()

    rows = []
    for length in lengths:
        high = length >= 65_536
        reps = 16 if high else 32
        warmups = 4 if high else 8
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
            "Track4B L={:>6} state={:>6} B cache={:>9} B "
            "tri={:.3f} ms/token tf={:.3f} ms/token redux={:.1f}x".format(
                length,
                state_tensor_bytes,
                cache_tensor_bytes,
                tri_ms,
                transformer_ms,
                row["reduction"],
            ),
            flush=True,
        )
        del state, caches, k_cache, v_cache
        if device == "cuda":
            torch.cuda.empty_cache()

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
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Module E: six-panel master figure
# ---------------------------------------------------------------------------

TRACK2_DEREF = {"tricore": 100.00, "transformer": 19.63}
TRACK2_PPL = {"tricore": 3.17, "transformer": 3.56}
TRACK3B = {
    "tricore": {"ppl": 18.616, "next_byte": 41.47},
    "transformer": {"ppl": 19.428, "next_byte": 28.15},
}


def _short_name(name: str) -> str:
    replacements = {
        "TriCore Full": "TriCore Full",
        "No Engine I: Surprise Gate": "No-I",
        "No Engine II: Fast-Weight": "No-II",
        "No Engine III: Adaptive Loop": "No-III",
        "Vanilla Transformer 2-Layer": "Transformer",
    }
    return replacements.get(name, name)


def plot_masterpiece(metrics: dict, output_paths=(FIGURE_PATH, MASTERPIECE_PATH)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
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
    }
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=180)

    # (0, 0) Depth scaling: 2-hop / 3-hop / 4-hop under 4x OOD gap.
    depth = metrics["track6"]["depth_reference"]
    hops = ["2-Hop\n(Track 1)", "3-Hop\n(Track 3A)", "4-Hop\n(Track 6)"]
    full = [
        depth["two_hop"]["full_v9_ood_pct"],
        depth["three_hop_v7"]["full_ood_pct"],
        depth["four_hop_v11"]["full_ood_pct"],
    ]
    shallow = [
        depth["two_hop"]["one_hop_limit_ood_pct"],
        depth["three_hop_v7"]["two_hop_limit_ood_pct"],
        depth["four_hop_v11"]["three_hop_limit_ood_pct"],
    ]
    transformer = [
        depth["two_hop"]["transformer_v9_ood_pct"],
        depth["three_hop_v7"]["transformer_ood_pct"],
        depth["four_hop_v11"]["transformer_ood_pct"],
    ]
    x = np.arange(len(hops))
    width = 0.26
    ax = axes[0, 0]
    b1 = ax.bar(
        x - width,
        full,
        width,
        color=colors["tricore"],
        label="TriCore matched depth",
    )
    b2 = ax.bar(
        x,
        shallow,
        width,
        color=colors["shallow"],
        label="TriCore one-hop shallow limit",
    )
    b3 = ax.bar(
        x,
        transformer,
        width,
        color=colors["transformer"],
        label="2-layer Transformer",
    )
    ax.axhline(
        TRACK6_CHANCE,
        color=colors["red"],
        linestyle="--",
        linewidth=1.2,
        label="25.0% chance",
    )
    ax.set_xticks(x, hops)
    ax.set_ylim(0, 150)
    ax.set_ylabel("Recall at 4x distractor gap (%)")
    ax.set_title("(a) Multi-hop depth scaling law")
    for bar in b1:
        height = bar.get_height()
        if height is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=0,
        )
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=7.8)

    # (0, 1) Track 5: three-seed infix EQ dereference and Welch p-values.
    t5 = metrics["track5"]
    names = list(t5["models"].keys())
    labels = ["TriCore\nFull", "Fair\nNo-Engine-I", "2-Layer\nTransformer"]
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
    ax.set_title("(b) Track 5: EQ dereference (3-seed mean ± std)")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 3,
            f"{mean:.2f} ± {std:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    sig = t5["significance"]
    p_fair = sig["full_vs_fair_no_engine_i_accuracy"]["p_value_two_tailed"]
    p_tf = sig["full_vs_transformer_accuracy"]["p_value_two_tailed"]
    ax.text(
        0.02,
        0.94,
        f"Welch p vs Full: Fair {fmt_p(p_fair)} | Transformer {fmt_p(p_tf)}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    full_model = t5["models"][names[0]]
    gate_ax = ax.inset_axes([0.60, 0.50, 0.36, 0.34])
    gate_values = [
        full_model["elig_gate_eq_mean"] or 0.0,
        full_model["elig_gate_entity_mean"] or 0.0,
    ]
    gate_ax.bar(
        ["EQ", "Entity"],
        gate_values,
        color=[colors["red"], colors["tricore"]],
        alpha=0.85,
    )
    gate_ax.set_ylim(0, 1.08)
    gate_ax.set_title("Eligibility gate", fontsize=7.5, pad=2)
    gate_ax.tick_params(labelsize=7)
    gate_ax.grid(False)

    # (0, 2) Track 4A: five-way 2-hop ablation over three seeds.
    track4a = metrics.get("track4a_reference", {})
    summary = track4a.get("summary", [])
    labels = [_short_name(row["name"]) for row in summary]
    in_means = [row["in_mean"] for row in summary]
    in_stds = [row["in_std"] for row in summary]
    ood_means = [row["ood_mean"] for row in summary]
    ood_stds = [row["ood_std"] for row in summary]
    ax = axes[0, 2]
    x = np.arange(len(summary))
    width = 0.36
    ax.bar(
        x - width / 2,
        in_means,
        width,
        yerr=in_stds,
        capsize=4,
        color=colors["blue"],
        label="In-domain",
    )
    ax.bar(
        x + width / 2,
        ood_means,
        width,
        yerr=ood_stds,
        capsize=4,
        color=colors["tricore_light"],
        label="4x OOD gap",
    )
    ax.axhline(25.0, color=colors["red"], linestyle="--", linewidth=1.1)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(c) Track 4A: 4-chain MQAR, 3-seed ablation")
    ax.legend(frameon=False, loc="upper right")

    # (1, 0) Real CUDA memory scaling to 128K.
    profile = metrics["track4b"]
    rows = profile["rows"]
    lengths = [row["length"] for row in rows]
    state_kib = [row["state_tensor_bytes"] / 1024.0 for row in rows]
    cache_kib = [row["cache_tensor_bytes"] / 1024.0 for row in rows]
    ax = axes[1, 0]
    ax.loglog(
        lengths,
        cache_kib,
        marker="o",
        linewidth=2.0,
        color=colors["transformer"],
        label="Transformer causal K+V cache",
    )
    ax.loglog(
        lengths,
        state_kib,
        marker="s",
        linewidth=2.2,
        color=colors["tricore"],
        label="TriCore O(1) state",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length L (tokens, log scale)")
    ax.set_ylabel("Decode memory (KiB, log scale)")
    ax.set_title("(d) Real CUDA memory scaling, L=128 to 131,072")
    ax.grid(True, which="both", alpha=0.18)
    final = rows[-1]
    ax.annotate(
        f"{final['cache_tensor_bytes'] / 1024 / 1024:.2f} MiB\n"
        f"({final['cache_tensor_bytes'] / 1024:.0f} KiB)",
        xy=(final["length"], final["cache_tensor_bytes"] / 1024.0),
        xytext=(-116, 10),
        textcoords="offset points",
        fontsize=8,
        color=colors["transformer"],
        arrowprops={"arrowstyle": "-", "color": colors["transformer"]},
    )
    ax.annotate(
        f"{final['state_tensor_bytes'] / 1024:.2f} KiB\n"
        f"{final['reduction']:,.1f}x reduction",
        xy=(final["length"], final["state_tensor_bytes"] / 1024.0),
        xytext=(-116, 26),
        textcoords="offset points",
        fontsize=8,
        color=colors["tricore"],
        arrowprops={"arrowstyle": "-", "color": colors["tricore"]},
    )
    ax.legend(frameon=False, loc="upper left")

    # (1, 1) Real CUDA decoding latency.
    tri_ms = [row["tri_ms_per_token"] for row in rows]
    tf_ms = [row["transformer_ms_per_token"] for row in rows]
    ax = axes[1, 1]
    ax.plot(
        lengths,
        tf_ms,
        marker="o",
        linewidth=2.0,
        color=colors["transformer"],
        label="Transformer cached decode",
    )
    ax.plot(
        lengths,
        tri_ms,
        marker="s",
        linewidth=2.2,
        color=colors["tricore"],
        label="TriCore forward_step",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length L (tokens, log2 scale)")
    ax.set_ylabel("Decode latency (ms / token)")
    ax.set_title("(e) Real CUDA decoding latency to 128K")
    ax.grid(True, which="both", alpha=0.18)
    crossing = min(
        (
            row
            for row in rows
            if row["transformer_ms_per_token"] > row["tri_ms_per_token"]
        ),
        key=lambda row: row["length"],
        default=None,
    )
    if crossing is not None:
        ax.axvline(
            crossing["length"],
            color=colors["red"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.text(
            crossing["length"] * 1.03,
            max(tf_ms) * 0.50,
            f"crossover\nL={crossing['length']:,}",
            fontsize=8,
            color=colors["red"],
        )
    ax.annotate(
        f"L={final['length']:,}: {final['transformer_ms_per_token']:.2f} ms",
        xy=(final["length"], final["transformer_ms_per_token"]),
        xytext=(-132, 8),
        textcoords="offset points",
        fontsize=8,
        color=colors["transformer"],
        arrowprops={"arrowstyle": "-", "color": colors["transformer"]},
    )
    ax.annotate(
        f"L={final['length']:,}: {final['tri_ms_per_token']:.2f} ms",
        xy=(final["length"], final["tri_ms_per_token"]),
        xytext=(-132, 20),
        textcoords="offset points",
        fontsize=8,
        color=colors["tricore"],
        arrowprops={"arrowstyle": "-", "color": colors["tricore"]},
    )
    ax.legend(frameon=False, loc="upper left")

    # (1, 2) Code language-modeling results retained from V6/V7.
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
        x - width / 2,
        tri_scores,
        width,
        color=colors["tricore"],
        label="TriCore",
    )
    ax.bar(
        x + width / 2,
        tf_scores,
        width,
        color=colors["transformer"],
        label="Transformer",
    )
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(f) Track 2 & Track 3B code modeling")
    for index, (tri_value, tf_value) in enumerate(zip(tri_scores, tf_scores)):
        ax.text(
            index - width / 2,
            tri_value + 2,
            f"{tri_value:.1f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            index + width / 2,
            tf_value + 2,
            f"{tf_value:.1f}",
            ha="center",
            fontsize=8,
        )
    ppl_ax = ax.inset_axes([0.56, 0.52, 0.39, 0.34])
    ppl_values = [
        TRACK3B["tricore"]["ppl"],
        TRACK3B["transformer"]["ppl"],
    ]
    ppl_ax.bar(
        [0, 1],
        ppl_values,
        color=[colors["tricore"], colors["transformer"]],
    )
    ppl_ax.set_xticks([0, 1], ["Tri", "TF"])
    ppl_ax.set_ylim(0, 24)
    ppl_ax.set_title("Track 3B PPL", fontsize=7.5, pad=2)
    ppl_ax.tick_params(labelsize=7)
    ppl_ax.grid(False)
    for index, value in enumerate(ppl_values):
        ppl_ax.text(
            index, value + 0.3, f"{value:.2f}", ha="center", fontsize=6.5
        )
    ax.text(
        0.02,
        0.94,
        "Track 2 PPL: 3.17 vs 3.56",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
    )
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "TriCore V11 — 4-hop latent chaining, 3-seed significance, "
        "128K CUDA scaling, and constant-state decoding",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Single machine, single seed for Track 6 / Track 4B; Track 5 is "
        "mean ± std over seeds 42/100/2026 with exact two-tailed Welch "
        "p-values. Track 6 and Track 4B are controlled comparisons, not "
        "population claims.",
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
    parser = argparse.ArgumentParser(description="Run the TriCore V11 flagship suite")
    parser.add_argument("--steps5", type=int, default=TRACK5_STEPS)
    parser.add_argument("--steps6", type=int, default=TRACK6_STEPS)
    parser.add_argument("--eval-batches6", type=int, default=TRACK6_EVAL_BATCHES)
    parser.add_argument(
        "--max-length",
        type=int,
        default=max(PROFILE_LENGTHS),
        help="largest Track 4B length to profile",
    )
    parser.add_argument("--skip-track5", action="store_true")
    parser.add_argument("--skip-track6", action="store_true")
    parser.add_argument("--skip-track4b", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TriCore V11 requires CUDA for the real 128K decode-state profile"
        )
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_path = Path(args.metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"=== TriCore V11 flagship on {torch.cuda.get_device_name(0)} ===",
        flush=True,
    )
    started = time.perf_counter()
    metrics = {
        "version": "1.4.0",
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "started_at_unix": time.time(),
        "protocol": {
            "track5_steps": args.steps5,
            "track6_steps": args.steps6,
            "seeds": list(SEEDS),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "track4a_reference": load_v9_track4a(),
        "track4a_welch_vs_full": track4a_welch(load_v9_track4a()),
    }

    if not args.skip_track5:
        print("\n--- Track 5: three-seed infix EQ dereference ---", flush=True)
        metrics["track5"] = run_track5(
            device, seeds=SEEDS, steps=args.steps5
        )
        write_metrics(metrics, output_path)
        print(f"Checkpoint written to {output_path}", flush=True)

    if not args.skip_track6:
        print("\n--- Track 6: four-hop depth scaling ---", flush=True)
        metrics["track6"] = run_track6(
            device, steps=args.steps6, eval_batches=args.eval_batches6
        )
        write_metrics(metrics, output_path)
        print(f"Checkpoint written to {output_path}", flush=True)

    if not args.skip_track4b:
        print("\n--- Track 4B: real CUDA profiling ---", flush=True)
        lengths = tuple(
            length for length in PROFILE_LENGTHS if length <= args.max_length
        )
        if not lengths:
            raise ValueError("--max-length excludes every profiling length")
        metrics["track4b"] = run_track4b(device, lengths=lengths)
        write_metrics(metrics, output_path)
        print(f"Checkpoint written to {output_path}", flush=True)

    metrics["runtime_seconds"] = time.perf_counter() - started
    if not args.skip_plot:
        figures = plot_masterpiece(metrics)
        metrics["figures"] = figures
        print("Wrote figures: " + ", ".join(figures), flush=True)
    write_metrics(metrics, output_path)

    print(
        f"\nV11_BENCHMARK_COMPLETED_SUCCESSFULLY in "
        f"{metrics['runtime_seconds']:.1f}s",
        flush=True,
    )
    if "track5" in metrics:
        for name, payload in metrics["track5"]["models"].items():
            print(
                f"TRACK5 | {name} | "
                f"{payload['accuracy_mean']:.2f} ± "
                f"{payload['accuracy_std']:.2f}%",
                flush=True,
            )
        for label, result in metrics["track5"]["significance"].items():
            if "accuracy" in label:
                print(
                    f"WELCH | {label} | t={result['t_statistic']:.3f} | "
                    f"df={result['degrees_of_freedom']:.3f} | "
                    f"p={fmt_p(result['p_value_two_tailed'])}",
                    flush=True,
                )
    if "track6" in metrics:
        for name, payload in metrics["track6"]["models"].items():
            print(
                f"TRACK6 | {name} | in={payload['in_accuracy_pct']:.2f}% | "
                f"4x-OOD={payload['ood_accuracy_pct']:.2f}%",
                flush=True,
            )
        print(
            "TRACK6 | hop router at query = "
            f"{metrics['track6']['hop_router']['query_position_softmax_mean']}",
            flush=True,
        )
    if "track4b" in metrics:
        final = metrics["track4b"]["rows"][-1]
        print(
            f"TRACK4B | L={final['length']} | state={final['state_tensor_bytes']} B | "
            f"KV={final['cache_tensor_bytes']} B | "
            f"reduction={final['reduction']:.1f}x | "
            f"tri={final['tri_ms_per_token']:.3f} ms/token | "
            f"tf={final['transformer_ms_per_token']:.3f} ms/token",
            flush=True,
        )
    print(f"Metrics: {output_path}", flush=True)


if __name__ == "__main__":
    main()

