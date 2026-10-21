import time

import torch
import torch.nn.functional as F

import run_v10_eligibility as task
from tricore import TriCoreLM


def mark(text):
    with open("_smoke2_progress", "w", encoding="utf-8") as handle:
        handle.write(text)


mark("imported")
torch.manual_seed(42)
model = TriCoreLM(
    vocab_size=64,
    d_model=64,
    n_layers=1,
    d_fast=16,
    max_loops=2,
    enable_surprise_gate=True,
    enable_fast_weight=True,
    enable_adaptive_loop=True,
)
mark("model")
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=250)
mark("optimizer")
start = time.perf_counter()
for step in range(1, 4):
    x, y = task.generate_infix_eq_batch(device="cpu")
    opt.zero_grad()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    mark(f"step={step} loss={loss.item():.4f}")
mark(f"done secs={time.perf_counter()-start:.2f}")
print("SMOKE2_OK", flush=True)
