import time

import torch
import torch.nn.functional as F

from run_v10_eligibility import generate_infix_eq_batch
from tricore import TriCoreLM


start = time.perf_counter()
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
x, y = generate_infix_eq_batch(device="cpu")
logits, _ = model(x)
loss = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
loss.backward()
open("_smoke_ok", "w").write(f"loss={loss.item():.4f} secs={time.perf_counter()-start:.2f}")
print(f"SMOKE_OK loss={loss.item():.4f}", flush=True)
