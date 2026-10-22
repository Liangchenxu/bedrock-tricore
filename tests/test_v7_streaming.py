import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from tricore import (
    TriCoreLM,
    generate_3hop_recall_batch,
    generate_4hop_recall_batch,
    load_real_python_corpus_batches,
)


def _iter_state_tensors(states):
    if torch.is_tensor(states):
        yield states
        return
    for item in states:
        yield from _iter_state_tensors(item)


def _state_bytes(states):
    return sum(t.numel() * t.element_size() for t in _iter_state_tensors(states))


def test_streaming_forward_step_matches_full_forward():
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = TriCoreLM(
        vocab_size=64,
        d_model=128,
        n_layers=1,
        d_fast=32,
        max_loops=2,
    ).to(device)
    model.eval()

    batch_size, seq_len = 2, 48
    idx = torch.randint(0, 64, (batch_size, seq_len), device=device)

    with torch.no_grad():
        full_logits, full_surprise = model(idx)
        states = model.init_state(batch_size, device, torch.float32)
        step_logits = []
        step_surprises = []
        for t in range(seq_len):
            logits_t, surprise_t, states = model.forward_step(idx[:, t], states, t)
            step_logits.append(logits_t)
            step_surprises.append(surprise_t)

        step_logits = torch.stack(step_logits, dim=1)
        step_surprises = torch.stack(step_surprises, dim=1).squeeze(-1)

    assert full_logits.shape == (batch_size, seq_len, 64)
    assert step_logits.shape == full_logits.shape
    assert full_surprise.shape == (batch_size, seq_len)
    assert step_surprises.shape == full_surprise.shape
    assert torch.allclose(full_logits, step_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full_surprise, step_surprises, atol=1e-5, rtol=1e-5)


def test_streaming_state_is_constant_17_kib_per_sample():
    torch.manual_seed(11)
    device = torch.device("cpu")
    batch_size = 2
    model = TriCoreLM(
        vocab_size=64,
        d_model=128,
        n_layers=1,
        d_fast=32,
        max_loops=2,
    ).to(device)
    model.eval()

    states = model.init_state(batch_size, device, torch.float32)
    expected_bytes = batch_size * 17_408
    observed = {}
    with torch.no_grad():
        for t in range(512):
            _, _, states = model.forward_step(
                torch.zeros(batch_size, dtype=torch.long, device=device),
                states,
                t,
            )
            if t + 1 in (16, 512):
                observed[t + 1] = _state_bytes(states)

    assert observed[16] == expected_bytes, observed
    assert observed[512] == expected_bytes, observed
    assert all(value == expected_bytes for value in observed.values())


def test_three_hop_and_real_python_corpus_shapes_and_gradients():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y = generate_3hop_recall_batch(
        batch_size=4,
        num_chains=4,
        num_queries=4,
        vocab_size=64,
        noise_len=12,
        device=device,
    )
    # 3 tiers of facts: 3 * 4 * 3 tokens, plus 12 noise and 8 query tokens.
    assert x.shape == (4, 56), x.shape
    assert y.shape == (4, 56), y.shape
    query_mask = y != -100
    assert int(query_mask.sum().item()) == 16
    assert torch.all(y[query_mask] >= 46)
    assert torch.all(y[query_mask] < 61)

    model = TriCoreLM(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        d_fast=8,
        max_loops=2,
    ).to(device)
    logits, surprise = model(x)
    assert logits.shape == (4, 56, 64)
    assert surprise.shape == (4, 56)
    loss = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
    loss.backward()
    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(torch.tensor(value)) for value in grads)

    sample_train_batch, iter_val_batches, vocab_size = load_real_python_corpus_batches(
        batch_size=8,
        seq_len=96,
        device=device,
    )
    assert vocab_size == 128
    train_x, train_y = sample_train_batch()
    assert train_x.shape == (8, 96), train_x.shape
    assert train_y.shape == (8, 96), train_y.shape
    assert int(train_x.min().item()) >= 0
    assert int(train_x.max().item()) < 128

    val_batches = list(iter_val_batches())
    assert val_batches
    val_x, val_y = val_batches[0]
    assert val_x.shape[1] == 96, val_x.shape
    assert val_y.shape == val_x.shape

    code_model = TriCoreLM(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        d_fast=8,
        max_loops=2,
    ).to(device)
    code_logits, _ = code_model(train_x)
    code_loss = F.cross_entropy(
        code_logits.view(-1, 128), train_y.view(-1), ignore_index=-100
    )
    code_loss.backward()
    code_grads = [
        p.grad.norm().item() for p in code_model.parameters() if p.grad is not None
    ]
    assert code_grads and all(
        torch.isfinite(torch.tensor(value)) for value in code_grads
    )


def test_v11_4hop_recall_and_gradients():
    torch.manual_seed(13)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y = generate_4hop_recall_batch(
        batch_size=4,
        num_chains=4,
        num_queries=4,
        vocab_size=96,
        noise_len=12,
        device=device,
    )
    # Four hops of 4 * 3 tokens, 12 noise tokens, and 4 * 2 query tokens.
    assert x.shape == (4, 68), x.shape
    assert y.shape == (4, 68), y.shape
    query_mask = y != -100
    assert int(query_mask.sum().item()) == 16
    targets = y[query_mask]
    assert torch.all(targets >= 77)
    assert torch.all(targets < 96)

    model = TriCoreLM(
        vocab_size=96,
        d_model=64,
        n_layers=1,
        d_fast=16,
        max_loops=3,
        separator_token=0,
        syntax_tokens=(0,),
        reset_trace_tokens=(0,),
    ).to(device)
    logits, surprise = model(x)
    assert logits.shape == (4, 68, 96)
    assert surprise.shape == (4, 68)
    loss = F.cross_entropy(logits.view(-1, 96), y.view(-1), ignore_index=-100)
    loss.backward()
    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(torch.tensor(value)) for value in grads)

    model.eval()
    with torch.no_grad():
        full_logits, full_surprise = model(x)
        states = model.init_state(x.shape[0], device, torch.float32)
        step_logits = []
        step_surprises = []
        for t in range(x.shape[1]):
            logits_t, surprise_t, states = model.forward_step(
                x[:, t], states, t
            )
            step_logits.append(logits_t)
            step_surprises.append(surprise_t)
        step_logits = torch.stack(step_logits, dim=1)
        step_surprises = torch.stack(step_surprises, dim=1).squeeze(-1)

    # The reset-mask path reassociates the final normalize multiply, so allow
    # one extra float32 ulp relative to the plain streaming parity check.
    assert torch.allclose(full_logits, step_logits, atol=1e-4, rtol=1e-4)
    assert torch.allclose(full_surprise, step_surprises, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_streaming_forward_step_matches_full_forward()
    test_streaming_state_is_constant_17_kib_per_sample()
    test_three_hop_and_real_python_corpus_shapes_and_gradients()
    test_v11_4hop_recall_and_gradients()
    print(
        "[PASS] V7 streaming, O(1) state, 3-hop, 4-hop and real-corpus checks passed."
    )
