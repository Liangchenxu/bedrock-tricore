import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from tricore import (
    TriCoreLM,
    generate_3hop_recall_batch,
    generate_4hop_recall_batch,
    generate_stateful_mutation_batch,
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


def test_v12_autonomous_and_delta_rule_parity():
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # No token-ID priors: the block must learn roles autonomously.
    model = TriCoreLM(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        d_fast=8,
        max_loops=2,
        separator_token=None,
        syntax_tokens=None,
        reset_trace_tokens=None,
    ).to(device)
    model.eval()

    batch_size, seq_len = 3, 48
    idx = torch.randint(0, 64, (batch_size, seq_len), device=device)
    full_by_mode = {}
    with torch.no_grad():
        for delta_enabled in (True, False):
            model.enable_delta_rule = delta_enabled
            full_logits, full_surprise = model(idx)
            states = model.init_state(batch_size, device, torch.float32)
            step_logits = []
            step_surprises = []
            for t in range(seq_len):
                logits_t, surprise_t, states = model.forward_step(
                    idx[:, t], states, t
                )
                step_logits.append(logits_t)
                step_surprises.append(surprise_t)
            step_logits = torch.stack(step_logits, dim=1)
            step_surprises = torch.stack(step_surprises, dim=1).squeeze(-1)
            assert torch.allclose(
                full_logits, step_logits, atol=1e-5, rtol=1e-5
            ), delta_enabled
            assert torch.allclose(
                full_surprise, step_surprises, atol=1e-5, rtol=1e-5
            ), delta_enabled
            full_by_mode[delta_enabled] = full_logits

    # The Delta erase term must be functionally active, not a dead flag.
    assert not torch.allclose(
        full_by_mode[True], full_by_mode[False], atol=1e-6, rtol=1e-6
    )

    # V13 Track 7 generator: 4 declarations + 2 early static assignments +
    # 2 hot variables x 4 rounds + 4 NOPs + 4 queries = 22 five-token
    # statements, hence a 110-token sequence when num_nops=4.
    x, y = generate_stateful_mutation_batch(
        batch_size=2,
        num_vars=4,
        num_queries=4,
        num_nops=4,
        vocab_size=64,
        device=device,
    )
    assert x.shape == (2, 110), x.shape
    assert y.shape == (2, 110), y.shape
    query_mask = y != -100
    assert int(query_mask.sum().item()) == 8
    assert torch.all(y[query_mask] >= 33)
    assert torch.all(y[query_mask] < 49)

    model.train()
    model.enable_delta_rule = True
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
    loss.backward()
    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(torch.tensor(value)) for value in grads)


def test_autonomous_masks_use_frequency_not_token_ids():
    """Zero-prior discovery must survive an arbitrary vocabulary permutation."""
    from run_v10_eligibility import generate_infix_eq_batch

    torch.manual_seed(23)
    model = TriCoreLM(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        d_fast=4,
        max_loops=1,
        separator_token=None,
        syntax_tokens=None,
        reset_trace_tokens=None,
        autonomous_syntax=True,
    )
    x, _ = generate_infix_eq_batch(
        batch_size=3, num_vars=4, num_queries=4, num_nops=4, device="cpu"
    )
    (elig, write), reset = model._discover_autonomous_masks(x)
    eq_positions = x == 59
    semi_positions = x == 0
    entity_positions = (x >= 1) & (x <= 48)

    assert torch.equal(elig, eq_positions)
    assert torch.equal(write, eq_positions | semi_positions)
    assert torch.equal(reset, semi_positions)
    assert not bool((elig & entity_positions).any())
    assert not bool((reset & ~semi_positions).any())

    # A bijective relabelling changes every ID but not a single discovered mask.
    perm = torch.randperm(64)
    x_relabelled = perm[x]
    (elig_perm, write_perm), reset_perm = model._discover_autonomous_masks(
        x_relabelled
    )
    assert torch.equal(elig_perm, elig)
    assert torch.equal(write_perm, write)
    assert torch.equal(reset_perm, reset)

    # V13's four-round overwrite layout makes a hot variable occur more than
    # four times, so it uses the same frequency rule with a task-scaled entity
    # bound.  Every EQ in this layout must still be discovered as an infix.
    v13_model = TriCoreLM(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        d_fast=4,
        max_loops=1,
        separator_token=None,
        syntax_tokens=None,
        reset_trace_tokens=None,
        autonomous_syntax=True,
        autonomous_entity_max_freq=6,
    )
    x7, y7 = generate_stateful_mutation_batch(
        batch_size=2, num_vars=4, num_queries=4, num_nops=6, device="cpu"
    )
    assert x7.shape == (2, 120), x7.shape
    assert y7.shape == (2, 120), y7.shape
    assert int((y7 != -100).sum().item()) == 8
    (elig7, write7), reset7 = v13_model._discover_autonomous_masks(x7)
    assert torch.equal(elig7, x7 == 59)
    assert torch.equal(write7, (x7 == 59) | (x7 == 0))
    assert torch.equal(reset7, x7 == 0)


if __name__ == "__main__":
    test_streaming_forward_step_matches_full_forward()
    test_streaming_state_is_constant_17_kib_per_sample()
    test_three_hop_and_real_python_corpus_shapes_and_gradients()
    test_v11_4hop_recall_and_gradients()
    test_v12_autonomous_and_delta_rule_parity()
    test_autonomous_masks_use_frequency_not_token_ids()
    print(
        "[PASS] V7 streaming, O(1) state, 3-hop, 4-hop, V12 Delta-rule and "
        "Track 7 mutation checks passed."
    )
