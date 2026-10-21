import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from tricore import TriCoreLM, generate_contextual_code_lm_batch
from tricore.model import tricore_fused_scan


def _reference_scan(
    u_feat, u_surp_bias, w_gate_logits, elig_gate_logits,
    hop_weights, Q, K, V,
    base_decay, fast_decay, fast_lr,
    hop_matrices,
    enable_surprise_gate, enable_fast_weight, enable_adaptive_loop,
):
    B, L, D = u_feat.shape
    H = Q.shape[2]
    Dh = Q.shape[3]
    h = torch.zeros(B, D, dtype=u_feat.dtype)
    F_mat = torch.zeros(B, H, Dh, Dh, dtype=u_feat.dtype)
    prev_k = torch.zeros(B, H, Dh, 1, dtype=u_feat.dtype)
    h_list, mem_list, surprise_list = [], [], []

    for t in range(L):
        u_t = u_feat[:, t, :]
        pred_error = torch.abs(u_t - h)
        if enable_surprise_gate:
            g_t = torch.sigmoid(u_surp_bias[:, t, :] + pred_error)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            s_4d = surprise_scalar.view(B, 1, 1, 1)
            write_gate = torch.sigmoid(w_gate_logits[:, t] + s_4d - 0.5)
            elig_gate = torch.sigmoid(elig_gate_logits[:, t] + s_4d - 0.5)
            eff_lr = fast_lr * write_gate
        else:
            g_t = torch.full_like(u_t, 0.5)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            elig_gate = torch.ones_like(elig_gate_logits[:, t])
            eff_lr = fast_lr

        h = base_decay * (1.0 - 0.5 * g_t) * h + g_t * u_t
        h_list.append(h)
        surprise_list.append(surprise_scalar)

        if enable_fast_weight:
            k_t = K[:, t].unsqueeze(-1)
            v_t = V[:, t].unsqueeze(-1)
            q_t = Q[:, t].unsqueeze(-1)
            if t > 0:
                v_pred = torch.matmul(F_mat, prev_k)
                synapse_err = v_t - v_pred
                F_mat = fast_decay * F_mat + torch.matmul(
                    eff_lr * synapse_err, prev_k.transpose(-1, -2)
                )
            q_read = torch.nn.functional.normalize(
                elig_gate * q_t + (1.0 - elig_gate) * prev_k,
                p=2.0,
                dim=-2,
            )
            r1 = torch.matmul(F_mat, q_read)
            if enable_adaptive_loop:
                r_accum = hop_weights[:, t, 0].view(B, 1, 1, 1) * r1
                r_curr = r1
                for step in range(hop_matrices.shape[0]):
                    q_next = torch.nn.functional.normalize(
                        r_curr + torch.matmul(hop_matrices[step], r_curr),
                        p=2.0, dim=-2,
                    )
                    r_curr = torch.matmul(F_mat, q_next)
                    r_accum = r_accum + hop_weights[
                        :, t, step + 1
                    ].view(B, 1, 1, 1) * r_curr
                mem_list.append(r_accum.squeeze(-1).reshape(B, H * Dh))
            else:
                mem_list.append(r1.squeeze(-1).reshape(B, H * Dh))
            prev_k = F.normalize(
                (1.0 - elig_gate) * prev_k + elig_gate * k_t,
                p=2.0, dim=-2,
            )

    h_seq = torch.stack(h_list, dim=1)
    surprise_trace = torch.cat(surprise_list, dim=1)
    if enable_fast_weight:
        mem_seq = torch.stack(mem_list, dim=1)
    else:
        mem_seq = torch.zeros(B, L, H * Dh, dtype=u_feat.dtype)
    return h_seq, mem_seq, surprise_trace


def test_jit_scan_matches_reference_without_f_seq():
    torch.manual_seed(7)
    B, L, D, H, Dh = 2, 9, 8, 2, 4
    u_feat = torch.randn(B, L, D)
    u_surp_bias = torch.randn(B, L, D)
    w_gate_logits = torch.randn(B, L, H, 1, 1)
    elig_gate_logits = torch.randn(B, L, H, 1, 1)
    hop_weights = torch.softmax(torch.randn(B, L, 3), dim=-1)
    Q = F.normalize(torch.randn(B, L, H, Dh), dim=-1)
    K = F.normalize(torch.randn(B, L, H, Dh), dim=-1)
    V = torch.randn(B, L, H, Dh)
    base_decay = torch.sigmoid(torch.randn(D))
    fast_decay = torch.sigmoid(torch.randn(1, H, 1, 1))
    fast_lr = torch.sigmoid(torch.randn(1, H, 1, 1))
    hop_matrices = torch.randn(2, 1, 1, H, Dh, Dh) * 0.02

    for surprise_gate, fast_weight, adaptive_loop in [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, False, False),
    ]:
        jit_out = tricore_fused_scan(
            u_feat, u_surp_bias, w_gate_logits, elig_gate_logits,
            hop_weights, Q, K, V,
            base_decay, fast_decay, fast_lr,
            hop_matrices,
            surprise_gate, fast_weight, adaptive_loop, 2,
        )
        ref_out = _reference_scan(
            u_feat, u_surp_bias, w_gate_logits, elig_gate_logits,
            hop_weights, Q, K, V,
            base_decay, fast_decay, fast_lr,
            hop_matrices,
            surprise_gate, fast_weight, adaptive_loop,
        )
        assert len(jit_out) == 3
        for jit_tensor, ref_tensor in zip(jit_out, ref_out):
            assert jit_tensor.shape == ref_tensor.shape, (
                f"shape mismatch for flags=({surprise_gate}, {fast_weight}, "
                f"{adaptive_loop}): {jit_tensor.shape} vs {ref_tensor.shape}"
            )
            assert torch.allclose(jit_tensor, ref_tensor, atol=1e-6, rtol=1e-5), (
                f"JIT/reference mismatch for flags=({surprise_gate}, "
                f"{fast_weight}, {adaptive_loop})"
            )


def test_forward_matches_forward_step_without_f_seq():
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TriCoreLM(
        vocab_size=64, d_model=32, n_layers=1, d_fast=8, max_loops=2
    ).to(device).eval()
    idx = torch.randint(0, 64, (2, 12), device=device)

    with torch.no_grad():
        full_logits, full_surprise = model(idx)
        states = model.init_state(idx.shape[0], device, torch.float32)
        step_logits = []
        step_surprises = []
        for t in range(idx.shape[1]):
            logits, surprise, states = model.forward_step(
                idx[:, t], states, t
            )
            step_logits.append(logits)
            step_surprises.append(surprise.squeeze(-1))
        step_logits = torch.stack(step_logits, dim=1)
        step_surprises = torch.stack(step_surprises, dim=1)

    logit_error = (full_logits - step_logits).abs().max().item()
    surprise_error = (full_surprise - step_surprises).abs().max().item()
    assert logit_error < 1e-5, f"logit error {logit_error:.3e}"
    assert surprise_error < 1e-5, f"surprise error {surprise_error:.3e}"



def test_contextual_code_lm_task_and_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, deref_mask = generate_contextual_code_lm_batch(
        batch_size=4, num_vars=6, num_queries=4, num_nops=4,
        vocab_size=64, device=device,
    )
    assert x.shape == (4, 80), x.shape
    assert y.shape == (4, 80), y.shape
    assert deref_mask.shape == (4, 80), deref_mask.shape
    assert int(deref_mask.sum().item()) == 16
    assert torch.all(y[deref_mask] >= 37)
    assert torch.all(y[deref_mask] < 55)
    assert torch.all(x[deref_mask] >= 1)
    assert torch.all(x[deref_mask] < 19)

    model = TriCoreLM(
        vocab_size=64, d_model=32, n_layers=1, d_fast=8, max_loops=2
    ).to(device)
    logits, surprise = model(x)
    assert logits.shape == (4, 80, 64)
    assert surprise.shape == (4, 80)
    dense = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
    deref = F.cross_entropy(logits[deref_mask], y[deref_mask])
    loss = 0.25 * dense + deref
    loss.backward()
    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(torch.tensor(g)) for g in grads)


def test_eligibility_gate_is_active_and_fair_ablation_keeps_parameters():
    torch.manual_seed(5)
    full_model = TriCoreLM(
        vocab_size=64, d_model=32, n_layers=1, d_fast=8, max_loops=2,
        enable_surprise_gate=True,
    ).eval()
    fair_model = TriCoreLM(
        vocab_size=64, d_model=32, n_layers=1, d_fast=8, max_loops=2,
        enable_surprise_gate=False,
    ).eval()
    fair_model.load_state_dict(full_model.state_dict())
    x = torch.randint(0, 64, (2, 16))
    full, _ = full_model(x)
    fair_no_i, _ = fair_model(x)
    assert full.shape == fair_no_i.shape
    assert full_model.layers[0].elig_gate_proj[-1].weight.numel() > 0
    assert not torch.allclose(full, fair_no_i)


if __name__ == "__main__":
    test_jit_scan_matches_reference_without_f_seq()
    test_forward_matches_forward_step_without_f_seq()
    test_contextual_code_lm_task_and_backward()
    test_eligibility_gate_is_active_and_fair_ablation_keeps_parameters()
    print("[PASS] V10 eligibility-trace fused scan, streaming parity, and code LM checks passed.")

