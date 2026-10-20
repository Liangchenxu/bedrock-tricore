import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from tricore import TriCoreLM, generate_contextual_code_lm_batch
from tricore.model import tricore_fused_scan


def _reference_scan(
    u_feat, u_surp_bias, w_gate_logits, hop_weights, Q, K, V,
    base_decay, fast_decay, fast_lr, hop_proj,
    enable_surprise_gate, enable_fast_weight, enable_adaptive_loop, max_loops,
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
            write_gate = torch.sigmoid(
                w_gate_logits[:, t] + surprise_scalar.view(B, 1, 1, 1) - 0.5
            )
            eff_lr = fast_lr * write_gate
            eff_decay = fast_decay
        else:
            g_t = torch.full_like(u_t, 0.5)
            surprise_scalar = g_t.mean(dim=-1, keepdim=True)
            eff_lr = torch.ones_like(fast_lr)
            eff_decay = torch.ones_like(fast_decay)

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
                F_mat = eff_decay * F_mat + torch.matmul(
                    eff_lr * synapse_err, prev_k.transpose(-1, -2)
                )
            r_hop = torch.matmul(F_mat, q_t)
            if enable_adaptive_loop:
                w_t = hop_weights[:, t]
                r_accum = w_t[:, 0].view(B, 1, 1, 1) * r_hop
                r_curr = r_hop
                for step in range(max_loops):
                    q_next = F.normalize(
                        r_curr + torch.matmul(hop_proj[step], r_curr),
                        p=2.0, dim=-2,
                    )
                    r_curr = torch.matmul(F_mat, q_next)
                    r_accum = r_accum + w_t[:, step + 1].view(B, 1, 1, 1) * r_curr
                recalled = r_accum.squeeze(-1).reshape(B, H * Dh)
            else:
                recalled = r_hop.squeeze(-1).reshape(B, H * Dh)
            mem_list.append(recalled)
            prev_k = k_t

    h_seq = torch.stack(h_list, dim=1)
    surprise_trace = torch.cat(surprise_list, dim=1)
    if enable_fast_weight:
        mem_seq = torch.stack(mem_list, dim=1)
    else:
        mem_seq = torch.zeros(B, L, H * Dh, dtype=u_feat.dtype)
    return h_seq, mem_seq, surprise_trace


def test_jit_scan_matches_reference_for_ablation_flags():
    torch.manual_seed(7)
    B, L, D, H, Dh = 2, 9, 8, 2, 4
    u_feat = torch.randn(B, L, D)
    u_surp_bias = torch.randn(B, L, D)
    w_gate_logits = torch.randn(B, L, H, 1, 1)
    hop_weights = torch.softmax(torch.randn(B, L, 3), dim=-1)
    Q = F.normalize(torch.randn(B, L, H, Dh), dim=-1)
    K = F.normalize(torch.randn(B, L, H, Dh), dim=-1)
    V = torch.randn(B, L, H, Dh)
    base_decay = torch.sigmoid(torch.randn(D))
    fast_decay = torch.sigmoid(torch.randn(1, H, 1, 1))
    fast_lr = torch.sigmoid(torch.randn(1, H, 1, 1))
    hop_proj = torch.randn(2, 1, H, Dh, Dh) * 0.02

    for flags in [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ]:
        jit_out = tricore_fused_scan(
            u_feat, u_surp_bias, w_gate_logits, hop_weights, Q, K, V,
            base_decay, fast_decay, fast_lr, hop_proj,
            flags[0], flags[1], flags[2], 2,
        )
        ref_out = _reference_scan(
            u_feat, u_surp_bias, w_gate_logits, hop_weights, Q, K, V,
            base_decay, fast_decay, fast_lr, hop_proj,
            flags[0], flags[1], flags[2], 2,
        )
        for jit_tensor, ref_tensor in zip(jit_out, ref_out):
            assert torch.allclose(jit_tensor, ref_tensor, atol=1e-6, rtol=1e-5), (
                f"JIT/reference mismatch for flags={flags}"
            )


def test_contextual_code_lm_task_and_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, deref_mask = generate_contextual_code_lm_batch(
        batch_size=4, num_vars=6, num_noise=6, vocab_size=64, device=device
    )
    assert x.shape == (4, 82), x.shape
    assert y.shape == (4, 82), y.shape
    assert deref_mask.shape == (4, 82), deref_mask.shape
    assert int(deref_mask.sum().item()) == 4
    assert torch.all(y[deref_mask] >= 20)
    assert torch.all(y[deref_mask] < 26)

    model = TriCoreLM(
        vocab_size=64, d_model=32, n_layers=1, d_fast=8, max_loops=2
    ).to(device)
    logits, surprise = model(x)
    assert logits.shape == (4, 82, 64)
    assert surprise.shape == (4, 82)
    loss = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
    loss.backward()
    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(torch.tensor(g)) for g in grads)


if __name__ == "__main__":
    test_jit_scan_matches_reference_for_ablation_flags()
    test_contextual_code_lm_task_and_backward()
    print("[PASS] V5 JIT equivalence and Contextual Code LM checks passed.")
