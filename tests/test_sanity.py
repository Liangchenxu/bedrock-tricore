import sys
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
