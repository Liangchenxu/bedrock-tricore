import torch

from tricore import BaselineTransformerLM, TriCoreLM, generate_associative_recall_batch


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = generate_associative_recall_batch(batch_size=8, num_kv_pairs=6, num_queries=4, vocab_size=64, noise_len=16, device=device)
    print(f"task: device={device} shape={tuple(x.shape)} targets={(y != -100).sum().item()}")

    configs = {
        "full": dict(enable_surprise_gate=True, enable_fast_weight=True, enable_adaptive_loop=True),
        "no_engine_i": dict(enable_surprise_gate=False, enable_fast_weight=True, enable_adaptive_loop=True),
        "no_engine_ii": dict(enable_surprise_gate=True, enable_fast_weight=False, enable_adaptive_loop=True),
        "no_engine_iii": dict(enable_surprise_gate=True, enable_fast_weight=True, enable_adaptive_loop=False),
    }
    for name, flags in configs.items():
        model = TriCoreLM(vocab_size=64, d_model=128, n_layers=1, d_fast=32, max_loops=2, **flags).to(device)
        logits, surprise = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100)
        loss.backward()
        assert logits.shape == (8, x.shape[1], 64)
        assert surprise.shape == (8, x.shape[1])
        assert all(p.grad is None or not torch.isnan(p.grad).any() for p in model.parameters())
        print(f"{name}: params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} loss={loss.item():.4f}")

    transformer = BaselineTransformerLM(vocab_size=64, d_model=128, n_layers=2, n_heads=4).to(device)
    logits, surprise = transformer(x)
    assert logits.shape == (8, x.shape[1], 64) and surprise is None
    print(f"transformer: params={sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")

    full = TriCoreLM(vocab_size=64, d_model=128, n_layers=1, d_fast=32, max_loops=2).to(device)
    block = full.layers[0]
    has_attention = any(isinstance(m, torch.nn.MultiheadAttention) for m in full.modules())
    print(f"full_params={sum(p.numel() for p in full.parameters() if p.requires_grad):,} attention_modules={has_attention}")
    print(f"state_shapes: h=[B,{block.d_model}] F=[B,{block.n_heads},{block.d_head},{block.d_head}] prev_k=[B,{block.n_heads},{block.d_head},1]")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
