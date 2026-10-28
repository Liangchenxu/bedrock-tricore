import argparse
import time

import torch

from . import __version__
from .model import TriCoreLM


def _make_model() -> TriCoreLM:
    return TriCoreLM(
        vocab_size=64,
        d_model=128,
        n_layers=1,
        d_fast=32,
        max_loops=2,
    ).eval()


def _state_bytes(states) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer_state in states
        for tensor in layer_state
    )


def cmd_info(_args) -> int:
    model = _make_model()
    states = model.init_state(1, torch.device("cpu"), torch.float32)
    fp32_bytes = _state_bytes(states)
    bf16_bytes = _state_bytes(
        [tuple(t.to(torch.bfloat16) for t in layer) for layer in states]
    )
    print(f"bedrock-tricore v{__version__}")
    print("Architecture: attention-free constant-state O(1) sequence operator")
    print("Engines: I surprise/eligibility | II causal shifted Delta rule | III latent multi-hop")
    print("Reference: d_model=128, n_layers=1, n_heads=4, d_head=32, max_loops=2")
    print(f"Persistent state FP32: {fp32_bytes:,} B / {fp32_bytes / 1024:.2f} KiB per sample")
    print(f"Persistent state BF16: {bf16_bytes:,} B / {bf16_bytes / 1024:.2f} KiB per sample")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return 0


def cmd_verify(_args) -> int:
    try:
        torch.manual_seed(17)
        model = _make_model()
        batch_size, length, chunk_size = 2, 19, 7
        idx = torch.randint(0, 64, (batch_size, length))
        with torch.no_grad():
            states = model.init_state(batch_size, torch.device("cpu"), torch.float32)
            if _state_bytes(states) != batch_size * 17_408:
                raise AssertionError("FP32 17,408 B/sample invariant failed")

            chunk_logits, chunk_surprise, chunk_states = model.forward_chunked(
                idx, chunk_size=chunk_size, return_states=True
            )
            step_states = model.init_state(
                batch_size, torch.device("cpu"), torch.float32
            )
            step_logits, step_surprises = [], []
            for t in range(length):
                logits_t, surprise_t, step_states = model.forward_step(
                    idx[:, t], step_states, t
                )
                step_logits.append(logits_t)
                step_surprises.append(surprise_t)
            step_logits = torch.stack(step_logits, dim=1)
            step_surprises = torch.stack(step_surprises, dim=1)
            if not torch.allclose(chunk_logits, step_logits, atol=1e-5, rtol=1e-5):
                raise AssertionError("forward_chunked logits differ from forward_step")
            if not torch.allclose(chunk_surprise, step_surprises, atol=1e-5, rtol=1e-5):
                raise AssertionError("forward_chunked surprise differs from forward_step")
            for chunk_layer, step_layer in zip(chunk_states, step_states):
                for chunk_tensor, step_tensor in zip(chunk_layer, step_layer):
                    if not torch.allclose(chunk_tensor, step_tensor, atol=1e-5, rtol=1e-5):
                        raise AssertionError("chunked state differs from forward_step")

            fp32_snapshot = model.save_state_snapshot(chunk_states)
            bf16_snapshot = model.save_state_snapshot(
                chunk_states, compress_bf16=True
            )
            if fp32_snapshot["state_bytes_per_sample"] != 17_408:
                raise AssertionError("FP32 snapshot byte accounting failed")
            if bf16_snapshot["state_bytes_per_sample"] != 8_704:
                raise AssertionError("BF16 snapshot byte accounting failed")
            loaded = model.load_state_snapshot(fp32_snapshot)
            next_tokens = torch.tensor([3, 4])
            logits_a, surprise_a, _ = model.forward_step(
                next_tokens, chunk_states, length
            )
            logits_b, surprise_b, _ = model.forward_step(next_tokens, loaded, length)
            if not torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5):
                raise AssertionError("snapshot-restored logits differ")
            if not torch.allclose(surprise_a, surprise_b, atol=1e-5, rtol=1e-5):
                raise AssertionError("snapshot-restored surprise differs")
        print("ALL INVARIANTS VERIFIED")
        return 0
    except Exception as exc:
        print(f"VERIFY FAILED: {type(exc).__name__}: {exc}")
        return 1


def cmd_benchmark(args) -> int:
    lengths = [length for length in args.lengths if length > 0]
    if not lengths:
        print("lengths must contain at least one positive integer")
        return 1
    model = _make_model()
    print(f"bedrock-tricore v{__version__} local benchmark")
    print(f"{'length':>8} {'latency_ms':>12} {'state_B/sample':>16}")
    with torch.no_grad():
        for length in lengths:
            idx = torch.randint(0, 64, (1, length))
            model.forward_chunked(idx, chunk_size=min(256, length))
            start = time.perf_counter()
            _, _, states = model.forward_chunked(
                idx, chunk_size=min(256, length), return_states=True
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{length:>8} {elapsed_ms:>12.3f} {_state_bytes(states):>16}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bedrock-tricore",
        description="Bedrock-TriCore O(1) sequence operator CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="show architecture and state ledger")
    info_parser.set_defaults(func=cmd_info)

    verify_parser = subparsers.add_parser("verify", help="run fast invariant checks")
    verify_parser.set_defaults(func=cmd_verify)

    benchmark_parser = subparsers.add_parser("benchmark", help="measure local state and latency")
    benchmark_parser.add_argument(
        "--lengths", nargs="+", type=int, default=[128, 512, 2048]
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

