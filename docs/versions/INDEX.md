# Bedrock-TriCore Version Vault

This directory is the append-only release ledger required by `.clinerules` rule 5.
Each release gets its own file; previous files are never overwritten.

## Current package releases

| Version | Engineering record | Commit | Core changes | Typical issue solved |
| :--- | :--- | :--- | :--- | :--- |
| **v1.6.4** | [v1.6.4_chunked_snapshot_cli.md](v1.6.4_chunked_snapshot_cli.md) | this release | `forward_chunked()`, FP32/BF16 state snapshots, `bedrock-tricore` CLI, 14 tests | chunk-boundary t=0 write, full-sequence mask slicing, snapshot SHA-256 |
| **v1.6.3** | [v1.6.3_blueprint_and_telemetry.md](v1.6.3_blueprint_and_telemetry.md) | `0ba497d` | 220-DPI architecture blueprint, `generate()`, `inspect_telemetry()`, 12 tests | ASCII misalignment, missing `_resolve_masks`, gate telemetry drift |
| **v1.6.2** | [v1.6.2_equation_rendering.md](v1.6.2_equation_rendering.md) | `0f2965c` / `e83d92e` | 8 Computer Modern equation PNGs, Unicode inline formulas, 11 tests | PyPI `nh3` sanitizer / no MathJax, ASCII formula misalignment |
| **v1.6.1** | [v1.6.1_pypi_packaging.md](v1.6.1_pypi_packaging.md) | `e789762` | `bedrock-tricore` naming, PyPI/GitHub metadata, Unicode formulas, 11 tests | `\operatorname` incompatibility, duplicate-version 403, SecureString token truncation |
| **v1.6.0** | [v1.6.0_v13_definitive_mutation.md](v1.6.0_v13_definitive_mutation.md) | `1411cfc` | V13 static retention + four-round hot mutation, three-seed Welch tests | single-round Hebbian survived via decay; PowerShell `| Out-File` swallowed OSC 633 |

## Historical V4–V13 slices

These files are raw, token-free extractions from the existing documentation; no
historical section was rewritten.

| Version | File |
| :--- | :--- |
| V13 | [historical_v4_to_v13/V13.md](historical_v4_to_v13/V13.md) |
| V12 | [historical_v4_to_v13/V12.md](historical_v4_to_v13/V12.md) |
| V11 | [historical_v4_to_v13/V11.md](historical_v4_to_v13/V11.md) |
| V10 | [historical_v4_to_v13/V10.md](historical_v4_to_v13/V10.md) |
| V9 | [historical_v4_to_v13/V9.md](historical_v4_to_v13/V9.md) |
| V8 | [historical_v4_to_v13/V8.md](historical_v4_to_v13/V8.md) |
| V7 | [historical_v4_to_v13/V7.md](historical_v4_to_v13/V7.md) |
| V6 | [historical_v4_to_v13/V6.md](historical_v4_to_v13/V6.md) |

## Quick lookup

- **17.00 KiB / 8.50 KiB state ledger** → v1.6.3 setup, v1.6.4 snapshots.
- **Formula rendering** → v1.6.1 Unicode, v1.6.2 PNG boards.
- **Architecture diagram** → v1.6.3 `results/tricore_architecture_blueprint.png`.
- **Chunked state carry** → v1.6.4 `TriCoreBlock.forward_with_state` / `TriCoreLM.forward_chunked`.
- **CLI** → `bedrock-tricore info|verify|benchmark` / `python -m tricore`.
