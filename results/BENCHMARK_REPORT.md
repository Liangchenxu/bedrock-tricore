# TriCore Experimental Benchmark Report

- **Compute Device:** `CUDA`

- **Task:** Multi-Query Associative Recall (MQAR) with Distractor Noise & 4x Length Extrapolation


## 1. Accuracy & Ablation Results

| Architecture Variant | Parameters | Final Val Loss | In-Domain Recall Acc (%) | 4x Length Extrapolation Acc (%) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore (Full 3-in-1)** | 348,432 | 0.0012 | **100.00%** | **99.45%** | 129.5s |
| **TriCore (No Fast-Weight)** | 348,432 | 3.4330 | **3.28%** | **3.83%** | 60.6s |
| **TriCore (No Adaptive Loop)** | 348,432 | 0.0012 | **100.00%** | **99.84%** | 126.0s |
| **Vanilla Transformer** | 543,744 | 2.8272 | **13.12%** | **13.05%** | 16.4s |

## 2. Inference State Cache Scaling (Per Sample)

| Sequence Length | Transformer KV Cache (KB) | TriCore O(1) State (KB) | Memory Reduction Factor |
| :---: | :---: | :---: | :---: |
| 128 | 256.00 KB | 33.00 KB | **7.8x smaller** |
| 512 | 1024.00 KB | 33.00 KB | **31.0x smaller** |
| 2,048 | 4096.00 KB | 33.00 KB | **124.1x smaller** |
| 8,192 | 16384.00 KB | 33.00 KB | **496.5x smaller** |
| 32,768 | 65536.00 KB | 33.00 KB | **1985.9x smaller** |
