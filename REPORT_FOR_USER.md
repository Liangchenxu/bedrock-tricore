# TriCore V8 完成报告

**日期：** 2026-10-20  
**版本：** `tricore-arch` 1.1.0  
**状态：** V8 完成；V7 原生 O(1) 流式推理、3-Hop 链式推理、真实 Python 源码 Held-Out LM 全部保留；新增 3-Seed 统计稳健性验证、真实 CUDA 解码缓存/显存实测与一键交互式演示  
**实测报告：** `results/BENCHMARK_REPORT.md`  
**图表：** `results/tricore_vs_transformer.png`  
**运行设备：** CUDA

---

## 0. 一页结论

**V8（v1.1.0）结论：** V6/V7 数据全部保留；新增 3 seeds（`42/100/2026`）的紧凑 2-Hop MQAR 统计稳健性验证，以及真实 CUDA 解码状态/KV-cache 分配与延迟实测。Full 在 4× 间隔上为 **100.00 ± 0.00%**，优于 Transformer 的 **77.29 ± 9.88%**、No Engine II 的 **65.73 ± 9.04%** 和 No Engine III 的 **31.56 ± 10.58%**。真实分配下 TriCore 状态恒为 **17,408 B**，`L=2,048` 时 Transformer KV cache 为 **4,194,304 B（240.9×）**。完整 V8 脚本实测 **94.6 s**，未达到 60 秒目标，报告如实保留。

V7 在不改变 V6 数学公式、权重命名和 V6 双赛道数据的前提下，完成了三件事：

1. **原生流式推理：** 为 `TriCoreBlock` / `TriCoreLM` 增加 `init_state` 与 `forward_step`，持久状态严格为 `(h, F_mat, prev_k)`，不携带训练期的 `F_seq`。完整序列与逐 token 解码的最大 logit 误差 **5.96e-07**，最大 surprise 误差 **1.19e-07**；状态恒为 **17,408 bytes = 17.00 KiB / 样本**；4,096 步解码实测 **2.160 ms/token**。见 §4.1。
2. **Track 3A（3-Hop 链式推理）：** 新增 `generate_3hop_recall_batch`，强制 `k_q -> m1_q -> m2_q -> v_q`。Full `max_loops=2` 达到 **100.00% 域内 / 100.00% 四倍间隔外推**；`max_loops=1` 的两跳上限只有 **25.39% / 23.91%**，2 层 Transformer 为 **23.83% / 23.44%**。见 §4.2。
3. **Track 3B（真实 Python 源码 Held-Out LM）：** 新增 `load_real_python_corpus_batches`，直接读取本仓库 `tricore/model.py`、`tricore/tasks.py`、`run_experiment.py` 的 ASCII 字节流，最后 15% 作为严格未见后缀。TriCore **PPL 18.616 / next-byte acc 41.47%**；参数匹配 Transformer **19.428 / 28.15%**。见 §4.3。

**必须同时看到的限制：** 全部结果仍是单 seed。Track 3A 是 4 链、4 查询、12 干扰 token 的合成任务；Track 3B 只使用本仓库的小型源码后缀，不是大规模人类代码基准。Full 在 Track 3A 的 100.00% 不能外推为通用推理能力，Track 3B 的 PPL 优势也不能外推为吞吐或调参后 Transformer 的全面优势；TriCore 在该训练循环中约慢 **38×**（57.7 s vs 1.5 s）。

---

## 1. V6 向量化多跳：训练耗时对比

| 测量项 | 耗时 | 说明 |
| :--- | :---: | :--- |
| V4 Full，Python 逐 token 循环 | 136.0 s | 历史参考运行 |
| V5 Full，TorchScript 融合扫描 | 125.3 s | 历史参考运行 |
| V6 Full，向量化多跳扫描 | **80.0 s** | 计时 500 步 |
| V6 / V4 速度比 | **1.70×** | >1 表示更快，<1 表示更慢 |
| V6 / V5 速度比 | **1.57×** | >1 表示更快，<1 表示更慢 |
| V6 首次调用预热 | 0.4 s | 不计入正式计时 |

计时范围包括数据生成、前向、反向和优化器步。V4 的 136.0 s 和 V5 的 125.3 s 是历史参考运行，不是同机同时测量；V6 的 80.0 s 是本次单机单 seed 墙钟结果，1.70× / 1.57× 是相对历史参考的比率，不是吞吐保证。

**结论：** V6 消除了 Engine III 的逐 token Python 循环，把每次固定跳的重查询变成一次跨 `L` 的批量 5-D `matmul`，因此 Full 从 V5 的 125.3 s 降到 80.0 s（1.57×），相对 V4 为 1.70×。但状态递归仍然串行，V6 不是并行前缀扫描，也不是自定义 CUDA kernel；小矩阵乘法和 kernel launch 仍是主要成本。相较 2 层 Transformer（18.6 s），TriCore 仍慢得多。

---

## 2. Track 1：2-Hop 链式联想召回（五组消融）

**协议：** 500 步，batch 64，`lr=3e-3`，seed 42；6 条链、4 个查询、16 个训练噪声 token；固定评测批次 seed 2026/2027；4× 外推把对抗噪声间隔从 16 拉到 64（总序列 108），不是把总长度乘 4。

| 架构变体 | 参数量 | 最终验证 loss | 域内 2-hop 召回 | 4× 噪声间隔外推 | 训练耗时 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full（三引擎全开）** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 80.0 s |
| TriCore No Engine I：去掉惊奇门控 | 207,503 | 3.0005 | 5.55% | 4.14% | 62.3 s |
| TriCore No Engine II：去掉快权重 | 207,503 | 3.0008 | 5.78% | 3.98% | 36.1 s |
| TriCore No Engine III：去掉自适应循环 | 207,503 | 1.8660 | 16.48% | 17.50% | 63.4 s |
| Vanilla Transformer（2 层） | 543,744 | 2.3131 | 16.72% | 14.92% | 18.6 s |

### Track 1 解读

1. **Full 解决任务。** 域内 100.00%，4× 间隔仍 100.00%，验证 loss 0.0010。
2. **Engine II 是联想记忆本体。** 去掉快权重后 5.78%/3.98%，低于 6 选 1 的 16.67% 随机水平。
3. **Engine III 是第二跳。** 去掉隐空间重查询后 16.48%/17.50%，基本等于随机；第一跳还能取到桥接 token，但无法把它转成最终值。
4. **Engine I 在本次 V6 运行中是必要的。** 去掉写保护后 5.55%/4.14%，直接跌到随机；这与 V5 的 91.88%/84.14% 不同。公式没变，但向量化多跳路径改变了单 seed 优化轨迹，因此不能把任一结果当作普适结论。
5. **Transformer 在本预算下是随机水平。** 16.72%/14.92%；它未调参、未做参数/算力匹配，因此这不是“注意力上限”的证据。
6. **向量化多跳带来实质加速。** Full 80.0 s，对 V4 136.0 s 为 1.70×，对 V5 125.3 s 为 1.57×；Transformer 仍快得多（18.6 s）。

---

## 3. Track 2：Causal Bytecode LM（语言建模 + 解引用）

**协议：** 450 步，batch 64，`lr=3e-3`，seed 42；6 个变量、4 条 NOP、序列长度 80；固定评测批次 seed 3030；训练目标为 `0.25 * dense_loss + deref_loss`；评测报告 dense 交叉熵/PPL 与解引用准确率，解引用只在 `[RET, alias_i, lit_i, SEMI]` 的 alias 位置（`pos + 1`）计算。

| 模型 | 参数量 | 验证 loss | 验证 PPL | 解引用准确率 | 训练耗时 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full（1 层）** | 207,503 | 1.1529 | **3.17** | **100.00%** | 78.5 s |
| Vanilla Transformer（2 层） | 543,744 | 1.2686 | 3.56 | 19.63% | 19.5 s |

**任务格式（语句随机排序）：**

```text
[LET,   alias_i, var_i, SEMI]   # alias_i -> var_i
[ALIAS, var_i,   lit_i, SEMI]   # var_i   -> lit_i
[NOP,   var_i,   SEMI,  SEMI]  × 4
...
[RET,   alias_i, lit_i, SEMI]   # predict lit_i at the alias position
```

V6 修正了 V5 的关键缺陷：V5 在 `[LET, var_i, EQ, lit_i]` 中把答案 `lit_i` 直接放在变量后面，且 `EQ` 参与哈希，导致 alias 与 variable 可能碰撞；V6 的因果字节码把 `LET` 绑定到 `alias -> var`，把 `ALIAS` 绑定到 `var -> lit`，返回时输入只有 `alias_i`，目标 `lit_i` 位于 `pos + 1`。模型必须完成 `alias_i -> var_i -> lit_i` 的两级指针解引用，而不能复制表面 token。6 个等概率字面量，随机基线 **16.67%**。

### Track 2 解读

- TriCore 的 PPL 更低（3.17 对 3.56），解引用准确率为 **100.00%**，Transformer 为 **19.63%**；6 类随机基线是 16.67%。
- 100.00% 说明模型在这个封闭语法上学会了 `alias_i -> var_i -> lit_i` 的两级指针查找；Transformer 仍接近随机。
- 因此正确结论是：**因果字节码管线跑通，目标能力可学习；但这不等于真实代码理解、组合泛化或全面优于 Transformer。**
- 任务本身是确定性的合成字节码，没有 held-out 模板；450 步、6 变量、单 seed 的预算也很小。

---

## 4. V7 原生流式与 Track 3 实测

### 4.1 原生 O(1) 流式推理

| 检查项 | 结果 |
| :--- | :---: |
| `forward` 与 `forward_step` 最大绝对 logit 误差 | **5.960e-07** |
| 最大绝对 surprise 误差 | **1.192e-07** |
| 流式延迟（batch=1，4,096 步） | **2.160 ms/token** |
| 持久状态 / 样本（FP32） | **17,408 bytes = 17.00 KiB** |

状态大小在 `L=128、512、2,048、4,096` 时均为 **17,408 bytes**。`forward_step` 只携带 `(h, F_mat, prev_k)`，不创建 `F_seq`；训练路径仍会物化 `F_seq`，因此 O(1) 只描述持久推理状态，不描述训练激活或吞吐。

### 4.2 Track 3A：3-Hop 链式推理

**协议：** 450 步，batch 64，`lr=3e-3`，seed 42；4 条链、4 个查询、12 个干扰 token；域内 `noise_len=12`，四倍间隔 `noise_len=48`；只在查询 key 位置计算准确率，随机基线 25%。

| 模型 | 参数量 | 域内验证 loss | 域内 3-hop 准确率 | 四倍间隔外推 | 训练耗时 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full（1 层，`max_loops=2`）** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 66.4 s |
| TriCore（`max_loops=1`，两跳上限） | 203,278 | 1.8395 | 25.39% | 23.91% | 62.8 s |
| Vanilla Transformer（2 层） | 297,984 | 2.2188 | 23.83% | 23.44% | 16.7 s |

**解读：** 第二次 refinement 真正完成了第三跳；去掉它后域内准确率从 **100.00%** 跌到 **25.39%**（随机水平），差距 **74.61 个百分点**。2 层 Transformer 在该预算下仍接近随机。这是合成 3-hop 任务，不是通用推理基准。

### 4.3 Track 3B：真实 Python 源码 Held-Out LM

**协议：** 300 步，batch 32，`lr=3e-3`，seed 42；语料为本仓库 `tricore/model.py`、`tricore/tasks.py`、`run_experiment.py` 的 ASCII 字节流（`vocab_size=128`）；前 85% 训练，最后 15% 为严格 held-out 后缀，训练窗口不跨越切分点。

| 模型 | 参数量 | Held-out loss | Held-out PPL | Next-byte 准确率 | 训练耗时 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full（1 层）** | 223,887 | **2.9240** | **18.616** | **41.47%** | 57.7 s |
| Vanilla Transformer（1 层，参数匹配） | 218,448 | 2.9667 | 19.428 | 28.15% | 1.5 s |

**解读：** 在接近匹配的参数量下（223,887 vs 218,448），TriCore 的 held-out loss、PPL 和 next-byte 准确率均更好；但训练耗时约 **38×**，因此这是质量/预算观察，不是吞吐优势。语料只是本仓库的小型后缀，不能据此宣称人类代码理解能力。

### 4.4 推理状态与 KV cache 对比

单样本、FP32、`D=128`、`H=4`、`d_head=32`、Transformer `n_layers=2`。

| 序列长度 | Transformer KV cache | TriCore O(1) 状态 | 缩减倍数 |
| :---: | :---: | :---: | :---: |
| 128 | 256.00 KB | 17.00 KB | 15.1× |
| 512 | 1,024.00 KB | 17.00 KB | 60.2× |
| 2,048 | 4,096.00 KB | 17.00 KB | 240.9× |
| 8,192 | 16,384.00 KB | 17.00 KB | 963.8× |
| 32,768 | 65,536.00 KB | 17.00 KB | **3,855.1×** |

TriCore 单核状态为 `h`（128 floats）+ `F_mat`（4,096 floats）+ `prev_k`（128 floats）= **4,352 floats = 17,408 bytes = 17.00 KiB**，与序列长度无关。Transformer 公式为 `n_layers * 2 * L * D * 4 / 1024` KiB。该表是状态核算，不是实测解码服务基准。

### 4.5 V8 Track 4A：3-Seed 统计稳健性（紧凑 2-Hop MQAR）

**协议：** seeds `42/100/2026`；350 步，batch 64，AdamW，`lr=3e-3`，weight decay `0.01`，cosine decay，clipping `1.0`；`d_model=32`，`d_fast=8`，1 条链、1 个查询、2 个训练噪声 token，序列长度 10；4× OOD 把噪声间隔从 2 拉到 8（序列长度 16）。每个数字是 3 个 seed 的 **mean ± std**，在 5 个固定评测 batch（每批 64 例）上计算。

| 架构 | 参数 | 域内召回 | 4× 间隔召回 | 平均训练耗时 |
| :--- | :---: | :---: | :---: | :---: |
| **TriCore Full（`max_loops=2`）** | 16,559 | **100.00 ± 0.00%** | **100.00 ± 0.00%** | 10.99 s |
| TriCore No Engine II: Fast-Weight | 16,559 | **100.00 ± 0.00%** | 65.73 ± 9.04% | 4.23 s |
| TriCore No Engine III: Adaptive Loop | 16,559 | **100.00 ± 0.00%** | 31.56 ± 10.58% | 7.83 s |
| Vanilla Transformer（2 层） | 23,296 | **100.00 ± 0.00%** | 77.29 ± 9.88% | 2.07 s |

**解读：** 紧凑域内任务对所有模型都饱和，因此域内 100% 不能区分架构；4× 间隔可以。Full 比 No Engine II 高 **+34.27** 个百分点，比 No Engine III 高 **+68.44**，比未调参 Transformer 高 **+22.71**。Full 的 seed 标准差为 0，所有非 Full 变体均非零。3 个 seed 只说明方差，不是显著性检验；该紧凑配置不能替代 V6/V7 的全尺寸单 seed 结论。

### 4.6 V8 Track 4B：真实 CUDA 解码状态与 KV-Cache 实测

batch 1，FP32，8 步 warm-up + 32 步计时；`CUDA Alloc Δ` 是 `empty_cache()` 后 `torch.cuda.memory_allocated()` 的增量。基线显式分配 `(K_cache, V_cache)`（形状 `(n_layers, L, 1, n_heads, d_head)`）并执行真实的 2 层 cached Transformer 解码（对完整因果 cache 做 scaled dot-product attention），不是公式估算。

| 序列长度 | TriCore 状态 bytes | CUDA 分配 Δ | Transformer K+V bytes | CUDA 分配 Δ | 缩减 | TriCore ms/token | Transformer ms/token |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 17,408 | 17,408 | 262,144 | 262,144 | **15.1×** | 2.086 | 1.199 |
| 512 | 17,408 | 17,408 | 1,048,576 | 1,048,576 | **60.2×** | 1.895 | 1.074 |
| 1,024 | 17,408 | 17,408 | 2,097,152 | 2,097,152 | **120.5×** | 1.774 | 1.165 |
| 2,048 | 17,408 | 17,408 | 4,194,304 | 4,194,304 | **240.9×** | 1.780 | 1.319 |

TriCore 状态为 `(D + H*d_head^2 + H*d_head) * 4 = 17,408` B = **17.00 KiB / 样本**（`D=128`，`H=4`，`d_head=32`），与 `L` 无关。基线 cache 为 `2 * n_layers * L * D * 4`，即每 token **2.00 KiB**，`L=2,048` 时 **4.00 MiB**。这是实测显存分配，不是公式；但 Transformer 在每个实测长度上仍然更快（1.07-1.32 ms vs 1.77-2.09 ms），所以这是内存优势，不是延迟优势。32K / 3,855.1× 仍是公式算术，未实测。

**一键演示：** `python demo_showcase.py` 用 `init_state` / `forward_step` 流式跑 `Key_A -> Bridge_1 -> Bridge_2 -> Target_Z`，打印 surprise-gate 轨迹和恒定的 17.00 KiB 仪表盘；无需 checkpoint，CUDA 不可用时自动回退 CPU。

**运行时间：** 参考 CUDA 机器上完整 `python run_v8_verification.py` 为 **94.6 s**（Track 4A 94.1 s，Track 4B 0.52 s，含数据生成与评测），**未达到 60 秒目标**，此处如实记录。

---

## 5. 开源文件就绪清单


| 文件 / 目录 | 状态 | 内容 |
| :--- | :---: | :--- |
| `run_v8_verification.py` | ✅ 已创建 | 3 seeds Track 4A + 真实 CUDA Track 4B；写出 `V8_METRICS.json` 与报告附录 |
| `demo_showcase.py` | ✅ 已创建 | 一键 3-hop 流式演示，无需 checkpoint，CPU/CUDA 均可运行 |
| `LICENSE` | ✅ 已创建 | MIT License |
| `.gitignore` | ✅ 已创建 | Python 缓存、虚拟环境、编辑器文件；保留 `results/` 报告与 PNG |
| `pyproject.toml` | ✅ 已更新 | `tricore-arch` 1.1.0，Python ≥3.10，torch/numpy/matplotlib，MIT |
| `README.md` | ✅ 已更新 | V8 标题、3-seed Track 4A、真实 CUDA Track 4B、V6/V7 数据保留、限制、复现命令 |
| `SPEC_V2.md` | ✅ 已更新 | V8 规范、3-seed Track 4A、真实 CUDA Track 4B、V6 JIT 四元组合约、状态核算、任务布局、协议 |
| `requirements.txt` | ✅ 已存在 | `torch>=2.0`、`numpy>=1.24`、`matplotlib>=3.7` |
| `tricore/__init__.py` | ✅ 已更新 | 导出 block/LM/baseline 与全部任务加载器 |
| `tricore/model.py` | ✅ 已更新 | `tricore_fused_scan`、双模型、`init_state` / `forward_step` 流式接口 |
| `tricore/tasks.py` | ✅ 已更新 | 2-Hop MQAR + 因果字节码 + 3-Hop 召回 + 真实 Python 源码 held-out 字节流 |
| `tests/test_sanity.py` | ✅ 已存在 | 前向/反向/形状/梯度流 |
| `tests/test_v5.py` | ✅ 已创建 | JIT 四元组对 Python 参考的四组 scan flag 等价性 + 因果字节码形状/反向 |
| `tests/test_v7_streaming.py` | ✅ 已创建 | 流式等价、17 KiB 恒定状态、3-Hop/真实语料梯度流 |
| `run_experiment.py` | ✅ 已更新 | V6 双赛道、Track 1 五组消融、grad clipping、报告与 2×2 图 |
| `run_v7_benchmark.py` | ✅ 已创建 | 仅运行 V7 流式检查与 Track 3，不重跑 V6 |
| `results/BENCHMARK_REPORT.md` | ✅ 已生成 | V6 表格保留 + V7 流式/Track 3 实测 |
| `results/tricore_vs_transformer.png` | ✅ 已生成 | 2×2 对比图 |
| `REPORT_FOR_USER.md` | ✅ 本文件 | 完整交付说明 |

**Git-ready 说明：** 仓库根目录是 Python 项目，无密钥、无本地绝对路径、无大数据集；`results/BENCHMARK_REPORT.md` 与 `results/tricore_vs_transformer.png` 作为可复现证据保留。提交前建议执行：

```bash
git status --short
python -m pytest -q
python run_experiment.py --steps 2 --code-steps 2   # 可选快速冒烟
```

---

## 6. 复现命令

**V8 新增（v1.1.0）：**

```bash
python run_v8_verification.py   # Track 4A 3-seed + Track 4B CUDA profiling
python demo_showcase.py         # 一键 3-hop 流式演示
```

```bash
# 安装
python -m pip install -e .

# 全量 V6 双赛道实验（生成报告与图）
python run_experiment.py

# V7 流式检查 + Track 3A/3B（保留 V6 表格）
python run_v7_benchmark.py

# 只做快速冒烟
python run_experiment.py --steps 2 --code-steps 2

# 测试
python -m pytest -q
python tests/test_v5.py
python tests/test_v7_streaming.py
```

**关键文件：**

- `tricore/model.py`：`@torch.jit.script tricore_fused_scan`（四元组返回）、向量化 5-D 多跳 `TriCoreBlock`、`TriCoreLM`、`BaselineTransformerLM`、`init_state` / `forward_step`
- `tricore/tasks.py`：2-Hop MQAR、因果字节码 `generate_contextual_code_lm_batch`、3-Hop `generate_3hop_recall_batch`、真实源码 `load_real_python_corpus_batches`
- `run_experiment.py`：Track 1 五组消融、Track 2 两模型、V6 计时、cache 核算、Markdown 报告、matplotlib 2×2 图
- `run_v7_benchmark.py`：流式等价/延迟/状态核算、Track 3A 三模型、Track 3B 两模型，并追加报告
- `tests/test_v5.py`：JIT 与 Python 参考在四种 `enable_surprise_gate` × `enable_fast_weight` 组合下对 `h_seq`、`r1_seq`、`F_seq`、`surprise_trace` 的 `allclose` 等价性；`TriCoreLM` 前向/反向；因果字节码的形状、`pos + 1` mask、反向梯度检查
- `tests/test_v7_streaming.py`：流式等价、17,408 bytes 恒定状态、3-Hop/真实语料形状与梯度检查；完整套件 6 passed

---

## 7. 结论：V8 到底证明了什么

V8 新增的证据是：在紧凑 2-Hop MQAR 上，3 个 seed 的 4× 间隔召回为 Full **100.00 ± 0.00%**、No Engine II **65.73 ± 9.04%**、No Engine III **31.56 ± 10.58%**、Transformer **77.29 ± 9.88%**；域内四者均为 100%，因此区分力来自 4× 间隔。真实 CUDA 分配下，TriCore 状态恒为 **17.00 KiB/样本**，`L=2,048` 的 Transformer KV cache 为 **4.00 MiB**，缩减 **240.9×**。但这仍是 3 个 seed 的方差检查，不是显著性检验；完整 V6/V7 表格仍为单 seed，V8 脚本耗时 **94.6 s**，未达到 60 秒目标。

### 已证明（在所述任务与预算内）

1. **Full TriCore 能完成 2-Hop 链式联想召回**，并在 4× 对抗噪声间隔外推下保持 100.00%。
2. **Engine II 和 Engine III 是该任务的关键**：去掉快权重或去掉隐空间重查询都会跌到随机水平附近。
3. **Engine I 在本次 V6 运行中是必要的**：去掉后跌到 5.55%/4.14%，与 V5 的 91.88%/84.14% 不同；这提醒我们单 seed 消融结论的脆弱性。
4. **V6 把多跳重查询从逐 token 循环改成跨全部位置的一次批量 5-D `matmul`**，保持公式等价；状态递归仍是串行，不能把它描述成并行前缀扫描或 kernel 级优化。
5. **O(1) 状态核算是真实的**：17.00 KiB 与序列长度无关，32K 时相对 2 层 KV cache 为 3,855.1×。
6. **第二赛道管线可运行**：TriCore 在因果字节码任务上 PPL 3.17、解引用 100.00%；2 层 Transformer 为 3.56 / 19.63%，接近 16.67% 随机水平。这证明封闭语法上的两级指针查找可学习，不证明真实代码理解。
7. **V7 流式路径与完整前向等价**：最大 logit 误差 5.96e-07，持久状态恒为 17,408 bytes，4,096 步实测 2.160 ms/token；训练期仍使用 `F_seq`。
8. **Track 3A 证明第二次 refinement 是必要的**：Full `max_loops=2` 为 100.00%/100.00%，`max_loops=1` 只有 25.39%/23.91%，差 74.61 个百分点；2 层 Transformer 为 23.83%/23.44%。
9. **Track 3B 在真实源码的后缀上完成了 held-out 比较**：TriCore 223,887 参数达到 PPL 18.616 / 41.47%，参数匹配 Transformer 218,448 参数为 19.428 / 28.15%；TriCore 训练慢约 38×，且语料仅为本仓库源码。

### 未证明 / 不应宣称

- 自然语言、通用推理或大规模真实代码能力；Track 3B 只是本仓库的小型源码后缀。
- 对调参后 Transformer 的全面优势；Track 1/3A 的 Transformer 未调参，Track 3B 只匹配参数量、未匹配算力。
- 多 seed 稳健性、统计显著性、置信区间。
- 生产吞吐、长上下文实测解码、分布式/混合精度/量化能力。
- 训练显存的 O(1) 性质：V7 仍会物化 `F_seq`，训练激活随 `L` 增长；只有持久推理状态是 O(1)。
- “Engine I 在所有任务上都必要”或“三个引擎缺一不可”的普适结论；本次 No-I 跌到随机，但 V5 没有，说明结论对优化轨迹和 seed 敏感。
- JIT 已接近硬件最优；它仍是编译后的 PyTorch 循环，状态递归仍串行。

---

## 8. 建议的下一步

1. **多 seed**：至少 3 个 seed，报告 mean ± std，尤其是 No-Engine-I 的方差和 Track 3A/3B 的稳定性。
2. **更强基线**：调参 Transformer，匹配非位置参数量、训练 token 与算力；当前 Track 3B 只匹配参数量。
3. **更大真实代码语料**：用 held-out 文件/仓库替代本仓库 15% 后缀，才能检验真实代码泛化。
4. **4–6 hop**：验证 latent re-query 是否随跳数扩展，以及 router 是否学会使用额外循环。
5. **真正 kernel**：在 JIT + 向量化多跳 + 原生流式的基础上做 fused/chunked CUDA/Triton scan；减少小 kernel launch 与串行状态更新成本。
6. **长上下文实测**：用真实 KV cache 解码对比 32K–128K 的延迟与显存，而不只做公式核算；V7 的 2.160 ms/token 只是单机 B=1 循环测量。

---

## 9. 交付状态

| 项目 | 状态 |
| :--- | :---: |
| V6 模型代码（JIT 四元组 + 向量化多跳 + LM + baseline） | ✅ 完成并保留 |
| Track 1 五组消融 | ✅ 完成并写入报告 |
| Track 2 因果字节码 Contextual Code LM | ✅ 完成并写入报告 |
| V6 / V5 / V4 耗时对比 | ✅ 80.0 s vs 125.3 s vs 136.0 s（1.57× / 1.70×） |
| V7 `init_state` / `forward_step` 流式接口 | ✅ 完成；最大 logit 误差 5.96e-07，2.160 ms/token |
| V7 推理状态核算 | ✅ 17,408 bytes = 17.00 KiB；L=128..4,096 恒定 |
| Track 3A：3-Hop 链式推理 | ✅ 完成；Full 100.00%/100.00%，max_loops=1 25.39%/23.91% |
| Track 3B：真实 Python 源码 held-out LM | ✅ 完成；TriCore PPL 18.616 / 41.47%，Transformer 19.428 / 28.15% |
| README / SPEC 同步 | ✅ 已更新为 V8 数据（V6/V7 保留） |
| LICENSE / .gitignore / pyproject | ✅ 已创建；版本 1.1.0 |
| 测试文件 | ✅ `tests/test_v5.py`、`tests/test_v7_streaming.py`；`python -m pytest -q` → **6 passed** |
| 结果报告与图 | ✅ 已生成；V6/V7 表格保留 + V8 附录追加 |
| 本汇报文件 | ✅ 已生成 |

> 一句话总结：**V7 在 V6 的恒定状态与向量化多跳之上补上了原生 `init_state` / `forward_step` 流式解码（17,408 bytes 恒定状态、5.96e-07 等价误差、2.160 ms/token），并用 Track 3A 证明第二次 refinement 是第三跳所必需（Full 100.00% vs 两跳上限 25.39%），用 Track 3B 在真实仓库源码的严格 15% held-out 后缀上取得 PPL 18.616 / 41.47%（参数匹配 Transformer 19.428 / 28.15%）。但全部结果仍是单 seed，Track 3A 是合成任务，Track 3B 语料很小且 TriCore 训练约慢 38×，不能据此宣称通用推理、真实代码理解或吞吐优势。**

---

<p align="center">
  <i>TriCore V7 — constant state, surprise-gated writes, causal delta memory, a vectorized latent loop, and native one-token streaming.</i>
</p>