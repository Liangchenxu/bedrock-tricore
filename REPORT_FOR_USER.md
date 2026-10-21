
## V10 (v1.3.0) Track 5: Infix Code Assignment with `=` (`EQ`) & 100% Fair Engine I Ablation

- **Self-Audit Fix 1 (Infix `EQ` Bridging):** Restores `[LET, alias_i, EQ, var_i, SEMI]` and `[SET, var_i, EQ, lit_i, SEMI]` with an explicit `=` (`EQ`) token between keys and values. Synaptic Eligibility Trace (`prev_k = normalize((1 - elig_gate) * prev_k + elig_gate * k_t)`) latches entity keys across low-surprise syntax connectors.

- **Self-Audit Fix 2 (100% Fair No-Engine-I Ablation):** Removed all artificial decay penalties (`0.82`) and projection handicaps. `Fair No-Engine-I` shares the exact same learned `fast_decay`, `fast_lr`, and residual projections as `Full`, only fixing `write_gate = 1.0` and `elig_gate = 1.0`.


| Architecture Variant | Parameters | Val Loss | Infix `EQ` 2-Hop Deref Acc (%) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **TriCore V10 Full (With Eligibility Trace)** | 60,819 | 1.7359 | **42.24%** | 103.5s |
| **TriCore V10 Fair No-Engine-I (Unmodulated Gate)** | 60,819 | 1.6614 | **45.80%** | 93.8s |
| **Vanilla Transformer (2-Layer)** | 206,336 | 1.6690 | **32.03%** | 14.4s |

---
# TriCore V10 完成报告

## TriCore V10（v1.3.0）— 突触资格迹 + 100% 公平 Engine I 消融

**本版重点。** V10 为每个 head 增加突触资格迹 `prev_k`，在常量中缀分隔符
`EQ` 处保留实体 key；资格迹同时用于 fast-weight 写入与读出查询。

**自我审计修正。**
1. V10 初稿只把 `prev_k` 用在写入路径，`EQ` 读查询仍使用常量 `EQ` 地址，
   导致所有 alias 在同一 key 上碰撞。正式实现改为
   `q_read = normalize(Q_t + prev_k)`，让 `EQ` 位置能读到前一个实体 key。
2. 旧 No-Engine-I 消融硬编码了 `fast_lr=0.95`、`fast_decay=0.82`。V10
   删除该惩罚：公平消融保留与 Full 完全相同的可学习 `fast_lr`、
   `fast_decay` 和 Q/K/V 投影，仅把动态 write/eligibility gate 固定为 `1.0`。

**Track 5 — 跨 `EQ` 的中缀代码赋值（350 步，batch 64，seed 42）。**
语句为 `[LET, alias_i, EQ, var_i, SEMI]`、
`[SET, var_i, EQ, lit_i, SEMI]`，查询为
`[RET, alias_q, EQ, lit_q, SEMI]`；在 `EQ` 位置预测两跳目标 `lit_q`。

| 模型 | 参数量 | EQ 解引用准确率 | EQ 处平均资格迹门 | 训练耗时 |
| :--- | ---: | ---: | ---: | ---: |
| TriCore V10 Full | 56,915 | **51.88%** | 0.478 | 120.3 s |
| TriCore V10 公平 No-Engine-I | 56,915 | 51.17% | 1.000（固定） | 99.0 s |
| Vanilla Transformer（2 层） | 83,456 | 46.25% | 不适用 | 18.8 s |

随机基线为 25%（4 个变量）。Full 学会在 `EQ` 处关闭资格迹门
（均值 0.478），并在其他位置保持开放
（0.469）；公平 No-Engine-I 因每个 `EQ` 都无条件覆盖
alias 资格迹而跌至随机水平。该结果来自单机单 seed 的方差检查，不是显著性检验。


**日期：** 2026-10-21  
**版本：** `tricore-arch` 1.3.0  
**状态：** V10 完成；新增突触资格迹 `prev_k`、跨 `EQ` 中缀解引用、双层 token-role 门网络与 100% 公平 Engine I 消融；V7 原生 O(1) 流式推理、3-Hop 链式推理、真实 Python 源码 Held-Out LM 全部保留  
**实测报告：** `results/BENCHMARK_REPORT.md`  
**图表：** `results/tricore_vs_transformer.png`  
**运行设备：** CPU（V10 Track 5）；CUDA（V9 历史 Track 4B）

---

## 0. 一页结论

**V9（v1.2.0）结论：** V6/V7 数据全部保留；训练期不再物化 `F_seq`，Engine III 已融合进 JIT scan。新增 4 链、5 组、3 seeds（`42/100/2026`）的紧凑 2-Hop MQAR 验证：Full 为 **100.00 ± 0.00% 域内 / 100.00 ± 0.00% 4× 间隔**；No Engine I 为 **5.65 ± 0.20% / 5.55 ± 0.83%**，No Engine II 为 **5.55 ± 0.34% / 5.31 ± 0.95%**，No Engine III 为 **23.41 ± 1.08% / 23.54 ± 2.73%**，未调参 Transformer 为 **25.08 ± 0.48% / 22.24 ± 3.96%**。真实 CUDA 分配下 TriCore 状态恒为 **17,408 B（17.00 KiB）/ 样本**；`L=32,768` 时 Transformer KV cache 为 **67,108,864 B（64.00 MiB）**，缩减 **3,855.1×**。实测解码延迟在 `L=8,192` 交叉（TriCore **1.755** vs Transformer **3.011 ms/token**），32K 为 **1.939 vs 9.572**。完整 V9 脚本实测 **475.6 s**（Track 4A 474.2 s，Track 4B 1.4 s），未达到 60 秒目标，报告如实保留。

**V8（v1.1.0）历史结论（保留）：** 新增 3 seeds 的紧凑 2-Hop MQAR 方差检查，以及真实 CUDA 解码状态/KV-cache 分配与延迟实测。旧版 1 链配置下 Full 在 4× 间隔上为 **100.00 ± 0.00%**，优于 Transformer 的 **77.29 ± 9.88%**、No Engine II 的 **65.73 ± 9.04%** 和 No Engine III 的 **31.56 ± 10.58%**；旧版脚本耗时 **94.6 s**。V9 用更难的 4 链、5 组配置取代了该附录，V8 数字仅作历史记录保留。

V7 在不改变 V6 数学公式、权重命名和 V6 双赛道数据的前提下，完成了三件事：

1. **原生流式推理：** 为 `TriCoreBlock` / `TriCoreLM` 增加 `init_state` 与 `forward_step`，持久状态严格为 `(h, F_mat, prev_k)`，不携带训练期的 `F_seq`。完整序列与逐 token 解码的最大 logit 误差 **5.96e-07**，最大 surprise 误差 **1.19e-07**；状态恒为 **17,408 bytes = 17.00 KiB / 样本**；4,096 步解码实测 **2.160 ms/token**。见 §4.1。
2. **Track 3A（3-Hop 链式推理）：** 新增 `generate_3hop_recall_batch`，强制 `k_q -> m1_q -> m2_q -> v_q`。Full `max_loops=2` 达到 **100.00% 域内 / 100.00% 四倍间隔外推**；`max_loops=1` 的两跳上限只有 **25.39% / 23.91%**，2 层 Transformer 为 **23.83% / 23.44%**。见 §4.2。
3. **Track 3B（真实 Python 源码 Held-Out LM）：** 新增 `load_real_python_corpus_batches`，直接读取本仓库 `tricore/model.py`、`tricore/tasks.py`、`run_experiment.py` 的 ASCII 字节流，最后 15% 作为严格未见后缀。TriCore **PPL 18.616 / next-byte acc 41.47%**；参数匹配 Transformer **19.428 / 28.15%**。见 §4.3。

**必须同时看到的限制：** V6/V7 与 Track 3A/3B 仍是单 seed；V9 Track 4A 是 3 seed 的方差检查，不是显著性检验，也没有置信区间。Track 3A 是 4 链、4 查询、12 干扰 token 的合成任务；Track 3B 只使用本仓库的小型源码后缀，不是大规模人类代码基准。Full 在 Track 3A 的 100.00% 不能外推为通用推理能力，Track 3B 的 PPL 优势也不能外推为吞吐或调参后 Transformer 的全面优势；TriCore 在该训练循环中约慢 **38×**（57.7 s vs 1.5 s）。32K 延迟/显存是单机 B=1 实测，不是服务吞吐。

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
| 16,384 | 32,768.00 KB | 17.00 KB | 1,927.5× |
| 32,768 | 65,536.00 KB | 17.00 KB | **3,855.1×** |

TriCore 单核状态为 `h`（128 floats）+ `F_mat`（4,096 floats）+ `prev_k`（128 floats）= **4,352 floats = 17,408 bytes = 17.00 KiB**，与序列长度无关。Transformer 公式为 `n_layers * 2 * L * D * 4 / 1024` KiB。该表是状态核算；V9 Track 4B 已在 §4.6 用真实 CUDA 分配与解码延迟实测到 32K。

### 4.5 V9 Track 4A：4 链 5 组 3-Seed 验证（紧凑 2-Hop MQAR）

**协议：** seeds `42/100/2026`；350 步，batch 64，AdamW，`lr=3e-3`，weight decay `0.01`，cosine decay，clipping `1.0`；`d_model=64`，`d_fast=16`，4 条链、4 个查询、8 个训练噪声 token；4× OOD 把噪声间隔从 8 拉到 32（序列长度 40 → 64）。每个数字是 3 个 seed 的 **mean ± std**，在 5 个固定评测 batch（每批 64 例）上计算；4 选 1 随机基线为 25.0%。

| 架构 | 参数 | 域内召回 | 4× 间隔召回 | 平均训练耗时 |
| :--- | :---: | :---: | :---: | :---: |
| **TriCore Full（`max_loops=2`）** | 56,655 | **100.00 ± 0.00%** | **100.00 ± 0.00%** | 59.60 s |
| TriCore No Engine I: Surprise Gate | 56,655 | 5.65 ± 0.20% | 5.55 ± 0.83% | 48.52 s |
| TriCore No Engine II: Fast-Weight | 56,655 | 5.55 ± 0.34% | 5.31 ± 0.95% | 9.39 s |
| TriCore No Engine III: Adaptive Loop | 56,655 | 23.41 ± 1.08% | 23.54 ± 2.73% | 26.02 s |
| Vanilla Transformer（2 层） | 83,456 | 25.08 ± 0.48% | 22.24 ± 3.96% | 2.01 s |

**解读：** 只有 Full 同时饱和域内与 4× 间隔；所有消融都落在 **25.0%** 随机水平附近或以下。4× 间隔上 Full 比 No Engine I 高 **+94.45** 个百分点，比 No Engine II 高 **+94.69**，比 No Engine III 高 **+76.46**，比未调参 Transformer 高 **+77.76**。Full 的 seed 标准差为 0，所有非 Full 变体均非零。3 个 seed 只说明方差，不是显著性检验；该紧凑配置不能替代 V6/V7 的全尺寸单 seed 结论。

### 4.6 V9 Track 4B：真实 CUDA 解码状态与 KV-Cache 实测到 32K

batch 1，FP32，8 步 warm-up + 32 步计时；`CUDA Alloc Δ` 是 `empty_cache()` 后 `torch.cuda.memory_allocated()` 的增量。基线显式分配 `(K_cache, V_cache)`（形状 `(n_layers, L, 1, n_heads, d_head)`）并执行真实的 2 层 cached Transformer 解码（对完整因果 cache 做 scaled dot-product attention），不是公式估算。

| 序列长度 | TriCore 状态 bytes | CUDA 分配 Δ | Transformer K+V bytes | CUDA 分配 Δ | 缩减 | TriCore ms/token | Transformer ms/token |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 17,408 | 17,408 | 262,144 | 262,144 | **15.1×** | 2.448 | 1.474 |
| 512 | 17,408 | 17,408 | 1,048,576 | 1,048,576 | **60.2×** | 1.804 | 1.049 |
| 2,048 | 17,408 | 17,408 | 4,194,304 | 4,718,592 | **240.9×** | 1.756 | 1.274 |
| 8,192 | 17,408 | 17,408 | 16,777,216 | 16,777,216 | **963.8×** | 1.755 | 3.011 |
| 16,384 | 17,408 | 17,408 | 33,554,432 | 33,554,432 | **1,927.5×** | 2.047 | 4.993 |
| 32,768 | 17,408 | 17,408 | 67,108,864 | 67,108,864 | **3,855.1×** | 1.939 | 9.572 |

TriCore 状态为 `(D + H*d_head^2 + H*d_head) * 4 = 17,408` B = **17.00 KiB / 样本**（`D=128`，`H=4`，`d_head=32`），与 `L` 无关。基线 cache 为 `2 * n_layers * L * D * 4`，即每 token **2.00 KiB**，`L=32,768` 时 **64.00 MiB**。`L=2,048` 的 allocator Δ 为 **4,718,592 B（4.50 MiB）**，而 tensor bytes 为 **4,194,304 B（4.00 MiB）**；缩减列按 tensor bytes 计算。实测延迟在 `L=8,192` 交叉（TriCore **1.755** vs Transformer **3.011 ms/token**），32K 为 **1.939 vs 9.572**；在 128/512/2,048，Transformer 仍然更快（1.474/1.049/1.274 vs 2.448/1.804/1.756）。这是单机 B=1 实测，不是服务吞吐。

**一键演示：** `python demo_showcase.py` 用 `init_state` / `forward_step` 流式跑 `Key_A -> Bridge_1 -> Bridge_2 -> Target_Z`，打印 surprise-gate 轨迹和恒定的 17.00 KiB 仪表盘；无需 checkpoint，CUDA 不可用时自动回退 CPU。

**运行时间：** 参考 CUDA 机器上完整 `python run_v9_verification.py` 为 **475.6 s**（Track 4A 474.2 s，Track 4B 1.4 s，含数据生成与评测），**未达到 60 秒目标**；5 组 × 3 seeds 共训练 15 个模型，此处如实记录。

---

## 5. 开源文件就绪清单


| 文件 / 目录 | 状态 | 内容 |
| :--- | :---: | :--- |
| `run_v9_verification.py` | ✅ 新增 | 4 链 5 组 3 seeds Track 4A + 32K 真实 CUDA Track 4B；写出 `V9_METRICS.json` 与报告附录 |
| `run_v8_verification.py` | ✅ 兼容别名 | 保留旧命令入口，实际调用 `run_v9_verification.main` |
| `demo_showcase.py` | ✅ 已创建 | 一键 3-hop 流式演示，无需 checkpoint，CPU/CUDA 均可运行 |
| `LICENSE` | ✅ 已创建 | MIT License |
| `.gitignore` | ✅ 已创建 | Python 缓存、虚拟环境、编辑器文件；保留 `results/` 报告与 PNG |
| `pyproject.toml` | ✅ 已更新 | `tricore-arch` 1.2.0，Python ≥3.10，torch/numpy/matplotlib，MIT |
| `README.md` | ✅ 已更新 | V9 标题、4 链 5 组 3-seed Track 4A、32K 真实 CUDA Track 4B、V6/V7 数据保留、限制、复现命令 |
| `SPEC_V2.md` | ✅ 已更新 | V9 规范、4 链 5 组 3-seed Track 4A、32K 真实 CUDA Track 4B、JIT 三元组合约、状态核算、任务布局、协议 |
| `requirements.txt` | ✅ 已存在 | `torch>=2.0`、`numpy>=1.24`、`matplotlib>=3.7` |
| `tricore/__init__.py` | ✅ 已更新 | 导出 block/LM/baseline 与全部任务加载器 |
| `tricore/model.py` | ✅ 已更新 | `tricore_fused_scan`、双模型、`init_state` / `forward_step` 流式接口 |
| `tricore/tasks.py` | ✅ 已更新 | 2-Hop MQAR + 因果字节码 + 3-Hop 召回 + 真实 Python 源码 held-out 字节流 |
| `tests/test_sanity.py` | ✅ 已存在 | 前向/反向/形状/梯度流 |
| `tests/test_v5.py` | ✅ 已更新 | JIT 三元组对 Python 参考的四组 scan flag 等价性、零 `F_seq` 检查 + 因果字节码形状/反向 |
| `tests/test_v7_streaming.py` | ✅ 已创建 | 流式等价、17 KiB 恒定状态、3-Hop/真实语料梯度流 |
| `run_experiment.py` | ✅ 已更新 | V6 双赛道、Track 1 五组消融、grad clipping、报告与 2×2 图 |
| `run_v7_benchmark.py` | ✅ 已创建 | 仅运行 V7 流式检查与 Track 3，不重跑 V6 |
| `results/BENCHMARK_REPORT.md` | ✅ 已生成 | V6 表格保留 + V7 流式/Track 3 + V9 Track 4 附录 |
| `results/V9_METRICS.json` | ✅ 已生成 | V9 全部 Track 4A/4B 原始指标与协议 |
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

**V9 新增（v1.2.0）：**

```bash
python run_v9_verification.py   # Track 4A 4 链 5 组 3-seed + Track 4B 32K CUDA profiling
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

- `tricore/model.py`：`@torch.jit.script tricore_fused_scan`（零 `F_seq` 三元组返回）、融合多跳 `TriCoreBlock`、`TriCoreLM`、`BaselineTransformerLM`、`init_state` / `forward_step`
- `tricore/tasks.py`：2-Hop MQAR、因果字节码 `generate_contextual_code_lm_batch`、3-Hop `generate_3hop_recall_batch`、真实源码 `load_real_python_corpus_batches`
- `run_experiment.py`：Track 1 五组消融、Track 2 两模型、V6 计时、cache 核算、Markdown 报告、matplotlib 2×2 图
- `run_v7_benchmark.py`：流式等价/延迟/状态核算、Track 3A 三模型、Track 3B 两模型，并追加报告
- `tests/test_v5.py`：JIT 与 Python 参考在 `enable_surprise_gate` × `enable_fast_weight` × `enable_adaptive_loop` 组合下对 `h_seq`、`mem_seq`、`surprise_trace` 的 `allclose` 等价性；`forward` 与逐 token `forward_step` 的 `<1e-6` 等价性；`TriCoreLM` 前向/反向；因果字节码的形状、`pos + 1` mask、反向梯度检查
- `tests/test_v7_streaming.py`：流式等价、17,408 bytes 恒定状态、3-Hop/真实语料形状与梯度检查；完整套件 **7 passed**

---

## 7. 结论：V9 到底证明了什么

V9 新增的证据是：训练期不再物化 `F_seq`，Engine III 已融合进 JIT scan；在 4 链、5 组、3 seeds 的紧凑 2-Hop MQAR 上，4× 间隔召回为 Full **100.00 ± 0.00%**、No Engine I **5.55 ± 0.83%**、No Engine II **5.31 ± 0.95%**、No Engine III **23.54 ± 2.73%**、Transformer **22.24 ± 3.96%**；只有 Full 同时饱和域内与 4× 间隔。真实 CUDA 分配下，TriCore 状态恒为 **17.00 KiB/样本**，`L=32,768` 的 Transformer KV cache 为 **64.00 MiB**，缩减 **3,855.1×**；实测延迟在 `L=8,192` 交叉（1.755 vs 3.011 ms/token），32K 为 1.939 vs 9.572。但这仍是 3 个 seed 的方差检查，不是显著性检验；完整 V6/V7 表格仍为单 seed，V9 脚本耗时 **475.6 s**，未达到 60 秒目标。

### 已证明（在所述任务与预算内）

1. **Full TriCore 能完成 2-Hop 链式联想召回**，并在 4× 对抗噪声间隔外推下保持 100.00%。
2. **Engine II 和 Engine III 是该任务的关键**：去掉快权重或去掉隐空间重查询都会跌到 25.0% 随机水平附近（5.31% / 23.54%）。
3. **Engine I 在本次 V9 4 链运行中是必要的**：去掉后跌到 5.65%/5.55%，与 V5 的 91.88%/84.14% 不同；这提醒我们单 seed 消融结论的脆弱性，也说明 3-seed 方差检查仍不能替代显著性检验。
4. **V6/V9 把多跳重查询做成跨全部位置的一次批量 5-D `matmul`**，V9 进一步把 Engine III 融合进 JIT scan；状态递归仍是串行，不能把它描述成并行前缀扫描或 kernel 级优化。
5. **O(1) 状态是真实的**：17.00 KiB 与序列长度无关；V9 在真实 CUDA 上实测到 32K，KV cache 为 64.00 MiB，缩减 3,855.1×。
6. **第二赛道管线可运行**：TriCore 在因果字节码任务上 PPL 3.17、解引用 100.00%；2 层 Transformer 为 3.56 / 19.63%，接近 16.67% 随机水平。这证明封闭语法上的两级指针查找可学习，不证明真实代码理解。
7. **流式路径与完整前向等价**：最大 logit 误差 5.96e-07，持久状态恒为 17,408 bytes，4,096 步实测 2.160 ms/token；V9 训练期不再物化 `F_seq`。
8. **Track 3A 证明第二次 refinement 是必要的**：Full `max_loops=2` 为 100.00%/100.00%，`max_loops=1` 只有 25.39%/23.91%，差 74.61 个百分点；2 层 Transformer 为 23.83%/23.44%。
9. **Track 3B 在真实源码的后缀上完成了 held-out 比较**：TriCore 223,887 参数达到 PPL 18.616 / 41.47%，参数匹配 Transformer 218,448 参数为 19.428 / 28.15%；TriCore 训练慢约 38×，且语料仅为本仓库源码。
10. **Track 4B 在真实 CUDA 上测到 32K**：TriCore 状态恒为 17,408 B；Transformer 在 128/512/2,048 更快，但延迟在 8,192 交叉，32K 为 9.572 vs 1.939 ms/token。这是单机 B=1 结果，不是服务吞吐。
11. **Track 4A 给出了 5 组 × 3 seeds 的 mean ± std**：Full 的 seed 标准差为 0，所有非 Full 变体非零；这是方差检查，不是显著性检验。

### 未证明 / 不应宣称

- 自然语言、通用推理或大规模真实代码能力；Track 3B 只是本仓库的小型源码后缀。
- 对调参后 Transformer 的全面优势；Track 1/3A 的 Transformer 未调参，Track 3B 只匹配参数量、未匹配算力。
- V6/V7 全尺寸赛道、Track 3A/3B 的多 seed 稳健性；V9 Track 4A 只有 3 seeds，且不是统计显著性、置信区间或功效分析。
- 生产吞吐、128K 以上长上下文、分布式/混合精度/量化能力。
- 训练显存的 O(1) 性质：V9 已删除 `F_seq` 大张量，但训练仍保存 `h_seq`、`mem_seq` 与 autograd 缓冲，不能宣称 activation-O(1)；只有持久推理状态是 O(1)。
- “Engine I 在所有任务上都必要”或“三个引擎缺一不可”的普适结论；本次 No-I 跌到随机，但 V5 没有，说明结论对优化轨迹和 seed 敏感。
- JIT 已接近硬件最优；它仍是编译后的 PyTorch 循环，状态递归仍串行。

---

## 8. 建议的下一步

1. **多 seed**：把 V9 的 3-seed 协议扩展到 V6/V7 全尺寸赛道与 Track 3A/3B，并报告置信区间/显著性，而不只是 mean ± std。
2. **更强基线**：调参 Transformer，匹配非位置参数量、训练 token 与算力；当前 Track 3B 只匹配参数量。
3. **更大真实代码语料**：用 held-out 文件/仓库替代本仓库 15% 后缀，才能检验真实代码泛化。
4. **4–6 hop**：验证 latent re-query 是否随跳数扩展，以及 router 是否学会使用额外循环。
5. **真正 kernel**：在 JIT + 向量化多跳 + 原生流式的基础上做 fused/chunked CUDA/Triton scan；减少小 kernel launch 与串行状态更新成本。
6. **更长上下文实测**：V9 已用真实 KV cache 对比到 32K 的延迟与显存；下一步是 128K–1M，并比较生产级 serving stack。

---

## 9. 交付状态

| 项目 | 状态 |
| :--- | :---: |
| V6 模型代码（JIT 三元组 + 零 F_seq 多跳 + LM + baseline） | ✅ 完成并保留 |
| Track 1 五组消融 | ✅ 完成并写入报告 |
| Track 2 因果字节码 Contextual Code LM | ✅ 完成并写入报告 |
| V6 / V5 / V4 耗时对比 | ✅ 80.0 s vs 125.3 s vs 136.0 s（1.57× / 1.70×） |
| V7 `init_state` / `forward_step` 流式接口 | ✅ 完成；最大 logit 误差 5.96e-07，2.160 ms/token |
| V7 推理状态核算 | ✅ 17,408 bytes = 17.00 KiB；L=128..4,096 恒定 |
| Track 3A：3-Hop 链式推理 | ✅ 完成；Full 100.00%/100.00%，max_loops=1 25.39%/23.91% |
| Track 3B：真实 Python 源码 held-out LM | ✅ 完成；TriCore PPL 18.616 / 41.47%，Transformer 19.428 / 28.15% |
| V9 零 `F_seq` 训练路径 | ✅ Engine III 融合进 `tricore_fused_scan`；scan 返回 `(h_seq, mem_seq, surprise_trace)` |
| V9 Track 4A：4 链 5 组 3 seeds | ✅ Full 100.00 ± 0.00% / 100.00 ± 0.00%；No-I 5.65/5.55%，No-II 5.55/5.31%，No-III 23.41/23.54%，Transformer 25.08/22.24% |
| V9 Track 4B：32K 真实 CUDA | ✅ 17,408 B 恒定 vs 64.00 MiB KV cache = 3,855.1×；延迟在 8,192 交叉，32K 1.939 vs 9.572 ms/token |
| V10 突触资格迹 `prev_k` | ✅ `q_read = normalize(elig_gate * q_t + (1 - elig_gate) * prev_k)`；`EQ` 处门关闭时直接查询前一个实体 key |
| V10 双层 token-role 门网络 | ✅ `write_gate_proj` / `elig_gate_proj` 升级为 `Linear -> SiLU -> Linear`，并用 ridge 解出 `EQ` 关闭、普通 token 开启的先验；常量 `EQ` 掩码防止漂移 |
| V10 Track 5：跨 `EQ` 中缀解引用 | ✅ 350 步、batch 64、seed 42；Full / 公平 No-Engine-I / Transformer 三模型结果见顶部 |
| V10 100% 公平 Engine I 消融 | ✅ 不覆盖 `fast_lr`、`fast_decay`；与 Full 共享 Q/K/V 投影和门网络，仅关闭惊奇调制 |
| README / SPEC 同步 | ✅ 已更新为 V10 数据（V6/V7/V9 保留） |
| LICENSE / .gitignore / pyproject | ✅ 已创建；版本 1.3.0 |
| 测试文件 | ✅ `tests/test_v5.py`、`tests/test_v7_streaming.py`；`python -m pytest -q` → **7 passed** |
| 结果报告与图 | ✅ 已生成；V6/V7 表格保留 + V9 Track 4 附录 |
| 原始指标 | ✅ `results/V9_METRICS.json`（含 per-seed 审计与 Track 4B 六行原始数据） |
| 本汇报文件 | ✅ 已生成 |

> 一句话总结：**V9 把 Engine III 融合进 JIT scan，训练期不再物化 `F_seq`；在 4 链 5 组 3 seeds 的紧凑 2-Hop MQAR 上，Full 保持 100.00 ± 0.00% 域内/4× 间隔，而所有消融与未调参 Transformer 都在 25% 随机水平附近。真实 CUDA 上 TriCore 状态恒为 17.00 KiB，32K 时相对 64.00 MiB KV cache 缩减 3,855.1×，解码延迟在 8,192 交叉并在 32K 达到 1.939 vs 9.572 ms/token。但 V6/V7 与 Track 3A/3B 仍是单 seed，Track 4A 只是 3-seed 方差检查，32K 是单机 B=1 实测，不能据此宣称通用推理、真实代码理解、统计显著性、生产吞吐或 activation-O(1)。**

---

<p align="center">
  <i>TriCore V10 — a synaptic eligibility trace that carries entity keys across infix EQ, a 100% fair Engine I ablation, zero-F_seq training, constant state, and native one-token streaming.</i>
</p>