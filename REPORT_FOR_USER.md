# TriCore V12 完成报告

<!-- V12_REPORT_SECTION -->
## V12（v1.5.0）— 零先验自主语法发现、Delta 规则动态覆写与 1,048,576-token CUDA 实测

**本版重点。** V12 不移除任何 V4–V11 历史结果，新增五项交付：

1. **零先验自主语法发现。** `TriCoreLM(..., autonomous_syntax=True)` 在 `separator_token=None`、`syntax_tokens=None`、`reset_trace_tokens=None` 下，仅用当前序列的 token 频率与前驱频率发现结构 token、终止分隔符和中缀连接符。不传任何 token ID，不使用可学习角色表，也不增加循环状态。
2. **正确的结构门方向。** 中缀连接符处的资格迹门关闭后读取冻结的前一个实体 key，并阻止 `prev_k` 前进；写门在连接符和分隔符处保持关闭。这产生 `F(alias)=var` 的移位绑定，正好匹配查询 `[RET, alias, EQ, ...]`。
3. **可切换 Delta 规则。** `enable_delta_rule=True` 使用 `F_t = decay*F_{t-1} + lr*(V_t - F_{t-1}K_{t-1})K_{t-1}^T`；`False` 在相同 decay/lr 下使用纯 Hebbian 外积。参数量与 17,408-byte 状态完全不变。
4. **Track 7 动态变量覆写。** 变量先写入旧字面量，再覆写为新字面量，经 NOP 干扰后通过 alias 查询；目标是**新值**，四路随机基线 25.0%。这直接测试旧值擦除，而不是静态绑定。
5. **Track 4B 扩展到 1,048,576。** 用真实 CUDA 分配量与真实缓存解码延迟测量两种模型，不使用公式估算。

**发现规则（无 token ID、无参数、无循环）。** 对 `[B, L]` 序列统计 token 频率与前驱频率：`freq >= max(6, L//10)` 为结构 token；最后一个位置的结构 token 为终止分隔符；结构、非终止、且前驱频率 `<= 4` 的 token 为中缀连接符。规则只做相等与频率比较，因此任意 64-token 词表置换都会得到逐位相同的掩码。复杂度 `O(B*L^2)`，1M profile 关闭该路径。

**Track 5 协议（seeds 42/100/2026，350 步，batch 64，CUDA）。** 四变量、四查询、四个对抗 NOP；目标在 `EQ` 位置读出；四路随机基线 **25.0%**。

| 模型 | 参数量 | 验证 Loss（mean ± std） | `EQ` 两跳准确率（mean ± std） | `EQ`/实体处资格迹门 | 平均训练耗时 |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V12 Zero-Prior Full** | 61,851 | **0.00237 ± 0.00021** | **100.00 ± 0.00%** | 0.009 / 0.909 | 171.7 s |
| TriCore V12 Fair No-Engine-I | 61,851 | 1.4731 ± 0.0185 | 25.13 ± 0.67% | 1.0000 / 1.0000 | 141.9 s |
| Vanilla Transformer（2 层） | 83,456 | 1.4397 ± 0.0075 | 25.01 ± 0.63% | n/a / n/a | 13.9 s |

**Welch 检验（双尾）。** Full vs Fair 准确率：`t=192.524`，`df=2.000`，**`p=2.70e-05`**；Full vs Transformer 准确率：`t=206.869`，`df=2.000`，**`p=2.34e-05`**；Full vs Fair loss：`t=-137.883`，`df=2.001`，**`p=5.25e-05`**；Full vs Transformer loss：`t=-333.867`，`df=2.003`，**`p=8.81e-06`**。每组只有 3 个 seed，这是对 3 个观测值的精确算术，不是总体推断。

**Track 7 协议（seeds 42/100/2026，400 步，batch 64，CUDA）。**

每条序列先声明变量并写入旧字面量，再覆写为新字面量，插入 NOP 后通过 alias 查询；查询 `EQ` 的监督目标是**新字面量**，四路随机基线 25.0%。两个 TriCore 臂共享同一惊奇门写路径，唯一差异是 `enable_delta_rule=True`（带擦除项）与 `False`（纯 Hebbian 外积）；Fair No-Engine-I 保留 Delta 但关闭 `enable_surprise_gate`；Transformer 不接收 token-role 先验。seeds `42/100/2026`，400 步，batch 64，CUDA。

| 变体 | 参数量 | 新值准确率（mean ± std） | 旧值混淆（mean ± std） | 验证 Loss（mean ± std） | 平均训练耗时 |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V12 Delta Full** | 61,851 | **100.00 ± 0.00%** | **0.00 ± 0.00%** | **0.00227 ± 0.00012** | 175.0 s |
| TriCore V12 Hebbian Full | 61,851 | 73.71 ± 42.22% | 1.06 ± 0.95% | 0.6164 ± 0.9452 | 167.2 s |
| TriCore V12 Fair No-Engine-I Delta | 61,851 | 25.12 ± 0.41% | 0.00 ± 0.00% | 1.7284 ± 0.0234 | 141.5 s |
| Vanilla Transformer（2 层） | 83,456 | 24.28 ± 0.41% | 0.00 ± 0.00% | 1.4840 ± 0.0674 | 3.3 s |

逐 seed 新值准确率：Delta **100.00/100.00/100.00%**；Hebbian **98.67/97.50/24.96%**；Fair No-Engine-I Delta **24.65/25.39/25.31%**；Transformer **24.22/24.73/23.91%**。Delta 在三个 seed 上全部稳定，且从未返回旧值。Hebbian 在 seed 42/100 接近满分，但在 seed 2026 退化为随机水平（24.96%，验证 loss 1.7077），因此其三 seed 均值（73.71 ± 42.22%）不是稳健效应。**精确双尾 Welch 检验：** Delta vs Fair 新值 `t=318.03`，`df=2.00`，**`p=9.89e-06`**；Delta vs Transformer `t=316.76`，`df=2.00`，**`p=9.97e-06`**；Delta vs Hebbian `t=1.078`，`df=2.00`，**`p=0.394`**（不显著），旧值 `p=0.194`。可以支持的结论是：动态写入引擎是必要的（Delta 与成功训练的 Hebbian 都远高于 25.0% 随机基线），而 Delta 是三个 seed 都保持稳定的版本；每组只有 3 个观测值，**不能**据此宣称总体层面的 Delta > Hebbian 优势。

**Track 4B 协议（batch 1，FP32，RTX 3060）。** Transformer 使用真实的 2 层缓存解码：每个长度都实际分配 K/V、写入并做全因果 cache 注意力。TriCore 使用原生 `forward_step` 状态路径。65K 以下 8 次 warm-up + 32 次计时；65K–131K 为 4 + 16；262K–1M 为 2 + 8。`Alloc` 是 `empty_cache()` 后 `torch.cuda.memory_allocated()` 的实测增量。

| 长度 | TriCore 状态（B） | Transformer K+V cache（B） | cache（二进制） | 缩减 | TriCore ms/token | Transformer ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 17,408 | 262,144 | 256.00 KiB | 15.1× | 2.596 | 1.277 |
| 512 | 17,408 | 1,048,576 | 1.00 MiB | 60.2× | 3.091 | 1.408 |
| 2,048 | 17,408 | 4,194,304 | 4.00 MiB | 240.9× | 3.249 | 1.675 |
| 8,192 | 17,408 | 16,777,216 | 16.00 MiB | 963.8× | 3.151 | 3.339 |
| 16,384 | 17,408 | 33,554,432 | 32.00 MiB | 1,927.5× | 2.893 | 5.311 |
| 32,768 | 17,408 | 67,108,864 | 64.00 MiB | 3,855.1× | 2.784 | 10.519 |
| 65,536 | 17,408 | 134,217,728 | 128.00 MiB | 7,710.1× | 3.221 | 18.257 |
| 131,072 | 17,408 | 268,435,456 | 256.00 MiB | 15,420.2× | 3.245 | 35.382 |
| 262,144 | 17,408 | 536,870,912 | 512.00 MiB | 30,840.5× | 3.039 | 71.015 |
| 524,288 | 17,408 | 1,073,741,824 | 1.00 GiB | 61,680.9× | 2.879 | 137.640 |
| **1,048,576** | **17,408** | **2,147,483,648** | **2.00 GiB** | **123,361.9×** | **3.794** | **274.275** |

TriCore 状态在所有长度恒为 **17,408 B = 17.00 KiB / 样本**。`L=1,048,576` 时，真实缓存 Transformer 的 K/V 分配为 **2,147,483,648 B（2.00 GiB）**，是 TriCore 状态的 **123,361.9×**；解码延迟为 **274.275 ms/token**，TriCore 为 **3.794 ms/token**，前者慢 **72.30×**。实测延迟交叉点在 `L=8,192`（3.339 vs 3.151 ms/token）；在其之下，小尺寸缓存 Transformer 更快。65,536–131,072 使用 16 次计时 + 4 次 warm-up，262,144–1,048,576 使用 8 次计时 + 2 次 warm-up；32,768 仍为 32 + 8。以上均为单机 RTX 3060、batch 1、FP32 的实测，不是服务吞吐结论。

> **范围。** Track 5/Track 7 每组只有 3 个观测值，不构成总体推断；Track 4B 是单机 B=1 实测，不是服务吞吐。自主发现是面向规则闭合语法的短序列角色先验，不是通用 parser。100% 结果来自闭合合成语法，不外推为通用推理能力。

**V12 实测耗时：** **493.8 s**（Track 5 三 seed + Track 7 四臂 × 三 seed + Track 4B 十一个长度；RTX 3060 参考机）。



## V11（v1.4.0）— 四跳深链召回、三 seed Welch 检验与 128K CUDA 实测

**本版重点。** V11 不修改 V10 的公式与参数命名，新增四项交付：

1. **Track 6：单次前向四跳链式召回。** 任务由 4 层随机打乱的事实三元组 `[key, value, 0]`、12 个干扰 token 和 4 个 `[key_q, 0]` 查询组成，目标为 `k_q -> m1_q -> m2_q -> m3_q -> v_q`。新增 `reset_trace_tokens=(0,)`：在每个语句分隔符后把每个 head 的资格迹 key `prev_k` 清零，避免上一个干扰桥接 token 泄漏到查询读出。reset 掩码同时接入 JIT `tricore_fused_scan` 与 `forward_step`，全序列与逐 token 路径在测试中以 `atol=1e-4` 对齐。
2. **Track 5 三 seed 显著性。** 42/100/2026 三个 seed 重复中缀 `EQ` 两跳解引用，使用 Welch 不等方差 t 检验（由正则化不完全 Beta 函数实现，无 SciPy 依赖）。
3. **Track 4B 扩展到 131,072。** 真实 CUDA 分配量与真实缓存解码延迟，覆盖 128 到 128K。
4. **2×3 六面板主图。** 同时写出 `results/tricore_vs_transformer.png` 与 `results/tricore_v11_masterpiece.png`。

**Track 6 协议（单 seed 42，500 步，batch 64，CUDA）。** 4 条链、4 个查询、12 个干扰 token（序列长度 68）；4× OOD 使用 48 个干扰 token（长度 104）；随机基线 **25.0%**。

| Track 6 变体 | 参数量 | 域内 | 4× OOD | 查询位置路由 `[h1,h2,h3,h4]` | 训练耗时 |
| :--- | ---: | ---: | ---: | :--- | ---: |
| **TriCore Full（1 层，`max_loops=3`）** | 236,436 | **100.00%** | **100.00%** | 0.005 / 0.004 / 0.003 / **0.988** | 214.2 s |
| TriCore（1 层，`max_loops=2`，三跳上限） | 232,211 | 24.95% | 24.76% | 0.002 / **0.997** / 0.001 / 0.000 | 184.0 s |
| TriCore（1 层，`max_loops=1`，两跳上限） | 227,986 | 25.73% | 26.27% | 0.019 / **0.981** / 0.000 / 0.000 | 148.8 s |
| 2 层 Transformer（`d_model=128`，4 heads） | 306,176 | 24.41% | 25.78% | n/a | 4.0 s |

**解读。** 只有 `max_loops=3` 的 Full 在单次前向中走通四条边，达到 **100.00% 域内 / 100.00% 四倍间隔外推**，并把 **98.8%** 的查询路由权重放在第 4 跳。`max_loops=2` 与 `max_loops=1` 分别把 **0.997 / 0.981** 的权重压到各自最后一个可用槽位，但任务准确率仍在随机水平；这正是“深度预算不足”的特征，而不是学到了捷径。Transformer 在相同小预算下同样处于随机水平。Track 6 是**单 seed 控制实验**，不做显著性宣称。三组 TriCore 深度变体使用相同的显式 token-role 先验（`separator_token=0`、`syntax_tokens=(0,)`、`reset_trace_tokens=(0,)`），该先验不是可学习 embedding；Transformer 不接收该先验。

**Track 5 协议（seeds 42/100/2026，350 步，batch 64，CUDA）。** 声明、赋值、对抗 NOP 与查询格式沿用 V10；目标在 `EQ` 位置读出，字面量候选 16 个，四路随机基线 25.0%。

| 模型 | 参数量 | 验证 Loss（mean ± std） | `EQ` 两跳准确率（mean ± std） | `EQ`/实体处资格迹门 | 平均训练耗时 |
| :--- | ---: | :---: | :---: | :---: | ---: |
| **TriCore V11 Full（资格迹）** | 60,819 | **0.00218 ± 0.00022** | **100.00 ± 0.00%** | 0.0032 / 0.9054 | 156.3 s |
| TriCore V11 Fair No-Engine-I（无调制门） | 60,819 | 1.4682 ± 0.0186 | 24.80 ± 1.12% | 1.0000 / 1.0000 | 129.0 s |
| Vanilla Transformer（2 层） | 83,456 | 1.4397 ± 0.0075 | 25.01 ± 0.63% | n/a / n/a | 14.8 s |

**Welch 检验（双尾，Welch-Satterthwaite 自由度）。** Full vs Fair 准确率：`t=116.649`，`df=2.000`，**`p=7.35e-05`**；Full vs Transformer 准确率：`t=206.869`，`df=2.000`，**`p=2.34e-05`**；Full vs Fair 验证 loss：**`p=5.34e-05`**；Full vs Transformer loss：**`p=8.81e-06`**。每组仅 3 个 seed，这是对 3 个观测值的精确算术，不是总体推断；正态性/等样本量假设也只是近似满足。

**Track 4B Extended 协议（单机，batch 1，FP32）。** Transformer 使用真实的 2 层缓存解码：每个长度都实际分配 K/V、写入并做全因果 cache 注意力；`Alloc` 是 `torch.cuda.memory_allocated()` 在 `empty_cache()` 后的增量，不是公式估算。65K 以下为 8 次 warm-up + 32 次计时，65K/128K 为 4 + 16 次。

| 长度 | TriCore 状态 | Transformer K+V cache | 缩减 | TriCore ms/token | Transformer ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 17,408 B | 262,144 B | 15.1× | 2.971 | 1.415 |
| 512 | 17,408 B | 1,048,576 B | 60.2× | 2.571 | 1.387 |
| 2,048 | 17,408 B | 4,194,304 B | 240.9× | 2.866 | 1.549 |
| 8,192 | 17,408 B | 16,777,216 B | 963.8× | 2.922 | 3.865 |
| 16,384 | 17,408 B | 33,554,432 B | 1,927.5× | 3.408 | 5.421 |
| 32,768 | 17,408 B | 67,108,864 B | 3,855.1× | 2.767 | 9.735 |
| 65,536 | 17,408 B | 134,217,728 B | 7,710.1× | 2.719 | 18.275 |
| **131,072** | **17,408 B** | **268,435,456 B（256.00 MiB）** | **15,420.2×** | **2.737** | **35.548** |

TriCore 状态在所有长度恒为 **17,408 B = 17.00 KiB / 样本**；128K 时 KV cache 为 **256.00 MiB**，张量字节缩减 **15,420.2×**。TriCore 延迟稳定在 **2.57–3.41 ms/token**，Transformer 从 **1.39 增至 35.55 ms/token**；实测交叉点为 `L=8,192`，128K 时 TriCore 每 token 快 **12.99×**。这是单机 RTX 3060、B=1 的测量，不是服务吞吐结论。

**六面板图。** `plot_masterpiece` 同时输出 `results/tricore_vs_transformer.png` 与 `results/tricore_v11_masterpiece.png`：多跳深度、Track 5 与 Welch p 值、Track 4A 五组消融、128K 显存、128K 延迟、代码建模。图脚注明确标注 Track 6/4B 为单 seed 控制实验。

## V10（v1.3.0）— 突触资格迹 + 100% 公平 Engine I 消融

**本版重点。** 每个 head 增加突触资格迹 `prev_k`，在常量中缀分隔符 `EQ` 处
保留实体 key；资格迹同时用于 fast-weight 写入与读出查询
`q_read = normalize(elig_gate * q_t + (1 - elig_gate) * prev_k)`。

**自我审计修正。**
1. 旧稿只在写入路径使用 `prev_k`，`EQ` 读查询仍使用常量地址，导致 alias 碰撞；
   现在 `EQ` 位置直接查询锁存的实体 key。
2. 旧 No-Engine-I 消融硬编码 `fast_lr=0.95`、`fast_decay=0.82`；公平消融保留
   与 Full 完全相同的可学习 `fast_lr`、`fast_decay` 与 Q/K/V 投影，仅把动态
   write/eligibility gate 固定为 `1.0`。
3. 旧 Track 5 查询把答案 `lit_q` 写进了输入；现在查询为
   `[RET, alias_q, EQ, SEMI, SEMI]`，只在 `EQ` 位置（`pos + 2`）读取两跳目标。

**Track 5 协议（350 步，batch 64，seed 42，CUDA）。**
- 声明：`[LET, alias_i, EQ, var_i, SEMI]`
- 赋值：`[SET, var_i, EQ, lit_i, SEMI]`
- 对抗干扰：`[NOP, var_i, EQ, SEMI, SEMI]`
- 查询：`[RET, alias_q, EQ, SEMI, SEMI]`，在 `EQ` 处评测两跳目标 `lit_q`。
- `EQ` 同时关闭 eligibility 与 write gate；`SEMI/LET/SET/RET/NOP` 只关闭
  write gate，使语句边界刷新 `prev_k` 而不覆写实体记忆。
- 目标从 16 个字面量候选中选取：均匀 16 路随机基线为 **6.25%**；若模型先锁定
  四个 SET 字面量，则四路随机基线为 **25.0%**。

| 模型 | 参数量 | 验证集 Loss | Infix `EQ` 两跳解引用准确率（%） | `EQ`/实体处平均资格迹门 | 训练耗时（s） |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **TriCore V10 Full (with eligibility trace)** | 60,819 | 0.0021 | **100.00%** | 0.0036 / 0.9011 | 166.8 |
| **TriCore V10 Fair No-Engine-I (unmodulated gates)** | 60,819 | 1.4884 | **24.18%** | 1.0000 / 1.0000 | 137.6 |
| **Vanilla Transformer (2-layer)** | 83,456 | 1.4349 | **25.47%** | n/a / n/a | 15.4 |

**解读。** Full 达到 **100.00%**，在 `EQ` 处关闭资格迹门
（均值 0.0036），在实体位置保持开启
（均值 0.9011）。公平 No-Engine-I 消融为
**24.18%**；没有门控时每个 `EQ` 都会覆写 alias 资格迹。
注意力基线在相同的小训练预算下仍明显低于 Full；这是单 seed 观察，不是对 Transformer 的普遍结论。 本结果来自单机单 seed，是方差检查，不是显著性检验。

---

**日期：** 2026-10-24  
**版本：** `tricore-arch` 1.5.0  
**状态：** V12 完成；新增零先验无监督语法角色发现、可切换 Delta rule、Track 7 动态变量覆写、1,048,576-token 真实 CUDA 显存/延迟实测与 2×3 主图；V11 四跳链、V10 突触资格迹、V9 零 `F_seq`、V7 原生 O(1) 流式与真实 Python 源码 Held-Out LM 全部保留  
**实测报告：** `results/BENCHMARK_REPORT.md`  
**图表：** `results/tricore_vs_transformer.png`、`results/tricore_v11_masterpiece.png`  
**运行设备：** CUDA（RTX 3060；V11 Track 5/6 与 Track 4B 实测）；CUDA（V9/V10 历史）

---

## 0. 一页结论

**V11（v1.4.0）结论：** Track 6 在单次前向中走通四跳，Full 达到 **100.00% 域内 / 100.00% 4× 干扰间隔**，查询路由把 **0.988** 的权重放在第 4 跳；三跳与两跳上限模型分别把 **0.997 / 0.981** 放在最后可用槽位，但准确率仍为 **24.95/24.76%** 与 **25.73/26.27%**，2 层 Transformer 为 **24.41/25.78%**。Track 5 三个 seed（42/100/2026）Full 为 **100.00 ± 0.00%**、公平 No-Engine-I 为 **24.80 ± 1.12%**、Transformer 为 **25.01 ± 0.63%**；Welch 双尾 `p=7.35e-05`（vs Fair）与 `p=2.34e-05`（vs Transformer）。Track 4B 在 **L=131,072** 实测 TriCore 状态仍为 **17,408 B（17.00 KiB）**，Transformer KV cache 为 **256.00 MiB**，缩减 **15,420.2×**；延迟 **2.737 vs 35.548 ms/token**（12.99×）。完整 V11 脚本耗时 **1,556.8 s**。Track 6/4B 为单 seed/单机控制实验，Track 5 的 3 个观测值也不构成总体推断。

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
| `run_v11_flagship.py` | ✅ 新增 | V11 Track 5 三 seed Welch + Track 6 四跳 + 128K Track 4B + 2×3 主图；写出 `V11_METRICS.json` |
| `run_v8_verification.py` | ✅ 兼容别名 | 保留旧命令入口，实际调用 `run_v9_verification.main` |
| `demo_showcase.py` | ✅ 已创建 | 一键 3-hop 流式演示，无需 checkpoint，CPU/CUDA 均可运行 |
| `LICENSE` | ✅ 已创建 | MIT License |
| `.gitignore` | ✅ 已创建 | Python 缓存、虚拟环境、编辑器文件；保留 `results/` 报告与 PNG |
| `pyproject.toml` | ✅ 已更新 | `tricore-arch` 1.5.0，Python ≥3.10，torch/numpy/matplotlib，MIT |
| `README.md` | ✅ 已更新 | V12 零先验/Delta/1M 顶部章节 + V11/V10/V9/V6/V7 历史数据保留 |
| `SPEC_V2.md` | ✅ 已更新 | V12 规范与 Track 5/7/4B，V11/V10/V9 历史规范保留 |
| `requirements.txt` | ✅ 已存在 | `torch>=2.0`、`numpy>=1.24`、`matplotlib>=3.7` |
| `tricore/__init__.py` | ✅ 已更新 | 导出 block/LM/baseline 与全部任务加载器 |
| `tricore/model.py` | ✅ 已更新 | `tricore_fused_scan`、双模型、`init_state` / `forward_step`、`reset_trace_tokens` / `reset_mask`、V12 `autonomous_syntax` 零先验 mask 与 `enable_delta_rule` |
| `tricore/tasks.py` | ✅ 已更新 | 2-Hop MQAR + 因果字节码 + 3-Hop/4-Hop 召回 + Track 7 动态覆写 + 真实 Python 源码 held-out 字节流 |
| `tests/test_sanity.py` | ✅ 已存在 | 前向/反向/形状/梯度流 |
| `tests/test_v5.py` | ✅ 已更新 | JIT 三元组对 Python 参考的四组 scan flag 等价性、零 `F_seq` 检查 + 因果字节码形状/反向 |
| `tests/test_v7_streaming.py` | ✅ 已更新 | 流式等价、17 KiB 恒定状态、3-Hop/真实语料梯度流、4-Hop reset parity、V12 Delta parity、零先验 mask 置换不变性 |
| `run_experiment.py` | ✅ 已更新 | V6 双赛道、Track 1 五组消融、grad clipping、报告与 2×2 图 |
| `run_v7_benchmark.py` | ✅ 已创建 | 仅运行 V7 流式检查与 Track 3，不重跑 V6 |
| `run_v12_apex.py` | ✅ 已创建 | V12 零先验 Track 5/7、Delta/Hebbian、1M CUDA 与 2×3 图 |
| `results/BENCHMARK_REPORT.md` | ✅ 已生成 | V12 顶部章节 + V11/V10/V9/V7/V6 历史表格 |
| `results/V9_METRICS.json` | ✅ 已生成 | V9 全部 Track 4A/4B 原始指标与协议 |
| `results/V11_METRICS.json` | ✅ 已生成 | V11 Track 5/6/4B 原始指标、Welch 检验与协议 |
| `results/V12_METRICS.json` | ✅ 已生成 | V12 零先验 Track 5/7、Delta/Hebbian、1M CUDA 原始指标与协议 |
| `results/tricore_vs_transformer.png` | ✅ 已生成 | V12 2×3 六面板主图（master 路径） |
| `results/tricore_v12_apex.png` | ✅ 已生成 | V12 2×3 主图发布路径 |
| `results/tricore_v11_masterpiece.png` | ✅ 已生成 | V11 历史六面板图 |
| `REPORT_FOR_USER.md` | ✅ 本文件 | 完整交付说明 |

**Git-ready 说明：** 仓库根目录是 Python 项目，无密钥、无本地绝对路径、无大数据集；`results/BENCHMARK_REPORT.md`、`results/V12_METRICS.json` 与主图作为可复现证据保留。提交前建议执行：

```bash
git status --short
python -m pytest -q
python run_experiment.py --steps 2 --code-steps 2   # 可选快速冒烟
```

---

## 6. 复现命令

**V12 新增（v1.5.0）：**

```bash
python run_v12_apex.py          # 零先验 Track 5 三 seed + Track 7 Delta/Hebbian 三 seed + 1M-token CUDA + 2×3 主图
python run_v12_apex.py --steps5 5 --steps7 10 --seeds 42 --max-length 512 --skip-plot --metrics _smoke_v12.json
```

**V11 新增（v1.4.0）：**

```bash
python run_v11_flagship.py      # Track 5 三 seed Welch + Track 6 四跳 + 128K Track 4B + 2×3 主图
```

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

- `tricore/model.py`：`@torch.jit.script tricore_fused_scan`（零 `F_seq` 三元组返回）、融合多跳 `TriCoreBlock`、零先验 `_discover_autonomous_masks`、`TriCoreLM`、`BaselineTransformerLM`、`init_state` / `forward_step`
- `tricore/tasks.py`：2-Hop MQAR、因果字节码 `generate_contextual_code_lm_batch`、3-Hop `generate_3hop_recall_batch`、真实源码 `load_real_python_corpus_batches`
- `run_experiment.py`：Track 1 五组消融、Track 2 两模型、V6 计时、cache 核算、Markdown 报告、matplotlib 2×2 图
- `run_v7_benchmark.py`：流式等价/延迟/状态核算、Track 3A 三模型、Track 3B 两模型，并追加报告
- `tests/test_v5.py`：JIT 与 Python 参考在 `enable_surprise_gate` × `enable_fast_weight` × `enable_adaptive_loop` 组合下对 `h_seq`、`mem_seq`、`surprise_trace` 的 `allclose` 等价性；`forward` 与逐 token `forward_step` 的 `<1e-6` 等价性；`TriCoreLM` 前向/反向；因果字节码的形状、`pos + 1` mask、反向梯度检查
- `run_v12_apex.py`：零先验 Track 5/7、Delta/Hebbian 对照、1,048,576-token CUDA 实测、2×3 主图
- `tests/test_v7_streaming.py`：流式等价、17,408 bytes 恒定状态、3-Hop/真实语料形状与梯度检查、4-Hop `reset_trace_tokens` 路径与 parity、V12 Delta parity、零先验 mask 的 Track 5/7 与词表置换不变性；完整套件 **11 passed**

---

## 7. 结论：V11 / V9 到底证明了什么

V11 新增的证据是：在单次前向中，Track 6 Full 走通四条指针边，域内 **100.00%**、4× 间隔 **100.00%**，查询路由第 4 跳权重 **0.988**；三跳/两跳上限模型把 **0.997/0.981** 放在最后可用槽位，但准确率只有 **24.95/24.76%** 与 **25.73/26.27%**，2 层 Transformer 为 **24.41/25.78%**。Track 5 用三个 seed 做 Welch 检验：Full **100.00 ± 0.00%**、Fair No-Engine-I **24.80 ± 1.12%**、Transformer **25.01 ± 0.63%**，准确率 p 值为 **7.35e-05** 与 **2.34e-05**。Track 4B 实测到 `L=131,072`：状态恒为 **17.00 KiB**，KV cache 为 **256.00 MiB**，缩减 **15,420.2×**，延迟 **2.737 vs 35.548 ms/token**。但 Track 6/4B 是单 seed/单机控制实验，Track 5 只有 3 个观测值，因此不是总体显著性、通用推理或生产吞吐结论。

V9 的证据是：训练期不再物化 `F_seq`，Engine III 已融合进 JIT scan；在 4 链、5 组、3 seeds 的紧凑 2-Hop MQAR 上，4× 间隔召回为 Full **100.00 ± 0.00%**、No Engine I **5.55 ± 0.83%**、No Engine II **5.31 ± 0.95%**、No Engine III **23.54 ± 2.73%**、Transformer **22.24 ± 3.96%**；只有 Full 同时饱和域内与 4× 间隔。真实 CUDA 分配下，TriCore 状态恒为 **17.00 KiB/样本**，`L=32,768` 的 Transformer KV cache 为 **64.00 MiB**，缩减 **3,855.1×**；实测延迟在 `L=8,192` 交叉（1.755 vs 3.011 ms/token），32K 为 1.939 vs 9.572。这仍是 3 个 seed 的方差检查；完整 V6/V7 表格仍为单 seed，V9 脚本耗时 **475.6 s**。

### 已证明（在所述任务与预算内）

1. **Full TriCore 能完成 2-Hop 链式联想召回**，并在 4× 对抗噪声间隔外推下保持 100.00%。
2. **Engine II 和 Engine III 是该任务的关键**：去掉快权重或去掉隐空间重查询都会跌到 25.0% 随机水平附近（5.31% / 23.54%）。
3. **Engine I 在本次 V9 4 链运行中是必要的**：去掉后跌到 5.65%/5.55%，与 V5 的 91.88%/84.14% 不同；这提醒我们单 seed 消融结论的脆弱性，也说明 3-seed 方差检查仍不能替代显著性检验。
4. **V6/V9 把多跳重查询做成跨全部位置的一次批量 5-D `matmul`**，V9 进一步把 Engine III 融合进 JIT scan；状态递归仍是串行，不能把它描述成并行前缀扫描或 kernel 级优化。
5. **O(1) 状态是真实的**：17.00 KiB 与序列长度无关；V11 在真实 CUDA 上实测到 128K，KV cache 为 256.00 MiB，缩减 15,420.2×。
6. **第二赛道管线可运行**：TriCore 在因果字节码任务上 PPL 3.17、解引用 100.00%；2 层 Transformer 为 3.56 / 19.63%，接近 16.67% 随机水平。这证明封闭语法上的两级指针查找可学习，不证明真实代码理解。
7. **流式路径与完整前向等价**：最大 logit 误差 5.96e-07，持久状态恒为 17,408 bytes，4,096 步实测 2.160 ms/token；V9 训练期不再物化 `F_seq`。
8. **Track 3A 证明第二次 refinement 是必要的**：Full `max_loops=2` 为 100.00%/100.00%，`max_loops=1` 只有 25.39%/23.91%，差 74.61 个百分点；2 层 Transformer 为 23.83%/23.44%。
9. **Track 3B 在真实源码的后缀上完成了 held-out 比较**：TriCore 223,887 参数达到 PPL 18.616 / 41.47%，参数匹配 Transformer 218,448 参数为 19.428 / 28.15%；TriCore 训练慢约 38×，且语料仅为本仓库源码。
10. **Track 4B 在真实 CUDA 上测到 128K**：TriCore 状态恒为 17,408 B；Transformer 在 128/512/2,048 更快，但延迟在 8,192 交叉，128K 为 35.548 vs 2.737 ms/token，KV cache 256.00 MiB vs 17.00 KiB。这是单机 B=1 结果，不是服务吞吐。
11. **Track 4A 给出了 5 组 × 3 seeds 的 mean ± std**：Full 的 seed 标准差为 0，所有非 Full 变体非零；这是方差检查。
12. **Track 6 证明四跳路由在单次前向中可学**：Full 第 4 跳权重 0.988 且召回 100.00%；三跳/两跳上限模型只能把权重压到最后一个槽位，任务仍为随机水平。
13. **Track 5 给出了三 seed Welch 检验**：Full 与 Fair/Transformer 的准确率和 loss 差异均达到 `p<1e-4`；但每组仅 3 个观测值，不是总体推断。

### 未证明 / 不应宣称

- 自然语言、通用推理或大规模真实代码能力；Track 3B 只是本仓库的小型源码后缀。
- 对调参后 Transformer 的全面优势；Track 1/3A 的 Transformer 未调参，Track 3B 只匹配参数量、未匹配算力。
- V6/V7 全尺寸赛道、Track 3A/3B 的多 seed 稳健性；V9 Track 4A 与 V11 Track 5 各只有 3 个观测值，不是置信区间或功效分析。
- 生产吞吐、128K 以上长上下文、分布式/混合精度/量化能力。
- 训练显存的 O(1) 性质：V9 已删除 `F_seq` 大张量，但训练仍保存 `h_seq`、`mem_seq` 与 autograd 缓冲，不能宣称 activation-O(1)；只有持久推理状态是 O(1)。
- “Engine I 在所有任务上都必要”或“三个引擎缺一不可”的普适结论；本次 No-I 跌到随机，但 V5 没有，说明结论对优化轨迹和 seed 敏感。
- JIT 已接近硬件最优；它仍是编译后的 PyTorch 循环，状态递归仍串行。

---

## 8. 建议的下一步

1. **多 seed**：V11 已把 Track 5 做到 3 seed Welch；下一步是把同样协议扩展到 V6/V7 全尺寸赛道与 Track 3A/3B，并报告置信区间/功效分析。
2. **更强基线**：调参 Transformer，匹配非位置参数量、训练 token 与算力；当前 Track 3B 只匹配参数量。
3. **更大真实代码语料**：用 held-out 文件/仓库替代本仓库 15% 后缀，才能检验真实代码泛化。
4. **5–8 hop**：V11 已验证单次前向四跳；下一步验证 latent re-query 是否随跳数继续扩展，以及 router 是否学会使用额外循环。
5. **真正 kernel**：在 JIT + 向量化多跳 + 原生流式的基础上做 fused/chunked CUDA/Triton scan；减少小 kernel launch 与串行状态更新成本。
6. **更长上下文实测**：V11 已用真实 KV cache 对比到 128K 的延迟与显存；下一步是 1M，并比较生产级 serving stack。

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
| V10 双层 token-role 门网络 | ✅ `write_gate_proj` / `elig_gate_proj` 为 `Linear -> SiLU -> Linear`，末层 `Normal(0, 0.02)` + bias `2.0` 保证梯度畅通；`EQ` 双掩码与语法 token 写掩码防止漂移 |
| V10 Track 5：跨 `EQ` 中缀解引用 | ✅ 350 步、batch 64、seed 42；Full 100.00% / 公平 No-Engine-I 24.18% / Transformer 25.47% |
| V10 100% 公平 Engine I 消融 | ✅ 不覆盖 `fast_lr`、`fast_decay`；与 Full 共享 Q/K/V 投影和门网络，仅关闭惊奇调制 |
| V11 `reset_trace_tokens` 四跳复位 | ✅ 分隔符后清零 `prev_k`；JIT scan 与 `forward_step` 同步，无新增可训练参数 |
| V11 Track 6：四跳单次前向 | ✅ Full 100.00%/100.00%，路由第 4 跳 0.988；三跳/两跳上限 24.95/24.76% 与 25.73/26.27% |
| V11 Track 5：三 seed Welch | ✅ Full 100.00 ± 0.00%，Fair 24.80 ± 1.12%，Transformer 25.01 ± 0.63%；p=7.35e-05 / 2.34e-05 |
| V11 Track 4B：128K 真实 CUDA | ✅ 17,408 B 恒定 vs 256.00 MiB = 15,420.2×；128K 延迟 2.737 vs 35.548 ms/token |
| V11 六面板主图 | ✅ `results/tricore_vs_transformer.png` + `results/tricore_v11_masterpiece.png` |
| V12 零先验自主语法发现 | ✅ 无 token ID / 无学习角色表；任意 64-token 词表置换后 EQ/SEMI mask 逐位一致；状态仍为 17,408 B |
| V12 Track 5：三 seed 零先验 | ✅ Full 100.00 ± 0.00%，Fair 25.13 ± 0.67%，Transformer 25.01 ± 0.63%；Welch p=2.70e-05 / 2.34e-05 |
| V12 Track 7：Delta vs Hebbian | ✅ Delta 新值 100.00 ± 0.00%、旧值 0.00%；Hebbian 73.71 ± 42.22%（seed 2026 失稳至 24.96%）；Fair 25.12%；Delta vs Hebbian p=0.394（不显著，如实保留） |
| V12 Track 4B：1M 真实 CUDA | ✅ 17,408 B 恒定 vs 2,147,483,648 B（2.00 GiB）= 123,361.9×；1M 延迟 3.794 vs 274.275 ms/token（72.30×） |
| V12 2×3 主图 | ✅ `results/tricore_v12_apex.png` + `results/tricore_vs_transformer.png` |
| V12 文档同步 | ✅ README / SPEC / REPORT / BENCHMARK 顶部 V12 章节；V4–V11 全部历史表格与审计记录保留 |
| README / SPEC 同步 | ✅ 已更新为 V11 数据（V10/V9/V6/V7 保留） |
| LICENSE / .gitignore / pyproject | ✅ 已创建；版本 1.5.0 |
| 测试文件 | ✅ `tests/test_v5.py`、`tests/test_v7_streaming.py`；`python -m pytest -q` → **11 passed**（含 V12 Delta parity 与零先验 mask 置换不变性） |
| 结果报告与图 | ✅ 已生成；V6/V7/V9/V10/V11 历史表格保留 + V12 Track 5/7/4B 顶部章节 |
| 原始指标 | ✅ `results/V12_METRICS.json` + `results/V11_METRICS.json` + `results/V10_METRICS.json` + `results/V9_METRICS.json` |
| 本汇报文件 | ✅ 已生成 |

> 一句话总结：**V11 在单次前向中走通四跳：Full 达到 100.00% 域内 / 100.00% 四倍间隔，第 4 跳路由权重 0.988；三跳与两跳上限仍在 24.76% / 26.27%，2 层 Transformer 为 25.78%。Track 5 三 seed Welch：Full 100.00%，Fair 24.80%，Transformer 25.01%，p=7.35e-05 / 2.34e-05。128K 实测状态恒为 17.00 KiB vs 256.00 MiB KV cache（15,420.2×），延迟 2.737 vs 35.548 ms/token。V10 突触资格迹、V9 零 `F_seq`、V7 原生 O(1) 流式与全部历史数据继续保留。Track 6/4B 仍是单 seed/单机结果，Track 5 只有 3 个观测值；不能据此宣称通用推理、真实代码理解、总体显著性、生产吞吐或 activation-O(1)。**

> V12 一句话总结：**零先验 Track 5 在三个 seed 上保持 100.00 ± 0.00%，无 token ID、无学习角色表；Track 7 的 Delta 臂新值 100.00 ± 0.00%、旧值 0.00%，Fair 与 Transformer 停在 ~25% 随机水平。Hebbian 在 seed 2026 失稳至 24.96%，因此 Delta vs Hebbian 的 p=0.394 不显著，报告不做优势宣称。Track 4B 在 L=1,048,576 实测 17,408 B 恒定状态 vs 2.00 GiB KV cache（123,361.9×），延迟 3.794 vs 274.275 ms/token（72.30×）。V4–V11 全部历史数据保留；以上为闭合合成语法、单机 B=1、每组 3 个观测值的实验记录，不是通用推理或生产吞吐结论。**

---

<p align="center">
  <i>TriCore V12 — zero-prior autonomous syntax discovery, Delta-rule live overwrite, one-million-token CUDA profiling, and constant 17 KiB state.</i>
</p>