# TriCore V6 完成报告

**日期：** 2026-10-20  
**版本：** `tricore-arch` 0.6.0  
**状态：** V6 双赛道实验完成；向量化多跳扫描与因果字节码解引用已接入；MIT 开源套件就绪  
**实测报告：** `results/BENCHMARK_REPORT.md`  
**图表：** `results/tricore_vs_transformer.png`  
**运行设备：** CUDA

---

## 0. 一页结论

V6 在不改变 V4/V5 数学公式、参数量和 O(1) 状态结构的前提下，完成了三件事：

1. **运行时：** 把 O(1) 状态演化保留在 JIT 循环中，把 Engine III 的多跳重查询改为 `F_seq: [B, L, H, d_head, d_head]` 上的批量 5-D `matmul`。Python 只循环固定的 `max_loops`，不再逐 token 循环。Full 训练 500 步的 V6 / V5 / V4 墙钟对比见 §1。
2. **第二赛道：** 把 V5 的 `EQ` 语句改为因果字节码 `[LET, alias, var, SEMI]`、`[ALIAS, var, lit, SEMI]`、`[NOP, var, SEMI, SEMI]`、`[RET, alias, lit, SEMI]`，并把解引用目标移到 `pos + 1` 的返回别名位置，强制 `alias_i -> var_i -> lit_i` 两级指针解析。PPL 与解引用准确率见 §3。
3. **开源交付：** 根目录已有 `LICENSE`（MIT）、`.gitignore`、`pyproject.toml`；README 与 SPEC 已同步为 V6 实测数据；实验报告和 2×2 图表已重新生成。

**必须同时看到的限制：** 两个赛道都是单 seed 合成任务。Track 1 的 Full 达到 100%/100%，但 No-Engine-I 本次为 **5.55%/4.14%**，跌到随机水平，和 V5 的 91.88%/84.14% 不同；公式没变，向量化多跳路径改变了单 seed 优化轨迹，不能把任一结果当作 Engine I 的普适结论。Track 2 的 **100.00%** 是在封闭、确定性的合成字节码语法上取得的，没有 held-out 模板，不能解读为“理解了真实代码”或“通用语言建模已被证明”。

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

## 4. 推理状态与 KV cache 对比

单样本、FP32、`D=128`、`H=4`、`d_head=32`、Transformer `n_layers=2`。

| 序列长度 | Transformer KV cache | TriCore O(1) 状态 | 缩减倍数 |
| :---: | :---: | :---: | :---: |
| 128 | 256.00 KB | 17.00 KB | 15.1× |
| 512 | 1,024.00 KB | 17.00 KB | 60.2× |
| 2,048 | 4,096.00 KB | 17.00 KB | 240.9× |
| 8,192 | 16,384.00 KB | 17.00 KB | 963.8× |
| 32,768 | 65,536.00 KB | 17.00 KB | **3,855.1×** |

TriCore 单核状态为 `h`（128 floats）+ `F_mat`（4,096 floats）+ `prev_k`（128 floats）= **4,352 floats = 17,408 bytes = 17.00 KiB**，与序列长度无关。Transformer 公式为 `n_layers * 2 * L * D * 4 / 1024` KiB。该表是状态核算，不是实测解码服务基准。

---

## 5. 开源文件就绪清单

| 文件 / 目录 | 状态 | 内容 |
| :--- | :---: | :--- |
| `LICENSE` | ✅ 已创建 | MIT License |
| `.gitignore` | ✅ 已创建 | Python 缓存、虚拟环境、编辑器文件；保留 `results/` 报告与 PNG |
| `pyproject.toml` | ✅ 已创建 | `tricore-arch` 0.6.0，Python ≥3.10，torch/numpy/matplotlib，MIT |
| `README.md` | ✅ 已更新 | V6 标题、向量化多跳、因果字节码、双赛道实测表、限制、复现命令、不变量 |
| `SPEC_V2.md` | ✅ 已更新 | V6 JIT 四元组合约、向量化 Engine III、状态核算、任务令牌布局、协议、实测记录 |
| `requirements.txt` | ✅ 已存在 | `torch>=2.0`、`numpy>=1.24`、`matplotlib>=3.7` |
| `tricore/__init__.py` | ✅ 已更新 | 导出 `TriCoreBlock`、`TriCoreLM`、`BaselineTransformerLM` 与两个任务生成器 |
| `tricore/model.py` | ✅ 已更新 | `tricore_fused_scan`、双模型 |
| `tricore/tasks.py` | ✅ 已更新 | 2-Hop MQAR + 因果字节码 Contextual Code LM |
| `tests/test_sanity.py` | ✅ 已存在 | 前向/反向/形状/梯度流 |
| `tests/test_v5.py` | ✅ 已创建 | JIT 四元组对 Python 参考的四组 scan flag 等价性 + 因果字节码形状/反向 |
| `run_experiment.py` | ✅ 已更新 | 双赛道、Track 1 五组消融、grad clipping、报告与 2×2 图 |
| `results/BENCHMARK_REPORT.md` | ✅ 已生成 | 本次 V6 实测报告 |
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

```bash
# 安装
python -m pip install -e .

# 全量双赛道实验（约数分钟；生成报告与图）
python run_experiment.py

# 只做快速冒烟
python run_experiment.py --steps 2 --code-steps 2

# 测试
python -m pytest -q
python tests/test_v5.py
```

**关键文件：**

- `tricore/model.py`：`@torch.jit.script tricore_fused_scan`（四元组返回）、向量化 5-D 多跳 `TriCoreBlock`、`TriCoreLM`、`BaselineTransformerLM`
- `tricore/tasks.py`：2-Hop MQAR 与因果字节码 `generate_contextual_code_lm_batch`
- `run_experiment.py`：Track 1 五组消融、Track 2 两模型、V6 计时、cache 核算、Markdown 报告、matplotlib 2×2 图
- `tests/test_v5.py`：JIT 与 Python 参考在四种 `enable_surprise_gate` × `enable_fast_weight` 组合下对 `h_seq`、`r1_seq`、`F_seq`、`surprise_trace` 的 `allclose` 等价性；`TriCoreLM` 前向/反向；因果字节码的形状、`pos + 1` mask、反向梯度检查

---

## 7. 结论：V6 到底证明了什么

### 已证明（在所述合成任务与预算内）

1. **Full TriCore 能完成 2-Hop 链式联想召回**，并在 4× 对抗噪声间隔外推下保持 100.00%。
2. **Engine II 和 Engine III 是该任务的关键**：去掉快权重或去掉隐空间重查询都会跌到随机水平附近。
3. **Engine I 在本次 V6 运行中是必要的**：去掉后跌到 5.55%/4.14%，与 V5 的 91.88%/84.14% 不同；这提醒我们单 seed 消融结论的脆弱性。
4. **V6 把多跳重查询从逐 token 循环改成跨全部位置的一次批量 5-D `matmul`**，保持公式等价；状态递归仍是串行，不能把它描述成并行前缀扫描或 kernel 级优化。
5. **O(1) 状态核算是真实的**：17.00 KiB 与序列长度无关，32K 时相对 2 层 KV cache 为 3,855.1×。
6. **第二赛道管线可运行**：TriCore 在因果字节码任务上 PPL 3.17、解引用 100.00%；2 层 Transformer 为 3.56 / 19.63%，接近 16.67% 随机水平。这证明封闭语法上的两级指针查找可学习，不证明真实代码理解。

### 未证明 / 不应宣称

- 自然语言、真实代码、推理或通用序列建模能力。
- 对调参后 Transformer 的全面优势；Track 1 的 Transformer 未调参、未匹配参数/算力。
- 多 seed 稳健性、统计显著性、置信区间。
- 生产吞吐、长上下文实测解码、分布式/混合精度/量化能力。
- 训练显存的 O(1) 性质：V6 会物化 `F_seq`，训练激活随 `L` 增长；只有持久推理状态是 O(1)。
- “Engine I 在所有任务上都必要”或“三个引擎缺一不可”的普适结论；本次 V6 No-I 跌到随机，但 V5 没有，说明结论对优化轨迹和 seed 敏感。
- JIT 已接近硬件最优；它仍是编译后的 PyTorch 循环，状态递归仍串行。

---

## 8. 建议的下一步

1. **多 seed**：至少 3 个 seed，报告 mean ± std，尤其是 No-Engine-I 的方差。
2. **更强基线**：调参 Transformer，匹配非位置参数量、训练 token 与算力。
3. **更长/更难 Code LM**：更多变量、更长训练、held-out 模板；当前封闭语法上的 100.00% 不足以支撑真实代码能力结论。
4. **3–5 hop**：验证 latent re-query 是否随跳数扩展，以及 router 是否学会使用额外循环。
5. **真正 kernel**：在 JIT + 向量化多跳的基础上做 fused/chunked CUDA/Triton scan；V6 已达到 1.57×（对 V5）/1.70×（对 V4），下一步是减少小 kernel launch 与串行状态更新成本。
6. **长上下文实测**：用真实 KV cache 解码对比 32K–128K 的延迟与显存，而不只做公式核算。

---

## 9. 交付状态

| 项目 | 状态 |
| :--- | :---: |
| V6 模型代码（JIT 四元组 + 向量化多跳 + LM + baseline） | ✅ 完成 |
| Track 1 五组消融 | ✅ 完成并写入报告 |
| Track 2 因果字节码 Contextual Code LM | ✅ 完成并写入报告 |
| V6 / V5 / V4 耗时对比 | ✅ 80.0 s vs 125.3 s vs 136.0 s（1.57× / 1.70×） |
| 推理状态核算 | ✅ 17.00 KiB；32K 时 3,855.1× |
| README / SPEC 同步 | ✅ 已更新为 V6 数据 |
| LICENSE / .gitignore / pyproject | ✅ 已创建 |
| 测试文件 | ✅ `tests/test_v5.py` 已创建；命令见上 |
| 结果报告与图 | ✅ 已生成 |
| 本汇报文件 | ✅ 已生成 |

> 一句话总结：**V6 把 Engine III 的多跳重查询向量化到全部时间步，Full 训练从 V5 的 125.3 s 降到 80.0 s（1.57×，对 V4 为 1.70×），并用因果字节码把 Track 2 的解引用目标移到返回别名 `pos + 1`，在该封闭语法上达到 100.00%；但状态递归仍串行，Track 2 不是真实代码，Track 1 的 Engine I 消融也再次显示单 seed 结论并不稳固。下一阶段的重点是多种子、强基线、held-out 模板和真正的 kernel。**

---

<p align="center">
  <i>TriCore V6 — constant state, surprise-gated writes, causal delta memory, and a vectorized latent loop.</i>
</p>