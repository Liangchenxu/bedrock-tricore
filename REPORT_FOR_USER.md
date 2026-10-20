# TriCore V5 完成报告

**日期：** 2026-10-20  
**版本：** `tricore-arch` 0.5.0  
**状态：** V5 双赛道实验完成；TorchScript-JIT 融合扫描已接入；MIT 开源套件就绪  
**实测报告：** `results/BENCHMARK_REPORT.md`  
**图表：** `results/tricore_vs_transformer.png`  
**运行设备：** CUDA

---

## 0. 一页结论

V5 在不改变 V4 数学公式、参数量和 O(1) 状态结构的前提下，完成了三件事：

1. **运行时：** 将逐 token 的 Python 循环替换为 `@torch.jit.script tricore_fused_scan`。Full 训练 500 步从 V4 的 **136.0 s** 降到 **125.3 s**，加速 **1.08×**（不含 0.5 s 首次编译/预热）。
2. **第二赛道：** 新增 Contextual Code LM（变量 / 别名 / 字面量 / 返回解引用），TriCore 1 层验证 **PPL 1.81、解引用准确率 21.09%**；2 层 Transformer 为 **PPL 1.96、18.75%**。6 类随机基线为 **16.67%**。
3. **开源交付：** 根目录已有 `LICENSE`（MIT）、`.gitignore`、`pyproject.toml`；README 与 SPEC 已同步为 V5 实测数据；实验报告和 2×2 图表已重新生成。

**必须同时看到的限制：** 两个赛道都是单 seed 合成任务。Track 1 的 Full 达到 100%/100%，但 No-Engine-I 本次为 91.88%/84.14%，没有复现 V4 的 63.44%/51.41%；这是单 seed 优化/评测批次差异，不是公式变化。Track 2 的 21.09% 只比随机高 4.42 个百分点，不能解读为“理解了代码”或“通用语言建模已被证明”。

---

## 1. JIT 加速前后：训练耗时对比

| 测量项 | 耗时 | 说明 |
| :--- | :---: | :--- |
| V4 Full，Python 逐 token 循环 | 136.0 s | 历史参考运行 |
| V5 Full，TorchScript 融合扫描 | **125.3 s** | 计时 500 步 |
| V5 / V4 速度比 | **1.08×** | >1 表示更快，<1 表示更慢 |
| V5 首次调用预热 | 0.5 s | 不计入正式计时 |

计时范围包括数据生成、前向、反向和优化器步。V4 的 136.0 s 是此前记录的 Python 循环 Full 运行，不是同机同时测量；因此 1.08× 是单机单 seed 的墙钟对比，不是吞吐保证。

**结论：** JIT 确实减少了 Python 逐 token 调度开销，但提升是温和的。它仍然是 PyTorch 张量算子的编译循环，不是自定义 CUDA kernel；小矩阵乘法和 kernel launch 仍是主要成本。相较 2 层 Transformer（15.9 s），TriCore 仍慢得多，因为 TriCore 是顺序递归。

---

## 2. Track 1：2-Hop 链式联想召回（五组消融）

**协议：** 500 步，batch 64，`lr=3e-3`，seed 42；6 条链、4 个查询、16 个训练噪声 token；固定评测批次 seed 2026/2027；4× 外推把对抗噪声间隔从 16 拉到 64（总序列 108），不是把总长度乘 4。

| 架构变体 | 参数量 | 最终验证 loss | 域内 2-hop 召回 | 4× 噪声间隔外推 | 训练耗时 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full（三引擎全开）** | 207,503 | **0.0010** | **100.00%** | **100.00%** | 125.3 s |
| TriCore No Engine I：去掉惊奇门控 | 207,503 | 0.4066 | 91.88% | 84.14% | 97.9 s |
| TriCore No Engine II：去掉快权重 | 207,503 | 3.0008 | 5.78% | 4.14% | 29.2 s |
| TriCore No Engine III：去掉自适应循环 | 207,503 | 1.8660 | 16.48% | 16.56% | 59.7 s |
| Vanilla Transformer（2 层） | 543,744 | 2.3131 | 16.72% | 15.23% | 15.9 s |

### Track 1 解读

1. **Full 解决任务。** 域内 100.00%，4× 间隔仍 100.00%，验证 loss 0.0010。
2. **Engine II 是联想记忆本体。** 去掉快权重后 5.78%/4.14%，低于 6 选 1 的 16.67% 随机水平。
3. **Engine III 是第二跳。** 去掉隐空间重查询后 16.48%/16.56%，基本等于随机；第一跳还能取到桥接 token，但无法把它转成最终值。
4. **Engine I 本次仍有贡献，但不是“缺了就崩”。** 去掉写保护后 91.88%/84.14%，相对 Full 下降 8.12/15.86 个百分点。V4 记录的 63.44%/51.41% 没有复现；相同公式下单 seed 优化轨迹和固定评测批次会带来差异，不能把两次数字当成同一实验的直接复现。
5. **Transformer 在本预算下是随机水平。** 16.72%/15.23%；它未调参、未做参数/算力匹配，因此这不是“注意力上限”的证据。
6. **JIT 只解决 Python 开销。** Full 125.3 s 对 V4 136.0 s，1.08×；Transformer 仍快得多。

---

## 3. Track 2：Contextual Code LM（语言建模 + 解引用）

**协议：** 300 步，batch 32，`lr=3e-3`，seed 42；6 个变量、6 条 NOP、序列长度 82；固定评测批次 seed 3030；dense next-token 交叉熵；解引用准确率只在 `[RET, alias_i, lit_i, SEMI]` 的 alias 位置计算。

| 模型 | 参数量 | 验证 loss | 验证 PPL | 解引用准确率 | 训练耗时 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TriCore Full（1 层）** | 207,503 | 0.5914 | **1.81** | **21.09%** | 83.4 s |
| Vanilla Transformer（2 层） | 543,744 | 0.6753 | 1.96 | 18.75% | 2.9 s |

**任务格式（语句随机排序）：**

```text
[LET,   var_i,   EQ, lit_i,  SEMI]
[ALIAS, alias_i, EQ, var_i,  SEMI]
[NOP,   NOISE,   SEMI]  × 6
...
[RET,   alias_i, lit_i,  SEMI]
```

变量、别名、字面量三个命名空间互不重叠。返回使用别名，所以模型必须在 alias 位置预测 `lit_i`，即完成 `alias_i -> var_i -> lit_i` 的两级解引用，而不是复制表面 token。6 个等概率字面量，随机基线 **16.67%**。

### Track 2 解读

- TriCore 的 PPL 更低（1.81 对 1.96），解引用准确率高 2.34 个百分点（21.09% 对 18.75%）。
- 但 21.09% 只比随机高 **4.42 个百分点**，Transformer 也只高 2.08 个百分点；两者都远未解决该任务。
- 因此正确结论是：**管线跑通，TriCore 出现非退化信号并有小幅优势；不足以宣称通用代码理解、组合泛化或全面优于 Transformer。**
- 任务本身是合成语法，不是人类代码；300 步、6 变量、单 seed 的预算也很小。

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
| `pyproject.toml` | ✅ 已创建 | `tricore-arch` 0.5.0，Python ≥3.10，torch/numpy/matplotlib，MIT |
| `README.md` | ✅ 已更新 | V5 标题、JIT、双赛道实测表、限制、复现命令、12 条不变量 |
| `SPEC_V2.md` | ✅ 已更新 | V5 JIT 合约、公式、状态核算、任务令牌布局、协议、实测记录 |
| `requirements.txt` | ✅ 已存在 | `torch>=2.0`、`numpy>=1.24`、`matplotlib>=3.7` |
| `tricore/__init__.py` | ✅ 已更新 | 导出 `TriCoreBlock`、`TriCoreLM`、`BaselineTransformerLM` 与两个任务生成器 |
| `tricore/model.py` | ✅ 已更新 | `tricore_fused_scan`、双模型 |
| `tricore/tasks.py` | ✅ 已更新 | 2-Hop MQAR + Contextual Code LM |
| `tests/test_sanity.py` | ✅ 已存在 | 前向/反向/形状/梯度流 |
| `tests/test_v5.py` | ✅ 已创建 | JIT 对 Python 参考的五组 flag 等价性 + Code LM 形状/反向 |
| `run_experiment.py` | ✅ 已更新 | 双赛道、五组消融、grad clipping、报告与 2×2 图 |
| `results/BENCHMARK_REPORT.md` | ✅ 已生成 | 本次 V5 实测报告 |
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

- `tricore/model.py`：`@torch.jit.script tricore_fused_scan`、`TriCoreBlock`、`TriCoreLM`、`BaselineTransformerLM`
- `tricore/tasks.py`：`generate_associative_recall_batch`、`generate_contextual_code_lm_batch`
- `run_experiment.py`：Track 1 五组消融、Track 2 两模型、JIT 计时、cache 核算、Markdown 报告、matplotlib 2×2 图
- `tests/test_v5.py`：JIT 与 Python 参考在 Full / No-I / No-II / No-III / all-off 五组配置下的 `allclose` 等价性；Code LM 的形状、mask、反向梯度检查

---

## 7. 结论：V5 到底证明了什么

### 已证明（在所述合成任务与预算内）

1. **Full TriCore 能完成 2-Hop 链式联想召回**，并在 4× 对抗噪声间隔外推下保持 100.00%。
2. **Engine II 和 Engine III 是该任务的关键**：去掉快权重或去掉隐空间重查询都会跌到随机水平附近。
3. **Engine I 有贡献但不总是决定性**：本次去掉后仍有 91.88%/84.14%，与 V4 记录的 63.44%/51.41% 不同；这提醒我们单 seed 消融结论的脆弱性。
4. **JIT 融合扫描保持公式等价并带来 1.08× 的 Full 训练加速**；它不是 kernel 级优化，不能期待数量级提升。
5. **O(1) 状态核算是真实的**：17.00 KiB 与序列长度无关，32K 时相对 2 层 KV cache 为 3,855.1×。
6. **第二赛道管线可运行**：TriCore 在合成 Code LM 上 PPL 1.81、解引用 21.09%，略优于 2 层 Transformer 的 1.96 / 18.75%，但两者都接近 16.67% 随机水平。

### 未证明 / 不应宣称

- 自然语言、真实代码、推理或通用序列建模能力。
- 对调参后 Transformer 的全面优势；Track 1 的 Transformer 未调参、未匹配参数/算力。
- 多 seed 稳健性、统计显著性、置信区间。
- 生产吞吐、长上下文实测解码、分布式/混合精度/量化能力。
- “Engine I 在所有任务上都必要”或“三个引擎缺一不可”的普适结论；本次 No-I 没有崩，V3/V4 的任务依赖性也说明结论是条件性的。
- JIT 已接近硬件最优；它仍是编译后的 PyTorch 循环。

---

## 8. 建议的下一步

1. **多 seed**：至少 3 个 seed，报告 mean ± std，尤其是 No-Engine-I 的方差。
2. **更强基线**：调参 Transformer，匹配非位置参数量、训练 token 与算力。
3. **更长/更难 Code LM**：更多变量、更长训练、held-out 模板；当前 21.09% 不足以支撑能力结论。
4. **3–5 hop**：验证 latent re-query 是否随跳数扩展，以及 router 是否学会使用额外循环。
5. **真正 kernel**：在 JIT 基础上做 fused/chunked CUDA/Triton scan，目标是把 1.08× 提升到有意义的量级。
6. **长上下文实测**：用真实 KV cache 解码对比 32K–128K 的延迟与显存，而不只做公式核算。

---

## 9. 交付状态

| 项目 | 状态 |
| :--- | :---: |
| V5 模型代码（JIT scan + LM + baseline） | ✅ 完成 |
| Track 1 五组消融 | ✅ 完成并写入报告 |
| Track 2 Contextual Code LM | ✅ 完成并写入报告 |
| JIT 前后耗时对比 | ✅ 125.3 s vs 136.0 s（1.08×） |
| 推理状态核算 | ✅ 17.00 KiB；32K 时 3,855.1× |
| README / SPEC 同步 | ✅ 已更新为 V5 数据 |
| LICENSE / .gitignore / pyproject | ✅ 已创建 |
| 测试文件 | ✅ `tests/test_v5.py` 已创建；命令见上 |
| 结果报告与图 | ✅ 已生成 |
| 本汇报文件 | ✅ 已生成 |

> 一句话总结：**V5 把 TriCore 的递归循环编译进了 TorchScript，保持了 O(1) 状态和 2-Hop 能力，并补上了第二赛道与开源交付；但 Track 2 仍只是略高于随机，Track 1 的 Engine I 消融也显示单 seed 结论并不稳固。下一阶段的重点是多种子、强基线、更难代码任务和真正的 kernel。**

---

<p align="center">
  <i>TriCore V5 — constant state, surprise-gated writes, causal delta memory, and a compiled latent loop.</i>
</p>