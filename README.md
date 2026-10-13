# CausalDualAD

> **Causal-invariant dual-path model for time series anomaly detection.**

基于深度学习的时间序列异常检测系统，融合**时序重构**与**多尺度原型匹配**双路径，通过不确定性估计自适应融合异常分数。

**本科毕设项目** · 指导老师：郭贤杰（南京邮电大学计算机学院）

---

## 文档

| 文件 | 内容 |
|---|---|
| [docs/目标体系与路线图.md](docs/目标体系与路线图.md) | **目标设定** —— 三个阶段、判定标准、时间线、风险登记 |
| [docs/项目现状报告.md](docs/项目现状报告.md) | 现状、实验结果、架构决策、待办清单 |
| [docs/工程发现记录.md](docs/工程发现记录.md) | ⚠️ **写论文前必读** —— 代码缺陷、数据陷阱、方法发现 |
| [docs/TSAD研究现状调研_2026-10.md](docs/TSAD研究现状调研_2026-10.md) | 领域研究现状 + 20 篇核心论文 + 可行性判断 |
| [docs/DATASET.md](docs/DATASET.md) | 数据集目录结构与获取方式 |
| [docs/thesis/](docs/thesis/) | 论文草稿（markdown + LaTeX） |

---

## 环境

```bash
conda activate MDAD          # Python 3.11 + torch 2.9.0 (CPU)
```

本机无独显，完整实验建议使用 GPU 环境（学校服务器或 Colab）。

---

## 快速开始

```bash
# 冒烟测试（合成数据，无需真实数据）
python -m src.ucr_pipeline.run_ucr --smoke-test    # 单变量
python -m src.ucr_pipeline.run_mv  --smoke-test    # 多变量

# 单变量（UCR）
python -m src.ucr_pipeline.run_ucr --ucr-path ./data --dataset-ids 1 2 3 4 5

# 多变量（SMAP / MSL / SMD）
python -m src.ucr_pipeline.run_mv --data-root ./data --datasets SMAP MSL SMD \
    --epochs 10 --save-dir results/mv

# 调试（CPU 友好）
python -m src.ucr_pipeline.run_mv --datasets SMAP \
    --max-points 8000 --start-offset 3000 --window-size 64 --batch-size 128 --epochs 5

# 消融：窗口池化对比（验证 max 池化的必要性）
python -m src.ucr_pipeline.run_mv --datasets SMAP --pool mean --assign-mode center  # 旧行为
python -m src.ucr_pipeline.run_mv --datasets SMAP --pool max  --assign-mode full   # 新默认
```

---

## 模型结构

```
输入 (B, W, C)
  ↓
时频双域预处理：FFT 沿时间维 → (B, freq_bins, C) → 取前 k 个频率分量
                → 展平 (B, k*C) → MLP → (B, W, 64)
  ↓ 拼接 → (B, W, C+64)
多尺度卷积（kernel 3/5/7）→ (B, W, hidden=128)
  ↓
Transformer 编码器（4 头 3 层）
  ├→ 重构路径：通道独立、权重共享 → (B, W, C) → max 聚合 → (B, W)
  ├→ 原型路径：作用于重构误差 → (B, W) → max 聚合
  └→ 不确定性估计 → 自适应权重
       ↓
   融合异常分数 (B, W)
```

**max 聚合语义**：任一通道异常 → 该时间点异常。

**三个正交配置维度**（消融实验必须正交，详见 [工程发现记录](docs/工程发现记录.md)）：

| 维度 | 参数 | 取值 | 默认 |
|---|---|---|---|
| 通道聚合 | `--aggregate` | max / mean / topk_mean | max |
| 窗口池化 | `--pool` | max / mean / topk_mean | **max** |
| 分数分配 | `--assign-mode` | full / center | full |

> ⚠️ 2026-10-13 修复：此前 `pool` 固定为 `mean`，导致稀疏异常被稀释
> （64 点窗口里 5 个异常点 → 分数稀释到 0.078 倍），SMAP AUROC 因此从 0.80 掉到 0.48。
> **池化必须是 max，否则 VUS-PR 与跨数据集比较都不可信。**

---

## 项目结构

```
CausalDualAD/
├── src/ucr_pipeline/          # 唯一活跃代码
│   ├── run_ucr.py             #   单变量入口（UCR）
│   ├── run_mv.py              #   多变量入口（SMAP/MSL/SMD）
│   ├── model.py               #   双路径模型（支持任意通道数）
│   ├── trainer.py             #   训练器（统一设备管理 + 批量推理）
│   ├── data.py                #   数据加载（通道化预处理 + 多变量加载器）
│   ├── evaluation.py          #   评估指标（AUROC + VUS-ROC/PR + Top-k + F1）
│   ├── config.py              #   配置管理
│   ├── prepare_data.py        #   数据下载/转换/验证
│   ├── ablation.py            #   消融实验框架（7 种配置）
│   └── baseline_simad.py      #   SimAD 基线对比
├── docs/                      # 文档（现状报告、调研、数据集说明、论文草稿）
├── data/                      # 数据集本地目录（gitignore，不入库）
├── results/                   # 实验结果（指标 JSON 入库，权重不入库）
├── _archive/                  # 旧代码归档，仅本地保留、不入库
└── requirements.txt
```

---

## 当前状态

**毕设进行中的工作版本，实验数据正在复现与校准中，暂不可作为结论引用。**

| 模块 | 状态 |
|---|---|
| 代码重构 | ✅ 单一入口、FFT/cdist 向量化、设备统一、通道化 |
| 评估指标 | ✅ 标准 Top-k + AUROC + VUS-ROC/PR + oracle-threshold F1 |
| 消融实验框架 | ✅ 7 种配置 |
| 基线对比框架 | ✅ SimAD 对比脚本 |
| 多变量管线 | ✅ SMAP 25 通道 / MSL 55 通道 / SMD 38 通道可跑 |
| 跨数据集实测 | 🔄 进行中 |
| 强基线复现 | ⏳ 待做 |
| 因果不变性改进点 | ⏳ 待做（毕设核心） |

详见 [项目现状报告](docs/项目现状报告.md)。

---

## 核心修复记录

| 问题 | 旧代码 | 修复后 |
|---|---|---|
| FFT 计算 | 双重 Python for 循环 | `torch.fft.rfft` 向量化 |
| 原型匹配 | 逐时间步 `pairwise_distance` | `torch.cdist` 一次性算完 |
| 设备管理 | forward 内强制 CPU | 统一 device 传递 |
| 入口混乱 | 7 个 main、6 个 model 变体 | 单一入口 |
| import 路径 | flat import 依赖 CWD | 包内相对导入 |
| 硬编码权重 | UCR_5 专用 0.2/0.8 | 参数化，默认自适应 |
| 数据路径 | 硬编码绝对路径 | 命令行参数 |
| 评估指标 | Top-k 宽容 + F1 用标签作弊 | 标准 Top-k + oracle threshold + AUROC |
| 滑窗创建 | for 循环逐窗构造 | `stride_tricks` 零拷贝 |
| 变量维度 | 全链路写死单变量 | 通道化，`n_features` 参数化 |

---

## 评估指标

| 指标 | 说明 |
|---|---|
| AUROC | 排序能力（0.5=随机，<0.5=比随机差） |
| VUS-ROC / VUS-PR | 体积 ROC/PR，**社区事实标准** |
| Top-1/3/5 | 最高分点是否落在异常区间 |
| F1 / P / R | oracle threshold 搜索（阈值用标签选取，**仅用于模型间对比，不作为绝对性能上报**） |

---

## 依赖

- Python 3.8+
- PyTorch 2.0+
- NumPy、SciPy、scikit-learn