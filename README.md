# CausalDualAD — 双路径时序异常检测（本科毕设）

> **Causal-invariant dual-path model for time series anomaly detection.**

基于深度学习的时间序列异常检测系统，结合**时序重构**与**多尺度原型匹配**双路径，通过不确定性估计自适应融合异常分数。

---

## 当前版本现状（2026-10-13）

**毕设进行中的工作版本，实验数据正在复现与校准中，暂不可作为结论引用。**

### 已完成

| 模块 | 状态 |
|---|---|
| 代码重构 | 单一入口 `run_ucr.py`，FFT / 原型匹配已向量化，设备统一管理 |
| 评估指标 | 标准 Top-k + AUROC + VUS-ROC/PR + oracle-threshold F1 |
| 消融实验框架 | 7 种配置（`ablation.py`） |
| 基线对比框架 | SimAD 对比脚本（`baseline_simad.py`） |
| 项目结构整理 | 旧代码移入 `_archive/`（不入库），数据集不再入库 |

### 进行中

| 事项 | 说明 |
|---|---|
| 多变量管线 | ✅ 已适配（`run_mv.py`），SMAP 25 通道 / MSL 55 通道 / SMD 38 通道可直接跑；SWaT / WADI 需先转 `.npy` |
| 跨数据集实测 | 🔄 SMAP / MSL / SMD / SWaT / WADI 的跨数据集性能差异待复现确认 |
| 论文文档口径同步 | `docs/thesis/experiments.md` 中基于 oracle threshold 的 P/R 与"优化前后对比"式表述需改为同 split 下与公开方法对比 |
| **因果不变性改进点** | **毕设核心待做内容**：用因果不变学习改善跨数据集泛化性能，尚未动工 |
| 强基线复现 | 待做。需在同一 split、同一指标下复现 MEMTO / DCdetector / ScatterAD |

### 环境

```bash
conda activate MDAD          # Python 3.11 + torch 2.9.0 (CPU)
```

```bash
# 单变量（UCR）
python -m src.ucr_pipeline.run_ucr --ucr-path ./data --dataset-ids 1 2 3 4 5

# 多变量（SMAP / MSL / SMD）
python -m src.ucr_pipeline.run_mv --data-root ./data --datasets SMAP MSL SMD \
    --epochs 10 --save-dir results/mv

# 调试：合成数据冒烟测试
python -m src.ucr_pipeline.run_mv --smoke-test

# 调试：只跑前N个时间点
python -m src.ucr_pipeline.run_mv --datasets SMAP --max-points 30000 --epochs 10
```

### 模型结构（多变量）

```
输入 (B, W, C)
  ↓
时频双域预处理: FFT 沿时间维 → (B, freq_bins, C) → 取前k 个频率 → MLP → (B, W, 64)
  ↓ 拼接 → (B, W, C+64)
多尺度卷积 (3/5/7)→ (B, W, hidden)
  ↓
Transformer 编码器
  ├→ 重构路径: 通道独立、权重共享 → (B, W, C) → max 聚合 → (B, W)
  ├→ 原型路径: 作用于重构误差 → (B, W) → max 聚合
  └→ 不确定性估计 → 自适应权重
       ↓
   融合异常分数 (B, W)
```

**max 聚合语义**：任一通道异常 → 该时间点异常。消融实验应补mean 聚合对比。

### 下一步

1. 完成 SMAP / MSL / SMD / SWaT / WADI 的跨数据集实测，确认性能差异是否真实存在
2. 复现强基线（MEMTO / DCdetector / ScatterAD），确认基线可复现
3. 统一论文文档的实验口径
4. 引入因果不变性模块，作为毕设核心改进点

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 烟雾测试（合成数据，无需真实数据，验证代码可运行）
python -m src.ucr_pipeline.run_ucr --smoke-test

# 正式运行（需要 UCR 数据）
python -m src.ucr_pipeline.run_ucr --ucr-path /path/to/UCR --dataset-ids 1 2 3 4 5 --device cuda:0
```

## 项目结构

```
CausalDualAD/
├── src/ucr_pipeline/          # ★ 唯一活跃代码
│   ├── run_ucr.py             #   单变量入口（UCR）：训练 → 检测 → 评估
│   ├── run_mv.py              #   多变量入口（SMAP/MSL/SMD）
│   ├── model.py               #   双路径模型（支持任意通道数）
│   ├── trainer.py             #   训练器（统一 GPU 设备管理 + 批量推理）
│   ├── data.py                #   数据加载（通道化预处理 + 多变量加载器）
│   ├── evaluation.py          #   评估指标（AUROC + F1 + VUS-ROC/PR + Top-k）
│   ├── config.py              #   配置管理
│   ├── prepare_data.py        #   UCR 数据下载/转换/验证工具
│   ├── ablation.py            #   消融实验框架（7 种配置）
│   └── baseline_simad.py      #   SimAD 基线对比框架
├── docs/
│   ├── DATASET.md             # 数据集目录结构与获取方式（数据不入库）
│   ├── thesis/                # 论文文档（markdown 草稿 + LaTeX 草稿）
│   └── figures/               # 实验可视化图
├── data/                      # 数据集本地目录（.npy/.csv 已 gitignore）
│   ├── UCR_*.npy              #   UCR 250 个子集
│   ├── SMAP/ MSL/ SMD/        #   多变量（npy，已可用）
│   └── PSM/ SWAT/ WaDi/       #   待转换（csv / 格式待适配）
├── results/                   # 实验结果（入库，保证可追溯）
├── _archive/                  # 旧代码归档，仅本地保留、不入库（见其 README）
└── requirements.txt
```

> **数据不入库**：`data/` 下的 `.npy` / `.csv` 已 gitignore，目录结构与获取方式见
> [docs/DATASET.md](docs/DATASET.md)。当前本机 UCR 真实数据尚未放入，
> 只有 `data/UCR_synthetic/`（合成数据，仅供冒烟测试，**不可用于论文结果**）。

## 核心修复（相对旧代码）

| 问题 | 旧代码 | 修复后 |
|------|--------|--------|
| FFT 计算 | 双重 Python for 循环 + `.cpu().numpy()` | `torch.fft.rfft` 一行向量化 |
| 原型匹配 | `seq_len` 次 for 循环 `pairwise_distance` | `torch.cdist` 一次性算完 |
| 设备管理 | forward 内强制 CPU，训练器硬编码 `batch.to('cpu')` | 统一 device 传递，全程不离开 GPU |
| 入口混乱 | 7 个 main、3 个 solution、6 个 model 变体 | 单一 `run_ucr.py` 入口 |
| import 路径 | 4 个文件 flat import 依赖 CWD | 包内相对导入 |
| 硬编码权重 | UCR_5 专用 0.2/0.8 固定权重 | `use_fixed_weights` 参数化，默认自适应 |
| 数据路径 | 硬编码 `E:\social_network\TSAD\data\UCR` | `--ucr-path` 命令行参数 |
| 评估指标 | Top-k 5 重宽容 + F1 用真实标签生成预测（作弊） | 标准 Top-k + oracle threshold + AUROC |
| 滑窗创建 | for 循环逐窗构造 | `np.lib.stride_tricks` 零拷贝 |
| 只支持单变量 | `in_channels=1`、`output_dim=1`、`squeeze()`、`[:, :, newaxis]` 写死 | 全链路通道化，`n_features` 参数化，支持 SMAP/MSL/SMD 等多变量基准 |
| 频域特征通道平均 | FFT 后对通道维求均值，多变量信息丢失 | FFT 保留通道维 → `(B, k, C)` 展平 → MLP，与通道数解耦 |

## 已知问题

见上文「当前版本现状」。核心两项：跨数据集性能差异待实测确认；SWaT / WADI 需先转格式。
评估管线已通过单变量与多变量双路径冒烟测试（AUROC 方向正确）。

## 使用方法

### 1. 训练 + 检测 + 评估（完整流程）

```bash
# 单个数据集
python -m src.ucr_pipeline.run_ucr --ucr-path /data/UCR --dataset-ids 1

# 批量运行
python -m src.ucr_pipeline.run_ucr --ucr-path /data/UCR --dataset-ids 1 2 3 4 5

# 自定义参数
python -m src.ucr_pipeline.run_ucr --ucr-path /data/UCR --dataset-ids 1 --epochs 100 --batch-size 64 --window-size 128

# 指定 GPU
python -m src.ucr_pipeline.run_ucr --ucr-path /data/UCR --dataset-ids 1 --device cuda:0

# 保存模型和结果
python -m src.ucr_pipeline.run_ucr --ucr-path /data/UCR --dataset-ids 1 --save-dir results/ucr
```

### 2. 消融实验

```bash
# 烟雾测试
python -m src.ucr_pipeline.ablation --smoke-test

# 正式运行（7 种配置：full / freq_only / time_only / no_prototype / no_fft / fixed_fusion / no_contrast）
python -m src.ucr_pipeline.ablation --ucr-path /data/UCR --dataset-ids 1 2 3 4 5 --device cuda:0 --epochs 50
```

### 3. SimAD 基线对比

```bash
# 烟雾测试
python -m src.ucr_pipeline.baseline_simad --smoke-test

# 正式运行
python -m src.ucr_pipeline.baseline_simad --ucr-path /data/UCR --dataset-ids 1 2 3 --device cuda:0 --epochs 50
```

### 4. 数据准备

```bash
# 生成合成数据（用于测试）
python -m src.ucr_pipeline.prepare_data --action synthetic --target-dir ./data/UCR_synthetic

# 转换原始 UCR Archive 数据
python -m src.ucr_pipeline.prepare_data --action convert --source-dir /path/to/UCR_AnomalyArchive --target-dir ./data/UCR

# 验证已有数据
python -m src.ucr_pipeline.prepare_data --action verify --target-dir ./data/UCR
```

## 模型架构

```
输入窗口 → 时频预处理(FFT) → 多尺度卷积 → Transformer编码器
                                                        ├→ 时序重构路径 → 重构误差
                                                        ├→ 原型匹配路径 → 原型距离
                                                        └→ 不确定性估计 → 自适应权重
                                                                    ↓
                                                              融合异常分数
```

## 评估指标

| 指标 | 说明 |
|------|------|
| AUROC | 排序能力（0.5=随机，<0.5=比随机差） |
| VUS-ROC / VUS-PR | 体积 ROC/PR |
| Top-1/3/5 | 最高分点是否落在异常区间 |
| F1 / Precision / Recall | oracle threshold 搜索（阈值用标签选取，**仅用于模型间对比，不作为绝对性能上报**） |

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn

## 快速验证

```bash
# 冒烟测试（合成数据，无需真实数据）
python -m src.ucr_pipeline.run_ucr --smoke-test

# 单数据集完整流程
python -m src.ucr_pipeline.run_ucr --ucr-path ./data/UCR --dataset-ids 1 --device cuda:0
```

