# 数据集说明

真实数据集**不入版本库**（体积过大，且多为第三方许可）。本目录只保留目录结构，
代码通过 `--ucr-path` / `data/` 下的相对路径读取。

## 需要的目录结构

```
data/
├── MSL/       MSL_train.npy, MSL_test.npy, MSL_test_label.npy
├── SMAP/      SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
├── SMD/       SMD_train.npy, SMD_test.npy, SMD_test_label.npy
├── PSM/       PSM_train.csv, PSM_test.csv, PSM_test_label.csv
├── SWAT/      swat_train2.csv, swat2.csv
└── UCR/       UCR_1_train.npy, UCR_1_test.npy, UCR_1_test_label.npy, ...
```

## 各数据集来源

| 数据集 | 变量数 | 来源 |
|---|---|---|
| UCR Anomaly Archive | 1 | https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ |
| SMAP / MSL | 55 | NASA SMAP/MSL 遥测数据（Hundman et al., KDD'18） |
| SMD | 25 | 来自 OmniAnomaly（Su et al., KDD'19） |
| PSM | 25 | https://github.com/eBay/RANSynCoders |
| SWaT | 51 | iTrust/SWaT 水处理测试床 |

## 生成合成数据（无需真实数据，用于冒烟测试）

```bash
python -m src.ucr_pipeline.prepare_data --action synthetic --target-dir ./data/UCR_synthetic
```

## UCR 原始数据转换

```bash
# 从 UCR 原始目录转换
python -m src.ucr_pipeline.prepare_data --action convert \
    --source-dir /path/to/UCR_AnomalyArchive --target-dir ./data/UCR

# 校验已有数据
python -m src.ucr_pipeline.prepare_data --action verify --target-dir ./data/UCR
```

## 当前本机已有

`data/` 下 MSL / SMAP / SMD / PSM / SWAT 已就位；**UCR 真实数据尚未放入**，
只有 `data/UCR_synthetic/`（合成数据，仅供冒烟测试，不可用于论文结果）。