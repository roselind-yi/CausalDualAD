#!/usr/bin/env python3
"""
data.py - UCR 数据加载与预处理（修复版）

关键修复：
1. 路径参数化：不再硬编码 UCR 数据路径
2. 标准化修复：用训练集统计量标准化测试集（防止数据泄漏）
3. 支持 UCR 1-250 批量加载
"""

import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_ucr_dataset(dataset_id, ucr_path):
    """
    加载单个 UCR 数据集

    参数:
        dataset_id: 数据集 ID (1-250)
        ucr_path: UCR 数据根目录
    返回:
        (train_data, test_data, test_label) 或 (None, None, None)
    """
    train_file = os.path.join(ucr_path, f"UCR_{dataset_id}_train.npy")
    test_file = os.path.join(ucr_path, f"UCR_{dataset_id}_test.npy")
    label_file = os.path.join(ucr_path, f"UCR_{dataset_id}_test_label.npy")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"[数据] UCR_{dataset_id} 文件不存在 (查找路径: {ucr_path})")
        return None, None, None

    train_data = np.load(train_file)
    test_data = np.load(test_file)
    test_label = np.load(label_file) if os.path.exists(label_file) else None

    print(f"[数据] UCR_{dataset_id}: train={train_data.shape}, test={test_data.shape}", end="")
    if test_label is not None:
        print(f", label={test_label.shape} (异常比例={np.mean(test_label):.4f})")
    else:
        print(" (无标签)")

    return train_data, test_data, test_label


def list_available_ucr(ucr_path):
    """列出指定路径下所有可用的 UCR 数据集 ID"""
    pattern = os.path.join(ucr_path, "UCR_*_train.npy")
    files = glob.glob(pattern)
    ids = []
    for f in files:
        try:
            # 从文件名提取 ID: UCR_{id}_train.npy
            name = os.path.basename(f)
            id_str = name.replace("UCR_", "").replace("_train.npy", "")
            ids.append(int(id_str))
        except (ValueError, IndexError):
            continue
    ids.sort()
    return ids


def preprocess_ucr(train_data, test_data):
    """
    数据预处理（支持单变量与多变量）

    关键点：
    1. 单变量 (T,) → 视为 (T, 1)
    2. 多变量 (T, C) → **按通道** 标准化（axis=0），避免跨通道量纲互相污染
    3. 用训练集统计量标准化测试集（防数据泄漏）

    参数:
        train_data: 训练数据 (T,) 或 (T, C)
        test_data: 测试数据 (T,) 或 (T, C)
    返回:
        (train_norm, test_norm, scaler)
    """
    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)

    # 统一成 (T, C)
    if train_data.ndim == 1:
        train_data = train_data[:, np.newaxis]
    if test_data.ndim == 1:
        test_data = test_data[:, np.newaxis]

    # 处理 NaN / Inf
    train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 按通道（axis=0）统计，单变量时等价于全局统计
    mean = train_data.mean(axis=0, keepdims=True)  # (1, C)
    std = train_data.std(axis=0, keepdims=True)    # (1, C)
    std = np.where(std < 1e-8, 1.0, std)

    train_norm = (train_data - mean) / std
    test_norm = (test_data - mean) / std

    # 范围限制
    train_norm = np.clip(train_norm, -5, 5)
    test_norm = np.clip(test_norm, -5, 5)

    scaler = {
        "mean": mean.squeeze().tolist() if mean.shape[1] > 1 else float(mean[0, 0]),
        "std": std.squeeze().tolist() if std.shape[1] > 1 else float(std[0, 0]),
        "n_features": int(train_norm.shape[1]),
    }
    if scaler["n_features"] == 1:
        print(f"[预处理] mean={mean[0,0]:.4f}, std={std[0,0]:.4f}")
    else:
        print(f"[预处理] 多变量 C={scaler['n_features']}, mean/std 按通道计算")
    print(f"[预处理] train_range=[{train_norm.min():.2f}, {train_norm.max():.2f}]")
    print(f"[预处理] test_range=[{test_norm.min():.2f}, {test_norm.max():.2f}]")

    return train_norm, test_norm, scaler


# 多变量数据集的目录名 → 文件名前缀映射
# 实际数据集的文件命名不统一（WADI/creditcard/Water），这里显式声明
_MV_NAME_MAP = {
    "SMAP": "SMAP",
    "MSL": "MSL",
    "SMD": "SMD",
    "WADI": "WADI",              # 目录 WaDi，文件前缀 WADI
    "NIPS_TS_CREDITCARD": "NIPS_TS_creditcard",   # 目录大写，文件小写
    "NIPS_TS_GECCO": "NIPS_TS_Water",            # 目录 GECCO，文件 Water
    "NIPS_TS_SWAN": "NIPS_TS_Swan",
}

# CSV 数据集映射：数据集名 → (train_file, test_file, label_file)
_MV_CSV_MAP = {
    "PSM": ("train.csv", "test.csv", "test_label.csv"),
    "SWAT": ("swat_train2.csv", "swat2.csv", None),  # SWaT 原始数据无独立标签
}


def load_mv_dataset(name, data_root):
    """
    加载多变量数据集

    支持两种格式：
    1. .npy  —— SMAP / MSL / SMD / WADI / NIPS_TS_*
    2. .csv  —— PSM / SWaT（CSV 含表头，第一列为时间戳需剔除）

    参数:
        name: 数据集名（不区分大小写，如 SMAP / WADI / PSM / SWAT / NIPS_TS_GECCO）
        data_root: data/ 根目录
    返回:
        (train_data, test_data, test_label)，形状 (T, C) / (T, C) / (T,)
    """
    key = name.upper()
    data_dir = os.path.join(data_root, key)
    # 目录名可能与请求名不一致（如 WaDi vs WADI），做存在性回退
    if not os.path.isdir(data_dir):
        try:
            candidates = os.listdir(data_root)
        except FileNotFoundError:
            candidates = []
        for cand in candidates:
            # 精确匹配（大小写不敏感）
            if cand.upper() == key:
                data_dir = os.path.join(data_root, cand)
                break
            # 允许缩写：NIPS_TS_GECCO 也接受 GECCO
            if key.endswith("_" + cand.upper()):
                data_dir = os.path.join(data_root, cand)
                break
        else:
            # 尝试用 _MV_NAME_MAP 的值反查目录（目录名可能不同于请求名）
            for req, prefix in _MV_NAME_MAP.items():
                if req == key and os.path.isdir(os.path.join(data_root, prefix)):
                    data_dir = os.path.join(data_root, prefix)
                    break

    # ---- .npy 路径 ----
    prefix = _MV_NAME_MAP.get(key)
    if prefix:
        tr_path = os.path.join(data_dir, f"{prefix}_train.npy")
        te_path = os.path.join(data_dir, f"{prefix}_test.npy")
        lb_path = os.path.join(data_dir, f"{prefix}_test_label.npy")
        if os.path.exists(tr_path) and os.path.exists(te_path):
            train = np.load(tr_path)
            test = np.load(te_path)
            label = np.load(lb_path) if os.path.exists(lb_path) else None
            return _finalize_mv(name, train, test, label)

    # ---- .csv 路径 ----
    if key in _MV_CSV_MAP:
        tr_f, te_f, lb_f = _MV_CSV_MAP[key]
        tr_path = os.path.join(data_dir, tr_f)
        te_path = os.path.join(data_dir, te_f)
        if os.path.exists(tr_path) and os.path.exists(te_path):
            train = _load_csv_features(tr_path)
            test = _load_csv_features(te_path)
            label = None
            if lb_f:
                lb_path = os.path.join(data_dir, lb_f)
                if os.path.exists(lb_path):
                    label = _load_csv_label(lb_path, len(test))
            return _finalize_mv(name, train, test, label)

    print(f"[数据] 找不到多变量数据集 {name}（查过 {data_dir}）")
    return None, None, None


def _load_csv_features(path):
    """
    读取特征 CSV：剔除表头与全 NaN 列/行

    用 pandas 而非 np.loadtxt，因为 PSM 等数据集存在空单元格，
    loadtxt 会直接抛 ValueError。
    """
    import pandas as pd

    df = pd.read_csv(path)
    # 全 NaN 列（某些传感器全程无数据）
    df = df.dropna(axis=1, how="all")
    arr = df.to_numpy(dtype=np.float32, na_value=np.nan)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    # 剔除首列时间戳：非负、单调递增、量级远大于其他列中位数
    if arr.shape[1] > 1:
        first = arr[:, 0]
        others = np.nanmedian(np.abs(arr[:, 1:]))
        with np.errstate(invalid="ignore"):
            if (
                np.all(first >= 0)
                and np.all(np.diff(first) >= 0)
                and others > 0
                and np.nanmax(first) > others * 1000
            ):
                arr = arr[:, 1:]

    n_nan = int(np.isnan(arr).sum())
    if n_nan:
        print(f"[数据] {os.path.basename(path)} 含 {n_nan} 个 NaN（预处理时会填充）")
    return arr


def _load_csv_label(path, expected_len):
    """读取标签 CSV：取最后一列为标签，二值化"""
    import pandas as pd

    df = pd.read_csv(path)
    label = (df.iloc[:, -1].to_numpy(dtype=np.float32, na_value=0.0) > 0).astype(np.int32)
    if len(label) != expected_len:
        print(f"[数据] 标签长度 {len(label)} 与测试集 {expected_len} 不一致，按较短的对齐")
    return label


def _finalize_mv(name, train, test, label):
    """统一后处理：剔除全 NaN 通道、补维度、处理标签"""
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if train.ndim == 1:
        train = train[:, np.newaxis]
    if test.ndim == 1:
        test = test[:, np.newaxis]
    if label is not None:
        label = np.asarray(label).astype(np.int32).squeeze()

    # WADI 等数据集存在部分全 NaN 通道（126 通道中 4 个），必须剔除，
    # 否则 nan_to_num 填 0 会把标准化统计量拉偏
    all_nan = np.isnan(train).all(axis=0) | np.isnan(test).all(axis=0)
    if all_nan.any():
        keep = ~all_nan
        train = train[:, keep]
        test = test[:, keep]
        print(f"[数据] 剔除 {int(all_nan.sum())} 个全NaN 通道，剩余 {train.shape[1]} 通道")

    # 剔除训练集中的部分 NaN 行（某时间点所有通道都缺失）
    bad_rows = np.isnan(train).all(axis=1)
    if bad_rows.any():
        train = train[~bad_rows]
        print(f"[数据] 剔除训练集中 {int(bad_rows.sum())} 个全 NaN 时间点")

    print(f"[数据] {name}: train={train.shape}, test={test.shape}", end="")
    if label is not None:
        print(f", label={label.shape} (异常比例={np.mean(label):.4f})")
    else:
        print(" (无标签)")

    return train, test, label


def analyze_dataset_characteristics(data, labels=None):
    """分析数据集基本特性（支持多变量）"""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    info = {
        "length": len(data),
        "n_features": int(data.shape[1]),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }
    if labels is not None:
        labels = np.asarray(labels)
        info["anomaly_ratio"] = float(np.mean(labels))
        info["num_anomalies"] = int(np.sum(labels))

        # 计算异常段
        segments = []
        in_anomaly = False
        seg_start = 0
        for i, l in enumerate(labels):
            if l == 1 and not in_anomaly:
                in_anomaly = True
                seg_start = i
            elif l == 0 and in_anomaly:
                in_anomaly = False
                segments.append(i - seg_start)
        if in_anomaly:
            segments.append(len(labels) - seg_start)

        info["num_anomaly_segments"] = len(segments)
        info["avg_anomaly_length"] = float(np.mean(segments)) if segments else 0.0
        info["max_anomaly_length"] = int(max(segments)) if segments else 0

    return info
