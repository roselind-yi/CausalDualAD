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


def load_mv_dataset(name, data_root):
    """
    加载多变量数据集

    参数:
        name: 数据集名（SMAP / MSL / SMD）
        data_root: data/ 根目录
    返回:
        (train_data, test_data, test_label)，形状 (T, C) / (T, C) / (T,)
    """
    name = name.upper()
    npy_dir = os.path.join(data_root, name)

    if os.path.isdir(npy_dir) and os.path.exists(os.path.join(npy_dir, f"{name}_train.npy")):
        train = np.load(os.path.join(npy_dir, f"{name}_train.npy"))
        test = np.load(os.path.join(npy_dir, f"{name}_test.npy"))
        label_path = os.path.join(npy_dir, f"{name}_test_label.npy")
        label = np.load(label_path) if os.path.exists(label_path) else None
    else:
        print(f"[数据] 找不到多变量数据集 {name}（期望 {npy_dir}/{name}_train.npy）")
        return None, None, None

    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if train.ndim == 1:
        train = train[:, np.newaxis]
    if test.ndim == 1:
        test = test[:, np.newaxis]
    if label is not None:
        label = np.asarray(label).astype(np.int32).squeeze()

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
