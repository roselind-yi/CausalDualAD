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
    UCR 数据预处理

    关键：用训练集的 mean/std 标准化测试集（防止数据泄漏）

    参数:
        train_data: 训练数据
        test_data: 测试数据
    返回:
        (train_norm, test_norm, scaler)
    """
    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)

    # 确保是一维序列
    if train_data.ndim > 1:
        train_data = train_data.squeeze()
    if test_data.ndim > 1:
        test_data = test_data.squeeze()

    # 处理 NaN
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    # 用训练集统计量标准化（防止数据泄漏）
    mean = train_data.mean()
    std = train_data.std()
    if std < 1e-8:
        std = 1.0

    train_norm = (train_data - mean) / std
    test_norm = (test_data - mean) / std

    # 范围限制
    train_norm = np.clip(train_norm, -5, 5)
    test_norm = np.clip(test_norm, -5, 5)

    scaler = {"mean": float(mean), "std": float(std)}
    print(f"[预处理] mean={mean:.4f}, std={std:.4f}")
    print(f"[预处理] train_range=[{train_norm.min():.2f}, {train_norm.max():.2f}]")
    print(f"[预处理] test_range=[{test_norm.min():.2f}, {test_norm.max():.2f}]")

    return train_norm, test_norm, scaler


def analyze_dataset_characteristics(data, labels=None):
    """分析数据集基本特性"""
    data = np.asarray(data)
    info = {
        "length": len(data),
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
