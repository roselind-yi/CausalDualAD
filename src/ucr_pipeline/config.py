#!/usr/bin/env python3
"""
config.py - UCR 实验配置
"""

import json
import os


def default_config():
    """默认配置"""
    return {
        "model_params": {
            "window_size": 100,
            "hidden_dim": 128,
            "freq_components": 16,
            "temperature": 0.1,
            "lambda1": 0.1,
            "lambda2": 0.01,
            "num_heads": 4,
            "num_layers": 3,
            "use_fixed_weights": False,  # 默认走自适应，不硬编码 0.2/0.8
        },
        "train_params": {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "patience": 15,
            "weight_decay": 1e-5,
            "max_grad_norm": 0.5,
        },
        "detect_params": {
            "stride": 1,
            "batch_size": 256,  # 推理批量，GPU 上可以开大
        },
        "data_params": {
            "ucr_path": "",  # 必须设置：UCR 数据根目录
            "dataset_ids": [1, 2, 3, 4, 5],  # 默认测试前 5 个
            "clip_range": 5.0,
        },
    }


def load_config(path):
    """从 JSON 加载配置"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, path):
    """保存配置到 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def merge_config(base, override):
    """合并配置（override 覆盖 base）"""
    result = {}
    for key in set(list(base.keys()) + list(override.keys())):
        if key in base and key in override:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                result[key] = {**base[key], **override[key]}
            else:
                result[key] = override[key]
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result
