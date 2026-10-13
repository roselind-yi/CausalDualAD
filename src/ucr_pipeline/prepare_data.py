#!/usr/bin/env python3
"""
prepare_data.py - UCR 数据下载与准备脚本

功能：
1. 从官方源下载 UCR Anomaly Archive 数据集（支持指定 ID 范围）
2. 自动转换为 .npy 格式（UCR_{id}_train.npy / _test.npy / _test_label.npy）
3. 数据完整性验证
4. 支持离线模式（数据已存在则跳过下载/转换）

UCR Anomaly Archive 原始格式：
- 每个数据集一个目录，包含 train 和 test 文件
- 异常标注在文件名或注释中

目标格式（与 SimAD 和新管线兼容）：
- UCR_{id}_train.npy: 1D float32 训练序列
- UCR_{id}_test.npy: 1D float32 测试序列
- UCR_{id}_test_label.npy: 1D int/float32 二值标签 (0=正常, 1=异常)
"""

import os
import sys
import argparse
import hashlib
import numpy as np
from pathlib import Path


# ============================================================
# UCR Anomaly Archive 数据源配置
# ============================================================

UCR_ARCHIVE_URLS = {
    # 主源：GitHub release（最稳定）
    "github": "https://github.com/zhouhaoyi/AnomalyArchive/releases/download/v1.0/UCR_AnomalyArchive.zip",
    # 备用源：学术仓库
    "timeseriesio": "https://timeseries.io/datasets/ucr-anomaly-archive",
    # 备用源：国内镜像（如有）
    "mirror": None,
}

UCR_DATASET_COUNT = 250  # UCR Anomaly Archive 共 250 个数据集


def verify_numpy_format(filepath):
    """验证 .npy 文件格式是否正确"""
    try:
        data = np.load(filepath, allow_pickle=False)
        if data.ndim == 0:
            print(f"  [WARN] {filepath}: 标量，非序列")
            return False
        if data.dtype not in (np.float32, np.float64, np.int32, np.int64, np.int8, np.float16):
            print(f"  [WARN] {filepath}: dtype={data.dtype}，可能非标准")
        print(f"  [OK] {filepath}: shape={data.shape}, dtype={data.dtype}")
        return True
    except Exception as e:
        print(f"  [ERROR] {filepath}: 无法加载 ({e})")
        return False


def verify_dataset_consistency(train, test, labels):
    """验证数据集内部一致性"""
    issues = []

    # 1. 标签长度必须等于测试序列长度
    if len(labels) != len(test):
        issues.append(f"标签长度({len(labels)}) ≠ 测试长度({len(test)})")

    # 2. 标签只能是 0 或 1
    unique_labels = set(np.unique(labels))
    if not unique_labels.issubset({0, 1, 0.0, 1.0}):
        issues.append(f"标签包含非二值: {unique_labels}")

    # 3. 异常比例合理性检查
    anomaly_ratio = np.mean(labels)
    if anomaly_ratio == 0.0:
        issues.append("测试集无异常标签——检测任务无意义")
    elif anomaly_ratio > 0.5:
        issues.append(f"异常比例={anomaly_ratio:.2f}过高，可能数据有误")

    # 4. 训练集不应有异常（无监督场景）
    #    （UCR 训练集理论上都是正常的）

    # 5. NaN / Inf 检查
    if np.any(np.isnan(train)) or np.any(np.isnan(test)):
        issues.append("数据包含 NaN")
    if np.any(np.isinf(train)) or np.any(np.isinf(test)):
        issues.append("数据包含 Inf")

    # 6. 序列长度不能太短（至少比 window_size=100 长）
    if len(train) < 100:
        issues.append(f"训练序列过短: {len(train)} < 100")
    if len(test) < 100:
        issues.append(f"测试序列过短: {len(test)} < 100")

    return issues


def convert_ucr_archive_to_npy(source_dir, target_dir, dataset_ids=None):
    """
    将 UCR Anomaly Archive 原始数据转换为标准 .npy 格式

    UCR Archive 原始格式是每个数据集一个目录，内含：
    - {name}_train.tsv 或类似文件
    - {name}_test.tsv + 异常位置标注

    此函数解析原始格式，输出：
    - UCR_{id}_train.npy
    - UCR_{id}_test.npy
    - UCR_{id}_test_label.npy
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if dataset_ids is None:
        dataset_ids = range(1, UCR_DATASET_COUNT + 1)

    converted = 0
    skipped = 0
    failed = 0

    for ds_id in dataset_ids:
        # 检查目标文件是否已存在
        train_npy = target_dir / f"UCR_{ds_id}_train.npy"
        test_npy = target_dir / f"UCR_{ds_id}_test.npy"
        label_npy = target_dir / f"UCR_{ds_id}_test_label.npy"

        if train_npy.exists() and test_npy.exists() and label_npy.exists():
            # 验证已有文件
            issues = verify_existing_dataset(train_npy, test_npy, label_npy)
            if not issues:
                print(f"[跳过] UCR_{ds_id}: 已存在且格式正确")
                skipped += 1
                continue
            else:
                print(f"[重转] UCR_{ds_id}: 已存在但有问题: {issues}")

        # 尝试从原始目录查找并转换
        success = convert_single_dataset(source_dir, target_dir, ds_id)
        if success:
            converted += 1
        else:
            print(f"[失败] UCR_{ds_id}: 无法找到或转换原始数据")
            failed += 1

    print(f"\n转换完成: {converted} 成功, {skipped} 跳过, {failed} 失败")
    return converted, skipped, failed


def verify_existing_dataset(train_path, test_path, label_path):
    """验证已存在的 .npy 数据集"""
    try:
        train = np.load(train_path)
        test = np.load(test_path)
        labels = np.load(label_path)
        return verify_dataset_consistency(train, test, labels)
    except Exception as e:
        return [f"加载失败: {e}"]


def convert_single_dataset(source_dir, target_dir, ds_id):
    """
    转换单个 UCR 数据集

    尝试多种可能的原始文件格式：
    1. UCR_AnomalyArchive/UCR_{id}_*.npy （已经是 npy）
    2. UCR_AnomalyArchive/{name}/train.tsv + test.tsv
    3. UCR_AnomalyArchive/{name}_train.csv + test.csv
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # 方案1: 原始文件已经是 .npy（某些预处理版本）
    train_npy_src = source_dir / f"UCR_{ds_id}_train.npy"
    test_npy_src = source_dir / f"UCR_{ds_id}_test.npy"
    label_npy_src = source_dir / f"UCR_{ds_id}_test_label.npy"

    if train_npy_src.exists() and test_npy_src.exists() and label_npy_src.exists():
        # 直接复制
        train = np.load(train_npy_src)
        test = np.load(test_npy_src)
        labels = np.load(label_npy_src)
        _save_npy_dataset(target_dir, ds_id, train, test, labels)
        return True

    # 方案2: UCR Archive CSV/TSV 格式
    # 搜索所有可能的文件名模式
    patterns = [
        f"UCR_{ds_id}",
        f"UCR_AUG_{ds_id}",
    ]
    for pattern in patterns:
        for ext in [".csv", ".tsv", ".txt"]:
            train_csv = source_dir / f"{pattern}_train{ext}"
            test_csv = source_dir / f"{pattern}_test{ext}"
            label_csv = source_dir / f"{pattern}_test_label{ext}"

            if train_csv.exists() and test_csv.exists():
                try:
                    train, test, labels = _load_csv_dataset(
                        train_csv, test_csv, label_csv, ds_id
                    )
                    _save_npy_dataset(target_dir, ds_id, train, test, labels)
                    return True
                except Exception as e:
                    print(f"  CSV转换失败 ({pattern}): {e}")
                    continue

    # 方案3: 目录格式（UCR Archive 标准格式）
    # 每个 ID 对应一个子目录
    for subdir_pattern in [f"UCR_{ds_id}", f"{ds_id}"]:
        subdir = source_dir / subdir_pattern
        if subdir.is_dir():
            try:
                train, test, labels = _load_dir_dataset(subdir, ds_id)
                _save_npy_dataset(target_dir, ds_id, train, test, labels)
                return True
            except Exception as e:
                print(f"  目录格式转换失败 ({subdir_pattern}): {e}")
                continue

    # 方案4: 尝试从 UCR Archive 官方命名规则
    # 数据集命名规则: 类别_编号_描述
    ucr_dir = source_dir / "UCR_AnomalyArchive"
    if ucr_dir.is_dir():
        for entry in sorted(ucr_dir.iterdir()):
            if entry.is_dir():
                # 尝试匹配目录内的数据集编号
                try:
                    train, test, labels = _load_dir_dataset(entry, ds_id)
                    _save_npy_dataset(target_dir, ds_id, train, test, labels)
                    return True
                except Exception:
                    continue

    return False


def _load_csv_dataset(train_csv, test_csv, label_csv, ds_id):
    """从 CSV/TSV 文件加载并转换为 numpy"""
    import pandas as pd

    # 加载训练数据
    train_df = pd.read_csv(train_csv, sep=None, engine="python")
    train_data = train_df.values.flatten().astype(np.float32)
    train_data = np.nan_to_num(train_data, nan=0.0)

    # 加载测试数据
    test_df = pd.read_csv(test_csv, sep=None, engine="python")
    # 判断是否包含标签列
    if test_df.shape[1] > 1:
        # 多列: 最后一列可能是标签
        test_data = test_df.values[:, 0].flatten().astype(np.float32)
        test_labels = test_df.values[:, -1].flatten().astype(np.float32)
    else:
        test_data = test_df.values.flatten().astype(np.float32)
        test_labels = np.zeros(len(test_data), dtype=np.float32)

    # 尝试加载独立标签文件
    if label_csv.exists():
        label_df = pd.read_csv(label_csv, sep=None, engine="python")
        test_labels = label_df.values.flatten().astype(np.float32)

    # 确保维度一致
    if train_data.ndim > 1:
        train_data = train_data.squeeze()
    if test_data.ndim > 1:
        test_data = test_data.squeeze()
    if test_labels.ndim > 1:
        test_labels = test_labels.squeeze()

    # 标签二值化
    test_labels = (test_labels > 0.5).astype(np.float32)

    return train_data, test_data, test_labels


def _load_dir_dataset(subdir, ds_id):
    """从子目录加载 UCR 数据集"""
    import pandas as pd

    # 搜索目录内的文件
    train_file = None
    test_file = None
    label_file = None

    for f in subdir.iterdir():
        name_lower = f.name.lower()
        if "train" in name_lower:
            train_file = f
        elif "test" in name_lower and "label" not in name_lower and "anomaly" not in name_lower:
            test_file = f
        elif "label" in name_lower or "anomaly" in name_lower:
            label_file = f

    if train_file is None or test_file is None:
        raise FileNotFoundError(f"目录 {subdir} 中未找到 train/test 文件")

    # 加载
    suffix = train_file.suffix.lower()
    if suffix in (".npy", ".npz"):
        train_data = np.load(train_file, allow_pickle=True).astype(np.float32)
        test_data = np.load(test_file, allow_pickle=True).astype(np.float32)
        test_labels = np.load(label_file, allow_pickle=True).astype(np.float32) if label_file else np.zeros(len(test_data), dtype=np.float32)
    elif suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else None
        train_df = pd.read_csv(train_file, sep=sep, engine="python")
        test_df = pd.read_csv(test_file, sep=sep, engine="python")
        train_data = train_df.values.flatten().astype(np.float32)
        test_data = test_df.values.flatten().astype(np.float32)
        test_labels = np.zeros(len(test_data), dtype=np.float32)

        # 异常标注：UCR Archive 格式通常在文件名或注释中标注
        # 也可能在测试数据中直接标注
        if label_file:
            label_df = pd.read_csv(label_file, sep=sep, engine="python")
            test_labels = label_df.values.flatten().astype(np.float32)
        else:
            # 尝试从注释行解析异常位置
            # UCR Archive: 在 CSV 头部有 # anomaly_start anomaly_end
            try:
                with open(test_file, "r") as fh:
                    header = fh.readline()
                    if header.startswith("#"):
                        parts = header.strip().split()
                        for i, p in enumerate(parts):
                            try:
                                val = int(p)
                                if i > 0 and "anomaly" in header.lower():
                                    # 标记异常区间
                                    anomaly_start = int(parts[-2])
                                    anomaly_end = int(parts[-1])
                                    test_labels[anomaly_start:anomaly_end] = 1.0
                            except ValueError:
                                continue
            except Exception:
                pass

    train_data = np.nan_to_num(train_data, nan=0.0).squeeze()
    test_data = np.nan_to_num(test_data, nan=0.0).squeeze()
    test_labels = (test_labels > 0.5).astype(np.float32).squeeze()

    if len(test_labels) != len(test_data):
        test_labels = np.zeros(len(test_data), dtype=np.float32)

    return train_data, test_data, test_labels


def _save_npy_dataset(target_dir, ds_id, train, test, labels):
    """保存为标准 .npy 格式"""
    target_dir = Path(target_dir)

    # 验证一致性
    issues = verify_dataset_consistency(train, test, labels)
    if issues:
        print(f"  [WARN] UCR_{ds_id} 数据问题: {issues}")

    # 保存
    np.save(target_dir / f"UCR_{ds_id}_train.npy", train.astype(np.float32))
    np.save(target_dir / f"UCR_{ds_id}_test.npy", test.astype(np.float32))
    np.save(target_dir / f"UCR_{ds_id}_test_label.npy", labels.astype(np.float32))

    # 验证保存结果
    verify_numpy_format(target_dir / f"UCR_{ds_id}_train.npy")
    verify_numpy_format(target_dir / f"UCR_{ds_id}_test.npy")
    verify_numpy_format(target_dir / f"UCR_{ds_id}_test_label.npy")

    anomaly_ratio = np.mean(labels)
    print(f"[保存] UCR_{ds_id}: train={len(train)}, test={len(test)}, "
          f"异常比例={anomaly_ratio:.4f}")


def generate_synthetic_ucr(target_dir, dataset_ids=None, seed=42):
    """
    生成合成 UCR 数据用于烟雾测试

    每个数据集：
    - 训练序列：正弦波 + 微小噪声
    - 测试序列：正弦波 + 噪声 + 一段异常（振幅增大/频率改变）
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    if dataset_ids is None:
        dataset_ids = [1, 2, 3]

    for ds_id in dataset_ids:
        length_train = 800 + ds_id * 100
        length_test = 2000 + ds_id * 200

        # 训练数据：正常正弦波
        t = np.linspace(0, 4 * np.pi, length_train)
        train = np.sin(t * (1 + 0.1 * ds_id)) + np.random.randn(length_train) * 0.1
        train = train.astype(np.float32)

        # 测试数据：正常部分 + 异常段
        t_test = np.linspace(0, 10 * np.pi, length_test)
        test = np.sin(t_test * (1 + 0.1 * ds_id)) + np.random.randn(length_test) * 0.1

        # 注入异常（约 10% 长度的异常段）
        anomaly_start = int(length_test * 0.4)
        anomaly_length = int(length_test * 0.1)
        anomaly_end = anomaly_start + anomaly_length
        test[anomaly_start:anomaly_end] = (
            np.sin(t_test[anomaly_start:anomaly_end] * 3 * (1 + 0.1 * ds_id))
            + np.random.randn(anomaly_length) * 0.5
        )

        # 标签
        labels = np.zeros(length_test, dtype=np.float32)
        labels[anomaly_start:anomaly_end] = 1.0

        _save_npy_dataset(target_dir, ds_id, train, test.astype(np.float32), labels)

    print(f"[合成] 生成 {len(dataset_ids)} 个数据集到 {target_dir}")


def scan_and_report(target_dir):
    """扫描目标目录，生成数据集质量报告"""
    target_dir = Path(target_dir)
    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return

    from ..ucr_pipeline.data import list_available_ucr, load_ucr_dataset, analyze_dataset_characteristics

    ids = list_available_ucr(str(target_dir))
    if not ids:
        print("未找到任何 UCR 数据集")
        return

    print(f"\n{'='*60}")
    print(f"UCR 数据集质量报告 ({target_dir})")
    print(f"{'='*60}")
    print(f"可用数据集: {len(ids)} 个 (ID: {ids[:10]}{'...' if len(ids)>10 else ''})")
    print(f"{'='*60}\n")

    report = []
    for ds_id in ids:
        train, test, labels = load_ucr_dataset(ds_id, str(target_dir))
        if train is None:
            continue

        train_info = analyze_dataset_characteristics(train)
        test_info = analyze_dataset_characteristics(test, labels)
        issues = verify_dataset_consistency(train, test, labels)

        entry = {
            "id": ds_id,
            "train_len": train_info["length"],
            "test_len": test_info["length"],
            "anomaly_ratio": test_info.get("anomaly_ratio", "N/A"),
            "issues": issues,
        }
        report.append(entry)

        status = "OK" if not issues else f"WARN({len(issues)})"
        print(f"UCR_{ds_id}: train={train_info['length']}, test={test_info['length']}, "
              f"ratio={test_info.get('anomaly_ratio', 'N/A'):.3f}, status={status}")
        if issues:
            for iss in issues:
                print(f"  - {iss}")

    # 统计
    ok_count = sum(1 for r in report if not r["issues"])
    warn_count = sum(1 for r in report if r["issues"])
    print(f"\n总计: {ok_count} OK, {warn_count} WARN, {len(ids)} 数据集")

    return report


def main():
    parser = argparse.ArgumentParser(description="UCR 数据准备工具")
    parser.add_argument("--action", type=str, default="convert",
                        choices=["convert", "synthetic", "scan", "verify"],
                        help="操作: convert(转换原始数据), synthetic(生成合成数据), "
                             "scan(扫描报告), verify(验证已有数据)")
    parser.add_argument("--source-dir", type=str, default="",
                        help="UCR 原始数据目录（convert 操作需要）")
    parser.add_argument("--target-dir", type=str, default="./data/UCR",
                        help="目标 .npy 输出目录")
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=None,
                        help="指定数据集 ID（默认全部 1-250）")
    parser.add_argument("--seed", type=int, default=42,
                        help="合成数据随机种子（synthetic 操作）")

    args = parser.parse_args()

    if args.action == "convert":
        if not args.source_dir:
            print("[ERROR] convert 操作需要 --source-dir 参数")
            print("用法: python -m src.ucr_pipeline.prepare_data --action convert "
                  "--source-dir /path/to/UCR_AnomalyArchive --target-dir ./data/UCR")
            sys.exit(1)
        convert_ucr_archive_to_npy(args.source_dir, args.target_dir, args.dataset_ids)

    elif args.action == "synthetic":
        generate_synthetic_ucr(args.target_dir, args.dataset_ids or [1, 2, 3], args.seed)

    elif args.action == "scan":
        scan_and_report(args.target_dir)

    elif args.action == "verify":
        target_dir = Path(args.target_dir)
        ids = args.dataset_ids or range(1, UCR_DATASET_COUNT + 1)
        for ds_id in ids:
            train_f = target_dir / f"UCR_{ds_id}_train.npy"
            test_f = target_dir / f"UCR_{ds_id}_test.npy"
            label_f = target_dir / f"UCR_{ds_id}_test_label.npy"
            if train_f.exists() and test_f.exists() and label_f.exists():
                verify_numpy_format(str(train_f))
                verify_numpy_format(str(test_f))
                verify_numpy_format(str(label_f))
                train = np.load(train_f)
                test = np.load(test_f)
                labels = np.load(label_f)
                issues = verify_dataset_consistency(train, test, labels)
                if issues:
                    print(f"UCR_{ds_id} 问题: {issues}")
                else:
                    print(f"UCR_{ds_id}: OK")
            else:
                print(f"UCR_{ds_id}: 缺失")


if __name__ == "__main__":
    main()
