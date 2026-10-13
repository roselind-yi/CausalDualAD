#!/usr/bin/env python3
"""
run_ucr.py — UCR 异常检测单一入口（修复版）

用法:
    # 在仓库根目录运行
    python -m src.ucr_pipeline.run_ucr --ucr-path /path/to/UCR --dataset-ids 1 2 3

    # 只跑前 5 个数据集
    python -m src.ucr_pipeline.run_ucr --ucr-path /path/to/UCR --dataset-ids 1 2 3 4 5

    # 自定义参数
    python -m src.ucr_pipeline.run_ucr --ucr-path /path/to/UCR --epochs 100 --batch-size 64 --window-size 128

    # 指定 GPU
    python -m src.ucr_pipeline.run_ucr --ucr-path /path/to/UCR --device cuda:0

    # 烟雾测试（合成数据，不需要真实数据）
    python -m src.ucr_pipeline.run_ucr --smoke-test

注意:
    旧代码有 7 个 main 入口、3 个 solution、6 个 model 变体，import 路径互相冲突。
    本文件是唯一的 UCR 入口，替换所有旧入口。
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# 确保能 import src.ucr_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ucr_pipeline import (
    DualPathModel,
    UCRTrainer,
    UCREvaluator,
    load_ucr_dataset,
    list_available_ucr,
    preprocess_ucr,
    analyze_dataset_characteristics,
    default_config,
    save_config,
)


def run_single_dataset(dataset_id, config, device=None, save_dir=None):
    """
    对单个 UCR 数据集执行完整流程: 加载 → 预处理 → 训练 → 检测 → 评估

    返回:
        metrics dict
    """
    print(f"\n{'='*70}")
    print(f"  UCR_{dataset_id}")
    print(f"{'='*70}")

    ucr_path = config["data_params"]["ucr_path"]

    # 1. 加载数据
    train_data, test_data, test_label = load_ucr_dataset(dataset_id, ucr_path)
    if train_data is None:
        print(f"  [跳过] UCR_{dataset_id} 数据不可用")
        return None

    # 数据特性分析
    char = analyze_dataset_characteristics(test_data, test_label)
    print(f"  [特性] 长度={char['length']}, 异常比例={char.get('anomaly_ratio', 'N/A')}")

    # 2. 预处理
    train_norm, test_norm, scaler = preprocess_ucr(train_data, test_data)

    # 3. 创建模型
    model = DualPathModel(config)

    # 4. 训练
    tp = config["train_params"]
    trainer = UCRTrainer(model, device=device)
    window_size = config["model_params"]["window_size"]

    train_losses, val_losses = trainer.train(
        train_norm,
        epochs=tp["epochs"],
        batch_size=tp["batch_size"],
        learning_rate=tp["learning_rate"],
        patience=tp["patience"],
        window_size=window_size,
    )

    # 5. 异常检测
    dp = config["detect_params"]
    anomaly_scores = trainer.detect_anomalies(
        test_norm,
        window_size=window_size,
        stride=dp["stride"],
        batch_size=dp["batch_size"],
    )

    # 6. 评估
    evaluator = UCREvaluator()
    if test_label is not None:
        # 确保 scores 和 labels 长度一致
        min_len = min(len(anomaly_scores), len(test_label))
        metrics = evaluator.evaluate(anomaly_scores[:min_len], test_label[:min_len])
        print(f"\n  [结果] UCR_{dataset_id}:")
        evaluator.print_metrics(metrics, prefix="    ")
    else:
        metrics = {"note": "no labels"}
        print(f"  [结果] UCR_{dataset_id}: 无标签，跳过评估")

    # 7. 保存模型和结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"UCR_{dataset_id}_model.pth")
        trainer.save_model(model_path)

        result_path = os.path.join(save_dir, f"UCR_{dataset_id}_results.json")
        result = {
            "dataset_id": dataset_id,
            "metrics": metrics,
            "train_losses": train_losses,
            "config": config,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [保存] 结果已保存到 {result_path}")

    return metrics


def run_batch(config, device=None, save_dir=None):
    """批量运行多个 UCR 数据集"""
    dataset_ids = config["data_params"]["dataset_ids"]
    ucr_path = config["data_params"]["ucr_path"]

    # 检查哪些数据集可用
    available = list_available_ucr(ucr_path)
    if not available:
        print(f"[错误] 路径 {ucr_path} 下没有找到任何 UCR 数据集")
        return {}

    print(f"\n[批量] 可用数据集: {len(available)} 个 (ID: {available[:10]}...)")
    # 过滤出请求的 ID
    to_run = [i for i in dataset_ids if i in available]
    skipped = [i for i in dataset_ids if i not in available]
    if skipped:
        print(f"[批量] 跳过不可用的: {skipped}")
    print(f"[批量] 将运行: {to_run}")

    all_results = {}
    all_metrics = []
    t0 = time.time()

    for ds_id in to_run:
        metrics = run_single_dataset(ds_id, config, device=device, save_dir=save_dir)
        if metrics is not None and "auroc" in metrics:
            all_results[ds_id] = metrics
            all_metrics.append(metrics)

    elapsed = time.time() - t0

    # 汇总
    print(f"\n{'='*70}")
    print(f"  批量结果汇总 ({len(all_results)} 个数据集, 耗时 {elapsed:.0f}s)")
    print(f"{'='*70}")

    if all_metrics:
        evaluator = UCREvaluator()
        avg_metrics, _ = evaluator.evaluate_batch(
            [m["_scores"] for m in all_metrics if "_scores" in m],
            [m["_labels"] for m in all_metrics if "_labels" in m],
        ) if False else (None, None)  # 不做跨数据集平均分数

        # 简单平均
        keys = ["auroc", "vus_roc", "vus_pr", "top_1", "top_3", "top_5", "f1", "precision", "recall"]
        print(f"\n  {'ID':>4} | {'AUROC':>7} | {'VUS-ROC':>7} | {'VUS-PR':>7} | {'Top-1':>5} | {'F1':>7} | {'P':>7} | {'R':>7}")
        print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for ds_id, m in all_results.items():
            print(f"  {ds_id:>4} | {m.get('auroc',0):>7.4f} | {m.get('vus_roc',0):>7.4f} | {m.get('vus_pr',0):>7.4f} | {m.get('top_1',0):>5.1f} | {m.get('f1',0):>7.4f} | {m.get('precision',0):>7.4f} | {m.get('recall',0):>7.4f}")

        # 平均
        print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
        avgs = {k: np.mean([m.get(k, 0) for m in all_results.values()]) for k in keys}
        print(f"  {'AVG':>4} | {avgs['auroc']:>7.4f} | {avgs['vus_roc']:>7.4f} | {avgs['vus_pr']:>7.4f} | {avgs['top_1']:>5.1f} | {avgs['f1']:>7.4f} | {avgs['precision']:>7.4f} | {avgs['recall']:>7.4f}")

        if save_dir:
            summary_path = os.path.join(save_dir, "batch_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"results": all_results, "averages": avgs, "elapsed_seconds": elapsed},
                    f, indent=2, ensure_ascii=False, default=str
                )
            print(f"\n  [保存] 汇总已保存到 {summary_path}")

    return all_results


def smoke_test(device=None):
    """
    烟雾测试 — 用合成数据验证代码能跑通，不需要真实 UCR 数据

    验证项:
    1. 模型能前向传播
    2. loss 能计算 + 反向传播
    3. detect_anomalies 能跑完
    4. 评估指标能计算
    5. GPU 可用时自动使用
    """
    print("\n" + "=" * 70)
    print("  烟雾测试 — 合成数据验证代码正确性")
    print("=" * 70)

    config = default_config()
    config["train_params"]["epochs"] = 2  # 只跑 2 轮验证
    config["train_params"]["batch_size"] = 8

    # 合成数据：正弦波 + 中间注入异常
    np.random.seed(42)
    t = np.arange(2000, dtype=np.float32)
    train_data = np.sin(t / 20) + 0.1 * np.random.randn(2000)

    test_data = np.sin(t / 20) + 0.1 * np.random.randn(2000)
    test_label = np.zeros(2000, dtype=np.int32)
    test_label[900:1100] = 1  # 注入异常段
    test_data[900:1100] += 2.0  # 异常段幅度增大

    print(f"\n[1/5] 合成数据: train={train_data.shape}, test={test_data.shape}, 异常比例={np.mean(test_label):.4f}")

    # 预处理
    train_norm, test_norm, scaler = preprocess_ucr(train_data, test_data)
    print(f"[2/5] 预处理完成: mean={scaler['mean']:.4f}, std={scaler['std']:.4f}")

    # 模型
    model = DualPathModel(config)
    trainer = UCRTrainer(model, device=device)
    print(f"[3/5] 模型创建完成, 参数量={sum(p.numel() for p in model.parameters()):,}")

    # 前向传播验证
    dummy = torch.randn(4, config["model_params"]["window_size"], 1).to(trainer.device)
    out = model(dummy)
    print(f"     前向传播: fused_score shape={out['fused_score'].shape}, device={out['fused_score'].device}")

    # 训练 2 轮
    print(f"\n[4/5] 训练 2 轮验证...")
    trainer.train(
        train_norm,
        epochs=2,
        batch_size=8,
        learning_rate=5e-4,
        patience=5,
        window_size=config["model_params"]["window_size"],
    )

    # 异常检测
    print(f"\n[5/5] 异常检测 + 评估...")
    scores = trainer.detect_anomalies(
        test_norm,
        window_size=config["model_params"]["window_size"],
        stride=1,
        batch_size=64,
    )

    evaluator = UCREvaluator()
    min_len = min(len(scores), len(test_label))
    metrics = evaluator.evaluate(scores[:min_len], test_label[:min_len])
    evaluator.print_metrics(metrics, prefix="     ")

    print("\n" + "=" * 70)
    print("  烟雾测试通过！代码可以正常运行。")
    print(f"  设备: {trainer.device}")
    if trainer.device.type == "cuda":
        print(f"  GPU 显存使用: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("=" * 70)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="UCR 异常检测 — 修复版入口")
    parser.add_argument("--ucr-path", type=str, default="", help="UCR 数据根目录")
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=None, help="数据集 ID 列表")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="训练批量")
    parser.add_argument("--window-size", type=int, default=None, help="滑窗长度")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--patience", type=int, default=None, help="早停 patience")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda / cuda:0 / cpu)")
    parser.add_argument("--save-dir", type=str, default=None, help="结果保存目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (JSON)")
    parser.add_argument("--smoke-test", action="store_true", help="运行烟雾测试（合成数据）")
    parser.add_argument("--list-datasets", action="store_true", help="列出可用数据集后退出")
    args = parser.parse_args()

    # 烟雾测试
    if args.smoke_test:
        smoke_test(device=args.device)
        return

    # 加载配置
    if args.config:
        from src.ucr_pipeline.config import load_config
        config = load_config(args.config)
    else:
        config = default_config()

    # 命令行参数覆盖
    if args.epochs is not None:
        config["train_params"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train_params"]["batch_size"] = args.batch_size
    if args.window_size is not None:
        config["model_params"]["window_size"] = args.window_size
    if args.lr is not None:
        config["train_params"]["learning_rate"] = args.lr
    if args.patience is not None:
        config["train_params"]["patience"] = args.patience
    if args.ucr_path:
        config["data_params"]["ucr_path"] = args.ucr_path
    if args.dataset_ids is not None:
        config["data_params"]["dataset_ids"] = args.dataset_ids

    ucr_path = config["data_params"]["ucr_path"]
    if not ucr_path:
        print("[错误] 必须指定 --ucr-path 或在配置中设置 ucr_path")
        print("  用法: python -m src.ucr_pipeline.run_ucr --ucr-path /path/to/UCR --dataset-ids 1 2 3")
        print("  烟雾测试: python -m src.ucr_pipeline.run_ucr --smoke-test")
        sys.exit(1)

    # 列出数据集
    if args.list_datasets:
        available = list_available_ucr(ucr_path)
        print(f"路径 {ucr_path} 下可用数据集 ({len(available)} 个):")
        print(f"  {available}")
        return

    # 批量运行
    run_batch(config, device=args.device, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
