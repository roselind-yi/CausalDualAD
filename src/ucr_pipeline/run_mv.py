#!/usr/bin/env python3
"""
run_mv.py — 多变量时序异常检测入口

用途:
    毕设的核心方向（因果不变性 / 分布偏移 / 跨数据集泛化）只在多变量场景下成立，
    本入口用于在 SMAP / MSL / SMD / SWaT / WADI 等多变量基准上运行双路径模型。

用法:
    # 冒烟测试（合成多变量数据，不需真实数据）
    python -m src.ucr_pipeline.run_mv --smoke-test

    # 单个数据集快速验证（截取前N 个时间点）
    python -m src.ucr_pipeline.run_mv --data-root ./data --datasets SMAP --max-points 20000

    # 完整跑 SMAP
    python -m src.ucr_pipeline.run_mv --data-root ./data --datasets SMAP --epochs 10

    # 多个数据集对比（观察跨数据集差异）
    python -m src.ucr_pipeline.run_mv --data-root ./data --datasets SMAP MSL SMD \
        --epochs 10 --save-dir results/mv

设计说明:
    - 双路径在通道维各自输出分数（recon_error_mv / prototype_score_mv，形状 (B, W, C)）
    - 通道分数用 max 聚合到时间点：任一通道异常 -> 该时间点异常
    - 标准化按通道做（axis=0），避免跨通道量纲污染
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ucr_pipeline import (
    DualPathModel,
    UCRTrainer,
    UCREvaluator,
    load_mv_dataset,
    preprocess_ucr,
    analyze_dataset_characteristics,
    default_config,
)

# 已适配的多变量数据集
# .npy：SMAP(25) / MSL(55) / SMD(38) / WADI(126) / NIPS_TS_Creditcard(29) / NIPS_TS_GECCO(9) / NIPS_TS_Swan(38)
# .csv：PSM(25) / SWaT(51)
SUPPORTED = [
    "SMAP", "MSL", "SMD", "WADI",
    "PSM", "SWAT",
    "NIPS_TS_Creditcard", "NIPS_TS_GECCO", "NIPS_TS_Swan",
]


def build_config(n_features, args):
    """按通道数构造配置"""
    config = default_config()
    mp = config["model_params"]
    mp["n_features"] = n_features
    mp["freq_feature_dim"] = 64
    # 通道分数聚合方式（消融实验用）
    if getattr(args, "aggregate", None):
        mp["aggregate"] = args.aggregate
        mp["aggregate_topk"] = getattr(args, "aggregate_topk", 3)
    # 多变量序列长，窗口适当加大
    if args.window_size is not None:
        mp["window_size"] = args.window_size
    if args.epochs is not None:
        config["train_params"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train_params"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train_params"]["learning_rate"] = args.lr
    config["train_params"]["patience"] = args.patience
    return config


def run_one(name, args, device=None, save_dir=None):
    """在单个多变量数据集上跑完整流程"""
    print(f"\n{'='*70}")
    f"  {name}"
    print(f"{'='*70}")

    train, test, label = load_mv_dataset(name, args.data_root)
    if train is None:
        return None

    # 可选：截取（CPU 上跑完整数据太慢，先看趋势）
    # 用滑动窗口而非前 N 点，保证截取范围内一定含异常
    if args.max_points and args.max_points < len(train):
        start = args.start_offset or 0
        train = train[start : start + args.max_points]
        print(f"[截取] train 取[{start}, {start+len(train)})，共 {len(train)} 点")
    if args.max_points and args.max_points < len(test):
        start = args.start_offset or 0
        test = test[start : start + args.max_points]
        if label is not None:
            label = label[start : start + args.max_points]
        print(f"[截取] test 取 [{start}, {start+len(test)})，共 {len(test)} 点，"
              f"其中异常比例={label.mean():.4f}" if label is not None else "")

    char = analyze_dataset_characteristics(test, label)
    print(f"  [特性] 长度={char['length']}, 通道={char['n_features']}, "
          f"异常比例={char.get('anomaly_ratio', 'N/A')}")

    train_norm, test_norm, scaler = preprocess_ucr(train, test)
    n_features = train_norm.shape[1]

    config = build_config(n_features, args)
    model = DualPathModel(config)
    trainer = UCRTrainer(model, device=device)
    window_size = config["model_params"]["window_size"]

    train_losses, val_losses = trainer.train(
        train_norm,
        epochs=config["train_params"]["epochs"],
        batch_size=config["train_params"]["batch_size"],
        learning_rate=config["train_params"]["learning_rate"],
        patience=config["train_params"]["patience"],
        window_size=window_size,
    )

    scores = trainer.detect_anomalies(
        test_norm,
        window_size=window_size,
        stride=config["detect_params"]["stride"],
        batch_size=config["detect_params"]["batch_size"],
    )

    evaluator = UCREvaluator()
    metrics = {}
    if label is not None:
        min_len = min(len(scores), len(label))
        metrics = evaluator.evaluate(scores[:min_len], label[:min_len])
        print(f"\n  [结果] {name} (C={n_features}):")
        evaluator.print_metrics(metrics, prefix="    ")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        agg = config["model_params"].get("aggregate", "max")
        suffix = "" if agg == "max" else f"_{agg}"
        result = {
            "dataset": name,
            "n_features": n_features,
            "aggregate": agg,
            "metrics": metrics,
            "train_losses": train_losses,
            "config": {k: v for k, v in config.items() if k != "data_params"},
        }
        path = os.path.join(save_dir, f"{name}{suffix}_mv_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        np.save(os.path.join(save_dir, f"{name}{suffix}_mv_scores.npy"), scores)
        print(f"  [保存] {path}")

    return {
        "dataset": name,
        "n_features": n_features,
        "aggregate": config["model_params"].get("aggregate", "max"),
        "metrics": metrics,
    }


def smoke_test(device=None):
    """合成多变量数据冒烟测试"""
    print("\n" + "=" * 70)
    print("  烟雾测试 — 合成多变量数据")
    print("=" * 70)

    np.random.seed(42)
    T, C = 3000, 5
    t = np.arange(T)

    def gen(normal):
        base = np.stack(
            [np.sin((t + i * 37) / (15 + i * 3)) + 0.1 * np.random.randn(T) for i in range(C)],
            axis=1,
        )
        if not normal:
            base[1200:1500, 2] += 3.0   # 第3 通道注入异常
            base[2000:2100, 0] -= 2.5   # 第 1 通道注入异常
        return base.astype(np.float32)

    train = gen(True)
    test = gen(False)
    label = np.zeros(T, dtype=np.int32)
    label[1200:1500] = 1
    label[2000:2100] = 1

    print(f"\n[1/4] 合成数据: train={train.shape}, 通道={C}, 异常比例={np.mean(label):.4f}")
    train_norm, test_norm, scaler = preprocess_ucr(train, test)

    args = argparse.Namespace(
        epochs=2, batch_size=8, lr=5e-4, patience=5, window_size=100
    )
    config = build_config(C, args)
    model = DualPathModel(config)
    trainer = UCRTrainer(model, device=device)
    print(f"[2/4] 模型参数量={sum(p.numel() for p in model.parameters()):,}")

    print("[3/4] 训练 2 轮验证...")
    trainer.train(train_norm, epochs=2, batch_size=8, learning_rate=5e-4,
                  patience=5, window_size=100)

    print("[4/4] 检测 + 评估...")
    scores = trainer.detect_anomalies(test_norm, window_size=100, stride=1, batch_size=64)

    evaluator = UCREvaluator()
    min_len = min(len(scores), len(label))
    metrics = evaluator.evaluate(scores[:min_len], label[:min_len])
    evaluator.print_metrics(metrics, prefix="     ")

    print("\n" + "=" * 70)
    print("  多变量烟雾测试通过！")
    print("=" * 70)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="多变量时序异常检测入口")
    parser.add_argument("--data-root", type=str, default="./data", help="data/ 根目录")
    parser.add_argument("--datasets", type=str, nargs="+", default=["SMAP"],
                        help=f"数据集列表，可选 {SUPPORTED}")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--aggregate", type=str, default=None,
                        choices=["max", "mean", "topk_mean"],
                        help="通道分数聚合方式（消融实验用）")
    parser.add_argument("--aggregate-topk", type=int, default=3,
                        help="topk_mean 策略下的 k 值")
    parser.add_argument("--max-points", type=int, default=None,
                        help="每个数据集最多用多少时间点（调试用）")
    parser.add_argument("--start-offset", type=int, default=0,
                        help="截取起始位置，配合 --max-points 使用")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(device=args.device)
        return

    unknown = [d for d in args.datasets if d.upper() not in SUPPORTED]
    if unknown:
        print(f"[警告] 这些数据集不在已适配列表中，将尝试直接加载: {unknown}")
        print(f"  已适配: {SUPPORTED}")

    results = {}
    t0 = time.time()
    for name in args.datasets:
        r = run_one(name.upper(), args, device=args.device, save_dir=args.save_dir)
        if r and r.get("metrics"):
            results[name.upper()] = r

    elapsed = time.time() - t0
    if results:
        print(f"\n{'='*70}")
        print(f"  多变量汇总 ({len(results)} 个数据集, 耗时 {elapsed:.0f}s)")
        print(f"{'='*70}")
        keys = ["auroc", "vus_roc", "vus_pr", "top_1", "f1", "precision", "recall"]
        print(f"\n  {'数据集':<10}| {'C':>4} | {'AUROC':>7} | {'VUS-ROC':>7} | {'VUS-PR':>7} | {'Top-1':>5} | {'F1':>7}")
        print(f"  {'-'*10}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}")
        for name, r in results.items():
            m = r["metrics"]
            print(f"  {name:<10}| {r['n_features']:>4} | {m.get('auroc',0):>7.4f} | "
                  f"{m.get('vus_roc',0):>7.4f} | {m.get('vus_pr',0):>7.4f} | "
                  f"{m.get('top_1',0):>5.1f} | {m.get('f1',0):>7.4f}")
        avg = {k: float(np.mean([r["metrics"].get(k, 0) for r in results.values()])) for k in keys}
        print(f"  {'-'*10}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}")
        print(f"  {'AVG':<10}| {'':>4} | {avg['auroc']:>7.4f} | {avg['vus_roc']:>7.4f} | "
              f"{avg['vus_pr']:>7.4f} | {avg['top_1']:>5.1f} | {avg['f1']:>7.4f}")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            p = os.path.join(args.save_dir, "mv_summary.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"results": results, "averages": avg,
                           "elapsed_seconds": elapsed},
                          f, indent=2, ensure_ascii=False, default=str)
            print(f"\n  [保存] 汇总已保存到 {p}")


if __name__ == "__main__":
    main()