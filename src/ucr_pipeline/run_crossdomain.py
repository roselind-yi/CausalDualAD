#!/usr/bin/env python3
"""
run_crossdomain.py — 跨数据集泛化测量（判定实验）

用途:
    回答一个决定毕设方向的问题：**分布偏移到底有多严重？**
    如果跨域掉点很小，因果不变性解决不了问题，立论就塌了。

核心概念:
    - **域内性能 (in-domain)**：在数据集 A 上训练，测试 A → 模型在自己领域的表现
    - **跨域性能 (cross-domain)**：在数据集 A 上训练，测试 B → 分布偏移的影响
    - **掉点 (drop)** = 域内 AUROC - 跨域 AUROC

    判定标准（见 docs/目标体系与路线图.md）:
        掉点 >= 0.15→ 分布偏移是真问题，因果不变性值得做
        掉点< 0.10   → 因果不变性不是银弹，需重新找切入点

通道数对齐:
    各数据集通道数不同（SMAP 25 / MSL 55 / SMD 38 / GECCO 9），
    模型输入维度必须固定。三种策略:
        - first_c: 取前 C 个通道（C = 训练集通道数，训练集通道少时补零）
        - last_c:  取后 C 个通道
        - interp:   线性插值/重采样到 C 维（保持全部信息，但会破坏物理语义）
    默认 first_c，简单可解释。

用法:
    # 单个训练集→ 测试集
    python -m src.ucr_pipeline.run_crossdomain \
        --data-root ./data --train-on SMAP --test-on MSL

    # 多组配对
    python -m src.ucr_pipeline.run_crossdomain \
        --data-root ./data \
        --pairs SMAP:MSL SMAP:SMD MSL:SMAP

    # 全部数据集两两互测
    python -m src.ucr_pipeline.run_crossdomain \
        --data-root ./data --all-pairs --datasets SMAP MSL SMD

    # 调试（CPU 友好）
    python -m src.ucr_pipeline.run_crossdomain --train-on SMAP --test-on MSL \
        --max-points 8000 --epochs 5 --window-size 64 --batch-size 128
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.ucr_pipeline import (
    DualPathModel,
    UCRTrainer,
    UCREvaluator,
    load_mv_dataset,
    preprocess_ucr,
    default_config,
)


def align_channels(data, target_c, strategy="first_c"):
    """
    把 (T, C_src) 对齐到 (T, C_target)

    first_c: 取前 target_c 个通道，不足补零
    last_c:  取后 target_c 个通道，不足补零
    interp:   线性插值到 target_c 维（保留整体趋势，但破坏通道语义）
    """
    data = np.asarray(data, dtype=np.float32)
    src_c = data.shape[1]
    if src_c == target_c:
        return data

    if strategy in ("first_c", "last_c"):
        out = np.zeros((len(data), target_c), dtype=np.float32)
        k = min(src_c, target_c)
        if strategy == "first_c":
            out[:, :k] = data[:, :k]
        else:
            out[:, target_c - k :] = data[:, src_c - k :]
        return out

    if strategy == "interp":
        # 沿通道维线性插值 src_c -> target_c，不引入零值，但破坏通道物理语义
        src_pos = np.arange(src_c, dtype=np.float32)
        tgt_pos = np.linspace(0, src_c - 1, target_c, dtype=np.float32)
        out = np.empty((len(data), target_c), dtype=np.float32)
        for i in range(len(data)):
            out[i] = np.interp(tgt_pos, src_pos, data[i])
        return out

    raise ValueError(f"未知对齐策略: {strategy}")


def _slice(data, label, max_points, start_offset):
    """截取（CPU 调试用）"""
    if max_points and max_points < len(data):
        s = start_offset or 0
        data = data[s : s + max_points]
        if label is not None:
            label = label[s : s + max_points]
    return data, label


def _build_config(n_features, args):
    config = default_config()
    mp = config["model_params"]
    mp["n_features"] = n_features
    mp["freq_feature_dim"] = 64
    mp["aggregate"] = getattr(args, "aggregate", "max") or "max"
    if args.window_size:
        mp["window_size"] = args.window_size
    config["train_params"]["epochs"] = args.epochs
    config["train_params"]["batch_size"] = args.batch_size
    config["train_params"]["patience"] = args.patience
    if args.lr:
        config["train_params"]["learning_rate"] = args.lr
    return config


def train_on(train_name, args, device=None):
    """
    在 train_name 上训练，返回 (model, trainer, n_features, train_norm)
    """
    print(f"\n{'-'*70}")
    print(f"  训练集: {train_name}")
    print(f"{'-'*70}")

    train, _, _ = load_mv_dataset(train_name, args.data_root)
    if train is None:
        return None
    train, _ = _slice(train, None, args.max_points, args.start_offset)

    train_norm, _, _ = preprocess_ucr(train, train)  # 仅标准化训练集
    n_features = train_norm.shape[1]
    print(f"[训练集] {train_name}: shape={train_norm.shape}, 通道={n_features}")

    config = _build_config(n_features, args)
    model = DualPathModel(config)
    trainer = UCRTrainer(model, device=device)
    window_size = config["model_params"]["window_size"]

    trainer.train(
        train_norm,
        epochs=config["train_params"]["epochs"],
        batch_size=config["train_params"]["batch_size"],
        learning_rate=config["train_params"]["learning_rate"],
        patience=config["train_params"]["patience"],
        window_size=window_size,
    )
    return trainer, n_features, train_norm, config


def evaluate_on(trainer, test_name, n_features, args, device=None):
    """在 test_name 上评估（用已训练好的 trainer）"""
    print(f"\n{'-'*70}")
    print(f"  测试集: {test_name}")
    print(f"{'-'*70}")

    _, test, label = load_mv_dataset(test_name, args.data_root)
    if test is None:
        return None
    test, label = _slice(test, label, args.max_points, args.start_offset)

    # 通道对齐：把测试集映射到训练集的通道数
    if test.shape[1] != n_features:
        print(f"[对齐] 测试集 {test.shape[1]} 通道 → 训练集 {n_features} 通道"
              f"（策略={args.align}）")
        test = align_channels(test, n_features, args.align)

    # 用**测试集自己的**统计量标准化（因为要用它的真实分布做检测）
    # 这模拟了"部署到新数据集"的真实场景
    test_norm, _, _ = preprocess_ucr(test, test)

    scores = trainer.detect_anomalies(
        test_norm,
        window_size=args.window_size or 100,
        stride=1,
        batch_size=256,
        pool=args.pool,
        assign_mode=args.assign_mode,
    )

    if label is None:
        print("  [警告] 该数据集无标签，跳过评估")
        return None

    evaluator = UCREvaluator()
    min_len = min(len(scores), len(label))
    metrics = evaluator.evaluate(scores[:min_len], label[:min_len])
    print(f"\n  [结果] {test_name}（{len(scores)} 点）:")
    evaluator.print_metrics(metrics, prefix="    ")
    return metrics


def run_pair(train_name, test_name, args, device=None):
    """跑一组 训练集→测试集"""
    trained = train_on(train_name, args, device=device)
    if trained is None:
        return None
    trainer, n_features, _, config = trained

    metrics = evaluate_on(trainer, test_name, n_features, args, device=device)
    if metrics is None:
        return None

    return {
        "train_on": train_name,
        "test_on": test_name,
        "n_features": n_features,
        "metrics": metrics,
        "auroc": metrics.get("auroc", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="跨数据集泛化测量（判定实验）")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--train-on", type=str, default=None, help="训练集名称")
    parser.add_argument("--test-on", type=str, default=None, help="测试集名称")
    parser.add_argument("--pairs", type=str, nargs="+", default=None,
                        help="多组配对，格式 TRAIN:TEST")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["SMAP", "MSL", "SMD"],
                        help="--all-pairs 时使用的数据集列表")
    parser.add_argument("--all-pairs", action="store_true",
                        help="所有数据集两两互测（含自身，用于拿域内基线）")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--align", type=str, default="first_c",
                        choices=["first_c", "last_c", "interp"],
                        help="测试集通道数对齐策略")

    # 检测参数（必须与域内实验一致）
    parser.add_argument("--pool", type=str, default="max",
                        choices=["max", "mean", "topk_mean"])
    parser.add_argument("--assign-mode", type=str, default="full",
                        choices=["full", "center"])
    parser.add_argument("--aggregate", type=str, default=None,
                        choices=["max", "mean", "topk_mean"])

    # 调试
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--start-offset", type=int, default=3000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="results/crossdomain")
    args = parser.parse_args()

    # ---- 组装要跑的配对 ----
    pairs = []
    if args.train_on and args.test_on:
        pairs.append((args.train_on, args.test_on))
    elif args.pairs:
        for p in args.pairs:
            if ":" not in p:
                print(f"[警告] 配对格式错误: {p}（应为 TRAIN:TEST）")
                continue
            tr, te = p.split(":")
            pairs.append((tr, te))
    elif args.all_pairs:
        for tr in args.datasets:
            for te in args.datasets:
                pairs.append((tr, te))
    else:
        print("必须指定 --train-on/--test-on、--pairs 或 --all-pairs")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  跨数据集泛化测量 — 共 {len(pairs)} 组")
    print(f"{'='*70}")

    results = []
    t0 = time.time()
    for tr, te in pairs:
        print(f"\n>>> {tr} → {te} <<<")
        r = run_pair(tr, te, args, device=args.device)
        if r:
            results.append(r)

    # ---- 汇总 ----
    elapsed = time.time() - t0
    if not results:
        print("无有效结果")
        return

    print(f"\n{'='*70}")
    print(f"  汇总 ({len(results)} 组, 耗时 {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"\n  {'训练集':<16}{'测试集':<16}{'C':>4}{'AUROC':>9}{'VUS-PR':>9}{'Top-1':>7}")
    print(f"  {'-'*16}-+-{'-'*16}-+-{'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")
    for r in results:
        m = r["metrics"]
        print(f"  {r['train_on']:<16}{r['test_on']:<16}{r['n_features']:>4}"
              f"{m.get('auroc',0):>9.4f}{m.get('vus_pr',0):>9.4f}{m.get('top_1',0):>7.1f}")

    # ---- 计算掉点 ----
    by_pair = {(r["train_on"], r["test_on"]): r["auroc"] for r in results}
    print(f"\n{'-'*70}")
    print("  掉点分析（域内 - 跨域）")
    print(f"{'-'*70}")
    drops = []
    for (tr, te), auroc in by_pair.items():
        if tr == te:
            continue
        in_domain = by_pair.get((tr, tr))
        if in_domain is None:
            continue
        drop = in_domain - auroc
        drops.append({"train_on": tr, "test_on": te,
                      "in_domain": in_domain, "cross_domain": auroc, "drop": drop})

    if drops:
        print(f"\n  {'训练集':<16}{'域内':>9}{'跨域':>9}{'掉点':>9}")
        print(f"  {'-'*16}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
        for d in drops:
            print(f"  {d['train_on']:<16}{d['in_domain']:>9.4f}"
                  f"{d['cross_domain']:>9.4f}{d['drop']:>+9.4f}")
        avg_drop = float(np.mean([d["drop"] for d in drops]))
        max_drop = max(d["drop"] for d in drops)
        min_drop = min(d["drop"] for d in drops)
        print(f"\n  平均掉点: {avg_drop:+.4f}   最大: {max_drop:+.4f}   最小: {min_drop:+.4f}")
        print()
        if avg_drop >= 0.15:
            print("  ✅ 判定：掉点 >= 0.15，分布偏移是真问题，因果不变性值得做")
        elif avg_drop < 0.10:
            print("  ⚠️ 判定：掉点 < 0.10，因果不变性不是银弹，需重新找切入点")
        else:
            print("  🟡 判定：边缘（0.10~0.15），推进但需准备备用方案")
    else:
        print("  （需要包含自身测试才能计算掉点，请用 --all-pairs）")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, "crossdomain_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results, "drops": drops,
                 "args": {k: v for k, v in vars(args).items()},
                 "elapsed_seconds": elapsed},
                f, indent=2, ensure_ascii=False, default=str,
            )
        print(f"\n  [保存] {path}")


if __name__ == "__main__":
    main()