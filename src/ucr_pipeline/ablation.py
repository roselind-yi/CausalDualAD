#!/usr/bin/env python3
"""
ablation.py - 消融实验框架

支持 7 种消融配置，每种禁用特定模块，在同一数据集上运行对比：

1. full        : 完整双路径模型（baseline）
2. freq_only   : 仅频域路径（禁用时域重构 + 时域原型匹配）
3. time_only   : 仅时域路径（禁用 FFT 预处理 + 频域原型匹配）
4. no_prototype: 无原型匹配（只用重构误差做异常分数）
5. no_fft      : 无 FFT（用原始窗口做频域替代，不做频域提取）
6. fixed_fusion: 固定权重融合（代替自适应不确定性融合）
7. no_contrast : 无对比/不确定性损失（仅重构损失）

用法（烟雾测试）:
    python -m src.ucr_pipeline.ablation --smoke-test

用法（正式，上云后）:
    python -m src.ucr_pipeline.ablation --ucr-path /data/UCR --dataset-ids 1 2 3 4 5 --device cuda:0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

from .model import DualPathModel
from .config import default_config, merge_config
from .trainer import UCRTrainer
from .evaluation import UCREvaluator
from .data import load_ucr_dataset, preprocess_ucr


# ============================================================
# 消融配置定义
# ============================================================

ABLATION_CONFIGS = {
    "full": {
        "description": "完整双路径模型",
        "model_overrides": {},  # 无修改
    },
    "freq_only": {
        "description": "仅频域路径（禁用时域重构 + 时域原型）",
        "model_overrides": {
            "disable_time_recon": True,
            "disable_prototype": True,
        },
    },
    "time_only": {
        "description": "仅时域路径（禁用 FFT 预处理）",
        "model_overrides": {
            "disable_fft": True,
        },
    },
    "no_prototype": {
        "description": "无原型匹配（只用重构误差）",
        "model_overrides": {
            "disable_prototype": True,
        },
    },
    "no_fft": {
        "description": "无 FFT 频域提取（用原始窗口替代）",
        "model_overrides": {
            "disable_fft": True,
        },
    },
    "fixed_fusion": {
        "description": "固定权重融合（0.5/0.5，代替自适应）",
        "model_overrides": {
            "use_fixed_weights": True,
            "fixed_recon_weight": 0.5,
            "fixed_proto_weight": 0.5,
        },
    },
    "no_contrast": {
        "description": "无不确定性损失（仅重构 + 原型匹配损失）",
        "model_overrides": {
            "disable_uncertainty_loss": True,
        },
    },
}


class AblationDualPathModel(DualPathModel):
    """
    支持消融模式的双路径模型

    在 DualPathModel 基础上增加以下消融开关：
    - disable_fft: 禁用 FFT 频域预处理，用原始窗口替代
    - disable_time_recon: 禁用时域重构路径
    - disable_prototype: 禁用原型匹配路径
    - disable_uncertainty_loss: 禁用不确定性损失
    """

    def __init__(self, config):
        # 合并消融开关到 config
        mp = config["model_params"]
        self.disable_fft = mp.get("disable_fft", False)
        self.disable_time_recon = mp.get("disable_time_recon", False)
        self.disable_prototype = mp.get("disable_prototype", False)
        self.disable_uncertainty_loss = mp.get("disable_uncertainty_loss", False)
        super().__init__(config)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # --- 频域预处理 ---
        if self.disable_fft:
            # 无 FFT：用原始时间序列做"频域替代"
            # 把原始窗口在时间维平均，复制成"伪频域特征"
            avg_feature = x.mean(dim=1, keepdim=True)  # (batch, 1, features)
            freq_feature = avg_feature.expand(-1, seq_len, -1)
            time_feature = x
        else:
            time_feature, freq_feature = self.preprocessing(x)

        # --- 特征融合 ---
        combined_feature = torch.cat([time_feature, freq_feature], dim=2)
        local_feature = self.multi_scale_conv(combined_feature)
        global_feature = self.transformer(local_feature)

        # --- 时域重构路径 ---
        if self.disable_time_recon:
            # 禁用重构：用零值替代，使得重构误差恒为 0
            recon = torch.zeros_like(x)
            recon_error = torch.zeros(batch_size, seq_len, device=x.device)
        else:
            recon = self.reconstruction_path(local_feature)
            recon_error = torch.mean(torch.abs(x - recon), dim=2)

        # --- 原型匹配路径 ---
        if self.disable_prototype:
            # 禁用原型：用零值替代
            prototype_score = torch.zeros(batch_size, seq_len, device=x.device)
        else:
            prototype_score = self.prototype_path(global_feature)

        # --- 不确定性融合 ---
        uncertainty_input = torch.cat(
            [global_feature, recon_error.unsqueeze(2), prototype_score.unsqueeze(2)],
            dim=2,
        )
        uncertainty = self.uncertainty_estimation(uncertainty_input)

        if self.use_fixed_weights:
            weight_recon = torch.ones_like(recon_error) * self.fixed_recon_weight
            weight_prototype = torch.ones_like(prototype_score) * self.fixed_proto_weight
        else:
            recon_unc = uncertainty[:, :, 0]
            proto_unc = uncertainty[:, :, 1]
            exp_recon = torch.exp(-recon_unc / self.temperature)
            exp_proto = torch.exp(-proto_unc / self.temperature)
            denom = exp_recon + exp_proto + 1e-8
            weight_recon = exp_recon / denom
            weight_prototype = exp_proto / denom

        fused_score = weight_recon * recon_error + weight_prototype * prototype_score

        return {
            "reconstruction": recon,
            "recon_error": recon_error,
            "prototype_score": prototype_score,
            "uncertainty": uncertainty,
            "weights": {"recon": weight_recon, "prototype": weight_prototype},
            "fused_score": fused_score,
        }

    def compute_loss(self, output, x):
        recon_loss = F.mse_loss(output["reconstruction"], x) if not self.disable_time_recon else torch.tensor(0.0, device=x.device)
        match_loss = output["prototype_score"].mean() if not self.disable_prototype else torch.tensor(0.0, device=x.device)

        if self.disable_uncertainty_loss:
            uncertainty_loss = torch.tensor(0.0, device=x.device)
        else:
            recon_unc = output["uncertainty"][:, :, 0]
            proto_unc = output["uncertainty"][:, :, 1]
            recon_error = output["recon_error"]
            proto_score = output["prototype_score"]

            # 如果重构或原型被禁用，score_diff 可能全为 0
            score_diff = torch.abs(recon_error - proto_score)
            uncertainty_loss = (
                (score_diff / torch.exp(recon_unc)).mean()
                + (score_diff / torch.exp(proto_unc)).mean()
                + recon_unc.mean()
                + proto_unc.mean()
            )

        total_loss = recon_loss + self.lambda1 * match_loss + self.lambda2 * uncertainty_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "match_loss": match_loss,
            "uncertainty_loss": uncertainty_loss,
        }


import torch.nn.functional as F  # noqa: E402 (needed for compute_loss)


class AblationTrainer(UCRTrainer):
    """消融实验训练器（继承 UCRTrainer，使用 AblationDualPathModel）"""

    def __init__(self, config, device=None):
        model = AblationDualPathModel(config)
        super().__init__(model, device)


def run_ablation_experiment(
    ablation_name,
    ablation_config_entry,
    base_config,
    train_data,
    test_data,
    test_labels,
    device=None,
    epochs=2,
):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"消融实验: {ablation_name} — {ablation_config_entry['description']}")
    print(f"{'='*60}")

    # 合并配置
    config = merge_config(base_config, {
        "model_params": ablation_config_entry["model_overrides"],
        "train_params": {"epochs": epochs},
    })

    # 创建训练器
    trainer = AblationTrainer(config, device=device)

    # 训练
    train_norm, test_norm, scaler = preprocess_ucr(train_data, test_data)
    tp = config["train_params"]
    trainer.train(
        train_norm,
        epochs=tp["epochs"],
        batch_size=tp["batch_size"],
        learning_rate=tp["learning_rate"],
        patience=tp["patience"],
        window_size=config["model_params"]["window_size"],
    )

    # 检测
    dp = config["detect_params"]
    anomaly_scores = trainer.detect_anomalies(
        test_norm,
        window_size=config["model_params"]["window_size"],
        stride=dp["stride"],
        batch_size=dp["batch_size"],
    )

    # 评估
    evaluator = UCREvaluator()
    min_len = min(len(anomaly_scores), len(test_labels))
    metrics = evaluator.evaluate(anomaly_scores[:min_len], test_labels[:min_len])

    # 记录结果
    result = {
        "ablation": ablation_name,
        "description": ablation_config_entry["description"],
        "metrics": metrics,
        "config_overrides": ablation_config_entry["model_overrides"],
    }
    print(f"  结果: AUROC={metrics.get('auroc', 0):.4f}, "
          f"F1={metrics.get('f1', 0):.4f}, "
          f"VUS-ROC={metrics.get('vus_roc', 0):.4f}")

    return result


def run_all_ablations(
    base_config,
    train_data,
    test_data,
    test_labels,
    device=None,
    epochs=2,
    ablation_names=None,
):
    """运行所有消融实验"""
    if ablation_names is None:
        ablation_names = list(ABLATION_CONFIGS.keys())

    results = []
    for name in ablation_names:
        if name not in ABLATION_CONFIGS:
            print(f"[WARN] 未知消融配置: {name}")
            continue
        entry = ABLATION_CONFIGS[name]
        result = run_ablation_experiment(
            name, entry, base_config, train_data, test_data, test_labels,
            device=device, epochs=epochs,
        )
        results.append(result)

    return results


def generate_ablation_table(results):
    """生成消融对比表格"""
    print(f"\n{'='*70}")
    print(f"消融实验对比表")
    print(f"{'='*70}")
    print(f"{'配置':<15} {'AUROC':>8} {'F1':>8} {'VUS-ROC':>8} {'描述'}")
    print(f"{'-'*70}")
    for r in results:
        m = r["metrics"]
        print(f"{r['ablation']:<15} {m.get('auroc', 0):>8.4f} {m.get('f1', 0):>8.4f} "
              f"{m.get('vus_roc', 0):>8.4f} {r['description']}")
    print(f"{'='*70}")

    return results


def save_ablation_results(results, output_dir):
    """保存消融结果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ablation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[保存] 消融结果已保存到 {output_path}")
    return output_path


def run_smoke_test():
    """烟雾测试：用合成数据验证消融框架"""
    print("=" * 60)
    print("消融框架烟雾测试")
    print("=" * 60)

    # 生成合成数据
    np.random.seed(42)
    train_length = 1000
    test_length = 2000

    t = np.linspace(0, 6 * np.pi, train_length)
    train_data = np.sin(t) + np.random.randn(train_length) * 0.1

    t_test = np.linspace(0, 10 * np.pi, test_length)
    test_data = np.sin(t_test) + np.random.randn(test_length) * 0.1
    anomaly_start = int(test_length * 0.4)
    anomaly_end = anomaly_start + int(test_length * 0.1)
    test_data[anomaly_start:anomaly_end] += 2.0
    test_labels = np.zeros(test_length)
    test_labels[anomaly_start:anomaly_end] = 1

    base_config = default_config()
    base_config["model_params"]["window_size"] = 100

    results = run_all_ablations(
        base_config, train_data.astype(np.float32),
        test_data.astype(np.float32), test_labels.astype(np.float32),
        device="cpu", epochs=2,
        ablation_names=["full", "freq_only", "time_only",
                        "no_prototype", "fixed_fusion"],
    )

    generate_ablation_table(results)

    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ablation")
    save_ablation_results(results, output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description="消融实验框架")
    parser.add_argument("--smoke-test", action="store_true", help="用合成数据做烟雾测试")
    parser.add_argument("--ucr-path", type=str, default="", help="UCR 数据路径")
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="数据集 ID 列表")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备 (cuda:0 / cpu)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮次")
    parser.add_argument("--output-dir", type=str, default="results/ablation",
                        help="结果输出目录")
    parser.add_argument("--ablations", type=str, nargs="+", default=None,
                        help="指定消融配置（默认全部）")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    if not args.ucr_path:
        print("[ERROR] 正式运行需要 --ucr-path 参数")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    base_config = default_config()

    ids = args.dataset_ids
    ablation_names = args.ablations or list(ABLATION_CONFIGS.keys())

    all_results = {}
    for ds_id in ids:
        train_data, test_data, test_labels = load_ucr_dataset(ds_id, args.ucr_path)
        if train_data is None:
            print(f"[跳过] UCR_{ds_id}: 数据不存在")
            continue

        results = run_all_ablations(
            base_config, train_data, test_data, test_labels,
            device=device, epochs=args.epochs,
            ablation_names=ablation_names,
        )

        all_results[f"UCR_{ds_id}"] = results
        generate_ablation_table(results)

    # 保存全部结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "all_ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 全部结果保存到 {output_dir}/all_ablation_results.json")


if __name__ == "__main__":
    main()
