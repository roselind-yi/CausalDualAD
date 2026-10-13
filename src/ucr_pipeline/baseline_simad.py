#!/usr/bin/env python3
"""
baseline_simad.py - SimAD 基线对比框架

功能：
1. 在相同 UCR 数据上运行 SimAD 基线模型
2. 收集 SimAD 的异常检测结果
3. 用相同评估指标（AUROC/F1/VUS-ROC）评估 SimAD
4. 与本模型的结果做对比表

用法（烟雾测试）:
    python -m src.ucr_pipeline.baseline_simad --smoke-test

用法（正式，上云后）:
    python -m src.ucr_pipeline.baseline_simad --ucr-path /data/UCR --dataset-ids 1 2 3 --device cuda:0

SimAD 原始代码在 SimAD/ 目录下，本框架做了以下适配：
- 统一设备管理（不再硬编码集群路径）
- 简化 SimAD 的数据加载（直接用 .npy 格式）
- 统一评估指标（与本模型相同）
- 去掉分布式训练依赖（单卡即可）
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .data import load_ucr_dataset, preprocess_ucr
from .evaluation import UCREvaluator
from .config import default_config


# ============================================================
# SimAD 模型（轻量适配版，独立于 SimAD/ 原始代码）
# ============================================================

class SimADRevIN(nn.Module):
    """可逆归一化（从 SimAD 移植）"""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x


class SimADPositionalEmbedding(nn.Module):
    """位置编码（从 SimAD 移植，简化版）"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.arange(0, max_len).float() / max_len
        pe = torch.sin(pe)
        pe = pe.unsqueeze(0).unsqueeze(-1)
        self.register_buffer("pe", pe)
        self.d_model = d_model

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SimADTokenEmbedding(nn.Module):
    """Token 嵌入（1D 卷积）"""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=1, padding_mode="circular", bias=False,
        )
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        # x: (batch, seq_len, c_in) → Conv1d expects (batch, c_in, seq_len)
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class SimADAttention(nn.Module):
    """SimAD 注意力模块（简化版）"""

    def __init__(self, d_model, n_head, max_len=5000, seq_len=128, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.v = nn.Embedding(max_len, d_model)
        self.v_proj = nn.Linear(max_len, seq_len)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

    def forward(self, x):
        B, N, D = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v.weight.unsqueeze(0).expand(B, -1, -1)
        v = self.v_proj(v.transpose(1, 2)).transpose(1, 2)
        if v.shape[1] != N:
            v = v[:, :N, :]

        # 多头拆分
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, v.shape[1], self.n_head, D // self.n_head).transpose(1, 2) if v.shape[2] == D else v

        attn = torch.matmul(q, k.transpose(-1, -2)) / (D // self.n_head ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class SimADEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, seq_len, d_feedforward=None, dropout=0.1):
        super().__init__()
        d_feedforward = d_feedforward or d_model * 4
        self.attn = SimADAttention(d_model, n_head, seq_len=seq_len, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward), nn.GELU(), nn.Linear(d_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class SimADModel(nn.Module):
    """
    SimAD 基线模型（轻量适配版）

    核心架构：
    1. RevIN 归一化
    2. 位置编码 + Patch 分块
    3. Transformer 编码器
    4. 重构输出 + 相似度输出
    5. 异常分数 = MSE 重构误差 + 余弦相似度误差
    """

    def __init__(self, c_dim=1, seq_len=100, patch_size=10, d_model=64,
                 n_head=4, n_layers=3, dropout=0.1, proj_dim=10):
        super().__init__()
        self.c_dim = c_dim
        self.patch_size = patch_size
        self.seq_len = seq_len
        assert seq_len % patch_size == 0, f"seq_len={seq_len} 必须能被 patch_size={patch_size} 整除"
        self.num_patches = seq_len // patch_size

        # 归一化 + 位置编码
        self.revin = SimADRevIN(c_dim)
        self.pos_enc = SimADPositionalEmbedding(d_model, max_len=seq_len)

        # Patch 嵌入
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_size * c_dim),
            nn.Linear(patch_size * c_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 编码器
        self.encoder = nn.Sequential(*[
            SimADEncoderLayer(d_model, n_head, seq_len=self.num_patches, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 输出投影
        self.recon_proj = nn.Linear(d_model, patch_size * c_dim)
        self.sim_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, patch_size * proj_dim),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, c_dim)
        返回: anomaly_score (batch, seq_len)
        """
        # RevIN
        x = self.revin(x, mode="norm")

        # Patch 分块: (batch, seq_len, c_dim) → (batch, num_patches, patch_size * c_dim)
        B, L, C = x.shape
        patches = x.reshape(B, self.num_patches, self.patch_size * C)
        patches = self.patch_embed(patches)  # (batch, num_patches, d_model)

        # 编码器
        encoded = self.encoder(patches)

        # 重构输出
        recon_patches = self.recon_proj(encoded)  # (batch, num_patches, patch_size * C)
        recon = recon_patches.reshape(B, L, C)

        # 计算异常分数
        # MSE 重构误差
        mse_score = nn.MSELoss(reduction="none")(recon, x).mean(dim=2)  # (batch, seq_len)

        # 余弦相似度误差
        cos_score = 1 - nn.CosineSimilarity(dim=2)(recon, x)  # (batch, seq_len)

        anomaly_score = mse_score + cos_score

        return anomaly_score


class SimADUCRDataset(Dataset):
    """UCR 数据集（SimAD 格式）"""

    def __init__(self, data, labels=None, win_size=100, step=1, mode="train"):
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim == 1:
            self.data = self.data[:, np.newaxis]
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None else None
        if self.labels is not None and self.labels.ndim == 1:
            self.labels = self.labels[:, np.newaxis]
        self.win_size = win_size
        self.step = step
        self.mode = mode
        self.n_samples = (len(self.data) - win_size) // step + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.step
        window = self.data[start:start + self.win_size]
        if self.labels is not None:
            label_window = self.labels[start:start + self.win_size]
            return torch.from_numpy(window), torch.from_numpy(label_window)
        return torch.from_numpy(window), torch.zeros(self.win_size, 1)


class SimADTrainer:
    """SimAD 训练器（轻量适配版）"""

    def __init__(self, config, device=None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)

        model_params = config.get("simad_params", {})
        seq_len = config["model_params"]["window_size"]
        c_dim = 1  # UCR 是单变量

        # patch_size 必须整除 seq_len
        patch_size = model_params.get("patch_size", min(10, seq_len // 4))
        while seq_len % patch_size != 0 and patch_size > 1:
            patch_size -= 1

        self.model = SimADModel(
            c_dim=c_dim, seq_len=seq_len, patch_size=patch_size,
            d_model=model_params.get("d_model", 64),
            n_head=model_params.get("n_head", 4),
            n_layers=model_params.get("n_layers", 3),
            dropout=model_params.get("dropout", 0.1),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["train_params"]["learning_rate"],
            weight_decay=config["train_params"]["weight_decay"],
        )

        self.win_size = seq_len
        self.batch_size = config["train_params"]["batch_size"]
        self.epochs = config["train_params"]["epochs"]

    def train(self, train_data):
        """训练 SimAD"""
        dataset = SimADUCRDataset(train_data, win_size=self.win_size, mode="train")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        mse_loss = nn.MSELoss()
        cos_loss = nn.CosineSimilarity(dim=2)

        print(f"[SimAD 训练] 开始, {self.epochs} epochs, device={self.device}")
        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)

                # 原始输入
                score = self.model(batch_x)

                # 噪声增强输入
                noise = torch.randn_like(batch_x) * 0.1
                x2 = batch_x + noise
                score2 = self.model(x2)

                # 重构损失
                recon_loss = mse_loss(score, score2.detach())

                # 对比损失（相似度一致性）
                sim_loss = torch.mean(1 - cos_loss(
                    score.unsqueeze(2) if score.dim() == 2 else score,
                    score2.unsqueeze(2) if score2.dim() == 2 else score2,
                ))

                loss = recon_loss + 0.1 * sim_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}, time={elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"[SimAD 训练] 完成, {total_time:.1f}s")

    def detect_anomalies(self, test_data, batch_size=256):
        """用 SimAD 检测异常"""
        dataset = SimADUCRDataset(test_data, win_size=self.win_size, mode="test")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                scores = self.model(batch_x)  # (batch, seq_len)
                all_scores.append(scores.cpu().numpy())

        if not all_scores:
            return np.array([])

        scores = np.concatenate(all_scores, axis=0)  # (n_windows, seq_len)

        # 滑窗分数映射到点级别
        n_windows = scores.shape[0]
        win_size = scores.shape[1]
        n_points = n_windows + win_size - 1
        point_scores = np.zeros(n_points, dtype=np.float32)
        point_counts = np.zeros(n_points, dtype=np.float32)

        for i in range(n_windows):
            point_scores[i:i + win_size] += scores[i]
            point_counts[i:i + win_size] += 1

        point_scores /= np.maximum(point_counts, 1)

        # 截取到测试序列实际长度
        actual_len = len(test_data) if hasattr(test_data, '__len__') else len(np.asarray(test_data))
        if len(point_scores) > actual_len:
            point_scores = point_scores[:actual_len]
        elif len(point_scores) < actual_len:
            point_scores = np.pad(point_scores, (0, actual_len - len(point_scores)), mode='edge')

        return point_scores


def run_simad_baseline(
    config,
    train_data,
    test_data,
    test_labels,
    device=None,
    epochs=None,
):
    """运行 SimAD 基线实验"""
    print(f"\n{'='*60}")
    print(f"SimAD 基线实验")
    print(f"{'='*60}")

    if epochs is not None:
        config["train_params"]["epochs"] = epochs

    trainer = SimADTrainer(config, device=device)

    # 数据预处理（与本模型相同的方式）
    train_norm, test_norm, scaler = preprocess_ucr(train_data, test_data)

    # 训练
    trainer.train(train_norm)

    # 检测
    anomaly_scores = trainer.detect_anomalies(test_norm)

    # 评估
    evaluator = UCREvaluator()
    metrics = evaluator.evaluate(anomaly_scores, test_labels)

    result = {
        "model": "SimAD",
        "metrics": metrics,
    }
    print(f"  SimAD 结果: AUROC={metrics.get('auroc', 0):.4f}, "
          f"F1={metrics.get('f1', 0):.4f}, "
          f"VUS-ROC={metrics.get('vus_roc', 0):.4f}")

    return result


def run_comparison(
    config,
    train_data,
    test_data,
    test_labels,
    device=None,
    epochs=None,
):
    """运行本模型 + SimAD 对比"""
    from .trainer import UCRTrainer

    # --- 本模型 ---
    print(f"\n{'='*60}")
    print(f"DualPathAD（本模型）实验")
    print(f"{'='*60}")

    from .model import DualPathModel
    our_model = DualPathModel(config)
    our_trainer = UCRTrainer(our_model, device=device)
    train_norm, test_norm, scaler = preprocess_ucr(train_data, test_data)
    tp = config["train_params"]
    our_trainer.train(
        train_norm,
        epochs=tp["epochs"],
        batch_size=tp["batch_size"],
        learning_rate=tp["learning_rate"],
        patience=tp["patience"],
        window_size=config["model_params"]["window_size"],
    )
    dp = config["detect_params"]
    our_scores = our_trainer.detect_anomalies(
        test_norm,
        window_size=config["model_params"]["window_size"],
        stride=dp["stride"],
        batch_size=dp["batch_size"],
    )

    evaluator = UCREvaluator()
    min_len = min(len(our_scores), len(test_labels))
    our_metrics = evaluator.evaluate(our_scores[:min_len], test_labels[:min_len])
    our_result = {"model": "DualPathAD", "metrics": our_metrics}
    print(f"  DualPathAD 结果: AUROC={our_metrics.get('auroc', 0):.4f}, "
          f"F1={our_metrics.get('f1', 0):.4f}, "
          f"VUS-ROC={our_metrics.get('vus_roc', 0):.4f}")

    # --- SimAD ---
    simad_result = run_simad_baseline(
        config, train_data, test_data, test_labels,
        device=device, epochs=epochs,
    )

    # --- 对比表 ---
    print(f"\n{'='*70}")
    print(f"模型对比表")
    print(f"{'='*70}")
    print(f"{'模型':<15} {'AUROC':>8} {'F1':>8} {'VUS-ROC':>8} {'Precision':>10} {'Recall':>8}")
    print(f"{'-'*70}")
    for r in [our_result, simad_result]:
        m = r["metrics"]
        print(f"{r['model']:<15} {m.get('auroc', 0):>8.4f} {m.get('f1', 0):>8.4f} "
              f"{m.get('vus_roc', 0):>8.4f} {m.get('precision', 0):>10.4f} "
              f"{m.get('recall', 0):>8.4f}")
    print(f"{'='*70}")

    return [our_result, simad_result]


def save_comparison_results(results, output_dir, dataset_id=None):
    """保存对比结果"""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"comparison_UCR_{dataset_id}.json" if dataset_id else "comparison_results.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[保存] 对比结果已保存到 {path}")
    return path


def run_smoke_test():
    """烟雾测试：用合成数据验证基线框架"""
    print("=" * 60)
    print("SimAD 基线对比框架烟雾测试")
    print("=" * 60)

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

    config = default_config()
    config["model_params"]["window_size"] = 100
    config["train_params"]["epochs"] = 2

    results = run_comparison(
        config, train_data.astype(np.float32),
        test_data.astype(np.float32), test_labels.astype(np.float32),
        device="cpu", epochs=2,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "baseline")
    save_comparison_results(results, output_dir, dataset_id="smoke")

    return results


def main():
    parser = argparse.ArgumentParser(description="SimAD 基线对比框架")
    parser.add_argument("--smoke-test", action="store_true", help="用合成数据做烟雾测试")
    parser.add_argument("--ucr-path", type=str, default="", help="UCR 数据路径")
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="数据集 ID 列表")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮次")
    parser.add_argument("--output-dir", type=str, default="results/baseline",
                        help="结果输出目录")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    if not args.ucr_path:
        print("[ERROR] 正式运行需要 --ucr-path 参数")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    config = default_config()

    all_results = {}
    for ds_id in args.dataset_ids:
        train_data, test_data, test_labels = load_ucr_dataset(ds_id, args.ucr_path)
        if train_data is None:
            print(f"[跳过] UCR_{ds_id}: 数据不存在")
            continue

        results = run_comparison(
            config, train_data, test_data, test_labels,
            device=device, epochs=args.epochs,
        )
        all_results[f"UCR_{ds_id}"] = results
        save_comparison_results(results, args.output_dir, dataset_id=ds_id)

    # 保存汇总
    with open(os.path.join(args.output_dir, "all_comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 全部结果保存到 {args.output_dir}/all_comparison_results.json")


if __name__ == "__main__":
    main()
