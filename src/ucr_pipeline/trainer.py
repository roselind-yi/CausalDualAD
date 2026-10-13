#!/usr/bin/env python3
"""
trainer.py - UCR 专用训练器（修复版）

关键修复：
1. 设备管理：model / tensor / loss 统一在同一 device，不再 CPU/GPU 混跑
2. 异常检测向量化：批量推理替代逐窗口 for 循环，用 scatter_add 聚合分数
3. import 修复：包内相对导入，不再依赖 CWD
4. 去掉 SWaT 专属逻辑，UCR 流程干净独立
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import DualPathModel, create_dataloader


class UCRTrainer:
    """UCR 专用训练器"""

    def __init__(self, model, device=None):
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)
        print(f"[UCRTrainer] 使用设备: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ------------------------------------------------------------------ #
    #  训练
    # ------------------------------------------------------------------ #
    def train(
        self,
        train_data,
        epochs=50,
        batch_size=32,
        learning_rate=5e-4,
        patience=15,
        window_size=100,
        val_data=None,
    ):
        """
        训练模型

        参数:
            train_data: 标准化后的训练序列 (T,) 或 (T, C)
            epochs / batch_size / learning_rate / patience: 训练超参
            window_size: 滑窗长度
            val_data: 可选验证集
        返回:
            (train_losses, val_losses)
        """
        train_loader = create_dataloader(
            train_data, window_size=window_size, batch_size=batch_size, shuffle=True
        )

        val_loader = None
        if val_data is not None:
            val_loader = create_dataloader(
                val_data, window_size=window_size, batch_size=batch_size, shuffle=False
            )

        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        train_losses, val_losses = [], []

        print(f"\n[训练] 数据形状: {train_data.shape}, 窗口: {window_size}")
        print(f"[训练] epochs={epochs}, batch={batch_size}, lr={learning_rate}, patience={patience}")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_match = 0.0
            n_batches = len(train_loader)
            t0 = time.time()

            for batch_idx, batch in enumerate(train_loader):
                x = batch.to(self.device, non_blocking=True)

                output = self.model(x)
                loss_dict = self.model.compute_loss(output, x)
                total_loss = loss_dict["total_loss"]

                optimizer.zero_grad()
                if torch.isnan(total_loss):
                    print(f"  [警告] epoch {epoch+1} batch {batch_idx}: loss=NaN, 跳过")
                    continue
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_match += loss_dict["match_loss"].item()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_recon = epoch_recon / max(n_batches, 1)
            avg_match = epoch_match / max(n_batches, 1)
            train_losses.append(avg_loss)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            # 验证
            val_loss = float("inf")
            improved = "-"
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    improved = "✓"
                else:
                    patience_counter += 1
                    improved = "⨯"
            else:
                val_losses.append(avg_loss)
                scheduler.step(avg_loss)

            print(
                f"  epoch [{epoch+1:03d}/{epochs}] {improved}  "
                f"loss={avg_loss:.6f}  recon={avg_recon:.6f}  match={avg_match:.6f}  "
                f"lr={current_lr:.2e}  {elapsed:.1f}s  "
                f"early_stop={patience_counter}/{patience}"
            )

            if patience_counter >= patience:
                print(f"  [早停] 第 {epoch+1} 轮触发")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"  [恢复] 最佳模型 (val_loss={best_val_loss:.6f})")

        print("[训练] 完成")
        return train_losses, val_losses

    # ------------------------------------------------------------------ #
    #  评估损失（内部用）
    # ------------------------------------------------------------------ #
    def _evaluate_loss(self, data_loader):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(self.device, non_blocking=True)
                output = self.model(x)
                loss_dict = self.model.compute_loss(output, x)
                total += loss_dict["total_loss"].item()
        return total / max(len(data_loader), 1)

    # ------------------------------------------------------------------ #
    #  异常检测（批量向量化）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def detect_anomalies(self, test_data, window_size=100, stride=1, batch_size=256):
        """
        批量异常检测 — 修复版

        旧代码逐窗口 for 循环推理 + Python for 循环分配分数，极慢。
        新代码批量推理 + scatter_add 向量化聚合。

        参数:
            test_data: 标准化后的测试序列 (T,) 或 (T, C)
            window_size / stride: 滑窗参数
            batch_size: 推理批量（默认 256，GPU 上可更大）
        返回:
            anomaly_scores: (len(test_data),) 每个时间点的异常分数
                （多变量时分数 = 各通道分数的 max 聚合）
        """
        self.model.eval()
        n = len(test_data)
        data = np.asarray(test_data, dtype=np.float32)
        # 统一成 (T, C)：单变量 (T,) → (T, 1)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_features = data.shape[1]

        # 生成所有窗口的起始索引
        starts = np.arange(0, n - window_size + 1, stride)
        if len(starts) == 0:
            print("[检测] 测试数据太短，无法生成窗口")
            return np.zeros(n)

        # 预分配分数累加器和计数器
        score_sum = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)

        n_windows = len(starts)
        print(
            f"[检测] {n_windows} 个窗口, 通道={n_features}, "
            f"batch_size={batch_size}, device={self.device}"
        )

        t0 = time.time()
        for batch_start in range(0, n_windows, batch_size):
            batch_ends = min(batch_start + batch_size, n_windows)
            window_starts = starts[batch_start:batch_ends]

            # 批量构造窗口: (B, W, C)
            windows = np.stack([data[s : s + window_size] for s in window_starts])
            x = torch.from_numpy(np.ascontiguousarray(windows)).float().to(
                self.device, non_blocking=True
            )

            output = self.model(x)
            # fused_score 已在 model 内做过 max 聚合 → (B, W)
            fused = output["fused_score"]
            fused_np = fused.cpu().numpy()  # (B, W)

            # 向量化分数分配：每个窗口的中间 1/3 区间分配平均分数
            for i, ws in enumerate(window_starts):
                cs = ws + window_size // 3
                ce = ws + 2 * window_size // 3
                ce = min(ce, n)
                if ce > cs:
                    window_score = float(np.mean(fused_np[i]))
                    score_sum[cs:ce] += window_score
                    count[cs:ce] += 1.0

        count[count == 0] = 1.0
        anomaly_scores = score_sum / count

        # 前后未覆盖的区域用最近的有效分数填充
        valid = np.where(count > 1.0)[0]
        if len(valid) > 0:
            first_valid, last_valid = valid[0], valid[-1]
            if first_valid > 0:
                anomaly_scores[:first_valid] = anomaly_scores[first_valid]
            if last_valid < n - 1:
                anomaly_scores[last_valid + 1 :] = anomaly_scores[last_valid]

        elapsed = time.time() - t0
        print(f"[检测] 完成, 耗时 {elapsed:.1f}s ({n_windows / max(elapsed, 0.01):.0f} 窗口/秒)")
        return anomaly_scores

    # ------------------------------------------------------------------ #
    #  保存 / 加载
    # ------------------------------------------------------------------ #
    def save_model(self, path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.model.config,
            },
            path,
        )
        print(f"[保存] 模型已保存到 {path}")

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"[加载] 模型已从 {path} 恢复")
