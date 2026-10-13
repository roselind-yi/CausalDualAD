#!/usr/bin/env python3
"""
model.py - 双路径时间序列异常检测模型（向量化修复版）

关键修复：
1. FFT 预处理：双重 Python for 循环 + .cpu().numpy() → 单次 torch.fft.rfft 向量化
2. 原型匹配：seq_len for 循环逐时间步算距离 → torch.cdist 一次性算完
3. 设备管理：forward 内不再强制 CPU，全程留在 model 所在设备
4. 权重策略：use_fixed_weights 参数化，默认走自适应（去掉 UCR_5 硬编码 0.2/0.8）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TimeFrequencyPreprocessing(nn.Module):
    """时频双域预处理模块（向量化版，支持多变量）

    多变量处理：FFT 沿时间维做，保留通道维
    → (B, freq_bins, C) → 取前 k 个频率分量 → 展平 (B, k*C) → MLP 投影
    这样 freq_feature_dim 与通道数解耦，下游结构不变。
    单变量（C=1）时与旧实现数值完全一致。
    """

    def __init__(self, freq_components=16, freq_feature_dim=64, n_features=1):
        super().__init__()
        self.freq_components = freq_components
        self.n_features = n_features
        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_components * n_features, 128),
            nn.ReLU(),
            nn.Linear(128, freq_feature_dim),
        )

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, C)
        返回:
            (time_feature, freq_feature)，两者都是 (batch, seq_len, dim)
        """
        batch_size, seq_len, features = x.shape
        time_feature = x

        # ---- 向量化 FFT ----
        # 沿时间维做，保留通道维
        fft_result = torch.fft.rfft(x, dim=1)  # (B, freq_bins, C)
        amplitude = torch.abs(fft_result)

        k = min(self.freq_components, amplitude.shape[1])
        # 取前 k 个频率分量，保留通道维 → (B, k, C)
        freq_features = amplitude[:, :k, :]

        # 不足 freq_components 则在频率维补零
        if k < self.freq_components:
            pad = torch.zeros(
                batch_size,
                self.freq_components - k,
                features,
                device=x.device,
                dtype=x.dtype,
            )
            freq_features = torch.cat([freq_features, pad], dim=1)

        # (B, k, C) → (B, k*C) → MLP → (B, freq_feature_dim)
        freq_flat = freq_features.reshape(batch_size, -1)
        freq_proj = self.freq_mlp(freq_flat)
        freq_feature = freq_proj.unsqueeze(1).expand(-1, seq_len, -1)

        return time_feature, freq_feature


class MultiScaleConv(nn.Module):
    """多尺度卷积模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        conv_channels = out_channels // 3
        remainder = out_channels % 3
        conv3_channels = conv_channels + (1 if remainder >= 1 else 0)
        conv5_channels = conv_channels + (1 if remainder >= 2 else 0)
        conv7_channels = conv_channels

        self.conv3 = nn.Conv1d(in_channels, conv3_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, conv5_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, conv7_channels, kernel_size=7, padding=3)

        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = None

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x_t = x.transpose(1, 2)
        out = torch.cat([self.conv3(x_t), self.conv5(x_t), self.conv7(x_t)], dim=1)
        out = out.transpose(1, 2)  # 回到 (batch, seq_len, out_channels)

        if self.residual is not None:
            out = out + self.residual(x)
        else:
            out = out + x
        return F.relu(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — batch_first 风格
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        # x: (batch, seq_len, d_model) — batch_first，不需要转置了
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)


class TemporalReconstructionPath(nn.Module):
    """时序重构路径

    多变量支持：权重在最后一维共享，对每个通道独立重构。
    输入 (..., output_dim) → 输出同形状。
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (B, W, hidden) → (B, W, output_dim)
        # 权重沿最后一维共享，天然对每个通道独立生效
        return self.decoder(x)


class MultiScalePrototypeMatchingPath(nn.Module):
    """多尺度原型匹配路径（逐通道，支持任意通道数）

    语义：每个通道有自己的一组"正常原型值"，通道 c 在时刻 t 的分数
    = 该通道当前值到该通道最近原型的距离。

    因此原型参数形状为 (n_features, prototypes_per_scale)，
    输入 (..., n_features) → 输出同形状，每个通道独立打分。
    """

    def __init__(self, n_features, scales=(1, 2, 4), prototypes_per_scale=8):
        super().__init__()
        self.n_features = n_features
        self.scales = scales
        self.prototypes_per_scale = prototypes_per_scale

        # 每个通道一套原型，初始在 [0,1] 区间随机（重构误差经Tanh 后大致落在此范围）
        self.prototypes = nn.ParameterDict()
        for scale in scales:
            self.prototypes[f"scale_{scale}"] = nn.Parameter(
                torch.rand(n_features, prototypes_per_scale)
            )
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))

    def forward(self, x):
        """
        参数:
            x: (..., n_features)，最后一维是通道维
        返回:
            与 x 同形状，每个元素是该通道到最近原型的加权距离
        """
        lead_shape = x.shape[:-1]
        normalized_weights = F.softmax(self.scale_weights, dim=0)

        # (..., C) -> (N, C)
        x_flat = x.reshape(-1, self.n_features)  # (N, C)

        total_score = torch.zeros_like(x_flat)
        for i, scale in enumerate(self.scales):
            # prototypes: (C, P) -> (1, C, P)，与 (N, C, 1) 广播
            protos = self.prototypes[f"scale_{scale}"].unsqueeze(0)  # (1, C, P)
            # (N, C, 1) - (1, C, P) -> (N, C, P)
            # 第2 维是 C（逐通道），第 3 维是 P（原型）
            # 距离定义：通道 c 的值到该通道 P 个原型的 L1 距离
            diff = (x_flat.unsqueeze(-1) - protos).abs()   # (N, C, P)
            # 先在 P 维取 min → 每通道一个最近原型距离 → (N, C)
            min_dist = diff.min(dim=2)[0]
            total_score = total_score + min_dist * normalized_weights[i]

        return total_score.reshape(*lead_shape, self.n_features)


class UncertaintyEstimation(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return F.softplus(self.uncertainty_mlp(x))


class DualPathModel(nn.Module):
    """双路径时间序列异常检测模型（支持单变量与多变量）

    输入:
        单变量: (B, W, 1)
        多变量: (B, W, C)

    内部两条路径都在通道维输出分数，最终用 max 聚合到 (B, W)：
        重构路径: (B, W, C)  每个通道一个重构误差
        原型路径: (B, W, C)  每个通道一个原型距离
    max 聚合的语义：任一通道异常 → 该时间点异常。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        mp = config["model_params"]
        self.window_size = mp["window_size"]
        self.hidden_dim = mp["hidden_dim"]
        self.freq_components = mp["freq_components"]
        self.freq_feature_dim = mp.get("freq_feature_dim", 64)
        self.temperature = mp["temperature"]
        self.lambda1 = mp["lambda1"]
        self.lambda2 = mp["lambda2"]
        # 输入通道数，1 = 单变量（UCR），>1 = 多变量
        self.n_features = mp.get("n_features", 1)
        # 默认走自适应权重，不再硬编码 UCR_5 的 0.2/0.8
        self.use_fixed_weights = mp.get("use_fixed_weights", False)
        # 如果要固定，允许配置权重比例（而非写死 0.2/0.8）
        self.fixed_recon_weight = mp.get("fixed_recon_weight", 0.5)
        self.fixed_proto_weight = mp.get("fixed_proto_weight", 0.5)
        # 通道分数聚合方式：max（任一通道异常即异常）/ mean（平均，抑制单通道噪声）
        # 可选 topk_mean：取分数最高的 k 个通道求平均，兼顾两者
        self.aggregate = mp.get("aggregate", "max")
        self.aggregate_topk = mp.get("aggregate_topk", 3)
        assert self.aggregate in ("max", "mean", "topk_mean"), \
            f"aggregate 必须是 max / mean / topk_mean，当前为 {self.aggregate}"

        self.preprocessing = TimeFrequencyPreprocessing(
            freq_components=self.freq_components,
            freq_feature_dim=self.freq_feature_dim,
            n_features=self.n_features,
        )
        # 输入 =原始 C 通道 + 频域投影 freq_feature_dim 通道
        self.multi_scale_conv = MultiScaleConv(
            in_channels=self.n_features + self.freq_feature_dim,
            out_channels=self.hidden_dim,
        )
        self.transformer = TransformerEncoder(
            d_model=self.hidden_dim,
            nhead=mp["num_heads"],
            num_layers=mp["num_layers"],
        )
        # 重构路径：输出 C 通道（逐通道独立重构，权重沿最后一维共享）
        self.reconstruction_path = TemporalReconstructionPath(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.n_features,
        )
        self.prototype_path = MultiScalePrototypeMatchingPath(
            n_features=self.n_features, scales=(1, 2, 4), prototypes_per_scale=8
        )
        self.uncertainty_estimation = UncertaintyEstimation(
            input_dim=self.hidden_dim + 2
        )

    def forward(self, x):
        """
        参数:
            x: (B, W, C)
        返回:
            dict: 各路径分数，融合后 fused_score 为 (B, W)
        """
        time_feature, freq_feature = self.preprocessing(x)   # (B,W,C), (B,W,F)
        combined_feature = torch.cat([time_feature, freq_feature], dim=2)
        local_feature = self.multi_scale_conv(combined_feature)    # (B,W,H)
        global_feature = self.transformer(local_feature)          # (B,W,H)

        # ---- 重构路径：逐通道重构误差 (B, W, C) ----
        recon = self.reconstruction_path(local_feature)           # (B,W,C)
        recon_error_mv = torch.abs(x - recon)                     # (B,W,C)

        # ---- 原型路径：逐通道原型距离 (B, W) ----
        # 原型匹配作用在重构误差特征上：语义是"这个点离正常原型有多远"，
        # 重构误差恰好是该语义的直接度量
        prototype_score_mv = self.prototype_path(recon_error_mv)     # (B,W,C)

        # ---- 通道分数聚合 (B,W,C) -> (B,W) ----
        if self.n_features == 1:
            prototype_score = prototype_score_mv.squeeze(2)          # (B,W)
            recon_error = recon_error_mv.squeeze(2)                  # (B,W)
        else:
            prototype_score = self._aggregate(prototype_score_mv)     # (B,W)
            recon_error = self._aggregate(recon_error_mv)            # (B,W)

        uncertainty_input = torch.cat(
            [global_feature, recon_error.unsqueeze(2), prototype_score.unsqueeze(2)],
            dim=2,
        )
        uncertainty = self.uncertainty_estimation(uncertainty_input)  # (B,W,2)

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
            "recon_error_mv": recon_error_mv,
            "prototype_score_mv": prototype_score_mv,
            "recon_error": recon_error,
            "prototype_score": prototype_score,
            "uncertainty": uncertainty,
            "weights": {"recon": weight_recon, "prototype": weight_prototype},
            "fused_score": fused_score,
        }

    def _aggregate(self, scores_mv):
        """
        通道分数聚合：(B, W, C) -> (B, W)

        三种策略：
        - max       取最大值。语义"任一通道异常即异常"，但对单通道噪声最敏感
        - mean      取平均值。抑制噪声，但会稀释稀疏异常信号
        - topk_mean 取分数最高的 k 个通道求平均。max 与 mean 的折中

        多变量异常检测中max 最常见，调研假设"SMAP/MSL 的异常分散、
        max 被噪声主导"，此方法用于验证该假设。
        """
        if self.aggregate == "max":
            return scores_mv.max(dim=2)[0]
        if self.aggregate == "mean":
            return scores_mv.mean(dim=2)
        # topk_mean
        k = min(self.aggregate_topk, scores_mv.size(2))
        topk_vals, _ = scores_mv.topk(k, dim=2)
        return topk_vals.mean(dim=2)

    def compute_loss(self, output, x):
        # 重构损失：逐通道
        recon_loss = F.mse_loss(output["reconstruction"], x)
        match_loss = output["prototype_score_mv"].mean()

        recon_unc = output["uncertainty"][:, :, 0]
        proto_unc = output["uncertainty"][:, :, 1]
        recon_error = output["recon_error"]
        proto_score = output["prototype_score"]

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


class UCRDataset(Dataset):
    """时序数据集（支持单变量 (T,) 与多变量 (T, C)）"""

    def __init__(self, data, window_size, slide_step=1):
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim == 1:
            self.data = self.data[:, np.newaxis]  # (T,) → (T, 1)
        self.window_size = window_size
        self.slide_step = slide_step
        self.indices = np.arange(
            0, len(self.data) - window_size + 1, slide_step
        ).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start : start + self.window_size]  # (W, C)
        return torch.from_numpy(np.ascontiguousarray(window)).float()


def create_dataloader(data, window_size, batch_size=64, shuffle=True, device=None):
    dataset = UCRDataset(data, window_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return dataloader


def create_windows_vectorized(data, window_size=100, stride=1):
    """向量化滑窗创建 — 比 for 循环快几十倍"""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_samples = data.shape[0]
    n_windows = (n_samples - window_size) // stride + 1
    if n_windows <= 0:
        return np.array([]).reshape(0, window_size, data.shape[1])
    # stride tricks
    shape = (n_windows, window_size, data.shape[1])
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return np.ascontiguousarray(windows)
