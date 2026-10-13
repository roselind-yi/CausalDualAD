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
    """时频双域预处理模块（向量化版）"""

    def __init__(self, freq_components=16, freq_feature_dim=64):
        super().__init__()
        self.freq_components = freq_components
        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_components, 128),
            nn.ReLU(),
            nn.Linear(128, freq_feature_dim),
        )

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, features)
        返回:
            (time_feature, freq_feature)
        """
        batch_size, seq_len, features = x.shape
        time_feature = x

        # ---- 向量化 FFT ----
        # 旧代码：for i in range(batch): for j in range(features): np.fft.fft(...)
        # 新代码：一次 rfft 搞定整个 batch
        # rfft 沿 seq_len 维做，输出 (batch, seq_len//2+1, features)
        fft_result = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(fft_result)  # (batch, freq_bins, features)

        k = min(self.freq_components, amplitude.shape[1])
        # 取前 k 个频率分量，对特征维求平均 → (batch, k)
        freq_features = amplitude[:, :k, :].mean(dim=2)

        # 不足 freq_components 则补零
        if k < self.freq_components:
            pad = torch.zeros(
                batch_size, self.freq_components - k, device=x.device, dtype=x.dtype
            )
            freq_features = torch.cat([freq_features, pad], dim=1)

        # MLP 投影后在时间维上复制对齐
        freq_proj = self.freq_mlp(freq_features)  # (batch, freq_feature_dim)
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(x)


class MultiScalePrototypeMatchingPath(nn.Module):
    """多尺度原型匹配路径（向量化版）"""

    def __init__(self, feature_dim, scales=(1, 2, 4), prototypes_per_scale=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.scales = scales
        self.prototypes_per_scale = prototypes_per_scale

        self.prototypes = nn.ParameterDict()
        for scale in scales:
            self.prototypes[f"scale_{scale}"] = nn.Parameter(
                torch.randn(prototypes_per_scale, feature_dim)
            )
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, feature_dim)
        返回:
            (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        normalized_weights = F.softmax(self.scale_weights, dim=0)

        # ---- 向量化原型匹配 ----
        # 旧代码：for t in range(seq_len): F.pairwise_distance(x[:,t,:], prototypes)
        # 新代码：cdst 一次性算完所有时间步
        x_flat = x.reshape(batch_size * seq_len, self.feature_dim)  # (B*S, D)

        total_score = torch.zeros(batch_size, seq_len, device=x.device, dtype=x.dtype)
        for i, scale in enumerate(self.scales):
            prototypes = self.prototypes[f"scale_{scale}"]  # (num_proto, D)
            # cdist: (B*S, num_proto)
            distances = torch.cdist(x_flat, prototypes, p=2.0)
            min_distance = distances.min(dim=1)[0]  # (B*S,)
            scale_scores = min_distance.reshape(batch_size, seq_len)
            total_score = total_score + scale_scores * normalized_weights[i]

        return total_score


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
    """双路径时间序列异常检测模型（修复版）"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        mp = config["model_params"]
        self.window_size = mp["window_size"]
        self.hidden_dim = mp["hidden_dim"]
        self.freq_components = mp["freq_components"]
        self.temperature = mp["temperature"]
        self.lambda1 = mp["lambda1"]
        self.lambda2 = mp["lambda2"]
        # 默认走自适应权重，不再硬编码 UCR_5 的 0.2/0.8
        self.use_fixed_weights = mp.get("use_fixed_weights", False)
        # 如果要固定，允许配置权重比例（而非写死 0.2/0.8）
        self.fixed_recon_weight = mp.get("fixed_recon_weight", 0.5)
        self.fixed_proto_weight = mp.get("fixed_proto_weight", 0.5)

        self.preprocessing = TimeFrequencyPreprocessing(
            freq_components=self.freq_components, freq_feature_dim=64
        )
        self.multi_scale_conv = MultiScaleConv(
            in_channels=1 + 64, out_channels=self.hidden_dim
        )
        self.transformer = TransformerEncoder(
            d_model=self.hidden_dim,
            nhead=mp["num_heads"],
            num_layers=mp["num_layers"],
        )
        self.reconstruction_path = TemporalReconstructionPath(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=1,
        )
        self.prototype_path = MultiScalePrototypeMatchingPath(
            feature_dim=self.hidden_dim, scales=(1, 2, 4), prototypes_per_scale=8
        )
        self.uncertainty_estimation = UncertaintyEstimation(
            input_dim=self.hidden_dim + 2
        )

    def forward(self, x):
        """
        参数:
            x: (batch, window_size, features)
        返回:
            dict: 模型输出
        """
        time_feature, freq_feature = self.preprocessing(x)
        combined_feature = torch.cat([time_feature, freq_feature], dim=2)
        local_feature = self.multi_scale_conv(combined_feature)
        global_feature = self.transformer(local_feature)

        recon = self.reconstruction_path(local_feature)  # (batch, seq_len, 1)
        recon_error = torch.mean(torch.abs(x - recon), dim=2)  # (batch, seq_len)

        prototype_score = self.prototype_path(global_feature)  # (batch, seq_len)

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
        recon_loss = F.mse_loss(output["reconstruction"], x)
        match_loss = output["prototype_score"].mean()

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
    """UCR 数据集（向量化滑窗）"""

    def __init__(self, data, window_size, slide_step=1):
        self.data = np.asarray(data, dtype=np.float32)
        self.window_size = window_size
        self.slide_step = slide_step
        self.indices = np.arange(
            0, len(self.data) - window_size + 1, slide_step
        ).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start : start + self.window_size]
        return torch.from_numpy(window).float().unsqueeze(-1)


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
