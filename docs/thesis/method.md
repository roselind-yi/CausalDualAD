# 方法

## 3.1 双路径异常检测模型

本文提出的双路径异常检测模型主要由以下组件组成：

1. **时频预处理模块**：提取时间域和频域特征
2. **多尺度卷积模块**：捕捉不同尺度的局部特征
3. **Transformer编码器**：提取全局特征
4. **时序重构路径**：通过重构误差检测异常
5. **多尺度原型匹配路径**：通过与原型的距离检测异常
6. **不确定性估计模块**：动态调整两条路径的权重
7. **分数融合模块**：结合两条路径的异常分数

模型的整体架构如图3-1所示：

![双路径异常检测模型架构](figures/model_architecture.png)

### 3.1.1 时频预处理模块

时频预处理模块的主要功能是同时提取时间域和频域的特征，具体实现如下：

```python
class TimeFrequencyPreprocessing(nn.Module):
    def __init__(self, freq_components=16, freq_feature_dim=64):
        super(TimeFrequencyPreprocessing, self).__init__()
        self.freq_components = freq_components
        self.freq_feature_dim = freq_feature_dim
        
        # 频域特征投影MLP
        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_components, 128),
            nn.ReLU(),
            nn.Linear(128, freq_feature_dim)
        )
    
    def forward(self, x):
        # 时间域特征（直接使用标准化后的数据）
        time_feature = x
        
        # 频域特征提取
        batch_size, seq_len, features = x.shape
        freq_features = np.zeros((batch_size, self.freq_components))
        
        for i in range(batch_size):
            seq_freq = []
            for j in range(features):
                # 对每个特征维度单独进行FFT
                signal = x[i, :, j].cpu().numpy()
                fft_result = np.fft.fft(signal)
                amplitude = np.abs(fft_result)[:len(signal)//2]
                
                # 截取前k个低频分量
                k = min(self.freq_components, len(amplitude))
                freq_component = amplitude[:k]
                
                # 补零到固定长度
                if len(freq_component) < self.freq_components:
                    freq_component = np.pad(freq_component, (0, self.freq_components - len(freq_component)))
                
                seq_freq.append(freq_component)
            
            # 计算每个样本的频域特征
            seq_freq = np.array(seq_freq).mean(axis=0)
            freq_features[i] = seq_freq
        
        # 将numpy数组转换为张量
        freq_features = torch.tensor(freq_features, dtype=torch.float32, device=x.device)
        
        # 通过MLP投影
        freq_proj = self.freq_mlp(freq_features)
        
        # 在时间维度上复制，与时间域特征对齐
        freq_feature = freq_proj.unsqueeze(1).repeat(1, seq_len, 1)
        
        return time_feature, freq_feature
```

### 3.1.2 多尺度卷积模块

多尺度卷积模块的主要功能是捕捉不同尺度的局部特征，具体实现如下：

```python
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        
        # 计算每个卷积的输出通道数
        conv_channels = out_channels // 3
        remainder = out_channels % 3
        
        # 分配余数
        conv3_channels = conv_channels
        conv5_channels = conv_channels
        conv7_channels = conv_channels
        
        if remainder >= 1:
            conv3_channels += 1
        if remainder >= 2:
            conv5_channels += 1
        
        # 三个不同尺度的卷积
        self.conv3 = nn.Conv1d(in_channels, conv3_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, conv5_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, conv7_channels, kernel_size=7, padding=3)
        
        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = None
    
    def forward(self, x):
        # 转换为(batch, features, seq_len)格式用于1D卷积
        x_transposed = x.transpose(1, 2)
        
        # 多尺度卷积
        out3 = self.conv3(x_transposed)
        out5 = self.conv5(x_transposed)
        out7 = self.conv7(x_transposed)
        
        # 拼接不同尺度的特征
        out = torch.cat([out3, out5, out7], dim=1)
        
        # 转换回(batch, seq_len, features)格式
        out = out.transpose(1, 2)
        
        # 残差连接
        if self.residual:
            residual = self.residual(x)
            out = out + residual
        else:
            out = out + x
        
        return F.relu(out)
```

### 3.1.3 Transformer编码器

Transformer编码器的主要功能是提取全局特征，具体实现如下：

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
    
    def forward(self, x):
        # 转换为(seq_len, batch, d_model)格式
        x = x.transpose(0, 1)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        output = self.transformer_encoder(x)
        
        # 转换回(batch, seq_len, d_model)格式
        output = output.transpose(0, 1)
        
        return output
```

### 3.1.4 时序重构路径

时序重构路径的主要功能是通过重构误差检测异常，具体实现如下：

```python
class TemporalReconstructionPath(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TemporalReconstructionPath, self).__init__()
        
        # 轻量级解码器
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 将输出约束在[-1,1]区间
        )
    
    def forward(self, x):
        # 重构输入
        recon = self.decoder(x)
        
        return recon
```

### 3.1.5 多尺度原型匹配路径

多尺度原型匹配路径的主要功能是通过与原型的距离检测异常，具体实现如下：

```python
class MultiScalePrototypeMatchingPath(nn.Module):
    def __init__(self, feature_dim, scales=[1, 2, 4], prototypes_per_scale=8):
        super(MultiScalePrototypeMatchingPath, self).__init__()
        self.feature_dim = feature_dim
        self.scales = scales
        self.prototypes_per_scale = prototypes_per_scale
        
        # 初始化多尺度原型
        self.prototypes = nn.ParameterDict()
        for scale in scales:
            self.prototypes[f'scale_{scale}'] = nn.Parameter(
                torch.randn(prototypes_per_scale, feature_dim)
            )
        
        # 尺度权重
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        matching_scores = []
        
        # 归一化尺度权重
        normalized_weights = F.softmax(self.scale_weights, dim=0)
        
        for i, scale in enumerate(self.scales):
            prototypes = self.prototypes[f'scale_{scale}']
            weight = normalized_weights[i]
            
            # 计算每个时间步与原型的距离
            scale_scores = []
            for t in range(seq_len):
                # 获取当前时间步的特征
                current_feature = x[:, t, :]
                
                # 计算与所有原型的距离
                distances = F.pairwise_distance(
                    current_feature.unsqueeze(1),
                    prototypes.unsqueeze(0)
                )
                
                # 取最小距离作为该尺度的分数
                min_distance = distances.min(dim=1)[0]
                scale_scores.append(min_distance)
            
            # 拼接所有时间步的分数
            scale_scores = torch.stack(scale_scores, dim=1)
            matching_scores.append(scale_scores * weight)
        
        # 加权融合不同尺度的分数
        total_score = sum(matching_scores)
        
        return total_score
```

### 3.1.6 不确定性估计模块

不确定性估计模块的主要功能是动态调整两条路径的权重，具体实现如下：

```python
class UncertaintyEstimation(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(UncertaintyEstimation, self).__init__()
        
        # 不确定性估计MLP
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 输出两条路径的不确定性
        )
    
    def forward(self, x):
        uncertainty = self.uncertainty_mlp(x)
        # 使用Softplus确保不确定性为正
        uncertainty = F.softplus(uncertainty)
        
        return uncertainty
```

### 3.1.7 模型前向传播

模型的前向传播过程如下：

```python
def forward(self, x):
    # 时频预处理
    time_feature, freq_feature = self.preprocessing(x)
    
    # 融合时间域和频域特征
    combined_feature = torch.cat([time_feature, freq_feature], dim=2)
    
    # 多尺度卷积提取局部特征
    local_feature = self.multi_scale_conv(combined_feature)
    
    # Transformer提取全局特征
    global_feature = self.transformer(local_feature)
    
    # 时序重构路径
    recon = self.reconstruction_path(local_feature)
    recon_error = torch.mean(torch.abs(x - recon), dim=2)
    
    # 多尺度原型匹配路径
    prototype_score = self.prototype_path(global_feature)
    
    # 不确定性估计输入
    uncertainty_input = torch.cat([
        global_feature,
        recon_error.unsqueeze(2),
        prototype_score.unsqueeze(2)
    ], dim=2)
    
    # 估计不确定性
    uncertainty = self.uncertainty_estimation(uncertainty_input)
    
    # 计算权重
    if self.use_fixed_weights:
        # 使用固定权重
        weight_recon = torch.ones_like(recon_error) * 0.5
        weight_prototype = torch.ones_like(prototype_score) * 0.5
    else:
        # 使用自适应权重
        recon_uncertainty = uncertainty[:, :, 0]
        prototype_uncertainty = uncertainty[:, :, 1]
        
        # Softmax计算权重
        exp_recon = torch.exp(-recon_uncertainty / self.temperature)
        exp_prototype = torch.exp(-prototype_uncertainty / self.temperature)
        weight_recon = exp_recon / (exp_recon + exp_prototype + 1e-8)
        weight_prototype = exp_prototype / (exp_recon + exp_prototype + 1e-8)
    
    # 融合两条路径的分数
    fused_score = weight_recon * recon_error + weight_prototype * prototype_score
    
    return {
        'reconstruction': recon,
        'recon_error': recon_error,
        'prototype_score': prototype_score,
        'uncertainty': uncertainty,
        'weights': {'recon': weight_recon, 'prototype': weight_prototype},
        'fused_score': fused_score
    }
```

## 3.2 无监督自适应策略

为了进一步提高异常检测的准确性，本文提出了一种无监督的异常分数自适应策略，主要包括以下步骤：

1. **多尺度局部对比度计算**：在不同尺度上计算局部对比度，捕捉不同尺度的异常信息。

2. **异常候选区域检测**：使用动态分位数阈值、移动平均阈值、融合对比度阈值等方法检测异常候选区域。

3. **区域类型分类**：根据长度将异常候选区域分为单点候选、短连续候选、中等连续候选和长连续候选，应用不同的增强策略。

4. **局部对比度增强**：对异常候选区域应用局部对比度增强，提高异常点的分数。

5. **非候选区域抑制**：对非候选区域的高分点进行抑制，减少误报。

6. **特殊处理**：检测单异常点，对突然变化的点进行增强。

7. **分数平滑**：对长连续异常进行高斯平滑，使分数更平滑。

8. **背景扣除**：使用训练集信息扣除背景分数，进一步提高异常检测的准确性。

## 3.3 损失函数

模型的损失函数包括重构损失、匹配路径损失和不确定性损失：

```python
def compute_loss(self, output, x):
    # 重构损失
    recon_loss = F.mse_loss(output['reconstruction'], x)
    
    # 匹配路径损失
    match_loss = output['prototype_score'].mean()
    
    # 不确定性损失
    recon_uncertainty = output['uncertainty'][:, :, 0]
    prototype_uncertainty = output['uncertainty'][:, :, 1]
    recon_error = output['recon_error']
    prototype_score = output['prototype_score']
    
    # 计算不确定性损失
    score_diff = torch.abs(recon_error - prototype_score)
    uncertainty_loss = (
        (score_diff / torch.exp(recon_uncertainty)).mean() +
        (score_diff / torch.exp(prototype_uncertainty)).mean() +
        recon_uncertainty.mean() +
        prototype_uncertainty.mean()
    )
    
    # 总损失
    total_loss = recon_loss + self.lambda1 * match_loss + self.lambda2 * uncertainty_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'match_loss': match_loss,
        'uncertainty_loss': uncertainty_loss
    }
```

## 3.4 模型训练

模型的训练过程如下：

1. **数据预处理**：对时间序列数据进行标准化处理，将数据缩放到[-1, 1]区间。

2. **窗口划分**：将时间序列数据划分为固定长度的滑动窗口。

3. **模型初始化**：初始化双路径异常检测模型的参数。

4. **训练优化**：使用Adam优化器，最小化总损失函数。

5. **模型保存**：训练完成后，保存模型权重和相关参数。

## 3.5 模型评估

模型的评估过程如下：

1. **加载模型**：加载训练好的模型权重。

2. **数据预处理**：对测试数据进行与训练数据相同的预处理。

3. **模型推理**：使用模型对测试数据进行推理，得到异常分数。

4. **应用自适应策略**：对异常分数应用无监督自适应策略，进一步优化分数。

5. **评估指标计算**：计算Top-1/3/5 Accuracy、VUS-ROC、VUS-PR、F1 Score、Precision、Recall等评估指标。

6. **结果分析**：分析模型在不同数据集上的表现，评估模型的泛化能力。