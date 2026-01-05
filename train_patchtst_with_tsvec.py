"""
PatchTST + TSVec时序对齐特征融合模型
在PatchTST基础上整合TSVec的时序嵌入特征

架构：
1. PatchTST分支：处理传统时序特征（12个特征）
2. TSVec时序分支：处理每个时间步的知识图谱嵌入（64维）
3. 时序对齐融合：在patch级别或时间步级别进行特征融合
4. 输出层：预测未来24小时的4个目标（lng, lat, speed, pressure）

关键改进：
- 不使用平均池化，保留时序维度
- 每个时间步的TSVec嵌入与对应时间的传统特征对齐融合
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置参数 ====================
TRADITIONAL_FEATURES_DIR = './data/processed_features'  # 传统特征
TSVEC_FEATURES_DIR = './data/temporal_kg_embeddings'  # TSVec嵌入特征
OUTPUT_DIR = './models/patchtst_with_tsvec'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
MODEL_CONFIG = {
    # PatchTST参数
    'patch_len': 4,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,
    'd_ff': 256,
    'dropout': 0.1,

    # TSVec特征融合参数
    'kg_embed_dim': 64,      # TSVec嵌入维度
    'fusion_method': 'concat',  # 融合方式: 'concat', 'attention', 'cross_attention'
}

# 训练参数
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
}

# ==================== 增强数据集 ====================
class TyphoonTSVecDataset(Dataset):
    """整合TSVec时序嵌入的台风数据集"""

    def __init__(self, X_trad, y, X_tsvec):
        """
        Args:
            X_trad: 传统特征 (samples, lookback, n_trad_features)
            y: 目标 (samples, forecast, n_output_features)
            X_tsvec: TSVec时序嵌入 (samples, lookback, kg_embed_dim)
        """
        self.X_trad = torch.FloatTensor(X_trad)
        self.y = torch.FloatTensor(y)
        self.X_tsvec = torch.FloatTensor(X_tsvec)

    def __len__(self):
        return len(self.X_trad)

    def __getitem__(self, idx):
        return self.X_trad[idx], self.X_tsvec[idx], self.y[idx]

# ==================== 模型组件 ====================
class PatchEmbedding(nn.Module):
    """Patch嵌入层"""

    def __init__(self, patch_len, n_features, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.linear = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        n_patches = seq_len // self.patch_len

        x = x[:, :n_patches * self.patch_len, :]
        x = x.reshape(batch_size, n_patches, self.patch_len, n_features)
        x = x.reshape(batch_size, n_patches, -1)
        x = self.linear(x)

        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class TSVecTemporalBranch(nn.Module):
    """TSVec时序特征处理分支（保留时序维度）"""

    def __init__(self, kg_embed_dim, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len

        # 将TSVec嵌入映射到d_model维度（逐时间步）
        self.temporal_projection = nn.Sequential(
            nn.Linear(kg_embed_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Patch级别的投影（将patch_len个时间步的特征聚合）
        self.patch_projection = nn.Linear(patch_len * d_model, d_model)

    def forward(self, x_tsvec):
        """
        Args:
            x_tsvec: (batch, lookback, kg_embed_dim)
        Returns:
            tsvec_patches: (batch, n_patches, d_model)
        """
        batch_size, seq_len, kg_dim = x_tsvec.shape
        n_patches = seq_len // self.patch_len

        # 1. 逐时间步投影
        x = self.temporal_projection(x_tsvec)  # (batch, lookback, d_model)

        # 2. 转换为patch表示
        x = x[:, :n_patches * self.patch_len, :]
        x = x.reshape(batch_size, n_patches, self.patch_len, -1)  # (batch, n_patches, patch_len, d_model)
        x = x.reshape(batch_size, n_patches, -1)  # (batch, n_patches, patch_len * d_model)

        # 3. Patch级别聚合
        tsvec_patches = self.patch_projection(x)  # (batch, n_patches, d_model)

        return tsvec_patches

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trad_feat, tsvec_feat):
        """
        Args:
            trad_feat: (batch, n_patches, d_model) 传统特征
            tsvec_feat: (batch, n_patches, d_model) TSVec特征
        Returns:
            fused: (batch, n_patches, d_model)
        """
        # 以传统特征为query，TSVec特征为key和value
        attn_out, attn_weights = self.cross_attn(trad_feat, tsvec_feat, tsvec_feat)
        fused = self.norm(trad_feat + self.dropout(attn_out))

        return fused

class PatchTSTWithTSVec(nn.Module):
    """PatchTST + TSVec时序对齐特征融合模型"""

    def __init__(self, lookback_window, forecast_horizon,
                 n_trad_features, n_output_features, kg_embed_dim,
                 patch_len, d_model, n_heads, n_layers, d_ff, dropout,
                 fusion_method='concat'):
        super().__init__()

        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.n_trad_features = n_trad_features
        self.n_output_features = n_output_features
        self.patch_len = patch_len
        self.fusion_method = fusion_method
        self.n_patches = lookback_window // patch_len

        # === PatchTST分支（传统特征）===
        self.patch_embedding = PatchEmbedding(patch_len, n_trad_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # === TSVec时序分支 ===
        self.tsvec_branch = TSVecTemporalBranch(kg_embed_dim, d_model, patch_len, dropout)

        # === 特征融合 ===
        if fusion_method == 'concat':
            # Patch级别拼接融合
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_method == 'attention':
            # 简单注意力融合（在patch维度聚合后融合）
            self.attn_query = nn.Linear(d_model, d_model)
            self.attn_key = nn.Linear(d_model, d_model)
            self.attn_value = nn.Linear(d_model, d_model)
        elif fusion_method == 'cross_attention':
            # 交叉注意力融合（patch级别交互）
            self.cross_fusion = CrossAttentionFusion(d_model, n_heads, dropout)

        # === 输出层 ===
        self.output_proj = nn.Linear(d_model, forecast_horizon * n_output_features)

    def forward(self, x_trad, x_tsvec):
        """
        Args:
            x_trad: (batch, lookback, n_trad_features) 传统特征
            x_tsvec: (batch, lookback, kg_embed_dim) TSVec时序嵌入
        Returns:
            y_pred: (batch, forecast, n_output_features)
        """
        batch_size = x_trad.shape[0]

        # === PatchTST分支 ===
        trad_patches = self.patch_embedding(x_trad)  # (batch, n_patches, d_model)
        trad_patches = trad_patches + self.pos_encoding
        trad_patches = self.dropout(trad_patches)

        for layer in self.encoder_layers:
            trad_patches = layer(trad_patches)

        # === TSVec分支（保留时序维度）===
        tsvec_patches = self.tsvec_branch(x_tsvec)  # (batch, n_patches, d_model)

        # === Patch级别特征融合 ===
        if self.fusion_method == 'concat':
            # 拼接融合
            fused_patches = torch.cat([trad_patches, tsvec_patches], dim=-1)  # (batch, n_patches, d_model*2)
            fused_patches = self.fusion_proj(fused_patches)  # (batch, n_patches, d_model)

            # 聚合patch维度
            fused = fused_patches.mean(dim=1)  # (batch, d_model)

        elif self.fusion_method == 'attention':
            # 简单注意力融合
            # 先聚合patch维度
            trad_feat = trad_patches.mean(dim=1)  # (batch, d_model)
            tsvec_feat = tsvec_patches.mean(dim=1)  # (batch, d_model)

            features = torch.stack([trad_feat, tsvec_feat], dim=1)  # (batch, 2, d_model)

            q = self.attn_query(trad_feat).unsqueeze(1)  # (batch, 1, d_model)
            k = self.attn_key(features)  # (batch, 2, d_model)
            v = self.attn_value(features)  # (batch, 2, d_model)

            attn_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(trad_feat.shape[-1])
            attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, 2)

            fused = torch.bmm(attn_weights, v).squeeze(1)  # (batch, d_model)

        elif self.fusion_method == 'cross_attention':
            # 交叉注意力融合（在patch级别交互）
            fused_patches = self.cross_fusion(trad_patches, tsvec_patches)  # (batch, n_patches, d_model)

            # 聚合patch维度
            fused = fused_patches.mean(dim=1)  # (batch, d_model)

        # === 输出层 ===
        output = self.output_proj(fused)  # (batch, forecast*n_output)
        output = output.reshape(batch_size, self.forecast_horizon, self.n_output_features)

        return output

# ==================== 数据加载 ====================
def load_data():
    """加载传统特征和TSVec时序嵌入，并进行对齐"""
    print("=" * 80)
    print("加载数据")
    print("=" * 80)

    # 1. 加载传统特征数据
    train_data = np.load(os.path.join(TRADITIONAL_FEATURES_DIR, 'train_data.npz'))
    val_data = np.load(os.path.join(TRADITIONAL_FEATURES_DIR, 'val_data.npz'))
    test_data = np.load(os.path.join(TRADITIONAL_FEATURES_DIR, 'test_data.npz'))

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    # 加载样本元信息（台风ID和时间索引）
    train_typhoon_ids = train_data['typhoon_ids']
    train_start_indices = train_data['start_indices']
    val_typhoon_ids = val_data['typhoon_ids']
    val_start_indices = val_data['start_indices']
    test_typhoon_ids = test_data['typhoon_ids']
    test_start_indices = test_data['start_indices']

    with open(os.path.join(TRADITIONAL_FEATURES_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    print(f"✓ 传统特征数据:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  样本元信息已加载 (typhoon_ids, start_indices)")

    # 2. 加载TSVec时序嵌入
    temporal_pkl_path = os.path.join(TSVEC_FEATURES_DIR, 'temporal_kg_features.pkl')

    if not os.path.exists(temporal_pkl_path):
        print(f"\n❌ 未找到TSVec时序特征: {temporal_pkl_path}")
        print(f"   请先运行: python train_tsvec_embedding.py")
        sys.exit(1)

    with open(temporal_pkl_path, 'rb') as f:
        temporal_data_full = pickle.load(f)

    temporal_kg_data = temporal_data_full['temporal_data']
    kg_dim = temporal_data_full['temporal_config']['rgcn_dim']

    print(f"\n✓ TSVec时序嵌入:")
    print(f"  台风数量: {len(temporal_kg_data)}")
    print(f"  嵌入维度: {kg_dim}")

    # 显示前5个台风的信息
    print(f"\n  示例台风:")
    for i, (tid, data) in enumerate(list(temporal_kg_data.items())[:5]):
        print(f"    {tid}: {data['seq_len']} 时间步, 特征维度={data['features'].shape}")

    # 3. 对齐TSVec时序嵌入到每个样本的每个时间步
    print(f"\n✓ 对齐TSVec时序嵌入到样本...")

    lookback_window = metadata['lookback_window']

    def align_tsvec_to_samples(typhoon_ids, start_indices, lookback):
        """
        根据台风ID和时间索引对齐TSVec嵌入

        关键：保留每个时间步的嵌入，不进行池化
        """
        X_tsvec_list = []
        matched_count = 0
        zero_padded_count = 0
        partial_padded_count = 0

        for typhoon_id, start_idx in zip(typhoon_ids, start_indices):
            if typhoon_id not in temporal_kg_data:
                # 如果该台风没有TSVec嵌入，使用零向量
                X_tsvec_list.append(np.zeros((lookback, kg_dim), dtype=np.float32))
                zero_padded_count += 1
                continue

            kg_info = temporal_kg_data[typhoon_id]
            kg_features = kg_info['features']  # (seq_len, kg_dim)
            kg_indices = kg_info.get('indices', None)  # 时间步索引

            # 如果有 indices 字段，使用精确对齐
            if kg_indices is not None:
                # 创建索引到特征的映射
                index_to_feat = {int(idx): kg_features[i] for i, idx in enumerate(kg_indices)}

                # 提取对应时间窗口的每个时间步特征
                X_tsvec = []
                for t in range(start_idx, start_idx + lookback):
                    if t in index_to_feat:
                        X_tsvec.append(index_to_feat[t])
                    else:
                        # 如果该时间步没有TSVec嵌入，填充零
                        X_tsvec.append(np.zeros(kg_dim, dtype=np.float32))

                X_tsvec = np.array(X_tsvec, dtype=np.float32)

                # 统计对齐情况
                num_matched = sum(1 for t in range(start_idx, start_idx + lookback) if t in index_to_feat)
                if num_matched == lookback:
                    matched_count += 1
                elif num_matched > 0:
                    partial_padded_count += 1
                else:
                    zero_padded_count += 1
            else:
                # 如果没有 indices 字段，使用连续索引（兼容旧格式）
                end_idx = start_idx + lookback
                if end_idx <= len(kg_features):
                    X_tsvec = kg_features[start_idx:end_idx]
                    matched_count += 1
                elif start_idx < len(kg_features):
                    # 部分可用
                    available = kg_features[start_idx:]
                    padding = np.zeros((lookback - len(available), kg_dim), dtype=np.float32)
                    X_tsvec = np.vstack([available, padding])
                    partial_padded_count += 1
                else:
                    # 完全超出范围
                    X_tsvec = np.zeros((lookback, kg_dim), dtype=np.float32)
                    zero_padded_count += 1

            X_tsvec_list.append(X_tsvec)

        # 打印对齐统计
        total = len(typhoon_ids)
        print(f"    完全匹配: {matched_count}/{total} ({matched_count/total*100:.1f}%)")
        print(f"    部分填充: {partial_padded_count}/{total} ({partial_padded_count/total*100:.1f}%)")
        print(f"    零填充: {zero_padded_count}/{total} ({zero_padded_count/total*100:.1f}%)")

        return np.array(X_tsvec_list, dtype=np.float32)

    # 对齐训练集
    print(f"\n  [训练集] 对齐TSVec嵌入...")
    X_tsvec_train = align_tsvec_to_samples(train_typhoon_ids, train_start_indices, lookback_window)
    print(f"    训练集TSVec特征: {X_tsvec_train.shape}")

    # 对齐验证集
    print(f"\n  [验证集] 对齐TSVec嵌入...")
    X_tsvec_val = align_tsvec_to_samples(val_typhoon_ids, val_start_indices, lookback_window)
    print(f"    验证集TSVec特征: {X_tsvec_val.shape}")

    # 对齐测试集
    print(f"\n  [测试集] 对齐TSVec嵌入...")
    X_tsvec_test = align_tsvec_to_samples(test_typhoon_ids, test_start_indices, lookback_window)
    print(f"    测试集TSVec特征: {X_tsvec_test.shape}")

    # 添加维度信息到metadata
    metadata['kg_embed_dim'] = kg_dim

    return (X_train, y_train, X_tsvec_train), (X_val, y_val, X_tsvec_val), (X_test, y_test, X_tsvec_test), metadata

def create_dataloaders(train_data, val_data, test_data, batch_size):
    """创建数据加载器"""
    X_train, y_train, X_tsvec_train = train_data
    X_val, y_val, X_tsvec_val = val_data
    X_test, y_test, X_tsvec_test = test_data

    train_dataset = TyphoonTSVecDataset(X_train, y_train, X_tsvec_train)
    val_dataset = TyphoonTSVecDataset(X_val, y_val, X_tsvec_val)
    test_dataset = TyphoonTSVecDataset(X_test, y_test, X_tsvec_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n✓ 数据加载器创建完成")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    print(f"  测试批次: {len(test_loader)}")

    return train_loader, val_loader, test_loader

# ==================== 训练/验证/测试 ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for X_trad, X_tsvec, y in train_loader:
        X_trad, X_tsvec, y = X_trad.to(device), X_tsvec.to(device), y.to(device)

        y_pred = model(X_trad, X_tsvec)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_trad, X_tsvec, y in val_loader:
            X_trad, X_tsvec, y = X_trad.to(device), X_tsvec.to(device), y.to(device)

            y_pred = model(X_trad, X_tsvec)
            loss = criterion(y_pred, y)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def test_model(model, test_loader, device, metadata):
    """测试模型"""
    print("\n" + "=" * 80)
    print("测试模型")
    print("=" * 80)

    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for X_trad, X_tsvec, y in test_loader:
            X_trad, X_tsvec = X_trad.to(device), X_tsvec.to(device)
            y_pred = model(X_trad, X_tsvec)

            predictions.append(y_pred.cpu().numpy())
            ground_truth.append(y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    # 归一化空间指标
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - ground_truth))

    print(f"\n归一化空间指标:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    # 反归一化
    if 'scaler_y' in metadata:
        scaler = metadata['scaler_y']
        n_output = metadata['n_target_features']

        pred_flat = predictions.reshape(-1, n_output)
        true_flat = ground_truth.reshape(-1, n_output)

        # 直接反归一化（目标特征的scaler）
        pred_original = scaler.inverse_transform(pred_flat)
        true_original = scaler.inverse_transform(true_flat)

        mae_original = np.mean(np.abs(pred_original - true_original))
        rmse_original = np.sqrt(np.mean((pred_original - true_original) ** 2))

        print(f"\n原始尺度指标:")
        print(f"  RMSE: {rmse_original:.6f}")
        print(f"  MAE:  {mae_original:.6f}")

        # 各特征误差
        if 'target_features' in metadata:
            print(f"\n各特征MAE（原始尺度）:")
            for i, feat in enumerate(metadata['target_features']):
                feat_mae = np.mean(np.abs(pred_original[:, i] - true_original[:, i]))
                print(f"  {feat:20s}: {feat_mae:.4f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

# ==================== 主训练函数 ====================
def train():
    """主训练流程"""
    print("\n" + "=" * 80)
    print("PatchTST + TSVec时序对齐特征融合模型训练")
    print("=" * 80)
    print(f"设备: {DEVICE}")
    print(f"融合方式: {MODEL_CONFIG['fusion_method']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    train_data, val_data, test_data, metadata = load_data()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, TRAIN_CONFIG['batch_size']
    )

    # 创建模型
    print("\n" + "=" * 80)
    print("创建PatchTST+TSVec模型")
    print("=" * 80)

    model = PatchTSTWithTSVec(
        lookback_window=metadata['lookback_window'],
        forecast_horizon=metadata['forecast_horizon'],
        n_trad_features=metadata['n_input_features'],
        n_output_features=metadata['n_target_features'],
        kg_embed_dim=metadata['kg_embed_dim'],
        patch_len=MODEL_CONFIG['patch_len'],
        d_model=MODEL_CONFIG['d_model'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_layers=MODEL_CONFIG['n_layers'],
        d_ff=MODEL_CONFIG['d_ff'],
        dropout=MODEL_CONFIG['dropout'],
        fusion_method=MODEL_CONFIG['fusion_method']
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建完成")
    print(f"  总参数量: {total_params:,}")

    # 训练
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                          lr=TRAIN_CONFIG['learning_rate'],
                          weight_decay=TRAIN_CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=TRAIN_CONFIG['scheduler_factor'],
        patience=TRAIN_CONFIG['scheduler_patience']
    )

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{TRAIN_CONFIG['epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'model_config': MODEL_CONFIG,
                'metadata': metadata
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型")
        else:
            patience_counter += 1

        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping")
            break

    print(f"\n✓ 训练完成！用时: {(time.time()-start_time)/60:.1f}分钟")

    # 测试
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = test_model(model, test_loader, DEVICE, metadata)

    # 保存历史
    with open(os.path.join(OUTPUT_DIR, 'training_history.pkl'), 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'test_results': test_results,
            'model_config': MODEL_CONFIG,
            'train_config': TRAIN_CONFIG
        }, f)

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"模型保存在: {OUTPUT_DIR}/best_model.pth")

if __name__ == '__main__':
    train()
