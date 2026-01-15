"""
PatchTST台风预测模型 - GPU/CPU自适应训练

特点：
1. 不使用RGCN特征增强（纯传统特征）
2. 自动检测GPU/CPU并优化训练
3. 使用12个输入特征预测4个目标特征
4. 预测目标：经度(lng)、纬度(lat)、风速(speed)、气压(pressure)
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置参数 ====================
DATA_DIR = './data/processed_features'
OUTPUT_DIR = './models/patchtst'

# 自动检测GPU/CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
MODEL_CONFIG = {
    'patch_len': 4,          # 每个patch的长度
    'd_model': 128,          # 模型维度
    'n_heads': 4,            # 注意力头数
    'n_layers': 3,           # Transformer层数
    'd_ff': 256,             # 前馈网络维度
    'dropout': 0.1,          # Dropout率
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

# ==================== 数据集类 ====================
class TyphoonDataset(Dataset):
    """台风时序数据集"""

    def __init__(self, X, y):
        """
        Args:
            X: numpy array, shape (samples, time_steps, features)
            y: numpy array, shape (samples, forecast_horizon, features)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== PatchTST模型 ====================
class PatchEmbedding(nn.Module):
    """将时序数据切分为patches并嵌入"""

    def __init__(self, patch_len, n_features, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.n_features = n_features

        # 线性投影
        self.linear = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            patches: (batch, n_patches, d_model)
        """
        batch_size, seq_len, n_features = x.shape

        # 计算patch数量
        n_patches = seq_len // self.patch_len

        # 重塑为patches: (batch, n_patches, patch_len, n_features)
        x = x[:, :n_patches * self.patch_len, :]  # 裁剪到可被整除
        x = x.reshape(batch_size, n_patches, self.patch_len, n_features)

        # 展平每个patch: (batch, n_patches, patch_len * n_features)
        x = x.reshape(batch_size, n_patches, -1)

        # 线性投影: (batch, n_patches, d_model)
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
        # Multi-head attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class PatchTST(nn.Module):
    """PatchTST模型"""

    def __init__(self, lookback_window, forecast_horizon, n_input_features, n_output_features,
                 patch_len, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()

        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.n_input_features = n_input_features
        self.n_output_features = n_output_features
        self.patch_len = patch_len

        # Patch嵌入 - 使用输入特征数
        self.patch_embedding = PatchEmbedding(patch_len, n_input_features, d_model)

        # 位置编码
        n_patches = lookback_window // patch_len
        self.pos_encoding = nn.Parameter(torch.randn(1, n_patches, d_model))

        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出投影层 - 使用输出特征数
        self.output_proj = nn.Linear(d_model, forecast_horizon * n_output_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, lookback_window, n_input_features)
        Returns:
            y: (batch, forecast_horizon, n_output_features)
        """
        batch_size = x.shape[0]

        # Patch嵌入
        x = self.patch_embedding(x)  # (batch, n_patches, d_model)

        # 添加位置编码
        x = x + self.pos_encoding
        x = self.dropout(x)

        # Transformer编码器
        for layer in self.encoder_layers:
            x = layer(x)

        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)

        # 输出投影
        x = self.output_proj(x)  # (batch, forecast_horizon * n_output_features)

        # 重塑为(batch, forecast_horizon, n_output_features)
        x = x.reshape(batch_size, self.forecast_horizon, self.n_output_features)

        return x

# ==================== 工具函数 ====================
def load_data():
    """加载数据"""
    print("=" * 80)
    print("加载数据")
    print("=" * 80)

    # 加载numpy数据
    train_data = np.load(os.path.join(DATA_DIR, 'train_data.npz'))
    val_data = np.load(os.path.join(DATA_DIR, 'val_data.npz'))
    test_data = np.load(os.path.join(DATA_DIR, 'test_data.npz'))

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    # 加载元数据
    with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    print(f"✓ 数据加载完成")
    print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"  验证集: X={X_val.shape}, y={y_val.shape}")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
    print(f"  输入特征数: {metadata['n_input_features']}")
    print(f"  目标特征数: {metadata['n_target_features']}")
    print(f"  输入特征: {metadata['input_features']}")
    print(f"  目标特征: {metadata['target_features']}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata

def create_dataloaders(train_data, val_data, test_data, batch_size):
    """创建数据加载器"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_dataset = TyphoonDataset(X_train, y_train)
    val_dataset = TyphoonDataset(X_val, y_val)
    test_dataset = TyphoonDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n✓ 数据加载器创建完成")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")

    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 前向传播
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def test_model(model, test_loader, device, metadata):
    """测试模型"""
    print("\n" + "=" * 80)
    print("测试模型")
    print("=" * 80)

    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)

            predictions.append(y_pred.cpu().numpy())
            ground_truth.append(y_batch.numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    # 计算指标（归一化空间）
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - ground_truth))

    print(f"\n归一化空间指标:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    # 反归一化到原始尺度
    scaler_y = metadata['scaler_y']
    n_target_features = metadata['n_target_features']

    pred_flat = predictions.reshape(-1, n_target_features)
    true_flat = ground_truth.reshape(-1, n_target_features)

    pred_original = scaler_y.inverse_transform(pred_flat)
    true_original = scaler_y.inverse_transform(true_flat)

    # 计算原始尺度指标
    mse_original = np.mean((pred_original - true_original) ** 2)
    rmse_original = np.sqrt(mse_original)
    mae_original = np.mean(np.abs(pred_original - true_original))

    print(f"\n原始尺度指标:")
    print(f"  MSE:  {mse_original:.6f}")
    print(f"  RMSE: {rmse_original:.6f}")
    print(f"  MAE:  {mae_original:.6f}")

    # 各特征的误差
    target_features = metadata['target_features']
    print(f"\n各特征MAE（原始尺度）:")
    for i, feat in enumerate(target_features):
        feat_mae = np.mean(np.abs(pred_original[:, i] - true_original[:, i]))
        print(f"  {feat:20s}: {feat_mae:.4f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mse_original': mse_original,
        'rmse_original': rmse_original,
        'mae_original': mae_original,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'predictions_original': pred_original,
        'ground_truth_original': true_original
    }

# ==================== 主训练函数 ====================
def train():
    """主训练流程"""
    print("\n" + "=" * 80)
    print("PatchTST台风预测模型训练")
    print("=" * 80)
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    train_data, val_data, test_data, metadata = load_data()

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, TRAIN_CONFIG['batch_size']
    )

    # 创建模型
    print("\n" + "=" * 80)
    print("创建PatchTST模型")
    print("=" * 80)

    model = PatchTST(
        lookback_window=metadata['lookback_window'],
        forecast_horizon=metadata['forecast_horizon'],
        n_input_features=metadata['n_input_features'],
        n_output_features=metadata['n_target_features'],
        patch_len=MODEL_CONFIG['patch_len'],
        d_model=MODEL_CONFIG['d_model'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_layers=MODEL_CONFIG['n_layers'],
        d_ff=MODEL_CONFIG['d_ff'],
        dropout=MODEL_CONFIG['dropout']
    ).to(DEVICE)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ 模型创建完成")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型配置:")
    for key, value in MODEL_CONFIG.items():
        print(f"    {key}: {value}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=TRAIN_CONFIG['learning_rate'],
                           weight_decay=TRAIN_CONFIG['weight_decay'])

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=TRAIN_CONFIG['scheduler_factor'],
        patience=TRAIN_CONFIG['scheduler_patience']
    )

    # 训练循环
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 记录
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{TRAIN_CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'model_config': MODEL_CONFIG,
                'metadata': metadata
            }

            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping触发 (patience={TRAIN_CONFIG['early_stopping_patience']})")
            break

    total_time = time.time() - start_time
    print(f"\n✓ 训练完成！")
    print(f"  总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  最佳验证损失: {best_val_loss:.6f}")

    # 加载最佳模型并测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = test_model(model, test_loader, DEVICE, metadata)

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'test_results': test_results,
        'total_time': total_time,
        'model_config': MODEL_CONFIG,
        'train_config': TRAIN_CONFIG
    }

    with open(os.path.join(OUTPUT_DIR, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print("\n" + "=" * 80)
    print("训练和测试完成！")
    print("=" * 80)
    print(f"\n保存文件:")
    print(f"  - {OUTPUT_DIR}/best_model.pth")
    print(f"  - {OUTPUT_DIR}/training_history.pkl")

if __name__ == '__main__':
    train()
