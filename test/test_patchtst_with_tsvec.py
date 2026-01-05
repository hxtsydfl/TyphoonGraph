"""
PatchTST+TSVec台风预测模型 - 独立测试脚本

用途：
1. 加载已训练好的TSVec融合模型
2. 在测试集上评估性能
3. 生成预测结果，便于与原始PatchTST对比
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置参数 ====================
TRADITIONAL_FEATURES_DIR = '../data/processed_features'  # 传统特征
TSVEC_FEATURES_DIR = '../data/temporal_kg_embeddings'  # TSVec嵌入特征
MODEL_DIR = '../models/patchtst_with_tsvec'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据集类 ====================
class TyphoonTSVecDataset(Dataset):
    """整合TSVec时序嵌入的台风数据集"""

    def __init__(self, X_trad, y, X_tsvec):
        self.X_trad = torch.FloatTensor(X_trad)
        self.y = torch.FloatTensor(y)
        self.X_tsvec = torch.FloatTensor(X_tsvec)

    def __len__(self):
        return len(self.X_trad)

    def __getitem__(self, idx):
        return self.X_trad[idx], self.X_tsvec[idx], self.y[idx]

# ==================== 模型组件 ====================
class PatchEmbedding(nn.Module):
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
    """TSVec时序特征处理分支"""

    def __init__(self, kg_embed_dim, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len

        self.temporal_projection = nn.Sequential(
            nn.Linear(kg_embed_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.patch_projection = nn.Linear(patch_len * d_model, d_model)

    def forward(self, x_tsvec):
        batch_size, seq_len, kg_dim = x_tsvec.shape
        n_patches = seq_len // self.patch_len

        x = self.temporal_projection(x_tsvec)
        x = x[:, :n_patches * self.patch_len, :]
        x = x.reshape(batch_size, n_patches, self.patch_len, -1)
        x = x.reshape(batch_size, n_patches, -1)
        tsvec_patches = self.patch_projection(x)

        return tsvec_patches

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trad_feat, tsvec_feat):
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

        # PatchTST分支
        self.patch_embedding = PatchEmbedding(patch_len, n_trad_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # TSVec时序分支
        self.tsvec_branch = TSVecTemporalBranch(kg_embed_dim, d_model, patch_len, dropout)

        # 特征融合
        if fusion_method == 'concat':
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_method == 'attention':
            self.attn_query = nn.Linear(d_model, d_model)
            self.attn_key = nn.Linear(d_model, d_model)
            self.attn_value = nn.Linear(d_model, d_model)
        elif fusion_method == 'cross_attention':
            self.cross_fusion = CrossAttentionFusion(d_model, n_heads, dropout)

        # 输出层
        self.output_proj = nn.Linear(d_model, forecast_horizon * n_output_features)

    def forward(self, x_trad, x_tsvec):
        batch_size = x_trad.shape[0]

        # PatchTST分支
        trad_patches = self.patch_embedding(x_trad)
        trad_patches = trad_patches + self.pos_encoding
        trad_patches = self.dropout(trad_patches)

        for layer in self.encoder_layers:
            trad_patches = layer(trad_patches)

        # TSVec分支
        tsvec_patches = self.tsvec_branch(x_tsvec)

        # 特征融合
        if self.fusion_method == 'concat':
            fused_patches = torch.cat([trad_patches, tsvec_patches], dim=-1)
            fused_patches = self.fusion_proj(fused_patches)
            fused = fused_patches.mean(dim=1)

        elif self.fusion_method == 'attention':
            trad_feat = trad_patches.mean(dim=1)
            tsvec_feat = tsvec_patches.mean(dim=1)

            features = torch.stack([trad_feat, tsvec_feat], dim=1)

            q = self.attn_query(trad_feat).unsqueeze(1)
            k = self.attn_key(features)
            v = self.attn_value(features)

            attn_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(trad_feat.shape[-1])
            attn_weights = F.softmax(attn_scores, dim=-1)

            fused = torch.bmm(attn_weights, v).squeeze(1)

        elif self.fusion_method == 'cross_attention':
            fused_patches = self.cross_fusion(trad_patches, tsvec_patches)
            fused = fused_patches.mean(dim=1)

        # 输出层
        output = self.output_proj(fused)
        output = output.reshape(batch_size, self.forecast_horizon, self.n_output_features)

        return output

# ==================== 数据加载函数 ====================
def load_test_data():
    """加载测试数据（传统特征 + TSVec嵌入）"""
    print("=" * 80)
    print("加载测试数据")
    print("=" * 80)

    # 1. 加载传统特征
    test_data = np.load(os.path.join(TRADITIONAL_FEATURES_DIR, 'test_data.npz'))
    X_test, y_test = test_data['X'], test_data['y']
    test_typhoon_ids = test_data['typhoon_ids']
    test_start_indices = test_data['start_indices']

    with open(os.path.join(TRADITIONAL_FEATURES_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    print(f"✓ 传统特征:")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
    print(f"  输入特征数: {metadata['n_input_features']}")
    print(f"  目标特征数: {metadata['n_target_features']}")
    print(f"  目标特征: {metadata['target_features']}")

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

    # 3. 对齐TSVec嵌入到测试集
    print(f"\n✓ 对齐TSVec嵌入到测试集...")

    lookback_window = metadata['lookback_window']
    X_tsvec_test_list = []
    matched_count = 0
    zero_padded_count = 0
    partial_padded_count = 0

    for typhoon_id, start_idx in zip(test_typhoon_ids, test_start_indices):
        if typhoon_id not in temporal_kg_data:
            X_tsvec_test_list.append(np.zeros((lookback_window, kg_dim), dtype=np.float32))
            zero_padded_count += 1
            continue

        kg_info = temporal_kg_data[typhoon_id]
        kg_features = kg_info['features']
        kg_indices = kg_info.get('indices', None)

        if kg_indices is not None:
            index_to_feat = {int(idx): kg_features[i] for i, idx in enumerate(kg_indices)}

            X_tsvec = []
            for t in range(start_idx, start_idx + lookback_window):
                if t in index_to_feat:
                    X_tsvec.append(index_to_feat[t])
                else:
                    X_tsvec.append(np.zeros(kg_dim, dtype=np.float32))

            X_tsvec = np.array(X_tsvec, dtype=np.float32)

            num_matched = sum(1 for t in range(start_idx, start_idx + lookback_window) if t in index_to_feat)
            if num_matched == lookback_window:
                matched_count += 1
            elif num_matched > 0:
                partial_padded_count += 1
            else:
                zero_padded_count += 1
        else:
            end_idx = start_idx + lookback_window
            if end_idx <= len(kg_features):
                X_tsvec = kg_features[start_idx:end_idx]
                matched_count += 1
            elif start_idx < len(kg_features):
                available = kg_features[start_idx:]
                padding = np.zeros((lookback_window - len(available), kg_dim), dtype=np.float32)
                X_tsvec = np.vstack([available, padding])
                partial_padded_count += 1
            else:
                X_tsvec = np.zeros((lookback_window, kg_dim), dtype=np.float32)
                zero_padded_count += 1

        X_tsvec_test_list.append(X_tsvec)

    X_tsvec_test = np.array(X_tsvec_test_list, dtype=np.float32)

    total = len(test_typhoon_ids)
    print(f"  完全匹配: {matched_count}/{total} ({matched_count/total*100:.1f}%)")
    print(f"  部分填充: {partial_padded_count}/{total} ({partial_padded_count/total*100:.1f}%)")
    print(f"  零填充: {zero_padded_count}/{total} ({zero_padded_count/total*100:.1f}%)")
    print(f"  TSVec特征: {X_tsvec_test.shape}")

    # 添加维度信息到metadata
    metadata['kg_embed_dim'] = kg_dim

    return X_test, y_test, X_tsvec_test, metadata

def load_model(metadata):
    """加载训练好的模型"""
    print("\n" + "=" * 80)
    print("加载训练好的模型")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_config = checkpoint['model_config']

    model = PatchTSTWithTSVec(
        lookback_window=metadata['lookback_window'],
        forecast_horizon=metadata['forecast_horizon'],
        n_trad_features=metadata['n_input_features'],
        n_output_features=metadata['n_target_features'],
        kg_embed_dim=metadata['kg_embed_dim'],
        patch_len=model_config['patch_len'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        d_ff=model_config['d_ff'],
        dropout=model_config['dropout'],
        fusion_method=model_config['fusion_method']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载完成")
    print(f"  训练epoch: {checkpoint['epoch']}")
    print(f"  验证损失: {checkpoint['val_loss']:.6f}")
    print(f"  融合方式: {model_config['fusion_method']}")
    print(f"  设备: {DEVICE}")

    return model

def test_model(model, X_test, y_test, X_tsvec_test, metadata, batch_size=32):
    """测试模型"""
    print("\n" + "=" * 80)
    print("模型测试")
    print("=" * 80)

    test_dataset = TyphoonTSVecDataset(X_test, y_test, X_tsvec_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for X_trad_batch, X_tsvec_batch, y_batch in test_loader:
            X_trad_batch = X_trad_batch.to(DEVICE)
            X_tsvec_batch = X_tsvec_batch.to(DEVICE)

            y_pred = model(X_trad_batch, X_tsvec_batch)

            predictions.append(y_pred.cpu().numpy())
            ground_truth.append(y_batch.numpy())

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
    scaler_y = metadata['scaler_y']
    n_target_features = metadata['n_target_features']
    target_features = metadata['target_features']

    pred_flat = predictions.reshape(-1, n_target_features)
    true_flat = ground_truth.reshape(-1, n_target_features)

    pred_original = scaler_y.inverse_transform(pred_flat)
    true_original = scaler_y.inverse_transform(true_flat)

    # 原始尺度指标
    mse_original = np.mean((pred_original - true_original) ** 2)
    rmse_original = np.sqrt(mse_original)
    mae_original = np.mean(np.abs(pred_original - true_original))

    print(f"\n原始尺度指标:")
    print(f"  MSE:  {mse_original:.6f}")
    print(f"  RMSE: {rmse_original:.6f}")
    print(f"  MAE:  {mae_original:.6f}")

    # 各特征误差
    print(f"\n各特征MAE（原始尺度）:")
    for i, feat in enumerate(target_features):
        feat_mae = np.mean(np.abs(pred_original[:, i] - true_original[:, i]))
        print(f"  {feat:20s}: {feat_mae:.4f}")

    # 保存结果
    results = {
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

    output_path = os.path.join(MODEL_DIR, 'test_results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ 测试结果已保存: {output_path}")
    return results

# ==================== 读取和分析测试结果 ====================
def load_and_analyze_results():
    """读取并详细分析测试结果"""
    results_path = os.path.join(MODEL_DIR, 'test_results.pkl')

    if not os.path.exists(results_path):
        print(f"❌ 测试结果文件不存在: {results_path}")
        print("   请先运行测试: python test/test_patchtst_with_tsvec.py")
        return None

    # 加载metadata
    with open(os.path.join(TRADITIONAL_FEATURES_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # 加载结果
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    print("=" * 80)
    print("详细测试结果分析 - PatchTST+TSVec模型")
    print("=" * 80)

    # 基本信息
    print(f"\n【1. 数据集信息】")
    print(f"  样本数: {len(results['predictions'])}")
    print(f"  预测时间步: {metadata['forecast_horizon']}")
    print(f"  目标特征: {', '.join(metadata['target_features'])}")

    # 归一化空间指标
    print(f"\n【2. 归一化空间指标】")
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE:  {results['mae']:.6f}")

    # 原始尺度指标
    print(f"\n【3. 原始尺度指标】")
    print(f"  MSE:  {results['mse_original']:.6f}")
    print(f"  RMSE: {results['rmse_original']:.6f}")
    print(f"  MAE:  {results['mae_original']:.6f}")

    # 各特征详细误差
    pred_orig = results['predictions_original']
    true_orig = results['ground_truth_original']
    target_features = metadata['target_features']

    print(f"\n【4. 各特征详细误差（原始尺度）】")
    for i, feat in enumerate(target_features):
        mae = np.mean(np.abs(pred_orig[:, i] - true_orig[:, i]))
        mse = np.mean((pred_orig[:, i] - true_orig[:, i]) ** 2)
        rmse = np.sqrt(mse)
        min_err = np.min(np.abs(pred_orig[:, i] - true_orig[:, i]))
        max_err = np.max(np.abs(pred_orig[:, i] - true_orig[:, i]))

        mape = np.mean(np.abs((pred_orig[:, i] - true_orig[:, i]) / (np.abs(true_orig[:, i]) + 1e-8))) * 100

        print(f"\n  {feat}:")
        print(f"    MAE:       {mae:.4f}")
        print(f"    RMSE:      {rmse:.4f}")
        print(f"    MAPE:      {mape:.2f}%")
        print(f"    最小误差:   {min_err:.4f}")
        print(f"    最大误差:   {max_err:.4f}")
        print(f"    平均真实值: {np.mean(true_orig[:, i]):.4f}")
        print(f"    平均预测值: {np.mean(pred_orig[:, i]):.4f}")

    # 各时间步误差
    print(f"\n【5. 各预测时间步的MAE（原始尺度）】")
    pred_reshaped = pred_orig.reshape(-1, metadata['forecast_horizon'], len(target_features))
    true_reshaped = true_orig.reshape(-1, metadata['forecast_horizon'], len(target_features))

    for t in range(metadata['forecast_horizon']):
        mae_t = np.mean(np.abs(pred_reshaped[:, t, :] - true_reshaped[:, t, :]))
        print(f"  +{t+1}小时: {mae_t:.4f}")

    # 预测示例
    print(f"\n【6. 前5个样本预测示例】")
    for idx in range(min(5, len(pred_reshaped))):
        print(f"\n  样本 #{idx+1}:")
        print(f"    预测 (+1小时): {pred_reshaped[idx, 0, :]}")
        print(f"    真实 (+1小时): {true_reshaped[idx, 0, :]}")
        print(f"    误差: {np.abs(pred_reshaped[idx, 0, :] - true_reshaped[idx, 0, :])}")

    return results

# ==================== 主函数 ====================
def main():
    import sys

    print("=" * 80)
    print("PatchTST+TSVec台风预测模型 - 独立测试")
    print("=" * 80)

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--load':
            print("\n模式: 读取已保存的测试结果\n")
            load_and_analyze_results()
            return
        elif sys.argv[1] == '--help':
            print("\n用法:")
            print("  python test/test_patchtst_with_tsvec.py        # 运行测试")
            print("  python test/test_patchtst_with_tsvec.py --load # 读取已保存的测试结果")
            return

    # 运行完整测试
    print(f"\n模式: 运行完整测试")
    print(f"设备: {DEVICE}\n")

    X_test, y_test, X_tsvec_test, metadata = load_test_data()
    model = load_model(metadata)
    results = test_model(model, X_test, y_test, X_tsvec_test, metadata)

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n提示: 运行 'python test/test_patchtst_with_tsvec.py --load' 查看详细分析")

if __name__ == '__main__':
    main()
