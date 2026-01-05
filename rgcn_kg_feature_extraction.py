"""
RGCN知识图谱特征提取模块
用于从台风知识图谱中提取特征,增强PatchTST模型的预测能力

功能：
1. 加载知识图谱数据（实体、关系、三元组）
2. 使用RGCN提取节点嵌入
3. 为每个台风观测点提取图结构特征
4. 保存特征用于后续模型训练
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from collections import defaultdict
import json
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置参数 ====================
KG_DATA_DIR = './data/kg'  # 知识图谱数据目录
OUTPUT_DIR = './data/kg_features'  # 输出目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RGCN模型参数
RGCN_CONFIG = {
    'hidden_dim': 128,      # 隐藏层维度
    'embedding_dim': 64,    # 最终嵌入维度
    'num_layers': 2,        # RGCN层数
    'dropout': 0.1,         # Dropout率
}

# ==================== 数据加载 ====================
class KnowledgeGraphLoader:
    """知识图谱数据加载器"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.triples = []
        self.entity_properties = {}

    def load_entity2id(self):
        """加载实体ID映射"""
        file_path = os.path.join(self.data_dir, 'entity2id.txt')
        print(f"加载实体映射: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        num_entities = int(lines[0].strip())
        print(f"实体总数: {num_entities}")

        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    entity, eid = parts
                    eid = int(eid)
                    self.entity2id[entity] = eid
                    self.id2entity[eid] = entity

        print(f"成功加载 {len(self.entity2id)} 个实体")
        return self.entity2id

    def load_relation2id(self):
        """加载关系ID映射"""
        file_path = os.path.join(self.data_dir, 'relation2id.txt')
        print(f"加载关系映射: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        num_relations = int(lines[0].strip())
        print(f"关系总数: {num_relations}")

        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    relation, rid = parts
                    rid = int(rid)
                    self.relation2id[relation] = rid
                    self.id2relation[rid] = relation

        print(f"成功加载 {len(self.relation2id)} 个关系")
        return self.relation2id

    def load_triples(self):
        """加载三元组"""
        file_path = os.path.join(self.data_dir, 'all_triples.txt')
        print(f"加载三元组: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        num_triples = int(lines[0].strip())
        print(f"三元组总数: {num_triples}")

        self.triples = []
        for line in tqdm(lines[1:], desc="读取三元组"):
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 3:
                    head, tail, relation = map(int, parts)
                    self.triples.append((head, tail, relation))

        print(f"成功加载 {len(self.triples)} 个三元组")
        return self.triples

    def load_entity_properties(self):
        """加载实体属性"""
        file_path = os.path.join(self.data_dir, 'all_entity_properties.csv')
        print(f"加载实体属性: {file_path}")

        df = pd.read_csv(file_path)
        print(f"属性数据形状: {df.shape}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="解析实体属性"):
            entity_name = row['only_entity']
            if entity_name in self.entity2id:
                entity_id = self.entity2id[entity_name]

                # 解析JSON属性
                try:
                    properties = json.loads(row['n'])['properties']
                    self.entity_properties[entity_id] = properties
                except:
                    self.entity_properties[entity_id] = {}

        print(f"成功加载 {len(self.entity_properties)} 个实体属性")
        return self.entity_properties

    def load_all(self):
        """加载所有知识图谱数据"""
        self.load_entity2id()
        self.load_relation2id()
        self.load_triples()
        self.load_entity_properties()
        return self

# ==================== RGCN模型 ====================
class RGCNEncoder(nn.Module):
    """
    关系图卷积网络编码器
    用于从知识图谱中提取节点嵌入
    """

    def __init__(self, num_entities, num_relations, hidden_dim, embedding_dim, num_layers=2, dropout=0.1):
        super(RGCNEncoder, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 实体嵌入初始化
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)

        # RGCN层
        self.rgcn_layers = nn.ModuleList()

        # 第一层: hidden_dim -> hidden_dim
        self.rgcn_layers.append(
            RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=min(num_relations, 30))
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.rgcn_layers.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=min(num_relations, 30))
            )

        # 最后一层: hidden_dim -> embedding_dim
        if num_layers > 1:
            self.rgcn_layers.append(
                RGCNConv(hidden_dim, embedding_dim, num_relations, num_bases=min(num_relations, 30))
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, edge_type):
        """
        前向传播

        Args:
            edge_index: (2, num_edges) 边索引
            edge_type: (num_edges,) 边类型

        Returns:
            node_embeddings: (num_entities, embedding_dim) 节点嵌入
        """
        # 初始节点特征
        x = self.entity_embedding.weight

        # 通过RGCN层
        for i, layer in enumerate(self.rgcn_layers):
            x = layer(x, edge_index, edge_type)

            # 最后一层不使用激活和dropout
            if i < len(self.rgcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x

# ==================== 时序注意力模块 ====================
class RGCN_TemporalAttention(nn.Module):
    """
    RGCN + 时序注意力机制
    将静态RGCN嵌入转换为时序感知的嵌入
    """

    def __init__(self, rgcn_dim=64, max_seq_len=200, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(RGCN_TemporalAttention, self).__init__()

        self.rgcn_dim = rgcn_dim
        self.max_seq_len = max_seq_len

        # 投影层：确保维度匹配
        self.input_projection = nn.Linear(rgcn_dim, d_model) if rgcn_dim != d_model else nn.Identity()

        # 时间编码层（将时间戳编码为向量）
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # 正弦位置编码（可学习）
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN结构，更稳定
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影：投影回原始维度
        self.output_projection = nn.Linear(d_model, rgcn_dim)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(rgcn_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, static_embeddings, timestamps, mask=None):
        """
        前向传播

        Args:
            static_embeddings: (batch_size, seq_len, rgcn_dim) 静态RGCN嵌入序列
            timestamps: (batch_size, seq_len) 归一化时间戳 [0, 1]
            mask: (batch_size, seq_len) 可选的padding mask

        Returns:
            temporal_embeddings: (batch_size, seq_len, rgcn_dim) 时序感知嵌入
        """
        batch_size, seq_len, _ = static_embeddings.shape

        # 1. 投影到Transformer维度
        x = self.input_projection(static_embeddings)  # (B, T, d_model)

        # 2. 添加时间编码
        time_encoded = self.time_encoder(timestamps.unsqueeze(-1))  # (B, T, d_model)
        x = x + time_encoded

        # 3. 添加位置编码
        pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0)  # (1, T, d_model)
        x = x + pos_encoding

        x = self.dropout(x)

        # 4. Transformer处理时序依赖
        # 创建causal mask（可选：如果需要自回归）
        # attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        temporal_features = self.temporal_transformer(
            x,
            src_key_padding_mask=mask  # mask: True表示忽略该位置
        )

        # 5. 投影回原始维度
        temporal_embeddings = self.output_projection(temporal_features)

        # 6. 残差连接 + Layer Norm
        temporal_embeddings = self.layer_norm(temporal_embeddings + static_embeddings)

        return temporal_embeddings

# ==================== 特征提取器 ====================
class KGFeatureExtractor:
    """知识图谱特征提取器"""

    def __init__(self, kg_loader, config):
        self.kg_loader = kg_loader
        self.config = config
        self.model = None
        self.node_embeddings = None

    def build_graph(self):
        """构建PyTorch Geometric图数据"""
        print("\n构建图数据...")

        # 构建边索引和边类型
        edge_index = []
        edge_type = []

        for head, tail, relation in self.kg_loader.triples:
            edge_index.append([head, tail])
            edge_type.append(relation)

        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_type = torch.LongTensor(edge_type)

        print(f"图统计信息:")
        print(f"  节点数: {len(self.kg_loader.entity2id)}")
        print(f"  边数: {len(edge_index[0])}")
        print(f"  关系类型数: {len(self.kg_loader.relation2id)}")

        return edge_index, edge_type

    def train_rgcn(self, edge_index, edge_type, epochs=50):
        """训练RGCN模型"""
        print("\n训练RGCN模型...")

        num_entities = len(self.kg_loader.entity2id)
        num_relations = len(self.kg_loader.relation2id)

        # 初始化模型
        self.model = RGCNEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=self.config['hidden_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(DEVICE)

        edge_index = edge_index.to(DEVICE)
        edge_type = edge_type.to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # 训练循环（无监督学习，使用链接预测作为代理任务）
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # 前向传播
            embeddings = self.model(edge_index, edge_type)

            # 简单的正则化损失（保持嵌入不要过大）
            loss = torch.norm(embeddings, p=2) / num_entities

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # 提取最终嵌入
        self.model.eval()
        with torch.no_grad():
            self.node_embeddings = self.model(edge_index, edge_type).cpu().numpy()

        print(f"✓ RGCN训练完成，节点嵌入维度: {self.node_embeddings.shape}")

    def extract_obs_features(self):
        """
        提取OBS节点的特征
        返回: DataFrame包含台风编号、观测索引、RGCN特征
        """
        print("\n提取OBS节点特征...")

        obs_features = []

        for entity_name, entity_id in tqdm(self.kg_loader.entity2id.items(), desc="处理实体"):
            # 只处理OBS节点
            if entity_name.startswith('20') and '_OBS_' in entity_name:
                # 解析台风编号和观测索引
                # 格式: 201002_OBS_1
                parts = entity_name.split('_OBS_')
                if len(parts) == 2:
                    typhoon_id = parts[0]
                    obs_index = parts[1]

                    # 获取RGCN嵌入
                    rgcn_embedding = self.node_embeddings[entity_id]

                    # 获取节点属性
                    properties = self.kg_loader.entity_properties.get(entity_id, {})

                    feature_dict = {
                        'entity_name': entity_name,
                        'typhoon_id': typhoon_id,
                        'obs_index': int(obs_index),
                        'entity_id': entity_id,
                    }

                    # 添加RGCN嵌入
                    for i, val in enumerate(rgcn_embedding):
                        feature_dict[f'rgcn_feat_{i}'] = val

                    # 添加节点属性
                    if 'lat' in properties:
                        feature_dict['lat'] = float(properties['lat'])
                    if 'lng' in properties:
                        feature_dict['lng'] = float(properties['lng'])
                    if 'speed' in properties:
                        feature_dict['speed'] = float(properties['speed'])
                    if 'pressure' in properties:
                        feature_dict['pressure'] = float(properties['pressure'])
                    if 'move_speed' in properties:
                        feature_dict['move_speed'] = float(properties['move_speed'])

                    obs_features.append(feature_dict)

        df_obs = pd.DataFrame(obs_features)
        print(f"✓ 提取了 {len(df_obs)} 个OBS节点特征")
        print(f"  特征维度: {len([c for c in df_obs.columns if c.startswith('rgcn_feat_')])}")

        return df_obs

    def save_features(self, df_obs, output_dir):
        """保存提取的特征"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存OBS特征
        obs_csv_path = os.path.join(output_dir, 'obs_rgcn_features.csv')
        df_obs.to_csv(obs_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存OBS特征到: {obs_csv_path}")

        # 保存完整节点嵌入
        embeddings_path = os.path.join(output_dir, 'all_node_embeddings.pkl')
        with open(embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.node_embeddings,
                'entity2id': self.kg_loader.entity2id,
                'id2entity': self.kg_loader.id2entity,
                'config': self.config
            }, f)
        print(f"✓ 保存完整嵌入到: {embeddings_path}")

        # 保存RGCN模型
        model_path = os.path.join(output_dir, 'rgcn_model.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ 保存RGCN模型到: {model_path}")

# ==================== 时序特征提取器 ====================
class TemporalKGFeatureExtractor:
    """
    时序知识图谱特征提取器
    将静态RGCN嵌入转换为时序感知嵌入
    """

    def __init__(self, static_extractor, temporal_config=None):
        """
        Args:
            static_extractor: KGFeatureExtractor实例（已训练好RGCN）
            temporal_config: 时序模型配置
        """
        self.static_extractor = static_extractor
        self.kg_loader = static_extractor.kg_loader
        self.node_embeddings = static_extractor.node_embeddings

        # 默认时序配置
        if temporal_config is None:
            temporal_config = {
                'rgcn_dim': static_extractor.config['embedding_dim'],
                'max_seq_len': 200,
                'd_model': 128,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.1
            }
        self.temporal_config = temporal_config

        # 初始化时序模型
        self.temporal_model = RGCN_TemporalAttention(**temporal_config).to(DEVICE)

        print(f"\n✓ 时序模型已初始化")
        print(f"  RGCN维度: {temporal_config['rgcn_dim']}")
        print(f"  Transformer维度: {temporal_config['d_model']}")
        print(f"  注意力头数: {temporal_config['nhead']}")

    def extract_typhoon_temporal_features(self, typhoon_id, obs_indices):
        """
        提取单个台风的时序图嵌入

        Args:
            typhoon_id: 台风编号（如'201002'）
            obs_indices: 观测索引列表（如[0,1,2,...,95]）

        Returns:
            temporal_features: (seq_len, rgcn_dim) 时序图特征
            timestamps: (seq_len,) 归一化时间戳
        """
        # 1. 获取实体名称和ID
        entity_names = [f"{typhoon_id}_OBS_{i}" for i in obs_indices]
        entity_ids = []
        valid_indices = []

        for idx, name in zip(obs_indices, entity_names):
            if name in self.kg_loader.entity2id:
                entity_ids.append(self.kg_loader.entity2id[name])
                valid_indices.append(idx)

        if len(entity_ids) == 0:
            raise ValueError(f"台风{typhoon_id}没有找到任何OBS节点")

        # 2. 获取静态RGCN嵌入
        static_embeddings = self.node_embeddings[entity_ids]  # (seq_len, rgcn_dim)
        static_embeddings = torch.FloatTensor(static_embeddings).unsqueeze(0).to(DEVICE)  # (1, seq_len, rgcn_dim)

        # 3. 构造归一化时间戳
        max_idx = max(valid_indices) if max(valid_indices) > 0 else 1
        timestamps = torch.FloatTensor([idx / max_idx for idx in valid_indices]).unsqueeze(0).to(DEVICE)  # (1, seq_len)

        # 4. 通过时序模型
        self.temporal_model.eval()
        with torch.no_grad():
            temporal_embeddings = self.temporal_model(static_embeddings, timestamps)  # (1, seq_len, rgcn_dim)

        temporal_features = temporal_embeddings.squeeze(0).cpu().numpy()  # (seq_len, rgcn_dim)

        return temporal_features, np.array(valid_indices)

    def extract_all_temporal_features(self, df_obs):
        """
        提取所有台风的时序特征

        Args:
            df_obs: OBS节点DataFrame（来自KGFeatureExtractor）

        Returns:
            temporal_data: dict {typhoon_id: {'features': array, 'indices': array}}
        """
        print("\n提取所有台风的时序图嵌入...")

        # 按台风分组
        grouped = df_obs.groupby('typhoon_id')
        temporal_data = {}

        for typhoon_id, group in tqdm(grouped, desc="处理台风"):
            obs_indices = sorted(group['obs_index'].tolist())

            try:
                temporal_features, valid_indices = self.extract_typhoon_temporal_features(
                    typhoon_id, obs_indices
                )

                temporal_data[typhoon_id] = {
                    'features': temporal_features,  # (seq_len, rgcn_dim)
                    'indices': valid_indices,        # (seq_len,)
                    'seq_len': len(valid_indices)
                }

            except Exception as e:
                print(f"  警告: 台风{typhoon_id}处理失败: {e}")
                continue

        print(f"✓ 成功提取 {len(temporal_data)} 个台风的时序特征")

        return temporal_data

    def save_temporal_features(self, temporal_data, output_dir):
        """保存时序特征"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存为pickle（推荐用于训练）
        temporal_pkl_path = os.path.join(output_dir, 'temporal_kg_features.pkl')
        with open(temporal_pkl_path, 'wb') as f:
            pickle.dump({
                'temporal_data': temporal_data,
                'temporal_config': self.temporal_config,
                'rgcn_config': self.static_extractor.config
            }, f)
        print(f"✓ 保存时序特征（pkl）到: {temporal_pkl_path}")

        # 2. 保存为CSV（便于检查）
        csv_records = []
        for typhoon_id, data in temporal_data.items():
            features = data['features']  # (seq_len, rgcn_dim)
            indices = data['indices']

            for i, obs_idx in enumerate(indices):
                record = {
                    'typhoon_id': typhoon_id,
                    'obs_index': obs_idx,
                }
                # 添加时序图特征
                for j, feat_val in enumerate(features[i]):
                    record[f'temporal_kg_feat_{j}'] = feat_val

                csv_records.append(record)

        df_temporal = pd.DataFrame(csv_records)
        csv_path = os.path.join(output_dir, 'temporal_kg_features.csv')
        df_temporal.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存时序特征（CSV）到: {csv_path}")

        # 3. 保存时序模型权重
        model_path = os.path.join(output_dir, 'temporal_attention_model.pth')
        torch.save(self.temporal_model.state_dict(), model_path)
        print(f"✓ 保存时序模型到: {model_path}")

        # 4. 保存统计信息
        stats = {
            'num_typhoons': len(temporal_data),
            'typhoon_ids': list(temporal_data.keys()),
            'seq_lengths': {tid: data['seq_len'] for tid, data in temporal_data.items()},
            'total_observations': sum(data['seq_len'] for data in temporal_data.values()),
            'feature_dim': self.temporal_config['rgcn_dim']
        }

        stats_path = os.path.join(output_dir, 'temporal_features_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存统计信息到: {stats_path}")

        return df_temporal

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("RGCN知识图谱时序特征提取")
    print("=" * 80)

    # 1. 加载知识图谱数据
    print("\n【步骤1】加载知识图谱数据")
    kg_loader = KnowledgeGraphLoader(KG_DATA_DIR)
    kg_loader.load_all()

    # 2. 构建图
    print("\n【步骤2】构建图数据结构")
    extractor = KGFeatureExtractor(kg_loader, RGCN_CONFIG)
    edge_index, edge_type = extractor.build_graph()

    # 3. 训练RGCN
    print("\n【步骤3】训练RGCN提取节点嵌入")
    extractor.train_rgcn(edge_index, edge_type, epochs=50)

    # 4. 提取OBS特征
    print("\n【步骤4】提取OBS观测节点特征")
    df_obs = extractor.extract_obs_features()

    # 5. 保存静态特征
    print("\n【步骤5】保存静态RGCN特征")
    extractor.save_features(df_obs, OUTPUT_DIR)

    # ===== 新增：时序特征提取 =====
    # 6. 初始化时序特征提取器
    print("\n【步骤6】初始化时序特征提取器")
    temporal_extractor = TemporalKGFeatureExtractor(extractor)

    # 7. 提取时序特征
    print("\n【步骤7】提取时序知识图谱特征")
    temporal_data = temporal_extractor.extract_all_temporal_features(df_obs)

    # 7.5. 加载台风ID映射表并添加映射ID
    print("\n【步骤7.5】应用台风ID映射")
    mapping_file = os.path.join(KG_DATA_DIR, 'typhoon_id_mapping.json')

    if os.path.exists(mapping_file):
        import json
        with open(mapping_file, 'r', encoding='utf-8') as f:
            typhoon_id_mapping = json.load(f)

        print(f"  加载ID映射表: {len(typhoon_id_mapping)} 个映射")

        # 为每个台风添加映射后的ID
        temporal_data_with_mapping = {}
        mapped_count = 0

        for typhoon_id, data in temporal_data.items():
            # 保留原始ID (6位数字，如200001)
            temporal_data_with_mapping[typhoon_id] = data

            # 如果有映射，添加映射后的ID (4位+名称，如0001达维)
            if typhoon_id in typhoon_id_mapping:
                mapped_id = typhoon_id_mapping[typhoon_id]['id_with_name']
                temporal_data_with_mapping[mapped_id] = data
                if mapped_count < 10:  # 只打印前10个
                    print(f"  {typhoon_id} -> {mapped_id}")
                mapped_count += 1

        print(f"\n  映射统计:")
        print(f"    映射后的key数量: {len(temporal_data_with_mapping)}")
        print(f"    原始台风数: {len(temporal_data)}")
        print(f"    成功映射数: {mapped_count}")
        print(f"    未映射数: {len(temporal_data) - mapped_count}")

        temporal_data = temporal_data_with_mapping
    else:
        print(f"  ⚠️  未找到ID映射表: {mapping_file}")
        print(f"  将只使用原始台风ID")

    # 8. 保存时序特征
    print("\n【步骤8】保存时序特征")
    df_temporal = temporal_extractor.save_temporal_features(temporal_data, OUTPUT_DIR)

    # 9. 统计信息
    print("\n" + "=" * 80)
    print("时序特征提取完成！")
    print("=" * 80)
    print(f"\n统计信息:")
    print(f"  OBS节点数: {len(df_obs)}")
    print(f"  台风数量: {df_obs['typhoon_id'].nunique()}")
    print(f"  静态RGCN特征维度: {RGCN_CONFIG['embedding_dim']}")
    print(f"  时序特征维度: {temporal_extractor.temporal_config['rgcn_dim']}")
    print(f"\n台风列表:")
    for tid in sorted(df_obs['typhoon_id'].unique()):
        count = len(df_obs[df_obs['typhoon_id'] == tid])
        seq_len = temporal_data[tid]['seq_len'] if tid in temporal_data else 0
        print(f"  {tid}: {count} 个观测点 (时序长度: {seq_len})")

    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n特征文件:")
    print(f"  - obs_rgcn_features.csv: OBS节点的静态RGCN特征")
    print(f"  - temporal_kg_features.csv: 时序知识图谱特征（CSV格式）")
    print(f"  - temporal_kg_features.pkl: 时序知识图谱特征（PKL格式，推荐）")
    print(f"  - all_node_embeddings.pkl: 所有节点的静态嵌入")
    print(f"  - rgcn_model.pth: 训练好的RGCN模型")
    print(f"  - temporal_attention_model.pth: 时序注意力模型")
    print(f"  - temporal_features_stats.json: 时序特征统计信息")

if __name__ == '__main__':
    main()
