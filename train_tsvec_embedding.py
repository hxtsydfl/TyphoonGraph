"""
TSvec时序知识图谱嵌入 - 简洁版
为每个台风的每个时间步生成64维嵌入向量
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置 ====================
DATA_DIR = './data/temporal_kg'
OUTPUT_DIR = './data/temporal_kg_embeddings'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'embedding_dim': 64,       # 输出嵌入维度（与PatchTST对齐）
    'batch_size': 512,
    'epochs': 100,
    'learning_rate': 0.001,
    'negative_samples': 5,
}

# ==================== 数据加载 ====================
def load_temporal_kg_data(data_dir):
    """加载时序知识图谱数据"""
    print("加载时序知识图谱数据...")

    # 1. 加载实体
    with open(os.path.join(data_dir, 'temporal_entities.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        num_entities = int(lines[0])
        entity2id = {}
        for line in lines[1:]:
            if line.strip():
                entity, eid = line.strip().split('\t')
                entity2id[entity] = int(eid)
    print(f"  实体数: {len(entity2id)}")

    # 2. 加载关系
    with open(os.path.join(data_dir, 'temporal_relations.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        num_relations = int(lines[0])
        relation2id = {}
        for line in lines[1:]:
            if line.strip():
                relation, rid = line.strip().split('\t')
                relation2id[relation] = int(rid)
    print(f"  关系数: {len(relation2id)}")

    # 3. 加载时序三元组
    triples = []
    with open(os.path.join(data_dir, 'temporal_triples.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines[1:], desc="  加载三元组"):
            if line.strip():
                h, t, r, time = line.strip().split('\t')
                triples.append([int(h), int(t), int(r)])
    print(f"  三元组数: {len(triples)}")

    # 4. 加载实体-时间映射
    df = pd.read_csv(os.path.join(data_dir, 'entity_time_mapping.csv'))
    entity_time_map = {}
    for _, row in df.iterrows():
        entity_time_map[row['entity_id']] = {
            'typhoon_id': row['typhoon_id'],
            'time_step': row['time_step'],
            'entity_name': row['entity_name']
        }
    print(f"  时序实体数: {len(entity_time_map)}")

    return entity2id, relation2id, triples, entity_time_map

# ==================== TSvec模型 ====================
class TSvec(nn.Module):
    """时序知识图谱嵌入模型（TransE风格）"""

    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TSvec, self).__init__()

        # 实体嵌入（这就是我们要的输出）
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # 关系嵌入
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 初始化
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head, relation, tail):
        """TransE评分函数: score = -||h + r - t||"""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        score = -torch.norm(h + r - t, p=2, dim=-1)
        return score

# ==================== 数据集 ====================
class TripleDataset(Dataset):
    def __init__(self, triples):
        self.triples = torch.LongTensor(triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

# ==================== 训练 ====================
def train_tsvec(model, data_loader, num_entities, config):
    """训练TSvec模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    print(f"\n开始训练TSvec模型...")
    print(f"  设备: {DEVICE}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  轮数: {config['epochs']}")

    model.train()

    for epoch in range(config['epochs']):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for batch in pbar:
            head, tail, relation = batch[:, 0], batch[:, 1], batch[:, 2]
            head, tail, relation = head.to(DEVICE), tail.to(DEVICE), relation.to(DEVICE)

            # 正样本得分
            pos_scores = model(head, relation, tail)

            # 负采样
            batch_size = head.size(0)
            neg_scores_list = []

            for _ in range(config['negative_samples']):
                if np.random.rand() < 0.5:
                    neg_head = torch.randint(0, num_entities, (batch_size,)).to(DEVICE)
                    neg_scores = model(neg_head, relation, tail)
                else:
                    neg_tail = torch.randint(0, num_entities, (batch_size,)).to(DEVICE)
                    neg_scores = model(head, relation, neg_tail)

                neg_scores_list.append(neg_scores)

            neg_scores = torch.stack(neg_scores_list, dim=1)

            # Margin ranking loss
            margin = 1.0
            loss = F.relu(margin - pos_scores.unsqueeze(1) + neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

    print("✓ 训练完成")

# ==================== 生成台风嵌入 ====================
def generate_typhoon_embeddings(model, entity_time_map, config):
    """为每个台风的每个时间步生成嵌入"""
    print("\n生成台风时序嵌入...")

    model.eval()

    # 按台风分组
    typhoon_entities = {}
    for eid, info in entity_time_map.items():
        typhoon_id = info['typhoon_id']
        time_step = info['time_step']

        if typhoon_id not in typhoon_entities:
            typhoon_entities[typhoon_id] = []

        typhoon_entities[typhoon_id].append({
            'entity_id': eid,
            'time_step': time_step
        })

    # 为每个台风生成嵌入
    typhoon_embeddings = {}

    with torch.no_grad():
        for typhoon_id in tqdm(sorted(typhoon_entities.keys()), desc="生成嵌入"):
            entities = typhoon_entities[typhoon_id]

            # 按时间步排序
            entities = sorted(entities, key=lambda x: x['time_step'])

            entity_ids = torch.LongTensor([e['entity_id'] for e in entities]).to(DEVICE)

            # 获取嵌入
            embeddings = model.entity_embeddings(entity_ids).cpu().numpy()

            typhoon_embeddings[typhoon_id] = {
                'features': embeddings,  # (seq_len, 64) - 关键：与PatchTST对齐
                'indices': np.array([e['time_step'] for e in entities]),
                'seq_len': len(entities)
            }

    print(f"✓ 生成了 {len(typhoon_embeddings)} 个台风的嵌入")

    return typhoon_embeddings

# ==================== 保存结果 ====================
def save_results(model, typhoon_embeddings, config, output_dir):
    """保存结果（格式与PatchTST对齐）"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n保存结果到 {output_dir}")

    # 1. 保存模型
    model_path = os.path.join(output_dir, 'tsvec_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ 模型: {model_path}")

    # 2. 保存台风嵌入（PKL格式 - 与PatchTST对齐）
    embeddings_pkl = os.path.join(output_dir, 'temporal_kg_features.pkl')
    with open(embeddings_pkl, 'wb') as f:
        pickle.dump({
            'temporal_data': typhoon_embeddings,  # 与rgcn_kg_feature_extraction.py格式一致
            'temporal_config': {'rgcn_dim': config['embedding_dim']},  # PatchTST读取这个
            'rgcn_config': {'embedding_dim': config['embedding_dim']}
        }, f)
    print(f"  ✓ 嵌入(PKL): {embeddings_pkl}")
    print(f"     格式与PatchTST对齐 ✓")

    # 3. 保存CSV（便于查看）
    csv_records = []
    for typhoon_id, data in typhoon_embeddings.items():
        features = data['features']
        indices = data['indices']

        for i, time_step in enumerate(indices):
            record = {'typhoon_id': typhoon_id, 'time_step': time_step}
            for j, val in enumerate(features[i]):
                record[f'tsvec_feat_{j}'] = val
            csv_records.append(record)

    df = pd.DataFrame(csv_records)
    csv_path = os.path.join(output_dir, 'temporal_kg_features.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 嵌入(CSV): {csv_path}")

    # 4. 保存统计
    stats = {
        'num_typhoons': len(typhoon_embeddings),
        'embedding_dim': config['embedding_dim'],
        'total_time_steps': sum(d['seq_len'] for d in typhoon_embeddings.values()),
        'typhoon_ids': list(typhoon_embeddings.keys())[:10],  # 只保存前10个
        'config': config
    }

    stats_path = os.path.join(output_dir, 'tsvec_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 统计信息: {stats_path}")

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("TSvec时序知识图谱嵌入")
    print("=" * 80)

    # 1. 加载数据
    print("\n【步骤1】加载数据")
    entity2id, relation2id, triples, entity_time_map = load_temporal_kg_data(DATA_DIR)

    # 2. 创建数据集
    print("\n【步骤2】创建数据集")
    dataset = TripleDataset(triples)
    data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    print(f"  数据集大小: {len(dataset)}")

    # 3. 创建模型
    print("\n【步骤3】初始化TSvec模型")
    model = TSvec(
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        embedding_dim=CONFIG['embedding_dim']
    ).to(DEVICE)

    print(f"  实体数: {len(entity2id)}")
    print(f"  关系数: {len(relation2id)}")
    print(f"  嵌入维度: {CONFIG['embedding_dim']}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 训练
    print("\n【步骤4】训练模型")
    train_tsvec(model, data_loader, len(entity2id), CONFIG)

    # 5. 生成嵌入
    print("\n【步骤5】生成台风嵌入")
    typhoon_embeddings = generate_typhoon_embeddings(model, entity_time_map, CONFIG)

    # 6. 保存
    print("\n【步骤6】保存结果")
    save_results(model, typhoon_embeddings, CONFIG, OUTPUT_DIR)

    # 7. 完成
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n统计:")
    print(f"  台风数: {len(typhoon_embeddings)}")
    print(f"  总时间步: {sum(d['seq_len'] for d in typhoon_embeddings.values())}")
    print(f"  嵌入维度: {CONFIG['embedding_dim']}")

    print(f"\n输出文件:")
    print(f"  - temporal_kg_features.pkl (用于PatchTST训练)")
    print(f"  - temporal_kg_features.csv (便于查看)")
    print(f"  - tsvec_model.pth (模型权重)")

    print(f"\n使用方式:")
    print(f"  将 {OUTPUT_DIR}/temporal_kg_features.pkl")
    print(f"  复制到 ./data/kg_features/temporal_kg_features.pkl")
    print(f"  然后运行: python train_patchtst_with_kg.py")

if __name__ == '__main__':
    main()
