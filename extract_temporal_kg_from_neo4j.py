"""
从Neo4j数据库导出时序对齐的知识图谱数据
用于TComplEx时序知识图谱嵌入模型

关键改进：
1. 提取时序信息（台风ID、观测序号作为时间戳）
2. 确保台风ID与训练数据格式对齐（4位数字+中文名）
3. 生成时序三元组 (head, relation, tail, timestamp)
4. 为每个台风的每个时间步生成对齐的实体嵌入索引

输出文件：
- temporal_entities.txt: 实体ID映射
- temporal_relations.txt: 关系ID映射
- temporal_triples.txt: 时序三元组 (head_id, tail_id, rel_id, timestamp)
- typhoon_temporal_index.json: 台风ID到时间步的索引映射
- entity_time_mapping.csv: 实体-时间对应表（用于特征对齐）
"""

import os
import sys
from neo4j import GraphDatabase
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置 ====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # 请修改为您的密码

OUTPUT_DIR = './data/temporal_kg'

# ==================== Neo4j时序KG导出器 ====================
class TemporalKGExporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def export_temporal_kg(self, output_dir):
        """导出时序对齐的知识图谱"""

        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print("从Neo4j导出时序对齐的知识图谱数据")
        print("=" * 80)

        # ========== 步骤1: 创建台风ID映射表 ==========
        print("\n【步骤1】创建台风ID映射表（6位 <-> 4位+名称）")
        typhoon_id_mapping = self._create_typhoon_id_mapping()

        # ========== 步骤2: 提取所有OBS节点及其时序信息 ==========
        print("\n【步骤2】提取OBS节点的时序数据")
        obs_nodes = self._extract_obs_nodes_with_time(typhoon_id_mapping)

        # ========== 步骤3: 构建实体映射 ==========
        print("\n【步骤3】构建实体映射")
        entity2id, id2entity, entity_time_mapping = self._build_entity_mapping(obs_nodes)

        # ========== 步骤4: 提取关系 ==========
        print("\n【步骤4】提取关系类型")
        relation2id, id2relation = self._extract_relations()

        # ========== 步骤5: 提取时序三元组 ==========
        print("\n【步骤5】提取时序三元组")
        temporal_triples = self._extract_temporal_triples(
            obs_nodes, entity2id, relation2id, typhoon_id_mapping
        )

        # ========== 步骤6: 构建台风时序索引 ==========
        print("\n【步骤6】构建台风时序索引（用于特征对齐）")
        typhoon_temporal_index = self._build_typhoon_temporal_index(obs_nodes, typhoon_id_mapping)

        # ========== 步骤7: 保存文件 ==========
        print("\n【步骤7】保存文件")
        self._save_files(
            output_dir,
            entity2id,
            relation2id,
            temporal_triples,
            typhoon_temporal_index,
            entity_time_mapping,
            typhoon_id_mapping
        )

        return {
            'entity2id': entity2id,
            'relation2id': relation2id,
            'temporal_triples': temporal_triples,
            'typhoon_temporal_index': typhoon_temporal_index,
            'entity_time_mapping': entity_time_mapping
        }

    def _create_typhoon_id_mapping(self):
        """创建台风ID映射：6位tfbh -> 4位+名称"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:OBS)
                WHERE n.tfbh IS NOT NULL AND n.name IS NOT NULL
                RETURN DISTINCT n.tfbh as tfbh, n.name as typhoon_name
                ORDER BY n.tfbh
            """)

            typhoon_id_mapping = {}

            for record in result:
                tfbh = record['tfbh']  # 6位：200401
                name = record['typhoon_name']  # 中文名：苏特

                if tfbh and name:
                    # 构造4位+名称格式
                    if tfbh.startswith('20') and len(tfbh) == 6:
                        short_id = tfbh[2:]  # 去掉'20'前缀：0401
                    else:
                        short_id = tfbh

                    id_with_name = f"{short_id}{name}"  # 0401苏特

                    typhoon_id_mapping[tfbh] = {
                        'tfbh': tfbh,
                        'short_id': short_id,
                        'name': name,
                        'id_with_name': id_with_name
                    }

            print(f"  映射台风数量: {len(typhoon_id_mapping)}")
            print(f"  示例映射: {list(typhoon_id_mapping.values())[:3]}")

            return typhoon_id_mapping

    def _extract_obs_nodes_with_time(self, typhoon_id_mapping):
        """提取OBS节点及其时序信息"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:OBS)
                RETURN
                    n.tfbh as tfbh,
                    n.observation_index as obs_index,
                    n.lat as lat,
                    n.lng as lng,
                    n.speed as speed,
                    n.pressure as pressure,
                    n.move_speed as move_speed,
                    n.move_dir as move_dir,
                    n.only_entity as entity_name,
                    id(n) as node_id
                ORDER BY n.tfbh, n.observation_index
            """)

            obs_nodes = []
            skipped = 0

            for record in result:
                tfbh = record['tfbh']

                # 转换为4位+名称格式
                if tfbh in typhoon_id_mapping:
                    typhoon_id = typhoon_id_mapping[tfbh]['id_with_name']
                else:
                    # 如果映射表中没有，尝试构造
                    if tfbh and tfbh.startswith('20') and len(tfbh) == 6:
                        typhoon_id = tfbh[2:]  # 只保留4位
                    else:
                        typhoon_id = tfbh
                    skipped += 1

                obs_index = record['obs_index']
                if obs_index is None:
                    obs_index = 0

                obs_nodes.append({
                    'typhoon_id': typhoon_id,  # 4位+名称格式
                    'tfbh': tfbh,  # 原始6位格式
                    'obs_index': int(obs_index),  # 观测序号作为时间戳
                    'node_id': record['node_id'],
                    'lat': record['lat'],
                    'lng': record['lng'],
                    'speed': record['speed'],
                    'pressure': record['pressure'],
                    'move_speed': record['move_speed'],
                    'move_dir': record['move_dir'],
                    'entity_name': record['entity_name']
                })

            print(f"  OBS节点数: {len(obs_nodes)}")
            print(f"  无法映射的节点: {skipped}")
            print(f"  示例: {obs_nodes[:3]}")

            return obs_nodes

    def _build_entity_mapping(self, obs_nodes):
        """构建实体映射和时序对齐表"""

        entity2id = {}
        id2entity = {}
        entity_time_mapping = []  # 用于对齐

        current_id = 0

        # 为每个OBS节点创建实体
        for obs in tqdm(obs_nodes, desc="构建实体映射"):
            typhoon_id = obs['typhoon_id']
            obs_index = obs['obs_index']

            # 实体名称格式：typhoon_id_OBS_index
            entity_name = f"{typhoon_id}_OBS_{obs_index}"

            if entity_name not in entity2id:
                entity2id[entity_name] = current_id
                id2entity[current_id] = entity_name

                # 记录实体-时间映射（用于特征对齐）
                entity_time_mapping.append({
                    'entity_id': current_id,
                    'entity_name': entity_name,
                    'typhoon_id': typhoon_id,
                    'time_step': obs_index,
                    'lat': obs['lat'],
                    'lng': obs['lng'],
                    'speed': obs['speed'],
                    'pressure': obs['pressure'],
                    'move_speed': obs['move_speed'],
                    'move_dir': obs['move_dir']
                })

                current_id += 1

        # 添加其他实体（台风实例、灾害类别等）
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE NOT n:OBS
                RETURN labels(n) as labels, properties(n) as props, id(n) as node_id
            """)

            for record in result:
                labels = record['labels']
                props = record['props']
                node_id = record['node_id']

                if 'typhoon_instance' in labels:
                    name = props.get('name', f'typhoon_{node_id}')
                    entity_name = f"{name}_instance"
                elif labels:
                    label = labels[0]
                    name = props.get('name', props.get('value', f'{label}_{node_id}'))
                    entity_name = f"{name}_{label}"
                else:
                    entity_name = f"entity_{node_id}"

                if entity_name not in entity2id:
                    entity2id[entity_name] = current_id
                    id2entity[current_id] = entity_name
                    current_id += 1

        print(f"  总实体数: {len(entity2id)}")
        print(f"  时序实体数: {len(entity_time_mapping)}")

        return entity2id, id2entity, entity_time_mapping

    def _extract_relations(self):
        """提取关系类型"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN DISTINCT type(r) as rel_type
            """)

            relations = [record['rel_type'] for record in result]

        relation2id = {rel: idx for idx, rel in enumerate(relations)}
        id2relation = {idx: rel for rel, idx in relation2id.items()}

        print(f"  关系类型数: {len(relation2id)}")
        print(f"  关系列表: {list(relation2id.keys())}")

        return relation2id, id2relation

    def _extract_temporal_triples(self, obs_nodes, entity2id, relation2id, typhoon_id_mapping):
        """提取时序三元组"""

        # 创建node_id到OBS信息的映射
        node_id_to_obs = {obs['node_id']: obs for obs in obs_nodes}

        with self.driver.session() as session:
            result = session.run("""
                MATCH (h)-[r]->(t)
                RETURN id(h) as head_node_id, type(r) as rel_type, id(t) as tail_node_id
            """)

            temporal_triples = []
            skipped = 0

            for record in tqdm(result, desc="提取三元组"):
                head_node_id = record['head_node_id']
                tail_node_id = record['tail_node_id']
                rel_type = record['rel_type']

                # 确定时间戳
                timestamp = 0  # 默认时间戳

                # 如果head或tail是OBS节点，使用其observation_index作为时间戳
                if head_node_id in node_id_to_obs:
                    obs = node_id_to_obs[head_node_id]
                    timestamp = obs['obs_index']
                    head_entity_name = f"{obs['typhoon_id']}_OBS_{obs['obs_index']}"
                elif tail_node_id in node_id_to_obs:
                    obs = node_id_to_obs[tail_node_id]
                    timestamp = obs['obs_index']
                    tail_entity_name = f"{obs['typhoon_id']}_OBS_{obs['obs_index']}"

                # 获取实体ID
                head_id = None
                tail_id = None

                # 查找head实体
                if head_node_id in node_id_to_obs:
                    obs = node_id_to_obs[head_node_id]
                    head_entity_name = f"{obs['typhoon_id']}_OBS_{obs['obs_index']}"
                    head_id = entity2id.get(head_entity_name)

                # 查找tail实体
                if tail_node_id in node_id_to_obs:
                    obs = node_id_to_obs[tail_node_id]
                    tail_entity_name = f"{obs['typhoon_id']}_OBS_{obs['obs_index']}"
                    tail_id = entity2id.get(tail_entity_name)

                rel_id = relation2id.get(rel_type)

                if head_id is not None and tail_id is not None and rel_id is not None:
                    temporal_triples.append({
                        'head': head_id,
                        'tail': tail_id,
                        'relation': rel_id,
                        'timestamp': timestamp
                    })
                else:
                    skipped += 1

            print(f"  有效三元组: {len(temporal_triples)}")
            print(f"  跳过三元组: {skipped}")

            return temporal_triples

    def _build_typhoon_temporal_index(self, obs_nodes, typhoon_id_mapping):
        """构建台风ID到时间步的索引（用于特征对齐）"""

        typhoon_temporal_index = defaultdict(lambda: {'time_steps': [], 'entity_ids': []})

        for obs in obs_nodes:
            typhoon_id = obs['typhoon_id']
            obs_index = obs['obs_index']
            entity_name = f"{typhoon_id}_OBS_{obs_index}"

            typhoon_temporal_index[typhoon_id]['time_steps'].append(obs_index)

        # 转换为普通dict并排序
        result = {}
        for typhoon_id, data in typhoon_temporal_index.items():
            sorted_steps = sorted(data['time_steps'])
            result[typhoon_id] = {
                'time_steps': sorted_steps,
                'num_steps': len(sorted_steps),
                'min_time': min(sorted_steps),
                'max_time': max(sorted_steps)
            }

        print(f"  索引的台风数: {len(result)}")
        print(f"  示例索引: {list(result.items())[:3]}")

        return result

    def _save_files(self, output_dir, entity2id, relation2id, temporal_triples,
                    typhoon_temporal_index, entity_time_mapping, typhoon_id_mapping):
        """保存所有文件"""

        # 1. 保存实体映射
        entity_file = os.path.join(output_dir, 'temporal_entities.txt')
        with open(entity_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(entity2id)}\n")
            for entity, eid in sorted(entity2id.items(), key=lambda x: x[1]):
                f.write(f"{entity}\t{eid}\n")
        print(f"  ✓ 实体映射: {entity_file}")

        # 2. 保存关系映射
        relation_file = os.path.join(output_dir, 'temporal_relations.txt')
        with open(relation_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(relation2id)}\n")
            for rel, rid in sorted(relation2id.items(), key=lambda x: x[1]):
                f.write(f"{rel}\t{rid}\n")
        print(f"  ✓ 关系映射: {relation_file}")

        # 3. 保存时序三元组
        triples_file = os.path.join(output_dir, 'temporal_triples.txt')
        with open(triples_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(temporal_triples)}\n")
            for triple in temporal_triples:
                f.write(f"{triple['head']}\t{triple['tail']}\t{triple['relation']}\t{triple['timestamp']}\n")
        print(f"  ✓ 时序三元组: {triples_file}")

        # 4. 保存台风时序索引
        index_file = os.path.join(output_dir, 'typhoon_temporal_index.json')
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(typhoon_temporal_index, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 台风时序索引: {index_file}")

        # 5. 保存实体-时间映射表
        mapping_file = os.path.join(output_dir, 'entity_time_mapping.csv')
        df_mapping = pd.DataFrame(entity_time_mapping)
        df_mapping.to_csv(mapping_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ 实体时间映射: {mapping_file}")

        # 6. 保存台风ID映射表
        typhoon_mapping_file = os.path.join(output_dir, 'typhoon_id_mapping.json')
        with open(typhoon_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(typhoon_id_mapping, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 台风ID映射: {typhoon_mapping_file}")

        # 7. 保存统计信息
        stats_file = os.path.join(output_dir, 'kg_statistics.json')
        stats = {
            'num_entities': len(entity2id),
            'num_relations': len(relation2id),
            'num_temporal_triples': len(temporal_triples),
            'num_typhoons': len(typhoon_temporal_index),
            'num_temporal_entities': len(entity_time_mapping),
            'time_range': {
                'min': min(t['timestamp'] for t in temporal_triples) if temporal_triples else 0,
                'max': max(t['timestamp'] for t in temporal_triples) if temporal_triples else 0
            }
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 统计信息: {stats_file}")

# ==================== 主函数 ====================
def main():
    print("\n请配置Neo4j连接信息：")
    print(f"  URI: {NEO4J_URI}")
    print(f"  User: {NEO4J_USER}")
    print(f"  Password: ******")
    print("\n如果需要修改，请编辑脚本顶部的配置项\n")

    response = input("开始导出时序知识图谱？(y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return

    try:
        exporter = TemporalKGExporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        result = exporter.export_temporal_kg(OUTPUT_DIR)
        exporter.close()

        print("\n" + "=" * 80)
        print("时序知识图谱导出完成！")
        print("=" * 80)
        print(f"\n输出目录: {OUTPUT_DIR}")
        print(f"\n统计信息:")
        print(f"  实体数: {len(result['entity2id'])}")
        print(f"  关系数: {len(result['relation2id'])}")
        print(f"  时序三元组数: {len(result['temporal_triples'])}")
        print(f"  台风数: {len(result['typhoon_temporal_index'])}")
        print(f"  时序实体数: {len(result['entity_time_mapping'])}")

        print("\n下一步：运行 TComplEx 模型训练")
        print("  python train_tcomplex_embedding.py")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查：")
        print("  1. Neo4j数据库是否运行")
        print("  2. 连接信息是否正确")
        print("  3. 数据库中是否有OBS节点")

if __name__ == '__main__':
    main()
