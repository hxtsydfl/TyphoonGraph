# 台风预测系统 - PatchTST 模型

基于 PatchTST (Patch Time Series Transformer) 的台风路径和强度预测系统，支持传统特征和知识图谱增强特征两种方式。

## 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
  - [本地运行](#本地运行)
  - [Docker 部署](#docker-部署)
- [详细使用说明](#详细使用说明)
- [模型说明](#模型说明)

---

## 项目概述

本项目实现了基于深度学习的台风预测系统，主要功能包括：
- **台风路径预测**：预测未来时刻的经度和纬度
- **台风强度预测**：预测未来时刻的风速和气压
- **多特征融合**：支持传统气象特征和知识图谱增强特征
- **GPU/CPU 自适应**：自动检测运行环境并优化性能

## 项目结构

```
newwork/
│
├── data/                           # 数据目录
│   ├── input_data/                 # 原始输入数据
│   │   └── past_2000-2022typhoons/ # 历史台风数据（CSV格式）
│   ├── kg/                         # 知识图谱数据
│   │   ├── entity2id.txt           # 实体ID映射
│   │   ├── relation2id.txt         # 关系ID映射
│   │   └── all_triples.txt         # 知识图谱三元组
│   ├── kg_features/                # 知识图谱特征（RGCN提取）
│   └── processed_features/         # 处理后的特征数据
│
├── models/                         # 模型保存目录
│   └── patchtst/                   # PatchTST模型文件
│       ├── best_model.pth          # 最佳模型权重
│       ├── scaler.pkl              # 数据标准化器
│       └── training_history.pkl    # 训练历史记录
│
├── feature_engineering_optimized.py  # 特征工程脚本
├── rgcn_kg_feature_extraction.py     # RGCN知识图谱特征提取
├── train_patchtst.py                 # PatchTST训练（传统特征）
├── train_patchtst_with_kg.py         # PatchTST训练（KG增强）
├── test_patchtst.py                  # 模型测试与评估
│
├── requirements.txt                  # Python依赖
├── Dockerfile                        # Docker镜像构建文件
├── .dockerignore                     # Docker忽略文件
└── README.md                         # 项目说明文档
```

## 功能特性

### 1. 特征工程
- **传统气象特征**：经纬度、风速、气压、移动速度、移动方向等
- **时序特征**：时间步长、历史统计特征
- **知识图谱特征**（可选）：通过 RGCN 提取的高阶关系特征

### 2. 模型架构
- **PatchTST**：基于 Transformer 的时序预测模型
- **Patch 机制**：将时序数据分割成 patches，提高建模效率
- **多头注意力**：捕获不同时间尺度的依赖关系

### 3. 预测目标
- **经度 (lng)**：台风中心经度位置
- **纬度 (lat)**：台风中心纬度位置
- **风速 (speed)**：台风最大风速
- **气压 (pressure)**：台风中心气压

## 环境要求

### 硬件要求
- **GPU**（推荐）：NVIDIA GPU with CUDA 11.8+
- **CPU**：支持 CPU 训练（速度较慢）
- **内存**：建议 16GB+
- **存储**：至少 5GB 可用空间

### 软件要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（GPU 版本）
- Docker 19.03+（Docker 部署时）
- NVIDIA Docker Runtime（GPU Docker 部署时）

## 快速开始

### 本地运行

#### 1. 安装依赖

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 准备数据

确保数据文件已放置在正确的目录：
- 原始台风数据：`data/input_data/past_2000-2022typhoons/`
- 知识图谱数据（可选）：`data/kg/`

#### 3. 运行特征工程

```bash
# 生成传统特征
python feature_engineering_optimized.py

# （可选）生成知识图谱特征
python rgcn_kg_feature_extraction.py
```

#### 4. 训练模型

```bash
# 使用传统特征训练
python train_patchtst.py

# 或使用知识图谱增强特征训练
python train_patchtst_with_kg.py
```

#### 5. 测试模型

```bash
python test_patchtst.py
```

---

## Docker 部署

Docker 部署方式适合在 GPU 服务器上快速部署和运行，无需手动配置环境。

### 前置条件

#### 1. 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 验证安装
docker --version
```

#### 2. 安装 NVIDIA Docker（GPU 支持）

```bash
# 添加仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 重启 Docker
sudo systemctl restart docker

# 验证 GPU 可用
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 构建 Docker 镜像

#### 方式一：本地构建

```bash
# 在项目根目录执行
cd /path/to/newwork

# 构建镜像
docker build -t taifeng-model:latest .

# 查看镜像
docker images | grep taifeng-model
```

#### 方式二：使用国内镜像加速

```bash
# 构建时使用清华镜像源
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t taifeng-model:latest .
```

### 运行 Docker 容器

#### 1. 运行特征工程

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  taifeng-model:latest \
  python feature_engineering_optimized.py
```

**参数说明：**
- `--gpus all`：使用所有可用 GPU
- `-v $(pwd)/data:/app/data`：将本地 data 目录挂载到容器（数据持久化）
- `taifeng-model:latest`：镜像名称和标签
- `python feature_engineering_optimized.py`：要执行的命令

#### 2. 运行模型训练

```bash
# 使用传统特征训练
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst.py

# 使用知识图谱增强特征训练
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst_with_kg.py
```

**参数说明：**
- `-v $(pwd)/models:/app/models`：挂载模型保存目录，训练结果会保存到本地

#### 3. 后台运行（推荐）

```bash
# 后台运行训练，并命名容器
docker run -d --name taifeng-train \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst.py

# 查看实时日志
docker logs -f taifeng-train

# 查看容器状态
docker ps

# 停止容器
docker stop taifeng-train

# 删除容器
docker rm taifeng-train
```

#### 4. 运行模型测试

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python test_patchtst.py
```

#### 5. 交互式运行（调试）

```bash
# 进入容器 Shell
docker run -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest

# 在容器内可以手动执行命令
root@container:/app# python train_patchtst.py
root@container:/app# python test_patchtst.py
root@container:/app# exit
```

#### 6. 指定 GPU 设备

```bash
# 只使用 GPU 0
docker run --gpus '"device=0"' \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst.py

# 使用 GPU 0 和 1
docker run --gpus '"device=0,1"' \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst.py
```

### 镜像传输到 GPU 服务器

#### 方式一：导出/导入镜像文件

```bash
# 1. 在本地导出镜像
docker save -o taifeng-model.tar taifeng-model:latest

# 2. 压缩镜像（可选，减小文件大小）
gzip taifeng-model.tar

# 3. 传输到 GPU 服务器
scp taifeng-model.tar.gz user@gpu-server:/path/to/

# 4. 在 GPU 服务器上解压和加载
ssh user@gpu-server
gunzip taifeng-model.tar.gz
docker load -i taifeng-model.tar

# 5. 验证镜像已加载
docker images | grep taifeng-model
```

#### 方式二：使用镜像仓库

**Docker Hub（公开）：**
```bash
# 1. 登录 Docker Hub
docker login

# 2. 打标签
docker tag taifeng-model:latest your-username/taifeng-model:latest

# 3. 推送镜像
docker push your-username/taifeng-model:latest

# 4. 在 GPU 服务器拉取
docker pull your-username/taifeng-model:latest
```

**私有镜像仓库：**
```bash
# 1. 打标签（替换为你的私有仓库地址）
docker tag taifeng-model:latest registry.example.com/taifeng-model:latest

# 2. 推送到私有仓库
docker push registry.example.com/taifeng-model:latest

# 3. 在 GPU 服务器拉取
docker pull registry.example.com/taifeng-model:latest
```

### Docker 常用命令

```bash
# 查看运行中的容器
docker ps

# 查看所有容器（包括已停止的）
docker ps -a

# 查看容器日志
docker logs taifeng-train
docker logs -f taifeng-train  # 实时跟踪日志
docker logs --tail 100 taifeng-train  # 查看最后100行

# 查看容器资源使用情况
docker stats taifeng-train

# 进入正在运行的容器
docker exec -it taifeng-train /bin/bash

# 从容器复制文件到主机
docker cp taifeng-train:/app/models/patchtst/best_model.pth ./

# 从主机复制文件到容器
docker cp ./config.py taifeng-train:/app/

# 停止容器
docker stop taifeng-train

# 启动已停止的容器
docker start taifeng-train

# 重启容器
docker restart taifeng-train

# 删除容器
docker rm taifeng-train

# 删除镜像
docker rmi taifeng-model:latest

# 清理未使用的资源
docker system prune -a  # 清理所有未使用的镜像和容器
docker system df        # 查看 Docker 磁盘使用情况
```

### 高级用法

#### 1. 设置环境变量

```bash
docker run --gpus all \
  -e BATCH_SIZE=64 \
  -e EPOCHS=200 \
  -e LEARNING_RATE=0.0005 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst.py
```

#### 2. 限制资源使用

```bash
# 限制内存和 CPU
docker run --gpus all \
  --memory=16g \
  --memory-swap=16g \
  --cpus=8 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst.py
```

#### 3. 使用 Docker Compose（批量管理）

创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'

services:
  taifeng-train:
    image: taifeng-model:latest
    container_name: taifeng-train
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    command: python train_patchtst.py
```

运行：
```bash
docker-compose up -d  # 后台运行
docker-compose logs -f  # 查看日志
docker-compose down  # 停止并删除容器
```

---

## 详细使用说明

### 1. 数据准备

#### 输入数据格式

原始台风数据应为 CSV 格式，包含以下字段：
- `time`：时间戳
- `lng`：经度
- `lat`：纬度
- `speed`：风速 (m/s)
- `pressure`：气压 (hPa)

示例：
```csv
time,lng,lat,speed,pressure
2022-01-01 00:00:00,125.5,25.3,30.0,980.0
2022-01-01 06:00:00,126.2,25.8,32.0,978.0
...
```

#### 知识图谱数据（可选）

如需使用 RGCN 增强特征，需准备：
- `entity2id.txt`：实体到ID的映射
- `relation2id.txt`：关系到ID的映射
- `all_triples.txt`：三元组 (head, relation, tail)

### 2. 特征工程

```bash
# 运行特征工程脚本
python feature_engineering_optimized.py
```

**生成的文件：**
- `data/processed_features/train_sequences.pkl`：训练集
- `data/processed_features/val_sequences.pkl`：验证集
- `data/processed_features/test_sequences.pkl`：测试集
- `data/processed_features/scaler.pkl`：标准化器

### 3. 模型训练

#### 训练参数配置

在 `train_patchtst.py` 中修改配置：

```python
# 模型参数
MODEL_CONFIG = {
    'patch_len': 4,          # patch长度
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
```

#### 开始训练

```bash
# 本地训练
python train_patchtst.py

# Docker 训练
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python train_patchtst.py
```

**训练输出：**
- 模型权重：`models/patchtst/best_model.pth`
- 标准化器：`models/patchtst/scaler.pkl`
- 训练历史：`models/patchtst/training_history.pkl`

### 4. 模型测试

```bash
# 本地测试
python test_patchtst.py

# Docker 测试
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  taifeng-model:latest \
  python test_patchtst.py
```

**测试指标：**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

## 模型说明

### PatchTST 架构

1. **Patch Embedding**：将时序数据分割成固定长度的 patches
2. **Transformer Encoder**：使用多头自注意力机制捕获时序依赖
3. **预测头**：将编码后的特征映射到预测目标

### 特征说明

#### 传统特征（12维）
- 位置特征：经度、纬度
- 强度特征：风速、气压
- 移动特征：移动速度、移动方向
- 变化特征：速度变化、方向变化、强度变化
- 时序特征：时间步长、历史统计特征

#### 知识图谱增强特征（可选）
- 通过 RGCN 从台风知识图谱中提取的高阶关系特征
- 维度：64 或 128（可配置）

---

## 常见问题

### 1. GPU 内存不足

**解决方案：**
- 减小 `batch_size`
- 减小模型维度 `d_model`
- 使用梯度累积

### 2. Docker GPU 不可用

**检查步骤：**
```bash
# 1. 检查 NVIDIA 驱动
nvidia-smi

# 2. 检查 nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 3. 重启 Docker
sudo systemctl restart docker
```

### 3. 数据路径错误

确保数据目录正确挂载：
```bash
# 检查挂载路径
docker run -it --gpus all \
  -v $(pwd)/data:/app/data \
  taifeng-model:latest \
  ls -la /app/data
```

### 4. 依赖安装失败

**使用国内镜像源：**
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

**更新日期：** 2025-12-19
