# 专家轨迹模仿学习服务文档

## 1. 核心文件
`afruits/core/services/imitation_learning_service.py`

## 2. 类结构
```python
class ImitationLearningService:
    def __init__(config, logger)  # 初始化服务
    def train_model(expert_trajectories, model_config)  # 模型训练入口
    def generate_trajectory(model_id, context)  # 轨迹生成
    def save_model(model_id, save_path)  # 模型保存
    def load_model(model_path, model_type)  # 模型加载
```

## 3. 训练方法详解

### 3.1 标准训练
```python
def _train_standard(model_type, expert_trajectories, model_config):
    # 支持模型：
    # - AutoencoderModel: 自编码器轨迹建模
    # - TransformerModel: 序列轨迹建模
    # - DiffusionTrajGenerator: 扩散轨迹生成
    # - VAETrajGenerator: 变分自编码轨迹生成
    
    # 训练流程：
    # 1. 准备数据加载器
    # 2. 配置模型参数
    # 3. 执行训练循环
    # 4. 返回训练指标
```

### 3.2 进化训练
```python
def _train_evolutionary(model_type, expert_trajectories, model_config):
    # 专用于TransformerModel的进化训练
    # 核心流程：
    # 1. 初始化种群 (population_size)
    # 2. 训练种群模型
    # 3. 评估适应度 (fitness_threshold)
    # 4. 选择/交叉/变异
    # 5. 返回最优模型
```

### 3.3 增量训练
```python
def _train_incremental(model_type, expert_trajectories, model_config):
    # 支持预训练模型微调
    # 关键技术：
    # - 弹性权重更新 (elastic_update)
    # - 经验回放 (replay_strategy)
    # - 漂移检测 (monitor_data_stream)
```

## 4. 轨迹生成流程
```python
def generate_trajectory(model_id, context):
    # 步骤：
    # 1. 加载指定模型
    # 2. 解析上下文参数
    # 3. 执行生成 (Diffusion/VAE)
    # 4. 物理约束检查
    # 5. 返回轨迹数据
```

## 5. 模型管理
| 方法 | 参数 | 返回值 | 功能说明 |
|------|------|--------|----------|
| save_model | model_id, save_path | 保存路径 | 持久化模型到磁盘 |
| load_model | model_path, model_type | 模型ID | 从磁盘加载模型 |
| get_model | model_id | 模型对象 | 获取内存中模型 |
| get_all_models | 无 | 模型字典 | 获取所有已加载模型 |