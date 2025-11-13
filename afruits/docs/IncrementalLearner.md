# 增量学习器文档

## 1. 核心文件
`afruits/utils/IncrementalLearner.py`

## 2. 类结构
```python
class IncrementalLearner:
    def __init__(base_model, model_type, optimizer_config, memory_buffer_size, replay_strategy)  # 初始化
    def setup_model(pretrained_weights)  # 模型设置
    def monitor_data_stream(new_data, drift_threshold)  # 数据流监控
    def elastic_update(model, importance_weights)  # 弹性权重更新
    def prepare_data(raw_data, batch_size)  # 数据预处理
    def incremental_train(train_loader, model, max_epochs, learning_rate)  # 增量训练
    def get_model()  # 获取模型
```

## 3. 核心方法详解

### 3.1 数据流监控
```python
def monitor_data_stream(new_data, drift_threshold):
    """
    功能：检测数据分布漂移
    流程：
        1. 提取历史数据特征
        2. 计算新数据与历史数据的分布差异
        3. 判断漂移分数是否超过阈值
    输出：
        drift_report: {
            'is_drift': bool,   # 是否发生显著漂移
            'drift_score': float, # 分布差异量化值
            'feature_contrib': dict # 各特征的贡献度
        }
    关键技术：
        - 均值差异计算
        - 标准差归一化
        - 特征级贡献分析
    """
```

### 3.2 弹性权重更新
```python
def elastic_update(model, importance_weights):
    """
    功能：防止灾难性遗忘
    数学原理：
        L = L_task + λ * Σ_i (F_i * (θ_i - θ_old_i)^2)
        其中：
          L_task: 新任务损失
          F_i: 参数重要性权重
          θ_i: 当前模型参数
          θ_old_i: 旧模型参数
    流程：
        1. 保存旧模型参数
        2. 计算正则化损失
        3. 混合损失反向传播
    """
```

### 3.3 增量训练
```python
def incremental_train(train_loader, model, max_epochs, learning_rate):
    """
    功能：执行增量训练
    流程：
        1. 根据模型类型选择训练方法
        2. 加载预训练权重
        3. 混合新旧数据训练
        4. 应用弹性权重更新
    支持模型类型：
        - AutoencoderModel
        - TransformerModel
        - DiffusionTrajGenerator
        - VAETrajGenerator
    """
```

## 4. 使用示例
```python
# 初始化增量学习器
incremental_learner = IncrementalLearner(
    base_model=pretrained_model,
    model_type='AutoencoderModel',
    optimizer_config={'lr': 0.001, 'type': 'Adam'},
    memory_buffer_size=1000,
    replay_strategy='random'
)

# 设置模型
incremental_learner.setup_model('pretrained_weights.pt')

# 监控数据流
new_data = ... # 新批次数据
drift_report = incremental_learner.monitor_data_stream(new_data, drift_threshold=0.3)

# 执行增量训练
train_loader = incremental_learner.prepare_data(new_data, batch_size=32)
training_result = incremental_learner.incremental_train(
    train_loader=train_loader,
    model=incremental_learner.model,
    max_epochs=10,
    learning_rate=0.0001
)

# 获取训练后模型
updated_model = incremental_learner.get_model()
```

## 5. 关键参数说明
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| base_model | nn.Module | None | 基础模型架构 |
| model_type | str | '' | 模型类型标识 |
| optimizer_config | dict | {} | 优化器配置 |
| memory_buffer_size | int | 1000 | 经验回放缓冲区大小 |
| replay_strategy | str | 'random' | 回放策略 (random/prioritized) |
| drift_threshold | float | 0.3 | 漂移检测阈值 |
| max_epochs | int | 10 | 增量训练最大轮数 |
| learning_rate | float | 0.001 | 增量学习率 |