# 算法小样本快速升级迭代软件

## 项目概述

本项目是"算法小样本快速升级迭代软件"的核心代码，旨在提供一个统一的框架，用于小样本博弈建模和专家轨迹模仿学习。系统支持在仅有少量离线博弈轨迹数据的情况下，快速构建并优化博弈模型/策略，同时利用人类专家或其他高水平策略的轨迹数据进行模仿和强化，提升策略水平。

## 系统目标

- **小样本数据的快速建模与策略迭代**：在仅有少量离线博弈轨迹数据的情况下，快速构建并优化博弈模型/策略。
- **专家轨迹模仿学习**：利用人类专家或其他高水平策略的轨迹数据进行模仿和强化，提升策略水平。
- **多种策略评估与可视化**：提供多指标、多维度的评估方式，并支持可视化分析策略效果。

## 系统架构

系统由以下几个主要模块组成：

### 1. 核心API层

- `AlgorithmAPI`：提供统一的接口，用于小样本博弈建模和专家轨迹模仿学习。

### 2. 服务层

- `GameModelingService`：小样本博弈建模服务，负责处理小样本博弈建模相关的功能。
- `ImitationLearningService`：小样本专家轨迹模仿学习服务，负责处理小样本专家轨迹模仿学习相关的功能。
- `VisualizationService`：可视化服务，负责处理系统的可视化输出功能。
- `LoggingService`：日志服务，负责处理系统的日志记录功能。

### 3. 功能模块

- **数据预处理**：
  - `DataPreprocessor`：数据读取、异常值处理、时间序列对齐、传感器数据融合、标准化。
  - `TrajectoryPreprocessor`：轨迹层面的预处理：标准化、轨迹分段、时序对齐、噪声处理、数据增强。

- **评估**：
  - `OfflineEvaluator`：离线评估器，用于评估离线强化学习和行为克隆算法。
  - `MultiMetricEvaluator`：多指标评估器，基于多维度指标系统对强化学习与多维度综合评估。

- **基础算法模块（Learner）**：
  - `BehaviorCloner`：行为克隆器，通过观察专家行为数据学习策略。
  - `AdversarialImitationLearner`：对抗模仿学习器，通过对抗训练学习专家策略。
  - `OfflineRLearner`：离线强化学习器，通过离线数据学习策略。
  - `OfflineFSPLearner`：离线虚构自我博弈学习器，通过虚构自我博弈学习策略。

- **轨迹建模与生成模型**：
  - `AutoencoderModel`：自编码器模型，用于轨迹压缩和重构。
  - `TransformerModel`：Transformer模型，基于注意力机制的序列处理模型。
  - `DiffusionTrajGenerator`：扩散轨迹生成器，基于扩散模型生成轨迹。
  - `VAETrajGenerator`：VAE轨迹生成器，基于变分自编码器生成轨迹。

- **训练方法模块**：
  - `EvolutionaryLearner`：进化学习器，基于种群进化机制实现策略群体持续优化。
  - `IncrementalLearner`：增量学习器，支持模型的增量更新和持续学习。
  - `FineTuneManager`：微调管理器，支持预训练模型的微调。

## 使用方法

### 1. 初始化API

```python
from core.api import AlgorithmAPI

# 初始化API
api = AlgorithmAPI(log_level="INFO")
```

### 2. 小样本博弈建模

```python
# 加载数据
training_data = api.load_data("path/to/data.json", data_format="json")

# 预处理数据
processed_data = api.preprocess_data(training_data, preprocess_config={
    'outlier_threshold': 3.0,
    'outlier_strategy': 'remove',
    'normalize': True
})

# 配置模型
model_config = {
    'model_type': 'BehaviorCloner',
    'batch_size': 32,
    'network_type': 'MLP',
    'max_epochs': 200,
    'dropout_rate': 0.2
}

# 训练模型
result = api.train_game_model(processed_data, model_config)

# 提取模型ID和训练指标
model_id = result['model_id']
model = result['model']
training_metrics = result['training_metrics']

# 评估模型
eval_config = {
    'method': 'offline',
    'method_type': 'IS'
}
eval_result = api.evaluate_model(model, test_data, eval_config)

# 可视化结果
vis_config = {
    'type': 'line',
    'title': '训练损失',
    'xlabel': 'Epoch',
    'ylabel': 'Loss'
}
vis_data = {
    'y': {
        'train_loss': training_metrics['train_loss']
    }
}
vis_result = api.visualize_results(vis_data, vis_config)

# 保存模型
save_path = api.save_model(model, "models/model.pt")
```

### 3. 小样本专家轨迹模仿学习

```python
# 加载专家轨迹数据
expert_trajectories = api.load_data("path/to/expert_data.json", data_format="json")

# 预处理轨迹数据
processed_trajectories = api.preprocess_trajectory(expert_trajectories, preprocess_config={
    'normalize': True,
    'segment_length': 50
})

# 配置模型
model_config = {
    'model_type': 'TransformerModel',
    'training_method': 'standard',
    'encoder_type': 'str',
    'input_dim': 32,
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 3,
    'max_seq_len': 100,
    'dropout_rate': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-4
}

# 训练模型
result = api.train_imitation_model(processed_trajectories, model_config)

# 提取模型ID和训练指标
model_id = result['model_id']
model = result['model']
training_metrics = result['training_metrics']

# 生成轨迹
context = {'input_seq': input_seq}
config = {'pred_steps': 10}
trajectory = api.imitation_learning_service.generate_trajectory(model_id, context, config)

# 可视化轨迹
vis_config = {
    'type': 'trajectory',
    'title': '生成轨迹',
    'xlabel': 'X',
    'ylabel': 'Y'
}
vis_data = {
    'trajectories': {
        'generated': trajectory['trajectory']
    }
}
vis_result = api.visualize_results(vis_data, vis_config)

# 保存模型
save_path = api.save_model(model, "models/transformer_model.pt")
```

## 高级功能

### 1. 进化学习

```python
# 配置进化学习
model_config = {
    'model_type': 'VAETrajGenerator',
    'training_method': 'evolutionary',
    'input_dim': 32,
    'hidden_dim': 128,
    'latent_dim': 16,
    'sequence_length': 50,
    'population_size': 50,
    'mutation_rate': 0.15,
    'crossover_rate': 0.7,
    'max_generations': 50,
    'fitness_threshold': 0.95
}

# 训练模型
result = api.train_imitation_model(expert_trajectories, model_config)
```

### 2. 增量学习

```python
# 配置增量学习
model_config = {
    'model_type': 'TransformerModel',
    'training_method': 'incremental',
    'base_model_id': 'TransformerModel_1234567890',
    'memory_size': 1000,
    'replay_ratio': 0.3,
    'learning_rate': 1e-4,
    'epochs': 50
}

# 训练模型
result = api.train_imitation_model(new_expert_trajectories, model_config)
```

### 3. 微调

```python
# 配置微调
model_config = {
    'model_type': 'TransformerModel',
    'training_method': 'fine_tune',
    'base_model_id': 'TransformerModel_1234567890',
    'learning_rate': 5e-5,
    'epochs': 20,
    'freeze_layers': ['encoder.0', 'encoder.1']
}

# 训练模型
result = api.train_imitation_model(expert_trajectories, model_config)
```

## 示例脚本

系统提供了两个示例脚本，展示如何使用API：

1. `examples/game_modeling_example.py`：展示如何使用小样本博弈建模API。
2. `examples/imitation_learning_example.py`：展示如何使用小样本专家轨迹模仿学习API。

可以通过以下命令运行示例脚本：

```bash
python examples/game_modeling_example.py
python examples/imitation_learning_example.py
```

## 系统要求

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/your-username/algorithm-framework.git
cd algorithm-framework
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装PyTorch（根据您的系统和CUDA版本选择合适的命令）：

```bash
pip install torch torchvision torchaudio
```

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。