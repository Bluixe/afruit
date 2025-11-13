# 博弈建模服务文档

## 1. 核心文件
`afruits/core/services/game_modeling_service.py` (需确认文件存在性)

## 2. 类结构
```python
class GameModelingService:
    def __init__(config, logger)  # 初始化服务
    def train_game_model(training_data, model_config)  # 博弈模型训练
    def evaluate_strategy(model_id, opponent_data)  # 策略评估
    def simulate_game(model_id, initial_state)  # 博弈模拟
```

## 3. 核心方法详解

### 3.1 小样本策略学习
```python
def train_game_model(training_data, model_config):
    """
    功能：基于小样本数据训练博弈策略模型
    输入参数：
        training_data: {
            'states': 状态序列,
            'actions': 动作序列,
            'rewards': 奖励序列
        }
        model_config: {
            'model_type': 'OfflineRLearner'/'OfflineFSPLearner',
            'learning_rate': 0.001,
            'batch_size': 32
        }
    输出：训练好的模型对象
    关键技术：
        - 离线强化学习
        - 自对弈策略优化
        - 小样本迁移学习
    """
```

### 3.2 策略评估
```python
def evaluate_strategy(model_id, opponent_data):
    """
    功能：评估当前策略对抗特定对手的表现
    输入参数：
        model_id: 已训练模型ID
        opponent_data: {
            'strategy_type': 'random'/'fixed'/'adaptive',
            'behavior_pattern': 对手行为模式
        }
    输出：评估指标字典
    关键技术：
        - 胜率计算
        - 收益期望分析
        - 策略鲁棒性测试
    """
```

### 3.3 博弈模拟
```python
def simulate_game(model_id, initial_state):
    """
    功能：模拟完整博弈过程
    输入参数：
        model_id: 已训练模型ID
        initial_state: 初始状态
    输出：博弈过程记录
    关键技术：
        - 状态转移模拟
        - 多智能体决策
        - 实时策略调整
    """
```

## 4. 支持的模型类型
| 模型类型 | 适用场景 | 关键技术 |
|----------|----------|----------|
| OfflineRLearner | 离线强化学习 | Q-learning, 策略梯度 |
| OfflineFSPLearner | 自对弈学习 | 虚拟自对弈, 策略蒸馏 |
| BehaviorCloner | 行为克隆 | 监督学习, 行为模仿 |
| AdversarialImitationLearner | 对抗模仿 | GAN, 逆强化学习 |

## 5. 使用示例
```python
# 初始化服务
game_service = GameModelingService()

# 训练博弈模型
training_data = {
    'states': np.load('states.npy'),
    'actions': np.load('actions.npy'),
    'rewards': np.load('rewards.npy')
}
model_config = {
    'model_type': 'OfflineFSPLearner',
    'learning_rate': 0.0005,
    'batch_size': 64
}
result = game_service.train_game_model(training_data, model_config)

# 策略评估
opponent_data = {
    'strategy_type': 'adaptive',
    'behavior_pattern': 'aggressive'
}
evaluation = game_service.evaluate_strategy('model_v1', opponent_data)

# 博弈模拟
initial_state = np.array([0, 0, 0])
simulation = game_service.simulate_game('model_v1', initial_state)