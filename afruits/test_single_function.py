import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入API
from core.api import AlgorithmAPI

def test_behavior_cloner():
    """测试行为克隆模型"""
    print("测试行为克隆模型")
    
    # 初始化API
    api = AlgorithmAPI(log_level="INFO")
    
    # 创建简单的训练数据
    training_data = create_simple_training_data()
    
    # 配置模型
    model_config = {
        'model_type': 'BehaviorCloner',  # 使用行为克隆模型
        'batch_size': 16,
        'network_type': 'MLP',
        'max_epochs': 5,  # 减少训练轮数以加快测试
        'dropout_rate': 0.1,
        'context_frames': 2,
        'validation_split': 0.2
    }
    
    # 训练模型
    print("开始训练行为克隆模型...")
    result = api.train_game_model(training_data, model_config)
    
    # 提取模型ID和训练指标
    model_id = result['model_id']
    model = result['model']
    training_metrics = result['training_metrics']
    
    print(f"模型ID: {model_id}")
    print(f"训练损失: {training_metrics['final_train_loss']:.4f}")
    print(f"验证准确率: {training_metrics['final_val_accuracy']:.4f}")
    
    # 测试预测功能
    state = np.random.rand(2, 5)  # 假设状态是2个时间步，每个时间步5个特征
    action = api.game_modeling_service.predict(model_id, state)
    
    print(f"预测动作形状: {action.shape}")
    print(f"预测动作值: {action}")
    
    print("行为克隆模型测试完成")
    
    return model_id, model

def test_transformer_model():
    """测试Transformer模型"""
    print("\n测试Transformer模型")
    
    # 初始化API
    api = AlgorithmAPI(log_level="INFO")
    
    # 创建简单的专家轨迹数据
    expert_trajectories = create_simple_expert_trajectories()
    
    # 配置Transformer模型
    transformer_config = {
        'model_type': 'TransformerModel',  # 使用Transformer模型
        'training_method': 'standard',     # 标准训练方法
        'encoder_type': 'str',
        'input_dim': 5,                    # 输入维度
        'd_model': 32,                     # 模型隐藏层维度
        'num_heads': 2,                    # 注意力头数量
        'num_layers': 1,                   # Transformer层数
        'max_seq_len': 20,                 # 最大序列长度
        'dropout_rate': 0.1,               # Dropout比率
        'epochs': 3,                       # 训练轮数（减少以加快测试）
        'batch_size': 8,                   # 批次大小
        'learning_rate': 1e-3              # 学习率
    }
    
    # 训练模型
    print("开始训练Transformer模型...")
    transformer_result = api.train_imitation_model(expert_trajectories, transformer_config)
    
    # 提取模型ID和训练指标
    transformer_model_id = transformer_result['model_id']
    transformer_metrics = transformer_result['training_metrics']
    
    print(f"Transformer模型ID: {transformer_model_id}")
    print(f"最终训练损失: {transformer_metrics['final_train_loss']:.4f}")
    print(f"最终验证损失: {transformer_metrics['final_val_loss']:.4f}")
    
    # 测试轨迹生成功能
    input_seq = np.random.rand(1, 5, 5)  # 批次大小为1，序列长度为5，特征维度为5
    input_seq_tensor = torch.FloatTensor(input_seq)
    
    # 生成轨迹
    print("生成轨迹...")
    context = {'input_seq': input_seq_tensor}
    config = {'pred_steps': 3}  # 预测3个步骤
    
    trajectory = api.imitation_learning_service.generate_trajectory(
        transformer_model_id, context, config
    )
    
    print(f"生成轨迹形状: {trajectory['trajectory'].shape}")
    print(f"生成轨迹值: {trajectory['trajectory']}")
    
    print("Transformer模型测试完成")
    
    return transformer_model_id, transformer_result['model']

def create_simple_training_data():
    """创建简单的训练数据"""
    # 创建2条轨迹，每条轨迹包含20个时间步
    num_trajectories = 2
    trajectory_length = 20
    state_dim = 5
    action_dim = 3
    
    trajectories = {}
    
    for i in range(num_trajectories):
        # 创建状态序列
        states = np.random.rand(trajectory_length, state_dim)
        
        # 创建动作序列（简单的线性映射加噪声）
        actions = np.zeros((trajectory_length, action_dim))
        for j in range(trajectory_length):
            # 简单的策略：将状态映射到动作
            actions[j] = np.tanh(np.dot(states[j], np.random.rand(state_dim, action_dim)) + np.random.randn(action_dim) * 0.1)
        
        # 创建奖励序列
        rewards = np.sum(actions, axis=1)  # 简单的奖励：动作的和
        
        # 添加到轨迹字典
        trajectories[f"trajectory_{i}"] = {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
    
    return trajectories

def create_simple_expert_trajectories():
    """创建简单的专家轨迹数据"""
    # 创建3条专家轨迹，每条轨迹包含20个时间步
    num_trajectories = 3
    trajectory_length = 20
    state_dim = 5
    action_dim = 3
    
    # 创建数据字典
    data = {
        'trajectories': []
    }
    
    for i in range(num_trajectories):
        # 创建状态序列
        states = np.random.rand(trajectory_length, state_dim)
        
        # 创建动作序列（专家策略）
        actions = np.zeros((trajectory_length, action_dim))
        for j in range(trajectory_length):
            # 模拟专家策略：非线性映射
            actions[j] = np.tanh(np.dot(states[j], np.random.rand(state_dim, action_dim)) + np.random.randn(action_dim) * 0.05)
        
        # 创建奖励序列
        rewards = np.sum(actions, axis=1) + np.random.randn(trajectory_length) * 0.1
        
        # 添加到轨迹列表
        data['trajectories'].append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'expert': True  # 标记为专家轨迹
        })
    
    # 添加批次数据（用于Transformer训练）
    batch_data = []
    for i in range(5):  # 5个批次
        inputs = np.random.rand(8, 5, state_dim)  # 批次大小8，序列长度5
        targets = np.random.rand(8, 5, action_dim)  # 对应的目标动作
        batch_data.append({
            'inputs': inputs,
            'targets': targets
        })
    
    data['batch_data'] = batch_data
    
    return data

if __name__ == "__main__":
    # 测试行为克隆模型
    bc_model_id, bc_model = test_behavior_cloner()
    
    # 测试Transformer模型
    transformer_model_id, transformer_model = test_transformer_model()
    
    print("\n所有测试完成")