import os
import sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from core.api import AlgorithmAPI

def main():
    """
    小样本博弈建模示例
    
    展示如何使用API进行小样本博弈建模
    """
    print("小样本博弈建模示例")
    
    # 初始化API
    api = AlgorithmAPI(log_level="INFO")
    
    # 创建示例数据
    training_data = create_example_data()
    
    # 配置模型
    model_config = {
        'model_type': 'BehaviorCloner',  # 使用行为克隆模型
        'batch_size': 32,
        'network_type': 'MLP',
        'max_epochs': 50,
        'dropout_rate': 0.2,
        'context_frames': 4,
        'validation_split': 0.2
    }
    
    # 训练模型
    print("\n1. 训练模型")
    result = api.train_game_model(training_data, model_config)
    
    # 提取模型ID和训练指标
    model_id = result['model_id']
    training_metrics = result['training_metrics']
    
    print(f"模型ID: {model_id}")
    print(f"训练损失: {training_metrics['final_train_loss']:.4f}")
    print(f"验证准确率: {training_metrics['final_val_accuracy']:.4f}")
    
    # 可视化训练过程
    print("\n2. 可视化训练过程")
    vis_config = {
        'type': 'line',
        'title': '训练损失',
        'xlabel': 'Epoch',
        'ylabel': 'Loss',
        'grid': True
    }
    
    vis_data = {
        'y': {
            'train_loss': training_metrics['train_loss']
        }
    }
    
    vis_result = api.visualize_results(vis_data, vis_config)
    print(f"可视化结果保存在: {vis_result['save_paths'][0]}")
    
    # 评估模型
    print("\n3. 评估模型")
    test_data = create_test_data()
    
    eval_config = {
        'method': 'offline',
        'method_type': 'IS'
    }
    
    eval_result = api.evaluate_model(result['model'], test_data, eval_config)
    
    print(f"评估结果:")
    print(f"动作准确率: {eval_result['action_accuracy']:.4f}")
    print(f"误差分布: {eval_result['error_distribution']}")
    
    # 保存模型
    print("\n4. 保存模型")
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_id}.pt")
    
    api.save_model(result['model'], save_path)
    print(f"模型保存在: {save_path}")
    
    # 使用模型进行预测
    print("\n5. 使用模型进行预测")
    state = np.random.rand(4, 10)  # 假设状态是4个时间步，每个时间步10个特征
    
    # 使用游戏建模服务进行预测
    action = api.game_modeling_service.predict(model_id, state)
    
    print(f"预测动作: {action}")
    
    print("\n示例完成")

def create_example_data():
    """创建示例训练数据"""
    # 创建10条轨迹，每条轨迹包含100个时间步
    num_trajectories = 10
    trajectory_length = 100
    state_dim = 10
    action_dim = 5
    
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

def create_test_data():
    """创建示例测试数据"""
    # 创建2条测试轨迹
    num_trajectories = 2
    trajectory_length = 50
    state_dim = 10
    action_dim = 5
    
    test_trajectories = []
    
    for i in range(num_trajectories):
        # 创建状态序列
        states = np.random.rand(trajectory_length, state_dim)
        
        # 创建动作序列
        actions = np.zeros((trajectory_length, action_dim))
        for j in range(trajectory_length):
            # 简单的策略：将状态映射到动作
            actions[j] = np.tanh(np.dot(states[j], np.random.rand(state_dim, action_dim)) + np.random.randn(action_dim) * 0.1)
        
        # 添加到测试轨迹列表
        test_trajectories.append({
            'states': states,
            'actions': actions
        })
    
    return test_trajectories

if __name__ == "__main__":
    main()