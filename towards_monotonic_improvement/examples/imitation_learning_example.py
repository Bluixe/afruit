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
    小样本专家轨迹模仿学习示例
    
    展示如何使用API进行小样本专家轨迹模仿学习
    """
    print("小样本专家轨迹模仿学习示例")
    
    # 初始化API
    api = AlgorithmAPI(log_level="INFO")
    
    # 创建示例专家轨迹数据
    expert_trajectories = create_expert_trajectories()
    
    # 1. 使用Transformer模型进行标准训练
    print("\n1. 使用Transformer模型进行标准训练")
    
    # 配置模型
    transformer_config = {
        'model_type': 'TransformerModel',  # 使用Transformer模型
        'training_method': 'standard',     # 标准训练方法
        'encoder_type': 'str',
        'input_dim': 10,                   # 输入维度
        'd_model': 64,                     # 模型隐藏层维度
        'num_heads': 4,                    # 注意力头数量
        'num_layers': 2,                   # Transformer层数
        'max_seq_len': 50,                 # 最大序列长度
        'dropout_rate': 0.1,               # Dropout比率
        'epochs': 30,                      # 训练轮数
        'batch_size': 16,                  # 批次大小
        'learning_rate': 1e-4              # 学习率
    }
    
    # 训练模型
    transformer_result = api.train_imitation_model(expert_trajectories, transformer_config)
    
    # 提取模型ID和训练指标
    transformer_model_id = transformer_result['model_id']
    transformer_metrics = transformer_result['training_metrics']
    
    print(f"Transformer模型ID: {transformer_model_id}")
    print(f"最终训练损失: {transformer_metrics['final_train_loss']:.4f}")
    print(f"最终验证损失: {transformer_metrics['final_val_loss']:.4f}")
    
    # 可视化训练过程
    vis_config = {
        'type': 'line',
        'title': 'Transformer模型训练损失',
        'xlabel': 'Epoch',
        'ylabel': 'Loss',
        'grid': True
    }
    
    vis_data = {
        'y': {
            'train_loss': transformer_metrics['train_loss'],
            'val_loss': transformer_metrics['val_loss']
        }
    }
    
    vis_result = api.visualize_results(vis_data, vis_config)
    print(f"可视化结果保存在: {vis_result['save_paths'][0]}")
    
    # 2. 使用扩散模型进行轨迹生成
    print("\n2. 使用扩散模型进行轨迹生成")
    
    # 配置模型
    diffusion_config = {
        'model_type': 'DiffusionTrajGenerator',  # 使用扩散轨迹生成器
        'training_method': 'standard',           # 标准训练方法
        'state_dim': 10,                         # 状态维度
        'action_dim': 5,                         # 动作维度
        'hidden_dim': 128,                       # 隐藏层维度
        'num_diffusion_steps': 50,               # 扩散步数
        'epochs': 20,                            # 训练轮数
        'batch_size': 16,                        # 批次大小
        'learning_rate': 5e-5                    # 学习率
    }
    
    # 训练模型
    diffusion_result = api.train_imitation_model(expert_trajectories, diffusion_config)
    
    # 提取模型ID和训练指标
    diffusion_model_id = diffusion_result['model_id']
    diffusion_metrics = diffusion_result['training_metrics']
    
    print(f"扩散模型ID: {diffusion_model_id}")
    print(f"最终训练损失: {diffusion_metrics['final_train_loss']:.4f}")
    
    # 3. 使用进化学习方法训练VAE模型
    print("\n3. 使用进化学习方法训练VAE模型")
    
    # 配置模型
    vae_config = {
        'model_type': 'VAETrajGenerator',    # 使用VAE轨迹生成器
        'training_method': 'evolutionary',   # 进化学习方法
        'input_dim': 10,                     # 输入维度
        'hidden_dim': 64,                    # 隐藏层维度
        'latent_dim': 8,                     # 潜在空间维度
        'sequence_length': 50,               # 序列长度
        'population_size': 10,               # 种群规模（示例中使用较小的值）
        'mutation_rate': 0.1,                # 变异概率
        'crossover_rate': 0.7,               # 交叉概率
        'max_generations': 5,                # 最大代数（示例中使用较小的值）
        'fitness_threshold': 0.9             # 适应度阈值
    }
    
    # 训练模型
    vae_result = api.train_imitation_model(expert_trajectories, vae_config)
    
    # 提取模型ID和训练指标
    vae_model_id = vae_result['model_id']
    vae_metrics = vae_result['training_metrics']
    
    print(f"VAE模型ID: {vae_model_id}")
    print(f"最终最大适应度: {vae_metrics['final_max_fitness']:.4f}")
    print(f"最终平均适应度: {vae_metrics['final_mean_fitness']:.4f}")
    print(f"进化代数: {vae_metrics['generations']}")
    
    # 可视化进化过程
    vis_config = {
        'type': 'line',
        'title': 'VAE模型进化过程',
        'xlabel': 'Generation',
        'ylabel': 'Fitness',
        'grid': True
    }
    
    vis_data = {
        'y': {
            'max_fitness': vae_metrics['max_fitness'],
            'mean_fitness': vae_metrics['mean_fitness']
        }
    }
    
    vis_result = api.visualize_results(vis_data, vis_config)
    print(f"可视化结果保存在: {vis_result['save_paths'][0]}")
    
    # 4. 使用Transformer模型生成轨迹
    print("\n4. 使用Transformer模型生成轨迹")
    
    # 创建输入序列
    input_seq = np.random.rand(1, 10, 10)  # 批次大小为1，序列长度为10，特征维度为10
    input_seq_tensor = torch.FloatTensor(input_seq)
    
    # 生成轨迹
    context = {'input_seq': input_seq_tensor}
    config = {'pred_steps': 5}  # 预测5个步骤
    
    trajectory = api.imitation_learning_service.generate_trajectory(
        transformer_model_id, context, config
    )
    
    print(f"生成轨迹形状: {trajectory['trajectory'].shape}")
    
    # 可视化注意力图
    vis_config = {
        'type': 'heatmap',
        'title': 'Transformer注意力图',
        'annot': False
    }
    
    vis_data = {
        'matrix': trajectory['attention_map']
    }
    
    vis_result = api.visualize_results(vis_data, vis_config)
    print(f"注意力图保存在: {vis_result['save_paths'][0]}")
    
    # 5. 使用扩散模型生成轨迹
    print("\n5. 使用扩散模型生成轨迹")
    
    # 创建初始状态
    initial_state = np.random.rand(10)  # 状态维度为10
    
    # 生成轨迹
    context = {'initial_state': initial_state}
    config = {'horizon': 20}  # 生成20个步骤
    
    diffusion_trajectory = api.imitation_learning_service.generate_trajectory(
        diffusion_model_id, context, config
    )
    
    print(f"扩散模型生成轨迹形状: {diffusion_trajectory['trajectory'].shape}")
    
    # 可视化轨迹
    vis_config = {
        'type': 'trajectory',
        'title': '扩散模型生成轨迹',
        'xlabel': 'X',
        'ylabel': 'Y'
    }
    
    # 提取轨迹的前两个维度用于可视化
    traj_data = diffusion_trajectory['trajectory']
    traj_2d = [(traj_data[i, 0], traj_data[i, 1]) for i in range(traj_data.shape[0])]
    
    vis_data = {
        'trajectories': {
            'generated': traj_2d
        }
    }
    
    vis_result = api.visualize_results(vis_data, vis_config)
    print(f"轨迹可视化保存在: {vis_result['save_paths'][0]}")
    
    # 保存模型
    print("\n6. 保存模型")
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存Transformer模型
    transformer_save_path = os.path.join(save_dir, f"{transformer_model_id}.pt")
    api.save_model(transformer_result['model'], transformer_save_path)
    print(f"Transformer模型保存在: {transformer_save_path}")
    
    # 保存扩散模型
    diffusion_save_path = os.path.join(save_dir, f"{diffusion_model_id}.pt")
    api.save_model(diffusion_result['model'], diffusion_save_path)
    print(f"扩散模型保存在: {diffusion_save_path}")
    
    print("\n示例完成")

def create_expert_trajectories():
    """创建示例专家轨迹数据"""
    # 创建20条专家轨迹，每条轨迹包含50个时间步
    num_trajectories = 20
    trajectory_length = 50
    state_dim = 10
    action_dim = 5
    
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
    for i in range(100):  # 100个批次
        inputs = np.random.rand(16, 10, state_dim)  # 批次大小16，序列长度10
        targets = np.random.rand(16, 10, action_dim)  # 对应的目标动作
        batch_data.append({
            'inputs': inputs,
            'targets': targets
        })
    
    data['batch_data'] = batch_data
    
    return data

if __name__ == "__main__":
    main()