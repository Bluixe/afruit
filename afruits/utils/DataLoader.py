import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time



class DataLoaderUtil:

    def load_bc_gail_data(self, raw_trajectories: Dict, context_frames: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据处理函数
        
        参数:
            raw_trajectories (Dict): 原始轨迹数据集，包含专家演示轨迹（每条轨迹包含时间序列状态和动作）
            context_frames (int): 上下文帧数，默认为4
        
        返回值:
            X_train: 三维数据（样本数×时间步×特征维度）
            y_train: 二维数据（样本数×动作维度）
        
        功能描述:
            1. 完成数据清洗、标准化和时间序列处理
            2. 自动填充缺失值
            3. 标准化特征到[-1, 1]区间
        """
        # 初始化结果
        expert_states = []
        expert_actions = []
        
        # 检查输入数据
        if not raw_trajectories or not isinstance(raw_trajectories, dict):
            raise ValueError("expert_trajectories必须是非空字典")
        
        # 处理轨迹数据
        for traj_id, trajectory in raw_trajectories.items():
            # 检查轨迹数据是否包含状态和动作
            if 'states' not in trajectory or 'actions' not in trajectory:
                print(f"警告: 轨迹 {traj_id} 缺少状态或动作数据，已跳过")
                continue
            
            states = trajectory['states']
            actions = trajectory['actions']
            
            # 检查状态和动作数据长度是否匹配
            if len(states) != len(actions):
                print(f"警告: 轨迹 {traj_id} 的状态和动作数据长度不匹配，已跳过")
                continue
            
            # 添加数据
            expert_states.extend(states)
            expert_actions.extend(actions)
        
        # 转换为numpy数组
        if expert_states and expert_actions:
            expert_states = np.array(expert_states)
            expert_actions = np.array(expert_actions)
        else:
            raise ValueError("处理后的数据为空，请检查输入数据")
        
        print(f"数据预处理完成: expert_states shape: {expert_states.shape}, expert_actions shape: {expert_actions.shape}")
        return expert_states, expert_actions
    
    def load_expert_data(self, data: List|Dict, seq_length: int, batch_size: int = 32) -> Dict:
        """
        数据加载
        
        参数:
            data_path (list|dict): 预处理后的数据
            batch_size (int): 批处理大小
            
        返回值:
            数据加载器 (DataLoader)
        """
        if type(data) == list:
            # list of dict {"states":..., "actions":...}
            # 将多个轨迹合并为一个整体
            combined_states = []
            combined_actions = []
            for traj in data:
                combined_states.append(traj['states'])
                combined_actions.append(traj['actions'])
            states = np.stack(combined_states, axis=0)
            actions = np.stack(combined_actions, axis=0)
        
        # 提取轨迹数据（支持同时包含状态和动作）
        elif 'state' in data and 'action' in data:
            # 新格式：分别包含状态和动作
            states = data['state']
            actions = data['action']
            
        # 检查状态和动作的序列长度是否匹配
        if states.shape[1] != actions.shape[1]:
            raise ValueError(f"状态序列长度 {states.shape[1]} 与动作序列长度 {actions.shape[1]} 不匹配")
        
        # 检查序列长度
        if states.shape[1] < seq_length:
            raise ValueError(f"轨迹序列长度 {states.shape[1]} 小于设定的序列长度 {seq_length}")
        
        # 如果轨迹长度大于设定长度，对于每条数据，随机截取指定长度的片段
        if states.shape[1] > seq_length:
            # 随机截取指定长度的片段
            start_indices = np.random.randint(0, states.shape[1] - seq_length, size=states.shape[0])
            sampled_states = np.array([
                states[i, start_idx:start_idx+seq_length] 
                for i, start_idx in enumerate(start_indices)
            ])
            sampled_actions = np.array([
                actions[i, start_idx:start_idx+seq_length] 
                for i, start_idx in enumerate(start_indices)
            ])
            states = sampled_states
            actions = sampled_actions
        
        # 获取状态和动作的维度,需要处理图像情况
        if len(states.shape) > 3:
            state_dim = states.shape[2:]  # 图像数据，保留通道和空间维度
        else:
            state_dim = states.shape[2] if len(states.shape) > 2 else 1
        
        # 检查动作是否为离散的
        is_discrete = False
        if np.issubdtype(actions.dtype, np.integer) or (actions.dtype == np.float64 and np.all(np.equal(np.mod(actions, 1), 0))):
            # 如果动作是整数类型，或者是浮点数但都是整数值，则认为是离散的
            is_discrete = True
            action_dim = int(max(actions.flatten()) + 1)  # 离散动作，从0开始编号
        else:
            # 连续动作
            action_dim = actions.shape[2] if len(actions.shape) > 2 else 1
        
        # 转换为PyTorch张量
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(states_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"数据加载完成: 样本数={len(dataset)}, 状态形状={states.shape}, 动作形状={actions.shape}")
        
        return {
            'dataloader': dataloader,
            'data_shape': {
                'states': states.shape,
                'actions': actions.shape
            },
            'feature_dim': {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'total_dim': state_dim + action_dim
            },
            'has_separate_action': True,
            'is_discrete_action': is_discrete
        }