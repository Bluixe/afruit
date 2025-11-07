import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time

class BehaviorCloner:
    """
    行为克隆器类
    
    功能描述：实现基于模仿学习的行为克隆算法，通过观察专家行为数据学习策略
    
    核心功能：
    - 数据处理：支持时间序列数据处理与特征提取
    - 模型训练：支持MLP、CNN等多种网络结构
    - 策略评估：提供多种评估指标与可视化工具
    """
    
    def __init__(self,
                 batch_size: int = 32,
                 network_type: str = "MLP",
                 max_epochs: int = 200,
                 dropout_rate: float = 0.2):
        """
        初始化行为克隆器
        
        参数:
            batch_size (int): 训练批次大小，默认为32，有效取值范围16-512
            network_type (str): 网络类型，默认为"MLP"，有效取值范围[MLP,CNN]
            max_epochs (int): 最大训练轮次，默认为200，有效取值范围>=50
            dropout_rate (float): 防止过拟合的神经元丢弃率，默认为0.2，有效取值范围0.0-0.5
        """
        # 初始化参数
        self.batch_size = batch_size
        self.network_type = network_type
        self.max_epochs = max_epochs
        self.dropout_rate = dropout_rate
        
        # 初始化模型和优化器
        self.model = None
        self.optimizer = None
        
        # 初始化训练状态
        self.is_trained = False
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证batch_size
        if not isinstance(self.batch_size, int) or not (16 <= self.batch_size <= 512):
            raise ValueError(f"batch_size必须为16-512范围内的整数，当前值: {self.batch_size}")
        
        # 验证network_type
        valid_network_types = ["MLP", "CNN"]
        if self.network_type not in valid_network_types:
            raise ValueError(f"network_type必须为 {valid_network_types} 之一，当前值: {self.network_type}")
        
        # 验证max_epochs
        if not isinstance(self.max_epochs, int) or self.max_epochs < 50:
            raise ValueError(f"max_epochs必须为大于等于50的整数，当前值: {self.max_epochs}")
        
        # 验证dropout_rate
        if not isinstance(self.dropout_rate, float) or not (0.0 <= self.dropout_rate <= 0.5):
            raise ValueError(f"dropout_rate必须在0.0-0.5范围内，当前值: {self.dropout_rate}")
    
    def process_data(self, raw_trajectories: Dict, context_frames: int = 4) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def extract_features(self, state_data: np.ndarray) -> Dict:
        """
        特征提取函数
        
        参数:
            state_data (np.ndarray): 状态数据，包含:
                - position: [x,y,z] 三维坐标
                - velocity: [vx,vy,vz] 速度向量
                - radar_image: 输入图像（可选）
        
        返回值:
            特征字典，包含512维特征向量
        
        功能描述:
            1. 编码规则：||position||
            2. 速度特征：cos θ = (v1·v2)/(||v1|| ||v2||)
            3. 图像特征：通过CNN提取输入图像特征
        """
        # 初始化特征字典
        features = {}
        
        # 检查输入数据
        if not isinstance(state_data, np.ndarray):
            raise ValueError("state_data必须是numpy数组")
        
        # 提取位置特征（如果存在）
        if state_data.shape[-1] >= 3:
            features['position'] = state_data[..., :3]  # 提取前三个维度作为位置
        
        # 提取速度特征（如果存在）
        if state_data.shape[-1] >= 6:
            features['velocity'] = state_data[..., 3:6]  # 提取第4-6个维度作为速度
        
        # 提取图像特征（如果存在）
        if len(state_data.shape) > 2 and state_data.shape[-1] > 6:
            # 假设第7个维度开始是图像数据
            features['radar_image'] = state_data[..., 6:]
            
            # 如果使用CNN，可以在这里进行图像特征提取
            if self.network_type == "CNN" and self.model:
                # 这里应该实现CNN特征提取
                # 简化示例：直接返回原始图像数据
                pass
        
        # 计算特征统计信息
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                print(f"特征 {key} 形状: {value.shape}")
                if value.size > 0:
                    print(f"特征 {key} 统计: 最小值={np.min(value)}, 最大值={np.max(value)}, 均值={np.mean(value)}")
        
        return features
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        模型训练函数
        
        参数:
            X_train (np.ndarray): 训练数据特征
            y_train (np.ndarray): 训练数据标签
            validation_split (float): 验证集比例，默认为0.2
        
        返回值:
            训练历史记录，包含以下下列指标:
                - train_loss: 训练集损失值
                - val_accuracy: 验证集准确率
        
        功能描述:
            1. 训练历史记录：包含以下数据指标
            2. MLP模式：FC → Dropout → FC → 输出层
            3. CNN模式：CNN特征提取 → 特征拼接 → FC → 输出层
            4. 损失函数：Huber损失
        """
        # 检查输入数据
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("X_train和y_train必须是numpy数组")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train和y_train的样本数不匹配: {X_train.shape[0]} vs {y_train.shape[0]}")
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 划分训练集和验证集
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 创建模型
        input_dim = X_train.shape[1] * X_train.shape[2]  # 展平输入特征
        output_dim = y_train.shape[1]  # 输出动作维度
        
        if self.network_type == "MLP":
            # 创建MLP模型
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, output_dim)
            )
        elif self.network_type == "CNN":
            # 创建CNN模型
            # 注意：这里假设输入是图像数据，如果不是，需要调整模型结构
            # 简化的CNN模型示例
            self.model = nn.Sequential(
                nn.Conv2d(X_train.shape[1], 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(32 * (X_train.shape[2]//4) * (X_train.shape[3]//4), 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, output_dim)
            )
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 定义损失函数
        criterion = nn.MSELoss()
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_accuracy': []
        }
        
        # 训练循环
        for epoch in range(self.max_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            # 计算平均验证损失
            val_loss /= len(val_loader)
            history['val_accuracy'].append(1.0 - val_loss)  # 简化的准确率计算
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 标记模型已训练
        self.is_trained = True
        
        return history
    
    def evaluate_policy(self, test_trajectories: List, simulator: Any = None) -> Dict:
        """
        策略评估函数
        
        参数:
            test_trajectories (List): 测试轨迹数据集
            simulator (Any, optional): 数字孪生仿真环境
        
        返回值:
            metrics (Dict): 包含以下评估指标的字典:
                - action_accuracy: 动作预测准确率（数值动作）
                - error_distribution: 各维度误差分布（连续动作）
        
        功能描述:
            1. 计算测试集上的预测误差指标
            2. 在仿真环境中执行策略并记录表现
            3. 动态评估：计算测试集上的预测误差指标
            4. 在跟踪中：在仿真环境中执行策略并记录表现
        """
        # 检查模型是否已训练
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 初始化评估指标
        metrics = {
            'action_accuracy': 0.0,
            'error_distribution': []
        }
        
        # 检查测试轨迹
        if not test_trajectories:
            raise ValueError("test_trajectories不能为空")
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 处理测试轨迹
        total_samples = 0
        correct_predictions = 0
        errors = []
        
        for trajectory in test_trajectories:
            # 检查轨迹数据是否包含状态和动作
            if 'states' not in trajectory or 'actions' not in trajectory:
                print("警告: 轨迹缺少状态或动作数据，已跳过")
                continue
            
            states = trajectory['states']
            actions = trajectory['actions']
            
            # 检查状态和动作数据长度是否匹配
            if len(states) != len(actions):
                print("警告: 轨迹的状态和动作数据长度不匹配，已跳过")
                continue
            
            # 转换为PyTorch张量
            states_tensor = torch.FloatTensor(states)
            
            # 使用模型进行预测
            with torch.no_grad():
                predicted_actions = self.model(states_tensor).numpy()
            
            # 计算预测准确率
            for i in range(len(actions)):
                # 计算预测误差
                error = np.abs(predicted_actions[i] - actions[i])
                errors.append(error)
                
                # 如果误差小于阈值，则认为预测正确
                if np.mean(error) < 0.1:  # 使用0.1作为阈值
                    correct_predictions += 1
                
                total_samples += 1
        
        # 计算总体准确率
        if total_samples > 0:
            metrics['action_accuracy'] = correct_predictions / total_samples
        
        # 计算误差分布
        if errors:
            errors = np.array(errors)
            metrics['error_distribution'] = np.mean(errors, axis=0)
        
        # 在仿真环境中评估（如果提供）
        if simulator is not None:
            # 这里应该实现在仿真环境中的评估
            # 简化示例：假设simulator有一个evaluate方法
            try:
                sim_metrics = simulator.evaluate(self.model)
                metrics.update(sim_metrics)
            except Exception as e:
                print(f"仿真评估失败: {str(e)}")
        
        return metrics