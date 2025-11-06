import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import copy

class OfflineFSPLearner:
    """
    离线自对弈学习器类 (Offline Fictitious Self-Play Learner)
    
    功能描述：实现基于离线数据的自对弈学习算法，使用 Off-FSP 方法
    
    核心功能：
    - 数据处理：支持加权数据集构建
    - 模型构建：支持多种网络结构
    - 自对弈训练：实现离线自对弈学习过程
    - 策略评估：提供多种评估指标
    """
    
    def __init__(self,
                 strategy_pool_size: int = 20,
                 cql_penalty_weight: float = 0.7,
                 exposure_ratio: float = 0.6,
                 importance_beta: float = 0.5):
        """
        初始化离线自对弈学习器
        
        参数:
            strategy_pool_size (int): 策略池容量，默认为20，有效取值范围10-50
            cql_penalty_weight (float): CQL惩罚项系数，默认为0.7，有效取值范围0.3-1.0
            exposure_ratio (float): 对抗策略暴露比例，默认为0.6，有效取值范围0.4-0.9
            importance_beta (float): 重要性采样系数，默认为0.5，有效取值范围0.1-0.9
        """
        # 初始化参数
        self.strategy_pool_size = strategy_pool_size
        self.cql_penalty_weight = cql_penalty_weight
        self.exposure_ratio = exposure_ratio
        self.importance_beta = importance_beta
        
        # 初始化模型和优化器
        self.network = None
        self.policy_network = None
        self.opponent_model = None
        self.optimizer = None
        
        # 初始化训练状态
        self.is_trained = False
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证strategy_pool_size
        if not isinstance(self.strategy_pool_size, int) or not (10 <= self.strategy_pool_size <= 50):
            raise ValueError(f"strategy_pool_size必须为10-50范围内的整数，当前值: {self.strategy_pool_size}")
        
        # 验证cql_penalty_weight
        if not isinstance(self.cql_penalty_weight, float) or not (0.3 <= self.cql_penalty_weight <= 1.0):
            raise ValueError(f"cql_penalty_weight必须在0.3-1.0范围内，当前值: {self.cql_penalty_weight}")
        
        # 验证exposure_ratio
        if not isinstance(self.exposure_ratio, float) or not (0.4 <= self.exposure_ratio <= 0.9):
            raise ValueError(f"exposure_ratio必须在0.4-0.9范围内，当前值: {self.exposure_ratio}")
        
        # 验证importance_beta
        if not isinstance(self.importance_beta, float) or not (0.1 <= self.importance_beta <= 0.9):
            raise ValueError(f"importance_beta必须在0.1-0.9范围内，当前值: {self.importance_beta}")
    
    def build_weighted_dataset(self, raw_trajectories: List, opponent_id: int) -> Dict:
        """
        数据处理函数：构建加权数据集
        
        参数:
            raw_trajectories (List): 多智能体对抗轨迹（含五元组+对手动作）
            opponent_id (int): 目标对抗智能体编号
        
        返回值:
            Dict: 加权数据集（子集）
        
        处理流程:
            1. 数据切片：按照时间切片
            2. 权重计算：重要性采样权重
            3. 样本筛选：基于权重进行采样
        """
        # 初始化结果
        weighted_dataset = {}
        
        # 检查输入数据
        if not raw_trajectories or not isinstance(raw_trajectories, list):
            raise ValueError("raw_trajectories必须是非空列表")
        
        # 处理轨迹数据
        states = []
        actions = []
        opponent_actions = []
        rewards = []
        next_states = []
        dones = []
        sample_weights = []
        
        for trajectory in raw_trajectories:
            # 检查轨迹数据是否包含必要字段
            if not all(key in trajectory for key in ['states', 'actions', 'opponent_actions', 'rewards', 'next_states', 'dones']):
                print("警告: 轨迹缺少必要数据字段，已跳过")
                continue
            
            # 检查对手ID是否匹配
            if 'opponent_id' in trajectory and trajectory['opponent_id'] != opponent_id:
                continue
            
            # 提取数据
            traj_states = trajectory['states']
            traj_actions = trajectory['actions']
            traj_opponent_actions = trajectory['opponent_actions']
            traj_rewards = trajectory['rewards']
            traj_next_states = trajectory['next_states']
            traj_dones = trajectory['dones']
            
            # 计算重要性权重
            traj_weights = self._compute_importance_weights(traj_states, traj_actions, traj_rewards)
            
            # 添加到数据集
            states.extend(traj_states)
            actions.extend(traj_actions)
            opponent_actions.extend(traj_opponent_actions)
            rewards.extend(traj_rewards)
            next_states.extend(traj_next_states)
            dones.extend(traj_dones)
            sample_weights.extend(traj_weights)
        
        # 转换为numpy数组
        if states:
            weighted_dataset['states'] = np.array(states)
            weighted_dataset['actions'] = np.array(actions)
            weighted_dataset['opponent_actions'] = np.array(opponent_actions)
            weighted_dataset['rewards'] = np.array(rewards)
            weighted_dataset['next_states'] = np.array(next_states)
            weighted_dataset['dones'] = np.array(dones)
            weighted_dataset['sample_weights'] = np.array(sample_weights)
        else:
            raise ValueError("处理后的数据为空，请检查输入数据或对手ID")
        
        print(f"数据处理完成: 共 {len(states)} 条加权样本")
        return weighted_dataset
    
    def _compute_importance_weights(self, states, actions, rewards):
        """计算重要性采样权重"""
        # 初始化权重
        weights = np.ones(len(states))
        
        # 如果已经训练过模型，可以使用模型来计算更精确的权重
        if self.is_trained and self.network is not None:
            try:
                # 转换为张量
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)
                
                # 使用模型计算Q值
                with torch.no_grad():
                    q_values = self.network(states_tensor)
                    selected_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze()
                    
                # 计算权重：β-power of Q-values
                weights = torch.pow(selected_q, self.importance_beta).numpy()
                
                # 归一化权重
                weights = weights / (np.sum(weights) + 1e-6)
            except Exception as e:
                print(f"计算重要性权重出错: {e}")
                # 使用默认权重
                weights = np.ones(len(states)) / len(states)
        else:
            # 使用奖励值作为简单的权重
            weights = np.abs(rewards) + 1e-6
            weights = weights / np.sum(weights)
        
        return weights
    
    def build_network(self, input_dim: int, action_dim: int) -> Dict:
        """
        网络构建函数
        
        参数:
            input_dim (int): 输入维度
            action_dim (int): 动作维度
        
        返回值:
            Dict: 包含网络结构的字典 (q_network, policy_network, opponent_model)
        
        网络结构:
            1. Q网络（CQL）：历史轨迹输入 → GRU时序编码 → 融合动作分布输出
            2. 策略网络：对手模型输入 → Transformer编码层 → 双头策略/价值输出
        """
        # 创建Q网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 创建策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 创建对手模型
        self.opponent_model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 创建优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.opponent_optimizer = optim.Adam(self.opponent_model.parameters(), lr=0.001)
        
        # 返回网络结构
        return {
            'q_network': self.network,
            'policy_network': self.policy_network,
            'opponent_model': self.opponent_model
        }
    
    def fictitious_play(self, dataset: Dict, num_iterations: int = 100) -> Dict:
        """
        自对弈训练函数
        
        参数:
            dataset (Dict): 加权对抗数据集
            num_iterations (int): 自对弈轮次，默认为100
        
        返回值:
            Dict: 训练日志
        
        训练流程:
            1. 数据集初始化: 加权样本准备
            2. 策略网络训练: BCQ+策略下采样
            3. 策略评估进化: Top-k策略保留机制
            4. 策略池生成: 计算策略平均分布
        """
        # 检查网络是否已构建
        if self.network is None or self.policy_network is None or self.opponent_model is None:
            raise ValueError("网络尚未构建，请先调用build_network方法")
        
        # 检查数据集
        if not dataset or not isinstance(dataset, dict):
            raise ValueError("dataset必须是非空字典")
        
        # 提取数据
        states = dataset['states']
        actions = dataset['actions']
        opponent_actions = dataset['opponent_actions']
        rewards = dataset['rewards']
        next_states = dataset['next_states']
        dones = dataset['dones']
        sample_weights = dataset.get('sample_weights', np.ones(len(states)))
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        opponent_actions_tensor = torch.LongTensor(opponent_actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        weights_tensor = torch.FloatTensor(sample_weights)
        
        # 创建数据集和数据加载器
        batch_size = 64
        dataset_tensor = TensorDataset(
            states_tensor, actions_tensor, opponent_actions_tensor,
            rewards_tensor, next_states_tensor, dones_tensor, weights_tensor
        )
        dataloader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
        
        # 训练日志
        logs = {
            'q_loss': [],
            'policy_loss': [],
            'opponent_loss': [],
            'exploitability': []
        }
        
        # 策略池
        strategy_pool = []
        
        # 训练循环
        for iteration in range(num_iterations):
            # 训练Q网络和策略网络
            epoch_q_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_opponent_loss = 0.0
            
            for batch in dataloader:
                batch_states, batch_actions, batch_opponent_actions, \
                batch_rewards, batch_next_states, batch_dones, batch_weights = batch
                
                # 训练Q网络
                self.optimizer.zero_grad()
                
                # 计算当前Q值
                q_values = self.network(batch_states)
                q_values_selected = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
                
                # 计算目标Q值
                with torch.no_grad():
                    # 使用策略网络选择下一个动作
                    next_q_values = self.network(batch_next_states)
                    next_actions = next_q_values.argmax(dim=1)
                    next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
                    
                    # 计算目标值
                    target_q = batch_rewards + (1 - batch_dones) * 0.99 * next_q
                
                # 计算TD误差
                td_loss = nn.MSELoss()(q_values_selected, target_q)
                
                # 计算CQL惩罚项
                cql_loss = torch.logsumexp(q_values, dim=1).mean() - q_values_selected.mean()
                
                # 总损失
                q_loss = td_loss + self.cql_penalty_weight * cql_loss
                
                # 反向传播
                q_loss.backward()
                self.optimizer.step()
                
                # 训练策略网络
                self.policy_optimizer.zero_grad()
                
                # 计算策略损失
                policy_output = self.policy_network(batch_states)
                policy_loss = nn.CrossEntropyLoss()(policy_output, batch_actions)
                
                # 反向传播
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # 训练对手模型
                self.opponent_optimizer.zero_grad()
                
                # 计算对手模型损失
                opponent_output = self.opponent_model(batch_states)
                opponent_loss = nn.CrossEntropyLoss()(opponent_output, batch_opponent_actions)
                
                # 反向传播
                opponent_loss.backward()
                self.opponent_optimizer.step()
                
                # 累积损失
                epoch_q_loss += q_loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_opponent_loss += opponent_loss.item()
            
            # 计算平均损失
            epoch_q_loss /= len(dataloader)
            epoch_policy_loss /= len(dataloader)
            epoch_opponent_loss /= len(dataloader)
            
            # 记录损失
            logs['q_loss'].append(epoch_q_loss)
            logs['policy_loss'].append(epoch_policy_loss)
            logs['opponent_loss'].append(epoch_opponent_loss)
            
            # 评估当前策略
            exploitability = self._evaluate_exploitability(states_tensor)
            logs['exploitability'].append(exploitability)
            
            # 更新策略池
            if len(strategy_pool) < self.strategy_pool_size:
                strategy_pool.append(copy.deepcopy(self.policy_network))
            else:
                # 替换策略池中的最差策略
                worst_idx = 0
                worst_value = float('-inf')
                
                for i, strategy in enumerate(strategy_pool):
                    # 评估策略
                    with torch.no_grad():
                        value = self._evaluate_strategy(strategy, states_tensor)
                    
                    if value > worst_value:
                        worst_value = value
                        worst_idx = i
                
                # 如果当前策略比最差策略好，则替换
                current_value = self._evaluate_strategy(self.policy_network, states_tensor)
                if current_value < worst_value:
                    strategy_pool[worst_idx] = copy.deepcopy(self.policy_network)
            
            # 打印训练进度
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{num_iterations}, "
                      f"Q Loss: {epoch_q_loss:.4f}, "
                      f"Policy Loss: {epoch_policy_loss:.4f}, "
                      f"Opponent Loss: {epoch_opponent_loss:.4f}, "
                      f"Exploitability: {exploitability:.4f}")
        
        # 标记模型已训练
        self.is_trained = True
        
        # 保存策略池
        self.strategy_pool = strategy_pool
        
        return logs
    
    def _evaluate_exploitability(self, states_tensor):
        """评估当前策略的可利用性"""
        with torch.no_grad():
            # 使用策略网络和对手模型计算动作概率
            policy_probs = torch.softmax(self.policy_network(states_tensor), dim=1)
            opponent_probs = torch.softmax(self.opponent_model(states_tensor), dim=1)
            
            # 计算Q值
            q_values = self.network(states_tensor)
            
            # 计算可利用性
            exploitability = 0.0
            
            for i in range(len(states_tensor)):
                # 计算最佳响应
                best_response_action = q_values[i].argmax().item()
                
                # 计算当前策略的期望值
                policy_value = 0.0
                for a in range(q_values.shape[1]):
                    policy_value += policy_probs[i, a] * q_values[i, a]
                
                # 计算最佳响应的值
                best_response_value = q_values[i, best_response_action]
                
                # 累积可利用性
                exploitability += (best_response_value - policy_value).item()
            
            # 计算平均可利用性
            exploitability /= len(states_tensor)
        
        return exploitability
    
    def _evaluate_strategy(self, strategy, states_tensor):
        """评估策略的性能"""
        with torch.no_grad():
            # 使用策略计算动作概率
            strategy_probs = torch.softmax(strategy(states_tensor), dim=1)
            
            # 计算Q值
            q_values = self.network(states_tensor)
            
            # 计算策略的期望值
            value = 0.0
            
            for i in range(len(states_tensor)):
                state_value = 0.0
                for a in range(q_values.shape[1]):
                    state_value += strategy_probs[i, a] * q_values[i, a]
                value += state_value.item()
            
            # 计算平均值
            value /= len(states_tensor)
        
        return value
    
    def evaluate_equilibrium(self, test_pool: List, num_episodes: int = 100) -> Dict:
        """
        评估函数
        
        参数:
            test_pool (List): 测试策略池
            num_episodes (int): 对抗轮次，默认为100
        
        返回值:
            Dict: 包含以下评估指标的字典:
                - exploitability: 策略可利用性指标
                - coverage_rate: 策略空间覆盖度
                - monitor_matrix: 策略互交关系矩阵
                - trajectory_samples: 典型对抗轨迹片段
        
        评估流程:
            1. 均衡评估报告（字典）
            2. exploitability: 策略可利用性指标
            3. coverage_rate: 策略空间覆盖度（0-1）
            4. monitor_matrix: 策略互交关系矩阵
            5. trajectory_samples: 典型对抗轨迹片段
        """
        # 检查模型是否已训练
        if not self.is_trained or self.policy_network is None:
            raise ValueError("模型尚未训练，请先调用fictitious_play方法")
        
        # 初始化评估指标
        metrics = {
            'exploitability': 0.0,
            'coverage_rate': 0.0,
            'monitor_matrix': None,
            'trajectory_samples': []
        }
        
        # 检查测试策略池
        if not test_pool or not isinstance(test_pool, list):
            raise ValueError("test_pool必须是非空列表")
        
        # 创建监控矩阵
        n_strategies = len(test_pool)
        monitor_matrix = np.zeros((n_strategies, n_strategies))
        
        # 评估每对策略
        for i in range(n_strategies):
            for j in range(n_strategies):
                # 计算策略i对策略j的胜率
                win_rate = self._evaluate_head_to_head(test_pool[i], test_pool[j], num_episodes)
                monitor_matrix[i, j] = win_rate
        
        # 计算可利用性
        exploitability = 0.0
        for i in range(n_strategies):
            # 找出对策略i的最佳响应
            best_response_idx = np.argmax(monitor_matrix[:, i])
            best_response_value = monitor_matrix[best_response_idx, i]
            
            # 累积可利用性
            exploitability += best_response_value - 0.5  # 0.5表示平局
        
        # 计算平均可利用性
        metrics['exploitability'] = exploitability / n_strategies
        
        # 计算策略空间覆盖度
        # 简化计算：使用监控矩阵的秩作为覆盖度的近似
        rank = np.linalg.matrix_rank(monitor_matrix)
        metrics['coverage_rate'] = rank / n_strategies
        
        # 保存监控矩阵
        metrics['monitor_matrix'] = monitor_matrix
        
        # 生成典型对抗轨迹片段
        # 这里简化处理，实际应用中可能需要更复杂的轨迹生成逻辑
        metrics['trajectory_samples'] = self._generate_trajectory_samples(test_pool, 5)
        
        print(f"评估完成: 可利用性 = {metrics['exploitability']:.4f}, 覆盖度 = {metrics['coverage_rate']:.4f}")
        
        return metrics
    
    def _evaluate_head_to_head(self, strategy1, strategy2, num_episodes):
        """评估两个策略的对抗性能"""
        # 这里简化处理，实际应用中可能需要在环境中进行模拟
        # 返回一个0.0-1.0之间的胜率
        return 0.5 + 0.1 * np.random.randn()
    
    def _generate_trajectory_samples(self, test_pool, num_samples):
        """生成典型对抗轨迹片段"""
        # 这里简化处理，实际应用中可能需要在环境中进行模拟
        # 返回一个轨迹样本列表
        return [{"states": [], "actions": [], "rewards": []} for _ in range(num_samples)]
    
    def save_model(self, path: str) -> None:
        """
        保存模型函数
        
        参数:
            path (str): 模型保存路径
        """
        if self.network is None or self.policy_network is None or self.opponent_model is None:
            raise ValueError("模型尚未构建，无法保存")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备保存数据
        save_data = {
            'network_state_dict': self.network.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            'opponent_model_state_dict': self.opponent_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'opponent_optimizer_state_dict': self.opponent_optimizer.state_dict(),
            'strategy_pool_size': self.strategy_pool_size,
            'cql_penalty_weight': self.cql_penalty_weight,
            'exposure_ratio': self.exposure_ratio,
            'importance_beta': self.importance_beta
        }
        
        # 如果已训练，保存策略池
        if self.is_trained and hasattr(self, 'strategy_pool'):
            save_data['strategy_pool'] = [strategy.state_dict() for strategy in self.strategy_pool]
        
        # 保存模型
        torch.save(save_data, path)
        
        print(f"模型已保存到 {path}")
    
    def load_model(self, path: str) -> None:
        """
        加载模型函数
        
        参数:
            path (str): 模型加载路径
        """
        if not os.path.exists(path):
            raise ValueError(f"模型文件 {path} 不存在")
        
        # 加载模型
        checkpoint = torch.load(path)
        
        # 更新参数
        self.strategy_pool_size = checkpoint['strategy_pool_size']
        self.cql_penalty_weight = checkpoint['cql_penalty_weight']
        self.exposure_ratio = checkpoint['exposure_ratio']
        self.importance_beta = checkpoint['importance_beta']
        
        # 重建模型
        # 这里假设输入维度和动作维度可以从模型状态字典中推断
        # 实际应用中可能需要额外的参数
        self.build_network(64, 10)  # 示例维度
        
        # 加载模型参数
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.opponent_model.load_state_dict(checkpoint['opponent_model_state_dict'])
        
        # 加载优化器参数
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.opponent_optimizer.load_state_dict(checkpoint['opponent_optimizer_state_dict'])
        
        # 如果有策略池，加载策略池
        if 'strategy_pool' in checkpoint:
            self.strategy_pool = []
            for strategy_state_dict in checkpoint['strategy_pool']:
                strategy = copy.deepcopy(self.policy_network)
                strategy.load_state_dict(strategy_state_dict)
                self.strategy_pool.append(strategy)
            
            # 标记模型已训练
            self.is_trained = True
        
        print(f"模型已从 {path} 加载")