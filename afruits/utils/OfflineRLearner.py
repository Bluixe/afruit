import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
from tianshou.data import Batch, ReplayBuffer
from nets.cql.discrete_cql import DiscreteCQLPolicy
from nets.cql.nets import QValueNet   
from utils.BCQModels import VAE, DiscreteBCQPolicy
import copy

class OfflineRLearner:
    """
    离线强化学习训练器类
    
    功能描述：实现基于离线数据的强化学习算法，使用Conservative Q-Learning (CQL)
    
    核心功能：
    - 数据预处理：支持离线数据的处理与转换
    - 模型构建：支持CQL算法的网络结构
    - 策略训练：实现离线强化学习训练过程
    - 策略评估：提供多种评估指标与可视化工具
    """
    
    def __init__(self,
                 cql_weight: float = 0.5,
                 vae_hidden_dim: int = 256,
                 perturbation_scale: float = 0,
                 replay_ratio: float = 0.8,
                 num_quantiles: int = 200,
                 discount_factor: float = 0.99,
                 estimation_step: int = 1,
                 target_update_freq: int = 0,
                 reward_normalization: bool = False,
                 device: str = "cpu"):
        """
        初始化离线强化学习训练器
        
        参数:
            state_dim (int): 状态空间维度，默认为7，有效取值范围64-512
            action_dim (int): 动作空间维度，默认为7，有效取值范围3-20
            cql_weight (float): CQL正则化项系数，默认为0.5，有效取值范围0.1-1.0
            vae_hidden_dim (int): VAE隐藏层维度，默认为256，有效取值范围64-512
            perturbation_scale (float): BCQ动作扰动幅度，默认为0.05，有效取值范围0.01-0.2
            replay_ratio (float): 历史轨迹回放比例，默认为0.8，有效取值范围0.5-0.95
            num_quantiles (int): 分位数数量，默认为200，有效取值范围50-400
            discount_factor (float): 折扣因子，默认为0.99，有效取值范围0.9-0.999
            estimation_step (int): 估计步数，默认为1，有效取值范围1-10
            target_update_freq (int): 目标网络更新频率，默认为0，有效取值范围0-1000
            reward_normalization (bool): 是否进行奖励归一化，默认为False
            device (str): 训练设备，默认为"cuda"如果可用，否则为"cpu"
        """
        # 初始化参数
        self.state_dim = None  # 需要在preprocess_data时设置
        self.action_dim = None  # 需要在preprocess_data时设置
        self.cql_weight = cql_weight
        self.vae_hidden_dim = vae_hidden_dim
        self.perturbation_scale = 0
        self.replay_ratio = replay_ratio
        self.num_quantiles = num_quantiles
        self.discount_factor = discount_factor
        self.estimation_step = estimation_step
        self.target_update_freq = target_update_freq
        self.reward_normalization = reward_normalization
        self.device = device

        self.config_to_save = {
            'cql_weight': cql_weight,
            'vae_hidden_dim': vae_hidden_dim,
            'perturbation_scale': perturbation_scale,
            'replay_ratio': replay_ratio,
            'num_quantiles': num_quantiles,
            'discount_factor': discount_factor,
            'estimation_step': estimation_step,
            'target_update_freq': target_update_freq,
            'reward_normalization': reward_normalization,
        }
        
        # 初始化模型和优化器
        self.model = None
        self.policy = None
        self.optimizer = None
        
        # 初始化训练状态
        self.is_trained = False
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证cql_weight
        if not isinstance(self.cql_weight, float) or not (0.0 <= self.cql_weight <= 10.0):
            raise ValueError(f"cql_weight必须在0.0-10.0范围内，当前值: {self.cql_weight}")
        
        # 验证vae_hidden_dim
        if not isinstance(self.vae_hidden_dim, int) or not (16 <= self.vae_hidden_dim <= 1024):
            raise ValueError(f"vae_hidden_dim必须为16-1024范围内的整数，当前值: {self.vae_hidden_dim}")
        
        # 验证perturbation_scale
        if not (self.perturbation_scale <= 1.0):
            raise ValueError(f"perturbation_scale必须在0.0-1.0范围内，当前值: {self.perturbation_scale}")
        
        # 验证replay_ratio
        if not isinstance(self.replay_ratio, float) or not (0.0 <= self.replay_ratio <= 1.0):
            raise ValueError(f"replay_ratio必须在0.0-1.0范围内，当前值: {self.replay_ratio}")
    
    def preprocess_data(self, raw_trajectories: Dict) -> ReplayBuffer:
        """
        数据预处理函数
        
        参数:
            raw_trajectories (Dict): 原始轨迹数据集，包含专家演示轨迹
                                    （每条轨迹包含时间序列状态、动作和奖励）
        
        返回值:
            ReplayBuffer: 处理后的数据缓冲区
        
        功能描述:
            1. 将原始轨迹数据转换为ReplayBuffer格式
            2. 标准化状态和奖励
            3. 处理终止状态和截断状态
        """
        # 创建ReplayBuffer
        buffer = ReplayBuffer(size=100000)
        
        # 处理轨迹数据
        if isinstance(raw_trajectories, list):
            for trajectory in raw_trajectories:
                
                states = np.array(trajectory['states'])
                actions = np.array(trajectory['actions'])
                rewards = np.array(trajectory['rewards'])

                if self.state_dim is None:
                    assert len(states.shape) == 2, "状态数据必须为二维数组 (时间步长, 状态维度)"
                    self.state_dim = states.shape[1]
                if self.action_dim is None:
                    self.action_dim = int(np.max(actions)) + 1  # 假设动作是从0开始的整数
                
                # 添加数据到ReplayBuffer
                for i in range(len(states) - 1):
                    # 当前状态、动作、奖励
                    obs = states[i]
                    act = actions[i]
                    rew = rewards[i]
                    
                    # 下一个状态
                    obs_next = states[i + 1]
                    
                    # 判断是否为终止状态
                    done = False
                    if i == len(states) - 2:  # 最后一个转换
                        done = True

                    
                    # 添加到buffer
                    buffer.add(
                        Batch(
                        obs=torch.tensor(obs).float(),
                        act=torch.tensor(act).long(),
                        rew=rew,
                        done=done,
                        terminated=done,
                        truncated=done,
                        obs_next=torch.tensor(obs_next).float(),
                        info={})
                    )
        
        elif isinstance(raw_trajectories, dict):
            for traj_id, trajectory in raw_trajectories.items():
                # 检查轨迹数据是否包含状态、动作和奖励
                if 'states' not in trajectory or 'actions' not in trajectory or 'rewards' not in trajectory:
                    print(f"警告: 轨迹 {traj_id} 缺少状态、动作或奖励数据，已跳过")
                    continue
                
                states = trajectory['states']
                actions = trajectory['actions']
                rewards = trajectory['rewards']

                if self.state_dim is None:
                    assert len(states.shape) == 2, "状态数据必须为二维数组 (时间步长, 状态维度)"
                    self.state_dim = states.shape[1]
                if self.action_dim is None:
                    self.action_dim = int(np.max(actions)) + 1  # 假设动作是从0开始的整数
                
                # 检查数据长度是否匹配
                if not (len(states) == len(actions) == len(rewards)):
                    print(f"警告: 轨迹 {traj_id} 的状态、动作和奖励数据长度不匹配，已跳过")
                    continue
                
                # 添加数据到ReplayBuffer
                for i in range(len(states) - 1):
                    # 当前状态、动作、奖励
                    obs = states[i]
                    act = actions[i]
                    rew = rewards[i]
                    
                    # 下一个状态
                    obs_next = states[i + 1]
                    
                    # 判断是否为终止状态
                    done = False
                    if i == len(states) - 2:  # 最后一个转换
                        done = True

                    
                    # 添加到buffer
                    buffer.add(
                        Batch(
                        obs=torch.tensor(obs).float(),
                        act=torch.tensor(act).long(),
                        rew=rew,
                        done=done,
                        terminated=done,
                        truncated=done,
                        obs_next=torch.tensor(obs_next).float(),
                        info={})
                    )
        
        print(f"数据预处理完成: 共添加 {len(buffer)} 条数据到ReplayBuffer")
        return buffer
    
    def build_model(self, state_dim, action_dim) -> None:
        """
        模型构建函数
        
        参数:
            args (Dict, optional): 模型参数字典，包含网络结构参数
        
        功能描述:
            1. 构建Q网络模型
            2. 创建VAE模型（用于BCQ算法）
            3. 创建策略（CQL或BCQ）
            4. 设置优化器
        """
        
        # 设置默认参数
        args = {
            "n_embd": 128,
            "dropout": 0.1
        }

        self.config_to_save.update({
            'state_dim': state_dim,
            'action_dim': action_dim
        })

        if type(state_dim) == tuple:
            self.state_dim = state_dim[0]
        else:
            self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建Q网络模型
        self.model = QValueNet(
            obs_space=self.state_dim,
            action_space=self.action_dim,
            num_quantiles=self.num_quantiles,
            device=self.device,
            args=args
        )
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        
        # 判断是否使用BCQ算法（根据perturbation_scale是否大于0）
        if self.perturbation_scale > 0:
            # 创建VAE模型
            self.vae = VAE(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.vae_hidden_dim,
                latent_dim=32,  # 默认潜在空间维度
                discrete=True   # 使用离散动作空间
            )
            
            # VAE优化器
            self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=3e-4)
            
            # 创建BCQ策略
            self.policy = DiscreteBCQPolicy(
                q_network=self.model,
                vae=self.vae,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device=self.device,
                perturbation_scale=self.perturbation_scale,
                num_samples=10,  # 默认采样数量
                threshold=0.3    # 默认阈值
            )
            
            print("BCQ模型构建完成")
        else:
            # 创建CQL策略
            self.policy = DiscreteCQLPolicy(
                model=self.model,
                optim=self.optimizer,
                discount_factor=self.discount_factor,
                num_quantiles=self.num_quantiles,
                estimation_step=self.estimation_step,
                target_update_freq=self.target_update_freq,
                reward_normalization=self.reward_normalization,
                min_q_weight=self.cql_weight
            )
            
            print("CQL模型构建完成")
    
    def train(self, buffer: ReplayBuffer, epochs: int = 100, batch_size: int = 64) -> Dict:
        """
        训练函数
        
        参数:
            buffer (ReplayBuffer): 数据缓冲区
            epochs (int): 训练轮次，默认为100
            batch_size (int): 批次大小，默认为64
        
        返回值:
            Dict: 训练历史记录，包含损失值等指标
        
        功能描述:
            1. 使用CQL或BCQ算法训练策略
            2. 支持历史轨迹回放比例控制
            3. 记录训练过程中的损失值
            4. 返回训练历史记录
        """
        # 检查模型是否已构建
        if self.policy is None:
            raise ValueError("模型尚未构建，请先调用build_model方法")
        
        # 训练历史记录
        history = {
            'loss': [],
            'qr_loss': [],
            'cql_loss': [],
            'vae_loss': []
        }
        
        # 判断是否使用BCQ算法
        is_bcq = hasattr(self, 'vae') and self.perturbation_scale > 0

        self.vae = self.vae.to(self.device) if is_bcq else None
        self.policy = self.policy.to(self.device)
        
        # 训练循环
        for epoch in range(epochs):
            # 如果使用BCQ算法，需要先训练VAE
            if is_bcq:
                try:
                    # 从buffer中采样数据
                    batch_indices = np.random.choice(len(buffer), batch_size, replace=False)
                    batch = buffer[batch_indices]
                    
                    # 转换为张量
                    states = torch.tensor(batch.obs, dtype=torch.float32).to(self.device)
                    actions = torch.tensor(batch.act, dtype=torch.long).to(self.device)
                    
                    # 训练VAE
                    self.vae_optimizer.zero_grad()
                    recon_actions, mean, log_var = self.vae(states, actions)
                    
                    # 计算重建损失（交叉熵损失）
                    recon_loss = nn.CrossEntropyLoss()(recon_actions, actions)
                    
                    # 计算KL散度
                    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                    
                    # 总损失
                    vae_loss = recon_loss + 0.5 * kl_loss
                    
                    # 反向传播
                    vae_loss.backward()
                    self.vae_optimizer.step()
                    
                    # 记录VAE损失
                    history['vae_loss'].append(vae_loss.item())
                except Exception as e:
                    print(f"VAE训练出错: {e}")
            else:
                batch_indices = np.random.choice(len(buffer), batch_size, replace=False)
                batch = buffer[batch_indices]
                
            # 根据replay_ratio决定是否使用历史轨迹回放
            if np.random.random() < self.replay_ratio:
                try:
                    # 更新策略
                    result = self.policy.update(batch_size, buffer, self.device)
                    
                    # 记录损失值
                    history['loss'].append(result['loss'])
                    if 'loss/qr' in result:
                        history['qr_loss'].append(result['loss/qr'])
                    if 'loss/cql' in result:
                        history['cql_loss'].append(result['loss/cql'])
                except Exception as e:
                    print(f"策略更新出错: {e}")
                    # 如果更新失败，添加占位值以保持历史记录的连续性
                    if len(history['loss']) < epoch:
                        history['loss'].append(float('nan'))
                    if 'loss/qr' in result and len(history['qr_loss']) < epoch:
                        history['qr_loss'].append(float('nan'))
                    if 'loss/cql' in result and len(history['cql_loss']) < epoch:
                        history['cql_loss'].append(float('nan'))
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                loss_str = f"Epoch {epoch+1}/{epochs}"
                if history['loss'] and len(history['loss']) > 0:
                    loss_str += f", Loss: {history['loss'][-1]:.4f}"
                if is_bcq and history['vae_loss'] and len(history['vae_loss']) > 0:
                    loss_str += f", VAE Loss: {history['vae_loss'][-1]:.4f}"
                if history['qr_loss'] and len(history['qr_loss']) > 0:
                    loss_str += f", QR Loss: {history['qr_loss'][-1]:.4f}"
                if history['cql_loss'] and len(history['cql_loss']) > 0:
                    loss_str += f", CQL Loss: {history['cql_loss'][-1]:.4f}"
                print(loss_str)
        
        # 标记模型已训练
        self.is_trained = True
        
        return history
    
    def evaluate(self, test_env, num_episodes: int = 10) -> Dict:
        """
        评估函数（离散动作）
        
        更新点:
        - 使用argmax从Q值(logits)中选出离散动作
        - 在可用情况下（环境提供专家动作）计算分类准确率accuracy
        
        参数:
            test_env: 测试环境（可选实现 get_expert_action(obs)->int）
            num_episodes (int): 测试轮次，默认为10
        
        返回值:
            metrics (Dict): 包含以下评估指标:
                - avg_reward: 平均奖励
                - success_rate: 成功率
                - safety_margin: 安全边界系数
                - accuracy: 分类准确率（若环境不提供expert动作则为None）
        """
        # 初始化评估指标
        metrics = {
            'avg_reward': 0.0,
            'success_rate': 0.0,
            'safety_margin': 0.0,
            'accuracy': None
        }
        
        total_reward = 0.0
        success_count = 0
        
        # 分类准确率统计
        correct_predictions = 0
        total_samples = 0
        errors = []
        
        for episode in range(num_episodes):
            obs = test_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # 获取专家动作（如果环境提供）
                expert_action = None
                if hasattr(test_env, 'get_expert_action'):
                    try:
                        expert_action = int(test_env.get_expert_action(obs))
                    except Exception:
                        expert_action = None
                
                # 基于当前模型/策略选取离散动作（argmax）
                try:
                    # 优先使用Q网络的argmax
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    if hasattr(self, 'device'):
                        obs_tensor = obs_tensor.to(self.device)
                    with torch.no_grad():
                        q_values = self.model(obs_tensor)  # [1, action_dim]
                        act_idx = int(torch.argmax(q_values, dim=-1).item())
                except Exception:
                    # 回退到策略接口
                    act_idx = int(self.policy.step(obs, self.device))
                
                # 交互环境（离散action索引）
                obs_next, rew, done, info = test_env.step(act_idx)
                
                # 统计accuracy（仅在有专家动作时）
                if expert_action is not None:
                    error = abs(act_idx - expert_action)
                    errors.append(error)
                    if act_idx == expert_action:
                        correct_predictions += 1
                    total_samples += 1
                
                # 奖励与状态更新
                episode_reward += float(rew)
                obs = obs_next
            
            total_reward += episode_reward
            if 'success' in info and info['success']:
                success_count += 1
        
        # 聚合指标
        metrics['avg_reward'] = total_reward / max(num_episodes, 1)
        metrics['success_rate'] = success_count / max(num_episodes, 1)
        metrics['safety_margin'] = 0.8 * metrics['success_rate'] + 0.2 * (metrics['avg_reward'] / 100.0)
        
        # 计算准确率
        if total_samples > 0:
            metrics['accuracy'] = correct_predictions / total_samples
            print(f"评估完成: 平均奖励 = {metrics['avg_reward']:.2f}, 成功率 = {metrics['success_rate']:.2f}, 准确率 = {metrics['accuracy']:.4f}")
        else:
            print(f"评估完成: 平均奖励 = {metrics['avg_reward']:.2f}, 成功率 = {metrics['success_rate']:.2f}, 准确率 = N/A (未提供专家动作)")
        
        return metrics
    
    def save_model(self, save_path = None) -> None:
        """
        保存模型函数
        
        参数:
            path (str): 模型保存路径
        """
        if self.policy is None:
            raise ValueError("模型尚未构建，无法保存")
        
        if save_path is None:
            # 默认保存路径
            save_path = f"models/offline_rl.pt"
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        
        # 准备保存数据
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'policy_state_dict': self.policy.state_dict() if hasattr(self.policy, 'state_dict') else None,
            'config': {k:v for k,v in self.config_to_save.items()}
        }
        
        # 保存模型
        torch.save(model_state, save_path)
        import json
        config_path = os.path.splitext(save_path)[0] + '_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_state['config'], f, ensure_ascii=False, indent=4)
        
        print(f"模型已保存到 {save_path}")
    
    @staticmethod
    def load_model(load_path, device: torch.device = None) -> 'OfflineRLearner':
        """
        加载模型函数
        
        参数:
            path (str): 模型加载路径
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_path is None:
            load_path = f"models/offline_rl.pt"
        
        # 加载模型
        checkpoint = torch.load(load_path, map_location=device)
        config = checkpoint['config']

        model = OfflineRLearner(
            cql_weight = config['cql_weight'],
            vae_hidden_dim = config['vae_hidden_dim'],
            perturbation_scale = config['perturbation_scale'],
            replay_ratio = config['replay_ratio'],
            num_quantiles = config['num_quantiles'],
            discount_factor = config['discount_factor'],
            estimation_step = config['estimation_step'],
            target_update_freq = config['target_update_freq'],
            reward_normalization = config['reward_normalization'],
            device = device
        )

        model.build_model(config['state_dim'], config['action_dim'])
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.policy.load_state_dict(checkpoint['policy_state_dict'])

        return model