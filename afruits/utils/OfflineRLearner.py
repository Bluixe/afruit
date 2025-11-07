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
                 perturbation_scale: float = 0.05,
                 replay_ratio: float = 0.8,
                 num_quantiles: int = 200,
                 discount_factor: float = 0.99,
                 estimation_step: int = 1,
                 target_update_freq: int = 0,
                 reward_normalization: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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
        self.perturbation_scale = perturbation_scale
        self.replay_ratio = replay_ratio
        self.num_quantiles = num_quantiles
        self.discount_factor = discount_factor
        self.estimation_step = estimation_step
        self.target_update_freq = target_update_freq
        self.reward_normalization = reward_normalization
        self.device = device
        
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
        if not isinstance(self.cql_weight, float) or not (0.0 < self.cql_weight <= 10.0):
            raise ValueError(f"cql_weight必须在0.0-10.0范围内，当前值: {self.cql_weight}")
        
        # 验证vae_hidden_dim
        if not isinstance(self.vae_hidden_dim, int) or not (16 <= self.vae_hidden_dim <= 1024):
            raise ValueError(f"vae_hidden_dim必须为16-1024范围内的整数，当前值: {self.vae_hidden_dim}")
        
        # 验证perturbation_scale
        if not isinstance(self.perturbation_scale, float) or not (0.0 <= self.perturbation_scale <= 1.0):
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
                    obs=obs,
                    act=act,
                    rew=rew,
                    done=done,
                    terminated=done,
                    truncated=done,
                    obs_next=obs_next,
                    info={})
                )
        
        print(f"数据预处理完成: 共添加 {len(buffer)} 条数据到ReplayBuffer")
        return buffer
    
    def build_model(self, args: Dict = None) -> None:
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
        if args is None:
            args = {}
        
        # 设置默认参数
        default_args = {
            "n_embd": 128,
            "dropout": 0.1
        }
        
        # 更新参数
        for key, value in default_args.items():
            if key not in args:
                args[key] = value
        
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

        self.vae.to(self.device) if is_bcq else None
        self.policy.to(self.device)
        
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
            
            # 根据replay_ratio决定是否使用历史轨迹回放
            if np.random.random() < self.replay_ratio:
                try:
                    # 更新策略
                    result = self.policy.update(batch_size, buffer)
                    
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
        评估函数
        
        参数:
            test_env: 测试环境
            num_episodes (int): 测试轮次，默认为10
        
        返回值:
            metrics (Dict): 包含以下评估指标的字典:
                - avg_reward: 平均奖励
                - success_rate: 成功率
                - safety_margin: 安全边界系数
        
        功能描述:
            1. 在测试环境中评估训练好的策略
            2. 计算平均奖励、成功率等指标
            3. 返回评估结果
        """
        # 检查模型是否已训练
        if not self.is_trained or self.policy is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 初始化评估指标
        metrics = {
            'avg_reward': 0.0,
            'success_rate': 0.0,
            'safety_margin': 0.0
        }
        
        # 评估循环
        total_reward = 0.0
        success_count = 0
        
        for episode in range(num_episodes):
            # 重置环境
            obs = test_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # 选择动作
                act = self.policy.step(obs, self.device)
                
                # 执行动作
                obs_next, rew, done, info = test_env.step(act)
                
                # 累积奖励
                episode_reward += rew
                
                # 更新观测
                obs = obs_next
            
            # 累积总奖励
            total_reward += episode_reward
            
            # 判断是否成功
            if 'success' in info and info['success']:
                success_count += 1
        
        # 计算平均奖励
        metrics['avg_reward'] = total_reward / num_episodes
        
        # 计算成功率
        metrics['success_rate'] = success_count / num_episodes
        
        # 计算安全边界系数（示例计算方法）
        metrics['safety_margin'] = 0.8 * metrics['success_rate'] + 0.2 * (metrics['avg_reward'] / 100)
        
        print(f"评估完成: 平均奖励 = {metrics['avg_reward']:.2f}, 成功率 = {metrics['success_rate']:.2f}")
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """
        保存模型函数
        
        参数:
            path (str): 模型保存路径
        """
        if self.policy is None:
            raise ValueError("模型尚未构建，无法保存")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 判断是否使用BCQ算法
        is_bcq = hasattr(self, 'vae') and self.perturbation_scale > 0
        
        # 准备保存数据
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'policy_state_dict': self.policy.state_dict() if hasattr(self.policy, 'state_dict') else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'cql_weight': self.cql_weight,
            'vae_hidden_dim': self.vae_hidden_dim,
            'perturbation_scale': self.perturbation_scale,
            'replay_ratio': self.replay_ratio,
            'num_quantiles': self.num_quantiles,
            'discount_factor': self.discount_factor,
            'is_bcq': is_bcq
        }
        
        # 如果使用BCQ算法，保存VAE模型
        if is_bcq:
            save_data['vae_state_dict'] = self.vae.state_dict()
            save_data['vae_optimizer_state_dict'] = self.vae_optimizer.state_dict()
        
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
        checkpoint = torch.load(path, map_location=self.device)
        
        # 更新参数
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.cql_weight = checkpoint['cql_weight']
        self.vae_hidden_dim = checkpoint.get('vae_hidden_dim', 256)  # 默认值为256
        self.perturbation_scale = checkpoint.get('perturbation_scale', 0.05)  # 默认值为0.05
        self.replay_ratio = checkpoint.get('replay_ratio', 0.8)  # 默认值为0.8
        self.num_quantiles = checkpoint['num_quantiles']
        self.discount_factor = checkpoint['discount_factor']
        
        # 重建模型
        self.build_model()
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['policy_state_dict'] is not None and hasattr(self.policy, 'load_state_dict'):
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 如果是BCQ模型，加载VAE参数
        if checkpoint.get('is_bcq', False) and hasattr(self, 'vae'):
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
            self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        
        # 标记模型已训练
        self.is_trained = True
        
        print(f"模型已从 {path} 加载")