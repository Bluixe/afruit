import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class VAE(nn.Module):
    """
    变分自编码器 (VAE) 模型，用于 BCQ 算法中的动作生成
    
    功能描述：
    - 编码器：将状态-动作对编码为潜在空间表示
    - 解码器：从潜在空间表示重建动作
    - 支持连续和离散动作空间
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, latent_dim: int = 32, discrete: bool = True):
        """
        初始化 VAE 模型
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_dim (int): 隐藏层维度，默认为256
            latent_dim (int): 潜在空间维度，默认为32
            discrete (bool): 是否为离散动作空间，默认为True
        """
        super(VAE, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.discrete = discrete
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和方差预测
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出层 - 根据动作空间类型不同
        if discrete:
            # 离散动作空间 - 使用 softmax 输出概率分布
            self.output_layer = nn.Linear(hidden_dim, action_dim)
        else:
            # 连续动作空间 - 输出动作值和方差
            self.output_mean = nn.Linear(hidden_dim, action_dim)
            self.output_log_var = nn.Linear(hidden_dim, action_dim)
    
    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将状态-动作对编码为潜在空间表示
        
        参数:
            state (torch.Tensor): 状态张量 [batch_size, state_dim]
            action (torch.Tensor): 动作张量 [batch_size, action_dim]
            
        返回:
            mean (torch.Tensor): 均值张量 [batch_size, latent_dim]
            log_var (torch.Tensor): 对数方差张量 [batch_size, latent_dim]
        """
        # 对于离散动作，将其转换为 one-hot 编码
        if self.discrete and action.dim() == 1:
            action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_one_hot = action
        
        # 拼接状态和动作
        x = torch.cat([state, action_one_hot], dim=1)
        x = self.encoder(x)
        
        # 计算均值和对数方差
        mean = self.mean(x)
        log_var = self.log_var(x)
        
        return mean, log_var
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        使用重参数化技巧从均值和方差中采样
        
        参数:
            mean (torch.Tensor): 均值张量 [batch_size, latent_dim]
            log_var (torch.Tensor): 对数方差张量 [batch_size, latent_dim]
            
        返回:
            z (torch.Tensor): 采样的潜在变量 [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间表示解码为动作
        
        参数:
            state (torch.Tensor): 状态张量 [batch_size, state_dim]
            z (torch.Tensor): 潜在变量 [batch_size, latent_dim]
            
        返回:
            action (torch.Tensor): 重建的动作 [batch_size, action_dim]
        """
        # 拼接状态和潜在变量
        x = torch.cat([state, z], dim=1)
        x = self.decoder(x)
        
        # 根据动作空间类型输出不同的结果
        if self.discrete:
            # 离散动作空间 - 输出 logits
            action_logits = self.output_layer(x)
            return action_logits
        else:
            # 连续动作空间 - 输出均值和方差
            action_mean = self.output_mean(x)
            action_log_var = self.output_log_var(x)
            return action_mean, action_log_var
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            state (torch.Tensor): 状态张量 [batch_size, state_dim]
            action (torch.Tensor): 动作张量 [batch_size, action_dim] 或 [batch_size]
            
        返回:
            recon_action (torch.Tensor): 重建的动作 [batch_size, action_dim]
            mean (torch.Tensor): 均值张量 [batch_size, latent_dim]
            log_var (torch.Tensor): 对数方差张量 [batch_size, latent_dim]
        """
        # 编码
        mean, log_var = self.encode(state, action)
        
        # 采样潜在变量
        z = self.reparameterize(mean, log_var)
        
        # 解码
        recon_action = self.decode(state, z)
        
        return recon_action, mean, log_var
    
    def sample_action(self, state: torch.Tensor, num_samples: int = 10, perturbation_scale: float = 0.05) -> torch.Tensor:
        """
        从 VAE 中采样动作
        
        参数:
            state (torch.Tensor): 状态张量 [batch_size, state_dim]
            num_samples (int): 采样数量，默认为10
            perturbation_scale (float): 扰动幅度，默认为0.05
            
        返回:
            sampled_action (torch.Tensor): 采样的动作 [batch_size]
        """
        batch_size = state.size(0)
        
        # 扩展状态以匹配采样数量
        state_expanded = state.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        
        # 从标准正态分布中采样潜在变量
        z = torch.randn(batch_size * num_samples, self.latent_dim, device=state.device)
        
        # 添加扰动
        z = z * perturbation_scale
        
        # 解码为动作
        action_logits = self.decode(state_expanded, z)
        
        if self.discrete:
            # 对于离散动作，重塑 logits 并计算 softmax
            action_probs = F.softmax(action_logits, dim=1)
            action_probs = action_probs.view(batch_size, num_samples, -1)
            
            # 计算每个动作的平均概率
            avg_action_probs = action_probs.mean(dim=1)
            
            # 从概率分布中采样动作
            sampled_action = torch.multinomial(avg_action_probs, num_samples=1).squeeze(-1)
        else:
            # 对于连续动作，重塑输出并计算平均值
            action_mean, _ = action_logits
            action_mean = action_mean.view(batch_size, num_samples, -1)
            
            # 计算每个动作的平均值
            avg_action = action_mean.mean(dim=1)
            
            # 添加扰动
            noise = torch.randn_like(avg_action) * perturbation_scale
            sampled_action = avg_action + noise
        
        return sampled_action


class DiscreteBCQPolicy(nn.Module):
    """
    离散 BCQ 策略类
    
    功能描述：
    - 实现离散动作空间的 Batch Constrained Q-learning (BCQ) 算法
    - 使用 VAE 生成动作，使用 Q 网络评估动作价值
    """
    
    def __init__(
        self,
        q_network: nn.Module,
        vae: VAE,
        state_dim: int,
        action_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        perturbation_scale: float = 0.05,
        num_samples: int = 10,
        threshold: float = 0.3
    ):
        """
        初始化离散 BCQ 策略
        
        参数:
            q_network (nn.Module): Q 网络模型
            vae (VAE): VAE 模型
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            device (str): 设备，默认为 "cuda" 如果可用，否则为 "cpu"
            perturbation_scale (float): 动作扰动幅度，默认为 0.05
            num_samples (int): 采样数量，默认为 10
            threshold (float): 阈值，用于过滤低概率动作，默认为 0.3
        """
        super(DiscreteBCQPolicy, self).__init__()
        
        self.q_network = q_network
        self.vae = vae
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.perturbation_scale = perturbation_scale
        self.num_samples = num_samples
        self.threshold = threshold
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            state (torch.Tensor): 状态张量 [batch_size, state_dim]
            
        返回:
            action (torch.Tensor): 选择的动作 [batch_size]
        """
        with torch.no_grad():
            # 从 VAE 中采样多个动作
            state_expanded = state.unsqueeze(1).repeat(1, self.num_samples, 1).view(-1, self.state_dim)
            z = torch.randn(state_expanded.size(0), self.vae.latent_dim, device=self.device)
            z = z * self.perturbation_scale
            
            # 解码为动作 logits
            action_logits = self.vae.decode(state_expanded, z)
            action_probs = F.softmax(action_logits, dim=1)
            
            # 重塑为 [batch_size, num_samples, action_dim]
            batch_size = state.size(0)
            action_probs = action_probs.view(batch_size, self.num_samples, -1)
            
            # 计算每个动作的平均概率
            avg_action_probs = action_probs.mean(dim=1)
            
            # 创建掩码，过滤低概率动作
            mask = (avg_action_probs >= self.threshold).float()
            
            # 如果所有动作都被过滤，则选择概率最高的动作
            if mask.sum(dim=1).min() == 0:
                mask = torch.zeros_like(mask).scatter_(1, avg_action_probs.argmax(dim=1, keepdim=True), 1.0)
            
            # 获取 Q 值
            q_values, _ = self.q_network(state)
            
            # 应用掩码
            q_values = q_values * mask - (1 - mask) * 1e8
            
            # 选择 Q 值最高的动作
            action = q_values.argmax(dim=1)
        
        return action
    
    def step(self, obs, device):
        """
        根据观测选择动作
        
        参数:
            obs: 观测
            device: 设备
            
        返回:
            act: 选择的动作
        """
        obs = torch.tensor(obs).float().to(device)
        return self.forward(obs).item()
    
    def get_q_value(self, obs, action, device):
        """
        获取指定观测和动作的 Q 值
        
        参数:
            obs: 观测
            action: 动作
            device: 设备
            
        返回:
            q: Q 值
        """
        obs = torch.tensor(obs).float().to(device)
        q, _ = self.q_network(obs)
        action = torch.tensor(action).long().to(device)
        q = q.gather(1, action.unsqueeze(1)).squeeze(1)
        return q