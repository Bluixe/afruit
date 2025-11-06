import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time

class DiffusionTrajGenerator:
    """
    扩散轨迹生成器
    
    功能描述：基于扩散模型的轨迹生成，实现多步噪声调度与物理约束集成
    
    核心特性：
    ◆ 渐进式生成：多步噪声调度实现高质量轨迹生成
    ◆ 物理约束：集成飞行动力学模型
    ◆ 条件生成：支持场景上下文输入
    ◆ 非平衡采样：加速推理过程
    """
    
    def __init__(self, 
                 diffusion_steps: int = 1000, 
                 noise_schedule: str = "cosine", 
                 physics_constraints: dict = None):
        """
        初始化扩散轨迹生成器
        
        参数:
            diffusion_steps (int): 扩散步数，取值范围10-2000
            noise_schedule (str): 噪声调度类型，可选["linear", "cosine"]
            physics_constraints (dict): 物理规则约束字典
        """
        # 参数有效性检查
        assert 10 <= diffusion_steps <= 2000, "diffusion_steps必须在10-2000范围内"
        assert noise_schedule in ["linear", "cosine"], "noise_schedule必须是'linear'或'cosine'"
        
        # 初始化参数
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.physics_constraints = physics_constraints if physics_constraints else {}
        
        # 初始化扩散模型参数
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # 初始化网络模型
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练相关参数
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.MSELoss()
        
    def _get_noise_schedule(self) -> np.ndarray:
        """
        获取噪声调度
        
        返回:
            betas (np.ndarray): 噪声方差序列
        """
        if self.noise_schedule == "linear":
            # 线性噪声调度
            scale = 1000 / self.diffusion_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, self.diffusion_steps)
        
        elif self.noise_schedule == "cosine":
            # 余弦噪声调度
            steps = self.diffusion_steps + 1
            t = np.linspace(0, self.diffusion_steps, steps) / self.diffusion_steps
            alphas_cumprod = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0001, 0.9999)
    
    def load_dataset(self, data_path: str, seq_len: int = None) -> Dict:
        """
        数据加载
        
        参数:
            data_path (str): 预处理后数据文件路径
            seq_len (int): 轨迹序列长度
            
        返回:
            dataloader (DataLoader): 训练数据加载器
        """
        # 加载数据
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件 {data_path} 不存在")
        
        # 加载数据（假设为numpy格式）
        data = np.load(data_path)
        
        # 提取轨迹数据
        trajectories = data['trajectories'] if 'trajectories' in data else data
        
        # 序列长度处理
        if seq_len is not None and trajectories.shape[1] > seq_len:
            # 随机截取指定长度的片段
            start_indices = np.random.randint(0, trajectories.shape[1] - seq_len, size=trajectories.shape[0])
            sampled_trajectories = np.array([
                trajectories[i, start_idx:start_idx+seq_len] 
                for i, start_idx in enumerate(start_indices)
            ])
            trajectories = sampled_trajectories
        
        # 转换为PyTorch张量
        trajectories_tensor = torch.FloatTensor(trajectories)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(trajectories_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        return {
            'dataloader': dataloader,
            'data_shape': trajectories.shape,
            'seq_len': trajectories.shape[1],
            'feature_dim': trajectories.shape[2] if len(trajectories.shape) > 2 else 1
        }
    
    def build_network(self, input_dim: int, cond_dim: int = 0) -> nn.Module:
        """
        构建网络
        
        参数:
            input_dim (int): 输入特征维度
            cond_dim (int): 条件信息维度
            
        返回:
            model (nn.Module): 扩散模型网络
        """
        # 构建U-Net结构的扩散模型
        class DiffusionUNet(nn.Module):
            def __init__(self, input_dim, cond_dim, time_emb_dim=128):
                super().__init__()
                self.time_embed = nn.Sequential(
                    nn.Linear(1, time_emb_dim),
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, time_emb_dim),
                )
                
                # 条件嵌入
                self.cond_embed = nn.Sequential(
                    nn.Linear(cond_dim, time_emb_dim),
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, time_emb_dim),
                ) if cond_dim > 0 else None
                
                # 编码器
                self.encoder = nn.ModuleList([
                    nn.Linear(input_dim + time_emb_dim, 128),
                    nn.Linear(128, 256),
                    nn.Linear(256, 512),
                ])
                
                # 中间层
                self.middle = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.SiLU(),
                    nn.Linear(512, 512),
                )
                
                # 解码器
                self.decoder = nn.ModuleList([
                    nn.Linear(512 + 256, 256),
                    nn.Linear(256 + 128, 128),
                    nn.Linear(128 + input_dim, input_dim),
                ])
                
                # 激活函数
                self.act = nn.SiLU()
                
            def forward(self, x, t, cond=None):
                # 时间嵌入
                t_emb = self.time_embed(t.unsqueeze(-1))
                
                # 条件嵌入
                if self.cond_embed is not None and cond is not None:
                    c_emb = self.cond_embed(cond)
                    t_emb = t_emb + c_emb
                
                # 初始特征
                h = torch.cat([x, t_emb], dim=-1)
                
                # 编码器前向传播
                skip_connections = [x]
                for layer in self.encoder:
                    h = self.act(layer(h))
                    skip_connections.append(h)
                
                # 中间层
                h = self.middle(h)
                
                # 解码器前向传播
                for i, layer in enumerate(self.decoder):
                    h = torch.cat([h, skip_connections[-(i+1)]], dim=-1)
                    h = self.act(layer(h)) if i < len(self.decoder) - 1 else layer(h)
                
                return h
        
        # 创建模型
        model = DiffusionUNet(input_dim, cond_dim)
        model = model.to(self.device)
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        self.model = model
        return model
    
    def train(self, dataloader: DataLoader, epochs: int = 100) -> Dict:
        """
        训练过程
        
        参数:
            dataloader (DataLoader): 训练数据加载器
            epochs (int): 训练轮数
            
        返回:
            训练统计 (Dict)
        """
        if self.model is None:
            raise ValueError("请先调用build_network构建模型")
        
        self.model.train()
        losses = []
        
        # 训练循环
        for epoch in range(epochs):
            epoch_losses = []
            start_time = time.time()
            
            for batch in dataloader:
                # 获取轨迹数据
                trajectories = batch[0].to(self.device)
                batch_size = trajectories.shape[0]
                
                # 随机选择时间步
                t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)
                
                # 添加噪声
                noise = torch.randn_like(trajectories)
                alphas_cumprod_t = torch.tensor(self.alphas_cumprod, device=self.device)[t]
                alphas_cumprod_t = alphas_cumprod_t.view(-1, 1, 1)
                
                noisy_trajectories = torch.sqrt(alphas_cumprod_t) * trajectories + \
                                    torch.sqrt(1 - alphas_cumprod_t) * noise
                
                # 预测噪声
                predicted_noise = self.model(noisy_trajectories, t / self.diffusion_steps)
                
                # 计算损失
                loss = self.loss_fn(predicted_noise, noise)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # 更新学习率
            self.scheduler.step()
            
            # 计算平均损失
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            # 打印训练信息
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Time: {elapsed:.2f}s")
        
        return {
            'loss_curve': losses,
            'final_loss': losses[-1],
            'epochs': epochs
        }
    
    def generate(self, batch_size: int = 1, cond_data: torch.Tensor = None, 
                 validity_flag: bool = True) -> Dict:
        """
        生成轨迹
        
        参数:
            batch_size (int): 生成批次大小
            cond_data (torch.Tensor): 条件信息数据
            validity_flag (bool): 是否检查物理有效性
            
        返回:
            生成结果 (Dict)
                - trajectories: 轨迹动作序列
                - validity_flags: 物理规则合规标记
        """
        if self.model is None:
            raise ValueError("请先调用build_network构建模型")
        
        self.model.eval()
        
        # 获取模型输入维度
        input_shape = next(self.model.parameters()).shape
        traj_dim = input_shape[0] if len(input_shape) == 1 else input_shape[1]
        
        # 初始化随机噪声
        x = torch.randn((batch_size, traj_dim), device=self.device)
        
        # 逐步去噪
        for i in reversed(range(self.diffusion_steps)):
            t = torch.ones(batch_size, device=self.device) * i / self.diffusion_steps
            
            # 无梯度计算
            with torch.no_grad():
                # 预测噪声
                predicted_noise = self.model(x, t, cond_data)
                
                # 计算去噪步骤
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # 更新x
                x = (1 / torch.sqrt(torch.tensor(alpha))) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
                ) + torch.sqrt(beta) * noise
        
        # 物理有效性检查
        validity_flags = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        if validity_flag and self.physics_constraints:
            validity_flags = self._check_physics_validity(x)
        
        return {
            'trajectories': x.cpu().numpy(),
            'validity_flags': validity_flags.cpu().numpy()
        }
    
    def _check_physics_validity(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        检查物理有效性
        
        参数:
            trajectories (torch.Tensor): 生成的轨迹
            
        返回:
            validity_flags (torch.Tensor): 有效性标志
        """
        batch_size = trajectories.shape[0]
        validity_flags = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # 检查物理约束
        if 'max_velocity' in self.physics_constraints:
            max_vel = self.physics_constraints['max_velocity']
            # 假设轨迹中包含速度信息
            velocities = trajectories[:, :, 3:6] if trajectories.dim() > 2 else trajectories
            vel_magnitude = torch.norm(velocities, dim=-1)
            # 标记超过最大速度的轨迹
            invalid_vel = torch.any(vel_magnitude > max_vel, dim=-1)
            validity_flags = validity_flags & ~invalid_vel
        
        # 可以添加更多物理约束检查
        
        return validity_flags
    
    def next_process(self, raw_trajectories: torch.Tensor, 
                    violation_threshold: float = 0.3) -> Dict:
        """
        后处理
        
        参数:
            raw_trajectories (torch.Tensor): 原始生成轨迹
            violation_threshold (float): 违规阈值
            
        返回:
            处理结果 (Dict)
        """
        # 转换为numpy数组
        trajectories = raw_trajectories.cpu().numpy() if isinstance(raw_trajectories, torch.Tensor) else raw_trajectories
        batch_size = trajectories.shape[0]
        
        # 初始化结果
        valid_trajectories = []
        validity_flags = np.ones(batch_size, dtype=bool)
        
        # 对每个轨迹进行处理
        for i, traj in enumerate(trajectories):
            # 物理约束修正
            corrected_traj = self._apply_physics_correction(traj)
            
            # 计算违规率
            violation_rate = self._compute_violation_rate(traj, corrected_traj)
            
            # 判断是否有效
            if violation_rate > violation_threshold:
                validity_flags[i] = False
            
            # 存储修正后的轨迹
            valid_trajectories.append(corrected_traj)
        
        return {
            'trajectories': np.array(valid_trajectories),
            'validity_flags': validity_flags
        }
    
    def _apply_physics_correction(self, trajectory: np.ndarray) -> np.ndarray:
        """
        应用物理约束修正
        
        参数:
            trajectory (np.ndarray): 单个轨迹
            
        返回:
            corrected_traj (np.ndarray): 修正后的轨迹
        """
        corrected_traj = trajectory.copy()
        
        # 应用物理约束
        if 'max_velocity' in self.physics_constraints:
            max_vel = self.physics_constraints['max_velocity']
            # 假设轨迹中包含速度信息
            if len(trajectory.shape) > 1 and trajectory.shape[1] >= 6:
                velocities = trajectory[:, 3:6]
                vel_magnitude = np.linalg.norm(velocities, axis=1)
                
                # 找到超过最大速度的点
                invalid_indices = np.where(vel_magnitude > max_vel)[0]
                
                # 修正速度
                for idx in invalid_indices:
                    scale_factor = max_vel / vel_magnitude[idx]
                    corrected_traj[idx, 3:6] = velocities[idx] * scale_factor
        
        # 可以添加更多物理约束修正
        
        return corrected_traj
    
    def _compute_violation_rate(self, original_traj: np.ndarray, corrected_traj: np.ndarray) -> float:
        """
        计算违规率
        
        参数:
            original_traj (np.ndarray): 原始轨迹
            corrected_traj (np.ndarray): 修正后的轨迹
            
        返回:
            violation_rate (float): 违规率
        """
        # 计算修正前后的差异
        diff = np.abs(original_traj - corrected_traj)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 计算违规率（简单示例）
        violation_rate = mean_diff / (max_diff + 1e-8)
        
        return violation_rate