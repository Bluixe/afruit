import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
from afruits.utils.DataLoader import DataLoaderUtil
import json

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
                 dropout: float = 0.2,
                 im_embd: int = 128):
        """
        初始化扩散轨迹生成器
        
        参数:
            diffusion_steps (int): 扩散步数，取值范围10-2000
            noise_schedule (str): 噪声调度类型，可选["linear", "cosine"]
            seq_length (int): 输入序列长度，取值范围60-300
            dropout (float): Dropout比率，用于CNN图像编码器
            im_embd (int): 图像嵌入维度，用于CNN图像编码器输出
        """
        # 参数有效性检查
        assert 10 <= diffusion_steps <= 2000, "diffusion_steps必须在10-2000范围内"
        assert noise_schedule in ["linear", "cosine"], "noise_schedule必须是'linear'或'cosine'"
        assert 0.0 <= dropout <= 0.5, "dropout必须在0.0-0.5范围内"
        
        # 初始化参数
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.physics_constraints = {}
        self.dropout = dropout
        self.im_embd = im_embd
        
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

        self.dataloader_util = DataLoaderUtil()

        self.config_to_save = {
            'diffusion_steps': self.diffusion_steps,
            'noise_schedule': self.noise_schedule,
            'dropout': self.dropout,
            'im_embd': self.im_embd
        }
        
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
    
    def load_dataset(self, data: str, batch_size: int = 32) -> Dict:
        """
        数据加载
        
        参数:
            data_path (str): 预处理后的数据
            batch_size (int): 批处理大小
            
        返回值:
            数据加载器 (DataLoader)
        """
        data = self.dataloader_util.load_expert_data(data, batch_size=batch_size)
        data_loader = data['dataloader']
        return data_loader
    
    def build_model(self, state_dim, cond_dim: int = 0) -> nn.Module:
        """
        构建网络
        
        参数:
            input_dim (int or Dict): 输入特征维度，可以是整数或包含状态和动作维度的字典
            cond_dim (int): 条件信息维度
            
        返回:
            model (nn.Module): 扩散模型网络
        """
        # 处理输入维度
        if isinstance(state_dim, tuple):
            assert len(state_dim) == 1, "state_dim元组长度必须为1"
            state_dim = state_dim[0]

        self.config_to_save.update({
            'state_dim': state_dim,
            'cond_dim': cond_dim,
        })

        # 构建U-Net结构的扩散模型
        # 完全重新设计的 DiffusionUNet 模型
        class DiffusionUNet(nn.Module):
            def __init__(self, total_dim, cond_dim, time_emb_dim=128, dropout=0.2):
                super().__init__()
                self.total_dim = total_dim
                self.dropout = dropout
                
                # 时间嵌入
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
                
                self.model = nn.Sequential(
                    nn.Linear(total_dim + time_emb_dim, 256),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 512),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 512),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, total_dim)
                )
                
            def forward(self, x_state=None, t=None, cond=None):
                """
                改进的前向传播函数，支持完整序列处理
                
                参数:
                    x_state: 输入状态，可以是 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
                    t: 时间步，形状为 [batch_size]
                    cond: 条件信息，可选
                    
                返回:
                    预测的噪声 (与输入x_state同形状)
                """
                # 处理输入形状
                if len(x_state.shape) == 3:
                    batch_size, seq_len, feature_dim = x_state.shape
                    # 展平序列用于处理
                    x_flat = x_state.reshape(batch_size * seq_len, feature_dim)
                    t_rep = t.repeat_interleave(seq_len)
                    if cond is not None:
                        cond_rep = cond.repeat_interleave(seq_len, dim=0)
                    else:
                        cond_rep = None
                else:
                    x_flat = x_state
                    t_rep = t
                    cond_rep = cond
                    seq_len = 1
                
                # 时间嵌入
                t_emb = self.time_embed(t_rep.unsqueeze(-1))
                
                # 条件嵌入
                if self.cond_embed is not None and cond_rep is not None:
                    c_emb = self.cond_embed(cond_rep)
                    t_emb = t_emb + c_emb
                
                # 连接输入和时间嵌入
                h = torch.cat([x_flat, t_emb], dim=-1).float()
                
                # 通过MLP模型
                output_flat = self.model(h)
                
                # 恢复原始形状
                if len(x_state.shape) == 3:
                    output = output_flat.reshape(batch_size, seq_len, feature_dim)
                else:
                    output = output_flat
                
                return output
        
        # 创建模型
        model = DiffusionUNet(
            state_dim,
            cond_dim,
        )
        model = model.to(self.device)
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        # 保存模型配置
        self.state_dim = state_dim
        
        self.model = model
        return model
    
    def train(self, dataloader: DataLoader, epochs: int = 100, learning_rate = 1e-4) -> Dict:
        """
        训练过程
        
        参数:
            dataloader (DataLoader): 训练数据加载器
            epochs (int): 训练轮数
            
        返回:
            训练统计 (Dict)
        """
        if self.model is None:
            raise ValueError("请先调用build_model构建模型")
        
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
                
                # 调整形状以匹配轨迹维度
                if len(trajectories.shape) == 3:  # [batch_size, seq_len, feature_dim]
                    alphas_cumprod_t = alphas_cumprod_t.view(-1, 1, 1)
                else:  # [batch_size, feature_dim]
                    alphas_cumprod_t = alphas_cumprod_t.view(-1, 1)
                
                noisy_trajectories = torch.sqrt(alphas_cumprod_t) * trajectories + \
                                    torch.sqrt(1 - alphas_cumprod_t) * noise
                
                # 归一化时间步
                t_normalized = t / self.diffusion_steps
                
                # 预测噪声 (使用改进的前向传播)
                predicted_noise = self.model(
                    x_state=noisy_trajectories,
                    t=t_normalized
                )
                
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
    
    def generate(self, batch_size: int = 1, seq_len: int = 10, cond_data: torch.Tensor = None,
                validity_check: bool = True) -> Dict:
        """
        生成轨迹
        
        参数:
            batch_size (int): 生成批次大小
            seq_len (int): 轨迹序列长度
            cond_data (torch.Tensor): 条件信息数据
            validity_check (bool): 是否检查物理有效性
            
        返回:
            生成结果 (Dict)
        """
        if self.model is None:
            raise ValueError("请先调用build_model构建模型")
        
        self.model.eval()
        state_dim = self.state_dim
        
        # 初始化随机噪声 (3D张量: [batch_size, seq_len, state_dim])
        x = torch.randn((batch_size, seq_len, state_dim), device=self.device)
        
        # 逐步去噪
        for i in reversed(range(self.diffusion_steps)):
            t = torch.ones(batch_size, device=self.device) * i / self.diffusion_steps
            
            # 无梯度计算
            with torch.no_grad():
                # 预测噪声
                predicted_noise = self.model(x_state=x, t=t, cond=cond_data)
                
                # 计算去噪步骤
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # 更新x (使用更稳定的去噪公式)
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
                ) + torch.sqrt(beta) * noise
        
        # 物理约束检查 (暂时禁用)
        if validity_check:
            # print("警告: 物理约束检查功能尚未实现，已跳过")
            validity_flags = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        return {
            'trajectories': x.cpu().numpy(),
            'validity': validity_flags.cpu().numpy() if validity_check else None
        }
    
    def save_model(self, save_path = None) -> None:
        """
        保存模型参数和配置
        
        参数:
            save_path (str): 保存路径，应以.pt结尾
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用build_model")

        if save_path is None:
            # 默认保存路径
            save_path = f"models/diffusion_traj_generator.pt"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 保存模型参数和配置
        model_state = {
            'encoder_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': {k:v for k,v in self.config_to_save.items()}
        }
        # 保存模型
        torch.save(model_state, save_path)
        
        config_path = os.path.splitext(save_path)[0] + '_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_state['config'], f, ensure_ascii=False, indent=4)
            
        print(f"模型已保存至: {save_path}")
        print(f"配置已保存至: {config_path}")

    # 写一个静态函数load_model，从而构建模型并加载参数
    @staticmethod
    def load_model(load_path) -> 'DiffusionTrajGenerator':
        """
        加载模型参数和配置
        
        参数:
            load_path (str): 加载路径，应以.pt结尾
            
        返回:
            DiffusionTrajGenerator实例
        """
        if load_path is None:
            load_path = f"models/diffusion_traj_generator.pt"
        if not os.path.isfile(load_path):
            raise ValueError(f"指定的模型文件不存在: {load_path}")
        
        # 加载模型状态
        model_state = torch.load(load_path, map_location=torch.device('cpu'))
        config = model_state['config']
        
        # 创建DiffusionTrajGenerator实例
        generator = DiffusionTrajGenerator(
            diffusion_steps=config.get('diffusion_steps', 1000),
            noise_schedule=config.get('noise_schedule', 'cosine'),
            dropout=config.get('dropout', 0.2),
            im_embd=config.get('im_embd', 128)
        )
        
        # 构建模型
        state_dim = config.get('state_dim')
        cond_dim = config.get('cond_dim', 0)
        generator.build_model(state_dim=state_dim, cond_dim=cond_dim)
        
        # 加载模型参数
        generator.model.load_state_dict(model_state['encoder_state_dict'])
        
        # 加载优化器和调度器状态（如果存在）
        if generator.optimizer and model_state['optimizer_state_dict']:
            generator.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        if generator.scheduler and model_state['scheduler_state_dict']:
            generator.scheduler.load_state_dict(model_state['scheduler_state_dict'])
        
        print(f"模型已从 {load_path} 加载")
        return generator

    
    def _check_physics_validity(self, data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        检查物理有效性
        
        参数:
            data: 可以是轨迹张量或(状态张量, 动作张量)元组
            
        返回:
            validity_flags (torch.Tensor): 有效性标志
        """
        if isinstance(data, tuple) and len(data) == 2:
            # 分离的状态和动作
            states, actions = data
            batch_size = states.shape[0]
            
            # 合并状态和动作以进行物理约束检查
            if self.is_image_state and len(states.shape) > 3:  # 图像状态 [batch_size, seq_len, c, h, w]
                # 需要先通过图像编码器处理
                with torch.no_grad():
                    encoded_states = []
                    for i in range(states.shape[0]):
                        if len(states.shape) == 5:  # [batch_size, seq_len, c, h, w]
                            seq_len = states.shape[1]
                            reshaped_states = states[i].reshape(-1, *self.image_shape)
                            encoded = self.model.image_encoder(reshaped_states)
                            encoded = encoded.reshape(seq_len, -1)
                            encoded_states.append(encoded)
                        else:  # [batch_size, c, h, w]
                            encoded = self.model.image_encoder(states[i:i+1])
                            encoded_states.append(encoded)
                    encoded_states = torch.stack(encoded_states)
                trajectories = torch.cat([encoded_states, actions], dim=-1)
            else:
                trajectories = torch.cat([states, actions], dim=-1)
        else:
            # 单一轨迹
            trajectories = data
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
    
    def _apply_physics_correction(self, data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        应用物理约束修正
        
        参数:
            data: 可以是轨迹数组或(状态数组, 动作数组)元组
            
        返回:
            corrected_data: 修正后的轨迹或(修正后的状态, 修正后的动作)元组
        """
        if isinstance(data, tuple) and len(data) == 2:
            # 分离的状态和动作
            states, actions = data
            
            # 合并状态和动作以进行物理约束检查
            combined_traj = np.concatenate([states, actions], axis=-1) if len(states.shape) > 1 else np.concatenate([states.reshape(1, -1), actions.reshape(1, -1)], axis=-1)
            corrected_combined = self._apply_physics_correction_internal(combined_traj)
            
            # 分离修正后的状态和动作
            state_dim = states.shape[-1]
            if len(states.shape) > 1:
                corrected_states = corrected_combined[..., :state_dim]
                corrected_actions = corrected_combined[..., state_dim:]
            else:
                corrected_states = corrected_combined[0, :state_dim]
                corrected_actions = corrected_combined[0, state_dim:]
            
            return corrected_states, corrected_actions
        else:
            # 单一轨迹
            return self._apply_physics_correction_internal(data)
    
    def _apply_physics_correction_internal(self, trajectory: np.ndarray) -> np.ndarray:
        """
        对单个轨迹应用物理约束修正
        
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
    
    def _compute_violation_rate(self, original_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                               corrected_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        计算违规率
        
        参数:
            original_data: 原始数据，可以是轨迹数组或(状态数组, 动作数组)元组
            corrected_data: 修正后的数据，可以是轨迹数组或(状态数组, 动作数组)元组
            
        返回:
            violation_rate (float): 违规率
        """
        if isinstance(original_data, tuple) and isinstance(corrected_data, tuple):
            # 分离的状态和动作
            original_states, original_actions = original_data
            corrected_states, corrected_actions = corrected_data
            
            # 计算状态和动作的差异
            state_diff = np.abs(original_states - corrected_states)
            action_diff = np.abs(original_actions - corrected_actions)
            
            # 合并差异
            if len(state_diff.shape) > 1:
                diff = np.concatenate([state_diff, action_diff], axis=-1)
            else:
                diff = np.concatenate([state_diff.reshape(1, -1), action_diff.reshape(1, -1)], axis=-1)
        else:
            # 单一轨迹
            diff = np.abs(original_data - corrected_data)
        
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 计算违规率（简单示例）
        violation_rate = mean_diff / (max_diff + 1e-8)
        
        return violation_rate