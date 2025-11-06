import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import json

class VAETrajGenerator:
    """
    VAE轨迹生成器
    
    功能定义：基于概率潜在空间实现轨迹生成与重构
    
    核心特性：
    ◆ 概率编码：支持潜在空间概率分布建模
    ◆ 双损失机制：重构损失+KL散度正则化
    ◆ 物理约束：集成飞行动力学模型
    ◆ 多模式生成：支持条件/无条件轨迹生成
    """
    
    def __init__(self, 
                 latent_dim: int = 64, 
                 seq_length: int = 120,
                 kl_weight: float = 0.001,
                 recon_loss_type: str = "mse",
                 physics_constraints: dict = None):
        """
        初始化VAE轨迹生成器
        
        参数:
            latent_dim (int): 潜在空间维度，取值范围16-256
            seq_length (int): 输入序列长度，取值范围60-300
            kl_weight (float): KL散度损失权重，取值范围0.0001-0.1
            recon_loss_type (str): 重构损失类型，可选["mse", "mae"]
            physics_constraints (dict): 物理规则约束字典
        """
        # 参数有效性检查
        assert 16 <= latent_dim <= 256, "latent_dim必须在16-256范围内"
        assert 60 <= seq_length <= 300, "seq_length必须在60-300范围内"
        assert 0.0001 <= kl_weight <= 0.1, "kl_weight必须在0.0001-0.1范围内"
        assert recon_loss_type in ["mse", "mae"], "recon_loss_type必须是'mse'或'mae'"
        
        # 初始化参数
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.kl_weight = kl_weight
        self.recon_loss_type = recon_loss_type
        self.physics_constraints = physics_constraints if physics_constraints else {}
        
        # 初始化网络模型
        self.encoder = None
        self.decoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练相关参数
        self.optimizer = None
        self.scheduler = None
        
    def load_dataset(self, data_path: str, batch_size: int = 32) -> Dict:
        """
        数据加载
        
        参数:
            data_path (str): 预处理后的数据文件路径
            batch_size (int): 批处理大小
            
        返回值:
            数据加载器 (DataLoader)
        """
        # 加载数据
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件 {data_path} 不存在")
        
        # 加载数据（假设为numpy格式）
        data = np.load(data_path)
        
        # 提取轨迹数据
        trajectories = data['trajectories'] if 'trajectories' in data else data
        
        # 检查序列长度
        if trajectories.shape[1] < self.seq_length:
            raise ValueError(f"轨迹序列长度 {trajectories.shape[1]} 小于设定的序列长度 {self.seq_length}")
        
        # 如果轨迹长度大于设定长度，截取或随机采样
        if trajectories.shape[1] > self.seq_length:
            # 随机截取指定长度的片段
            start_indices = np.random.randint(0, trajectories.shape[1] - self.seq_length, size=trajectories.shape[0])
            sampled_trajectories = np.array([
                trajectories[i, start_idx:start_idx+self.seq_length] 
                for i, start_idx in enumerate(start_indices)
            ])
            trajectories = sampled_trajectories
        
        # 转换为PyTorch张量
        trajectories_tensor = torch.FloatTensor(trajectories)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(trajectories_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"数据加载完成: 样本数={len(dataset)}, 输入形状={trajectories.shape}")
        
        return {
            'dataloader': dataloader,
            'data_shape': trajectories.shape,
            'feature_dim': trajectories.shape[2] if len(trajectories.shape) > 2 else 1
        }
    
    def build_model(self, input_dim: int) -> Tuple[nn.Module, nn.Module]:
        """
        模型构建
        
        参数:
            input_dim (int): 输入特征维度
            
        返回值:
            模型组 (tuple): (encoder, decoder)
        """
        # 构建编码器
        class Encoder(nn.Module):
            def __init__(self, input_dim, seq_len, latent_dim):
                super().__init__()
                self.input_dim = input_dim
                self.seq_len = seq_len
                self.latent_dim = latent_dim
                
                # LSTM编码器
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True
                )
                
                # 均值和方差输出层
                self.fc_mu = nn.Linear(256, latent_dim)  # 256 = 128*2 (bidirectional)
                self.fc_logvar = nn.Linear(256, latent_dim)
                
                # 均值/方差输出层
                self.fc_out = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.SiLU()
                )
                
            def forward(self, x):
                # x shape: [batch_size, seq_len, input_dim]
                batch_size = x.size(0)
                
                # LSTM编码
                lstm_out, (h_n, c_n) = self.lstm(x)
                
                # 使用最后一个时间步的输出
                hidden = lstm_out[:, -1, :]
                
                # 特征处理
                hidden = self.fc_out(hidden)
                
                # 计算均值和对数方差
                mu = self.fc_mu(lstm_out[:, -1, :])
                logvar = self.fc_logvar(lstm_out[:, -1, :])
                
                return mu, logvar
        
        # 构建解码器
        class Decoder(nn.Module):
            def __init__(self, latent_dim, seq_len, output_dim):
                super().__init__()
                self.latent_dim = latent_dim
                self.seq_len = seq_len
                self.output_dim = output_dim
                
                # 潜在向量到初始隐藏状态的映射
                self.fc_hidden = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.LayerNorm(128),
                    nn.SiLU(),
                    nn.Linear(128, 256)  # 256 = 128*2 (for h_0 and c_0)
                )
                
                # LSTM解码器
                self.lstm = nn.LSTM(
                    input_size=latent_dim,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True
                )
                
                # 输出层
                self.fc_out = nn.Linear(128, output_dim)
                
            def forward(self, z):
                # z shape: [batch_size, latent_dim]
                batch_size = z.size(0)
                
                # 生成初始隐藏状态
                hidden = self.fc_hidden(z)
                h_0 = hidden.view(2, batch_size, 128)  # 2层LSTM
                c_0 = torch.zeros_like(h_0)
                
                # 重复潜在向量作为每个时间步的输入
                z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
                
                # LSTM解码
                lstm_out, _ = self.lstm(z_repeated, (h_0, c_0))
                
                # 生成轨迹输出
                output = self.fc_out(lstm_out)
                
                return output
        
        # 创建编码器和解码器
        encoder = Encoder(input_dim, self.seq_length, self.latent_dim).to(self.device)
        decoder = Decoder(self.latent_dim, self.seq_length, input_dim).to(self.device)
        
        # 设置优化器
        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.encoder = encoder
        self.decoder = decoder
        
        return encoder, decoder
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧
        
        参数:
            mu (torch.Tensor): 均值
            logvar (torch.Tensor): 对数方差
            
        返回值:
            z (torch.Tensor): 采样的潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def train(self, dataloader: DataLoader, epochs: int = 100) -> Dict:
        """
        模型训练
        
        参数:
            dataloader (DataLoader): 训练数据加载器
            epochs (int): 训练轮数
            
        返回值:
            训练日志 (dict):
                total_loss: 总损失曲线
                kl_divergence: KL散度变化
                recon_error: 重构误差趋势
        """
        if self.encoder is None or self.decoder is None:
            raise ValueError("请先调用build_model构建模型")
        
        self.encoder.train()
        self.decoder.train()
        
        # 训练日志
        history = {
            'total_loss': [],
            'kl_divergence': [],
            'recon_error': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_recon_loss = 0.0
            start_time = time.time()
            
            for batch in dataloader:
                # 获取轨迹数据
                trajectories = batch[0].to(self.device)
                batch_size = trajectories.shape[0]
                
                # 前向传播
                mu, logvar = self.encoder(trajectories)
                z = self.reparameterize(mu, logvar)
                reconstructed = self.decoder(z)
                
                # 计算重构损失
                if self.recon_loss_type == "mse":
                    recon_loss = F.mse_loss(reconstructed, trajectories)
                else:  # mae
                    recon_loss = F.l1_loss(reconstructed, trajectories)
                
                # 计算KL散度
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / batch_size  # 归一化
                
                # 总损失
                loss = recon_loss + self.kl_weight * kl_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 累计损失
                epoch_loss += loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_recon_loss += recon_loss.item()
            
            # 计算平均损失
            avg_loss = epoch_loss / len(dataloader)
            avg_kl_loss = epoch_kl_loss / len(dataloader)
            avg_recon_loss = epoch_recon_loss / len(dataloader)
            
            # 更新学习率
            self.scheduler.step(avg_loss)
            
            # 记录训练日志
            history['total_loss'].append(avg_loss)
            history['kl_divergence'].append(avg_kl_loss)
            history['recon_error'].append(avg_recon_loss)
            
            # 打印训练信息
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, KL: {avg_kl_loss:.6f}, Recon: {avg_recon_loss:.6f}, Time: {elapsed:.2f}s")
            
            # 早停机制：验证损失连续多轮次不下降则停止
            if epoch > 10 and history['total_loss'][-1] > history['total_loss'][-2] > history['total_loss'][-3]:
                print("早停：验证损失连续3轮未下降")
                break
        
        return history
    
    def generate(self, num_samples: int = 1, cond_vector: torch.Tensor = None) -> Dict:
        """
        轨迹生成
        
        参数:
            num_samples (int): 生成数量
            cond_vector (torch.Tensor): 条件向量
            
        返回值:
            生成轨迹 (dict):
                trajectories: 状态-动作序列
                latent_codes: 潜在空间编码
        """
        if self.encoder is None or self.decoder is None:
            raise ValueError("请先调用build_model构建模型")
        
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # 生成潜在向量
            if cond_vector is not None:
                # 条件生成
                cond_vector = cond_vector.to(self.device)
                z = cond_vector
            else:
                # 随机生成
                z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            # 解码生成轨迹
            generated_trajectories = self.decoder(z)
        
        return {
            'trajectories': generated_trajectories.cpu().numpy(),
            'latent_codes': z.cpu().numpy()
        }
    
    def validate_physics(self, trajectories: List) -> List:
        """
        物理规则校验
        
        参数:
            trajectories (List): 待校验轨迹
            
        返回值:
            合规轨迹集 (List):
                ◆ 通过空气动力学束的轨迹
        """
        if not trajectories:
            return []
        
        valid_trajectories = []
        
        # 转换为numpy数组（如果是张量）
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.cpu().numpy()
        
        # 遍历每条轨迹进行校验
        for traj in trajectories:
            # 检查物理约束
            is_valid = True
            
            # 检查速度约束（如果存在）
            if 'max_velocity' in self.physics_constraints and traj.shape[-1] >= 6:
                max_vel = self.physics_constraints['max_velocity']
                # 假设轨迹中包含速度信息在索引3-5
                velocities = traj[:, 3:6]
                vel_magnitude = np.linalg.norm(velocities, axis=1)
                
                # 检查是否有速度超过阈值
                if np.any(vel_magnitude > max_vel):
                    is_valid = False
            
            # 检查加速度约束（如果存在）
            if 'max_acceleration' in self.physics_constraints and traj.shape[-1] >= 6:
                max_acc = self.physics_constraints['max_acceleration']
                # 计算加速度（速度的差分）
                velocities = traj[:, 3:6]
                accelerations = np.diff(velocities, axis=0)
                acc_magnitude = np.linalg.norm(accelerations, axis=1)
                
                # 检查是否有加速度超过阈值
                if np.any(acc_magnitude > max_acc):
                    is_valid = False
            
            # 如果通过所有约束检查，则添加到有效轨迹集
            if is_valid:
                valid_trajectories.append(traj)
        
        return valid_trajectories
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型参数和配置
        
        参数:
            save_path (str): 保存路径，应以.pt结尾
        """
        if self.encoder is None or self.decoder is None:
            raise ValueError("模型尚未构建，请先调用build_model")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 保存模型参数和配置
        model_state = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': {
                'latent_dim': self.latent_dim,
                'seq_length': self.seq_length,
                'kl_weight': self.kl_weight,
                'recon_loss_type': self.recon_loss_type,
                'physics_constraints': self.physics_constraints
            }
        }
        
        # 保存模型
        torch.save(model_state, save_path)
        
        # 保存配置信息到JSON文件（可选，便于查看）
        config_path = os.path.splitext(save_path)[0] + '_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_state['config'], f, ensure_ascii=False, indent=4)
            
        print(f"模型已保存至: {save_path}")
        print(f"配置已保存至: {config_path}")
    
    def load_model(self, load_path: str, input_dim: int = None) -> None:
        """
        加载模型参数和配置
        
        参数:
            load_path (str): 模型文件路径，应以.pt结尾
            input_dim (int, optional): 输入特征维度，如果为None则使用保存的配置
        
        返回值:
            成功加载返回True，否则抛出异常
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件 {load_path} 不存在")
        
        # 加载模型状态
        model_state = torch.load(load_path, map_location=self.device)
        
        # 更新配置
        config = model_state.get('config', {})
        self.latent_dim = config.get('latent_dim', self.latent_dim)
        self.seq_length = config.get('seq_length', self.seq_length)
        self.kl_weight = config.get('kl_weight', self.kl_weight)
        self.recon_loss_type = config.get('recon_loss_type', self.recon_loss_type)
        self.physics_constraints = config.get('physics_constraints', self.physics_constraints)
        
        # 如果没有提供input_dim，尝试从模型结构推断
        if input_dim is None:
            # 尝试从解码器的输出层获取维度
            decoder_dict = model_state['decoder_state_dict']
            for key in decoder_dict:
                if 'fc_out.weight' in key:
                    input_dim = decoder_dict[key].size(0)
                    break
            
            if input_dim is None:
                raise ValueError("无法从模型中推断input_dim，请手动提供")
        
        # 构建模型
        self.build_model(input_dim)
        
        # 加载模型参数
        self.encoder.load_state_dict(model_state['encoder_state_dict'])
        self.decoder.load_state_dict(model_state['decoder_state_dict'])
        
        # 加载优化器状态（如果存在）
        if 'optimizer_state_dict' in model_state and model_state['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
            
        # 加载调度器状态（如果存在）
        if 'scheduler_state_dict' in model_state and model_state['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(model_state['scheduler_state_dict'])
            
        print(f"模型已从 {load_path} 加载")
        print(f"配置: latent_dim={self.latent_dim}, seq_length={self.seq_length}, kl_weight={self.kl_weight}")
        
        return True