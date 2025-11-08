import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import json
from afruits.utils.DataLoader import DataLoaderUtil

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
                 dropout: float = 0.2,
                 im_embd: int = 128,
                 discrete_action: bool = False):
        """
        初始化VAE轨迹生成器
        
        参数:
            latent_dim (int): 潜在空间维度，取值范围16-256
            seq_length (int): 输入序列长度，取值范围60-300
            kl_weight (float): KL散度损失权重，取值范围0.0001-0.1
            recon_loss_type (str): 重构损失类型，可选["mse", "mae"]
            physics_constraints (dict): 物理规则约束字典
            dropout (float): Dropout比率，用于CNN图像编码器
            im_embd (int): 图像嵌入维度，用于CNN图像编码器输出
            discrete_action (bool): 是否使用离散动作，如果为True，则使用交叉熵损失
                                  并在生成时进行离散化处理
        """
        # 参数有效性检查
        assert 16 <= latent_dim <= 256, "latent_dim必须在16-256范围内"
        assert 60 <= seq_length <= 300, "seq_length必须在60-300范围内"
        assert 0.0001 <= kl_weight <= 0.1, "kl_weight必须在0.0001-0.1范围内"
        assert recon_loss_type in ["mse", "mae"], "recon_loss_type必须是'mse'或'mae'"
        assert 0.0 <= dropout <= 0.5, "dropout必须在0.0-0.5范围内"
        
        # 初始化参数
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.kl_weight = kl_weight
        self.recon_loss_type = recon_loss_type
        self.physics_constraints = {}
        self.dropout = dropout
        self.im_embd = im_embd
        self.discrete_action = discrete_action
        
        # 初始化网络模型
        self.encoder = None
        self.decoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练相关参数
        self.optimizer = None
        self.scheduler = None

        self.dataloader_util = DataLoaderUtil()
        
    def load_dataset(self, data: str, batch_size: int = 32) -> Dict:
        """
        数据加载
        
        参数:
            data_path (str): 预处理后的数据
            batch_size (int): 批处理大小
            
        返回值:
            数据加载器 (DataLoader)
        """
        data = self.dataloader_util.load_expert_data(data, batch_size)
        return data
    
    def build_model(self, state_dim, action_dim) -> Tuple[nn.Module, nn.Module]:
        """
        模型构建
        
        参数:
            input_dim (int or Dict): 输入特征维度，可以是整数或包含状态和动作维度的字典
            
        返回值:
            模型组 (tuple): (encoder, decoder)
        """
        # 处理输入维度
        has_separate_action = False
        state_dim = None
        action_dim = None
        is_image_state = False
        image_shape = None
        
        # 新格式：分别包含状态和动作维度
        if isinstance(state_dim, tuple) and len(state_dim) == 1:
            state_dim = state_dim[0]
        has_separate_action = True
        state_dim = state_dim
        action_dim = action_dim
        total_dim = state_dim + action_dim
        
        # 检查状态是否为图像（四维张量）
        if isinstance(state_dim, tuple) and len(state_dim) == 3:
            is_image_state = True
            image_shape = state_dim  # (c, h, w)
            # 对于图像状态，我们将使用CNN编码器，因此这里的state_dim将是CNN的输出维度
            state_dim = self.im_embd
            total_dim = state_dim + action_dim
        
        # 构建编码器
        class Encoder(nn.Module):
            def __init__(self, input_dim, seq_len, latent_dim, has_separate_action=False, state_dim=None, action_dim=None,
                         is_image_state=False, image_shape=None, dropout=0.2, im_embd=128):
                super().__init__()
                self.input_dim = input_dim
                self.seq_len = seq_len
                self.latent_dim = latent_dim
                self.has_separate_action = has_separate_action
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.is_image_state = is_image_state
                self.image_shape = image_shape
                self.dropout = dropout
                self.im_embd = im_embd
                
                # 如果状态是图像，添加CNN图像编码器
                if is_image_state and image_shape is not None:
                    c, h, w = image_shape
                    self.image_encoder = nn.Sequential(
                        nn.Conv2d(c, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Flatten(start_dim=1),
                        nn.Linear(int(16 * h * w), self.im_embd),
                        nn.ReLU(),
                    )
                
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
                
            def forward(self, x_state, x_action=None):
                # 处理输入
                if self.is_image_state:
                    # 如果状态是图像，首先通过CNN编码器处理每个时间步的图像
                    batch_size, seq_len = x_state.shape[0], x_state.shape[1]
                    # 重塑为 [batch_size * seq_len, c, h, w]
                    x_state_reshaped = x_state.view(-1, *self.image_shape)
                    # 通过CNN编码器
                    x_state_encoded = self.image_encoder(x_state_reshaped)
                    # 重塑回 [batch_size, seq_len, im_embd]
                    x_state = x_state_encoded.view(batch_size, seq_len, -1)
                
                if self.has_separate_action and x_action is not None:
                    # 合并状态和动作
                    # x_state shape: [batch_size, seq_len, state_dim] 或 [batch_size, seq_len, im_embd]
                    # x_action shape: [batch_size, seq_len, action_dim]
                    x = torch.cat([x_state, x_action], dim=2)
                else:
                    # 单一输入
                    x = x_state
                
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
            def __init__(self, latent_dim, seq_len, output_dim, has_separate_action=False, state_dim=None, action_dim=None, is_image_state=False, image_shape=None, im_embd=128):
                super().__init__()
                self.latent_dim = latent_dim
                self.seq_len = seq_len
                self.output_dim = output_dim
                self.has_separate_action = has_separate_action
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.is_image_state = is_image_state
                self.image_shape = image_shape
                self.im_embd = im_embd
                
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
                if has_separate_action:
                    # 分别输出状态和动作
                    if is_image_state and image_shape is not None:
                        # 对于图像状态，先输出到im_embd维度
                        self.fc_state = nn.Linear(128, im_embd)
                        
                        # 添加图像解码器
                        c, h, w = image_shape
                        self.image_decoder = nn.Sequential(
                            nn.Linear(im_embd, 16 * h * w),
                            nn.ReLU(),
                            nn.Unflatten(1, (16, h, w)),
                            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.ConvTranspose2d(16, c, kernel_size=3, padding=1),
                        )
                    else:
                        # 普通状态
                        self.fc_state = nn.Linear(128, state_dim)
                    
                    # 对于离散动作，输出logits
                    self.fc_action = nn.Linear(128, action_dim)  # 输出每个动作类别的logits
                else:
                    # 单一输出
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
                if self.has_separate_action:
                    # 分别输出状态和动作
                    if self.is_image_state and self.image_shape is not None:
                        # 对于图像状态，先输出到im_embd维度
                        state_embd = self.fc_state(lstm_out)  # [batch_size, seq_len, im_embd]
                        
                        # 重塑为 [batch_size * seq_len, im_embd]
                        state_embd_flat = state_embd.reshape(-1, self.im_embd)
                        
                        # 通过图像解码器
                        image_flat = self.image_decoder(state_embd_flat)  # [batch_size * seq_len, c, h, w]
                        
                        # 重塑回 [batch_size, seq_len, c, h, w]
                        c, h, w = self.image_shape
                        state_output = image_flat.reshape(batch_size, self.seq_len, c, h, w)
                    else:
                        # 普通状态
                        state_output = self.fc_state(lstm_out)
                    
                    action_output = self.fc_action(lstm_out)
                    return state_output, action_output
                else:
                    # 单一输出
                    output = self.fc_out(lstm_out)
                    return output
        
        # 创建编码器和解码器
        encoder = Encoder(
            total_dim,
            self.seq_length,
            self.latent_dim,
            has_separate_action,
            state_dim,
            action_dim,
            is_image_state,
            image_shape,
            self.dropout,
            self.im_embd
        ).to(self.device)
        
        decoder = Decoder(
            self.latent_dim,
            self.seq_length,
            total_dim if not has_separate_action else None,
            has_separate_action,
            state_dim,
            action_dim,
            is_image_state,
            image_shape,
            self.im_embd
        ).to(self.device)
        
        # 设置优化器
        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.encoder = encoder
        self.decoder = decoder
        
        # 保存模型配置
        self.has_separate_action = has_separate_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_image_state = is_image_state
        self.image_shape = image_shape
        
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
                if len(batch) == 2 and self.has_separate_action:
                    # 分别获取状态和动作数据
                    states = batch[0].to(self.device)
                    actions = batch[1].to(self.device)
                    batch_size = states.shape[0]
                    
                    # 检查状态是否为图像格式
                    if self.is_image_state and len(states.shape) == 5:  # [batch_size, seq_len, c, h, w]
                        # 前向传播
                        mu, logvar = self.encoder(states, actions)
                    else:
                        # 前向传播
                        mu, logvar = self.encoder(states, actions)
                    z = self.reparameterize(mu, logvar)
                    reconstructed_states, reconstructed_actions = self.decoder(z)
                    
                    # 计算重构损失
                    # 状态仍使用MSE或MAE
                    if self.recon_loss_type == "mse":
                        state_recon_loss = F.mse_loss(reconstructed_states, states)
                    else:  # mae
                        state_recon_loss = F.l1_loss(reconstructed_states, states)
                    
                    # 根据动作类型选择损失函数
                    if self.discrete_action:
                        # 对于离散动作，使用交叉熵损失
                        # 重塑为 [batch*seq, action_dim] 和 [batch*seq]
                        batch_size, seq_len = actions.shape[0], actions.shape[1]
                        action_recon_loss = F.cross_entropy(
                            reconstructed_actions.reshape(-1, self.action_dim),  # [batch*seq, action_dim]
                            actions.reshape(-1).long()  # [batch*seq]
                        )
                    else:
                        # 对于连续动作，使用MSE或MAE
                        if self.recon_loss_type == "mse":
                            action_recon_loss = F.mse_loss(reconstructed_actions, actions)
                        else:  # mae
                            action_recon_loss = F.l1_loss(reconstructed_actions, actions)
                    
                    # 总重构损失（可以根据需要调整状态和动作的权重）
                    recon_loss = state_recon_loss + action_recon_loss
                else:
                    # 旧格式：单一轨迹数据
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
    
    def generate(self, num_samples: int = 1, cond_vector: torch.Tensor = None, temperature: float = 1.0) -> Dict:
        """
        轨迹生成
        
        参数:
            num_samples (int): 生成数量
            cond_vector (torch.Tensor): 条件向量
            temperature (float): 温度参数，控制采样随机性。较低的值（接近0）使生成更确定性，
                               较高的值增加随机性。默认为1.0
            
        返回值:
            生成轨迹 (dict):
                state: 状态序列（如果是图像状态，则为[batch_size, seq_len, c, h, w]格式）
                action: 离散动作序列（整数索引）
                action_probs: 动作概率分布
                action_logits: 原始logits输出
                trajectories: 状态-动作序列（仅在旧格式模式下返回）
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
            if self.has_separate_action:
                # 分别生成状态和动作
                generated_states, generated_actions = self.decoder(z)
                
                # 对于图像状态，generated_states已经是[batch_size, seq_len, c, h, w]格式
                # 由于decoder.forward中的处理，不需要额外的转换
                
                # 根据动作类型选择处理方式
                if self.discrete_action:
                    # 对于离散动作，需要进行处理
                    # 使用温度参数调整logits
                    scaled_logits = generated_actions / temperature
                    
                    # 计算softmax概率
                    action_probs = F.softmax(scaled_logits, dim=-1)
                    
                    # 可以使用两种方式获取离散动作：
                    # 1. argmax: 取概率最大的类别（确定性）
                    if temperature <= 0.01:  # 接近于0的温度，使用argmax
                        discrete_actions = torch.argmax(scaled_logits, dim=-1)
                    # 2. 按概率采样（随机性）
                    else:
                        # 重塑为 [batch*seq, action_dim]，采样后再重塑回 [batch, seq]
                        probs_flat = action_probs.reshape(-1, self.action_dim)
                        sampled_flat = torch.multinomial(probs_flat, 1).squeeze(-1)
                        discrete_actions = sampled_flat.reshape(num_samples, self.seq_length)
                    
                    return {
                        'state': generated_states.cpu().numpy(),
                        'action': discrete_actions.cpu().numpy(),
                        'action_probs': action_probs.cpu().numpy(),  # 返回概率分布
                        'action_logits': scaled_logits.cpu().numpy(),  # 返回原始logits
                        'latent_codes': z.cpu().numpy()
                    }
                else:
                    # 对于连续动作，直接返回
                    return {
                        'state': generated_states.cpu().numpy(),
                        'action': generated_actions.cpu().numpy(),
                        'latent_codes': z.cpu().numpy()
                    }
            else:
                # 旧格式：单一轨迹
                generated_trajectories = self.decoder(z)
                
                return {
                    'trajectories': generated_trajectories.cpu().numpy(),
                    'latent_codes': z.cpu().numpy()
                }
    
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
                'physics_constraints': self.physics_constraints,
                'dropout': self.dropout,
                'im_embd': self.im_embd,
                'is_image_state': self.is_image_state,
                'image_shape': self.image_shape
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
        self.dropout = config.get('dropout', self.dropout)
        self.im_embd = config.get('im_embd', self.im_embd)
        self.is_image_state = config.get('is_image_state', False)
        self.image_shape = config.get('image_shape', None)
        
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
        if self.is_image_state:
            print(f"图像状态配置: image_shape={self.image_shape}, dropout={self.dropout}, im_embd={self.im_embd}")
        
        return True