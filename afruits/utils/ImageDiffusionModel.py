import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time

class ImageDiffusionModel:
    """
    图像扩散模型
    
    功能描述：专为图像输入设计的扩散模型，直接在图像空间进行扩散过程
    
    核心特性：
    ◆ 直接图像扩散：无需编码器，直接在图像空间进行扩散
    ◆ U-Net架构：采用U-Net结构处理图像
    ◆ 时间条件：支持时间步嵌入
    ◆ 多尺度特征：多层次特征提取与融合
    ◆ 注意力机制：整合空间注意力提升生成质量
    """
    
    def __init__(self,
                 image_size: Tuple[int, int, int] = (3, 64, 64),  # (C, H, W)
                 diffusion_steps: int = 1000,
                 noise_schedule: str = "cosine",
                 base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4, 8],
                 attention_resolutions: List[int] = [8, 16],
                 dropout: float = 0.1,
                 use_checkpoint: bool = False):
        """
        初始化图像扩散模型
        
        参数:
            image_size (Tuple[int, int, int]): 图像尺寸 (通道数, 高度, 宽度)
            diffusion_steps (int): 扩散步数，取值范围10-2000
            noise_schedule (str): 噪声调度类型，可选["linear", "cosine"]
            base_channels (int): 基础通道数
            channel_multipliers (List[int]): 通道数乘数列表
            attention_resolutions (List[int]): 使用注意力机制的分辨率列表
            dropout (float): Dropout比率
            use_checkpoint (bool): 是否使用梯度检查点以节省内存
        """
        # 参数有效性检查
        assert 10 <= diffusion_steps <= 2000, "diffusion_steps必须在10-2000范围内"
        assert noise_schedule in ["linear", "cosine"], "noise_schedule必须是'linear'或'cosine'"
        assert 0.0 <= dropout <= 0.5, "dropout必须在0.0-0.5范围内"
        
        # 初始化参数
        self.image_size = image_size
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        
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
        data = np.load(data_path, allow_pickle=True)
        
        # 提取图像数据
        if 'images' in data:
            images = data['images']
        else:
            raise ValueError("数据文件必须包含'images'键")
        
        # 检查图像形状
        if len(images.shape) != 4:  # (N, C, H, W)
            raise ValueError(f"图像数据形状应为(N, C, H, W)，但得到{images.shape}")
        
        # 转换为PyTorch张量
        images_tensor = torch.FloatTensor(images)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(images_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"数据加载完成: 样本数={len(dataset)}, 图像形状={images.shape}")
        
        return {
            'dataloader': dataloader,
            'data_shape': images.shape
        }
    
    def build_model(self) -> nn.Module:
        """
        构建U-Net模型
        
        返回:
            model (nn.Module): U-Net模型
        """
        # 定义U-Net模型
        class TimeEmbedding(nn.Module):
            def __init__(self, time_dim):
                super().__init__()
                self.time_dim = time_dim
                self.time_embed = nn.Sequential(
                    nn.Linear(1, time_dim),
                    nn.SiLU(),
                    nn.Linear(time_dim, time_dim),
                )
            
            def forward(self, t):
                # t: [batch_size]
                t = t.unsqueeze(-1)  # [batch_size, 1]
                return self.time_embed(t)  # [batch_size, time_dim]
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, time_dim, dropout, use_attention=False, resolution=None):
                super().__init__()
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, out_channels)
                )
                
                self.block1 = nn.Sequential(
                    nn.GroupNorm(32, in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, out_channels, 3, padding=1)
                )
                
                self.block2 = nn.Sequential(
                    nn.GroupNorm(32, out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)
                )
                
                if in_channels != out_channels:
                    self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
                else:
                    self.shortcut = nn.Identity()
                
                self.use_attention = use_attention
                if use_attention:
                    self.attention = SelfAttention(out_channels, resolution)
            
            def forward(self, x, t):
                # x: [batch_size, in_channels, h, w]
                # t: [batch_size, time_dim]
                h = self.block1(x)
                time_emb = self.time_mlp(t)[:, :, None, None]  # [batch_size, out_channels, 1, 1]
                h = h + time_emb
                h = self.block2(h)
                
                h = h + self.shortcut(x)
                
                if self.use_attention:
                    h = self.attention(h)
                
                return h
        
        class SelfAttention(nn.Module):
            def __init__(self, channels, resolution):
                super().__init__()
                self.channels = channels
                self.resolution = resolution
                
                self.norm = nn.GroupNorm(32, channels)
                self.qkv = nn.Conv2d(channels, channels * 3, 1)
                self.proj = nn.Conv2d(channels, channels, 1)
                
                # 计算注意力图的尺寸
                self.attention_size = resolution * resolution
                
            def forward(self, x):
                # x: [batch_size, channels, h, w]
                batch_size, channels, h, w = x.shape
                
                # 归一化
                x_norm = self.norm(x)
                
                # 计算q, k, v
                qkv = self.qkv(x_norm)  # [batch_size, channels*3, h, w]
                q, k, v = torch.chunk(qkv, 3, dim=1)  # 每个 [batch_size, channels, h, w]
                
                # 重塑以计算注意力
                q = q.reshape(batch_size, channels, -1)  # [batch_size, channels, h*w]
                k = k.reshape(batch_size, channels, -1)  # [batch_size, channels, h*w]
                v = v.reshape(batch_size, channels, -1)  # [batch_size, channels, h*w]
                
                # 计算注意力分数
                attn = torch.bmm(q.transpose(1, 2), k)  # [batch_size, h*w, h*w]
                attn = attn * (channels ** -0.5)  # 缩放
                attn = F.softmax(attn, dim=2)
                
                # 应用注意力
                out = torch.bmm(v, attn.transpose(1, 2))  # [batch_size, channels, h*w]
                out = out.reshape(batch_size, channels, h, w)  # [batch_size, channels, h, w]
                
                # 投影回原始维度
                out = self.proj(out)
                
                return out + x
        
        class DownBlock(nn.Module):
            def __init__(self, in_channels, out_channels, time_dim, dropout, use_attention=False, resolution=None):
                super().__init__()
                self.res = ResidualBlock(in_channels, out_channels, time_dim, dropout, use_attention, resolution)
                self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            
            def forward(self, x, t):
                x = self.res(x, t)
                return self.downsample(x)
        
        class UpBlock(nn.Module):
            def __init__(self, in_channels, out_channels, time_dim, dropout, use_attention=False, resolution=None):
                super().__init__()
                self.res = ResidualBlock(in_channels + out_channels, out_channels, time_dim, dropout, use_attention, resolution)
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)
                )
            
            def forward(self, x, skip_x, t):
                x = self.upsample(x)
                x = torch.cat([x, skip_x], dim=1)
                return self.res(x, t)
        
        class UNet(nn.Module):
            def __init__(self, image_size, base_channels, channel_multipliers, attention_resolutions, dropout, use_checkpoint=False):
                super().__init__()
                self.image_size = image_size
                self.use_checkpoint = use_checkpoint
                
                # 图像通道数
                in_channels = image_size[0]
                
                # 时间嵌入维度
                time_dim = base_channels * 4
                self.time_embed = TimeEmbedding(time_dim)
                
                # 初始卷积层
                self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
                
                # 计算每个分辨率的通道数
                channels = [base_channels]
                for mult in channel_multipliers:
                    channels.append(base_channels * mult)
                
                # 下采样块
                self.downs = nn.ModuleList()
                curr_res = image_size[1]  # 当前分辨率
                
                for i in range(len(channel_multipliers)):
                    use_attn = curr_res in attention_resolutions
                    self.downs.append(
                        DownBlock(
                            channels[i], 
                            channels[i+1], 
                            time_dim, 
                            dropout, 
                            use_attn,
                            curr_res
                        )
                    )
                    curr_res = curr_res // 2
                
                # 中间块
                self.mid = ResidualBlock(
                    channels[-1], 
                    channels[-1], 
                    time_dim, 
                    dropout, 
                    True,
                    curr_res
                )
                
                # 上采样块
                self.ups = nn.ModuleList()
                
                for i in range(len(channel_multipliers)-1, -1, -1):
                    use_attn = curr_res in attention_resolutions
                    self.ups.append(
                        UpBlock(
                            channels[i+1], 
                            channels[i], 
                            time_dim, 
                            dropout, 
                            use_attn,
                            curr_res
                        )
                    )
                    curr_res = curr_res * 2
                
                # 输出层
                self.final_conv = nn.Sequential(
                    nn.GroupNorm(32, base_channels),
                    nn.SiLU(),
                    nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
                )
            
            def forward(self, x, t):
                # x: [batch_size, in_channels, h, w]
                # t: [batch_size]
                
                # 时间嵌入
                t_emb = self.time_embed(t)
                
                # 初始卷积
                h = self.init_conv(x)
                
                # 保存下采样特征用于跳跃连接
                skips = [h]
                
                # 下采样路径
                for down in self.downs:
                    h = down(h, t_emb)
                    skips.append(h)
                
                # 中间块
                h = self.mid(h, t_emb)
                
                # 上采样路径
                for up in self.ups:
                    h = up(h, skips.pop(), t_emb)
                
                # 输出层
                return self.final_conv(h)
        
        # 创建模型
        model = UNet(
            self.image_size,
            self.base_channels,
            self.channel_multipliers,
            self.attention_resolutions,
            self.dropout,
            self.use_checkpoint
        )
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
            raise ValueError("请先调用build_model构建模型")
        
        self.model.train()
        losses = []
        
        # 训练循环
        for epoch in range(epochs):
            epoch_losses = []
            start_time = time.time()
            
            for batch in dataloader:
                # 获取图像数据
                images = batch[0].to(self.device)
                batch_size = images.shape[0]
                
                # 随机选择时间步
                t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)
                
                # 添加噪声
                noise = torch.randn_like(images)
                alphas_cumprod_t = torch.tensor(self.alphas_cumprod, device=self.device)[t]
                
                # 调整形状以匹配图像维度
                alphas_cumprod_t = alphas_cumprod_t.view(-1, 1, 1, 1)
                
                noisy_images = torch.sqrt(alphas_cumprod_t) * images + \
                              torch.sqrt(1 - alphas_cumprod_t) * noise
                
                # 预测噪声
                predicted_noise = self.model(noisy_images, t / self.diffusion_steps)
                
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
    
    def generate(self, batch_size: int = 1, cond_data: torch.Tensor = None) -> Dict:
        """
        生成图像
        
        参数:
            batch_size (int): 生成批次大小
            cond_data (torch.Tensor): 条件信息数据
            
        返回:
            生成结果 (Dict)
                - images: 生成的图像
        """
        if self.model is None:
            raise ValueError("请先调用build_model构建模型")
        
        self.model.eval()
        
        # 初始化随机噪声
        x = torch.randn((batch_size, *self.image_size), device=self.device)
        
        # 逐步去噪
        for i in reversed(range(self.diffusion_steps)):
            t = torch.ones(batch_size, device=self.device) * i / self.diffusion_steps
            
            # 无梯度计算
            with torch.no_grad():
                # 预测噪声
                predicted_noise = self.model(x, t)
                
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
        
        # 将像素值归一化到[0, 1]范围
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        
        return {
            'images': x.cpu().numpy()
        }
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型
        
        参数:
            save_path (str): 保存路径
        """
        if self.model is None:
            raise ValueError("请先调用build_model构建模型")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'image_size': self.image_size,
                'diffusion_steps': self.diffusion_steps,
                'noise_schedule': self.noise_schedule,
                'base_channels': self.base_channels,
                'channel_multipliers': self.channel_multipliers,
                'attention_resolutions': self.attention_resolutions,
                'dropout': self.dropout,
                'use_checkpoint': self.use_checkpoint
            }
        }, save_path)
        
        print(f"模型已保存至: {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        加载模型
        
        参数:
            load_path (str): 加载路径
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件 {load_path} 不存在")
        
        # 加载模型
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 更新配置
        config = checkpoint['config']
        self.image_size = config['image_size']
        self.diffusion_steps = config['diffusion_steps']
        self.noise_schedule = config['noise_schedule']
        self.base_channels = config['base_channels']
        self.channel_multipliers = config['channel_multipliers']
        self.attention_resolutions = config['attention_resolutions']
        self.dropout = config['dropout']
        self.use_checkpoint = config['use_checkpoint']
        
        # 重新初始化扩散参数
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # 构建模型
        self.build_model()
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"模型已从 {load_path} 加载")