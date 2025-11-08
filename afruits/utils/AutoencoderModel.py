import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from afruits.utils.DataLoader import DataLoaderUtil
import math

class AutoencoderModel(nn.Module):
    """
    通用的自编码器模型类，基于LSTM/Transformer架构实现编码解码功能。
    
    功能定位：
    - 时序特征提取：支持长短时序列数据
    - 双向信息流：通过编码器和解码器实现
    - 自动降维：自动保存特征模型检点
    - 可视化分析：隐空间向量可视化比较
    """
    
    def __init__(self,
                 encoder_type="lstm",  # 编码器类型: "str"或"transformer"
                 latent_dim=32,       # 隐空间维度
                 seq_length=100,      # 序列长度
                 kl_weight=0.001,     # 正则化权重
                 dropout_rate=0.2,    # Dropout比率
                 action_dim=None):    # 动作空间维度（用于one-hot编码）
        """
        初始化AutoencoderModel
        
        参数:
            encoder_type (str): 编码器类型，可选"lstm"或"transformer"
            latent_dim (int): 隐空间维度
            seq_length (int): 序列长度
            input_dim (int): 输入特征维度
            kl_weight (float): 正则化权重
            dropout_rate (float): Dropout比率
        """
        super(AutoencoderModel, self).__init__()
        
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.kl_weight = kl_weight
        self.dropout_rate = dropout_rate
        self.action_dim = action_dim
        self.device = device
        
        # 根据编码器类型初始化不同的编码器

        # 图像输入检测和处理
        self.is_image_input = False
        self.image_encoder = None
        self.image_decoder = None
        self.dataloader_util = DataLoaderUtil()
    
    def _init_position_embedding(self):
        """初始化位置编码"""
        if self.encoder_type == "transformer":
            position = torch.arange(0, self.seq_length).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.input_dim, 2) * -(np.log(10000.0) / self.input_dim)
            )
            
            pe = torch.zeros(self.seq_length, self.input_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.position_embedding.data = pe.unsqueeze(0)
    
    def load_sequences(self, raw_data, batch_size=32, mode="train_test"):
        """
        加载序列数据
        
        参数:
            raw_data (dict): 原始数据
            batch_size (int): 批处理大小
            
        返回:
            DataLoader: 数据加载器
        """
        data = self.dataloader_util.load_expert_data(raw_data, batch_size)
        data_loader = data['dataloader']
        
        return data_loader
    
    def build_model(self, input_dim, output_dim=None, action_dim=None):
        """
        构建模型
        
        参数:
            input_dim (int): 输入维度
            output_dim (int, optional): 输出维度，默认与输入维度相同
            
        返回:
            tuple: (encoder, decoder)
        """
        # 检测是否为图像输入并设置图像编码器和解码器
        self.input_dim = input_dim
        if isinstance(self.input_dim, tuple) and len(self.input_dim) == 3:
            self.is_image_input = True
            c, h, w = self.input_dim
            im_embd = self.latent_dim
            
            # 图像编码器
            self.image_encoder = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * h * w, im_embd)
            )
            
            # 图像解码器
            self.image_decoder = nn.Sequential(
                nn.Linear(im_embd, 16 * h * w),
                nn.ReLU(),
                nn.Unflatten(1, (16, h, w)),
                nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, c, kernel_size=3, padding=1),
            )
            input_dim = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        elif isinstance(self.input_dim, tuple):
            input_dim = self.input_dim[0]
        else:
            input_dim = self.input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.action_dim = action_dim
        
        if self.encoder_type == "lstm":
            # LSTM编码器
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.latent_dim,
                num_layers=2,
                batch_first=True,
                dropout=self.dropout_rate,
                bidirectional=True
            )
            
            # LSTM解码器
            self.decoder = nn.LSTM(
                input_size=self.latent_dim,
                hidden_size=input_dim,
                num_layers=2,
                batch_first=True,
                dropout=self.dropout_rate,
                bidirectional=False
            )
            
            # 输出层
            self.output_layer = nn.Linear(input_dim, input_dim)
            
            # 隐空间映射（双向LSTM输出到单向输入）
            self.hidden_map = nn.Linear(self.latent_dim * 2, self.latent_dim)
            
        elif self.encoder_type == "transformer":
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=self.latent_dim * 4,
                dropout=self.dropout_rate,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
            
            # Transformer解码器
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=self.latent_dim * 4,
                dropout=self.dropout_rate,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=2
            )
            
            # 隐空间映射
            self.hidden_map = nn.Linear(input_dim, self.latent_dim)
            
            # 隐空间逆映射
            self.hidden_unmap = nn.Linear(self.latent_dim, input_dim)
            
            # 位置编码
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.seq_length, input_dim)
            )
            self._init_position_embedding()
        
        else:
            raise ValueError(f"不支持的编码器类型")
        
        if action_dim is not None:
            self.action_embedding = nn.Embedding(action_dim, action_dim)
            self.action_linear = nn.Linear(action_dim, action_dim)
        
        return self
    
    def encode(self, x):
        """
        编码过程
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, input_dim]
            
        返回:
            torch.Tensor: 隐空间表示 [batch_size, latent_dim]
        """
        if self.encoder_type == "lstm":
            # LSTM编码
            _, (hidden, _) = self.encoder(x)
            # 合并双向LSTM的隐状态
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            # 映射到隐空间
            latent = self.hidden_map(hidden)
            
        elif self.encoder_type == "transformer":
            # 添加位置编码
            x = x + self.position_embedding[:, :x.size(1), :]
            # Transformer编码
            memory = self.encoder(x)
            # 取序列的平均值作为隐空间表示
            latent = self.hidden_map(torch.mean(memory, dim=1))
        
        return latent
    
    def decode(self, z, seq_len=None):
        """
        解码过程
        
        参数:
            z (torch.Tensor): 隐空间表示 [batch_size, latent_dim]
            seq_len (int, optional): 输出序列长度，默认使用初始化时的序列长度
            
        返回:
            torch.Tensor: 重构输出 [batch_size, seq_len, input_dim]
        """
        if seq_len is None:
            seq_len = self.seq_length
        
        batch_size = z.size(0)
        
        if self.encoder_type == "lstm":
            # 准备初始隐状态
            h0 = torch.zeros((2, batch_size, self.input_dim)).to(device)
            c0 = torch.zeros((2, batch_size, self.input_dim)).to(device)
            
            # 重复隐空间表示以创建输入序列
            z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
            
            # LSTM解码
            output, _ = self.decoder(z_seq, (h0, c0))
            
            # 应用输出层
            output = self.output_layer(output)
            
        elif self.encoder_type == "transformer":
            # 将隐空间表示映射回输入维度
            z_mapped = self.hidden_unmap(z)
            
            # 创建目标序列（全零）
            tgt = torch.zeros(batch_size, seq_len, self.input_dim).to(device)
            
            # 添加位置编码
            tgt = tgt + self.position_embedding[:, :seq_len, :]
            
            # 创建记忆序列
            memory = z_mapped.unsqueeze(1).repeat(1, seq_len, 1)
            
            # Transformer解码
            output = self.decoder(tgt, memory)
        
        return output
    
    def one_hot_encode_action(self, action):
        """
        将离散动作转换为one-hot编码
        
        参数:
            action (torch.Tensor): 离散动作张量 [batch_size] 或 [batch_size, seq_len]
            
        返回:
            torch.Tensor: one-hot编码后的动作
        """
        if self.action_dim is None:
            raise ValueError("未设置action_dim，无法进行one-hot编码")
        
        # 处理不同形状的动作输入
        if len(action.shape) == 1:  # [batch_size]
            # 使用Embedding层进行one-hot编码
            return self.action_embedding(action)
        elif len(action.shape) == 2:  # [batch_size, seq_len]
            # 重塑为一维，进行编码，然后恢复形状
            batch_size, seq_len = action.shape
            action_flat = action.reshape(-1)
            action_emb = self.action_embedding(action_flat)
            return action_emb.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"不支持的动作形状: {action.shape}")
    
    def process_image_input(self, x):
        """
        处理图像输入
        
        参数:
            x (torch.Tensor): 图像输入 [batch_size, C, H, W] 或 [batch_size, seq_len, C, H, W]
            
        返回:
            torch.Tensor: 处理后的特征
        """
        if not self.is_image_input:
            return x
        
        # 处理不同形状的图像输入
        if len(x.shape) == 4:  # [batch_size, C, H, W]
            return self.image_encoder(x)
        elif len(x.shape) == 5:  # [batch_size, seq_len, C, H, W]
            batch_size, seq_len = x.shape[:2]
            x_flat = x.reshape(-1, *x.shape[2:])
            x_enc = self.image_encoder(x_flat)
            return x_enc.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"不支持的图像形状: {x.shape}")
    
    def forward(self, batch):
        """
        前向传播
        
        参数:
            batch (tuple): 包含状态和动作的批次数据 (states, actions)
            
        返回:
            tuple: (重构输出, 隐空间表示)
        """
        # 解包批次数据
        states = batch[0].to(self.device)
        actions = batch[1].to(self.device) if len(batch) > 1 else None
        
        # 处理图像输入
        if self.is_image_input:
            states = self.process_image_input(states)
        
        # 编码状态
        z = self.encode(states)
        
        # 解码状态
        output = self.decode(z, states.size(1))
        
        # 如果有动作数据，进行one-hot编码
        if actions is not None and self.action_dim is not None:
            actions_encoded = self.one_hot_encode_action(actions)
            # 这里只返回编码后的动作，不对动作进行自编码重构
        
        return output, z
    
    def train_model(self, data_loader, val_loader=None, epochs=10, learning_rate=0.001, process_actions=True):
        """
        训练模型
        
        参数:
            data_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            
        返回:
            dict: 训练历史
        """
        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.train()
            train_loss = 0.0
            
            for batch in data_loader:
                # 解包批次数据
                states = batch[0].to(self.device)
                actions = batch[1].to(self.device) if len(batch) > 1 and process_actions else None
                
                # 前向传播
                output, z = self((states, actions) if actions is not None else (states,))
                
                # 计算重构损失 (只针对状态进行重构)
                recon_loss = F.mse_loss(output, states)
                
                # 计算正则化损失（KL散度或其他）
                if self.kl_weight > 0:
                    # 简单的L2正则化
                    reg_loss = torch.mean(z ** 2)
                    loss = recon_loss + self.kl_weight * reg_loss
                else:
                    loss = recon_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(data_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        
        return history
    
    def evaluate(self, data_loader):
        """
        评估模型
        
        参数:
            data_loader (DataLoader): 数据加载器
            
        返回:
            dict: 评估结果
        """
        # 评估模式
        self.eval()
        eval_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                # 解包批次数据
                states = batch[0].to(self.device)
                actions = batch[1].to(self.device) if len(batch) > 1 else None
                
                # 前向传播
                output, z = self((states, actions) if actions is not None else (states,))
                
                # 计算重构损失 (只针对状态进行重构)
                recon_loss = F.mse_loss(output, states)
                
                # 计算正则化损失
                if self.kl_weight > 0:
                    reg_loss = torch.mean(z ** 2)
                    loss = recon_loss + self.kl_weight * reg_loss
                else:
                    loss = recon_loss
                
                eval_loss += loss.item()
        
        # 计算平均评估损失
        eval_loss /= len(data_loader)
        
        # 计算其他评估指标
        results = {
            'loss': eval_loss,
            'latent_stats': {
                'mean': torch.mean(z).item(),
                'std': torch.std(z).item()
            },
            'tensor_embeddings': z.cpu().numpy()  # 用于可视化
        }
        
        return results