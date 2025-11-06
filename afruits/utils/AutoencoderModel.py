import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                 encoder_type="str",  # 编码器类型: "str"或"transformer"
                 latent_dim=32,       # 隐空间维度
                 seq_length=100,      # 序列长度
                 input_dim=512,       # 输入特征维度
                 h_weight=0.001,      # 正则化权重
                 dropout_rate=0.2):   # Dropout比率
        """
        初始化AutoencoderModel
        
        参数:
            encoder_type (str): 编码器类型，可选"str"或"transformer"
            latent_dim (int): 隐空间维度
            seq_length (int): 序列长度
            input_dim (int): 输入特征维度
            h_weight (float): 正则化权重
            dropout_rate (float): Dropout比率
        """
        super(AutoencoderModel, self).__init__()
        
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.h_weight = h_weight
        self.dropout_rate = dropout_rate
        
        # 根据编码器类型初始化不同的编码器
        if encoder_type == "str":
            # LSTM编码器
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=latent_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=True
            )
            
            # LSTM解码器
            self.decoder = nn.LSTM(
                input_size=latent_dim,
                hidden_size=input_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=False
            )
            
            # 输出层
            self.output_layer = nn.Linear(input_dim, input_dim)
            
            # 隐空间映射（双向LSTM输出到单向输入）
            self.hidden_map = nn.Linear(latent_dim * 2, latent_dim)
            
        elif encoder_type == "transformer":
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=latent_dim * 4,
                dropout=dropout_rate,
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
                dim_feedforward=latent_dim * 4,
                dropout=dropout_rate,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=2
            )
            
            # 隐空间映射
            self.hidden_map = nn.Linear(input_dim, latent_dim)
            
            # 隐空间逆映射
            self.hidden_unmap = nn.Linear(latent_dim, input_dim)
            
            # 位置编码
            self.position_embedding = nn.Parameter(
                torch.zeros(1, seq_length, input_dim)
            )
            self._init_position_embedding()
        
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
    
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
            mode (str): 数据模式，"train_test"或"train_val_test"
            
        返回:
            DataLoader: 数据加载器
        """
        # 提取特征和标签
        features = raw_data.get('features', None)
        
        if features is None:
            raise ValueError("输入数据必须包含'features'键")
        
        # 转换为张量
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # 创建数据集
        dataset = TensorDataset(features, features)  # 自编码器输入输出相同
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        return data_loader
    
    def build_model(self, input_dim, output_dim=None):
        """
        构建模型
        
        参数:
            input_dim (int): 输入维度
            output_dim (int, optional): 输出维度，默认与输入维度相同
            
        返回:
            tuple: (encoder, decoder)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # 重新初始化模型
        self.__init__(
            encoder_type=self.encoder_type,
            latent_dim=self.latent_dim,
            seq_length=self.seq_length,
            input_dim=self.input_dim,
            h_weight=self.h_weight,
            dropout_rate=self.dropout_rate
        )
        
        return self
    
    def encode(self, x):
        """
        编码过程
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, input_dim]
            
        返回:
            torch.Tensor: 隐空间表示 [batch_size, latent_dim]
        """
        if self.encoder_type == "str":
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
        
        if self.encoder_type == "str":
            # 准备初始隐状态
            h0 = torch.zeros(2, batch_size, self.input_dim).to(device)
            c0 = torch.zeros(2, batch_size, self.input_dim).to(device)
            
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
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, input_dim]
            
        返回:
            tuple: (重构输出, 隐空间表示)
        """
        # 编码
        z = self.encode(x)
        
        # 解码
        output = self.decode(z, x.size(1))
        
        return output, z
    
    def train(self, data_loader, val_loader=None, epochs=10, learning_rate=0.001):
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
            
            for batch_x, _ in data_loader:
                # 移动到设备
                batch_x = batch_x.to(device)
                
                # 前向传播
                output, z = self(batch_x)
                
                # 计算重构损失
                recon_loss = F.mse_loss(output, batch_x)
                
                # 计算正则化损失（KL散度或其他）
                if self.h_weight > 0:
                    # 简单的L2正则化
                    reg_loss = torch.mean(z ** 2)
                    loss = recon_loss + self.h_weight * reg_loss
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
            for batch_x, _ in data_loader:
                # 移动到设备
                batch_x = batch_x.to(device)
                
                # 前向传播
                output, z = self(batch_x)
                
                # 计算重构损失
                recon_loss = F.mse_loss(output, batch_x)
                
                # 计算正则化损失
                if self.h_weight > 0:
                    reg_loss = torch.mean(z ** 2)
                    loss = recon_loss + self.h_weight * reg_loss
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