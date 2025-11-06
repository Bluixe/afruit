import torch
import torch.nn as nn
import transformers
from transformers import GPT2Config, GPT2Model
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel(nn.Module):
    """
    通用的Transformer模型类，基于注意力机制的序列处理模型。
    
    核心特性:
    - 多头注意力机制，支持多头自注意力计算
    - 信息瓶颈设计，通过正则化和注意力机制
    - 多任务支持，支持多种预测和训练任务
    - 位置编码，支持序列位置信息的编码
    """
    
    def __init__(self, 
                 encoder_type="str",  # 编码器类型: "str"或"transformer"
                 input_dim=32,        # 输入特征维度
                 d_model=128,         # 模型隐藏层维度
                 num_heads=4,         # 注意力头数量
                 num_layers=3,        # Transformer层数
                 max_seq_len=100,     # 最大序列长度
                 dropout_rate=0.2,    # Dropout比率
                 lr_weight=0.01):     # 信息瓶颈权重
        """
        初始化TransformerModel
        
        参数:
            encoder_type (str): 编码器类型，可选"str"或"transformer"
            input_dim (int): 输入特征维度
            d_model (int): 模型隐藏层维度
            num_heads (int): 注意力头数量
            num_layers (int): Transformer层数
            max_seq_len (int): 最大序列长度
            dropout_rate (float): Dropout比率
            lr_weight (float): 信息瓶颈权重
        """
        super(TransformerModel, self).__init__()
        
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.lr_weight = lr_weight
        
        # 创建GPT2配置
        self.config = GPT2Config(
            n_positions=self.max_seq_len,
            n_embd=self.d_model,
            n_layer=self.num_layers,
            n_head=self.num_heads,
            resid_pdrop=self.dropout_rate,
            embd_pdrop=self.dropout_rate,
            attn_pdrop=self.dropout_rate,
            use_cache=False,
        )
        
        # 初始化Transformer模型
        self.transformer = GPT2Model(self.config)
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_seq_len, self.d_model)
        )
        
        # 初始化位置编码
        self._init_position_embedding()
    
    def _init_position_embedding(self):
        """初始化位置编码"""
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )
        
        pe = torch.zeros(self.max_seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.position_embedding.data = pe.unsqueeze(0)
    
    def load_sequences(self, raw_data, batch_size=32):
        """
        加载序列数据
        
        参数:
            raw_data (dict): 原始数据字典
            batch_size (int): 批处理大小
            
        返回:
            DataLoader: 数据加载器
        """
        # 这里可以实现数据加载逻辑，返回DataLoader
        # 由于这是一个模型类，实际上数据加载可能在外部完成
        pass
    
    def build_model(self, input_dim, output_dim, encoder_type="str", decoder_type="fc"):
        """
        构建模型
        
        参数:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            encoder_type (str): 编码器类型
            decoder_type (str): 解码器类型
            
        返回:
            tuple: (encoder, decoder)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 更新输入嵌入层
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # 输出层
        if decoder_type == "fc":
            self.output_layer = nn.Linear(self.d_model, output_dim)
        else:
            # 可以实现其他类型的解码器
            self.output_layer = nn.Linear(self.d_model, output_dim)
        
        return self
    
    def forward(self, x, attention_mask=None, output_attentions=False):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, input_dim]
            attention_mask (torch.Tensor, optional): 注意力掩码
            output_attentions (bool): 是否输出注意力权重
            
        返回:
            torch.Tensor: 输出张量 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 确保序列长度不超过最大长度
        if seq_len > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            seq_len = self.max_seq_len
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]
        
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 添加位置编码
        x = x + self.position_embedding[:, :seq_len, :]
        
        # 层归一化
        x = self.layer_norm(x)
        
        # Transformer前向传播
        transformer_outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        # 获取隐藏状态
        hidden_states = transformer_outputs.last_hidden_state
        
        # 输出层
        outputs = self.output_layer(hidden_states)
        
        if output_attentions:
            return outputs, transformer_outputs.attentions
        else:
            return outputs
    
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
        
        # 损失函数
        criterion = nn.MSELoss()
        
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
                # 前向传播
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # 添加信息瓶颈正则化
                if self.lr_weight > 0:
                    # 计算注意力权重的熵作为正则化项
                    _, attentions = self(inputs, output_attentions=True)
                    attention_entropy = 0
                    for attention in attentions:
                        # 计算注意力权重的熵
                        entropy = -torch.sum(attention * torch.log(attention + 1e-10)) / attention.size(0)
                        attention_entropy += entropy
                    
                    # 添加到损失中
                    loss += self.lr_weight * attention_entropy
                
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
                val_loss = self.evaluate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        
        return history
    
    def evaluate(self, data_loader, criterion=None):
        """
        评估模型
        
        参数:
            data_loader (DataLoader): 数据加载器
            criterion (nn.Module, optional): 损失函数
            
        返回:
            float: 评估损失
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        # 评估模式
        self.eval()
        eval_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                # 前向传播
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                eval_loss += loss.item()
        
        # 计算平均评估损失
        eval_loss /= len(data_loader)
        
        return eval_loss
    
    def predict(self, input_seq, pred_steps=1):
        """
        预测
        
        参数:
            input_seq (torch.Tensor): 输入序列
            pred_steps (int): 预测步数
            
        返回:
            dict: 预测结果
        """
        # 评估模式
        self.eval()
        
        with torch.no_grad():
            # 确保输入是张量
            if not isinstance(input_seq, torch.Tensor):
                input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)
            
            # 添加批次维度（如果需要）
            if len(input_seq.shape) == 2:
                input_seq = input_seq.unsqueeze(0)
            
            # 前向传播
            outputs = self(input_seq)
            
            # 获取预测结果
            predictions = outputs[:, -pred_steps:, :]
            
            # 计算注意力权重（如果需要）
            _, attentions = self(input_seq, output_attentions=True)
            attention_map = torch.cat([att.mean(dim=1) for att in attentions], dim=0)
            
            return {
                'trajectory': predictions.cpu().numpy(),
                'attention_map': attention_map.cpu().numpy()
            }