import torch
import torch.nn as nn
import transformers
from transformers import GPT2Config, GPT2Model
import numpy as np
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from afruits.utils.DataLoader import DataLoaderUtil

class TransformerModel(nn.Module):
    """
    通用的Transformer模型类，基于注意力机制的序列处理模型。
    
    核心特性:
    - 多头注意力机制，支持多头自注意力计算
    - 信息瓶颈设计，通过正则化和注意力机制
    - 多任务支持，支持多种预测和训练任务
    - 位置编码，支持序列位置信息的编码
    - 支持处理state和action数据，可处理图像输入
    """
    
    def __init__(self,
                 d_model=128,         # 模型隐藏层维度
                 num_heads=4,         # 注意力头数量
                 num_layers=3,        # Transformer层数
                 max_seq_len=100,     # 最大序列长度
                 dropout_rate=0.2,    # Dropout比率
                 lr_weight=0.01,      # 信息瓶颈权重
                 device=None):        # 设备
        """
        初始化TransformerModel
        
        参数:
            input_dim (int): 输入特征维度
            action_dim (int): 动作维度
            d_model (int): 模型隐藏层维度
            num_heads (int): 注意力头数量
            num_layers (int): Transformer层数
            max_seq_len (int): 最大序列长度
            dropout_rate (float): Dropout比率
            lr_weight (float): 信息瓶颈权重
            device: 计算设备
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = None
        self.action_dim = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.lr_weight = lr_weight
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.im_embd = 64  # 图像嵌入维度
        
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
        
        # 层归一化
        self.embed_ln = nn.LayerNorm(self.d_model)
        
        # 输出层 - 预测动作
        self.predict_actions = None
        
        # 是否使用图像编码器（将在forward中根据输入维度决定）
        self.use_image_encoder = False
    
    def _init_image_encoder(self, h, w, c):
        """
        初始化图像编码器
        
        参数:
            h (int): 图像高度
            w (int): 图像宽度
            c (int): 图像通道数
        """
        self.use_image_encoder = True
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * h * w), self.im_embd),
            nn.ReLU(),
        )
        
        # 图像解码器
        self.image_decoder = nn.Sequential(
            nn.Linear(self.im_embd, 16 * h * w),
            nn.ReLU(),
            nn.Unflatten(1, (16, h, w)),
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, kernel_size=3, padding=1),
        )
        
        # 更新嵌入层
        new_dim = self.im_embd + self.action_dim + 1  # 图像编码 + one-hot动作 + 奖励
        self.embed_transition = nn.Linear(new_dim, self.d_model)
    
    def _init_vector_encoder(self):
        """初始化向量编码器"""
        self.use_image_encoder = False
        
        # 更新嵌入层
        new_dim = self.input_dim + self.action_dim + 1  # 状态向量 + one-hot动作 + 奖励
        self.embed_transition = nn.Linear(new_dim, self.d_model)
    
    def build_model(self, input_dim, output_dim):
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
        if isinstance(input_dim, tuple) and len(input_dim) == 1:
            input_dim = input_dim[0]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = output_dim
        
        # 检测输入是否为图像
        if isinstance(input_dim, tuple) and len(input_dim) == 3:  # (C, H, W)
            # 图像输入
            c, h, w = input_dim
            self._init_image_encoder(h, w, c)
        elif isinstance(input_dim, int) and input_dim > 0:
            # 向量输入
            self._init_vector_encoder()
        
        # 更新输入嵌入层
        if not self.use_image_encoder:
            self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # 输出层
        self.predict_actions = nn.Linear(self.d_model, output_dim)
        
        return self
    
    def forward(self, batch, output_attentions=False):
        """
        前向传播
        
        参数:
            batch: 包含states和actions的批次数据
            output_attentions (bool): 是否输出注意力权重
            
        返回:
            torch.Tensor: 预测的动作概率
        """
        # 从batch中获取数据
        states = batch[0].to(self.device)
        actions = batch[1].to(self.device)
        
        # 确保action是long类型并转换为one-hot编码
        if len(actions.shape) == 1:  # [batch_size]
            actions = actions.unsqueeze(1)  # [batch_size, 1]
        
        if len(actions.shape) == 2:  # [batch_size, seq_len]
            actions = actions.long()
        else:  # [batch_size, seq_len, 1]
            actions = actions.squeeze(-1).long()
        
        # 转换为one-hot编码
        actions_one_hot = torch.nn.functional.one_hot(
            actions, num_classes=self.action_dim).float()
        
        # 创建奖励占位符（在这个模型中我们不使用奖励，但保持接口一致）
        rewards = torch.zeros(actions.shape[0], actions.shape[1], 1).to(self.device)
        
        # 处理状态
        if self.use_image_encoder:
            # 处理图像状态
            batch_size = states.shape[0]
            
            # 重塑图像序列以便通过编码器处理
            if len(states.shape) == 5:  # [batch_size, seq_len, channels, height, width]
                seq_len = states.shape[1]
                states_reshaped = states.view(-1, states.shape[2], states.shape[3], states.shape[4])
            else:  # [batch_size, channels, height, width]
                seq_len = 1
                states_reshaped = states
            
            # 编码图像
            encoded_states = self.image_encoder(states_reshaped)
            encoded_states = encoded_states.view(batch_size, seq_len, self.im_embd)
        else:
            encoded_states = states
        
        # 将状态、动作和奖励在特征维度上拼接
        # print(actions_one_hot.shape, rewards.shape, encoded_states.shape)
        stacked_inputs = torch.cat([
            actions_one_hot,
            rewards,
            encoded_states,
        ], dim=2)
        
        # 应用线性变换和层归一化
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Transformer前向传播
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            output_attentions=output_attentions
        )
        
        # 预测动作
        preds = self.predict_actions(transformer_outputs.last_hidden_state)
        
        if output_attentions:
            return preds, transformer_outputs.attentions
        else:
            return preds


class TransformerTrainer:
    """
    Transformer模型训练器，负责模型的训练、评估和预测。
    """
    
    def __init__(self,
                 d_model=128,         # 模型隐藏层维度
                 num_heads=4,         # 注意力头数量
                 num_layers=3,        # Transformer层数
                 max_seq_len=100,     # 最大序列长度
                 dropout_rate=0.2,    # Dropout比率
                 lr_weight=0.01,      # 信息瓶颈权重
                 device=None):        # 设备
        """
        初始化TransformerModel
        
        参数:
            input_dim (int): 输入特征维度
            action_dim (int): 动作维度
            d_model (int): 模型隐藏层维度
            num_heads (int): 注意力头数量
            num_layers (int): Transformer层数
            max_seq_len (int): 最大序列长度
            dropout_rate (float): Dropout比率
            lr_weight (float): 信息瓶颈权重
            device: 计算设备
        """
        self.input_dim = None
        self.action_dim = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.lr_weight = lr_weight
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.im_embd = 64  # 图像嵌入维度

        self.model = TransformerModel(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            dropout_rate=self.dropout_rate,
            lr_weight=self.lr_weight,
            device=self.device
        ).to(self.device)

        self.config_to_save = {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'lr_weight': self.lr_weight,
        }


    def build_model(self, input_dim, output_dim):
        """
        构建模型
        
        参数:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            
        返回:
            TransformerTrainer: 自身实例
        """
        self.config_to_save.update({
            'input_dim': input_dim,
            'output_dim': output_dim,
        })
        self.input_dim = input_dim
        self.action_dim = output_dim
        
        self.model.build_model(input_dim, output_dim)
        
        return self.model

    def load_sequences(self, raw_data, batch_size=32):
        """
        加载序列数据
        
        参数:
            raw_data (dict): 原始数据字典
            batch_size (int): 批处理大小
            
        返回:
            DataLoader: 数据加载器
        """
        dataloader_util = DataLoaderUtil()
        data = dataloader_util.load_expert_data(raw_data, batch_size=batch_size)
        data_loader = data['dataloader']
        return data_loader
    
    def train_model(self, train_loader, val_loader=None, epochs=10, learning_rate=0.001):
        """
        训练模型
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            
        返回:
            dict: 训练历史
        """
        # 优化器
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # 损失函数 - 交叉熵损失用于分类任务
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # 前向传播
                pred_actions = self.model(batch)
                
                # 获取真实动作
                true_actions = batch[1].to(self.device)
                
                # 重塑预测和真实动作以适应损失函数
                true_actions = true_actions.reshape(-1).long()
                pred_actions = pred_actions.reshape(-1, self.model.action_dim)
                
                # 计算损失
                loss = criterion(pred_actions, true_actions)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, criterion)
                val_loss = float(val_metrics.get('eval_loss', 0.0))
                history['val_loss'].append(val_loss)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_metrics.get("accuracy", 0.0):.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        
        return history
    
    def evaluate(self, data_loader, criterion=None):
        """
        评估模型（离散动作，增强指标版）
        - 指标对齐 BehaviorCloner.evaluate_policy
        - 计算交叉熵损失与分类准确率
        - 额外指标：mean_abs_error, error_distribution(兼容旧版), per_action_accuracy,
                  action_hist, action_entropy, action_switch_rate, unique_actions_ratio
        
        参数:
            data_loader (DataLoader): 数据加载器
            criterion (nn.Module, optional): 损失函数（默认CrossEntropyLoss）
            
        返回:
            dict: 指标字典
                  {
                    accuracy, eval_loss, mean_abs_error, error_distribution,
                    per_action_accuracy, action_hist, action_entropy,
                    action_switch_rate, unique_actions_ratio
                  }
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        import torch.nn.functional as F
        
        # 评估模式
        self.model.eval()
        eval_loss = 0.0

        total_samples = 0
        correct_predictions = 0
        all_abs_errors = []
        per_class_correct = {}
        per_class_total = {}
        hist_counts = {}
        total_switches = 0
        total_switch_den = 0
        entropies = []
        predicted_unique_actions = set()
        
        with torch.no_grad():
            for batch in data_loader:
                # 前向传播得到logits: [B, T, A] 或 [N, A]
                logits = self.model(batch)
                
                # 真实动作（保持原始序列形状以供序列指标使用）
                true_actions_seq = batch[1].to(self.device)

                # 展平用于交叉熵损失
                if logits.dim() == 3:
                    B, T, A = logits.shape
                    logits_flat = logits.reshape(-1, A)
                    true_flat = true_actions_seq.view(-1).long()
                else:
                    N, A = logits.shape
                    logits_flat = logits.view(-1, A)
                    true_flat = true_actions_seq.view(-1).long()

                # 损失
                loss = criterion(logits_flat, true_flat)
                eval_loss += loss.item()

                # 预测与准确率
                preds_flat = torch.argmax(logits_flat, dim=-1)
                correct_predictions += (preds_flat == true_flat).sum().item()
                total_samples += true_flat.numel()

                # 绝对误差、类别直方图与每类统计
                for pr, gt in zip(preds_flat.tolist(), true_flat.tolist()):
                    pr_i, gt_i = int(pr), int(gt)
                    all_abs_errors.append(abs(pr_i - gt_i))
                    if pr_i == gt_i:
                        per_class_correct[gt_i] = per_class_correct.get(gt_i, 0) + 1
                    per_class_total[gt_i] = per_class_total.get(gt_i, 0) + 1
                    hist_counts[pr_i] = hist_counts.get(pr_i, 0) + 1
                    predicted_unique_actions.add(pr_i)

                # 信息熵：-sum p log p
                probs_flat = F.softmax(logits_flat, dim=-1).clamp(min=1e-12)
                entropy_flat = (-probs_flat * probs_flat.log()).sum(dim=-1)
                entropies.extend(entropy_flat.detach().cpu().numpy().tolist())

                # 动作切换率（序列场景）
                if logits.dim() == 3 and T > 1:
                    preds_seq = preds_flat.view(B, T)
                    switches = (preds_seq[:, 1:] != preds_seq[:, :-1]).sum().item()
                    total_switches += int(switches)
                    total_switch_den += int(B * (T - 1))
        
        # 平均评估损失
        eval_loss /= max(len(data_loader), 1)
        
        # 准确率
        accuracy = float(correct_predictions / total_samples) if total_samples > 0 else 0.0
        
        # 汇总指标
        metrics = {
            'accuracy': accuracy,
            'eval_loss': float(eval_loss),
            'mean_abs_error': float(np.mean(all_abs_errors)) if all_abs_errors else 0.0,
            'error_distribution': float(np.mean(all_abs_errors)) if all_abs_errors else 0.0,  # 兼容旧版（标量）
            'action_hist': {int(k): int(v) for k, v in hist_counts.items()},
            'action_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'action_switch_rate': float(total_switches / total_switch_den) if total_switch_den > 0 else 0.0,
        }
        
        # 唯一动作比率
        try:
            action_space = int(getattr(self.model, 'action_dim', 0))
            if action_space <= 0 and len(hist_counts) > 0:
                action_space = int(max(hist_counts.keys())) + 1
            metrics['unique_actions_ratio'] = float(len(predicted_unique_actions) / action_space) if action_space > 0 else 0.0
        except Exception:
            metrics['unique_actions_ratio'] = 0.0
        
        # 各类别准确率
        per_action_acc = {}
        for cls, tot in per_class_total.items():
            cor = per_class_correct.get(cls, 0)
            per_action_acc[int(cls)] = float(cor / tot) if tot > 0 else 0.0
        metrics['per_action_accuracy'] = per_action_acc
        
        return metrics
    
    def predict(self, states):
        """
        预测动作（离散）：
        - 返回离散动作索引（通过对logits执行argmax）
        
        参数:
            states (torch.Tensor): 状态序列（支持批/序列）
            
        返回:
            np.ndarray: 离散动作索引（形状与输入序列维度匹配）
        """
        # 评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 确保输入是张量并移动到正确的设备
            if not isinstance(states, torch.Tensor):
                states = torch.tensor(states, dtype=torch.float32).to(self.device)
            else:
                states = states.to(self.device)
            
            # 创建一个虚拟的动作序列（全零）
            if len(states.shape) == 4:  # 单个图像 [batch_size, channels, height, width]
                batch_size = states.shape[0]
                dummy_actions = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
            elif len(states.shape) == 5:  # 图像序列 [batch_size, seq_len, channels, height, width]
                batch_size, seq_len = states.shape[0], states.shape[1]
                dummy_actions = torch.zeros(batch_size, seq_len, dtype=torch.long).to(self.device)
            elif len(states.shape) == 2:  # 单个向量状态 [batch_size, features]
                batch_size = states.shape[0]
                dummy_actions = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
            elif len(states.shape) == 3:  # 向量状态序列 [batch_size, seq_len, features]
                batch_size, seq_len = states.shape[0], states.shape[1]
                dummy_actions = torch.zeros(batch_size, seq_len, dtype=torch.long).to(self.device)
            else:
                raise ValueError(f"不支持的状态形状: {states.shape}")
            
            # 创建批次
            batch = [states, dummy_actions]
            
            # 前向传播得到logits
            logits = self.model(batch)
            
            # 取argmax得到离散动作索引
            action_indices = torch.argmax(logits, dim=-1)
            
            # 返回CPU numpy数组，便于后续评估/统计
            return action_indices.detach().cpu().numpy()
        
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
            save_path = f"models/transformer.pt"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 保存模型参数和配置
        model_state = {
            'state_dict': self.model.state_dict(),
            'config': {k:v for k,v in self.config_to_save.items()}
        }
        
        # 保存模型
        torch.save(model_state, save_path)
        
        config_path = os.path.splitext(save_path)[0] + '_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_state['config'], f, ensure_ascii=False, indent=4)
            
        print(f"模型已保存至: {save_path}")
        print(f"配置已保存至: {config_path}")

    @staticmethod
    def load_model(load_path, device: torch.device = None) -> 'TransformerModel':
        """
        静态方法：加载模型参数和配置，返回TransformerModel实例
        
        参数:
            load_path (str): 模型加载路径
            device (torch.device, optional): 计算设备
            
        返回值:
            TransformerModel实例
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_path is None:
            load_path = f"models/transformer.pt"
        
        # 加载模型状态
        checkpoint = torch.load(load_path, map_location=device)
        config = checkpoint['config']
        
        # 创建TransformerModel实例
        model_trainer = TransformerTrainer(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            max_seq_len=config['max_seq_len'],
            dropout_rate=config['dropout_rate'],
            lr_weight=config['lr_weight'],
            device=device
        )
        
        # 构建模型
        model_trainer.build_model(config['input_dim'], config['output_dim'])
        
        # 加载模型参数
        model_trainer.model.load_state_dict(checkpoint['state_dict'])
        
        print(f"成功加载模型: {load_path}")
        
        return model_trainer