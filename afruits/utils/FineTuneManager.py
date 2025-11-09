import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Any
import time
import copy

from afruits.utils.DataLoader import DataLoaderUtil

class FineTuneManager:
    """
    微调管理器类 - 简化版
    
    功能定位：使用新数据读取模型并根据model_type进行微调训练
    """
    
    def __init__(self,
                 base_model: Optional[nn.Module] = None,
                 model_type: str = "",
                 optimizer_config: Dict = {}):
        """
        初始化微调管理器
        
        参数:
            base_model (nn.Module, optional): 预训练模型对象
            model_type (str): 模型类型，用于确定使用哪种训练方法
            optimizer_config (Dict): 优化器参数配置
        """
        # 初始化参数
        self.base_model = base_model
        self.model_type = model_type
        assert model_type in ["AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator"]
        self.optimizer_config = optimizer_config
        
        # 初始化模型和优化器
        self.model = None
        self.optimizer = None
        
        # 训练状态跟踪
        self.is_trained = False
        self.training_history = {}
    
    def setup_model(self, pretrained_weights: Optional[str] = None):
        """
        模型加载与配置
        
        参数:
            pretrained_weights (str, optional): 预训练权重路径
        
        返回值:
            nn.Module: 设置好的模型
        """
        # 检查基础模型是否存在
        if self.base_model is None:
            raise ValueError("基础模型未设置，请先设置base_model")
        
        # 创建模型副本以避免修改原始模型
        self.model = copy.deepcopy(self.base_model)
        
        # 加载预训练权重（如果提供）
        if pretrained_weights:
            try:
                # 尝试加载权重
                state_dict = torch.load(pretrained_weights)
                self.model.load_state_dict(state_dict)
                print(f"成功加载预训练权重: {pretrained_weights}")
            except Exception as e:
                print(f"加载预训练权重失败: {str(e)}")
        
        # 设置优化器
        self._setup_optimizer()
        
        return self.model
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 获取优化器配置
        lr = self.optimizer_config.get('lr', 0.001)
        weight_decay = self.optimizer_config.get('weight_decay', 0.0)
        optimizer_type = self.optimizer_config.get('type', 'Adam')
        
        # 使用统一的学习率
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 创建优化器
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            momentum = self.optimizer_config.get('momentum', 0.9)
            self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        return self.optimizer
    
    def prepare_data(self, raw_data, batch_size: int = 32):
        """
        数据预处理
        
        参数:
            raw_data (Dataset): 原始微调数据集
            batch_size (int): 批次大小，默认为32
        
        返回值:
            DataLoader: 训练数据加载器
        """
        dataloader_util = DataLoaderUtil()
        data = dataloader_util.load_expert_data(raw_data, batch_size=batch_size)
        data_loader = data['dataloader']
        return data_loader
    
    def train(self, train_loader, model, max_epochs=10, learning_rate=1e-4):
        """
        微调训练
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            max_epochs (int): 最大训练轮次
            criterion: 损失函数，默认为None（使用MSELoss）
            device (str): 设备，默认为None（自动选择）
        
        返回值:
            Dict: 训练报告，包含训练指标
        """
        # 检查模型和优化器是否已设置
        if self.model is None:
            raise ValueError("模型未设置，请先调用setup_model方法")
        if self.optimizer is None:
            raise ValueError("优化器未设置，请先调用setup_model方法")
        
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 设置损失函数
        if criterion is None:
            criterion = nn.MSELoss()
        
        # 初始化训练状态
        training_history = {
            'train_loss': [],
            'learning_rates': []
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练循环
        for epoch in range(max_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                # 将数据移动到设备
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # 前向传播
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            training_history['train_loss'].append(train_loss)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            training_history['learning_rates'].append(current_lr)
            
            # 打印训练进度
            print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.6f}, LR: {current_lr:.6f}")
        
        # 计算训练时间
        training_time = (time.time() - start_time) / 60.0  # 转换为分钟
        
        # 标记模型已训练
        self.is_trained = True
        
        # 保存训练历史
        self.training_history = training_history
        
        # 生成训练报告
        training_report = {
            'final_train_loss': train_loss,
            'training_time': training_time,
            'train_loss_history': training_history['train_loss']
        }
        
        return training_report
    
    def _train_evolutionary(self, train_loader, max_epochs: int = 10, criterion=None, device: str = None):
        """进化学习方法训练"""
        print("使用进化学习方法训练")
        return self.train(train_loader, max_epochs, criterion, device)
    
    def _train_incremental(self, train_loader, max_epochs: int = 10, criterion=None, device: str = None):
        """增量学习方法训练"""
        print("使用增量学习方法训练")
        return self.train(train_loader, max_epochs, criterion, device)
    
    def _train_vae(self, train_loader, max_epochs: int = 10, criterion=None, device: str = None):
        """VAE方法训练"""
        print("使用VAE方法训练")
        return self.train(train_loader, max_epochs, criterion, device)
    
    def _train_diffusion(self, train_loader, max_epochs: int = 10, criterion=None, device: str = None):
        """扩散模型方法训练"""
        print("使用扩散模型方法训练")
        return self.train(train_loader, max_epochs, criterion, device)
    
    def fine_tune(self, train_loader, model, max_epochs=10, learning_rate=1e-4):
        """
        根据模型类型选择合适的训练方法进行微调
        
        参数:
            expert_trajectories: 专家轨迹数据
            epochs: 训练轮次
            criterion: 损失函数
            device: 训练设备
            
        返回值:
            Dict: 训练报告
        """
        if self.model_type == "AutoencoderModel":
            training_history = model.train_model(train_loader, epochs=max_epochs, learning_rate=learning_rate)
        elif self.model_type == "TransformerModel":
            training_history = model.train_model(train_loader, epochs=max_epochs, learning_rate=learning_rate)
        elif self.model_type == "DiffusionTrajGenerator":
            training_history = model.train(train_loader, epochs=max_epochs, learning_rate=learning_rate)
        elif self.model_type == "VAETrajGenerator":
            training_history = model.train(train_loader, epochs=max_epochs, learning_rate=learning_rate)

        model.save_model()

        return training_history
    
    def get_model(self):
        """获取训练后的模型"""
        return self.model