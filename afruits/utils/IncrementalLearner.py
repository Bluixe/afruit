import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import random
from collections import deque

class IncrementalLearner:
    """
    增量学习器类 - 简化版
    
    功能定位：使用新数据读取模型并进行增量训练
    
    核心特性：
    • 实时性适应：支持回放/正则化/多任务机制
    • 动态适应：支持数据分布漂移检测与自适应调整
    • 弹性更新：混合精度训练与新旧经验平衡
    • 多策略协同：可配置增量学习模式组合
    """
    
    def __init__(self,
                 base_model: Optional[nn.Module] = None,
                 model_type: str = "",
                 optimizer_config: Dict = {},
                 memory_buffer_size: int = 1000,
                 replay_strategy: str = "random"):
        """
        初始化增量学习器
        
        参数:
            base_model (nn.Module, optional): 预训练模型对象
            model_type (str): 模型类型，用于确定使用哪种训练方法
            optimizer_config (Dict): 优化器参数配置
            memory_buffer_size (int): 历史样本回放缓冲区容量
            replay_strategy (str): 回放策略("random", "prioritized", "generative")
        """
        # 初始化参数
        self.base_model = base_model
        self.model_type = model_type
        assert model_type in ["AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator"]
        self.optimizer_config = optimizer_config
        self.memory_buffer_size = memory_buffer_size
        self.replay_strategy = replay_strategy
        
        # 初始化内部状态
        self.memory_buffer = deque(maxlen=self.memory_buffer_size)
        self.model = None
        self.optimizer = None
        self.drift_detector = None
        
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
    def monitor_data_stream(self, new_data, drift_threshold: float = 0.3) -> dict:
        """
        数据流监控
        
        输入参数:
            new_data (Dataset): 新增数据集对象
            drift_threshold (float): 分布漂移检测阈值
        
        返回值:
            漂移报告 (dict):
                is_drift: 是否发生显著分布漂移
                drift_score: 分布差异量化值
                feature_contrib: 各特征维度贡献度
        """
        # 检查输入参数
        if new_data is None:
            raise ValueError("new_data不能为None")
        
        if not isinstance(drift_threshold, float) or not (0.0 < drift_threshold < 1.0):
            raise ValueError(f"drift_threshold必须在0.0-1.0范围内，当前值: {drift_threshold}")
        
        # 初始化漂移报告
        drift_report = {
            'is_drift': False,
            'drift_score': 0.0,
            'feature_contrib': {}
        }
        
        # 如果没有历史数据，无法检测漂移
        if len(self.memory_buffer) == 0:
            print("警告: 内存缓冲区为空，无法检测分布漂移")
            # 将新数据添加到内存缓冲区
            self._update_memory_buffer(new_data)
            return drift_report
        
        # 提取历史数据特征
        historical_features = self._extract_features_from_buffer()
        
        # 提取新数据特征
        new_features = self._extract_features(new_data)
        
        # 计算分布差异
        drift_score, feature_contributions = self._calculate_distribution_drift(
            historical_features, new_features
        )
        
        # 更新漂移报告
        drift_report['drift_score'] = drift_score
        drift_report['feature_contrib'] = feature_contributions
        
        # 判断是否发生显著漂移
        if drift_score > drift_threshold:
            drift_report['is_drift'] = True
            print(f"检测到显著分布漂移，漂移分数: {drift_score:.4f}")
        
        # 更新内存缓冲区
        self._update_memory_buffer(new_data)
        
        return drift_report
    
    def _extract_features(self, data):
        """从数据中提取特征"""
        # 简单实现：假设数据已经是特征形式
        features = []
        for batch in data:
            if isinstance(batch, tuple) and len(batch) >= 1:
                # 如果是DataLoader格式 (inputs, targets)
                inputs = batch[0]
                features.append(inputs.detach().cpu().numpy())
            else:
                # 如果直接是特征数据
                features.append(batch.detach().cpu().numpy())
        
        return np.vstack(features) if features else np.array([])
    
    def _extract_features_from_buffer(self):
        """从内存缓冲区提取特征"""
        features = []
        for item in self.memory_buffer:
            if isinstance(item, tuple) and len(item) >= 1:
                # 如果是 (inputs, targets) 格式
                inputs = item[0]
                if isinstance(inputs, torch.Tensor):
                    features.append(inputs.detach().cpu().numpy())
                else:
                    features.append(inputs)
            else:
                # 如果直接是特征数据
                if isinstance(item, torch.Tensor):
                    features.append(item.detach().cpu().numpy())
                else:
                    features.append(item)
        
        return np.vstack(features) if features else np.array([])
    
    def _calculate_distribution_drift(self, historical_features, new_features):
        """计算分布漂移程度"""
        if len(historical_features) == 0 or len(new_features) == 0:
            return 0.0, {}
        
        # 计算均值差异
        hist_mean = np.mean(historical_features, axis=0)
        new_mean = np.mean(new_features, axis=0)
        mean_diff = np.abs(hist_mean - new_mean)
        
        # 计算标准差
        hist_std = np.std(historical_features, axis=0)
        new_std = np.std(new_features, axis=0)
        # 避免除零错误
        combined_std = np.maximum(hist_std + new_std, 1e-8)
        
        # 计算标准化差异
        normalized_diff = mean_diff / combined_std
        
        # 计算总体漂移分数
        drift_score = np.mean(normalized_diff)
        
        # 计算各特征的贡献度
        feature_contrib = {}
        for i in range(len(normalized_diff)):
            feature_contrib[f"feature_{i}"] = float(normalized_diff[i])
        
        return drift_score, feature_contrib
    def elastic_update(self, model, importance_weights: dict = None):
        """
        弹性权重更新
        
        输入参数:
            model (nn.Module): 待更新模型
            importance_weights (dict): 参数重要性矩阵
        
        处理流程:
            1. 计算参数重要性矩阵
            2. 应用正则化损失函数
            3. 混合损失反向传播
        """
        # 检查模型是否有效
        if model is None:
            raise ValueError("model不能为None")
        
        # 如果没有提供重要性权重，则初始化为均匀分布
        if importance_weights is None:
            importance_weights = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    importance_weights[name] = torch.ones_like(param.data)
        
        # 保存当前模型参数的副本
        old_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                old_params[name] = param.data.clone()
        
        # 设置优化器（如果尚未设置）
        if self.optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 设置弹性权重正则化
        def elastic_weight_consolidation_loss(model, old_params, importance_weights, reg_strength):
            """计算弹性权重正则化损失"""
            reg_loss = 0
            for name, param in model.named_parameters():
                if name in old_params and name in importance_weights:
                    # 计算参数变化的正则化损失
                    reg_loss += (importance_weights[name] * (param - old_params[name]).pow(2)).sum()
            return reg_strength * reg_loss
        
        # 返回弹性更新函数
        def update_with_ewc(inputs, targets, criterion):
            """使用弹性权重正则化进行更新"""
            # 前向传播
            outputs = model(inputs)
            
            # 计算任务损失
            task_loss = criterion(outputs, targets)
            
            # 计算正则化损失
            ewc_loss = elastic_weight_consolidation_loss(
                model, old_params, importance_weights, self.regularization_strength
            )
            
            # 计算总损失
            total_loss = task_loss + ewc_loss
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return {
                'task_loss': task_loss.item(),
                'ewc_loss': ewc_loss.item(),
                'total_loss': total_loss.item()
            }
            
            # 更新模型引用
            self.model = model
            
            return update_with_ewc
    def prepare_data(self, raw_data, batch_size: int = 32):
        """
        数据预处理
        
        参数:
            raw_data (Dataset): 原始增量数据集
            batch_size (int): 批次大小，默认为32
        
        返回值:
            DataLoader: 训练数据加载器
        """
        from afruits.utils.DataLoader import DataLoaderUtil
        dataloader_util = DataLoaderUtil()
        data = dataloader_util.load_expert_data(raw_data, batch_size=batch_size)
        data_loader = data['dataloader']
        return data_loader
    
    def _train_on_new_data(self, new_data, epochs):
        """仅在新数据上训练"""
        # 设置损失函数
        criterion = nn.CrossEntropyLoss() if self._is_classification_task() else nn.MSELoss()
        
        # 训练过程
        for epoch in range(epochs):
            epoch_loss = 0.0
            samples = 0
            
            for inputs, targets in new_data:
                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                samples += inputs.size(0)
            
            # 计算平均损失
            epoch_loss /= samples
            
            # 打印训练进度
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        # 返回简化的训练指标
        return {
            'old_task_acc': 0.0,  # 没有旧任务
            'new_task_acc': self._evaluate_on_data(new_data),
            'forgetting_rate': 0.0  # 没有遗忘
        }
    
    def _evaluate_on_data(self, data_loader):
        """在数据集上评估模型性能"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # 设置损失函数
        criterion = nn.CrossEntropyLoss() if self._is_classification_task() else nn.MSELoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        # 恢复训练模式
        self.model.train()
        
        # 计算平均损失
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # 转换为准确率（简化：1 - 归一化损失）
        accuracy = max(0.0, 1.0 - avg_loss / 10.0)  # 假设损失最大为10
        return min(1.0, accuracy)  # 确保准确率不超过1.0
    
    def incremental_train(self, train_loader, model, max_epochs=10, learning_rate=1e-4):
        """
        根据模型类型选择合适的训练方法进行增量训练
        
        参数:
            train_loader: 训练数据加载器
            model: 模型对象
            max_epochs: 训练轮次
            learning_rate: 学习率
            
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

        # 保存模型
        model.save_model()
        
        # 更新内存缓冲区
        self._update_memory_buffer(train_loader)
        
        return training_history
    
    def get_model(self):
        """获取训练后的模型"""
        return self.model
    
    def _update_memory_buffer(self, new_data):
        """更新内存缓冲区"""
        # 将新数据添加到内存缓冲区
        for batch in new_data:
            if isinstance(batch, tuple):
                # 如果是 (inputs, targets) 格式
                for i in range(len(batch[0])):
                    item = tuple(tensor[i] for tensor in batch)
                    self.memory_buffer.append(item)
            else:
                # 如果直接是特征数据
                for item in batch:
                    self.memory_buffer.append(item)
    
    def _update_memory_buffer(self, new_data):
        """更新内存缓冲区"""
        # 将新数据添加到内存缓冲区
        for batch in new_data:
            if isinstance(batch, tuple):
                # 如果是 (inputs, targets) 格式
                for i in range(len(batch[0])):
                    item = tuple(tensor[i] for tensor in batch)
                    self.memory_buffer.append(item)
            else:
                # 如果直接是特征数据
                for item in batch:
                    self.memory_buffer.append(item)