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
    增量学习器类
    
    功能定位：实现动态数据流下的持续模型优化与知识保留
    
    核心特性：
    • 实时性适应：支持回放/正则化/多任务机制
    • 动态适应：支持数据分布漂移检测与自适应调整
    • 弹性更新：混合精度训练与新旧经验平衡
    • 多策略协同：可配置增量学习模式组合
    """
    
    def __init__(self,
                 memory_buffer_size: int = 1000,
                 regularization_strength: float = 0.5,
                 replay_strategy: str = "generative",
                 task_heads: dict = {},
                 adaptive_lr: bool = True):
        """
        初始化增量学习器
        
        参数:
            memory_buffer_size (int): 历史样本回放缓冲区容量
            regularization_strength (float): 弹性权重保持系数(0.1-2.0)
            replay_strategy (str): 回放策略("random", "prioritized", "generative")
            task_heads (dict): 多任务配置字典
            adaptive_lr (bool): 启用自适应学习率机制
        """
        # 初始化参数
        self.memory_buffer_size = memory_buffer_size
        self.regularization_strength = regularization_strength
        self.replay_strategy = replay_strategy
        self.task_heads = task_heads
        self.adaptive_lr = adaptive_lr
        
        # 初始化内部状态
        self.memory_buffer = deque(maxlen=self.memory_buffer_size)
        self.model = None
        self.optimizer = None
        self.drift_detector = None
        self.is_initialized = False
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证memory_buffer_size
        if not isinstance(self.memory_buffer_size, int) or self.memory_buffer_size <= 0:
            raise ValueError(f"memory_buffer_size必须为正整数，当前值: {self.memory_buffer_size}")
        
        # 验证regularization_strength
        if not isinstance(self.regularization_strength, float) or not (0.1 <= self.regularization_strength <= 2.0):
            raise ValueError(f"regularization_strength必须在0.1-2.0范围内，当前值: {self.regularization_strength}")
        
        # 验证replay_strategy
        valid_strategies = ["random", "prioritized", "generative"]
        if self.replay_strategy not in valid_strategies:
            raise ValueError(f"replay_strategy必须为 {valid_strategies} 之一，当前值: {self.replay_strategy}")
        
        # 验证task_heads
        if not isinstance(self.task_heads, dict):
            raise ValueError(f"task_heads必须为字典类型，当前类型: {type(self.task_heads)}")
        
        # 验证adaptive_lr
        if not isinstance(self.adaptive_lr, bool):
            raise ValueError(f"adaptive_lr必须为布尔类型，当前类型: {type(self.adaptive_lr)}")
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
    def train_with_replay(self, new_data, epochs: int = 10):
        """
        混合训练
        
        输入参数:
            new_data (DataLoader): 新增数据加载器
            epochs (int): 训练轮次
        
        返回值:
            训练指标 (dict):
                old_task_acc: 历史任务准确率
                new_task_acc: 新任务准确率
                forgetting_rate: 遗忘速率
        """
        # 检查输入参数
        if new_data is None:
            raise ValueError("new_data不能为None")
        
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"epochs必须为正整数，当前值: {epochs}")
        
        # 检查模型是否已初始化
        if self.model is None:
            raise ValueError("模型尚未初始化，请先初始化模型")
        
        # 初始化训练指标
        training_metrics = {
            'old_task_acc': 0.0,
            'new_task_acc': 0.0,
            'forgetting_rate': 0.0
        }
        
        # 准备回放数据
        replay_data = self._prepare_replay_data()
        
        # 如果没有回放数据，直接使用新数据训练
        if replay_data is None or len(replay_data) == 0:
            print("警告: 没有可用的回放数据，仅使用新数据训练")
            return self._train_on_new_data(new_data, epochs)
        
        # 设置损失函数
        criterion = nn.CrossEntropyLoss() if self._is_classification_task() else nn.MSELoss()
        
        # 评估旧任务初始性能
        initial_old_task_acc = self._evaluate_on_data(replay_data)
        
        # 混合训练过程
        for epoch in range(epochs):
            # 在新数据上训练
            new_task_loss = 0.0
            new_samples = 0
            
            for inputs, targets in new_data:
                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                new_task_loss += loss.item() * inputs.size(0)
                new_samples += inputs.size(0)
            
            # 计算新任务平均损失
            new_task_loss /= new_samples
            
            # 在回放数据上训练
            replay_loss = 0.0
            replay_samples = 0
            
            for inputs, targets in replay_data:
                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                replay_loss += loss.item() * inputs.size(0)
                replay_samples += inputs.size(0)
            
            # 计算回放数据平均损失
            if replay_samples > 0:
                replay_loss /= replay_samples
            
            # 打印训练进度
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, New Task Loss: {new_task_loss:.4f}, Replay Loss: {replay_loss:.4f}")
        
        # 评估训练后性能
        training_metrics['old_task_acc'] = self._evaluate_on_data(replay_data)
        training_metrics['new_task_acc'] = self._evaluate_on_data(new_data)
        
        # 计算遗忘率
        if initial_old_task_acc > 0:
            forgetting = (initial_old_task_acc - training_metrics['old_task_acc']) / initial_old_task_acc
            training_metrics['forgetting_rate'] = max(0.0, forgetting)  # 确保遗忘率非负
        
        return training_metrics
    
    def _prepare_replay_data(self):
        """准备回放数据"""
        if len(self.memory_buffer) == 0:
            return None
        
        # 根据回放策略选择样本
        if self.replay_strategy == "random":
            # 随机采样
            replay_samples = random.sample(list(self.memory_buffer), 
                                          min(len(self.memory_buffer), self.memory_buffer_size // 2))
        elif self.replay_strategy == "prioritized":
            # 优先采样（这里简化为最近的样本优先）
            replay_samples = list(self.memory_buffer)[-self.memory_buffer_size // 2:]
        elif self.replay_strategy == "generative":
            # 生成式回放（简化实现）
            replay_samples = list(self.memory_buffer)
            # 在实际应用中，这里应该使用生成模型生成样本
        
        # 转换为DataLoader格式
        # 注意：这里假设样本是(inputs, targets)格式
        inputs = []
        targets = []
        
        for sample in replay_samples:
            if isinstance(sample, tuple) and len(sample) >= 2:
                inputs.append(sample[0])
                targets.append(sample[1])
        
        if not inputs or not targets:
            return None
        
        # 转换为张量
        inputs_tensor = torch.stack(inputs) if isinstance(inputs[0], torch.Tensor) else torch.tensor(inputs)
        targets_tensor = torch.stack(targets) if isinstance(targets[0], torch.Tensor) else torch.tensor(targets)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        return DataLoader(dataset, batch_size=32, shuffle=True)
    
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
    
    def _is_classification_task(self):
        """判断是否为分类任务"""
        # 简化实现：根据任务头判断
        if not self.task_heads:
            # 默认为回归任务
            return False
        
        # 检查第一个任务头的输出类型
        first_task = next(iter(self.task_heads.values()))
        if isinstance(first_task, dict) and 'type' in first_task:
            return first_task['type'] == 'classification'
        
        return False
        return False
    
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