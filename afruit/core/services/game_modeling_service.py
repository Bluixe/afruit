import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
import time
import copy

# 导入基础算法模块
from utils.BehaviorCloner import BehaviorCloner
from utils.OfflineRLearner import OfflineRLearner
from utils.OfflineFSPLearner import OfflineFSPLearner

class GameModelingService:
    """
    小样本博弈建模服务类
    
    负责小样本博弈建模相关的功能，包括模型训练、评估和预测
    """
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        初始化小样本博弈建模服务
        
        参数:
            config (Dict): 配置参数字典
            logger (logging.Logger): 日志记录器
        """
        # 初始化配置
        self.config = config or {}
        
        # 设置日志记录器
        self.logger = logger or logging.getLogger(__name__)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型字典
        self.models = {}
        
        # 初始化训练历史
        self.training_history = {}
        
        self.logger.info(f"小样本博弈建模服务初始化完成，使用设备: {self.device}")
    
    def train_model(self, training_data: Dict, model_config: Dict) -> Dict:
        """
        训练小样本博弈模型
        
        参数:
            training_data (Dict): 训练数据
            model_config (Dict): 模型配置
            
        返回:
            Dict: 训练结果，包含模型和训练指标
        """
        # 获取模型类型
        model_type = model_config.get('model_type', 'BehaviorCloner')
        model_id = model_config.get('model_id', f"{model_type}_{int(time.time())}")
        
        self.logger.info(f"开始训练模型: {model_id}, 类型: {model_type}")
        
        # 初始化结果字典
        result = {
            'model_id': model_id,
            'model_type': model_type,
            'training_metrics': {},
            'model': None
        }
        
        try:
            # 根据模型类型创建模型
            if model_type == 'BehaviorCloner':
                model, metrics = self._train_behavior_cloner(training_data, model_config)
            elif model_type == 'OfflineRLearner':
                model, metrics = self._train_offline_rl(training_data, model_config)
            elif model_type == 'OfflineFSPLearner':
                model, metrics = self._train_offline_fsp(training_data, model_config)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 更新结果
            result['model'] = model
            result['training_metrics'] = metrics
            
            # 保存模型
            self.models[model_id] = model
            
            # 保存训练历史
            self.training_history[model_id] = metrics
            
            self.logger.info(f"模型训练完成: {model_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def _train_behavior_cloner(self, training_data: Dict, model_config: Dict) -> Tuple[BehaviorCloner, Dict]:
        """训练行为克隆模型"""
        self.logger.info("训练行为克隆模型")
        
        # 提取模型参数
        batch_size = model_config.get('batch_size', 32)
        network_type = model_config.get('network_type', 'MLP')
        max_epochs = model_config.get('max_epochs', 200)
        dropout_rate = model_config.get('dropout_rate', 0.2)
        context_frames = model_config.get('context_frames', 4)
        validation_split = model_config.get('validation_split', 0.2)
        
        # 创建模型
        model = BehaviorCloner(
            batch_size=batch_size,
            network_type=network_type,
            max_epochs=max_epochs,
            dropout_rate=dropout_rate
        )
        
        # 处理数据
        X_train, y_train = model.process_data(training_data, context_frames=context_frames)
        
        # 训练模型
        training_history = model.train_model(X_train, y_train, validation_split=validation_split)
        
        # 提取训练指标
        metrics = {
            'train_loss': training_history.get('train_loss', []),
            'val_accuracy': training_history.get('val_accuracy', []),
            'final_train_loss': training_history.get('train_loss', [-1])[-1],
            'final_val_accuracy': training_history.get('val_accuracy', [-1])[-1]
        }
        
        self.logger.info(f"行为克隆模型训练完成，最终训练损失: {metrics['final_train_loss']:.4f}, 验证准确率: {metrics['final_val_accuracy']:.4f}")
        
        return model, metrics
    
    def _train_offline_rl(self, training_data: Dict, model_config: Dict) -> Tuple[OfflineRLearner, Dict]:
        """训练离线强化学习模型"""
        self.logger.info("训练离线强化学习模型")
        
        # 提取模型参数
        batch_size = model_config.get('batch_size', 64)
        network_arch = model_config.get('network_arch', [256, 256])
        learning_rate = model_config.get('learning_rate', 3e-4)
        discount_factor = model_config.get('discount_factor', 0.99)
        cql_weight = model_config.get('cql_weight', 1.0)
        target_update_interval = model_config.get('target_update_interval', 100)
        num_iterations = model_config.get('num_iterations', 10000)
        
        # 创建模型
        model = OfflineRLearner(
            batch_size=batch_size,
            network_arch=network_arch,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            cql_weight=cql_weight,
            target_update_interval=target_update_interval
        )
        
        # 训练模型
        training_history = model.train(training_data, num_iterations=num_iterations)
        
        # 提取训练指标
        metrics = {
            'q_loss': training_history.get('q_loss', []),
            'cql_loss': training_history.get('cql_loss', []),
            'policy_loss': training_history.get('policy_loss', []),
            'final_q_loss': training_history.get('q_loss', [-1])[-1] if training_history.get('q_loss') else None,
            'final_cql_loss': training_history.get('cql_loss', [-1])[-1] if training_history.get('cql_loss') else None,
            'final_policy_loss': training_history.get('policy_loss', [-1])[-1] if training_history.get('policy_loss') else None
        }
        
        self.logger.info(f"离线强化学习模型训练完成，最终Q损失: {metrics['final_q_loss']}, CQL损失: {metrics['final_cql_loss']}, 策略损失: {metrics['final_policy_loss']}")
        
        return model, metrics
    
    def _train_offline_fsp(self, training_data: Dict, model_config: Dict) -> Tuple[OfflineFSPLearner, Dict]:
        """训练离线虚构自我博弈模型"""
        self.logger.info("训练离线虚构自我博弈模型")
        
        # 提取模型参数
        batch_size = model_config.get('batch_size', 64)
        network_arch = model_config.get('network_arch', [256, 256])
        learning_rate = model_config.get('learning_rate', 3e-4)
        discount_factor = model_config.get('discount_factor', 0.99)
        num_iterations = model_config.get('num_iterations', 10000)
        br_weight = model_config.get('br_weight', 0.5)
        
        # 创建模型
        model = OfflineFSPLearner(
            batch_size=batch_size,
            network_arch=network_arch,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            br_weight=br_weight
        )
        
        # 训练模型
        training_history = model.train(training_data, num_iterations=num_iterations)
        
        # 提取训练指标
        metrics = {
            'avg_policy_loss': training_history.get('avg_policy_loss', []),
            'br_policy_loss': training_history.get('br_policy_loss', []),
            'exploitability': training_history.get('exploitability', []),
            'final_avg_policy_loss': training_history.get('avg_policy_loss', [-1])[-1] if training_history.get('avg_policy_loss') else None,
            'final_br_policy_loss': training_history.get('br_policy_loss', [-1])[-1] if training_history.get('br_policy_loss') else None,
            'final_exploitability': training_history.get('exploitability', [-1])[-1] if training_history.get('exploitability') else None
        }
        
        self.logger.info(f"离线虚构自我博弈模型训练完成，最终平均策略损失: {metrics['final_avg_policy_loss']}, BR策略损失: {metrics['final_br_policy_loss']}, 可利用性: {metrics['final_exploitability']}")
        
        return model, metrics
    
    def evaluate_model(self, model_id: str, test_data: Dict, eval_config: Dict = None) -> Dict:
        """
        评估模型
        
        参数:
            model_id (str): 模型ID
            test_data (Dict): 测试数据
            eval_config (Dict): 评估配置
            
        返回:
            Dict: 评估结果
        """
        self.logger.info(f"开始评估模型: {model_id}")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 根据模型类型选择不同的评估方法
            if isinstance(model, BehaviorCloner):
                # 评估行为克隆模型
                metrics = model.evaluate_policy(test_data)
            elif isinstance(model, OfflineRLearner):
                # 评估离线强化学习模型
                metrics = model.evaluate(test_data)
            elif isinstance(model, OfflineFSPLearner):
                # 评估离线虚构自我博弈模型
                metrics = model.evaluate(test_data)
            else:
                raise ValueError(f"不支持的模型类型: {type(model).__name__}")
            
            self.logger.info(f"模型评估完成: {model_id}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def predict(self, model_id: str, state: Any) -> Any:
        """
        使用模型进行预测
        
        参数:
            model_id (str): 模型ID
            state (Any): 状态
            
        返回:
            Any: 预测结果
        """
        self.logger.info(f"使用模型进行预测: {model_id}")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 根据模型类型选择不同的预测方法
            if isinstance(model, BehaviorCloner):
                # 使用行为克隆模型预测
                if hasattr(model, 'model') and model.model is not None:
                    # 转换为张量
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    # 预测
                    with torch.no_grad():
                        action = model.model(state_tensor).numpy()
                    return action
                else:
                    raise ValueError("模型未初始化")
            elif isinstance(model, OfflineRLearner):
                # 使用离线强化学习模型预测
                return model.act(state)
            elif isinstance(model, OfflineFSPLearner):
                # 使用离线虚构自我博弈模型预测
                return model.act(state)
            else:
                raise ValueError(f"不支持的模型类型: {type(model).__name__}")
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {str(e)}")
            raise
    
    def save_model(self, model_id: str, save_path: str) -> str:
        """
        保存模型
        
        参数:
            model_id (str): 模型ID
            save_path (str): 保存路径
            
        返回:
            str: 模型保存路径
        """
        self.logger.info(f"保存模型: {model_id} 到 {save_path}")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存模型
            torch.save(model.state_dict(), save_path)
            
            self.logger.info(f"模型保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def load_model(self, model_path: str, model_type: str, model_config: Dict = None) -> str:
        """
        加载模型
        
        参数:
            model_path (str): 模型路径
            model_type (str): 模型类型
            model_config (Dict): 模型配置
            
        返回:
            str: 模型ID
        """
        self.logger.info(f"加载模型: {model_path}, 类型: {model_type}")
        
        try:
            # 根据模型类型创建模型
            if model_type == 'BehaviorCloner':
                model = BehaviorCloner(**(model_config or {}))
            elif model_type == 'OfflineRLearner':
                model = OfflineRLearner(**(model_config or {}))
            elif model_type == 'OfflineFSPLearner':
                model = OfflineFSPLearner(**(model_config or {}))
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 加载模型参数
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # 生成模型ID
            model_id = f"{model_type}_{int(time.time())}"
            
            # 保存模型
            self.models[model_id] = model
            
            self.logger.info(f"模型加载成功: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def get_model(self, model_id: str) -> Any:
        """
        获取模型
        
        参数:
            model_id (str): 模型ID
            
        返回:
            Any: 模型
        """
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        return self.models[model_id]
    
    def get_training_history(self, model_id: str) -> Dict:
        """
        获取训练历史
        
        参数:
            model_id (str): 模型ID
            
        返回:
            Dict: 训练历史
        """
        # 检查训练历史是否存在
        if model_id not in self.training_history:
            raise ValueError(f"训练历史不存在: {model_id}")
        
        return self.training_history[model_id]
    
    def get_all_models(self) -> Dict:
        """
        获取所有模型
        
        返回:
            Dict: 模型字典
        """
        return self.models