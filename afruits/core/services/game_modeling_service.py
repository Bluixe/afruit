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
from utils.AdversarialImitationLearner import AdversarialImitationLearner

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
            elif model_type == 'AdversarialImitationLearner':
                model, metrics = self._train_adversial_imitation_learner(training_data, model_config)
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
        
        data, state_dim, action_dim = training_data["data"], training_data["state_dim"], training_data["action_dim"]
        # 处理数据
        X_train, y_train = model.process_data(data, context_frames=context_frames)
        
        # 训练模型
        training_history = model.train_model(X_train, y_train, state_dim, action_dim, validation_split=validation_split)
        
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
        cql_weight = model_config.get('cql_weight', 0.5)
        vae_hidden_dim = model_config.get('vae_hidden_dim', 256)
        perturbation_scale = model_config.get('perturbation_scale', 0)
        replay_ratio = model_config.get('replay_ratio', 0.8)
        num_quantiles = model_config.get('num_quantiles', 200)
        discount_factor = model_config.get('discount_factor', 0.99)
        estimation_step = model_config.get('estimation_step', 1)
        target_update_freq = model_config.get('target_update_freq', 0)
        reward_normalization = model_config.get('reward_normalization', False)
        
        # 创建模型
        model = OfflineRLearner(
            cql_weight=cql_weight,
            vae_hidden_dim=vae_hidden_dim,
            perturbation_scale=perturbation_scale,
            replay_ratio=replay_ratio,
            num_quantiles=num_quantiles,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
        )

        data, state_dim, action_dim = training_data["data"], training_data["state_dim"], training_data["action_dim"]

        # 预处理数据
        training_data = model.preprocess_data(data)

        # 构建模型
        model.build_model(state_dim, action_dim)
        
        # 训练模型
        training_history = model.train(training_data)
        
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
        strategy_pool_size = model_config.get('strategy_pool_size', 20)
        cql_penalty_weight = model_config.get('cql_penalty_weight', 0.7)
        exposure_ratio = model_config.get('exposure_ratio', 0.6)
        importance_beta = model_config.get('importance_beta', 0.5)
        num_iterations = model_config.get('num_iterations', 50)
        
        # 创建模型
        model = OfflineFSPLearner(
            strategy_pool_size=strategy_pool_size,
            cql_penalty_weight=cql_penalty_weight,
            exposure_ratio=exposure_ratio,
            importance_beta=importance_beta
        )
        
        data, state_dim, action_dim = training_data["data"], training_data["state_dim"], training_data["action_dim"]
        model.build_network(state_dim, action_dim)
        # 预处理数据
        training_data = model.build_weighted_dataset(data)
        # 训练模型
        training_history = model.fictitious_play(training_data, num_iterations=num_iterations)
        
        # 提取训练指标
        metrics = {
            'q_loss': training_history.get('q_loss', []),
            'policy_loss': training_history.get('policy_loss', []),
            'opponent_loss': training_history.get('opponent_loss', []),
            'exploitability': training_history.get('exploitability', []),
        }
        
        self.logger.info(f"离线虚构自我博弈模型训练完成")
        
        return model, metrics
    
    def _train_adversial_imitation_learner(self, training_data: Dict, model_config: Dict) -> Tuple[AdversarialImitationLearner, Dict]:
        """训练对抗模仿学习模型"""
        self.logger.info("训练对抗模仿学习模型")
        
        # 提取模型参数
        gen_learning_rate = model_config.get('gen_learning_rate', 1e-4)
        disc_learning_rate = model_config.get('disc_learning_rate', 5e-5)
        update_ratio = model_config.get('update_ratio', 5)
        gp_lambda = model_config.get('gp_lambda', 10.0)
        batch_size = model_config.get('batch_size', 64)
        epochs = model_config.get('epochs', 100)
        
        # 创建模型
        model = AdversarialImitationLearner(
            gen_learning_rate=gen_learning_rate,
            disc_learning_rate=disc_learning_rate,
            update_ratio=update_ratio,
            gp_lambda=gp_lambda,
            device=self.device
        )

        data, state_dim, action_dim = training_data["data"], training_data["state_dim"], training_data["action_dim"]
        model.build_models(state_dim, action_dim)
        # 处理数据
        expert_states, expert_actions = model.preprocess_data(data)
        
        # 准备训练数据
        expert_data = {
            'states': expert_states,
            'actions': expert_actions
        }
        
        # 训练模型
        training_history = model.train(expert_data, batch_size=batch_size, epochs=epochs)
        
        # 提取训练指标
        metrics = {
            'gen_loss': training_history.get('gen_loss', []),
            'disc_loss': training_history.get('disc_loss', []),
            'wasserstein_dist': training_history.get('wasserstein_dist', []),
            'final_gen_loss': training_history.get('gen_loss', [-1])[-1] if training_history.get('gen_loss') else None,
            'final_disc_loss': training_history.get('disc_loss', [-1])[-1] if training_history.get('disc_loss') else None,
            'final_wasserstein_dist': training_history.get('wasserstein_dist', [-1])[-1] if training_history.get('wasserstein_dist') else None
        }
        
        self.logger.info(f"对抗模仿学习模型训练完成，最终生成器损失: {metrics['final_gen_loss']:.4f}, 判别器损失: {metrics['final_disc_loss']:.4f}, Wasserstein距离: {metrics['final_wasserstein_dist']:.4f}")
        
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
                metrics = model.evaluate_equilibrium(test_data)
            elif isinstance(model, AdversarialImitationLearner):
                # 评估对抗模仿学习模型
                metrics = model.evaluate(test_data.get('test_env'), num_episodes=eval_config.get('num_episodes', 10))
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
                    state_tensor = torch.FloatTensor(state)
                    # 预测
                    with torch.no_grad():
                        action = model.model(state_tensor).numpy()
                    return action
                else:
                    raise ValueError("模型未初始化")
            elif isinstance(model, OfflineRLearner):
                # 使用离线强化学习模型预测
                state_tensor = torch.FloatTensor(state)
                return model.policy.step(state_tensor, 'cpu')
            elif isinstance(model, OfflineFSPLearner):
                # 使用离线虚构自我博弈模型预测
                state_tensor = torch.FloatTensor(state)
                return model.policy_network(state_tensor)
            elif isinstance(model, AdversarialImitationLearner):
                # 使用对抗模仿学习模型预测
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = model.generator(state_tensor).cpu().numpy()[0]
                return action
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
            elif model_type == 'AdversarialImitationLearner':
                model = AdversarialImitationLearner(**(model_config or {}))
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