import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
import time
import copy

# 导入轨迹建模与生成模型
from utils.AutoencoderModel import AutoencoderModel
from utils.TransformerModel import TransformerModel
from utils.DiffusionTrajGenerator import DiffusionTrajGenerator
from utils.VAETrajGenerator import VAETrajGenerator

# 导入训练方法模块
from utils.EvolutionaryLearner import EvolutionaryLearner
from utils.IncrementalLearner import IncrementalLearner
from utils.FineTuneManager import FineTuneManager

# 导入模仿学习模块
from utils.AdversarialImitationLearner import AdversarialImitationLearner

class ImitationLearningService:
    """
    小样本专家轨迹模仿学习服务类
    
    负责小样本专家轨迹模仿学习相关的功能，包括模型训练、评估和预测
    """
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        初始化小样本专家轨迹模仿学习服务
        
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
        
        # 初始化训练方法
        self.training_methods = {}
        
        self.logger.info(f"小样本专家轨迹模仿学习服务初始化完成，使用设备: {self.device}")
    
    def train_model(self, expert_trajectories: Dict, model_config: Dict) -> Dict:
        """
        训练小样本专家轨迹模仿学习模型
        
        参数:
            expert_trajectories (Dict): 专家轨迹数据
            model_config (Dict): 模型配置
            
        返回:
            Dict: 训练结果，包含模型和训练指标
        """
        # 获取模型类型
        model_type = model_config.get('model_type', 'TransformerModel')
        model_id = model_config.get('model_id', f"{model_type}_{int(time.time())}")
        
        # 获取训练方法
        training_method = model_config.get('training_method', 'standard')
        
        self.logger.info(f"开始训练模型: {model_id}, 类型: {model_type}, 训练方法: {training_method}")
        
        # 初始化结果字典
        result = {
            'model_id': model_id,
            'model_type': model_type,
            'training_method': training_method,
            'training_metrics': {},
            'model': None
        }
        
        try:
            # 根据模型类型和训练方法选择不同的训练流程
            if training_method == 'standard':
                # 标准训练方法
                model, metrics = self._train_standard(model_type, expert_trajectories, model_config)
            elif training_method == 'evolutionary':
                # 进化学习方法
                model, metrics = self._train_evolutionary(model_type, expert_trajectories, model_config)
            elif training_method == 'incremental':
                # 增量学习方法
                model, metrics = self._train_incremental(model_type, expert_trajectories, model_config)
            elif training_method == 'fine_tune':
                # 微调方法
                model, metrics = self._train_fine_tune(model_type, expert_trajectories, model_config)
            else:
                raise ValueError(f"不支持的训练方法: {training_method}")
            
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
    
    def _train_standard(self, model_type: str, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """标准训练方法"""
        self.logger.info(f"使用标准训练方法训练 {model_type} 模型")
        
        # 根据模型类型选择不同的训练函数
        if model_type == 'AutoencoderModel':
            return self._train_autoencoder(expert_trajectories, model_config)
        elif model_type == 'TransformerModel':
            return self._train_transformer(expert_trajectories, model_config)
        elif model_type == 'DiffusionTrajGenerator':
            return self._train_diffusion(expert_trajectories, model_config)
        elif model_type == 'VAETrajGenerator':
            return self._train_vae(expert_trajectories, model_config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _train_autoencoder(self, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """自编码器模型训练方法"""
        self.logger.info("使用自编码器训练方法")
        
        # 创建模型
        model = self._create_model('AutoencoderModel', model_config)
        
        # 提取训练参数
        epochs = model_config.get('epochs', 100)
        batch_size = model_config.get('batch_size', 32)
        learning_rate = model_config.get('learning_rate', 1e-4)
        
        # 准备数据加载器
        data_loader = model.load_sequences(expert_trajectories, batch_size=batch_size)
        
        # 训练模型
        training_history = model.train(data_loader, epochs=epochs, learning_rate=learning_rate)
        
        # 提取训练指标
        metrics = {
            'train_loss': training_history.get('train_loss', []),
            'val_loss': training_history.get('val_loss', []),
            'final_train_loss': training_history.get('train_loss', [-1])[-1] if training_history.get('train_loss') else None,
            'final_val_loss': training_history.get('val_loss', [-1])[-1] if training_history.get('val_loss') else None
        }
        
        self.logger.info(f"自编码器训练完成，最终训练损失: {metrics['final_train_loss']}, 验证损失: {metrics['final_val_loss']}")
        
        return model, metrics
    
    def _train_transformer(self, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """Transformer模型训练方法"""
        self.logger.info("使用Transformer训练方法")
        
        # 创建模型
        model = self._create_model('TransformerModel', model_config)
        
        # 提取训练参数
        epochs = model_config.get('epochs', 100)
        batch_size = model_config.get('batch_size', 32)
        learning_rate = model_config.get('learning_rate', 1e-4)
        
        # 准备数据
        # 注意：TransformerModel的load_sequences方法可能需要特殊处理
        # 这里假设数据已经是正确的格式，或者在模型内部进行了处理
        data_loader = model.load_sequences(expert_trajectories, batch_size=batch_size)
        
        # 训练模型
        training_history = model.train(data_loader, epochs=epochs, learning_rate=learning_rate)
        
        # 提取训练指标
        metrics = {
            'train_loss': training_history.get('train_loss', []),
            'val_loss': training_history.get('val_loss', []),
            'final_train_loss': training_history.get('train_loss', [-1])[-1] if training_history.get('train_loss') else None,
            'final_val_loss': training_history.get('val_loss', [-1])[-1] if training_history.get('val_loss') else None
        }
        
        self.logger.info(f"Transformer训练完成，最终训练损失: {metrics['final_train_loss']}, 验证损失: {metrics['final_val_loss']}")
        
        return model, metrics
    
    def _train_diffusion(self, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """扩散轨迹生成器训练方法"""
        self.logger.info("使用扩散模型训练方法")
        
        # 创建模型
        model = self._create_model('DiffusionTrajGenerator', model_config)
        
        # 提取训练参数
        epochs = model_config.get('epochs', 100)
        batch_size = model_config.get('batch_size', 32)
        
        # 准备数据
        # DiffusionTrajGenerator需要特殊的数据处理
        # 首先检查数据是否已经是DataLoader格式
        if 'dataloader' in expert_trajectories:
            data_loader = expert_trajectories['dataloader']
        else:
            # 否则，使用模型的load_dataset方法加载数据
            # 这里假设expert_trajectories中包含data_path
            data_path = expert_trajectories.get('data_path')
            if not data_path:
                # 如果没有data_path，可能需要先将数据保存到临时文件
                import tempfile
                import numpy as np
                
                temp_dir = tempfile.mkdtemp()
                data_path = f"{temp_dir}/temp_trajectories.npz"
                np.savez(data_path, trajectories=expert_trajectories.get('trajectories', []))
            
            data_info = model.load_dataset(data_path, seq_len=model_config.get('seq_length'))
            data_loader = data_info['dataloader']
        
        # 确保模型网络已构建
        if not hasattr(model, 'model') or model.model is None:
            input_dim = expert_trajectories.get('feature_dim', model_config.get('input_dim', 32))
            cond_dim = model_config.get('cond_dim', 0)
            model.build_network(input_dim, cond_dim)
        
        # 训练模型
        training_history = model.train(data_loader, epochs=epochs)
        
        # 提取训练指标
        metrics = {
            'loss_curve': training_history.get('loss_curve', []),
            'final_loss': training_history.get('final_loss'),
            'epochs': training_history.get('epochs')
        }
        
        self.logger.info(f"扩散模型训练完成，最终损失: {metrics['final_loss']}")
        
        return model, metrics
    
    def _train_vae(self, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """VAE轨迹生成器训练方法"""
        self.logger.info("使用VAE训练方法")
        
        # 创建模型
        model = self._create_model('VAETrajGenerator', model_config)
        
        # 提取训练参数
        epochs = model_config.get('epochs', 100)
        batch_size = model_config.get('batch_size', 32)
        
        # 准备数据
        # VAETrajGenerator需要特殊的数据处理
        # 首先检查数据是否已经是DataLoader格式
        if 'dataloader' in expert_trajectories:
            data_loader = expert_trajectories['dataloader']
        else:
            # 否则，使用模型的load_dataset方法加载数据
            # 这里假设expert_trajectories中包含data_path
            data_path = expert_trajectories.get('data_path')
            if not data_path:
                # 如果没有data_path，可能需要先将数据保存到临时文件
                import tempfile
                import numpy as np
                
                temp_dir = tempfile.mkdtemp()
                data_path = f"{temp_dir}/temp_trajectories.npz"
                np.savez(data_path, trajectories=expert_trajectories.get('trajectories', []))
            
            data_info = model.load_dataset(data_path, batch_size=batch_size)
            data_loader = data_info['dataloader']
        
        # 确保模型网络已构建
        if not hasattr(model, 'encoder') or model.encoder is None:
            input_dim = expert_trajectories.get('feature_dim', model_config.get('input_dim', 32))
            model.build_model(input_dim)
        
        # 训练模型
        training_history = model.train(data_loader, epochs=epochs)
        
        # 提取训练指标
        metrics = {
            'total_loss': training_history.get('total_loss', []),
            'kl_divergence': training_history.get('kl_divergence', []),
            'recon_error': training_history.get('recon_error', []),
            'final_total_loss': training_history.get('total_loss', [-1])[-1] if training_history.get('total_loss') else None,
            'final_kl_divergence': training_history.get('kl_divergence', [-1])[-1] if training_history.get('kl_divergence') else None,
            'final_recon_error': training_history.get('recon_error', [-1])[-1] if training_history.get('recon_error') else None
        }
        
        self.logger.info(f"VAE训练完成，最终总损失: {metrics['final_total_loss']}, KL散度: {metrics['final_kl_divergence']}, 重构误差: {metrics['final_recon_error']}")
        
        return model, metrics
    
    def _train_evolutionary(self, model_type: str, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """进化学习方法"""
        self.logger.info(f"使用进化学习方法训练 {model_type} 模型")
        
        # 提取进化学习参数
        population_size = model_config.get('population_size', 50)
        mutation_rate = model_config.get('mutation_rate', 0.15)
        crossover_rate = model_config.get('crossover_rate', 0.7)
        selection_method = model_config.get('selection_method', 'tournament')
        elitism_ratio = model_config.get('elitism_ratio', 0.1)
        max_generations = model_config.get('max_generations', 50)
        fitness_threshold = model_config.get('fitness_threshold', 0.95)
        
        # 创建进化学习器
        evolutionary_learner = EvolutionaryLearner(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            selection_method=selection_method,
            elitism_ratio=elitism_ratio
        )
        
        # 创建模型模板
        model_template = self._create_model(model_type, model_config)
        
        # 创建混合种群
        population = evolutionary_learner.create_mixed_population(expert_trajectories)
        
        # 如果种群为空，使用模板初始化种群
        if not population:
            population = evolutionary_learner.initialize_population(model_template)
        
        # 训练种群
        training_result = evolutionary_learner.train_population(expert_trajectories)
        
        # 创建评估环境
        eval_env = self._create_eval_env(model_config)
        
        # 运行进化
        evolution_result = evolutionary_learner.run_evolution(
            max_generations=max_generations,
            fitness_threshold=fitness_threshold,
            eval_env=eval_env
        )
        
        # 获取最佳模型
        best_model = evolution_result.get('best_policy')
        
        # 提取训练指标
        metrics = {
            'mean_fitness': evolution_result.get('mean_fitness', []),
            'max_fitness': evolution_result.get('max_fitness', []),
            'min_fitness': evolution_result.get('min_fitness', []),
            'diversity': evolution_result.get('diversity', []),
            'final_mean_fitness': evolution_result.get('mean_fitness', [-1])[-1] if evolution_result.get('mean_fitness') else None,
            'final_max_fitness': evolution_result.get('max_fitness', [-1])[-1] if evolution_result.get('max_fitness') else None,
            'generations': evolution_result.get('generations', 0)
        }
        
        self.logger.info(f"进化学习完成，最终最大适应度: {metrics['final_max_fitness']}, 平均适应度: {metrics['final_mean_fitness']}, 代数: {metrics['generations']}")
        
        # 保存进化学习器
        learner_id = f"evolutionary_{int(time.time())}"
        self.training_methods[learner_id] = evolutionary_learner
        
        return best_model, metrics
    
    def _train_incremental(self, model_type: str, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """增量学习方法"""
        self.logger.info(f"使用增量学习方法训练 {model_type} 模型")
        
        # 提取增量学习参数
        base_model_id = model_config.get('base_model_id')
        memory_size = model_config.get('memory_size', 1000)
        replay_ratio = model_config.get('replay_ratio', 0.3)
        learning_rate = model_config.get('learning_rate', 1e-4)
        epochs = model_config.get('epochs', 50)
        
        # 创建增量学习器
        incremental_learner = IncrementalLearner(
            memory_size=memory_size,
            replay_ratio=replay_ratio,
            learning_rate=learning_rate
        )
        
        # 获取基础模型
        if base_model_id and base_model_id in self.models:
            base_model = self.models[base_model_id]
        else:
            # 创建新模型
            base_model = self._create_model(model_type, model_config)
        
        # 设置基础模型
        incremental_learner.set_base_model(base_model)
        
        # 训练模型
        training_result = incremental_learner.train(expert_trajectories, epochs=epochs)
        
        # 获取更新后的模型
        updated_model = incremental_learner.get_model()
        
        # 提取训练指标
        metrics = {
            'train_loss': training_result.get('train_loss', []),
            'val_loss': training_result.get('val_loss', []),
            'catastrophic_forgetting': training_result.get('catastrophic_forgetting', []),
            'final_train_loss': training_result.get('train_loss', [-1])[-1] if training_result.get('train_loss') else None,
            'final_val_loss': training_result.get('val_loss', [-1])[-1] if training_result.get('val_loss') else None,
            'final_catastrophic_forgetting': training_result.get('catastrophic_forgetting', [-1])[-1] if training_result.get('catastrophic_forgetting') else None
        }
        
        self.logger.info(f"增量学习完成，最终训练损失: {metrics['final_train_loss']}, 验证损失: {metrics['final_val_loss']}, 灾难性遗忘: {metrics['final_catastrophic_forgetting']}")
        
        # 保存增量学习器
        learner_id = f"incremental_{int(time.time())}"
        self.training_methods[learner_id] = incremental_learner
        
        return updated_model, metrics
    
    def _train_fine_tune(self, model_type: str, expert_trajectories: Dict, model_config: Dict) -> Tuple[Any, Dict]:
        """微调方法"""
        self.logger.info(f"使用微调方法训练 {model_type} 模型")
        
        # 提取微调参数
        base_model_id = model_config.get('base_model_id')
        learning_rate = model_config.get('learning_rate', 5e-5)
        epochs = model_config.get('epochs', 20)
        freeze_layers = model_config.get('freeze_layers', [])
        
        # 创建微调管理器
        fine_tune_manager = FineTuneManager(
            learning_rate=learning_rate
        )
        
        # 获取基础模型
        if base_model_id and base_model_id in self.models:
            base_model = self.models[base_model_id]
        else:
            # 创建新模型
            base_model = self._create_model(model_type, model_config)
        
        # 设置基础模型
        fine_tune_manager.set_model(base_model)
        
        # 冻结指定层
        if freeze_layers:
            fine_tune_manager.freeze_layers(freeze_layers)
        
        # 训练模型
        training_result = fine_tune_manager.fine_tune(expert_trajectories, epochs=epochs)
        
        # 获取微调后的模型
        fine_tuned_model = fine_tune_manager.get_model()
        
        # 提取训练指标
        metrics = {
            'train_loss': training_result.get('train_loss', []),
            'val_loss': training_result.get('val_loss', []),
            'final_train_loss': training_result.get('train_loss', [-1])[-1] if training_result.get('train_loss') else None,
            'final_val_loss': training_result.get('val_loss', [-1])[-1] if training_result.get('val_loss') else None
        }
        
        self.logger.info(f"微调完成，最终训练损失: {metrics['final_train_loss']}, 验证损失: {metrics['final_val_loss']}")
        
        # 保存微调管理器
        manager_id = f"fine_tune_{int(time.time())}"
        self.training_methods[manager_id] = fine_tune_manager
        
        return fine_tuned_model, metrics
    
    def _create_model(self, model_type: str, model_config: Dict) -> Any:
        """创建模型"""
        self.logger.info(f"创建模型: {model_type}")
        
        # 根据模型类型创建模型
        if model_type == 'AutoencoderModel':
            # 提取自编码器模型参数
            input_dim = model_config.get('input_dim', 32)
            hidden_dims = model_config.get('hidden_dims', [64, 128, 64])
            latent_dim = model_config.get('latent_dim', 16)
            
            # 创建自编码器模型
            model = AutoencoderModel(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim
            )
            
        elif model_type == 'TransformerModel':
            # 提取Transformer模型参数
            encoder_type = model_config.get('encoder_type', 'str')
            input_dim = model_config.get('input_dim', 32)
            d_model = model_config.get('d_model', 128)
            num_heads = model_config.get('num_heads', 4)
            num_layers = model_config.get('num_layers', 3)
            max_seq_len = model_config.get('max_seq_len', 100)
            dropout_rate = model_config.get('dropout_rate', 0.2)
            
            # 创建Transformer模型
            model = TransformerModel(
                encoder_type=encoder_type,
                input_dim=input_dim,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                dropout_rate=dropout_rate
            )
            
        elif model_type == 'DiffusionTrajGenerator':
            # 提取扩散轨迹生成器参数
            state_dim = model_config.get('state_dim', 32)
            action_dim = model_config.get('action_dim', 8)
            hidden_dim = model_config.get('hidden_dim', 128)
            num_diffusion_steps = model_config.get('num_diffusion_steps', 100)
            
            # 创建扩散轨迹生成器
            model = DiffusionTrajGenerator(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_diffusion_steps=num_diffusion_steps
            )
            
        elif model_type == 'VAETrajGenerator':
            # 提取VAE轨迹生成器参数
            input_dim = model_config.get('input_dim', 32)
            hidden_dim = model_config.get('hidden_dim', 128)
            latent_dim = model_config.get('latent_dim', 16)
            sequence_length = model_config.get('sequence_length', 50)
            
            # 创建VAE轨迹生成器
            model = VAETrajGenerator(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                sequence_length=sequence_length
            )
            
        elif model_type == 'AdversarialImitationLearner':
            # 提取对抗模仿学习器参数
            state_dim = model_config.get('state_dim', 32)
            action_dim = model_config.get('action_dim', 8)
            hidden_dim = model_config.get('hidden_dim', 128)
            
            # 创建对抗模仿学习器
            model = AdversarialImitationLearner(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            )
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model
    
    def _create_eval_env(self, model_config: Dict) -> Any:
        """创建评估环境"""
        # 这里应该根据实际需求创建评估环境
        # 简化示例：返回一个简单的评估环境
        class SimpleEvalEnv:
            def __init__(self):
                self.state = None
            
            def reset(self):
                self.state = np.zeros(model_config.get('state_dim', 32))
                return self.state
            
            def step(self, action):
                # 简单的状态转移
                self.state = np.random.randn(model_config.get('state_dim', 32)) * 0.1 + self.state
                reward = np.random.rand()
                done = np.random.rand() > 0.95
                info = {}
                return self.state, reward, done, info
        
        return SimpleEvalEnv()
    
    def generate_trajectory(self, model_id: str, context: Dict = None, config: Dict = None) -> Dict:
        """
        生成轨迹
        
        参数:
            model_id (str): 模型ID
            context (Dict): 上下文信息
            config (Dict): 生成配置
            
        返回:
            Dict: 生成的轨迹
        """
        self.logger.info(f"使用模型生成轨迹: {model_id}")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 根据模型类型选择不同的生成方法
            if isinstance(model, TransformerModel):
                # 使用Transformer模型生成轨迹
                input_seq = context.get('input_seq') if context else None
                pred_steps = config.get('pred_steps', 1) if config else 1
                
                if input_seq is None:
                    raise ValueError("缺少输入序列")
                
                # 生成轨迹
                result = model.predict(input_seq, pred_steps=pred_steps)
                
            elif isinstance(model, DiffusionTrajGenerator):
                # 使用扩散轨迹生成器生成轨迹
                initial_state = context.get('initial_state') if context else None
                horizon = config.get('horizon', 50) if config else 50
                
                if initial_state is None:
                    raise ValueError("缺少初始状态")
                
                # 生成轨迹
                result = model.generate(initial_state, horizon=horizon)
                
            elif isinstance(model, VAETrajGenerator):
                # 使用VAE轨迹生成器生成轨迹
                latent_code = context.get('latent_code') if context else None
                
                if latent_code is None:
                    # 随机采样潜在编码
                    latent_code = torch.randn(1, model.latent_dim)
                
                # 生成轨迹
                result = model.generate(latent_code)
                
            else:
                raise ValueError(f"不支持的模型类型: {type(model).__name__}")
            
            self.logger.info(f"轨迹生成完成: {model_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"轨迹生成失败: {str(e)}")
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
            # 创建模型
            model = self._create_model(model_type, model_config or {})
            
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
    
    def encode(self, model_id: str, input_data: Any) -> Any:
        """
        使用模型编码输入数据
        
        参数:
            model_id (str): 模型ID
            input_data (Any): 输入数据
            
        返回:
            Any: 编码结果
        """
        self.logger.info(f"使用模型 {model_id} 编码数据")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 确保输入数据是张量
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            else:
                input_data = input_data.to(self.device)
            
            # 根据模型类型选择不同的编码方法
            if isinstance(model, AutoencoderModel):
                # 使用自编码器模型编码
                with torch.no_grad():
                    encoded = model.encode(input_data)
                return encoded
            else:
                raise ValueError(f"不支持的模型类型: {type(model).__name__}")
            
        except Exception as e:
            self.logger.error(f"编码失败: {str(e)}")
            raise
    
    def decode(self, model_id: str, latent_code: Any) -> Any:
        """
        使用模型解码潜在编码
        
        参数:
            model_id (str): 模型ID
            latent_code (Any): 潜在编码
            
        返回:
            Any: 解码结果
        """
        self.logger.info(f"使用模型 {model_id} 解码数据")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 确保潜在编码是张量
            if not isinstance(latent_code, torch.Tensor):
                latent_code = torch.tensor(latent_code, dtype=torch.float32).to(self.device)
            else:
                latent_code = latent_code.to(self.device)
            
            # 根据模型类型选择不同的解码方法
            if isinstance(model, AutoencoderModel):
                # 使用自编码器模型解码
                with torch.no_grad():
                    decoded = model.decode(latent_code)
                return decoded
            else:
                raise ValueError(f"不支持的模型类型: {type(model).__name__}")
            
        except Exception as e:
            self.logger.error(f"解码失败: {str(e)}")
            raise
    
    def predict_next(self, model_id: str, input_seq: Any, steps: int = 1) -> Any:
        """
        预测下一个时间步
        
        参数:
            model_id (str): 模型ID
            input_seq (Any): 输入序列
            steps (int): 预测步数
            
        返回:
            Any: 预测结果
        """
        self.logger.info(f"使用模型 {model_id} 预测下一个时间步")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 确保输入序列是张量
            if not isinstance(input_seq, torch.Tensor):
                input_seq = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
            else:
                input_seq = input_seq.to(self.device)
            
            # 根据模型类型选择不同的预测方法
            if isinstance(model, TransformerModel):
                # 使用Transformer模型预测
                with torch.no_grad():
                    result = model.predict(input_seq, pred_steps=steps)
                return result.get('trajectory', None)
            else:
                raise ValueError(f"不支持的模型类型: {type(model).__name__}")
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            raise
    
    def generate(self, model_id: str, num_samples: int = 1, cond_vector: Any = None) -> Dict:
        """
        生成轨迹
        
        参数:
            model_id (str): 模型ID
            num_samples (int): 生成数量
            cond_vector (Any): 条件向量
            
        返回:
            Dict: 生成结果
        """
        self.logger.info(f"使用模型 {model_id} 生成轨迹")
        
        # 检查模型是否存在
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 获取模型
        model = self.models[model_id]
        
        try:
            # 确保条件向量是张量（如果提供）
            if cond_vector is not None and not isinstance(cond_vector, torch.Tensor):
                cond_vector = torch.tensor(cond_vector, dtype=torch.float32).to(self.device)
            elif cond_vector is not None:
                cond_vector = cond_vector.to(self.device)
            
            # 根据模型类型选择不同的生成方法
            if isinstance(model, DiffusionTrajGenerator):
                # 使用扩散轨迹生成器生成
                result = model.generate(batch_size=num_samples)
                return result
            elif isinstance(model, VAETrajGenerator):
                # 使用VAE轨迹生成器生成
                result = model.generate(num_samples=num_samples, cond_vector=cond_vector)
                return result
            else:
                raise ValueError(f"不支持的模型类型: {type(model).__name__}")
            
        except Exception as e:
            self.logger.error(f"生成失败: {str(e)}")
            raise