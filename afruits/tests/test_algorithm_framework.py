import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from afruits.core.api import AlgorithmAPI

class TestAlgorithmFramework(unittest.TestCase):
    """
    算法框架测试类
    
    测试小样本博弈建模和专家轨迹模仿学习功能
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
        
    #---------- 小样本博弈建模模块测试 ----------#
    
    def test_behavior_cloner(self):
        """测试行为克隆功能"""
        print("\n测试行为克隆功能 (BehaviorCloner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'BehaviorCloner',  # 使用行为克隆模型
            'batch_size': 32,
            'network_type': 'MLP',
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'dropout_rate': 0.2,
            'context_frames': 4,
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_game_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"训练损失: {training_metrics['final_train_loss']:.4f}")
        print(f"验证准确率: {training_metrics['final_val_accuracy']:.4f}")
        
        # 测试预测功能
        state = np.random.rand(4, 10)  # 假设状态是4个时间步，每个时间步10个特征
        action = self.api.game_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_game_modeling_test_data()
        eval_config = {
            'method': 'offline',
            'method_type': 'IS'
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('action_accuracy', eval_result)
        print(f"动作准确率: {eval_result['action_accuracy']:.4f}")
    
    def test_adversarial_imitation_learner(self):
        """测试对抗模仿学习功能"""
        print("\n测试对抗模仿学习功能 (AdversarialImitationLearner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'AdversarialImitationLearner',  # 使用对抗模仿学习模型
            'gen_hidden_dim': 64,
            'disc_hidden_dim': 64,
            'gen_learning_rate': 1e-4,
            'disc_learning_rate': 1e-5,
            'batch_size': 32,
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'gp_lambda': 10.0,  # 梯度惩罚系数
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_game_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"生成器损失: {training_metrics.get('final_gen_loss', 'N/A')}")
        print(f"判别器损失: {training_metrics.get('final_disc_loss', 'N/A')}")
        
        # 测试预测功能
        state = np.random.rand(4, 10)  # 假设状态是4个时间步，每个时间步10个特征
        action = self.api.game_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_game_modeling_test_data()
        eval_config = {
            'method': 'offline',
            'method_type': 'IS'
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('action_accuracy', eval_result)
        print(f"动作准确率: {eval_result['action_accuracy']:.4f}")
    
    def test_offline_r_learner(self):
        """测试离线强化学习功能"""
        print("\n测试离线强化学习功能 (OfflineRLearner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'OfflineRLearner',  # 使用离线强化学习模型
            'algorithm': 'CQL',  # 使用保守Q学习算法
            'hidden_dim': 64,
            'learning_rate': 3e-4,
            'batch_size': 32,
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'cql_weight': 0.5,  # CQL正则化权重
            'target_update_interval': 5,
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_game_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"Q损失: {training_metrics.get('final_q_loss', 'N/A')}")
        print(f"CQL损失: {training_metrics.get('final_cql_loss', 'N/A')}")
        
        # 测试预测功能
        state = np.random.rand(1, 10)  # 假设状态是1个时间步，10个特征
        action = self.api.game_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_game_modeling_test_data()
        eval_config = {
            'method': 'offline',
            'method_type': 'IS'
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('action_accuracy', eval_result)
        print(f"动作准确率: {eval_result['action_accuracy']:.4f}")
    
    def test_offline_fsp_learner(self):
        """测试离线自对弈功能"""
        print("\n测试离线自对弈功能 (OfflineFSPLearner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'OfflineFSPLearner',  # 使用离线自对弈模型
            'br_hidden_dim': 64,
            'avg_hidden_dim': 64,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'br_update_interval': 5,
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_game_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"BR损失: {training_metrics.get('final_br_loss', 'N/A')}")
        print(f"平均策略损失: {training_metrics.get('final_avg_loss', 'N/A')}")
        
        # 测试预测功能
        state = np.random.rand(1, 10)  # 假设状态是1个时间步，10个特征
        action = self.api.game_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_game_modeling_test_data()
        eval_config = {
            'method': 'offline',
            'method_type': 'IS'
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('action_accuracy', eval_result)
        print(f"动作准确率: {eval_result['action_accuracy']:.4f}")
    
    #---------- 小样本专家轨迹模仿学习模块测试 ----------#
    
    def test_autoencoder_model(self):
        """测试自编码器模型功能"""
        print("\n测试自编码器模型功能 (AutoencoderModel)")
        
        # 创建专家轨迹数据
        expert_trajectories = self.create_expert_trajectories()
        
        # 配置自编码器模型
        autoencoder_config = {
            'model_type': 'AutoencoderModel',  # 使用自编码器模型
            'training_method': 'standard',     # 标准训练方法
            'encoder_type': 'mlp',
            'input_dim': 10,                   # 输入维度
            'latent_dim': 32,                  # 潜在空间维度
            'hidden_dims': [64, 32],           # 隐藏层维度
            'dropout_rate': 0.1,               # Dropout比率
            'epochs': 5,                       # 训练轮数（减少以加快测试）
            'batch_size': 16,                  # 批次大小
            'learning_rate': 1e-4              # 学习率
        }
        
        # 训练模型
        autoencoder_result = self.api.train_imitation_model(expert_trajectories, autoencoder_config)
        
        # 验证结果
        self.assertIn('model_id', autoencoder_result)
        self.assertIn('model', autoencoder_result)
        self.assertIn('training_metrics', autoencoder_result)
        
        # 提取模型ID和训练指标
        autoencoder_model_id = autoencoder_result['model_id']
        autoencoder_metrics = autoencoder_result['training_metrics']
        
        print(f"自编码器模型ID: {autoencoder_model_id}")
        print(f"最终训练损失: {autoencoder_metrics.get('final_train_loss', 'N/A')}")
        print(f"最终验证损失: {autoencoder_metrics.get('final_val_loss', 'N/A')}")
        
        # 测试轨迹生成功能
        input_seq = np.random.rand(1, 10, 10)  # 批次大小为1，序列长度为10，特征维度为10
        input_seq_tensor = torch.FloatTensor(input_seq)
        
        # 生成轨迹
        context = {'input_seq': input_seq_tensor}
        config = {'pred_steps': 5}  # 预测5个步骤
        
        trajectory = self.api.imitation_learning_service.generate_trajectory(
            autoencoder_model_id, context, config
        )
        
        # 验证生成的轨迹
        self.assertIn('trajectory', trajectory)
        print(f"生成轨迹形状: {trajectory['trajectory'].shape}")
    
    def test_transformer_model(self):
        """测试Transformer模型功能"""
        print("\n测试Transformer模型功能 (TransformerModel)")
        
        # 创建专家轨迹数据
        expert_trajectories = self.create_expert_trajectories()
        
        # 配置Transformer模型
        transformer_config = {
            'model_type': 'TransformerModel',  # 使用Transformer模型
            'training_method': 'standard',     # 标准训练方法
            'encoder_type': 'str',
            'input_dim': 10,                   # 输入维度
            'd_model': 64,                     # 模型隐藏层维度
            'num_heads': 4,                    # 注意力头数量
            'num_layers': 2,                   # Transformer层数
            'max_seq_len': 50,                 # 最大序列长度
            'dropout_rate': 0.1,               # Dropout比率
            'epochs': 5,                       # 训练轮数（减少以加快测试）
            'batch_size': 16,                  # 批次大小
            'learning_rate': 1e-4              # 学习率
        }
        
        # 训练模型
        transformer_result = self.api.train_imitation_model(expert_trajectories, transformer_config)
        
        # 验证结果
        self.assertIn('model_id', transformer_result)
        self.assertIn('model', transformer_result)
        self.assertIn('training_metrics', transformer_result)
        
        # 提取模型ID和训练指标
        transformer_model_id = transformer_result['model_id']
        transformer_metrics = transformer_result['training_metrics']
        
        print(f"Transformer模型ID: {transformer_model_id}")
        print(f"最终训练损失: {transformer_metrics.get('final_train_loss', 'N/A')}")
        print(f"最终验证损失: {transformer_metrics.get('final_val_loss', 'N/A')}")
        
        # 测试轨迹生成功能
        input_seq = np.random.rand(1, 10, 10)  # 批次大小为1，序列长度为10，特征维度为10
        input_seq_tensor = torch.FloatTensor(input_seq)
        
        # 生成轨迹
        context = {'input_seq': input_seq_tensor}
        config = {'pred_steps': 5}  # 预测5个步骤
        
        trajectory = self.api.imitation_learning_service.generate_trajectory(
            transformer_model_id, context, config
        )
        
        # 验证生成的轨迹
        self.assertIn('trajectory', trajectory)
        print(f"生成轨迹形状: {trajectory['trajectory'].shape}")
    
    def test_diffusion_traj_generator(self):
        """测试扩散轨迹生成器功能"""
        print("\n测试扩散轨迹生成器功能 (DiffusionTrajGenerator)")
        
        # 创建专家轨迹数据
        expert_trajectories = self.create_expert_trajectories()
        
        # 配置扩散轨迹生成器模型
        diffusion_config = {
            'model_type': 'DiffusionTrajGenerator',  # 使用扩散轨迹生成器模型
            'training_method': 'standard',           # 标准训练方法
            'input_dim': 10,                         # 输入维度
            'hidden_dim': 64,                        # 隐藏层维度
            'num_layers': 3,                         # 网络层数
            'noise_steps': 50,                       # 噪声步数
            'beta_start': 1e-4,                      # β起始值
            'beta_end': 0.02,                        # β结束值
            'epochs': 5,                             # 训练轮数（减少以加快测试）
            'batch_size': 16,                        # 批次大小
            'learning_rate': 1e-4                    # 学习率
        }
        
        # 训练模型
        diffusion_result = self.api.train_imitation_model(expert_trajectories, diffusion_config)
        
        # 验证结果
        self.assertIn('model_id', diffusion_result)
        self.assertIn('model', diffusion_result)
        self.assertIn('training_metrics', diffusion_result)
        
        # 提取模型ID和训练指标
        diffusion_model_id = diffusion_result['model_id']
        diffusion_metrics = diffusion_result['training_metrics']
        
        print(f"扩散模型ID: {diffusion_model_id}")
        print(f"最终训练损失: {diffusion_metrics.get('final_train_loss', 'N/A')}")
        
        # 测试轨迹生成功能
        # 生成轨迹
        context = {'batch_size': 1, 'seq_length': 10, 'feature_dim': 10}
        config = {'num_samples': 1}  # 生成1个样本
        
        trajectory = self.api.imitation_learning_service.generate_trajectory(
            diffusion_model_id, context, config
        )
        
        # 验证生成的轨迹
        self.assertIn('trajectory', trajectory)
        print(f"生成轨迹形状: {trajectory['trajectory'].shape}")
    
    def test_vae_traj_generator(self):
        """测试VAE轨迹生成器功能"""
        print("\n测试VAE轨迹生成器功能 (VAETrajGenerator)")
        
        # 创建专家轨迹数据
        expert_trajectories = self.create_expert_trajectories()
        
        # 配置VAE轨迹生成器模型
        vae_config = {
            'model_type': 'VAETrajGenerator',  # 使用VAE轨迹生成器模型
            'training_method': 'standard',     # 标准训练方法
            'input_dim': 10,                   # 输入维度
            'latent_dim': 32,                  # 潜在空间维度
            'seq_length': 30,                  # 序列长度
            'kl_weight': 0.001,                # KL散度权重
            'recon_loss_type': 'mse',          # 重构损失类型
            'epochs': 5,                       # 训练轮数（减少以加快测试）
            'batch_size': 16,                  # 批次大小
            'learning_rate': 1e-4              # 学习率
        }
        
        # 训练模型
        vae_result = self.api.train_imitation_model(expert_trajectories, vae_config)
        
        # 验证结果
        self.assertIn('model_id', vae_result)
        self.assertIn('model', vae_result)
        self.assertIn('training_metrics', vae_result)
        
        # 提取模型ID和训练指标
        vae_model_id = vae_result['model_id']
        vae_metrics = vae_result['training_metrics']
        
        print(f"VAE模型ID: {vae_model_id}")
        print(f"最终训练损失: {vae_metrics.get('final_train_loss', 'N/A')}")
        print(f"KL散度: {vae_metrics.get('final_kl_divergence', 'N/A')}")
        
        # 测试轨迹生成功能
        # 生成轨迹
        context = {'batch_size': 1, 'latent_dim': 32}
        config = {'num_samples': 1}  # 生成1个样本
        
        trajectory = self.api.imitation_learning_service.generate_trajectory(
            vae_model_id, context, config
        )
        
        # 验证生成的轨迹
        self.assertIn('trajectory', trajectory)
        print(f"生成轨迹形状: {trajectory['trajectory'].shape}")
    
    #---------- 训练方法模型测试 ----------#
    
    def test_evolutionary_learner(self):
        """测试进化学习器功能"""
        print("\n测试进化学习器功能 (EvolutionaryLearner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'EvolutionaryLearner',  # 使用进化学习器模型
            'population_size': 10,                # 减少种群大小以加快测试
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'selection_method': 'tournament',
            'elitism_ratio': 0.1,
            'max_generations': 3,                 # 减少代数以加快测试
            'base_policy': 'BehaviorCloner'       # 基础策略类型
        }
        
        # 训练模型
        result = self.api.train_game_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"最佳适应度: {training_metrics.get('best_fitness', 'N/A')}")
        print(f"最终种群大小: {training_metrics.get('population_size', 'N/A')}")
        
        # 测试预测功能
        state = np.random.rand(1, 10)  # 假设状态是1个时间步，10个特征
        action = self.api.game_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_game_modeling_test_data()
        eval_config = {
            'method': 'offline',
            'method_type': 'IS'
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('action_accuracy', eval_result)
        print(f"动作准确率: {eval_result['action_accuracy']:.4f}")
    
    def test_incremental_learner(self):
        """测试增量学习器功能"""
        print("\n测试增量学习器功能 (IncrementalLearner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'IncrementalLearner',  # 使用增量学习器模型
            'base_model_type': 'BehaviorCloner', # 基础模型类型
            'memory_size': 1000,                 # 记忆库大小
            'batch_size': 32,
            'learning_rate': 1e-4,
            'replay_ratio': 0.3,                 # 重放比例
            'max_epochs': 5,                     # 减少训练轮数以加快测试
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_game_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"训练损失: {training_metrics.get('final_train_loss', 'N/A')}")
        print(f"验证准确率: {training_metrics.get('final_val_accuracy', 'N/A')}")
        
        # 测试预测功能
        state = np.random.rand(1, 10)  # 假设状态是1个时间步，10个特征
        action = self.api.game_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_game_modeling_test_data()
        eval_config = {
            'method': 'offline',
            'method_type': 'IS'
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('action_accuracy', eval_result)
        print(f"动作准确率: {eval_result['action_accuracy']:.4f}")
    
    def test_fine_tune_manager(self):
        """测试微调管理器功能"""
        print("\n测试微调管理器功能 (FineTuneManager)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'FineTuneManager',     # 使用微调管理器模型
            'base_model_type': 'BehaviorCloner', # 基础模型类型
            'learning_rate': 5e-5,               # 微调学习率
            'batch_size': 16,
            'max_epochs': 5,                     # 减少训练轮数以加快测试
            'freeze_layers': ['encoder.0'],      # 冻结层
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_game_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"训练损失: {training_metrics.get('final_train_loss', 'N/A')}")
        print(f"验证准确率: {training_metrics.get('final_val_accuracy', 'N/A')}")
        
        # 测试预测功能
        state = np.random.rand(1, 10)  # 假设状态是1个时间步，10个特征
        action = self.api.game_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_game_modeling_test_data()
        eval_config = {
            'method': 'offline',
            'method_type': 'IS'
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
