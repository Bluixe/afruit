import sys
import os
import numpy as np
import torch
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.EvolutionaryLearner import EvolutionaryLearner
from utils.AutoencoderModel import AutoencoderTrainer
from utils.TransformerModel import TransformerTrainer
from utils.DiffusionTrajGenerator import DiffusionTrajGenerator
from utils.VAETrajGenerator import VAETrajGenerator
from core.services.imitation_learning_service import ImitationLearningService

class TestEvolutionaryLearner(unittest.TestCase):
    
    def setUp(self):
        # 创建模拟数据
        self.mock_trajectories = {
            'data': [np.random.rand(10, 5) for _ in range(20)],  # 20个轨迹，每个轨迹10个时间步，5维状态
            'state_dim': 5,
            'action_dim': 2,
            'traj_length': 10
        }
        
        # 创建进化学习器
        self.evolutionary_learner = EvolutionaryLearner(
            population_size=3,  # 小规模种群用于测试
            mutation_rate=0.15,
            crossover_rate=0.7,
            selection_method="tournament",
            elitism_ratio=0.1,
            model_type="AutoencoderModel"
        )
    
    @patch('utils.AutoencoderModel.AutoencoderTrainer.train_model')
    def test_train_autoencoder(self, mock_train_model):
        # 设置模拟返回值
        mock_train_model.return_value = {'train_loss': [0.5, 0.4, 0.3]}
        
        # 创建模型
        model = AutoencoderTrainer(
            encoder_type='lstm',
            latent_dim=32,
            kl_weight=0.001,
            dropout_rate=0.2
        )
        model.build_model(5, 2, 10)
        model.id = "autoencoder_0"
        
        # 创建模拟数据加载器
        mock_loader = MagicMock()
        
        # 测试训练方法
        result = self.evolutionary_learner._train_autoencoder(model, mock_loader, epochs=3)
        
        # 验证结果
        self.assertEqual(result, {'train_loss': [0.5, 0.4, 0.3]})
        mock_train_model.assert_called_once_with(mock_loader, epochs=3, learning_rate=1e-4)
    
    @patch('utils.TransformerModel.TransformerTrainer.train_model')
    def test_train_transformer(self, mock_train_model):
        # 设置模拟返回值
        mock_train_model.return_value = {'train_loss': [0.6, 0.5, 0.4]}
        
        # 创建模型
        model = TransformerTrainer(
            d_model=128,
            num_heads=4,
            num_layers=3,
            max_seq_len=100,
            dropout_rate=0.2
        )
        model.build_model(5, 2)
        model.id = "transformer_0"
        
        # 创建模拟数据加载器
        mock_loader = MagicMock()
        
        # 测试训练方法
        result = self.evolutionary_learner._train_transformer(model, mock_loader, epochs=3)
        
        # 验证结果
        self.assertEqual(result, {'train_loss': [0.6, 0.5, 0.4]})
        mock_train_model.assert_called_once_with(mock_loader, epochs=3, learning_rate=1e-4)
    
    @patch('utils.DiffusionTrajGenerator.DiffusionTrajGenerator.train')
    def test_train_diffusion(self, mock_train):
        # 设置模拟返回值
        mock_train.return_value = {'loss_curve': [0.7, 0.6, 0.5]}
        
        # 创建模型
        model = DiffusionTrajGenerator(
            diffusion_steps=1000,
            noise_schedule='cosine',
            dropout=0.2,
            im_embd=128
        )
        model.build_model(5)
        model.id = "diffusion_0"
        
        # 创建模拟数据加载器
        mock_loader = MagicMock()
        
        # 测试训练方法
        result = self.evolutionary_learner._train_diffusion(model, mock_loader, epochs=3)
        
        # 验证结果
        self.assertEqual(result, {'loss_curve': [0.7, 0.6, 0.5]})
        mock_train.assert_called_once_with(mock_loader, epochs=3, learning_rate=1e-4)
    
    @patch('utils.VAETrajGenerator.VAETrajGenerator.train')
    def test_train_vae(self, mock_train):
        # 设置模拟返回值
        mock_train.return_value = {'total_loss': [0.8, 0.7, 0.6]}
        
        # 创建模型
        model = VAETrajGenerator(
            latent_dim=64,
            kl_weight=0.001,
            recon_loss_type='mse',
            dropout=0.2,
            im_embd=128
        )
        model.build_model(5, 2, 10)
        model.id = "vae_0"
        
        # 创建模拟数据加载器
        mock_loader = MagicMock()
        
        # 测试训练方法
        result = self.evolutionary_learner._train_vae(model, mock_loader, epochs=3)
        
        # 验证结果
        self.assertEqual(result, {'total_loss': [0.8, 0.7, 0.6]})
        mock_train.assert_called_once_with(mock_loader, epochs=3, learning_rate=1e-4)
    
    @patch('utils.EvolutionaryLearner.EvolutionaryLearner._train_autoencoder')
    @patch('utils.AutoencoderModel.AutoencoderTrainer.load_sequences')
    def test_train_population(self, mock_load_sequences, mock_train_autoencoder):
        # 设置模拟返回值
        mock_load_sequences.return_value = MagicMock()
        mock_train_autoencoder.return_value = {'train_loss': [0.5, 0.4, 0.3]}
        
        # 创建模型
        model = AutoencoderTrainer(
            encoder_type='lstm',
            latent_dim=32,
            kl_weight=0.001,
            dropout_rate=0.2
        )
        model.build_model(5, 2, 10)
        model.id = "autoencoder_0"
        
        # 初始化种群
        self.evolutionary_learner.population = [model]
        self.evolutionary_learner.is_initialized = True
        
        # 测试训练种群方法
        result = self.evolutionary_learner.train_population(self.mock_trajectories, epochs=3, batch_size=32)
        
        # 验证结果
        self.assertIn("autoencoder_0", result)
        self.assertEqual(result["autoencoder_0"], {'train_loss': [0.5, 0.4, 0.3]})
        mock_load_sequences.assert_called_once()
        mock_train_autoencoder.assert_called_once()

class TestImitationLearningService(unittest.TestCase):
    
    def setUp(self):
        # 创建模拟数据
        self.mock_trajectories = {
            'data': [np.random.rand(10, 5) for _ in range(20)],  # 20个轨迹，每个轨迹10个时间步，5维状态
            'state_dim': 5,
            'action_dim': 2,
            'traj_length': 10
        }
        
        # 创建模型配置
        self.model_config = {
            'model_type': 'AutoencoderModel',
            'training_method': 'evolutionary',
            'population_size': 3,  # 小规模种群用于测试
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'selection_method': 'tournament',
            'elitism_ratio': 0.1,
            'max_generations': 2,
            'fitness_threshold': 0.95,
            'encoder_type': 'lstm',
            'latent_dim': 32,
            'kl_weight': 0.001,
            'dropout_rate': 0.2
        }
        
        # 创建服务
        self.service = ImitationLearningService()
    
    @patch('utils.EvolutionaryLearner.EvolutionaryLearner.initialize_population')
    @patch('utils.EvolutionaryLearner.EvolutionaryLearner.train_population')
    @patch('utils.EvolutionaryLearner.EvolutionaryLearner.run_evolution')
    @patch('core.services.imitation_learning_service.ImitationLearningService._create_eval_env')
    def test_train_evolutionary(self, mock_create_eval_env, mock_run_evolution, mock_train_population, mock_initialize_population):
        # 设置模拟返回值
        mock_initialize_population.return_value = [MagicMock() for _ in range(3)]
        mock_train_population.return_value = {'model_0': {'train_loss': [0.5, 0.4, 0.3]}}
        mock_create_eval_env.return_value = MagicMock()
        
        best_policy = MagicMock()
        mock_run_evolution.return_value = {
            'best_policy': best_policy,
            'mean_fitness': [0.5, 0.6],
            'max_fitness': [0.7, 0.8],
            'min_fitness': [0.3, 0.4],
            'diversity': [0.2, 0.1],
            'generations': 2
        }
        
        # 测试进化训练方法
        model, metrics = self.service._train_evolutionary('AutoencoderModel', self.mock_trajectories, self.model_config)
        
        # 验证结果
        self.assertEqual(model, best_policy)
        self.assertEqual(metrics['final_max_fitness'], 0.8)
        self.assertEqual(metrics['final_mean_fitness'], 0.6)
        self.assertEqual(metrics['generations'], 2)
        
        # 验证方法调用
        mock_initialize_population.assert_called_once()
        mock_train_population.assert_called_once()
        mock_create_eval_env.assert_called_once()
        mock_run_evolution.assert_called_once()

if __name__ == '__main__':
    unittest.main()