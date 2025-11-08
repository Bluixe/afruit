import os
import sys
import numpy as np
import torch
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from afruits.core.api import AlgorithmAPI

class TestModelAndGenerateAlgorithm(unittest.TestCase):
    """
    轨迹建模与生成模型测试类
    
    测试小样本专家轨迹模仿学习模块的轨迹建模与生成功能
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
    
    #---------- 辅助方法 ----------#
    
    def create_trajectory_data(self):
        """创建轨迹数据"""
        # 创建模拟的轨迹数据
        trajectories = []
        # 创建长序列轨迹(batch_size, seq_length, state_dim)
        
        # 创建10条轨迹
        for i in range(200):
            # 每条轨迹包含40个时间步
            states = np.random.rand(80, 10)  # 10维状态空间
            # 离散动作空间，动作维度为5
            actions = np.random.randint(0, 5, size=(80,))  # 离散动作空间
            
            trajectories.append({
                'states': states,
                'actions': actions,
            })
        
        return {"data": trajectories,
                "state_dim": (10,),
                "action_dim": 5,
                "traj_length": 80}
    
    
    def create_trajectory_test_data(self):
        """创建轨迹测试数据"""
        # 创建模拟的轨迹数据
        test_trajectories = []
        
        # 创建5条测试轨迹
        for i in range(60):
            # 每条轨迹包含40个时间步
            states = np.random.rand(80, 10)  # 10维状态空间
            # 离散动作空间，动作维度为5
            actions = np.random.randint(0, 5, size=(80,))  # 离散动作空间
            
            test_trajectories.append({
                'states': states,
                'actions': actions,
            })
        
        return {"data": test_trajectories,
                "state_dim": (10,),
                "action_dim": 5,
                "traj_length": 80}
    
    #---------- 测试方法 ----------#
    
    def test_autoencoder_model(self):
        """测试小样本专家轨迹模仿学习模块的基于自编码器的轨迹建模功能"""
        print("\n测试小样本专家轨迹模仿学习模块的基于自编码器的轨迹建模功能 (AutoencoderModel)")
        
        # 创建测试数据
        training_data = self.create_trajectory_data()
        
        # 配置模型
        model_config = {
            'model_type': 'AutoencoderModel',  # 使用自编码器模型
            'latent_dim': 32,
            'hidden_dim': 64,
            'batch_size': 32,
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'learning_rate': 1e-3,
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_imitation_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        # print(f"训练损失: {training_metrics['final_train_loss']:.4f}")
        # print(f"验证损失: {training_metrics['final_val_loss']:.4f}")
        
        # 测试编码功能
        state = np.random.rand(1, 10, 10)  # 假设状态是1个批次，10个时间步，每个时间步10个特征
        encoded = self.api.trajectory_modeling_service.encode(model_id, state)
        
        # 验证编码结果
        self.assertIsNotNone(encoded)
        print(f"编码结果形状: {encoded.shape}")
        
        # 测试解码功能
        decoded = self.api.trajectory_modeling_service.decode(model_id, encoded)
        
        # 验证解码结果
        self.assertIsNotNone(decoded)
        print(f"解码结果形状: {decoded.shape}")
        
        # 测试评估功能
        test_data = self.create_trajectory_test_data()
        eval_config = {
            'method': 'reconstruction',
            'metrics': ['mse', 'mae']
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('reconstruction_mse', eval_result)
        self.assertIn('reconstruction_mae', eval_result)
        print(f"重构MSE: {eval_result['reconstruction_mse']:.4f}")
        print(f"重构MAE: {eval_result['reconstruction_mae']:.4f}")
    
    def test_transformer_model(self):
        """测试小样本专家轨迹模仿学习模块的基于Transformer的轨迹建模功能"""
        print("\n测试小样本专家轨迹模仿学习模块的基于Transformer的轨迹建模功能 (TransformerModel)")
        
        # 创建测试数据
        training_data = self.create_trajectory_data()
        
        # 配置模型
        model_config = {
            'model_type': 'TransformerModel',  # 使用Transformer模型
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'batch_size': 32,
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'learning_rate': 1e-3,
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_trajectory_model(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        # print(f"训练损失: {training_metrics['final_train_loss']:.4f}")
        # print(f"验证损失: {training_metrics['final_val_loss']:.4f}")
        
        # 测试预测功能
        state = np.random.rand(1, 5, 10)  # 假设状态是1个批次，5个时间步，每个时间步10个特征
        predicted = self.api.trajectory_modeling_service.predict_next(model_id, state, steps=5)
        
        # 验证预测结果
        self.assertIsNotNone(predicted)
        print(f"预测结果形状: {predicted.shape}")
        
        # 测试评估功能
        test_data = self.create_trajectory_test_data()
        eval_config = {
            'method': 'prediction',
            'metrics': ['mse', 'mae'],
            'prediction_steps': 5
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('prediction_mse', eval_result)
        self.assertIn('prediction_mae', eval_result)
        print(f"预测MSE: {eval_result['prediction_mse']:.4f}")
        print(f"预测MAE: {eval_result['prediction_mae']:.4f}")
    
    def test_diffusion_traj_generator(self):
        """测试小样本专家轨迹模仿学习模块的基于扩散模型的轨迹生成模型功能"""
        print("\n测试小样本专家轨迹模仿学习模块的基于扩散模型的轨迹生成模型功能 (DiffusionTrajGenerator)")
        
        # 创建测试数据
        training_data = self.create_trajectory_data()
        
        # 配置模型
        model_config = {
            'model_type': 'DiffusionTrajGenerator',  # 使用扩散模型
            'diffusion_steps': 100,  # 减少步数以加快测试
            'noise_schedule': 'cosine',
            'batch_size': 32,
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'learning_rate': 1e-4,
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_trajectory_generator(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"训练损失: {training_metrics['final_loss']:.4f}")
        
        # 测试生成功能
        # generated = self.api.trajectory_generation_service.generate(model_id, num_samples=5)
        
        # # 验证生成结果
        # self.assertIsNotNone(generated)
        # self.assertIn('trajectories', generated)
        # self.assertIn('validity_flags', generated)
        # print(f"生成轨迹数量: {len(generated['trajectories'])}")
        # print(f"有效轨迹数量: {np.sum(generated['validity_flags'])}")
        
        # 测试评估功能
        # test_data = self.create_trajectory_test_data()
        # eval_config = {
        #     'method': 'generation',
        #     'metrics': ['fid', 'diversity'],
        #     'num_samples': 10
        # }
        
        # eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # # 验证评估结果
        # self.assertIn('fid_score', eval_result)
        # self.assertIn('diversity_score', eval_result)
        # print(f"FID分数: {eval_result['fid_score']:.4f}")
        # print(f"多样性分数: {eval_result['diversity_score']:.4f}")
    
    def test_vae_traj_generator(self):
        """测试小样本专家轨迹模仿学习模块的基于变分自编码器的轨迹生成模型功能"""
        print("\n测试小样本专家轨迹模仿学习模块的基于变分自编码器的轨迹生成模型功能 (VAETrajGenerator)")
        
        # 创建测试数据
        training_data = self.create_trajectory_data()
        
        # 配置模型
        model_config = {
            'model_type': 'VAETrajGenerator',  # 使用VAE模型
            'latent_dim': 32,
            'seq_length': 20,
            'kl_weight': 0.001,
            'batch_size': 32,
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'learning_rate': 1e-3,
            'validation_split': 0.2
        }
        
        # 训练模型
        result = self.api.train_trajectory_generator(training_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"总损失: {training_metrics['final_total_loss']:.4f}")
        # print(f"重构损失: {training_metrics['final_recon_loss']:.4f}")
        # print(f"KL散度: {training_metrics['final_kl_loss']:.4f}")
        
        # 测试生成功能
        generated = self.api.trajectory_generation_service.generate(model_id, num_samples=5)
        
        # 验证生成结果
        self.assertIsNotNone(generated)
        self.assertIn('trajectories', generated)
        self.assertIn('latent_codes', generated)
        print(f"生成轨迹数量: {len(generated['trajectories'])}")
        print(f"潜在编码形状: {generated['latent_codes'].shape}")
        
        # 测试条件生成
        cond_vector = np.random.randn(1, 32)  # 假设潜在空间维度为32
        cond_generated = self.api.trajectory_generation_service.generate(model_id, cond_vector=cond_vector)
        
        # 验证条件生成结果
        self.assertIsNotNone(cond_generated)
        print(f"条件生成轨迹形状: {cond_generated['trajectories'].shape}")
        
        # 测试评估功能
        test_data = self.create_trajectory_test_data()
        eval_config = {
            'method': 'generation',
            'metrics': ['reconstruction', 'kl_divergence'],
            'num_samples': 10
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('reconstruction_loss', eval_result)
        self.assertIn('kl_divergence', eval_result)
        print(f"重构损失: {eval_result['reconstruction_loss']:.4f}")
        print(f"KL散度: {eval_result['kl_divergence']:.4f}")

if __name__ == "__main__":
    unittest.main()