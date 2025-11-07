import os
import sys
import numpy as np
import torch
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from afruits.core.api import AlgorithmAPI
from afruits.utils.enhanced_test_runner import EnhancedTestRunner

class TestBasicAlgorithm(unittest.TestCase):
    """
    基础算法模型测试类
    
    测试小样本博弈建模模块的基础算法模型功能
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
    
    #---------- 辅助方法 ----------#
    
    def create_game_modeling_data(self, algorithm_type='BehaviorCloner'):
        """创建博弈建模训练数据"""
        # 创建模拟的轨迹数据
        trajectories = {}
        
        # 创建10条轨迹
        for i in range(10):
            # 每条轨迹包含20个时间步
            states = np.random.rand(20, 10)  # 10维状态空间
            actions = np.random.rand(20, 5)  # 5维动作空间
            opponent_actions = np.random.rand(20, 5)  # 对手动作
            next_states = np.random.rand(20, 10)  # 下一个状态

            if algorithm_type == 'OfflineRLearner':
                # 离线强化学习需要单独的轨迹格式
                trajectories[f'traj_{i}'] = {
                    'states': states,
                    'actions': actions,
                    'rewards': np.random.rand(20),  # 随机奖励
                    'next_states': next_states,
                    'dones': np.zeros(20),  # 完成标志
                    'infos': [{} for _ in range(20)]  # 额外信息
                }
            elif algorithm_type == 'OfflineFSPLearner':
                # 离线自对弈需要单独的轨迹格式
                trajectories[f'traj_{i}'] = {
                    'states': states,
                    'actions': actions,
                    'opponent_actions': opponent_actions,
                    'next_states': next_states,
                    'rewards': np.random.rand(20),  # 随机奖励
                    'dones': np.zeros(20),  # 完成标志
                    'infos': [{} for _ in range(20)]  # 额外信息
                }
            else:
            
                trajectories[f'traj_{i}'] = {
                    'states': states,
                    'actions': actions,
                    # 'opponent_actions': opponent_actions,
                    # 'next_states': next_states,
                    'rewards': np.random.rand(20),  # 随机奖励
                    'dones': np.zeros(20),  # 完成标志
                    'infos': [{} for _ in range(20)]  # 额外信息
                }
        
        return trajectories
    
    def create_game_modeling_test_data(self):
        """创建博弈建模测试数据"""
        # 创建模拟的轨迹数据
        test_trajectories = []
        
        # 创建5条测试轨迹
        for i in range(5):
            # 每条轨迹包含10个时间步
            states = np.random.rand(10, 10)  # 10维状态空间
            actions = np.random.rand(10, 5)  # 5维动作空间
            opponent_actions = np.random.rand(20, 5)  # 对手动作
            next_states = np.random.rand(20, 10)  # 下一个状态
            
            test_trajectories.append({
                'states': states,
                'actions': actions,
                # 'opponent_actions': opponent_actions,
                # 'next_states': next_states,
                'rewards': np.random.rand(10),  # 随机奖励
                'dones': np.zeros(10),  # 完成标志
                'infos': [{} for _ in range(10)]  # 额外信息
            })
        
        return test_trajectories
    
    #---------- 测试方法 ----------#
    
    def test_behavior_cloner(self):
        """测试小样本博弈建模模块的行为克隆功能"""
        print("\n测试小样本博弈建模模块的行为克隆功能 (BehaviorCloner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'BehaviorCloner',  # 使用行为克隆模型
            'batch_size': 32,
            'network_type': 'MLP',
            'max_epochs': 50,  # 减少训练轮数以加快测试
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
        print(f"评估结果: {eval_result}")
    
    def test_adversarial_imitation_learner(self):
        """测试小样本博弈建模模块的对抗模仿学习功能"""
        print("\n测试小样本博弈建模模块的对抗模仿学习功能 (AdversarialImitationLearner)")
        
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
            'max_epochs': 50, 
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
        print(f"评估结果: {eval_result}")
    
    def test_offline_r_learner(self):
        """测试小样本博弈建模模块的离线强化学习功能"""
        print("\n测试小样本博弈建模模块的离线强化学习功能 (OfflineRLearner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'OfflineRLearner',  # 使用离线强化学习模型
            'algorithm': 'CQL',  # 使用保守Q学习算法
            'hidden_dim': 64,
            'learning_rate': 3e-4,
            'batch_size': 32,
            'max_epochs': 50,  # 减少训练轮数以加快测试
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
        print(f"评估结果: {eval_result}")
    
    def test_offline_fsp_learner(self):
        """测试小样本博弈建模模块的离线自对弈功能"""
        print("\n测试小样本博弈建模模块的离线自对弈功能 (OfflineFSPLearner)")
        
        # 创建测试数据
        training_data = self.create_game_modeling_data()
        
        # 配置模型
        model_config = {
            'model_type': 'OfflineFSPLearner',  # 使用离线自对弈模型
            'br_hidden_dim': 64,
            'avg_hidden_dim': 64,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'max_epochs': 50,  # 减少训练轮数以加快测试
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
        print(f"评估结果: {eval_result}")

if __name__ == "__main__":
    # 使用增强的测试运行器而不是默认的
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicAlgorithm)
    runner = EnhancedTestRunner(verbosity=2)
    runner.run(test_suite)