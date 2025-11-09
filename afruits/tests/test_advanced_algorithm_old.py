import os
import sys
import numpy as np
import torch
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from afruits.core.api import AlgorithmAPI

class TestAdvancedAlgorithm(unittest.TestCase):
    """
    高级算法模型测试类
    
    测试小样本专家轨迹模仿学习模块的高级算法模型功能
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
    
    #---------- 辅助方法 ----------#
    
    def create_trajectory_data(self, num_trajectories=10, steps_per_trajectory=20):
        """
        创建轨迹数据
        
        参数:
            num_trajectories (int): 轨迹数量
            steps_per_trajectory (int): 每条轨迹的时间步数
            
        返回:
            Dict: 轨迹数据字典
        """
        # 创建模拟的轨迹数据
        trajectories = {}
        
        # 创建指定数量的轨迹
        for i in range(num_trajectories):
            # 每条轨迹包含指定数量的时间步
            states = np.random.rand(steps_per_trajectory, 10)  # 10维状态空间
            actions = np.random.rand(steps_per_trajectory, 5)  # 5维动作空间
            
            trajectories[f'traj_{i}'] = {
                'states': states,
                'actions': actions,
                'rewards': np.random.rand(steps_per_trajectory),  # 随机奖励
                'dones': np.zeros(steps_per_trajectory),  # 完成标志
                'infos': [{} for _ in range(steps_per_trajectory)]  # 额外信息
            }
        
        return trajectories
    
    def create_trajectory_test_data(self):
        """创建轨迹测试数据"""
        # 创建模拟的轨迹数据
        test_trajectories = []
        
        # 创建5条测试轨迹
        for i in range(5):
            # 每条轨迹包含10个时间步
            states = np.random.rand(10, 10)  # 10维状态空间
            actions = np.random.rand(10, 5)  # 5维动作空间
            
            test_trajectories.append({
                'states': states,
                'actions': actions,
                'rewards': np.random.rand(10),  # 随机奖励
                'dones': np.zeros(10),  # 完成标志
                'infos': [{} for _ in range(10)]  # 额外信息
            })
        
        return test_trajectories
    
    def create_policy_template(self, input_dim=10, hidden_dim=64, output_dim=5):
        """
        创建策略模板
        
        参数:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
            
        返回:
            torch.nn.Module: 策略网络模板
        """
        # 简单的策略网络
        class SimplePolicy(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(SimplePolicy, self).__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim // 2, output_dim)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return SimplePolicy(input_dim, hidden_dim, output_dim)
    
    def create_dummy_env(self):
        """创建模拟环境"""
        class DummyEnv:
            def __init__(self):
                self.state_dim = 10
                self.action_dim = 5
                self.action_space = type('obj', (object,), {
                    'sample': lambda: np.random.rand(5)
                })
                self.observation_space = type('obj', (object,), {
                    'shape': (10,)
                })
                self.current_step = 0
                self.max_steps = 20
            
            def reset(self):
                self.current_step = 0
                return np.random.rand(10)
            
            def step(self, action):
                self.current_step += 1
                done = self.current_step >= self.max_steps
                return np.random.rand(10), np.random.rand(), done, {}
        
        return DummyEnv()
    
    def create_structured_data(self, num_samples=50, feature_dim=10, target_dim=5):
        """
        创建结构化数据（用于微调）
        
        参数:
            num_samples (int): 样本数量
            feature_dim (int): 特征维度
            target_dim (int): 目标维度
            
        返回:
            Dict: 包含特征和目标的数据字典
        """
        # 创建特征和目标
        features = np.random.rand(num_samples, feature_dim)
        targets = np.random.rand(num_samples, target_dim)
        
        # 返回字典格式
        return {
            'x': features,
            'y': targets
        }
    
    #---------- 测试方法 ----------#
    
    def test_evolutionary_learner(self):
        """
        测试小样本专家轨迹模仿学习模块的进化学习功能
        
        该测试验证EvolutionaryLearner的以下功能:
        1. 种群初始化与进化
        2. 适应度评估
        3. 最佳策略选择
        4. 策略预测
        """
        print("\n测试小样本专家轨迹模仿学习模块的进化学习功能 (EvolutionaryLearner)")
        
        # 创建测试数据
        training_data = self.create_trajectory_data()
        
        # 创建策略模板
        policy_template = self.create_policy_template()
        
        # 创建模拟环境
        eval_env = self.create_dummy_env()
        
        # 配置模型
        model_config = {
            'model_type': 'EvolutionaryLearner',  # 使用进化学习模型
            'population_size': 20,  # 减少种群规模以加快测试
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'selection_method': 'tournament',
            'elitism_ratio': 0.1,
            'max_generations': 5  # 减少代数以加快测试
        }
        
        # 训练模型
        result = self.api.train_advanced_model(training_data, model_config,
                                              policy_template=policy_template,
                                              eval_env=eval_env)
        
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
        print(f"最终种群规模: {training_metrics.get('final_population_size', 'N/A')}")
        print(f"收敛代数: {training_metrics.get('convergence_generation', 'N/A')}")
        
        # 测试获取最佳策略
        best_policy = self.api.advanced_modeling_service.get_best_policy(model_id)
        
        # 验证最佳策略
        self.assertIsNotNone(best_policy)
        print(f"最佳策略类型: {type(best_policy).__name__}")
        
        # 测试策略预测
        state = np.random.rand(10)  # 假设状态是10维
        action = self.api.advanced_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_trajectory_test_data()
        eval_config = {
            'method': 'policy_evaluation',
            'metrics': ['reward', 'success_rate'],
            'num_episodes': 5
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('avg_reward', eval_result)
        self.assertIn('success_rate', eval_result)
        print(f"平均奖励: {eval_result['avg_reward']:.4f}")
        print(f"成功率: {eval_result['success_rate']:.4f}")
    
    def test_incremental_learner(self):
        """
        测试小样本专家轨迹模仿学习模块的增量学习功能
        
        该测试验证IncrementalLearner的以下功能:
        1. 初始模型训练
        2. 数据流监控与分布漂移检测
        3. 增量模型更新
        4. 模型预测与评估
        """
        print("\n测试小样本专家轨迹模仿学习模块的增量学习功能 (IncrementalLearner)")
        
        # 创建初始训练数据
        initial_data = self.create_trajectory_data()
        
        # 创建增量训练数据（模拟新数据流）
        incremental_data = {}
        for i in range(5):  # 5条新轨迹
            states = np.random.rand(15, 10)  # 15个时间步，10维状态空间
            actions = np.random.rand(15, 5)  # 5维动作空间
            
            incremental_data[f'new_traj_{i}'] = {
                'states': states,
                'actions': actions,
                'rewards': np.random.rand(15),
                'dones': np.zeros(15),
                'infos': [{} for _ in range(15)]
            }
        
        # 配置模型
        model_config = {
            'model_type': 'IncrementalLearner',  # 使用增量学习模型
            'memory_buffer_size': 500,
            'regularization_strength': 0.5,
            'replay_strategy': 'generative',
            'adaptive_lr': True,
            'batch_size': 32,
            'max_epochs': 10  # 减少训练轮数以加快测试
        }
        
        # 初始训练模型
        result = self.api.train_advanced_model(initial_data, model_config)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        initial_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"初始训练损失: {initial_metrics.get('final_loss', 'N/A')}")
        
        # 测试数据流监控
        drift_report = self.api.advanced_modeling_service.monitor_data_stream(
            model_id, incremental_data, drift_threshold=0.3
        )
        
        # 验证漂移报告
        self.assertIn('is_drift', drift_report)
        self.assertIn('drift_score', drift_report)
        print(f"检测到分布漂移: {drift_report['is_drift']}")
        print(f"漂移分数: {drift_report['drift_score']:.4f}")
        
        # 测试增量训练
        incremental_result = self.api.advanced_modeling_service.update_model(
            model_id, incremental_data, epochs=5
        )
        
        # 验证增量训练结果
        self.assertIn('model', incremental_result)
        self.assertIn('training_metrics', incremental_result)
        
        updated_metrics = incremental_result['training_metrics']
        print(f"增量训练后损失: {updated_metrics.get('final_loss', 'N/A')}")
        print(f"遗忘率: {updated_metrics.get('forgetting_rate', 'N/A'):.4f}")
        
        # 测试预测功能
        state = np.random.rand(1, 10)  # 假设状态是1个批次，10个特征
        action = self.api.advanced_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_trajectory_test_data()
        eval_config = {
            'method': 'incremental',
            'metrics': ['accuracy', 'forgetting'],
            'old_data_ratio': 0.3
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('accuracy', eval_result)
        self.assertIn('forgetting_measure', eval_result)
        print(f"准确率: {eval_result['accuracy']:.4f}")
        print(f"遗忘度量: {eval_result['forgetting_measure']:.4f}")
    
    def test_fine_tune_manager(self):
        """
        测试小样本专家轨迹模仿学习模块的少样本微调功能
        
        该测试验证FineTuneManager的以下功能:
        1. 预训练模型加载
        2. 选择性层冻结
        3. 少样本数据微调
        4. 微调模型评估
        """
        print("\n测试小样本专家轨迹模仿学习模块的少样本微调功能 (FineTuneManager)")
        
        # 创建预训练模型
        pretrained_model = self.create_policy_template()
        
        # 创建少量微调数据
        fine_tune_data = {}
        # 只创建3条轨迹，模拟少样本场景
        for i in range(3):
            states = np.random.rand(10, 10)  # 10个时间步，10维状态空间
            actions = np.random.rand(10, 5)  # 5维动作空间
            
            fine_tune_data[f'ft_traj_{i}'] = {
                'states': states,
                'actions': actions,
                'rewards': np.random.rand(10),
                'dones': np.zeros(10),
                'infos': [{} for _ in range(10)]
            }
        
        # 配置模型
        model_config = {
            'model_type': 'FineTuneManager',  # 使用微调管理器
            'trainable_layers': ["*last*"],  # 只微调最后几层
            'freeze_strategy': 'selective',
            'optimizer_config': {
                'type': 'Adam',
                'lr': 0.001,
                'weight_decay': 0.0001,
                'use_layer_lr': True
            },
            'regularization_mode': 'adaptive',
            'max_epochs': 10,  # 减少训练轮数以加快测试
            'batch_size': 16,
            'augment_data': True  # 启用数据增强
        }
        
        # 训练模型
        result = self.api.train_advanced_model(fine_tune_data, model_config,
                                              base_model=pretrained_model)
        
        # 验证结果
        self.assertIn('model_id', result)
        self.assertIn('model', result)
        self.assertIn('training_metrics', result)
        
        # 提取模型ID和训练指标
        model_id = result['model_id']
        model = result['model']
        training_metrics = result['training_metrics']
        
        print(f"模型ID: {model_id}")
        print(f"最佳验证损失: {training_metrics.get('best_val_loss', 'N/A')}")
        print(f"训练时间(分钟): {training_metrics.get('training_time', 'N/A')}")
        print(f"收敛轮次: {training_metrics.get('convergence_epoch', 'N/A')}")
        
        # 测试预测功能
        state = np.random.rand(1, 10)  # 假设状态是1个批次，10个特征
        action = self.api.advanced_modeling_service.predict(model_id, state)
        
        # 验证预测结果
        self.assertIsNotNone(action)
        print(f"预测动作形状: {action.shape}")
        
        # 测试评估功能
        test_data = self.create_trajectory_test_data()
        eval_config = {
            'method': 'fine_tune',
            'metrics': ['accuracy', 'generalization'],
            'num_samples': 5
        }
        
        eval_result = self.api.evaluate_model(model, test_data, eval_config)
        
        # 验证评估结果
        self.assertIn('accuracy', eval_result)
        self.assertIn('generalization_score', eval_result)
        print(f"准确率: {eval_result['accuracy']:.4f}")
        print(f"泛化分数: {eval_result['generalization_score']:.4f}")

if __name__ == "__main__":
    unittest.main()