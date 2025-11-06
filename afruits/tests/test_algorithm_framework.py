import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from core.api import AlgorithmAPI

class TestAlgorithmFramework(unittest.TestCase):
    """
    算法框架测试类
    
    测试小样本博弈建模和专家轨迹模仿学习功能
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
        
    def test_game_modeling(self):
        """测试小样本博弈建模功能"""
        print("\n测试小样本博弈建模功能")
        
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
        
    def test_imitation_learning(self):
        """测试小样本专家轨迹模仿学习功能"""
        print("\n测试小样本专家轨迹模仿学习功能")
        
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
        print(f"最终训练损失: {transformer_metrics['final_train_loss']:.4f}")
        print(f"最终验证损失: {transformer_metrics['final_val_loss']:.4f}")
        
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
        
    def create_game_modeling_data(self):
        """创建博弈建模测试数据"""
        # 创建5条轨迹，每条轨迹包含50个时间步
        num_trajectories = 5
        trajectory_length = 50
        state_dim = 10
        action_dim = 5
        
        trajectories = {}
        
        for i in range(num_trajectories):
            # 创建状态序列
            states = np.random.rand(trajectory_length, state_dim)
            
            # 创建动作序列（简单的线性映射加噪声）
            actions = np.zeros((trajectory_length, action_dim))
            for j in range(trajectory_length):
                # 简单的策略：将状态映射到动作
                actions[j] = np.tanh(np.dot(states[j], np.random.rand(state_dim, action_dim)) + np.random.randn(action_dim) * 0.1)
            
            # 创建奖励序列
            rewards = np.sum(actions, axis=1)  # 简单的奖励：动作的和
            
            # 添加到轨迹字典
            trajectories[f"trajectory_{i}"] = {
                'states': states,
                'actions': actions,
                'rewards': rewards
            }
        
        return trajectories
    
    def create_game_modeling_test_data(self):
        """创建博弈建模测试数据"""
        # 创建2条测试轨迹
        num_trajectories = 2
        trajectory_length = 20
        state_dim = 10
        action_dim = 5
        
        test_trajectories = []
        
        for i in range(num_trajectories):
            # 创建状态序列
            states = np.random.rand(trajectory_length, state_dim)
            
            # 创建动作序列
            actions = np.zeros((trajectory_length, action_dim))
            for j in range(trajectory_length):
                # 简单的策略：将状态映射到动作
                actions[j] = np.tanh(np.dot(states[j], np.random.rand(state_dim, action_dim)) + np.random.randn(action_dim) * 0.1)
            
            # 添加到测试轨迹列表
            test_trajectories.append({
                'states': states,
                'actions': actions
            })
        
        return test_trajectories
    
    def create_expert_trajectories(self):
        """创建专家轨迹数据"""
        # 创建10条专家轨迹，每条轨迹包含30个时间步
        num_trajectories = 10
        trajectory_length = 30
        state_dim = 10
        action_dim = 5
        
        # 创建数据字典
        data = {
            'trajectories': []
        }
        
        for i in range(num_trajectories):
            # 创建状态序列
            states = np.random.rand(trajectory_length, state_dim)
            
            # 创建动作序列（专家策略）
            actions = np.zeros((trajectory_length, action_dim))
            for j in range(trajectory_length):
                # 模拟专家策略：非线性映射
                actions[j] = np.tanh(np.dot(states[j], np.random.rand(state_dim, action_dim)) + np.random.randn(action_dim) * 0.05)
            
            # 创建奖励序列
            rewards = np.sum(actions, axis=1) + np.random.randn(trajectory_length) * 0.1
            
            # 添加到轨迹列表
            data['trajectories'].append({
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'expert': True  # 标记为专家轨迹
            })
        
        # 添加批次数据（用于Transformer训练）
        batch_data = []
        for i in range(20):  # 20个批次
            inputs = np.random.rand(16, 10, state_dim)  # 批次大小16，序列长度10
            targets = np.random.rand(16, 10, action_dim)  # 对应的目标动作
            batch_data.append({
                'inputs': inputs,
                'targets': targets
            })
        
        data['batch_data'] = batch_data
        
        return data

if __name__ == "__main__":
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(TestAlgorithmFramework('test_game_modeling'))
    test_suite.addTest(TestAlgorithmFramework('test_imitation_learning'))
    
    # 运行测试
    unittest.TextTestRunner().run(test_suite)