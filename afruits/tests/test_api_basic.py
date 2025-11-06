import os
import sys
import numpy as np
import torch
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from afruits.core.api import AlgorithmAPI

class TestAPIBasic(unittest.TestCase):
    """
    API基本功能测试类
    
    测试API的基本功能，包括初始化、数据加载和预处理
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
        
    def test_api_initialization(self):
        """测试API初始化"""
        print("\n测试API初始化")
        
        # 验证API实例
        self.assertIsNotNone(self.api)
        self.assertIsNotNone(self.api.game_modeling_service)
        self.assertIsNotNone(self.api.imitation_learning_service)
        self.assertIsNotNone(self.api.visualization_service)
        self.assertIsNotNone(self.api.logging_service)
        
        print("API初始化测试通过")
        
    def test_data_preprocessing(self):
        """测试数据预处理功能"""
        print("\n测试数据预处理功能")
        
        # 创建简单的测试数据
        raw_data = self.create_simple_data()
        
        # 配置预处理参数
        preprocess_config = {
            'normalize': True
        }
        
        # 预处理数据
        try:
            processed_data = self.api.preprocess_data(raw_data, preprocess_config)
            self.assertIsNotNone(processed_data)
            print("数据预处理测试通过")
        except Exception as e:
            self.fail(f"数据预处理失败: {str(e)}")
        
    def test_trajectory_preprocessing(self):
        """测试轨迹预处理功能"""
        print("\n测试轨迹预处理功能")
        
        # 创建简单的轨迹数据
        trajectories = self.create_simple_trajectories()
        
        # 配置预处理参数
        preprocess_config = {
            'normalize': True,
            'segment_length': 10
        }
        
        # 预处理轨迹数据
        try:
            processed_trajectories = self.api.preprocess_trajectory(trajectories, preprocess_config)
            self.assertIsNotNone(processed_trajectories)
            print("轨迹预处理测试通过")
        except Exception as e:
            self.fail(f"轨迹预处理失败: {str(e)}")
        
    def create_simple_data(self):
        """创建简单的测试数据"""
        # 创建一个简单的数据字典
        data = {
            'features': np.random.rand(100, 10),
            'labels': np.random.rand(100, 5)
        }
        
        return data
    
    def create_simple_trajectories(self):
        """创建简单的轨迹数据"""
        # 创建3条轨迹，每条轨迹包含20个时间步
        num_trajectories = 3
        trajectory_length = 20
        state_dim = 5
        action_dim = 2
        
        # 创建数据字典
        data = {
            'trajectories': []
        }
        
        for i in range(num_trajectories):
            # 创建状态序列
            states = np.random.rand(trajectory_length, state_dim)
            
            # 创建动作序列
            actions = np.random.rand(trajectory_length, action_dim)
            
            # 创建奖励序列
            rewards = np.random.rand(trajectory_length)
            
            # 添加到轨迹列表
            data['trajectories'].append({
                'states': states,
                'actions': actions,
                'rewards': rewards
            })
        
        return data

if __name__ == "__main__":
    unittest.main()