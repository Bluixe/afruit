import os
import sys
import numpy as np
import torch
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API
from afruits.core.api import AlgorithmAPI
from afruits.utils.DataPreprocessor import DataPreprocessor  # 导入数据预处理器

class TestAPIBasic(unittest.TestCase):
    """
    API基本功能测试类
    
    测试API的基本功能，包括初始化、数据加载和预处理
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
        # 初始化数据预处理器
        self.preprocessor = DataPreprocessor()
        
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
            'outlier_threshold': 3.0,
            "alignment_mode": "linear"
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
            
    def test_training_data_preprocessing(self):
        """测试训练数据预处理功能"""
        print("\n测试训练数据预处理功能")
        
        # 创建符合新格式的轨迹数据
        raw_data = {
            "trajectories": [
                {
                    "states": np.random.rand(5, 4),  # 5步轨迹，4维状态
                    "actions": np.random.rand(5, 2), # 5步轨迹，2维动作
                    "rewards": np.random.rand(5),
                    "next_states": np.random.rand(5, 4),
                    "dones": np.array([0,0,0,0,1]),
                    "infos": [{}, {}, {}, {}, {}],
                    "opponent_actions": np.random.rand(5, 2)
                },
                {
                    "states": np.random.rand(3, 4),  # 3步轨迹，4维状态
                    "actions": np.random.rand(3, 2), # 3步轨迹，2维动作
                    "rewards": np.random.rand(3),
                    "next_states": np.random.rand(3, 4),
                    "dones": np.array([0,0,1]),
                    "infos": [{}, {}, {}],
                    "opponent_actions": np.random.rand(3, 2)
                }
            ],
            "state_dim": (4,),
            "action_dim": 2
        }
        
        # 测试不同模型类型的数据预处理
        model_types = [
            "AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator",
            "OfflineRLearner", "OfflineFSPLearner", "BehaviorCloner", "AdversarialImitationLearner"
        ]
        
        for model_type in model_types:
            print(f"测试模型类型: {model_type}")
            try:
                # 预处理训练数据
                processed_data = self.preprocessor.preprocess_for_training(raw_data, model_type)
                
                # 验证基本数据结构
                self.assertIn("data", processed_data)
                self.assertIn("state_dim", processed_data)
                self.assertIn("action_dim", processed_data)
                self.assertEqual(processed_data["state_dim"], (4,))
                self.assertEqual(processed_data["action_dim"], 2)
                
                # 验证特定模型类型的数据格式
                if model_type in ["AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator"]:
                    self.assertIsInstance(processed_data["data"], list)
                    self.assertEqual(len(processed_data["data"]), 2)
                    self.assertIn("traj_length", processed_data)
                    
                    # 验证轨迹长度
                    self.assertEqual(len(processed_data["data"][0]['states']), 5)
                    self.assertEqual(len(processed_data["data"][1]['states']), 3)
                else:
                    self.assertIsInstance(processed_data["data"], dict)
                    self.assertEqual(len(processed_data["data"]), 2)
                    
                    # 验证轨迹字段
                    traj0 = processed_data["data"]["traj_0"]
                    self.assertIn('states', traj0)
                    self.assertIn('actions', traj0)
                    
                    # 验证模型特定字段
                    if model_type == 'OfflineRLearner':
                        self.assertIn('rewards', traj0)
                        self.assertIn('next_states', traj0)
                    elif model_type == 'OfflineFSPLearner':
                        self.assertIn('opponent_actions', traj0)
                    
                print(f"  {model_type} 数据格式验证通过")
                
            except Exception as e:
                self.fail(f"{model_type} 预处理失败: {str(e)}")
        
        print("训练数据预处理测试通过")
        
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
        # 创建20条轨迹，每条轨迹包含200个时间步
        num_trajectories = 20
        trajectory_length = 200
        state_dim = 5
        action_dim = 2
        
        # 创建数据字典
        data = {
            'states': [],
            "actions": [],
            "rewards": [],
            "timestamps": [],
        }
        
        for i in range(num_trajectories):
            # 创建状态序列
            states = np.random.rand(trajectory_length, state_dim)
            
            # 创建动作序列
            actions = np.random.rand(trajectory_length, action_dim)
            
            # 创建奖励序列
            rewards = np.random.rand(trajectory_length)
            
            # 创建时间戳序列（假设每个时间步间隔为0.1秒）
            timestamps = np.array([0.1 * j for j in range(trajectory_length)])
            
            # 添加到数据字典
            data['states'].append(states)
            data['actions'].append(actions)
            data['rewards'].append(rewards)
            data['timestamps'].append(timestamps)
        
        return data

if __name__ == "__main__":
    unittest.main()