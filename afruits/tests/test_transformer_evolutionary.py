import sys
import os
import torch
import numpy as np
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.TransformerModel import TransformerTrainer
from utils.EvolutionaryLearner import EvolutionaryLearner
from core.services.imitation_learning_service import ImitationLearningService

class TestTransformerEvolutionary(unittest.TestCase):
    """测试Transformer模型进化学习功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建模拟的专家轨迹数据
        self.expert_trajectories = {
            "data": self._create_mock_data(),
            "state_dim": 4,
            "action_dim": 2,
            "traj_length": 10
        }
        
        # 创建模型配置
        self.model_config = {
            "d_model": 64,
            "num_heads": 2,
            "num_layers": 2,
            "max_seq_len": 10,
            "dropout_rate": 0.1,
            "population_size": 4,
            "mutation_rate": 0.2,
            "crossover_rate": 0.7,
            "selection_method": "tournament",
            "elitism_ratio": 0.25,
            "max_generations": 2,
            "fitness_threshold": 0.9,
            "epochs": 2,
            "batch_size": 8
        }
    
    def _create_mock_data(self):
        """创建模拟数据"""
        # 创建10个轨迹，每个轨迹包含10个状态-动作对
        num_trajectories = 10
        traj_length = 10
        state_dim = 4
        action_dim = 2
        
        # 创建状态数据
        states = torch.randn(num_trajectories, traj_length, state_dim)
        
        # 创建动作数据（离散动作）
        actions = torch.randint(0, action_dim, (num_trajectories, traj_length))
        
        # 返回数据字典
        return {
            "states": states,
            "actions": actions
        }
    
    def _create_mock_env(self):
        """创建模拟环境"""
        mock_env = MagicMock()
        
        # 模拟reset方法
        mock_env.reset.return_value = torch.randn(4)
        
        # 模拟step方法
        def mock_step(action):
            return torch.randn(4), float(np.random.rand()), bool(np.random.rand() > 0.8), {}
        
        mock_env.step = mock_step
        
        return mock_env
    
    def test_evolutionary_learner_initialization(self):
        """测试EvolutionaryLearner初始化"""
        # 创建进化学习器
        learner = EvolutionaryLearner(
            population_size=self.model_config["population_size"],
            mutation_rate=self.model_config["mutation_rate"],
            crossover_rate=self.model_config["crossover_rate"],
            selection_method=self.model_config["selection_method"],
            elitism_ratio=self.model_config["elitism_ratio"],
            model_config=self.model_config
        )
        
        # 检查初始化是否成功
        self.assertEqual(learner.population_size, self.model_config["population_size"])
        self.assertEqual(learner.mutation_rate, self.model_config["mutation_rate"])
        self.assertEqual(learner.crossover_rate, self.model_config["crossover_rate"])
        self.assertEqual(learner.selection_method, self.model_config["selection_method"])
        self.assertEqual(learner.elitism_ratio, self.model_config["elitism_ratio"])
        self.assertEqual(learner.model_config, self.model_config)
        
        # 检查种群和适应度是否为空
        self.assertEqual(len(learner.population), 0)
        self.assertEqual(len(learner.fitness_scores), 0)
    
    def test_initialize_population(self):
        """测试种群初始化"""
        # 创建进化学习器
        learner = EvolutionaryLearner(
            population_size=self.model_config["population_size"],
            model_config=self.model_config
        )
        
        # 创建模板模型
        template_model = TransformerTrainer(
            d_model=self.model_config["d_model"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            max_seq_len=self.model_config["max_seq_len"],
            dropout_rate=self.model_config["dropout_rate"]
        )
        
        # 构建模型
        template_model.build_model(self.expert_trajectories["state_dim"], self.expert_trajectories["action_dim"])
        
        # 初始化种群
        population = learner.initialize_population(template_model, seed=42)
        
        # 检查种群大小
        self.assertEqual(len(population), self.model_config["population_size"])
        
        # 检查种群中的模型类型
        for model in population:
            self.assertIsInstance(model, TransformerTrainer)
            self.assertEqual(model.d_model, self.model_config["d_model"])
            self.assertEqual(model.num_heads, self.model_config["num_heads"])
            self.assertEqual(model.num_layers, self.model_config["num_layers"])
    
    def test_train_population(self):
        """测试种群训练"""
        # 创建进化学习器
        learner = EvolutionaryLearner(
            population_size=self.model_config["population_size"],
            model_config=self.model_config
        )
        
        # 创建模板模型
        template_model = TransformerTrainer(
            d_model=self.model_config["d_model"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            max_seq_len=self.model_config["max_seq_len"],
            dropout_rate=self.model_config["dropout_rate"]
        )
        
        # 构建模型
        template_model.build_model(self.expert_trajectories["state_dim"], self.expert_trajectories["action_dim"])
        
        # 初始化种群
        population = learner.initialize_population(template_model, seed=42)
        
        # 模拟train_model方法
        with patch.object(TransformerTrainer, 'train_model', return_value={'train_loss': [0.5, 0.4]}):
            with patch.object(TransformerTrainer, 'load_sequences', return_value=MagicMock()):
                # 训练种群
                training_results = learner.train_population(
                    self.expert_trajectories,
                    epochs=self.model_config["epochs"],
                    batch_size=self.model_config["batch_size"]
                )
                
                # 检查训练结果
                self.assertEqual(len(training_results), self.model_config["population_size"])
    
    def test_evolutionary_process(self):
        """测试进化过程"""
        # 创建进化学习器
        learner = EvolutionaryLearner(
            population_size=self.model_config["population_size"],
            mutation_rate=self.model_config["mutation_rate"],
            crossover_rate=self.model_config["crossover_rate"],
            selection_method=self.model_config["selection_method"],
            elitism_ratio=self.model_config["elitism_ratio"],
            model_config=self.model_config
        )
        
        # 创建模板模型
        template_model = TransformerTrainer(
            d_model=self.model_config["d_model"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            max_seq_len=self.model_config["max_seq_len"],
            dropout_rate=self.model_config["dropout_rate"]
        )
        
        # 构建模型
        template_model.build_model(self.expert_trajectories["state_dim"], self.expert_trajectories["action_dim"])
        
        # 初始化种群
        population = learner.initialize_population(template_model, seed=42)
        
        # 创建模拟环境
        mock_env = self._create_mock_env()
        
        # 模拟evaluate_fitness方法
        with patch.object(EvolutionaryLearner, '_evaluate_transformer', return_value=0.5):
            # 评估适应度
            fitness_scores = learner.evaluate_fitness(population, mock_env)
            
            # 检查适应度分数
            self.assertEqual(len(fitness_scores), self.model_config["population_size"])
            
            # 选择父代
            parents = learner.select_parents(fitness_scores, 2)
            
            # 检查父代数量
            self.assertEqual(len(parents), 2)
            
            # 进化生成新一代
            new_population = learner.evolve_population(parents, 1.0)
            
            # 检查新一代种群大小
            self.assertEqual(len(new_population), self.model_config["population_size"])
    
    def test_imitation_learning_service(self):
        """测试ImitationLearningService中的_train_evolutionary方法"""
        # 创建ImitationLearningService
        service = ImitationLearningService()
        
        # 模拟_create_eval_env方法
        service._create_eval_env = MagicMock(return_value=self._create_mock_env())
        
        # 模拟EvolutionaryLearner的方法
        with patch('utils.EvolutionaryLearner.EvolutionaryLearner') as mock_learner:
            # 设置模拟的返回值
            mock_instance = mock_learner.return_value
            mock_instance.initialize_population.return_value = []
            mock_instance.train_population.return_value = {}
            mock_instance.run_evolution.return_value = {
                'generation_stats': [
                    {'generation': 1, 'mean_fitness': 0.4, 'max_fitness': 0.6, 'min_fitness': 0.2},
                    {'generation': 2, 'mean_fitness': 0.5, 'max_fitness': 0.7, 'min_fitness': 0.3}
                ],
                'best_policy': MagicMock(),
                'history': {'diversity': [0.1, 0.2]}
            }
            
            # 调用_train_evolutionary方法
            best_model, metrics = service._train_evolutionary(
                'TransformerModel',
                self.expert_trajectories,
                self.model_config
            )
            
            # 检查是否调用了正确的方法
            mock_learner.assert_called_once()
            mock_instance.initialize_population.assert_called_once()
            mock_instance.train_population.assert_called_once()
            mock_instance.run_evolution.assert_called_once()
            
            # 检查返回的指标
            self.assertIn('mean_fitness', metrics)
            self.assertIn('max_fitness', metrics)
            self.assertIn('min_fitness', metrics)
            self.assertIn('diversity', metrics)
            self.assertIn('final_mean_fitness', metrics)
            self.assertIn('final_max_fitness', metrics)
            self.assertIn('generations', metrics)

if __name__ == '__main__':
    unittest.main()