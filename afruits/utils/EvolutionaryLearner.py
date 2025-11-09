import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import copy
import random

from afruits.utils.DataLoader import DataLoaderUtil
from utils.TransformerModel import TransformerTrainer, TransformerModel

class EvolutionaryLearner:
    """
    Transformer模型进化学习器类
    
    功能定义：基于种群进化机制实现Transformer模型策略群体持续优化
    
    核心特性：
    - 多策略并行：支持Transformer模型种群管理
    - 进化算子：集成随机变异/均匀交叉/高斯变异
    - 多样性保护：自适应相似度惩罚机制
    - 断点续训：支持进化过程状态保存与恢复
    """
    
    def __init__(self,
                 population_size: int = 10,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.7,
                 selection_method: str = "tournament",
                 elitism_ratio: float = 0.1,
                 model_config: Dict = {}):
        """
        初始化Transformer进化学习器
        
        参数:
            population_size (int): 种群规模，默认为10，有效取值范围10-50
            mutation_rate (float): 变异概率，默认为0.15，有效取值范围0.05-0.3
            crossover_rate (float): 交叉概率，默认为0.7，有效取值范围0.4-0.9
            selection_method (str): 选择策略类型，默认为"tournament"，有效取值范围["tournament", "roulette"]
            elitism_ratio (float): 精英保留比例，默认为0.1，有效取值范围0.0-0.2
            model_config (Dict): Transformer模型配置
        """
        # 初始化参数
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.elitism_ratio = elitism_ratio
        self.model_config = model_config
        
        # 初始化种群和适应度
        self.population = []
        self.fitness_scores = {}
        
        # 初始化训练状态
        self.is_initialized = False
        self.generation = 0
        self.best_policy = None
        self.best_fitness = -float('inf')
        self.history = {
            'mean_fitness': [],
            'max_fitness': [],
            'min_fitness': [],
            'diversity': []
        }
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def initialize_population(self, policy_template: TransformerTrainer, seed: int = None) -> List:
        """
        种群初始化方法
        
        参数:
            policy_template (TransformerTrainer): Transformer模型模板
            seed (int): 随机种子
            
        返回值:
            List: 初始化的Transformer模型种群
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # 检查模板是否为TransformerTrainer类型
        if not isinstance(policy_template, TransformerTrainer):
            raise TypeError("策略模板必须是TransformerTrainer类型")
        
        # 初始化种群
        self.population = []
        
        # 获取模型配置
        d_model = self.model_config.get('d_model', 128)
        num_heads = self.model_config.get('num_heads', 4)
        num_layers = self.model_config.get('num_layers', 3)
        max_seq_len = self.model_config.get('max_seq_len', 100)
        dropout_rate = self.model_config.get('dropout_rate', 0.2)
        
        # 获取输入和输出维度
        input_dim = policy_template.input_dim
        action_dim = policy_template.action_dim
        
        # 创建种群
        for i in range(self.population_size):
            # 创建新的Transformer模型
            model = TransformerTrainer(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                dropout_rate=dropout_rate
            )
            
            # 构建模型
            model.build_model(input_dim, action_dim)
            
            # 随机初始化模型参数
            for param in model.model.parameters():
                # 使用正态分布初始化参数
                nn.init.normal_(param, mean=0.0, std=0.02)
            
            # 设置模型ID
            model.id = f"transformer_{i}"
            
            # 添加到种群
            self.population.append(model)
        
        # 设置初始化标志
        self.is_initialized = True
        
        print(f"种群初始化完成: {len(self.population)} 个Transformer模型")
        
        return self.population
    
    def evaluate_fitness(self, population: List, eval_env: object) -> Dict:
        """
        适应度评估方法
        
        参数:
            population (List): 当前策略种群
            eval_env (object): 评估环境实例
        
        返回值:
            适应度字典 (dict): {"policy_id": (score, metadata)}
        """
        # 检查种群是否为空
        if not population:
            raise ValueError("种群不能为空")
        
        # 初始化适应度字典
        fitness_scores = {}
        
        # 评估每个策略
        for policy in population:
            # 检查策略是否有ID
            if not hasattr(policy, 'id'):
                policy.id = f"transformer_{id(policy)}"
            
            # 评估策略
            try:
                # 使用环境评估策略
                score = self._evaluate_transformer(policy, eval_env)
                
                # 存储适应度分数和元数据
                metadata = {
                    'type': 'TransformerTrainer',
                    'params': {
                        'd_model': policy.d_model,
                        'num_heads': policy.num_heads,
                        'num_layers': policy.num_layers,
                        'dropout_rate': policy.dropout_rate
                    }
                }
                
                fitness_scores[policy.id] = (score, metadata)
                
            except Exception as e:
                print(f"评估策略 {policy.id} 时出错: {str(e)}")
                # 分配一个很低的适应度分数
                fitness_scores[policy.id] = (-1000, {'error': str(e)})
        
        # 更新适应度字典
        self.fitness_scores = fitness_scores
        
        # 计算并打印统计信息
        scores = [score for score, _ in fitness_scores.values()]
        if scores:
            print(f"适应度评估完成: 平均={np.mean(scores):.4f}, 最大={np.max(scores):.4f}, 最小={np.min(scores):.4f}")
        
        return fitness_scores
    
    def _evaluate_transformer(self, policy, eval_env, num_episodes=5):
        """评估Transformer模型的适应度"""
        total_reward = 0.0
        
        for _ in range(num_episodes):
            # 重置环境
            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # 使用Transformer模型预测动作
                action_probs = policy.predict(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                action = torch.argmax(action_probs, dim=-1).item()
                
                # 执行动作
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        # 返回平均奖励
        return total_reward / num_episodes
    
    def select_parents(self, fitness_scores: Dict, num_parents: int) -> List:
        """
        父代选择方法
        
        参数:
            fitness_scores (Dict): 适应度评分字典，格式为 {"policy_id": (score, metadata)}
            num_parents (int): 选择数量
        
        返回值:
            父代策略列表 (list): 被选中的父代策略对象列表
        """
        # 检查适应度分数是否为空
        if not fitness_scores:
            raise ValueError("适应度分数字典不能为空")
        
        # 检查选择数量是否合法
        if not isinstance(num_parents, int) or num_parents <= 0:
            raise ValueError(f"num_parents必须为正整数，当前值: {num_parents}")
        
        # 如果选择数量大于种群大小，调整为种群大小
        if num_parents > len(self.population):
            num_parents = len(self.population)
            print(f"警告: 选择数量大于种群大小，已调整为 {num_parents}")
        
        # 提取策略ID和对应的适应度分数
        policy_ids = list(fitness_scores.keys())
        scores = np.array([score for score, _ in fitness_scores.values()])
        
        # 创建策略ID到策略对象的映射
        id_to_policy = {policy.id: policy for policy in self.population if hasattr(policy, 'id')}
        
        # 选择父代
        selected_parents = []
        
        if self.selection_method == "tournament":
            # 锦标赛选择
            for _ in range(num_parents):
                # 随机选择一部分策略进行比较
                tournament_size = max(2, int(len(policy_ids) * 0.2))  # 至少选择2个策略
                tournament_indices = np.random.choice(len(policy_ids), tournament_size, replace=False)
                tournament_scores = scores[tournament_indices]
                
                # 选择适应度最高的策略
                winner_idx = tournament_indices[np.argmax(tournament_scores)]
                winner_id = policy_ids[winner_idx]
                
                # 添加到选中的父代列表
                if winner_id in id_to_policy:
                    selected_parents.append(id_to_policy[winner_id])
                    
                    # 从候选池中移除已选中的策略（避免重复选择）
                    mask = np.ones(len(policy_ids), dtype=bool)
                    mask[winner_idx] = False
                    policy_ids = [policy_ids[i] for i in range(len(policy_ids)) if mask[i]]
                    scores = scores[mask]
                    
                    # 如果候选池为空，退出循环
                    if len(policy_ids) == 0:
                        break
        elif self.selection_method == "roulette":
            # 轮盘赌选择
            # 将分数转换为非负数
            min_score = min(0, np.min(scores))
            adjusted_scores = scores - min_score + 1e-6  # 确保所有分数为正
            
            # 计算选择概率
            selection_probs = adjusted_scores / np.sum(adjusted_scores)
            
            # 选择父代
            selected_indices = np.random.choice(
                len(policy_ids),
                size=num_parents,
                replace=False,
                p=selection_probs
            )
            
            for idx in selected_indices:
                policy_id = policy_ids[idx]
                if policy_id in id_to_policy:
                    selected_parents.append(id_to_policy[policy_id])
        
        print(f"父代选择完成: 选择了 {len(selected_parents)} 个策略")
        return selected_parents
    
    def evolve_population(self, parents: List, mutation_strength: float = 1.0) -> List:
        """
        进化操作方法
        
        参数:
            parents (List): 选定的父代策略列表
            mutation_strength (float): 变异强度系数，默认为1.0
        
        返回值:
            新一代种群 (list): 进化后的策略对象列表
        """
        # 检查父代列表是否为空
        if not parents:
            raise ValueError("父代列表不能为空")
        
        # 初始化新一代种群
        new_population = []
        
        # 计算精英数量
        elite_count = max(1, int(self.population_size * self.elitism_ratio))
        
        # 如果有适应度分数，选择精英
        if self.fitness_scores:
            # 按适应度分数排序
            sorted_policies = sorted(
                [(policy_id, score) for policy_id, (score, _) in self.fitness_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # 选择精英
            elite_ids = [policy_id for policy_id, _ in sorted_policies[:elite_count]]
            
            # 将精英添加到新一代
            for policy in self.population:
                if hasattr(policy, 'id') and policy.id in elite_ids:
                    # 创建精英的深拷贝
                    elite = copy.deepcopy(policy)
                    elite.id = f"elite_{elite.id}"
                    new_population.append(elite)
        
        # 计算需要生成的后代数量
        offspring_count = self.population_size - len(new_population)
        
        # 生成后代
        for i in range(offspring_count):
            # 随机选择两个父代
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
            else:
                # 如果父代不足两个，使用同一个父代
                parent1 = parent2 = parents[0]
            
            # 创建子代（从父代1复制）
            child = copy.deepcopy(parent1)
            child.id = f"offspring_{i}"
            
            # 均匀交叉
            if random.random() < self.crossover_rate and parent1 != parent2:
                self._crossover_transformer(child, parent2)
            
            # 高斯变异
            if random.random() < self.mutation_rate:
                self._mutate_transformer(child, mutation_strength)
            
            # 添加到新一代
            new_population.append(child)
        
        # 更新种群
        self.population = new_population
        
        print(f"进化操作完成: 生成了 {len(new_population)} 个新策略")
        return new_population
    
    def _crossover_transformer(self, child, parent2):
        """Transformer模型均匀交叉操作"""
        # 对于模型参数进行交叉
        if hasattr(child, 'model') and hasattr(parent2, 'model') and \
           child.model is not None and parent2.model is not None:
            # 获取两个父代的模型参数
            child_params = dict(child.model.named_parameters())
            parent2_params = dict(parent2.model.named_parameters())
            
            # 对每个参数层进行均匀交叉
            for name, param in child.model.named_parameters():
                if name in parent2_params:
                    # 生成随机掩码，决定从哪个父代继承参数
                    mask = torch.rand_like(param) > 0.5
                    # 应用掩码进行交叉
                    param.data = torch.where(mask, param.data, parent2_params[name].data)
        
        # 对于超参数进行交叉
        for attr in ['d_model', 'num_heads', 'num_layers', 'dropout_rate']:
            if hasattr(parent2, attr):
                # 50%的概率从parent2继承属性
                if random.random() > 0.5:
                    try:
                        setattr(child, attr, copy.deepcopy(getattr(parent2, attr)))
                    except:
                        pass  # 忽略无法复制的属性
    
    def _mutate_transformer(self, policy, mutation_strength):
        """Transformer模型高斯变异操作"""
        # 对模型参数进行变异
        if hasattr(policy, 'model') and policy.model is not None:
            for param in policy.model.parameters():
                # 生成高斯噪声
                noise = torch.randn_like(param) * mutation_strength * 0.1
                # 应用噪声
                param.data.add_(noise)
        
        # 对超参数进行变异
        if random.random() < 0.3:
            # 变异dropout_rate
            policy.dropout_rate = max(0.1, min(0.5, policy.dropout_rate + np.random.normal(0, 0.05 * mutation_strength)))
            # 更新模型中的dropout
            for module in policy.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = policy.dropout_rate
    
    def run_evolution(self, max_generations: int, fitness_threshold: float, eval_env: object) -> Dict:
        """
        进化迭代方法
        
        参数:
            max_generations (int): 最大进化代数
            fitness_threshold (float): 适应度终止阈值
            eval_env (object): 评估环境实例
        
        返回值:
            进化历史 (dict): 包含以下内容:
                - generation_stats: 各代适应度统计
                - best_policy: 历史最优策略对象
        """
        # 检查种群是否已初始化
        if not self.is_initialized or not self.population:
            raise ValueError("种群尚未初始化，请先调用initialize_population方法")
        
        # 检查参数
        if not isinstance(max_generations, int) or max_generations <= 0:
            raise ValueError(f"max_generations必须为正整数，当前值: {max_generations}")
        
        # 初始化进化历史
        evolution_history = {
            'generation_stats': [],
            'best_policy': None
        }
        
        # 初始化最佳适应度
        best_fitness = -float('inf')
        
        # 进化循环
        for generation in range(max_generations):
            self.generation = generation + 1
            
            print(f"\n===== 第 {self.generation} 代进化 =====")
            
            # 评估当前种群
            fitness_scores = self.evaluate_fitness(self.population, eval_env)
            
            # 计算统计信息
            scores = [score for score, _ in fitness_scores.values()]
            if scores:
                mean_fitness = np.mean(scores)
                max_fitness = np.max(scores)
                min_fitness = np.min(scores)
                
                # 记录统计信息
                generation_stats = {
                    'generation': self.generation,
                    'mean_fitness': mean_fitness,
                    'max_fitness': max_fitness,
                    'min_fitness': min_fitness,
                    'population_size': len(self.population)
                }
                
                evolution_history['generation_stats'].append(generation_stats)
                
                # 更新历史记录
                self.history['mean_fitness'].append(mean_fitness)
                self.history['max_fitness'].append(max_fitness)
                self.history['min_fitness'].append(min_fitness)
                
                # 计算种群多样性（简单实现：使用适应度标准差）
                diversity = np.std(scores)
                self.history['diversity'].append(diversity)
                
                # 打印统计信息
                print(f"统计信息: 平均适应度={mean_fitness:.4f}, 最大适应度={max_fitness:.4f}, 最小适应度={min_fitness:.4f}, 多样性={diversity:.4f}")
                
                # 更新最佳策略
                if max_fitness > best_fitness:
                    best_fitness = max_fitness
                    best_policy_id = max(fitness_scores.items(), key=lambda x: x[1][0])[0]
                    
                    # 找到最佳策略对象
                    for policy in self.population:
                        if hasattr(policy, 'id') and policy.id == best_policy_id:
                            self.best_policy = copy.deepcopy(policy)
                            evolution_history['best_policy'] = self.best_policy
                            print(f"发现新的最佳策略: ID={best_policy_id}, 适应度={max_fitness:.4f}")
                            break
                
                # 检查是否达到终止条件
                if max_fitness >= fitness_threshold:
                    print(f"达到适应度阈值 {fitness_threshold}，提前终止进化")
                    break
            
            # 如果已经是最后一代，跳过选择和进化
            if generation == max_generations - 1:
                break
            
            # 选择父代
            num_parents = max(2, int(len(self.population) * 0.4))  # 至少选择2个父代
            parents = self.select_parents(fitness_scores, num_parents)
            
            # 进化生成新一代
            mutation_strength = 1.0 - 0.5 * (generation / max_generations)  # 随着代数增加，变异强度减小
            self.evolve_population(parents, mutation_strength)
        
        print("\n===== 进化完成 =====")
        print(f"总代数: {self.generation}")
        print(f"最佳适应度: {best_fitness:.4f}")
        
        return evolution_history
    
    def save_model(self, path: str) -> None:
        """
        保存模型函数
        
        参数:
            path (str): 模型保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备保存数据
        save_data = {
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'selection_method': self.selection_method,
            'elitism_ratio': self.elitism_ratio,
            'generation': self.generation,
            'history': self.history
        }
        
        # 如果有最佳策略，保存最佳策略
        if self.best_policy is not None:
            # 创建策略保存路径
            policy_path = os.path.join(os.path.dirname(path), 'best_policy.pt')
            
            # 保存最佳策略
            self.best_policy.save_model(policy_path)
            
            # 保存策略类型和ID
            save_data['best_policy_type'] = 'TransformerTrainer'
            save_data['best_policy_id'] = self.best_policy.id if hasattr(self.best_policy, 'id') else None
            save_data['best_policy_path'] = policy_path
        
        # 保存进化器状态
        torch.save(save_data, path)
        
        print(f"模型已保存到 {path}")
    
    def load_model(self, path: str) -> None:
        """
        加载模型函数
        
        参数:
            path (str): 模型加载路径
        """
        if not os.path.exists(path):
            raise ValueError(f"模型文件 {path} 不存在")
        
        # 加载模型
        checkpoint = torch.load(path)
        
        # 更新参数
        self.population_size = checkpoint['population_size']
        self.mutation_rate = checkpoint['mutation_rate']
        self.crossover_rate = checkpoint['crossover_rate']
        self.selection_method = checkpoint['selection_method']
        self.elitism_ratio = checkpoint['elitism_ratio']
        self.generation = checkpoint['generation']
        self.history = checkpoint['history']
        
        # 如果有最佳策略信息，尝试加载最佳策略
        if 'best_policy_type' in checkpoint and 'best_policy_path' in checkpoint:
            policy_path = checkpoint['best_policy_path']
            
            if os.path.exists(policy_path):
                # 加载Transformer模型
                self.best_policy = TransformerTrainer.load_model(policy_path)
                
                # 设置策略ID
                if 'best_policy_id' in checkpoint:
                    self.best_policy.id = checkpoint['best_policy_id']
        
        print(f"模型已从 {path} 加载")
        print(f"当前代数: {self.generation}")
    
    def load_sequences(self, raw_data, batch_size=32):
        """
        加载序列数据
        
        参数:
            raw_data (dict): 原始数据
            batch_size (int): 批处理大小
            
        返回:
            DataLoader: 数据加载器
        """
        dataloader_util = DataLoaderUtil()
        data = dataloader_util.load_expert_data(raw_data, batch_size=batch_size)
        data_loader = data['dataloader']
        
        return data_loader
    
    def _train_transformer(self, policy, data_loader, epochs=10):
        """
        训练Transformer模型
        
        参数:
            policy (TransformerTrainer): Transformer模型
            data_loader (DataLoader): 数据加载器
            epochs (int): 训练轮次
            
        返回:
            dict: 训练历史
        """
        # 训练模型
        learning_rate = self.model_config.get('learning_rate', 0.001)
        history = policy.train_model(data_loader, epochs=epochs, learning_rate=learning_rate)
        
        return history
    
    def train_population(self, raw_trajectories: Dict, epochs: int = 10, batch_size: int = 64) -> Dict:
        """
        训练Transformer种群
        
        参数:
            raw_trajectories (Dict): 原始轨迹数据
            epochs (int): 训练轮次，默认为10
            batch_size (int): 批次大小，默认为64
        
        返回值:
            训练结果 (Dict): 包含每个策略的训练历史
        """
        # 检查种群是否已初始化
        if not self.is_initialized or not self.population:
            raise ValueError("种群尚未初始化，请先调用initialize_population方法")
        
        # 初始化训练结果
        training_results = {}
        
        # 从原始轨迹数据中提取必要信息
        data = raw_trajectories["data"]
        
        # 训练每个策略
        for policy in self.population:
            try:
                # 准备数据加载器
                data_loader = policy.load_sequences(data, batch_size=batch_size)
                
                # 训练Transformer模型
                history = self._train_transformer(policy, data_loader, epochs=epochs)
                
                # 记录训练历史
                training_results[policy.id] = history
                print(f"策略 {policy.id} 训练完成")
                
            except Exception as e:
                print(f"训练策略 {policy.id} 时出错: {str(e)}")
                training_results[policy.id] = {'error': str(e)}
        
        print(f"种群训练完成: 共训练了 {len(training_results)} 个策略")
        return training_results