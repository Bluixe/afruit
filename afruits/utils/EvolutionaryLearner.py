import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import copy
import random
from utils.OfflineRLearner import OfflineRLearner
from utils.BehaviorCloner import BehaviorCloner
from utils.AdversarialImitationLearner import AdversarialImitationLearner

class EvolutionaryLearner:
    """
    进化学习器类
    
    功能定义：基于种群进化机制实现策略群体持续优化
    
    核心特性：
    - 多策略并行：支持大规模策略种群管理
    - 进化算子：集成随机变异/均匀交叉/高斯变异
    - 多样性保护：自适应相似度惩罚机制
    - 断点续训：支持进化过程状态保存与恢复
    """
    
    def __init__(self,
                 population_size: int = 10,
                 model_type: str = "",
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.7,
                 selection_method: str = "tournament",
                 elitism_ratio: float = 0.1,
                 model_config: Dict = {}):
        """
        初始化进化学习器
        
        参数:
            population_size (int): 种群规模，默认为50，有效取值范围20-200
            mutation_rate (float): 变异概率，默认为0.15，有效取值范围0.05-0.3
            crossover_rate (float): 交叉概率，默认为0.7，有效取值范围0.4-0.9
            selection_method (str): 选择策略类型，默认为"tournament"，有效取值范围["tournament", "roulette"]
            elitism_ratio (float): 精英保留比例，默认为0.1，有效取值范围0.0-0.2
        """
        # 初始化参数
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.elitism_ratio = elitism_ratio
        assert model_type in ["AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator"]
        self.model_type = model_type
        
        # 初始化种群和适应度
        self.population = []
        self.fitness_scores = {}
        
        # 初始化训练器
        self.offline_rl_learner = None
        self.behavior_cloner = None
        self.adversarial_learner = None
        
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
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证population_size
        if not isinstance(self.population_size, int) or not (20 <= self.population_size <= 200):
            raise ValueError(f"population_size必须为20-200范围内的整数，当前值: {self.population_size}")
        
        # 验证mutation_rate
        if not isinstance(self.mutation_rate, float) or not (0.05 <= self.mutation_rate <= 0.3):
            raise ValueError(f"mutation_rate必须在0.05-0.3范围内，当前值: {self.mutation_rate}")
        
        # 验证crossover_rate
        if not isinstance(self.crossover_rate, float) or not (0.4 <= self.crossover_rate <= 0.9):
            raise ValueError(f"crossover_rate必须在0.4-0.9范围内，当前值: {self.crossover_rate}")
        
        # 验证selection_method
        valid_selection_methods = ["tournament", "roulette"]
        if self.selection_method not in valid_selection_methods:
            raise ValueError(f"selection_method必须为 {valid_selection_methods} 之一，当前值: {self.selection_method}")
        
        # 验证elitism_ratio
        if not isinstance(self.elitism_ratio, float) or not (0.0 <= self.elitism_ratio <= 0.2):
            raise ValueError(f"elitism_ratio必须在0.0-0.2范围内，当前值: {self.elitism_ratio}")
    
    def initialize_population(self, policy_template: object, seed: int = None) -> List:
        """
        种群初始化方法
        
        参数:
            policy_template (object): 策略原型对象
            seed (int, optional): 随机种子，默认为None
        
        返回值:
            初始种群 (list): 包含差异化策略对象的列表
        
        处理流程:
            1. 网络权重差异化初始化
            2. 超参数空间随机采样
            3. 结构变异
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        # 初始化种群列表
        population = []
        
        # 创建初始种群
        for i in range(self.population_size):
            # 复制策略模板
            policy = copy.deepcopy(policy_template)
            
            # 根据策略类型进行差异化初始化
            if isinstance(policy, OfflineRLearner):
                # 离线强化学习策略初始化
                policy.cql_weight = np.random.uniform(0.1, 1.0)
                policy.perturbation_scale = np.random.uniform(0.01, 0.2)
                
                # 重新构建模型以应用新参数
                if hasattr(policy, 'build_model'):
                    policy.build_model()
                
            elif isinstance(policy, BehaviorCloner):
                # 行为克隆策略初始化
                policy.dropout_rate = np.random.uniform(0.1, 0.5)
                policy.batch_size = np.random.choice([16, 32, 64, 128])
                
            elif isinstance(policy, AdversarialImitationLearner):
                # 对抗模仿学习策略初始化
                policy.gen_learning_rate = np.random.uniform(1e-5, 1e-3)
                policy.disc_learning_rate = np.random.uniform(1e-6, 1e-4)
                policy.gp_lambda = np.random.uniform(1.0, 20.0)
            
            # 如果策略有网络模型，进行权重随机初始化
            if hasattr(policy, 'model') and policy.model is not None:
                for param in policy.model.parameters():
                    # 添加高斯噪声到权重
                    noise = torch.randn_like(param) * 0.1
                    param.data.add_(noise)
            
            # 为策略添加唯一ID
            policy.id = f"policy_{i}"
            
            # 添加到种群
            population.append(policy)
        
        # 更新状态
        self.population = population
        self.is_initialized = True
        
        print(f"种群初始化完成: 创建了 {len(population)} 个差异化策略")
        return population
    
    def evaluate_fitness(self, population: List, eval_env: object) -> Dict:
        """
        适应度评估方法
        
        参数:
            population (List): 当前策略种群
            eval_env (object): 评估环境实例
        
        返回值:
            适应度字典 (dict): {"policy_id": (score, metadata)}
        
        功能描述:
            1. 对种群中的每个策略进行评估
            2. 计算适应度分数
            3. 返回包含策略ID和对应适应度的字典
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
                policy.id = f"policy_{id(policy)}"
            
            # 根据策略类型进行评估
            try:
                # 评估策略
                if hasattr(policy, 'evaluate'):
                    # 使用策略自带的评估方法
                    metrics = policy.evaluate(eval_env)
                    
                    # 提取主要指标作为适应度分数
                    if isinstance(metrics, dict):
                        if 'avg_reward' in metrics:
                            score = metrics['avg_reward']
                        elif 'success_rate' in metrics:
                            score = metrics['success_rate']
                        else:
                            # 使用字典中的第一个值
                            score = list(metrics.values())[0]
                    else:
                        # 如果不是字典，直接使用返回值
                        score = metrics
                else:
                    # 手动评估策略
                    score = self._manual_evaluate(policy, eval_env)
                
                # 存储适应度分数和元数据
                metadata = {
                    'type': type(policy).__name__,
                    'params': {
                        attr: getattr(policy, attr)
                        for attr in dir(policy)
                        if not attr.startswith('_') and not callable(getattr(policy, attr))
                        and attr not in ['model', 'optimizer', 'device', 'population', 'fitness_scores']
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
    
    def _manual_evaluate(self, policy, eval_env, num_episodes=5):
        """手动评估策略的适应度"""
        total_reward = 0.0
        
        for _ in range(num_episodes):
            # 重置环境
            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # 根据策略类型选择动作
                if hasattr(policy, 'step'):
                    action = policy.step(obs)
                elif hasattr(policy, 'model') and policy.model is not None:
                    # 使用模型预测动作
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        action = policy.model(obs_tensor).argmax().item()
                else:
                    # 随机动作
                    action = eval_env.action_space.sample()
                
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
        
        功能描述:
            1. 根据选择方法（锦标赛或轮盘赌）选择父代
            2. 选择概率分布基于适应度分数
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
            # 处理负分数：将所有分数平移到非负区间
            min_score = min(0, np.min(scores))
            adjusted_scores = scores - min_score + 1e-6  # 添加小值避免零分
            
            # 计算选择概率
            selection_probs = adjusted_scores / np.sum(adjusted_scores)
            
            # 选择父代
            selected_indices = np.random.choice(
                len(policy_ids),
                size=min(num_parents, len(policy_ids)),
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
        
        进化算子:
            1. 均匀交叉: 随机交换父代网络层参数
            2. 高斯变异: N(0, mutation_strength)
            3. 精英保留: 保留一定比例的最优策略
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
                self._crossover(child, parent2)
            
            # 高斯变异
            if random.random() < self.mutation_rate:
                self._mutate(child, mutation_strength)
            
            # 添加到新一代
            new_population.append(child)
        
        # 更新种群
        self.population = new_population
        
        print(f"进化操作完成: 生成了 {len(new_population)} 个新策略")
        return new_population
    
    def _crossover(self, child, parent2):
        """均匀交叉操作"""
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
        for attr in dir(child):
            # 只考虑非私有、非方法的属性
            if not attr.startswith('_') and not callable(getattr(child, attr)) and \
               attr not in ['model', 'optimizer', 'device', 'population', 'fitness_scores', 'id']:
                # 50%的概率从parent2继承属性
                if random.random() > 0.5 and hasattr(parent2, attr):
                    try:
                        setattr(child, attr, copy.deepcopy(getattr(parent2, attr)))
                    except:
                        pass  # 忽略无法复制的属性
    
    def _mutate(self, policy, mutation_strength):
        """高斯变异操作"""
        # 对模型参数进行变异
        if hasattr(policy, 'model') and policy.model is not None:
            for param in policy.model.parameters():
                # 生成高斯噪声
                noise = torch.randn_like(param) * mutation_strength * 0.1
                # 应用噪声
                param.data.add_(noise)
        
        # 对超参数进行变异
        if isinstance(policy, OfflineRLearner):
            # 变异离线强化学习策略的超参数
            if random.random() < 0.3:
                policy.cql_weight = max(0.1, min(1.0, policy.cql_weight + np.random.normal(0, 0.1 * mutation_strength)))
            if random.random() < 0.3:
                policy.perturbation_scale = max(0.01, min(0.2, policy.perturbation_scale + np.random.normal(0, 0.02 * mutation_strength)))
            
        elif isinstance(policy, BehaviorCloner):
            # 变异行为克隆策略的超参数
            if random.random() < 0.3:
                policy.dropout_rate = max(0.1, min(0.5, policy.dropout_rate + np.random.normal(0, 0.05 * mutation_strength)))
            
        elif isinstance(policy, AdversarialImitationLearner):
            # 变异对抗模仿学习策略的超参数
            if random.random() < 0.3:
                policy.gen_learning_rate = max(1e-5, min(1e-3, policy.gen_learning_rate * (1 + np.random.normal(0, 0.1 * mutation_strength))))
            if random.random() < 0.3:
                policy.disc_learning_rate = max(1e-6, min(1e-4, policy.disc_learning_rate * (1 + np.random.normal(0, 0.1 * mutation_strength))))
            if random.random() < 0.3:
                policy.gp_lambda = max(1.0, min(20.0, policy.gp_lambda + np.random.normal(0, 1.0 * mutation_strength)))
    
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
        
        功能描述:
            1. 执行完整的进化过程
            2. 记录每代的适应度统计信息
            3. 保存历史最优策略
            4. 支持断点续训和提前终止
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
            
            # 根据策略类型保存
            if isinstance(self.best_policy, OfflineRLearner):
                self.best_policy.save_model(policy_path)
            elif isinstance(self.best_policy, BehaviorCloner) and hasattr(self.best_policy, 'model'):
                torch.save(self.best_policy.model.state_dict(), policy_path)
            elif isinstance(self.best_policy, AdversarialImitationLearner):
                # 保存生成器和判别器
                if hasattr(self.best_policy, 'generator') and self.best_policy.generator is not None:
                    torch.save(self.best_policy.generator.state_dict(), os.path.join(os.path.dirname(path), 'generator.pt'))
                if hasattr(self.best_policy, 'discriminator') and self.best_policy.discriminator is not None:
                    torch.save(self.best_policy.discriminator.state_dict(), os.path.join(os.path.dirname(path), 'discriminator.pt'))
            
            # 保存策略类型和ID
            save_data['best_policy_type'] = type(self.best_policy).__name__
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
            policy_type = checkpoint['best_policy_type']
            policy_path = checkpoint['best_policy_path']
            
            if os.path.exists(policy_path):
                # 根据策略类型加载
                if policy_type == 'OfflineRLearner':
                    self.best_policy = OfflineRLearner()
                    self.best_policy.load_model(policy_path)
                elif policy_type == 'BehaviorCloner':
                    self.best_policy = BehaviorCloner()
                    if hasattr(self.best_policy, 'model') and self.best_policy.model is not None:
                        self.best_policy.model.load_state_dict(torch.load(policy_path))
                elif policy_type == 'AdversarialImitationLearner':
                    self.best_policy = AdversarialImitationLearner()
                    # 加载生成器和判别器
                    generator_path = os.path.join(os.path.dirname(path), 'generator.pt')
                    discriminator_path = os.path.join(os.path.dirname(path), 'discriminator.pt')
                    
                    if os.path.exists(generator_path) and hasattr(self.best_policy, 'generator'):
                        self.best_policy.generator.load_state_dict(torch.load(generator_path))
                    if os.path.exists(discriminator_path) and hasattr(self.best_policy, 'discriminator'):
                        self.best_policy.discriminator.load_state_dict(torch.load(discriminator_path))
                
                # 设置策略ID
                if 'best_policy_id' in checkpoint:
                    self.best_policy.id = checkpoint['best_policy_id']
        
        print(f"模型已从 {path} 加载")
        print(f"当前代数: {self.generation}")
    
    def create_mixed_population(self, raw_trajectories: Dict, env_config: Dict = None) -> List:
        """
        创建混合策略种群
        
        参数:
            raw_trajectories (Dict): 原始轨迹数据
            env_config (Dict, optional): 环境配置
        
        返回值:
            混合策略种群 (list): 包含不同类型策略的列表
        
        功能描述:
            1. 创建包含三种不同学习方法的策略种群
            2. 使用相同的轨迹数据初始化不同类型的策略
            3. 返回混合策略种群
        """
        # 初始化种群
        mixed_population = []
        
        # 确定每种策略的数量
        offline_rl_count = int(self.population_size * 0.4)  # 40%
        bc_count = int(self.population_size * 0.4)          # 40%
        adversarial_count = self.population_size - offline_rl_count - bc_count  # 20%
        
        # 处理环境配置
        if env_config is None:
            env_config = {}
        
        # 提取状态和动作维度
        state_dim = env_config.get('state_dim', 7)
        action_dim = env_config.get('action_dim', 7)
        
        # 预处理数据
        processed_data = self._preprocess_trajectories(raw_trajectories)
        
        # 创建离线强化学习策略
        print(f"创建 {offline_rl_count} 个离线强化学习策略...")
        for i in range(offline_rl_count):
            # 创建策略
            policy = OfflineRLearner(
                state_dim=state_dim,
                action_dim=action_dim,
                cql_weight=np.random.uniform(0.1, 1.0),
                perturbation_scale=np.random.uniform(0.01, 0.2)
            )
            
            # 构建模型
            policy.build_model()
            
            # 设置ID
            policy.id = f"offline_rl_{i}"
            
            # 添加到种群
            mixed_population.append(policy)
        
        # 创建行为克隆策略
        print(f"创建 {bc_count} 个行为克隆策略...")
        for i in range(bc_count):
            # 创建策略
            policy = BehaviorCloner(
                batch_size=np.random.choice([16, 32, 64, 128]),
                network_type="MLP",
                dropout_rate=np.random.uniform(0.1, 0.5)
            )
            
            # 设置ID
            policy.id = f"bc_{i}"
            
            # 添加到种群
            mixed_population.append(policy)
        
        # 创建对抗模仿学习策略
        print(f"创建 {adversarial_count} 个对抗模仿学习策略...")
        for i in range(adversarial_count):
            # 创建策略
            policy = AdversarialImitationLearner(
                state_dim=state_dim,
                action_dim=action_dim,
                gen_learning_rate=np.random.uniform(1e-5, 1e-3),
                disc_learning_rate=np.random.uniform(1e-6, 1e-4),
                gp_lambda=np.random.uniform(1.0, 20.0)
            )
            
            # 设置ID
            policy.id = f"adversarial_{i}"
            
            # 添加到种群
            mixed_population.append(policy)
        
        # 更新种群
        self.population = mixed_population
        self.is_initialized = True
        
        print(f"混合种群创建完成: 总共 {len(mixed_population)} 个策略")
        return mixed_population
    
    def _preprocess_trajectories(self, raw_trajectories: Dict) -> Dict:
        """预处理轨迹数据"""
        processed_data = {}
        
        # 提取状态和动作
        states = []
        actions = []
        
        for traj_id, trajectory in raw_trajectories.items():
            if 'states' in trajectory and 'actions' in trajectory:
                states.extend(trajectory['states'])
                actions.extend(trajectory['actions'])
        
        if states and actions:
            processed_data['states'] = np.array(states)
            processed_data['actions'] = np.array(actions)
        
        return processed_data
    
    def _train_autoencoder(self, model, train_loader, epochs: int = 10, learning_rate: float = 1e-4) -> Dict:
        """自编码器模型训练方法"""
        print("使用自编码器训练方法")
        training_history = model.train_model(train_loader, epochs=epochs, learning_rate=learning_rate)
        return training_history
    
    def _train_transformer(self, model, train_loader, epochs: int = 10, learning_rate: float = 1e-4) -> Dict:
        """Transformer模型训练方法"""
        print("使用Transformer训练方法")
        training_history = model.train_model(train_loader, epochs=epochs, learning_rate=learning_rate)
        return training_history
    
    def _train_diffusion(self, model, train_loader, epochs: int = 10, learning_rate: float = 1e-4) -> Dict:
        """扩散模型训练方法"""
        print("使用扩散模型训练方法")
        training_history = model.train(train_loader, epochs=epochs, learning_rate=learning_rate)
        return training_history
    
    def _train_vae(self, model, train_loader, epochs: int = 10, learning_rate: float = 1e-4) -> Dict:
        """VAE训练方法"""
        print("使用VAE训练方法")
        training_history = model.train(train_loader, epochs=epochs, learning_rate=learning_rate)
        return training_history
    
    def train_population(self, raw_trajectories: Dict, epochs: int = 10, batch_size: int = 64) -> Dict:
        """
        训练种群中的策略
        
        参数:
            raw_trajectories (Dict): 原始轨迹数据
            epochs (int): 训练轮次，默认为10
            batch_size (int): 批次大小，默认为64
        
        返回值:
            训练结果 (Dict): 包含每个策略的训练历史
        """
        # 检查种群是否已初始化
        if not self.is_initialized or not self.population:
            raise ValueError("种群尚未初始化，请先调用initialize_population或create_mixed_population方法")
        
        # 初始化训练结果
        training_results = {}
        
        # 预处理数据
        processed_data = self._preprocess_trajectories(raw_trajectories)
        
        # 从原始轨迹数据中提取必要信息
        data, state_dim, action_dim = raw_trajectories["data"], raw_trajectories["state_dim"], raw_trajectories["action_dim"]
        seq_length = raw_trajectories.get("traj_length", None)
        
        # 训练每个策略
        for policy in self.population:
            try:
                if self.model_type == "AutoencoderModel":
                    # 准备数据加载器
                    data_loader = policy.load_sequences(data, batch_size=batch_size)
                    # 训练自编码器模型
                    history = self._train_autoencoder(policy, data_loader, epochs=epochs)
                    
                elif self.model_type == "TransformerModel":
                    # 准备数据加载器
                    data_loader = policy.load_sequences(data, batch_size=batch_size)
                    # 训练Transformer模型
                    history = self._train_transformer(policy, data_loader, epochs=epochs)
                    
                elif self.model_type == "DiffusionTrajGenerator":
                    # 准备数据加载器
                    data_loader = policy.load_dataset(data, batch_size=batch_size)
                    # 训练扩散模型
                    history = self._train_diffusion(policy, data_loader, epochs=epochs)
                    
                elif self.model_type == "VAETrajGenerator":
                    # 准备数据加载器
                    data_loader = policy.load_dataset(data, batch_size=batch_size)
                    # 训练VAE模型
                    history = self._train_vae(policy, data_loader, epochs=epochs)
                
                # 记录训练历史
                training_results[policy.id] = history
                print(f"策略 {policy.id} 训练完成")
                
            except Exception as e:
                print(f"训练策略 {policy.id} 时出错: {str(e)}")
                training_results[policy.id] = {'error': str(e)}
        
        print(f"种群训练完成: 共训练了 {len(training_results)} 个策略")
        return training_results


# 示例用法
def example_usage():
    """
    EvolutionaryLearner 使用示例
    """
    # 创建进化学习器
    learner = EvolutionaryLearner(
        population_size=50,
        mutation_rate=0.15,
        crossover_rate=0.7,
        selection_method="tournament",
        elitism_ratio=0.1
    )
    
    # 假设我们有一些轨迹数据
    raw_trajectories = {
        'traj_1': {
            'states': [np.random.rand(7) for _ in range(100)],
            'actions': [np.random.randint(0, 7) for _ in range(100)],
            'rewards': [np.random.rand() for _ in range(100)]
        },
        'traj_2': {
            'states': [np.random.rand(7) for _ in range(100)],
            'actions': [np.random.randint(0, 7) for _ in range(100)],
            'rewards': [np.random.rand() for _ in range(100)]
        }
    }
    
    # 环境配置
    env_config = {
        'state_dim': 7,
        'action_dim': 7
    }
    
    # 创建混合种群
    population = learner.create_mixed_population(raw_trajectories, env_config)
    
    # 训练种群
    training_results = learner.train_population(raw_trajectories, epochs=5, batch_size=32)
    
    # 假设我们有一个评估环境
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'sample': lambda: np.random.randint(0, 7)})
            
        def reset(self):
            return np.random.rand(7)
            
        def step(self, action):
            next_obs = np.random.rand(7)
            reward = np.random.rand()
            done = np.random.random() > 0.9
            info = {'success': np.random.random() > 0.7}
            return next_obs, reward, done, info
    
    # 创建评估环境
    eval_env = DummyEnv()
    
    # 运行进化
    evolution_history = learner.run_evolution(
        max_generations=10,
        fitness_threshold=0.9,
        eval_env=eval_env
    )
    
    # 保存模型
    learner.save_model("models/evolutionary_learner.pt")
    
    # 加载模型
    new_learner = EvolutionaryLearner()
    new_learner.load_model("models/evolutionary_learner.pt")
    
    # 获取最佳策略
    best_policy = evolution_history['best_policy']
    print(f"最佳策略ID: {best_policy.id if hasattr(best_policy, 'id') else 'Unknown'}")
    
    return best_policy

# 如果直接运行此文件
if __name__ == "__main__":
    example_usage()