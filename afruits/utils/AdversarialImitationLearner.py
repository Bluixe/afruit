import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import copy
from afruits.utils.DataLoader import DataLoaderUtil

class AdversarialImitationLearner:
    """
    对抗模仿学习器类 (GAIL算法实现)
    
    功能描述：实现基于生成对抗网络的模仿学习算法，通过判别器区分专家行为和生成行为
    
    核心功能：
    - 数据预处理：支持专家轨迹数据处理与特征提取
    - 模型构建：支持生成器和判别器网络结构
    - 策略训练：实现GAIL算法的训练过程
    - 策略评估：提供多种评估指标与可视化工具
    """
    
    def __init__(self,
                 gen_learning_rate: float = 1e-4,
                 disc_learning_rate: float = 5e-5,
                 update_ratio: int = 5,
                 gp_lambda: float = 10.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化对抗模仿学习器
        
        参数:
            gen_learning_rate (float): 生成器学习率，默认为1e-4，有效取值范围1e-5~1e-3
            disc_learning_rate (float): 判别器学习率，默认为5e-5，有效取值范围1e-6~1e-4
            update_ratio (int): 更新比例，默认为5，有效取值范围1~10
            gp_lambda (float): 梯度惩罚系数，默认为10.0，有效取值范围1.0~100.0
            device (str): 训练设备，默认为"cuda"如果可用，否则为"cpu"
        """
        # 初始化参数
        self.state_dim = None
        self.action_dim = None
        self.gen_learning_rate = gen_learning_rate
        self.disc_learning_rate = disc_learning_rate
        self.update_ratio = update_ratio
        self.gp_lambda = gp_lambda
        self.device = device

        # 保存配置以便持久化
        self.config_to_save = {
            'gen_learning_rate': self.gen_learning_rate,
            'disc_learning_rate': self.disc_learning_rate,
            'update_ratio': self.update_ratio,
            'gp_lambda': self.gp_lambda
        }
        
        # 初始化模型和优化器
        self.generator = None
        self.discriminator = None
        self.gen_optimizer = None
        self.disc_optimizer = None
        
        # 初始化训练状态
        self.is_trained = False

        self.dataloader_util = DataLoaderUtil()
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证gen_learning_rate
        if not isinstance(self.gen_learning_rate, float) or not (1e-5 <= self.gen_learning_rate <= 1e-3):
            raise ValueError(f"gen_learning_rate必须在1e-5~1e-3范围内，当前值: {self.gen_learning_rate}")
        
        # 验证disc_learning_rate
        if not isinstance(self.disc_learning_rate, float) or not (1e-6 <= self.disc_learning_rate <= 1e-4):
            raise ValueError(f"disc_learning_rate必须在1e-6~1e-4范围内，当前值: {self.disc_learning_rate}")
        
        # 验证update_ratio
        if not isinstance(self.update_ratio, int) or not (1 <= self.update_ratio <= 10):
            raise ValueError(f"update_ratio必须为1-10范围内的整数，当前值: {self.update_ratio}")
        
        # 验证gp_lambda
        if not isinstance(self.gp_lambda, float) or not (1.0 <= self.gp_lambda <= 100.0):
            raise ValueError(f"gp_lambda必须在1.0-100.0范围内，当前值: {self.gp_lambda}")
    
    def preprocess_data(self, expert_trajectories: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理函数
        
        参数:
            expert_trajectories (Dict): 专家轨迹数据集（例如：原始轨迹数据，包含状态-动作序列）
        
        返回值:
            expert_states: 专家状态数据（三维张量）
            expert_actions: 专家动作数据（二维张量）
        
        功能描述:
            1. 处理专家轨迹数据，提取状态和动作
            2. 标准化特征
            3. 构建数据集用于训练判别器
        """
        expert_states, expert_actions = self.dataloader_util.load_bc_gail_data(expert_trajectories)
        return expert_states, expert_actions
    
    def build_models(self, state_dim: int, action_dim: int, generator_args: Dict = None, discriminator_args: Dict = None) -> Tuple[nn.Module, nn.Module]:
        """
        模型构建函数
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            generator_args (Dict, optional): 生成器参数字典
            discriminator_args (Dict, optional): 判别器参数字典
        
        返回值:
            generator: 生成器模型（策略网络）
            discriminator: 判别器模型
        
        功能描述:
            1. 构建生成器网络（策略网络）
            2. 构建判别器网络
            3. 设置优化器
        """
        # 设置状态和动作维度
        if type(state_dim) == tuple:
            assert len(state_dim) == 1, "仅支持一维状态输入" 
            state_dim = state_dim[0]
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 更新保存配置中的维度信息
        if not hasattr(self, 'config_to_save'):
            self.config_to_save = {}
        self.config_to_save.update({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        })
        
        # 验证状态和动作维度
        if not isinstance(self.state_dim, int) or not (1 <= self.state_dim <= 1024):
            raise ValueError(f"state_dim必须为1-1024范围内的整数，当前值: {self.state_dim}")
        
        if not isinstance(self.action_dim, int) or not (1 <= self.action_dim <= 100):
            raise ValueError(f"action_dim必须为1-100范围内的整数，当前值: {self.action_dim}")
        
        # 设置默认参数
        if generator_args is None:
            generator_args = {}
        if discriminator_args is None:
            discriminator_args = {}
        
        # 生成器默认参数
        gen_default_args = {
            "hidden_dim": 128,
            "n_layers": 2,
            "activation": "relu",
            "dropout": 0.1
        }
        
        # 判别器默认参数
        disc_default_args = {
            "hidden_dim": 128,
            "n_layers": 2,
            "activation": "relu",
            "dropout": 0.1
        }
        
        # 更新参数
        for key, value in gen_default_args.items():
            if key not in generator_args:
                generator_args[key] = value
        
        for key, value in disc_default_args.items():
            if key not in discriminator_args:
                discriminator_args[key] = value
        
        # 构建生成器（策略网络）
        self.generator = self._build_generator(generator_args)
        
        # 构建判别器
        self.discriminator = self._build_discriminator(discriminator_args)
        
        # 设置优化器
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.gen_learning_rate)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.disc_learning_rate)
        
        print("模型构建完成")
        return self.generator, self.discriminator
    
    def _build_generator(self, args: Dict) -> nn.Module:
        """
        构建生成器（策略网络）
        
        参数:
            args (Dict): 生成器参数字典
        
        返回值:
            generator: 生成器模型
        """
        if self.state_dim is None:
            raise ValueError("state_dim未设置，请先调用build_models方法")
        if self.action_dim is None:
            raise ValueError("action_dim未设置，请先调用build_models方法")
            
        hidden_dim = args["hidden_dim"]
        n_layers = args["n_layers"]
        dropout = args["dropout"]
        
        # 创建MLP策略网络
        layers = [nn.Linear(self.state_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层 - 动作分布参数（概率分布）
        layers.append(nn.Linear(hidden_dim, self.action_dim))
        layers.append(nn.Softmax(dim=1))  # 添加Softmax确保输出是概率分布
        
        # 创建模型
        generator = nn.Sequential(*layers)
        generator.to(self.device)
        
        return generator
    
    def _build_discriminator(self, args: Dict) -> nn.Module:
        """
        构建判别器
        
        参数:
            args (Dict): 判别器参数字典
        
        返回值:
            discriminator: 判别器模型
        """
        if self.state_dim is None:
            raise ValueError("state_dim未设置，请先调用build_models方法")
        if self.action_dim is None:
            raise ValueError("action_dim未设置，请先调用build_models方法")
            
        hidden_dim = args["hidden_dim"]
        n_layers = args["n_layers"]
        dropout = args["dropout"]
        
        # 创建判别器网络
        layers = [nn.Linear(self.state_dim + self.action_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层 - 二分类（真/假）
        layers.append(nn.Linear(hidden_dim, 1))
        
        # 创建模型
        discriminator = nn.Sequential(*layers)
        discriminator.to(self.device)
        
        return discriminator
    
    def _compute_gradient_penalty(self, discriminator, real_data, fake_data):
        """计算梯度惩罚项"""
        batch_size = real_data.size(0)
        
        # 创建随机插值系数
        alpha = torch.rand(batch_size, 1, device=self.device)
        
        # 在真实数据和生成数据之间进行插值
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # 计算判别器在插值点的输出
        disc_interpolates = discriminator(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train(self, expert_data: Dict, batch_size: int = 64, epochs: int = 100) -> Dict:
        """
        训练函数
        
        参数:
            expert_data (Dict): 处理后的专家数据
            batch_size (int): 训练批次大小，默认为64
            epochs (int): 训练轮次，默认为100
            generator_args (Dict, optional): 生成器参数字典
            discriminator_args (Dict, optional): 判别器参数字典
        
        返回值:
            Dict: 训练历史记录，包含以下指标:
                - gen_loss: 生成器损失
                - disc_loss: 判别器损失
                - wasserstein_dist: Wasserstein距离
        
        功能描述:
            1. 使用GAIL算法训练生成器和判别器
            2. 实现Wasserstein GAN with Gradient Penalty (WGAN-GP)
            3. 记录训练过程中的损失值
            4. 返回训练历史记录
        """
        # 检查输入数据
        if 'states' not in expert_data or 'actions' not in expert_data:
            raise ValueError("expert_data必须包含'expert_states'和'expert_actions'")
        
        expert_states = expert_data['states']
        expert_actions = expert_data['actions']
        
        # 转换为PyTorch张量
        expert_states_tensor = torch.FloatTensor(expert_states).to(self.device)
        # 处理离散动作，转换为one-hot编码
        if len(expert_actions.shape) == 1:  # 离散动作 (batch_size,)
            # 创建one-hot编码
            expert_actions_one_hot = np.zeros((expert_actions.shape[0], self.action_dim))
            for i, action in enumerate(expert_actions):
                expert_actions_one_hot[i, int(action)] = 1.0
            expert_actions_tensor = torch.FloatTensor(expert_actions_one_hot).to(self.device)
        else:  # 已经是连续形式 (batch_size, action_dim)
            expert_actions_tensor = torch.FloatTensor(expert_actions).to(self.device)
        
        # 创建数据集和数据加载器
        expert_dataset = TensorDataset(expert_states_tensor, expert_actions_tensor)
        expert_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练历史记录
        history = {
            'gen_loss': [],
            'disc_loss': [],
            'wasserstein_dist': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            epoch_gen_loss = 0.0
            epoch_disc_loss = 0.0
            epoch_wasserstein_dist = 0.0
            
            for expert_states_batch, expert_actions_batch in expert_loader:
                batch_size = expert_states_batch.size(0)
                
                # 训练判别器
                for _ in range(self.update_ratio):
                    self.disc_optimizer.zero_grad()
                    
                    # 生成动作
                    with torch.no_grad():
                        generated_actions = self.generator(expert_states_batch)  # (batch_size, action_dim)
                    
                    # 真实数据
                    real_data = torch.cat([expert_states_batch, expert_actions_batch], dim=1)
                    # 生成数据
                    fake_data = torch.cat([expert_states_batch, generated_actions], dim=1)
                    
                    # 判别器输出
                    real_output = self.discriminator(real_data)
                    fake_output = self.discriminator(fake_data)
                    
                    # 计算Wasserstein距离
                    wasserstein_dist = fake_output.mean() - real_output.mean()
                    
                    # 计算梯度惩罚
                    gradient_penalty = self._compute_gradient_penalty(self.discriminator, real_data, fake_data)
                    
                    # 判别器损失
                    disc_loss = wasserstein_dist + self.gp_lambda * gradient_penalty
                    
                    # 反向传播和优化
                    disc_loss.backward()
                    self.disc_optimizer.step()
                    
                    epoch_disc_loss += disc_loss.item()
                    epoch_wasserstein_dist += wasserstein_dist.item()
                
                # 训练生成器
                self.gen_optimizer.zero_grad()
                
                # 生成动作
                generated_actions = self.generator(expert_states_batch)  # (batch_size, action_dim)
                
                # 生成数据
                fake_data = torch.cat([expert_states_batch, generated_actions], dim=1)
                
                # 判别器输出
                fake_output = self.discriminator(fake_data)
                
                # 生成器损失 - 最小化判别器对生成数据的负输出
                gen_loss = -fake_output.mean()
                
                # 反向传播和优化
                gen_loss.backward()
                self.gen_optimizer.step()
                
                epoch_gen_loss += gen_loss.item()
            
            # 计算平均损失
            avg_gen_loss = epoch_gen_loss / len(expert_loader)
            avg_disc_loss = epoch_disc_loss / (len(expert_loader) * self.update_ratio)
            avg_wasserstein_dist = epoch_wasserstein_dist / (len(expert_loader) * self.update_ratio)
            
            # 记录历史
            history['gen_loss'].append(avg_gen_loss)
            history['disc_loss'].append(avg_disc_loss)
            history['wasserstein_dist'].append(avg_wasserstein_dist)
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Wasserstein Dist: {avg_wasserstein_dist:.4f}")
        
        # 标记模型已训练
        self.is_trained = True
        
        return history
    
    def evaluate(self, test_env, num_episodes: int = 10) -> Dict:
        """
        评估函数（离散动作）
        
        更新点:
        - 使用argmax从动作概率中选出离散动作索引
        - 若环境提供专家动作(get_expert_action)，计算动作分类准确率accuracy
        - 保留原有判别器准确率disc_accuracy与其他指标
        
        参数:
            test_env: 测试环境（可选实现 get_expert_action(obs)->int）
            num_episodes (int): 测试轮次，默认为10
        
        返回值:
            metrics (Dict): 包含以下评估指标:
                - disc_accuracy: 判别器区分专家/生成行为的准确率
                - accuracy: 策略动作分类准确率（基于专家动作，若不可用则为None）
                - policy_jsd: 策略Jensen-Shannon散度（占位）
                - trajectory_overlap: 轨迹重叠度（占位）
        """
        # 维度检查
        if self.state_dim is None or self.action_dim is None:
            raise ValueError("state_dim或action_dim未设置，请先调用train/build_models方法")
        
        metrics = {
            'disc_accuracy': 0.0,
            'accuracy': None,
            'policy_jsd': 0.0,
            'trajectory_overlap': 0.0
        }
        
        total_reward = 0.0
        # 判别器区分准确率统计
        expert_correct = 0
        generated_correct = 0
        disc_total = 0
        
        # 策略动作准确率统计
        correct_predictions = 0
        total_samples = 0
        errors = []
        
        generated_trajectories = []
        
        for episode in range(num_episodes):
            obs = test_env.reset()
            done = False
            episode_reward = 0.0
            trajectory = []
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # 生成离散动作（argmax）
                with torch.no_grad():
                    action_probs = self.generator(obs_tensor)  # (1, action_dim)
                    action_idx = int(torch.argmax(action_probs, dim=1).item())
                    action_tensor = action_probs  # for discriminator concatenation
                    action = action_probs.cpu().numpy()[0]
                
                # 在交互前获取专家动作（若可用），用于分类准确率
                expert_action_idx = None
                if hasattr(test_env, 'get_expert_action'):
                    try:
                        expert_action_idx = int(test_env.get_expert_action(obs))
                    except Exception:
                        expert_action_idx = None
                
                # 环境交互（离散索引）
                next_obs, reward, done, info = test_env.step(action_idx)
                
                # 记录轨迹
                trajectory.append((obs, action_idx, action))
                
                # 累积奖励
                episode_reward += float(reward)
                
                # 判别器评估（基于当前obs_tensor与action_tensor）
                with torch.no_grad():
                    # 专家样本
                    if expert_action_idx is not None:
                        expert_action_vec = np.zeros(self.action_dim, dtype=np.float32)
                        expert_action_vec[expert_action_idx] = 1.0
                        expert_data = torch.cat(
                            [obs_tensor, torch.from_numpy(expert_action_vec).unsqueeze(0).to(self.device)],
                            dim=1
                        )
                        expert_output = self.discriminator(expert_data)
                        if expert_output.item() > 0:
                            expert_correct += 1
                    
                    # 生成样本
                    generated_data = torch.cat([obs_tensor, action_tensor], dim=1)
                    generated_output = self.discriminator(generated_data)
                    if generated_output.item() <= 0:
                        generated_correct += 1
                    
                    disc_total += 1
                
                # 统计策略动作准确率
                if expert_action_idx is not None:
                    errors.append(abs(action_idx - expert_action_idx))
                    if action_idx == expert_action_idx:
                        correct_predictions += 1
                    total_samples += 1
                
                # 更新观测
                obs = next_obs
            
            total_reward += episode_reward
            generated_trajectories.append(trajectory)
        
        # 判别器准确率
        if disc_total > 0:
            metrics['disc_accuracy'] = (expert_correct + generated_correct) / (2 * disc_total)
        
        # 策略分类准确率
        if total_samples > 0:
            metrics['accuracy'] = correct_predictions / total_samples
        
        # 其他占位指标
        metrics['policy_jsd'] = 0.5
        metrics['trajectory_overlap'] = 0.6
        
        avg_reward = total_reward / max(num_episodes, 1)
        if metrics['accuracy'] is not None:
            print(f"评估完成: 平均奖励 = {avg_reward:.2f}, 判别器准确率 = {metrics['disc_accuracy']:.4f}, 策略准确率 = {metrics['accuracy']:.4f}")
        else:
            print(f"评估完成: 平均奖励 = {avg_reward:.2f}, 判别器准确率 = {metrics['disc_accuracy']:.4f}, 策略准确率 = N/A")
        
        return metrics
    
    def save_model(self, save_path = None) -> None:
        """
        保存模型函数
        
        参数:
            path (str): 模型保存路径
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("模型尚未构建，无法保存")
        
        if save_path is None:
            save_path = f"models/gail.pt"
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备保存数据
        model_state = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'config': {k:v for k,v in self.config_to_save.items()}
        }
        
        # 保存模型
        torch.save(model_state, save_path)
        import json
        config_path = os.path.splitext(save_path)[0] + '_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_state['config'], f, ensure_ascii=False, indent=4)
        
        print(f"模型已保存到 {save_path}")
        print(f"配置已保存至: {config_path}")
    
    @staticmethod
    def load_model(load_path, device: torch.device = None) -> 'AdversarialImitationLearner':
        """
        加载模型函数
        
        参数:
            path (str): 模型加载路径
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_path is None:
            load_path = f"models/gail.pt"
        
        # 加载模型
        checkpoint = torch.load(load_path, map_location=device)
        config = checkpoint['config']
        
        # 创建实例并重建模型
        model = AdversarialImitationLearner(
            gen_learning_rate = config['gen_learning_rate'],
            disc_learning_rate = config['disc_learning_rate'],
            update_ratio = config['update_ratio'],
            gp_lambda = config['gp_lambda'],
            device = device
        )
        
        model.build_models(config['state_dim'], config['action_dim'])
        
        # 加载模型参数
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print(f"成功加载模型: {load_path}")
        return model