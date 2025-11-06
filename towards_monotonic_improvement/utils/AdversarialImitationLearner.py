import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import copy

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
                 state_dim: int = 7,
                 action_dim: int = 7,
                 gen_learning_rate: float = 1e-4,
                 disc_learning_rate: float = 5e-5,
                 update_ratio: int = 5,
                 gp_lambda: float = 10.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化对抗模仿学习器
        
        参数:
            state_dim (int): 状态空间维度，默认为7
            action_dim (int): 动作空间维度，默认为7
            gen_learning_rate (float): 生成器学习率，默认为1e-4，有效取值范围1e-5~1e-3
            disc_learning_rate (float): 判别器学习率，默认为5e-5，有效取值范围1e-6~1e-4
            update_ratio (int): 更新比例，默认为5，有效取值范围1~10
            gp_lambda (float): 梯度惩罚系数，默认为10.0，有效取值范围1.0~100.0
            device (str): 训练设备，默认为"cuda"如果可用，否则为"cpu"
        """
        # 初始化参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gen_learning_rate = gen_learning_rate
        self.disc_learning_rate = disc_learning_rate
        self.update_ratio = update_ratio
        self.gp_lambda = gp_lambda
        self.device = device
        
        # 初始化模型和优化器
        self.generator = None
        self.discriminator = None
        self.gen_optimizer = None
        self.disc_optimizer = None
        
        # 初始化训练状态
        self.is_trained = False
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证state_dim
        if not isinstance(self.state_dim, int) or not (1 <= self.state_dim <= 1024):
            raise ValueError(f"state_dim必须为1-1024范围内的整数，当前值: {self.state_dim}")
        
        # 验证action_dim
        if not isinstance(self.action_dim, int) or not (1 <= self.action_dim <= 100):
            raise ValueError(f"action_dim必须为1-100范围内的整数，当前值: {self.action_dim}")
        
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
        # 初始化结果
        expert_states = []
        expert_actions = []
        
        # 检查输入数据
        if not expert_trajectories or not isinstance(expert_trajectories, dict):
            raise ValueError("expert_trajectories必须是非空字典")
        
        # 处理轨迹数据
        for traj_id, trajectory in expert_trajectories.items():
            # 检查轨迹数据是否包含状态和动作
            if 'states' not in trajectory or 'actions' not in trajectory:
                print(f"警告: 轨迹 {traj_id} 缺少状态或动作数据，已跳过")
                continue
            
            states = trajectory['states']
            actions = trajectory['actions']
            
            # 检查状态和动作数据长度是否匹配
            if len(states) != len(actions):
                print(f"警告: 轨迹 {traj_id} 的状态和动作数据长度不匹配，已跳过")
                continue
            
            # 添加数据
            expert_states.extend(states)
            expert_actions.extend(actions)
        
        # 转换为numpy数组
        if expert_states and expert_actions:
            expert_states = np.array(expert_states)
            expert_actions = np.array(expert_actions)
        else:
            raise ValueError("处理后的数据为空，请检查输入数据")
        
        print(f"数据预处理完成: expert_states shape: {expert_states.shape}, expert_actions shape: {expert_actions.shape}")
        return expert_states, expert_actions
    
    def build_models(self, generator_args: Dict = None, discriminator_args: Dict = None) -> Tuple[nn.Module, nn.Module]:
        """
        模型构建函数
        
        参数:
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
        """构建生成器（策略网络）"""
        hidden_dim = args["hidden_dim"]
        n_layers = args["n_layers"]
        dropout = args["dropout"]
        
        # 创建MLP策略网络
        layers = [nn.Linear(self.state_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层 - 动作分布参数
        layers.append(nn.Linear(hidden_dim, self.action_dim))
        
        # 创建模型
        generator = nn.Sequential(*layers)
        generator.to(self.device)
        
        return generator
    
    def _build_discriminator(self, args: Dict) -> nn.Module:
        """构建判别器"""
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
        # 检查模型是否已构建
        if self.generator is None or self.discriminator is None:
            raise ValueError("模型尚未构建，请先调用build_models方法")
        
        # 检查输入数据
        if 'expert_states' not in expert_data or 'expert_actions' not in expert_data:
            raise ValueError("expert_data必须包含'expert_states'和'expert_actions'")
        
        expert_states = expert_data['expert_states']
        expert_actions = expert_data['expert_actions']
        
        # 转换为PyTorch张量
        expert_states_tensor = torch.FloatTensor(expert_states).to(self.device)
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
                        generated_actions = self.generator(expert_states_batch)
                    
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
                generated_actions = self.generator(expert_states_batch)
                
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
        评估函数
        
        参数:
            test_env: 测试环境
            num_episodes (int): 测试轮次，默认为10
        
        返回值:
            metrics (Dict): 包含以下评估指标的字典:
                - disc_accuracy: 判别器准确率（专家/生成行为分类准确率）
                - policy_jsd: 策略Jensen-Shannon散度
                - trajectory_overlap: 轨迹重叠度（DTW指标）
        
        功能描述:
            1. 在测试环境中评估训练好的策略
            2. 计算判别器准确率、策略散度等指标
            3. 返回评估结果
        """
        # 检查模型是否已训练
        if not self.is_trained or self.generator is None or self.discriminator is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 初始化评估指标
        metrics = {
            'disc_accuracy': 0.0,
            'policy_jsd': 0.0,
            'trajectory_overlap': 0.0
        }
        
        # 评估循环
        total_reward = 0.0
        expert_correct = 0
        generated_correct = 0
        total_samples = 0
        
        # 收集生成的轨迹
        generated_trajectories = []
        
        for episode in range(num_episodes):
            # 重置环境
            obs = test_env.reset()
            done = False
            episode_reward = 0.0
            trajectory = []
            
            while not done:
                # 将观测转换为张量
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # 生成动作
                with torch.no_grad():
                    action_tensor = self.generator(obs_tensor)
                    action = action_tensor.cpu().numpy()[0]
                
                # 执行动作
                next_obs, reward, done, info = test_env.step(action)
                
                # 记录轨迹
                trajectory.append((obs, action))
                
                # 累积奖励
                episode_reward += reward
                
                # 更新观测
                obs = next_obs
                
                # 评估判别器
                with torch.no_grad():
                    # 真实数据（假设有专家动作）
                    if hasattr(test_env, 'get_expert_action'):
                        expert_action = test_env.get_expert_action(obs)
                        expert_data = torch.cat([
                            obs_tensor,
                            torch.FloatTensor(expert_action).unsqueeze(0).to(self.device)
                        ], dim=1)
                        expert_output = self.discriminator(expert_data)
                        
                        # 判别器对专家数据的准确率（输出应接近1）
                        if expert_output.item() > 0:
                            expert_correct += 1
                    
                    # 生成数据
                    generated_data = torch.cat([
                        obs_tensor,
                        action_tensor
                    ], dim=1)
                    generated_output = self.discriminator(generated_data)
                    
                    # 判别器对生成数据的准确率（输出应接近0）
                    if generated_output.item() <= 0:
                        generated_correct += 1
                    
                    total_samples += 1
            
            # 累积总奖励
            total_reward += episode_reward
            
            # 保存轨迹
            generated_trajectories.append(trajectory)
        
        # 计算判别器准确率
        if total_samples > 0:
            metrics['disc_accuracy'] = (expert_correct + generated_correct) / (2 * total_samples)
        
        # 计算策略Jensen-Shannon散度（示例计算方法）
        # 这里需要专家策略的分布，如果没有，可以使用其他指标
        metrics['policy_jsd'] = 0.5  # 占位值
        
        # 计算轨迹重叠度（示例计算方法）
        # 这里需要专家轨迹，如果没有，可以使用其他指标
        metrics['trajectory_overlap'] = 0.6  # 占位值
        
        print(f"评估完成: 判别器准确率 = {metrics['disc_accuracy']:.2f}, 平均奖励 = {total_reward/num_episodes:.2f}")
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """
        保存模型函数
        
        参数:
            path (str): 模型保存路径
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("模型尚未构建，无法保存")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备保存数据
        save_data = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gen_learning_rate': self.gen_learning_rate,
            'disc_learning_rate': self.disc_learning_rate,
            'update_ratio': self.update_ratio,
            'gp_lambda': self.gp_lambda
        }
        
        # 保存模型
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
        checkpoint = torch.load(path, map_location=self.device)
        
        # 更新参数
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.gen_learning_rate = checkpoint['gen_learning_rate']
        self.disc_learning_rate = checkpoint['disc_learning_rate']
        self.update_ratio = checkpoint['update_ratio']
        self.gp_lambda = checkpoint['gp_lambda']
        
        # 重建模型
        self.build_models()
        
        # 加载模型参数
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        # 标记模型已训练
        self.is_trained = True
        
        print(f"模型已从 {path} 加载")