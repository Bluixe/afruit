import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TSNE
from sklearn.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class OfflineEvaluator:
    """
    离线评估器类
    
    功能描述：实现对离线强化学习和行为克隆算法的评估
    
    核心功能：
    - 数据处理：支持离线数据的处理与转换
    - 策略评估：提供多种评估指标
    - 可视化工具：支持多种可视化方法
    """
    
    def __init__(self,
                 is_gamma: float = 0.99,
                 cis_threshold: float = 0.3,
                 variance_reduction: bool = True,
                 quality_check_level: int = 2):
        """
        初始化离线评估器
        
        参数:
            is_gamma (float): 重要性采样折扣因子，默认为0.99，有效取值范围0.9-0.999
            cis_threshold (float): 条件重要性采样阈值，默认为0.3，有效取值范围0.1-0.5
            variance_reduction (bool): 是否使用方差减小技术，默认为True
            quality_check_level (int): 数据质量检查等级，默认为2，有效取值范围1-3
        """
        # 初始化参数
        self.is_gamma = is_gamma
        self.cis_threshold = cis_threshold
        self.variance_reduction = variance_reduction
        self.quality_check_level = quality_check_level
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证is_gamma
        if not isinstance(self.is_gamma, float) or not (0.9 <= self.is_gamma <= 0.999):
            raise ValueError(f"is_gamma必须在0.9-0.999范围内，当前值: {self.is_gamma}")
        
        # 验证cis_threshold
        if not isinstance(self.cis_threshold, float) or not (0.1 <= self.cis_threshold <= 0.5):
            raise ValueError(f"cis_threshold必须在0.1-0.5范围内，当前值: {self.cis_threshold}")
        
        # 验证variance_reduction
        if not isinstance(self.variance_reduction, bool):
            raise ValueError(f"variance_reduction必须为布尔值，当前值: {self.variance_reduction}")
        
        # 验证quality_check_level
        if not isinstance(self.quality_check_level, int) or not (1 <= self.quality_check_level <= 3):
            raise ValueError(f"quality_check_level必须为1-3范围内的整数，当前值: {self.quality_check_level}")
    
    def preprocess_data(self, raw_trajectories: Dict) -> Dict:
        """
        数据预处理函数
        
        参数:
            raw_trajectories (Dict): 原始轨迹数据集（包含状态-动作-奖励序列）
        
        返回值:
            处理后的数据集（字典）
        
        功能描述:
            - valid_states: 有效状态采样量
            - action_probs: 行为策略动作概率
            - flags: 轨迹质量标记信息
        
        处理流程:
            1. 数据完整性检查：验证元组数据完整性
            2. 噪声过滤：基于阈值过滤噪声数据
            3. 轨迹标记：记录"_D(s,a)"值
        """
        # 检查输入数据
        if not raw_trajectories or not isinstance(raw_trajectories, dict):
            raise ValueError("raw_trajectories必须是非空字典")
        
        # 初始化结果字典
        processed_data = {
            'valid_states': [],
            'action_probs': [],
            'flags': []
        }
        
        # 处理轨迹数据
        for traj_id, trajectory in raw_trajectories.items():
            # 检查轨迹数据是否包含状态、动作和奖励
            if 'states' not in trajectory or 'actions' not in trajectory or 'rewards' not in trajectory:
                print(f"警告: 轨迹 {traj_id} 缺少状态、动作或奖励数据，已跳过")
                continue
            
            states = trajectory['states']
            actions = trajectory['actions']
            rewards = trajectory['rewards']
            
            # 检查数据长度是否匹配
            if not (len(states) == len(actions) == len(rewards)):
                print(f"警告: 轨迹 {traj_id} 的状态、动作和奖励数据长度不匹配，已跳过")
                continue
            
            # 数据质量检查
            valid_indices = self._check_data_quality(states, actions, rewards)
            
            # 提取有效数据
            for idx in valid_indices:
                processed_data['valid_states'].append(states[idx])
                
                # 计算动作概率（简化示例，实际应根据策略计算）
                action_prob = np.ones(len(actions[idx])) / len(actions[idx])  # 均匀分布
                processed_data['action_probs'].append(action_prob)
                
                # 设置标记
                flag = 1 if self._is_valid_transition(states[idx], actions[idx], rewards[idx]) else 0
                processed_data['flags'].append(flag)
        
        # 转换为numpy数组
        for key in processed_data:
            if processed_data[key]:
                processed_data[key] = np.array(processed_data[key])
            else:
                processed_data[key] = np.array([])
        
        print(f"数据预处理完成: 有效状态数量 = {len(processed_data['valid_states'])}")
        return processed_data
    
    def _check_data_quality(self, states, actions, rewards) -> List[int]:
        """检查数据质量并返回有效数据的索引"""
        valid_indices = []
        
        for i in range(len(states)):
            # 基本检查：非空数据
            if len(states[i]) == 0 or len(actions[i]) == 0:
                continue
            
            # 根据质量检查等级进行筛选
            if self.quality_check_level >= 1:
                # 级别1：检查数值范围
                if np.any(np.isnan(states[i])) or np.any(np.isnan(actions[i])):
                    continue
            
            if self.quality_check_level >= 2:
                # 级别2：检查动作合法性
                if not self._is_valid_action(actions[i]):
                    continue
            
            if self.quality_check_level >= 3:
                # 级别3：检查状态转移合理性
                if i > 0 and not self._is_reasonable_transition(states[i-1], states[i], actions[i-1]):
                    continue
            
            valid_indices.append(i)
        
        return valid_indices
    
    def evaluate_policy(self, target_policy, behavior_policy, method_type: str = 'IS') -> Dict:
        """
        策略评估函数
        
        参数:
            target_policy (object): 待评估策略
            behavior_policy (object): 行为策略
            method_type (str): 评估方法，'IS', 'DR', 'CIS'之一
        
        返回值:
            dict: 评估结果
        
        算法实现:
            重要性采样（IS）：
            V = (Σπ_t * R_t) / (Σπ_b)
            
            π_t = π(π_target/π_behavior)
            
            双重鲁棒（DR）：
            
            条件重要性采样（CIS）：
            V = 1/n Σ(π + w(Φ))
        """
        # 检查评估方法类型
        valid_methods = ['IS', 'DR', 'CIS']
        if method_type not in valid_methods:
            raise ValueError(f"method_type必须为 {valid_methods} 之一，当前值: {method_type}")
        
        # 初始化结果字典
        result = {
            'value': 0.0,
            'confidence_interval': [0.0, 0.0],
            'method': method_type
        }
        
        # 根据评估方法类型选择不同的评估策略
        if method_type == 'IS':
            # 重要性采样方法
            result = self._importance_sampling(target_policy, behavior_policy)
        elif method_type == 'DR':
            # 双重鲁棒方法
            result = self._doubly_robust(target_policy, behavior_policy)
        elif method_type == 'CIS':
            # 条件重要性采样方法
            result = self._conditional_importance_sampling(target_policy, behavior_policy)
        
        return result
    
    def _importance_sampling(self, target_policy, behavior_policy) -> Dict:
        """重要性采样方法实现"""
        # 初始化结果
        result = {
            'value': 0.0,
            'confidence_interval': [0.0, 0.0],
            'method': 'IS'
        }
        
        try:
            # 获取目标策略和行为策略的动作概率
            target_probs = self._get_policy_probs(target_policy)
            behavior_probs = self._get_policy_probs(behavior_policy)
            
            # 计算重要性权重
            weights = []
            for t_prob, b_prob in zip(target_probs, behavior_probs):
                # 避免除以零
                if np.any(b_prob < 1e-10):
                    b_prob = np.maximum(b_prob, 1e-10)
                
                # 计算权重比率
                weight = t_prob / b_prob
                weights.append(weight)
            
            # 转换为numpy数组
            weights = np.array(weights)
            
            # 获取奖励
            rewards = self._get_rewards(behavior_policy)
            
            # 计算加权奖励
            weighted_rewards = weights * rewards
            
            # 计算价值估计
            value = np.sum(weighted_rewards) / np.sum(weights)
            
            # 计算置信区间
            std_error = np.std(weighted_rewards) / np.sqrt(len(weighted_rewards))
            confidence_interval = [value - 1.96 * std_error, value + 1.96 * std_error]
            
            # 更新结果
            result['value'] = float(value)
            result['confidence_interval'] = [float(ci) for ci in confidence_interval]
            
        except Exception as e:
            print(f"重要性采样评估出错: {str(e)}")
        
        return result
    
    def _doubly_robust(self, target_policy, behavior_policy) -> Dict:
        """双重鲁棒方法实现"""
        # 初始化结果
        result = {
            'value': 0.0,
            'confidence_interval': [0.0, 0.0],
            'method': 'DR'
        }
        
        try:
            # 获取目标策略和行为策略的动作概率
            target_probs = self._get_policy_probs(target_policy)
            behavior_probs = self._get_policy_probs(behavior_policy)
            
            # 计算重要性权重
            weights = []
            for t_prob, b_prob in zip(target_probs, behavior_probs):
                # 避免除以零
                if np.any(b_prob < 1e-10):
                    b_prob = np.maximum(b_prob, 1e-10)
                
                # 计算权重比率
                weight = t_prob / b_prob
                weights.append(weight)
            
            # 转换为numpy数组
            weights = np.array(weights)
            
            # 获取奖励和状态值估计
            rewards = self._get_rewards(behavior_policy)
            state_values = self._estimate_state_values(target_policy)
            
            # 计算双重鲁棒估计
            dr_estimates = []
            for i in range(len(rewards)):
                # 直接奖励项
                direct_reward = rewards[i]
                
                # 修正项
                correction = weights[i] * (direct_reward - state_values[i])
                
                # 双重鲁棒估计
                dr_estimate = state_values[i] + correction
                dr_estimates.append(dr_estimate)
            
            # 转换为numpy数组
            dr_estimates = np.array(dr_estimates)
            
            # 计算价值估计
            value = np.mean(dr_estimates)
            
            # 计算置信区间
            std_error = np.std(dr_estimates) / np.sqrt(len(dr_estimates))
            confidence_interval = [value - 1.96 * std_error, value + 1.96 * std_error]
            
            # 更新结果
            result['value'] = float(value)
            result['confidence_interval'] = [float(ci) for ci in confidence_interval]
            
        except Exception as e:
            print(f"双重鲁棒评估出错: {str(e)}")
        
        return result
    
    def _conditional_importance_sampling(self, target_policy, behavior_policy) -> Dict:
        """条件重要性采样方法实现"""
        # 初始化结果
        result = {
            'value': 0.0,
            'confidence_interval': [0.0, 0.0],
            'method': 'CIS'
        }
        
        try:
            # 获取目标策略和行为策略的动作概率
            target_probs = self._get_policy_probs(target_policy)
            behavior_probs = self._get_policy_probs(behavior_policy)
            
            # 计算条件重要性权重
            weights = []
            for t_prob, b_prob in zip(target_probs, behavior_probs):
                # 避免除以零
                if np.any(b_prob < 1e-10):
                    b_prob = np.maximum(b_prob, 1e-10)
                
                # 计算权重比率
                weight = t_prob / b_prob
                
                # 应用阈值截断
                weight = np.minimum(weight, self.cis_threshold)
                
                weights.append(weight)
            
            # 转换为numpy数组
            weights = np.array(weights)
            
            # 获取奖励
            rewards = self._get_rewards(behavior_policy)
            
            # 计算加权奖励
            weighted_rewards = weights * rewards
            
            # 计算价值估计
            value = np.mean(weighted_rewards)
            
            # 计算置信区间
            std_error = np.std(weighted_rewards) / np.sqrt(len(weighted_rewards))
            confidence_interval = [value - 1.96 * std_error, value + 1.96 * std_error]
            
            # 更新结果
            result['value'] = float(value)
            result['confidence_interval'] = [float(ci) for ci in confidence_interval]
            
        except Exception as e:
            print(f"条件重要性采样评估出错: {str(e)}")
        
        return result
    
    def _get_policy_probs(self, policy) -> np.ndarray:
        """获取策略的动作概率"""
        # 简化示例：假设策略对象有一个get_action_probs方法
        try:
            if hasattr(policy, 'get_action_probs'):
                return policy.get_action_probs()
            elif hasattr(policy, 'action_probs'):
                return policy.action_probs
            else:
                # 默认返回均匀分布
                return np.ones((10, 5)) / 5  # 假设10个状态，每个状态5个动作
        except Exception as e:
            print(f"获取策略概率出错: {str(e)}")
            return np.ones((10, 5)) / 5  # 返回默认值
    
    def dim_reduction(self, eval_result: Dict, dim_reduction_type: str = 'tsne') -> Dict:
        """
        降维可视化方法
        
        参数:
            eval_result (Dict): 评估结果数据
            dim_reduction_type (str): 降维方法，'tsne'或'pca'
        
        返回值:
            dict: 可视化数据（包含降维后的片段）
        """
        # 检查降维方法类型
        valid_methods = ['tsne', 'pca']
        if dim_reduction_type not in valid_methods:
            raise ValueError(f"dim_reduction_type必须为 {valid_methods} 之一，当前值: {dim_reduction_type}")
        
        # 检查输入数据
        if not isinstance(eval_result, dict) or 'value' not in eval_result:
            raise ValueError("eval_result必须是包含'value'键的字典")
        
        # 初始化结果字典
        result = {
            'reduced_data': None,
            'method': dim_reduction_type,
            'original_shape': None
        }
        
        try:
            # 提取数据
            data = self._extract_data_for_visualization(eval_result)
            
            # 记录原始形状
            result['original_shape'] = data.shape
            
            # 数据预处理：展平高维数据
            if len(data.shape) > 2:
                # 如果数据是高维的，将其展平为2D
                n_samples = data.shape[0]
                data_reshaped = data.reshape(n_samples, -1)
            else:
                data_reshaped = data
            
            # 应用降维方法
            if dim_reduction_type == 'tsne':
                # t-SNE降维
                reduced_data = TSNE(n_components=2, random_state=42).fit_transform(data_reshaped)
            elif dim_reduction_type == 'pca':
                # PCA降维
                reduced_data = PCA(n_components=2, random_state=42).fit_transform(data_reshaped)
            
            # 存储降维结果
            result['reduced_data'] = reduced_data
            
            print(f"降维完成: 原始形状 {result['original_shape']} -> 降维后形状 {reduced_data.shape}")
            
        except Exception as e:
            print(f"降维处理出错: {str(e)}")
        
        return result
    
    def vis_heatmap(self, eval_result: Dict) -> Dict:
        """
        热力图可视化方法
        
        参数:
            eval_result (Dict): 评估结果数据
        
        返回值:
            dict: 可视化数据（包含可视化图片）
        """
        # 检查输入数据
        if not isinstance(eval_result, dict):
            raise ValueError("eval_result必须是字典")
        
        # 初始化结果字典
        result = {
            'heatmap_data': None,
            'figure': None
        }
        
        try:
            # 提取数据
            data = self._extract_data_for_visualization(eval_result)
            
            # 确保数据是2D的
            if len(data.shape) > 2:
                # 如果数据是高维的，将其展平为2D
                n_samples = data.shape[0]
                data_reshaped = data.reshape(n_samples, -1)
            elif len(data.shape) == 1:
                # 如果数据是1D的，将其转换为2D
                data_reshaped = data.reshape(-1, 1)
            else:
                data_reshaped = data
            
            # 计算相关性矩阵（如果数据列数大于1）
            if data_reshaped.shape[1] > 1:
                heatmap_data = np.corrcoef(data_reshaped.T)
            else:
                # 如果只有一列，创建一个简单的热力图数据
                heatmap_data = np.ones((1, 1))
            
            # 存储热力图数据
            result['heatmap_data'] = heatmap_data
            
            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('特征相关性热力图')
            
            # 保存图像到内存
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # 存储图像数据
            result['figure'] = buf
            
            print(f"热力图生成完成: 形状 {heatmap_data.shape}")
            
            # 关闭图像以释放内存
            plt.close()
            
        except Exception as e:
            print(f"热力图生成出错: {str(e)}")
        
        return result
    
    def vis_3d(self, eval_result: Dict) -> Dict:
        """
        3D可视化方法
        
        参数:
            eval_result (Dict): 评估结果数据
        
        返回值:
            dict: 可视化数据（包含可视化图片）
        """
        # 检查输入数据
        if not isinstance(eval_result, dict):
            raise ValueError("eval_result必须是字典")
        
        # 初始化结果字典
        result = {
            '3d_data': None,
            'figure': None
        }
        
        try:
            # 提取数据
            data = self._extract_data_for_visualization(eval_result)
            
            # 确保数据是2D的
            if len(data.shape) > 2:
                # 如果数据是高维的，将其展平为2D
                n_samples = data.shape[0]
                data_reshaped = data.reshape(n_samples, -1)
            else:
                data_reshaped = data
            
            # 如果数据维度不足3维，使用PCA降维到3维
            if data_reshaped.shape[1] < 3:
                # 使用PCA将数据扩展到3维
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3, random_state=42)
                data_3d = pca.fit_transform(data_reshaped)
            elif data_reshaped.shape[1] > 3:
                # 使用PCA将数据降维到3维
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3, random_state=42)
                data_3d = pca.fit_transform(data_reshaped)
            else:
                # 数据已经是3维的
                data_3d = data_reshaped
            
            # 存储3D数据
            result['3d_data'] = data_3d
            
            # 创建3D图
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制散点图
            scatter = ax.scatter(
                data_3d[:, 0],
                data_3d[:, 1],
                data_3d[:, 2],
                c=np.arange(len(data_3d)),  # 使用索引作为颜色
                cmap='viridis',
                s=50,
                alpha=0.6
            )
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label='样本索引')
            
            # 设置标题和标签
            ax.set_title('3D数据可视化')
            ax.set_xlabel('维度1')
            ax.set_ylabel('维度2')
            ax.set_zlabel('维度3')
            
            # 保存图像到内存
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # 存储图像数据
            result['figure'] = buf
            
            print(f"3D可视化生成完成: 数据形状 {data_3d.shape}")
            
            # 关闭图像以释放内存
            plt.close()
            
        except Exception as e:
            print(f"3D可视化生成出错: {str(e)}")
        
        return result
    
    def _extract_data_for_visualization(self, eval_result: Dict) -> np.ndarray:
        """从评估结果中提取数据用于可视化"""
        # 简化示例：从评估结果中提取数据
        if isinstance(eval_result['value'], (int, float)):
            # 如果value是标量，创建一个随机数组用于演示
            return np.random.normal(0, 1, (100, 10))
        elif isinstance(eval_result['value'], np.ndarray):
            return eval_result['value']
        elif isinstance(eval_result['value'], list):
            return np.array(eval_result['value'])
        else:
            # 默认返回一个随机数组
            return np.random.normal(0, 1, (100, 10))
    
    def _get_rewards(self, policy) -> np.ndarray:
        """获取策略的奖励"""
        # 简化示例：假设策略对象有一个rewards属性
        try:
            if hasattr(policy, 'get_rewards'):
                return policy.get_rewards()
            elif hasattr(policy, 'rewards'):
                return policy.rewards
            else:
                # 默认返回随机奖励
                return np.random.normal(0, 1, 10)  # 假设10个状态
        except Exception as e:
            print(f"获取奖励出错: {str(e)}")
            return np.random.normal(0, 1, 10)  # 返回默认值
    
    def _estimate_state_values(self, policy) -> np.ndarray:
        """估计状态值函数"""
        # 简化示例：假设策略对象有一个estimate_values方法
        try:
            if hasattr(policy, 'estimate_values'):
                return policy.estimate_values()
            elif hasattr(policy, 'state_values'):
                return policy.state_values
            else:
                # 默认返回随机状态值
                return np.random.normal(0, 1, 10)  # 假设10个状态
        except Exception as e:
            print(f"估计状态值出错: {str(e)}")
            return np.random.normal(0, 1, 10)  # 返回默认值
    
    def _is_valid_action(self, action) -> bool:
        """检查动作是否有效"""
        # 简化示例：检查动作是否在合理范围内
        return np.all(np.abs(action) <= 1.0)
    
    def _is_reasonable_transition(self, state_prev, state_curr, action_prev) -> bool:
        """检查状态转移是否合理"""
        # 简化示例：检查状态变化是否在合理范围内
        state_diff = np.abs(state_curr - state_prev)
        return np.all(state_diff <= 0.5)  # 假设状态变化不应过大
    
    def _is_valid_transition(self, state, action, reward) -> bool:
        """检查状态-动作-奖励三元组是否有效"""
        # 简化示例：检查奖励是否在合理范围内
        return -100 <= reward <= 100