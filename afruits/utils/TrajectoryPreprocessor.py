import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any

class TrajectoryPreprocessor:
    """
    轨迹预处理器：实现多源长序列预处理轨迹的标准化、分段切片与噪声清洗。
    
    核心特性：
    - 多源数据兼容：支持模拟器/传感器/数据库等多数据源输入
    - 智能分段：自适应动窗口与等长切片机制
    - 时序一致性：跨模态数据对齐与重采样
    - 噪声鲁棒性：噪点去除与波动值修复
    """
    
    def __init__(self, 
                 window_size: int = 50, 
                 overlap_rate: float = 0.3, 
                 resample_freq: int = 10, 
                 noise_threshold: float = 3.0,
                 norm_method: str = "minmax"):
        """
        初始化轨迹预处理器
        
        参数:
            window_size (int): 滑动窗口切片长度，取值范围50-500
            overlap_rate (float): 窗口重叠比例，取值范围0.1-0.6
            resample_freq (int): 重采样频率，取值范围1-50
            noise_threshold (float): 异常值判定标准，取值范围2.0-5.0
            norm_method (str): 数据标准化方法，可选["minmax", "zscore"]
        """
        # 参数有效性检查
        assert 50 <= window_size <= 500, "window_size必须在50-500范围内"
        assert 0.1 <= overlap_rate <= 0.6, "overlap_rate必须在0.1-0.6范围内"
        assert 1 <= resample_freq <= 50, "resample_freq必须在1-50范围内"
        assert 2.0 <= noise_threshold <= 5.0, "noise_threshold必须在2.0-5.0范围内"
        assert norm_method in ["minmax", "zscore"], "norm_method必须是'minmax'或'zscore'"
        
        # 初始化参数
        self.window_size = window_size
        self.overlap_rate = overlap_rate
        self.resample_freq = resample_freq
        self.noise_threshold = noise_threshold
        self.norm_method = norm_method
        
        # 元数据记录：记录数据源类型与来集参数
        self.metadata = {}
        
    def load_data(self, data_sources: List, source_type: str = "simulator") -> Dict:
        """
        数据加载：加载原始轨迹数据
        
        参数:
            data_sources (List): 数据路径列表（支持CSV/HDF5/JSON）
            source_type (str): 数据类型 ["simulator", "sensor", "database"]
            
        返回:
            raw_dataset (Dict): 原始数据集字典
                - states: 原始状态序列 (N×T×D_s)
                - actions: 原始动作序列 (N×T×D_a)
                - timestamps: 时间戳向量 (N×T)
        """
        # 验证参数
        assert source_type in ["simulator", "sensor", "database"], "source_type必须是'simulator'、'sensor'或'database'"
        
        # 初始化数据容器
        states_list = []
        actions_list = []
        timestamps_list = []
        
        # 根据数据源类型处理数据加载
        for source in data_sources:
            # 这里简化处理，实际应用中需要根据不同数据源类型实现具体的加载逻辑
            if source_type == "simulator":
                # 模拟器数据加载逻辑
                data = self._load_simulator_data(source)
            elif source_type == "sensor":
                # 传感器数据加载逻辑
                data = self._load_sensor_data(source)
            elif source_type == "database":
                # 数据库数据加载逻辑
                data = self._load_database_data(source)
            
            # 提取数据
            states_list.append(data["states"])
            actions_list.append(data["actions"])
            timestamps_list.append(data["timestamps"])
        
        # 合并数据
        raw_dataset = {
            "states": np.vstack(states_list) if states_list else np.array([]),
            "actions": np.vstack(actions_list) if actions_list else np.array([]),
            "timestamps": np.vstack(timestamps_list) if timestamps_list else np.array([])
        }
        
        # 记录元数据
        self.metadata["source_type"] = source_type
        self.metadata["data_shape"] = {
            "states": raw_dataset["states"].shape,
            "actions": raw_dataset["actions"].shape,
            "timestamps": raw_dataset["timestamps"].shape
        }
        
        return raw_dataset
    
    def format_unification(self, raw_dataset: Dict) -> Dict:
        """
        格式标准化：统一不同来源数据的格式
        
        参数:
            raw_dataset (Dict): 原始数据集
            
        返回:
            standardized_data (Dict): 标准化数据
                - aligned_states: 对齐后的状态矩阵
                - aligned_actions: 对齐后的动作矩阵
                - meta_info: 标准化元数据标签
        """
        # 数据标准化处理
        if self.norm_method == "minmax":
            # Min-Max标准化
            states_normalized = self._minmax_normalize(raw_dataset["states"])
            actions_normalized = self._minmax_normalize(raw_dataset["actions"])
        elif self.norm_method == "zscore":
            # Z-Score标准化
            states_normalized = self._zscore_normalize(raw_dataset["states"])
            actions_normalized = self._zscore_normalize(raw_dataset["actions"])
        
        # 构建标准化数据集
        standardized_data = {
            "aligned_states": states_normalized,
            "aligned_actions": actions_normalized,
            "meta_info": {
                "norm_method": self.norm_method,
                "state_dims": states_normalized.shape[-1],
                "action_dims": actions_normalized.shape[-1],
                "sequence_length": states_normalized.shape[1]
            }
        }
        
        return standardized_data
    
    def segment_trajs(self, standardized_data: Dict, min_length: int = 10) -> Dict:
        """
        轨迹分段：按窗口大小切分轨迹
        
        参数:
            standardized_data (Dict): 标准化数据
            min_length (int): 最小有效片段长度
            
        返回:
            sliced_dataset (Dict): 切片数据集
                - state_segments: 切片状态序列
                - action_segments: 切片动作序列
                - valid_flags: 片段有效标记
        """
        # 提取数据
        states = standardized_data["aligned_states"]
        actions = standardized_data["aligned_actions"]
        
        # 初始化结果容器
        state_segments = []
        action_segments = []
        valid_flags = []
        
        # 计算步长（基于重叠率）
        stride = int(self.window_size * (1 - self.overlap_rate))
        
        # 对每个轨迹进行分段
        for i in range(len(states)):
            seq_len = states[i].shape[0]
            
            # 滑动窗口切分
            for start_idx in range(0, seq_len - self.window_size + 1, stride):
                end_idx = start_idx + self.window_size
                
                # 提取片段
                state_segment = states[i][start_idx:end_idx]
                action_segment = actions[i][start_idx:end_idx]
                
                # 判断片段是否有效（长度检查）
                is_valid = (end_idx - start_idx) >= min_length
                
                # 存储结果
                state_segments.append(state_segment)
                action_segments.append(action_segment)
                valid_flags.append(is_valid)
        
        # 转换为numpy数组
        sliced_dataset = {
            "state_segments": np.array(state_segments),
            "action_segments": np.array(action_segments),
            "valid_flags": np.array(valid_flags)
        }
        
        return sliced_dataset
    
    def time_align(self, sliced_dataset: Dict, reference_clock: str = "host") -> Dict:
        """
        时序对齐：对齐不同来源的时序数据
        
        参数:
            sliced_dataset (Dict): 切片数据集
            reference_clock (str): 参考时钟源 ["host", "global"]
            
        返回:
            aligned_dataset (Dict): 对齐后数据集
                - resampled_states: 重采样后的状态序列
                - resampled_actions: 重采样后的动作序列
                - time_deltas: 时间偏差校正值
        """
        # 验证参数
        assert reference_clock in ["host", "global"], "reference_clock必须是'host'或'global'"
        
        # 提取数据
        state_segments = sliced_dataset["state_segments"]
        action_segments = sliced_dataset["action_segments"]
        
        # 初始化结果容器
        resampled_states = []
        resampled_actions = []
        time_deltas = []
        
        # 对每个片段进行时序对齐
        for i in range(len(state_segments)):
            # 计算重采样索引（基于重采样频率）
            original_indices = np.arange(len(state_segments[i]))
            
            if reference_clock == "host":
                # 主机时钟对齐（均匀采样）
                new_indices = np.linspace(0, len(state_segments[i]) - 1, self.resample_freq)
            else:
                # 全局时钟对齐（可能有非均匀采样）
                # 这里简化为随机扰动的均匀采样
                new_indices = np.linspace(0, len(state_segments[i]) - 1, self.resample_freq)
                new_indices += np.random.normal(0, 0.1, size=len(new_indices))
                new_indices = np.clip(new_indices, 0, len(state_segments[i]) - 1)
            
            # 线性插值重采样
            resampled_state = self._interpolate_sequence(state_segments[i], original_indices, new_indices)
            resampled_action = self._interpolate_sequence(action_segments[i], original_indices, new_indices)
            
            # 计算时间偏差
            time_delta = np.mean(np.diff(new_indices)) - np.mean(np.diff(original_indices))
            
            # 存储结果
            resampled_states.append(resampled_state)
            resampled_actions.append(resampled_action)
            time_deltas.append(time_delta)
        
        # 转换为numpy数组
        aligned_dataset = {
            "resampled_states": np.array(resampled_states),
            "resampled_actions": np.array(resampled_actions),
            "time_deltas": np.array(time_deltas)
        }
        
        return aligned_dataset
    
    def denoise(self, aligned_dataset: Dict, filter_type: str = "kalman") -> Dict:
        """
        噪声处理：去除异常值与波动修复
        
        参数:
            aligned_dataset (Dict): 对齐后数据集
            filter_type (str): 滤波方法 ["kalman", "median"]
            
        返回:
            cleaned_dataset (Dict): 清洗后数据集
                - filtered_states: 滤波后状态数据
                - repaired_actions: 修复后动作序列
                - noise_errors: 噪点统计报告
        """
        # 验证参数
        assert filter_type in ["kalman", "median"], "filter_type必须是'kalman'或'median'"
        
        # 提取数据
        states = aligned_dataset["resampled_states"]
        actions = aligned_dataset["resampled_actions"]
        
        # 初始化结果容器
        filtered_states = []
        repaired_actions = []
        noise_counts = []
        
        # 对每个片段进行噪声处理
        for i in range(len(states)):
            # 异常值检测
            state_mask = self._detect_outliers(states[i], self.noise_threshold)
            action_mask = self._detect_outliers(actions[i], self.noise_threshold)
            
            # 根据滤波方法处理
            if filter_type == "kalman":
                # 卡尔曼滤波（简化为移动平均）
                filtered_state = self._apply_moving_average(states[i], state_mask)
                repaired_action = self._apply_moving_average(actions[i], action_mask)
            else:
                # 中值滤波
                filtered_state = self._apply_median_filter(states[i], state_mask)
                repaired_action = self._apply_median_filter(actions[i], action_mask)
            
            # 统计噪点数量
            noise_count = np.sum(state_mask) + np.sum(action_mask)
            
            # 存储结果
            filtered_states.append(filtered_state)
            repaired_actions.append(repaired_action)
            noise_counts.append(noise_count)
        
        # 转换为numpy数组
        cleaned_dataset = {
            "filtered_states": np.array(filtered_states),
            "repaired_actions": np.array(repaired_actions),
            "noise_errors": {
                "total_noise_points": np.sum(noise_counts),
                "noise_ratio": np.sum(noise_counts) / (len(states) * states[0].shape[0]),
                "per_trajectory_noise": noise_counts
            }
        }
        
        return cleaned_dataset
    
    def data_enhance(self, trajectories: List, data_num: int = 0, method: str = None) -> List:
        """
        数据增强：扩充原始数据集
        
        参数:
            trajectories (List): 原始数据列表
            data_num (int): 增强的数据数量
            method (str): 数据增强方法
            
        返回:
            增强后轨迹集 (List)
        """
        if method is None or data_num <= 0:
            return trajectories
        
        enhanced_trajectories = trajectories.copy()
        
        # 简单数据增强
        if "simple" in method.lower():
            # 随机变换：随机选择平移/旋转/缩放组合
            for _ in range(data_num):
                traj = np.random.choice(trajectories)
                # 随机平移
                if np.random.random() < 0.5:
                    traj = self._random_shift(traj)
                # 随机旋转
                if np.random.random() < 0.5:
                    traj = self._random_rotate(traj)
                # 随机缩放
                if np.random.random() < 0.5:
                    traj = self._random_scale(traj)
                
                enhanced_trajectories.append(traj)
        
        # 隐空间生成
        elif "latent" in method.lower():
            # 检查是否提供了模型保存路径
            if save_path is None:
                raise ValueError("使用隐空间生成方法时必须提供save_path参数")
            
            # 导入VAETrajGenerator
            from afruits.utils.VAETrajGenerator import VAETrajGenerator
            
            # 创建VAE轨迹生成器实例
            vae_generator = VAETrajGenerator()
            
            # 准备数据格式
            # 假设轨迹数据是numpy数组列表，需要转换为适合VAE的格式
            trajectories_array = np.array(trajectories)
            
            # 从save_path加载预训练的VAE模型
            import torch
            import os
            
            # 检查模型文件是否存在
            save_path = "models"
            model_path = os.path.join(save_path, "vae_model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"在{model_path}找不到VAE模型文件")
            
            # 加载模型参数
            vae_generator.load_model(model_path)
            
            # 设置为评估模式
            vae_generator.encoder.eval()
            vae_generator.decoder.eval()
            
            # 使用VAE生成新轨迹
            generated_data = vae_generator.generate(num_samples=data_num)
            
            # 将生成的轨迹添加到增强数据集
            generated_trajectories = generated_data['trajectories']
            for traj in generated_trajectories:
                enhanced_trajectories.append(traj)
        
        return enhanced_trajectories
    
    # 辅助方法
    def _load_simulator_data(self, source):
        """加载模拟器数据"""
        states = source.get("states", np.random.random((10, 100, 5)))
        actions = source.get("actions", np.random.random((10, 100, 2)))
        timestamps = source.get("timestamps", np.arange(100).reshape(10, 10))
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "timestamps": np.array(timestamps)
        }
    
    def _load_sensor_data(self, source):
        """加载传感器数据"""
        states = source.get("states", np.random.random((10, 100, 5)))
        actions = source.get("actions", np.random.random((10, 100, 2)))
        timestamps = source.get("timestamps", np.arange(100).reshape(10, 10))
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "timestamps": np.array(timestamps)
        }
    
    
    def _load_database_data(self, source):
        """加载数据库数据"""
        states = source.get("states", np.random.random((10, 100, 5)))
        actions = source.get("actions", np.random.random((10, 100, 2)))
        timestamps = source.get("timestamps", np.arange(100).reshape(10, 10))
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "timestamps": np.array(timestamps)
        }
    
    
    def _minmax_normalize(self, data):
        """Min-Max标准化"""
        min_vals = np.min(data, axis=(0, 1), keepdims=True)
        max_vals = np.max(data, axis=(0, 1), keepdims=True)
        return (data - min_vals) / (max_vals - min_vals + 1e-8)
    
    def _zscore_normalize(self, data):
        """Z-Score标准化"""
        mean = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True)
        return (data - mean) / (std + 1e-8)
    
    def _interpolate_sequence(self, sequence, original_indices, new_indices):
        """线性插值序列"""
        # 对每个特征维度进行插值
        result = np.zeros((len(new_indices), sequence.shape[1]))
        for dim in range(sequence.shape[1]):
            result[:, dim] = np.interp(new_indices, original_indices, sequence[:, dim])
        return result
    
    def _detect_outliers(self, data, threshold):
        """检测异常值"""
        # 使用Z-Score方法检测异常值
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-8))
        return z_scores > threshold
    
    def _apply_moving_average(self, data, mask, window=3):
        """应用移动平均滤波"""
        result = data.copy()
        for i in range(len(data)):
            if np.any(mask[i]):
                # 对异常值应用移动平均
                start = max(0, i - window // 2)
                end = min(len(data), i + window // 2 + 1)
                result[i] = np.mean(data[start:end], axis=0)
        return result
    
    def _apply_median_filter(self, data, mask, window=3):
        """应用中值滤波"""
        result = data.copy()
        for i in range(len(data)):
            if np.any(mask[i]):
                # 对异常值应用中值滤波
                start = max(0, i - window // 2)
                end = min(len(data), i + window // 2 + 1)
                result[i] = np.median(data[start:end], axis=0)
        return result
    
    def _random_shift(self, trajectory, max_shift=0.1):
        """随机平移变换"""
        shift = np.random.uniform(-max_shift, max_shift, size=trajectory.shape[-1])
        return trajectory + shift
    
    def _random_rotate(self, trajectory, max_angle=0.1):
        """随机旋转变换（简化为2D旋转）"""
        # 注：实际应用中需要根据数据维度实现适当的旋转
        angle = np.random.uniform(-max_angle, max_angle)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        # 假设前两维是位置坐标
        result = trajectory.copy()
        result[:, :2] = np.dot(trajectory[:, :2], rot_matrix.T)
        return result
    
    def _random_scale(self, trajectory, scale_range=(0.9, 1.1)):
        """随机缩放变换"""
        scale = np.random.uniform(scale_range[0], scale_range[1], size=trajectory.shape[-1])
        return trajectory * scale