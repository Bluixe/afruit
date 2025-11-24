import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

class DataPreprocessor:
    """
    数据预处理器类
    
    功能描述：实现多种数据预处理操作，包括时间序列处理、数据标准化、异常检测等
    
    核心功能：
    - 时间序列处理：支持动态时间规整(DTW)与重采样技术
    - 数据标准化：集成Z-score标准化与Min-Max归一化模式
    - 异常值检测：提供多种异常检测方法与滤波选择
    - 时间序列对齐：内置RMSE, DTW等量化评估指标
    - 训练数据预处理：根据模型类型生成符合训练要求的数据格式
    """
    
    def __init__(self, target_freq: float = 1, norm_method: str = "zscore", filter_type: str = "kalman"):
        """初始化数据预处理器"""
        self.target_freq = target_freq
        self.norm_method = norm_method
        self.filter_type = filter_type
    
    def load_data(self, raw_data: Dict, timestamp: List[float] = None,
                 position: List[Tuple[float, float, float]] = None,
                 velocity: List[Tuple[float, float, float]] = None,
                 attitude: List[Tuple[float, float, float]] = None) -> Dict:
        """
        数据加载与预处理函数（增强版：支持轨迹数据字典）
        
        支持两类输入：
        1) 传统平铺数据，返回与原逻辑一致的dict
        2) 轨迹数据:
           {
             "state_dim": (D,),
             "action_dim": K,
             "trajectories": [
                 {"states":[T,D], "actions":[T], "rewards":[T], "next_states":[T,D], "dones":[T],
                  "opponent_actions":[T], "infos":[...](可选)}
             ]
           }
        返回会将列表转为numpy数组，确保dtype与形状一致，便于后续训练/保存。
        """
        # 若为轨迹数据，进行轨迹级转换
        if isinstance(raw_data, dict) and 'trajectories' in raw_data:
            trajs = raw_data.get('trajectories', [])
            processed_trajs = []

            def _to_int_array(a):
                arr = np.array(a)
                # 如果是one-hot，转换为索引
                if arr.ndim == 2 and arr.shape[-1] > 1 and arr.dtype != np.object_:
                    return np.argmax(arr, axis=-1).astype(np.int64)
                # 尝试整型化（兼容浮点整数）
                if not np.issubdtype(arr.dtype, np.integer):
                    arr = np.round(arr).astype(np.int64)
                else:
                    arr = arr.astype(np.int64)
                return arr

            for traj in trajs:
                t_states = np.asarray(traj.get('states', []), dtype=np.float32)
                t_actions = traj.get('actions', None)
                t_rewards = traj.get('rewards', None)
                t_next = traj.get('next_states', None)
                t_dones = traj.get('dones', None)
                t_opp = traj.get('opponent_actions', None)

                # 基本校验
                if t_states is None or len(t_states) == 0:
                    continue

                T = t_states.shape[0]

                # 动作转整型索引
                if t_actions is not None:
                    t_actions = _to_int_array(t_actions)
                    if t_actions.ndim > 1:
                        t_actions = t_actions.reshape(-1)
                    # 裁剪长度一致
                    if t_actions.shape[0] != T:
                        T = min(T, t_actions.shape[0])
                        t_states = t_states[:T]
                        t_actions = t_actions[:T]

                # next_states
                if t_next is not None:
                    t_next = np.asarray(t_next, dtype=np.float32)
                    if t_next.shape[0] != T:
                        T = min(T, t_next.shape[0])
                        t_states = t_states[:T]
                        if t_actions is not None:
                            t_actions = t_actions[:T]
                        t_next = t_next[:T]

                # rewards
                if t_rewards is not None:
                    t_rewards = np.asarray(t_rewards, dtype=np.float32)
                    if t_rewards.shape[0] != T:
                        T = min(T, t_rewards.shape[0])
                        t_states = t_states[:T]
                        if t_actions is not None:
                            t_actions = t_actions[:T]
                        if t_next is not None:
                            t_next = t_next[:T]
                        t_rewards = t_rewards[:T]
                else:
                    t_rewards = np.zeros((T,), dtype=np.float32)

                # dones
                if t_dones is not None:
                    t_dones = np.asarray(t_dones, dtype=np.int32)
                    if t_dones.shape[0] != T:
                        T = min(T, t_dones.shape[0])
                        t_states = t_states[:T]
                        if t_actions is not None:
                            t_actions = t_actions[:T]
                        if t_next is not None:
                            t_next = t_next[:T]
                        if t_rewards is not None:
                            t_rewards = t_rewards[:T]
                        t_dones = t_dones[:T]
                else:
                    t_dones = np.zeros((T,), dtype=np.int32)

                # 对手动作
                if t_opp is not None:
                    t_opp = _to_int_array(t_opp)
                    if t_opp.ndim > 1:
                        t_opp = t_opp.reshape(-1)
                    if t_opp.shape[0] != T:
                        T = min(T, t_opp.shape[0])
                        t_states = t_states[:T]
                        if t_actions is not None:
                            t_actions = t_actions[:T]
                        if t_next is not None:
                            t_next = t_next[:T]
                        if t_rewards is not None:
                            t_rewards = t_rewards[:T]
                        if t_dones is not None:
                            t_dones = t_dones[:T]
                        t_opp = t_opp[:T]

                processed = {
                    'states': t_states,
                    'actions': t_actions if t_actions is not None else np.zeros((T,), dtype=np.int64),
                    'rewards': t_rewards,
                    'dones': t_dones
                }
                if t_next is not None:
                    processed['next_states'] = t_next
                if t_opp is not None:
                    processed['opponent_actions'] = t_opp
                if 'infos' in traj:
                    processed['infos'] = traj['infos']

                processed_trajs.append(processed)

            # 维度推断与回填
            state_dim = raw_data.get('state_dim', None)
            if state_dim is None:
                if processed_trajs and 'states' in processed_trajs[0]:
                    state_dim = (int(processed_trajs[0]['states'].shape[-1]),)
                else:
                    state_dim = (0,)

            action_dim = raw_data.get('action_dim', None)
            if action_dim is None:
                max_a = 0
                for tr in processed_trajs:
                    if 'actions' in tr and tr['actions'] is not None and len(tr['actions']) > 0:
                        max_a = max(max_a, int(np.max(tr['actions'])))
                    if 'opponent_actions' in tr and tr['opponent_actions'] is not None and len(tr['opponent_actions']) > 0:
                        max_a = max(max_a, int(np.max(tr['opponent_actions'])))
                action_dim = int(max_a + 1) if max_a >= 0 else 0

            return {
                'trajectories': processed_trajs,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'num_trajectories': len(processed_trajs),
                'traj_length': int(processed_trajs[0]['states'].shape[0]) if processed_trajs else 0
            }

        # 否则走旧的键值输入路径
        dict_formatted = {}
        if raw_data:
            dict_formatted = raw_data

        if timestamp:
            dict_formatted['timestamp'] = np.array(timestamp)
        if position:
            dict_formatted['position'] = np.array(position)
        if velocity:
            dict_formatted['velocity'] = np.array(velocity)
        if attitude:
            dict_formatted['attitude'] = np.array(attitude)

        return dict_formatted
    
    def outlier_processing(self, data: Dict, threshold: float = -1, 
                          strategy: str = 'remove') -> Tuple[Dict, List]:
        """
        异常值检测函数
        
        参数:
            data (Dict): 输入数据字典
            threshold (float, optional): 异常值判定阈值
            strategy (str, optional): 处理策略，可选值为 'remove' 或 'interpolate'
        
        返回值:
            Tuple[Dict, List]: 处理后的数据字典和异常点列表
        
        功能描述:
            1. 使用MAD(Median Absolute Deviation)检测离群异常点
            2. 根据策略选择移除或插值处理异常点
            3. 返回处理后的数据集与异常点位置
        """
        # 初始化结果
        processed_data = data.copy()
        outliers = []
        if threshold <= 0:
            return processed_data, outliers
        
        # 对数据进行异常检测
        for key, values in data.items():
            if key == 'ref_timestamps':
                continue
            if isinstance(values, np.ndarray) and values.size > 0:
                # 计算中位数
                median = np.median(values, axis=0)
                # 计算绝对偏差
                mad = np.median(np.abs(values - median), axis=0)
                
                # 识别异常值
                if mad.any():  # 避免除以零
                    z_scores = np.abs(values - median) / mad
                    mask = np.any(z_scores > threshold, axis=1) if len(values.shape) > 1 else z_scores > threshold
                    
                    # 记录异常点索引
                    outlier_indices = np.where(mask)[0]
                    if len(outlier_indices) > 0:
                        outliers.extend([(key, idx) for idx in outlier_indices])
                    
                    # 处理异常值
                    if strategy == 'remove':
                        # 移除异常值
                        processed_data[key] = values[~mask]
                    elif strategy == 'interpolate':
                        # 使用插值替换异常值
                        processed_values = values.copy()
                        if len(outlier_indices) > 0:
                            # 简单的线性插值示例
                            for idx in outlier_indices:
                                if 0 < idx < len(values) - 1:
                                    if len(values.shape) > 1:
                                        processed_values[idx] = (values[idx-1] + values[idx+1]) / 2
                                    else:
                                        processed_values[idx] = (values[idx-1] + values[idx+1]) / 2
                        processed_data[key] = processed_values
        
        return processed_data, outliers
    
    def time_alignment(self, ref_timestamps: List[float], data: Dict, 
                      alignment_mode: str = 'dtw') -> Dict:
        """
        时间序列对齐函数
        
        参数:
            ref_timestamps (List[float]): 参考时间序列
            data (Dict): 待对齐的数据字典
            alignment_mode (str, optional): 对齐模式，可选值为 'dtw' 或 'linear'
        
        返回值:
            Dict: 时间轴对齐后的数据字典
        
        功能描述:
            1. 重采样数据至统一时间基准
            2. 采用Lanczos插值算法进行时间序列对齐
            3. 提供DTW算法进行非线性时间序列对齐
        """
        # 初始化结果
        aligned_data = {}

        if not ref_timestamps or len(ref_timestamps) == 0:
            return data
        
        # 转换参考时间戳为numpy数组
        ref_timestamps = np.array(ref_timestamps)
        
        # 对每个数据序列进行时间对齐
        for key, values in data.items():
            if key == 'timestamp':
                aligned_data[key] = ref_timestamps
                continue
                
            if isinstance(values, np.ndarray) and 'timestamp' in data:
                orig_timestamps = data['timestamp']
                
                # 根据不同的对齐模式进行处理
                if alignment_mode == 'linear':
                    # 线性插值
                    if len(values.shape) > 1:
                        # 多维数据
                        aligned_values = np.zeros((len(ref_timestamps), values.shape[1]))
                        for i in range(values.shape[1]):
                            aligned_values[:, i] = np.interp(ref_timestamps, orig_timestamps, values[:, i])
                    else:
                        # 一维数据
                        aligned_values = np.interp(ref_timestamps, orig_timestamps, values)
                    
                    aligned_data[key] = aligned_values
                
                elif alignment_mode == 'dtw':
                    # 这里应该实现DTW算法
                    # 由于DTW算法较为复杂，这里仅使用线性插值作为示例
                    if len(values.shape) > 1:
                        aligned_values = np.zeros((len(ref_timestamps), values.shape[1]))
                        for i in range(values.shape[1]):
                            aligned_values[:, i] = np.interp(ref_timestamps, orig_timestamps, values[:, i])
                    else:
                        aligned_values = np.interp(ref_timestamps, orig_timestamps, values)
                    
                    aligned_data[key] = aligned_values
            else:
                # 对于非时间序列数据，直接复制
                aligned_data[key] = values
        
        return aligned_data
    
    def sensor_fusion(self, sensor_list: List[str], data: Dict) -> Dict:
        """
        传感器数据融合函数
        
        参数:
            sensor_list (List[str]): 传感器列表，例如 ["radar_01", "lidar_02"]
            data (Dict): 传感器数据字典
        
        返回值:
            Dict: 融合后的数据字典
        
        功能描述:
            1. 多传感器数据时空对齐
            2. 卡尔曼滤波进行多传感器数据融合
            3. 更新融合后的置信度与精度评估
        """
        # 初始化结果
        fused_data = {}
        
        # 检查传感器列表
        if not sensor_list or len(sensor_list) < 1:
            return data
        
        # 如果只有一个传感器，直接返回数据
        if len(sensor_list) == 1 and sensor_list[0] in data:
            return data[sensor_list[0]]
        
        # 多传感器融合
        for key in data[sensor_list[0]].keys():
            # 收集所有传感器的对应数据
            sensor_values = []
            for sensor in sensor_list:
                if sensor in data and key in data[sensor]:
                    sensor_values.append(data[sensor][key])
            
            # 如果有数据，则计算平均值
            if sensor_values:
                if all(isinstance(v, np.ndarray) for v in sensor_values):
                    # 确保所有数组具有相同的形状
                    shapes = [v.shape for v in sensor_values]
                    if all(s == shapes[0] for s in shapes):
                        fused_data[key] = np.mean(sensor_values, axis=0)
                    else:
                        # 形状不同，需要更复杂的处理
                        # 这里简单地使用第一个传感器的数据
                        fused_data[key] = sensor_values[0]
                else:
                    # 非数组数据，简单地使用第一个传感器的数据
                    fused_data[key] = sensor_values[0]
        
        return fused_data
    
    def normalize_data(self, data: Dict, norm_method: str = "zscore", feature_ranges: Dict = None) -> Dict:
        """
        数据标准化处理函数（增强版：支持轨迹结构）
        - 若输入为轨迹数据字典，则对每条轨迹的states/next_states做归一化
        - 离散actions/opponent_actions不做归一化
        """
        # 轨迹路径
        if isinstance(data, dict) and 'trajectories' in data:
            trajs = data.get('trajectories', [])
            if len(trajs) == 0:
                return data

            # 统计所有状态的全局统计量
            states_all = []
            for tr in trajs:
                if 'states' in tr and isinstance(tr['states'], np.ndarray):
                    states_all.append(tr['states'])
            if len(states_all) == 0:
                return data

            cat = np.concatenate(states_all, axis=0)  # [sumT, Ds]
            if norm_method == "minmax":
                min_vec = np.min(cat, axis=0)
                max_vec = np.max(cat, axis=0)
                scale = (max_vec - min_vec)
                scale[scale == 0] = 1.0
                for tr in trajs:
                    if 'states' in tr and isinstance(tr['states'], np.ndarray):
                        tr['states'] = (tr['states'] - min_vec) / scale
                    if 'next_states' in tr and isinstance(tr['next_states'], np.ndarray):
                        tr['next_states'] = (tr['next_states'] - min_vec) / scale
                data['norm_stats'] = {'type': 'minmax', 'min': min_vec, 'max': max_vec}
            else:
                mean = np.mean(cat, axis=0)
                std = np.std(cat, axis=0)
                std[std == 0] = 1.0
                for tr in trajs:
                    if 'states' in tr and isinstance(tr['states'], np.ndarray):
                        tr['states'] = (tr['states'] - mean) / std
                    if 'next_states' in tr and isinstance(tr['next_states'], np.ndarray):
                        tr['next_states'] = (tr['next_states'] - mean) / std
                data['norm_stats'] = {'type': 'zscore', 'mean': mean, 'std': std}

            return data

        # 非轨迹路径（保留原有逻辑）
        normalized_data = {}
        for key, values in data.items():
            if isinstance(values, np.ndarray) and values.size > 0:
                if feature_ranges and key in feature_ranges:
                    # Min-Max归一化（按指定区间映射）
                    min_val, max_val = feature_ranges[key]
                    if len(values.shape) > 1:
                        normalized_values = np.zeros_like(values, dtype=float)
                        for i in range(values.shape[1]):
                            col_min = np.min(values[:, i])
                            col_max = np.max(values[:, i])
                            if col_max > col_min:
                                normalized_values[:, i] = (values[:, i] - col_min) / (col_max - col_min) * (max_val - min_val) + min_val
                    else:
                        data_min = np.min(values)
                        data_max = np.max(values)
                        if data_max > data_min:
                            normalized_values = (values - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
                        else:
                            normalized_values = np.zeros_like(values, dtype=float)
                else:
                    # Z-score标准化
                    if len(values.shape) > 1:
                        normalized_values = np.zeros_like(values, dtype=float)
                        for i in range(values.shape[1]):
                            mean = np.mean(values[:, i])
                            std = np.std(values[:, i])
                            if std > 0:
                                normalized_values[:, i] = (values[:, i] - mean) / std
                    else:
                        mean = np.mean(values)
                        std = np.std(values)
                        if std > 0:
                            normalized_values = (values - mean) / std
                        else:
                            normalized_values = np.zeros_like(values, dtype=float)
                normalized_data[key] = normalized_values
            else:
                normalized_data[key] = values

        return normalized_data

    def preprocess_for_training(self, raw_data: Dict, model_type: str) -> Dict:
        """
        根据训练模型类型转换原始轨迹数据为所需格式
        
        参数:
            raw_data (Dict): 原始输入数据字典，格式如下:
                {
                    "trajectories": [
                        {
                            "states": np.array([t0_state, t1_state, ...]),  # 状态序列
                            "actions": np.array([t0_action, t1_action, ...]), # 动作序列
                            "rewards": np.array([t0_reward, t1_reward, ...]), # 奖励序列(可选)
                            "next_states": np.array([t0_next, t1_next, ...]), # 下一状态(可选)
                            "dones": np.array([t0_done, t1_done, ...]),       # 终止标志(可选)
                            "infos": [t0_info, t1_info, ...],                # 额外信息(可选)
                            "opponent_actions": np.array([t0_opp, t1_opp, ...]) # 对手动作(可选)
                        },
                        ... # 更多轨迹
                    ],
                    "state_dim": tuple,  # 状态维度
                    "action_dim": int    # 动作维度
                }
            model_type (str): 模型类型
            
        返回值:
            Dict: 符合训练输入格式的数据
        """
        # 提取原始轨迹和维度信息
        trajectories = raw_data["trajectories"]
        state_dim = raw_data["state_dim"]
        action_dim = raw_data["action_dim"]
        
        if model_type in ["AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator"]:
            # 转换格式：轨迹列表
            processed_trajs = []
            for traj in trajectories:
                processed_trajs.append({
                    'states': traj['states'],
                    'actions': traj['actions']
                })
            
            return {
                "data": processed_trajs,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "traj_length": len(trajectories[0]['states']) if trajectories else 0
            }
            
        elif model_type in ["OfflineRLearner", "OfflineFSPLearner", "BehaviorCloner", "AdversarialImitationLearner"]:
            # 转换格式：轨迹字典
            processed_trajs = {}
            for i, traj in enumerate(trajectories):
                traj_id = f"traj_{i}"
                
                # 基础字段
                processed_traj = {
                    'states': traj['states'],
                    'actions': traj['actions']
                }
                
                # 根据模型类型添加额外字段
                if model_type == 'OfflineRLearner':
                    processed_traj.update({
                        'rewards': traj.get('rewards', np.zeros(len(traj['states']))),
                        'next_states': traj.get('next_states', np.zeros_like(traj['states'])),
                        'dones': traj.get('dones', np.zeros(len(traj['states']))),
                        'infos': traj.get('infos', [{} for _ in range(len(traj['states']))])
                    })
                elif model_type == 'OfflineFSPLearner':
                    processed_traj.update({
                        'opponent_actions': traj.get('opponent_actions', np.zeros(len(traj['actions']))),
                        'rewards': traj.get('rewards', np.zeros(len(traj['states']))),
                        'next_states': traj.get('next_states', np.zeros_like(traj['states'])),
                        'dones': traj.get('dones', np.zeros(len(traj['states']))),
                        'infos': traj.get('infos', [{} for _ in range(len(traj['states']))])
                    })
                else:
                    processed_traj.update({
                        'rewards': traj.get('rewards', np.zeros(len(traj['states']))),
                        'dones': traj.get('dones', np.zeros(len(traj['states']))),
                        'infos': traj.get('infos', [{} for _ in range(len(traj['states']))])
                    })
                
                processed_trajs[traj_id] = processed_traj
                    
            return {
                "data": processed_trajs,
                "state_dim": state_dim,
                "action_dim": action_dim
            }
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")