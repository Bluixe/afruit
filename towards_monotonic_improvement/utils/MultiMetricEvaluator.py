import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class MultiMetricEvaluator:
    """
    多指标评估器类
    
    功能描述：基于多维度指标系统对强化学习与多维度综合评估
    
    核心功能：
    • 分层评估体系：支/微观指标分层计算与整合
    • 多种指标聚合：支持传感器数据与专家评分的细分分析
    • 闭环优化流动：评估结果直连强化优化模块
    """
    
    def __init__(self, agg_weights: Dict = None):
        """
        初始化多指标评估器
        
        参数:
            agg_weights (Dict): 层次分析法指标权重配置
        """
        # 默认权重配置
        self.agg_weights = {
            'macro_metrics': 0.6,
            'micro_metrics': 0.4
        } if agg_weights is None else agg_weights
        
        # 验证权重配置
        self._validate_weights()
    
    def _validate_weights(self):
        """验证权重配置是否合法"""
        # 检查权重和是否为1
        weight_sum = sum(self.agg_weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-5):
            raise ValueError(f"权重总和必须为1.0，当前值: {weight_sum}")
        
        # 检查权重是否为非负值
        for key, weight in self.agg_weights.items():
            if weight < 0:
                raise ValueError(f"权重必须为非负值，{key}的当前值: {weight}")
    
    def load_data(self, trajectory_data: List = None, 
                 expert_scores: Dict = None, 
                 resource_stats: Dict = None) -> Dict:
        """
        数据加载函数
        
        参数:
            trajectory_data (List): 原始轨迹数据集
            expert_scores (Dict): 专家评分数据
            resource_stats (DataFrames): 资源消耗统计表
        
        返回值:
            Dict: 标准化数据集
        
        处理流程:
            1. 数据完整性检查
            2. 时间序列对齐
            3. 异常值过滤
        """
        # 初始化结果字典
        processed_data = {
            'macro_metrics': {},
            'micro_metrics': {},
            'expert_annotations': {}
        }
        
        # 处理轨迹数据
        if trajectory_data:
            # 数据完整性检查
            valid_trajectories = self._validate_trajectories(trajectory_data)
            processed_data['macro_metrics'] = valid_trajectories
        
        # 处理专家评分数据
        if expert_scores:
            processed_data['expert_annotations'] = expert_scores
        
        # 处理资源统计数据
        if resource_stats:
            processed_data['micro_metrics'] = resource_stats
        
        return processed_data
    
    def _validate_trajectories(self, trajectories: List) -> Dict:
        """验证轨迹数据并返回有效数据"""
        valid_data = {}
        
        for i, traj in enumerate(trajectories):
            # 检查轨迹数据结构
            if not isinstance(traj, dict):
                print(f"警告: 轨迹 {i} 不是字典格式，已跳过")
                continue
            
            # 检查必要字段
            required_fields = ['states', 'actions', 'rewards']
            if not all(field in traj for field in required_fields):
                print(f"警告: 轨迹 {i} 缺少必要字段，已跳过")
                continue
            
            # 检查数据长度一致性
            lengths = [len(traj[field]) for field in required_fields]
            if len(set(lengths)) > 1:
                print(f"警告: 轨迹 {i} 的字段长度不一致，已跳过")
                continue
            
            # 添加到有效数据
            valid_data[f'trajectory_{i}'] = traj
        
        return valid_data
    
    def calculate_metrics(self, metric_config: Dict, analysis_mode: str = "micro") -> Dict:
        """
        指标计算函数
        
        参数:
            metric_config (Dict): 指标计算规则配置
            analysis_mode (str): 分析模式，"micro"或"macro"
        
        返回值:
            指标计算结果 (Dict)
        
        算法实现:
            macro_scores: 宏指标得分
            micro_scores: 微指标得分
            correlation_matrix: 指标相关性矩阵
        """
        # 检查分析模式
        valid_modes = ["micro", "macro"]
        if analysis_mode not in valid_modes:
            raise ValueError(f"analysis_mode必须为 {valid_modes} 之一，当前值: {analysis_mode}")
        
        # 初始化结果字典
        result = {
            'macro_scores': {},
            'micro_scores': {},
            'correlation_matrix': None
        }
        
        # 根据分析模式计算指标
        if analysis_mode == "macro":
            result['macro_scores'] = self._calculate_macro_metrics(metric_config)
        else:  # micro
            result['micro_scores'] = self._calculate_micro_metrics(metric_config)
        
        # 计算指标相关性矩阵
        if result['macro_scores'] and result['micro_scores']:
            result['correlation_matrix'] = self._calculate_correlation_matrix(
                result['macro_scores'], result['micro_scores']
            )
        
        return result
    
    def _calculate_macro_metrics(self, config: Dict) -> Dict:
        """计算宏观指标"""
        macro_scores = {}
        
        # 检查配置
        if not config or not isinstance(config, dict):
            return macro_scores
        
        # 提取指标计算参数
        metrics = config.get('metrics', [])
        data = config.get('data', {})
        
        # 计算各项指标
        for metric in metrics:
            metric_name = metric.get('name')
            metric_func = metric.get('function')
            
            if not metric_name or not metric_func:
                continue
            
            # 根据函数名调用相应的计算方法
            if metric_func == 'rmse':
                if 'predictions' in data and 'targets' in data:
                    macro_scores[metric_name] = np.sqrt(mean_squared_error(
                        data['targets'], data['predictions']
                    ))
            elif metric_func == 'mae':
                if 'predictions' in data and 'targets' in data:
                    macro_scores[metric_name] = mean_absolute_error(
                        data['targets'], data['predictions']
                    )
            elif metric_func == 'cosine_sim':
                if 'vec1' in data and 'vec2' in data:
                    macro_scores[metric_name] = cosine_similarity(
                        [data['vec1']], [data['vec2']]
                    )[0][0]
        
        return macro_scores
    
    def _calculate_micro_metrics(self, config: Dict) -> Dict:
        """计算微观指标"""
        micro_scores = {}
        
        # 检查配置
        if not config or not isinstance(config, dict):
            return micro_scores
        
        # 提取指标计算参数
        metrics = config.get('metrics', [])
        data = config.get('data', {})
        
        # 计算各项指标
        for metric in metrics:
            metric_name = metric.get('name')
            metric_func = metric.get('function')
            
            if not metric_name or not metric_func:
                continue
            
            # 根据函数名调用相应的计算方法
            if metric_func == 'resource_usage':
                if 'cpu_usage' in data and 'memory_usage' in data:
                    # 计算资源使用率
                    cpu_weight = 0.7
                    mem_weight = 0.3
                    micro_scores[metric_name] = (
                        cpu_weight * np.mean(data['cpu_usage']) + 
                        mem_weight * np.mean(data['memory_usage'])
                    )
            elif metric_func == 'time_efficiency':
                if 'execution_times' in data:
                    # 计算时间效率
                    micro_scores[metric_name] = 1.0 / (1.0 + np.mean(data['execution_times']))
            elif metric_func == 'stability':
                if 'metrics_variance' in data:
                    # 计算稳定性指标
                    micro_scores[metric_name] = 1.0 / (1.0 + np.mean(data['metrics_variance']))
        
        return micro_scores
    
    def _calculate_correlation_matrix(self, macro_scores: Dict, micro_scores: Dict) -> np.ndarray:
        """计算指标相关性矩阵"""
        # 合并所有指标
        all_metrics = {}
        all_metrics.update(macro_scores)
        all_metrics.update(micro_scores)
        
        # 转换为DataFrame
        df = pd.DataFrame({key: [value] for key, value in all_metrics.items()})
        
        # 计算相关性矩阵
        if len(df.columns) > 1:
            return df.corr().values
        else:
            return np.array([[1.0]])
    
    def fuse_metrics(self, custom_weights: Dict = None) -> Dict:
        """
        指标融合函数
        
        参数:
            custom_weights (Dict): 自定义权重配置（可选）
        
        返回值:
            融合评估结果 (Dict)
        
        评估结果:
            composite_scores: 融合综合得分
            dimension_breakdown: 各维度分解分析
            optimization_suggestions: 优化建议参数表
        """
        # 使用自定义权重或默认权重
        weights = custom_weights if custom_weights else self.agg_weights
        
        # 初始化结果字典
        result = {
            'composite_scores': {},
            'dimension_breakdown': {},
            'optimization_suggestions': {}
        }
        
        # 模拟计算综合得分
        # 在实际应用中，这里应该使用真实的指标数据
        macro_score = 0.85  # 示例值
        micro_score = 0.72  # 示例值
        
        # 计算综合得分
        composite_score = (
            weights.get('macro_metrics', 0.6) * macro_score + 
            weights.get('micro_metrics', 0.4) * micro_score
        )
        
        result['composite_scores'] = {
            'overall': composite_score,
            'macro': macro_score,
            'micro': micro_score
        }
        
        # 维度分解分析
        result['dimension_breakdown'] = {
            'performance': 0.88,
            'efficiency': 0.76,
            'stability': 0.82
        }
        
        # 优化建议
        result['optimization_suggestions'] = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'regularization': 0.0001
        }
        
        return result
    
    def eval_diversity(self, data_for_eval: Dict) -> Dict:
        """
        多样性评估函数
        
        参数:
            data_for_eval (Dict): 预处理后用于多样性评估的数据
        
        返回值:
            结构化数据 (Dict): 包含评估结果集群和可视化结果
        
        流程实现:
            1. 高维空间聚类：使用KMeans/K-means聚类方法
            2. 行为模式分析：计算行为特征的分布特性
        """
        # 检查输入数据
        if not data_for_eval or not isinstance(data_for_eval, dict):
            raise ValueError("data_for_eval必须是非空字典")
        
        # 初始化结果字典
        result = {
            'clusters': {},
            'diversity_metrics': {},
            'visualization_data': {}
        }
        
        # 提取数据
        if 'states' in data_for_eval:
            states = np.array(data_for_eval['states'])
            
            # 确保数据是2D的
            if len(states.shape) > 2:
                n_samples = states.shape[0]
                states_2d = states.reshape(n_samples, -1)
            else:
                states_2d = states
            
            # 执行K-means聚类
            if states_2d.shape[0] > 1:  # 确保有足够的样本
                n_clusters = min(5, states_2d.shape[0] - 1)  # 最多5个簇
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(states_2d)
                
                # 存储聚类结果
                result['clusters'] = {
                    'labels': cluster_labels.tolist(),
                    'centers': kmeans.cluster_centers_.tolist()
                }
                
                # 计算多样性指标
                # 1. 簇间距离
                inter_cluster_distances = []
                centers = kmeans.cluster_centers_
                for i in range(n_clusters):
                    for j in range(i+1, n_clusters):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        inter_cluster_distances.append(dist)
                
                # 2. 簇内方差
                intra_cluster_variances = []
                for i in range(n_clusters):
                    cluster_points = states_2d[cluster_labels == i]
                    if len(cluster_points) > 1:
                        variance = np.var(cluster_points, axis=0).mean()
                        intra_cluster_variances.append(variance)
                
                # 存储多样性指标
                result['diversity_metrics'] = {
                    'inter_cluster_distance': np.mean(inter_cluster_distances) if inter_cluster_distances else 0,
                    'intra_cluster_variance': np.mean(intra_cluster_variances) if intra_cluster_variances else 0,
                    'cluster_count': n_clusters
                }
                
                # 准备可视化数据
                if states_2d.shape[1] > 2:
                    # 使用PCA降维到2D用于可视化
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    vis_data = pca.fit_transform(states_2d)
                else:
                    vis_data = states_2d
                
                result['visualization_data'] = {
                    'points': vis_data.tolist(),
                    'labels': cluster_labels.tolist()
                }
        
        return result