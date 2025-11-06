import os
import sys
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具模块
from utils.DataPreprocessor import DataPreprocessor
from utils.TrajectoryPreprocessor import TrajectoryPreprocessor
from utils.OflineEvaluator import OfflineEvaluator
from utils.MultiMetricEvaluator import MultiMetricEvaluator

# 导入基础算法模块
from utils.BehaviorCloner import BehaviorCloner
from utils.AdversarialImitationLearner import AdversarialImitationLearner
from utils.OfflineRLearner import OfflineRLearner
from utils.OfflineFSPLearner import OfflineFSPLearner

# 导入轨迹建模与生成模型
from utils.AutoencoderModel import AutoencoderModel
from utils.TransformerModel import TransformerModel
from utils.DiffusionTrajGenerator import DiffusionTrajGenerator
from utils.VAETrajGenerator import VAETrajGenerator

# 导入训练方法模块
from utils.EvolutionaryLearner import EvolutionaryLearner
from utils.IncrementalLearner import IncrementalLearner
from utils.FineTuneManager import FineTuneManager

# 导入服务层模块
from core.services.game_modeling_service import GameModelingService
from core.services.imitation_learning_service import ImitationLearningService
from core.services.visualization_service import VisualizationService
from core.services.logging_service import LoggingService

class AlgorithmAPI:
    """
    算法小样本快速升级迭代软件的主API类
    
    提供统一的接口，用于小样本博弈建模和专家轨迹模仿学习
    """
    
    def __init__(self, config: Dict = None, log_level: str = "INFO"):
        """
        初始化API
        
        参数:
            config (Dict): 配置参数字典
            log_level (str): 日志级别，可选值为 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        """
        # 初始化配置
        self.config = config or {}
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化日志服务
        self.logging_service = LoggingService(log_level)
        self.logger = self.logging_service.get_logger()
        self.logger.info(f"初始化API，使用设备: {self.device}")
        
        # 初始化服务
        self.game_modeling_service = GameModelingService(self.config, self.logger)
        self.imitation_learning_service = ImitationLearningService(self.config, self.logger)
        self.visualization_service = VisualizationService(self.config, self.logger)
        
        # 初始化数据预处理器
        self.data_preprocessor = DataPreprocessor()
        self.trajectory_preprocessor = TrajectoryPreprocessor()
        
        # 初始化评估器
        self.offline_evaluator = OfflineEvaluator()
        self.multi_metric_evaluator = MultiMetricEvaluator()
        
        self.logger.info("API初始化完成")
    
    def load_data(self, data_path: str, data_format: str = "json") -> Dict:
        """
        加载数据
        
        参数:
            data_path (str): 数据文件路径
            data_format (str): 数据格式，支持 "json", "csv", "npy"
            
        返回:
            Dict: 加载的数据
        """
        self.logger.info(f"加载数据: {data_path}, 格式: {data_format}")
        
        try:
            # 根据数据格式选择不同的加载方法
            if data_format == "json":
                import json
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif data_format == "csv":
                import pandas as pd
                data = pd.read_csv(data_path).to_dict('records')
            elif data_format == "npy":
                data = np.load(data_path, allow_pickle=True).item()
            else:
                raise ValueError(f"不支持的数据格式: {data_format}")
            
            self.logger.info(f"数据加载成功，包含 {len(data)} 条记录")
            return data
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def preprocess_data(self, raw_data: Dict, preprocess_config: Dict = None) -> Dict:
        """
        预处理数据
        
        参数:
            raw_data (Dict): 原始数据
            preprocess_config (Dict): 预处理配置
            
        返回:
            Dict: 预处理后的数据
        """
        self.logger.info("开始数据预处理")
        
        try:
            # 使用数据预处理器处理数据
            processed_data = self.data_preprocessor.load_data(raw_data)
            
            # 如果有异常值处理配置
            if preprocess_config and 'outlier_threshold' in preprocess_config:
                processed_data, outliers = self.data_preprocessor.outlier_processing(
                    processed_data, 
                    threshold=preprocess_config.get('outlier_threshold', 3.0),
                    strategy=preprocess_config.get('outlier_strategy', 'remove')
                )
                self.logger.info(f"异常值处理完成，检测到 {len(outliers)} 个异常点")
            
            # 如果有时间对齐配置
            if preprocess_config and 'time_alignment' in preprocess_config:
                if 'ref_timestamps' in processed_data:
                    processed_data = self.data_preprocessor.time_alignment(
                        processed_data['ref_timestamps'],
                        processed_data,
                        alignment_mode=preprocess_config.get('alignment_mode', 'linear')
                    )
                    self.logger.info("时间序列对齐完成")
            
            # 如果有数据标准化配置
            if preprocess_config and 'normalize' in preprocess_config and preprocess_config['normalize']:
                feature_ranges = preprocess_config.get('feature_ranges', None)
                processed_data = self.data_preprocessor.normalize_data(processed_data, feature_ranges)
                self.logger.info("数据标准化完成")
            
            self.logger.info("数据预处理完成")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def preprocess_trajectory(self, trajectories: Dict, preprocess_config: Dict = None) -> Dict:
        """
        预处理轨迹数据
        
        参数:
            trajectories (Dict): 轨迹数据
            preprocess_config (Dict): 预处理配置
            
        返回:
            Dict: 预处理后的轨迹数据
        """
        self.logger.info("开始轨迹数据预处理")
        
        try:
            # 使用轨迹预处理器处理数据
            processed_trajectories = self.trajectory_preprocessor.process_trajectories(
                trajectories,
                config=preprocess_config
            )
            
            self.logger.info("轨迹数据预处理完成")
            return processed_trajectories
            
        except Exception as e:
            self.logger.error(f"轨迹数据预处理失败: {str(e)}")
            raise
    
    def train_game_model(self, training_data: Dict, model_config: Dict) -> Dict:
        """
        训练小样本博弈模型
        
        参数:
            training_data (Dict): 训练数据
            model_config (Dict): 模型配置
            
        返回:
            Dict: 训练结果，包含模型和训练指标
        """
        self.logger.info("开始训练小样本博弈模型")
        
        try:
            # 使用博弈建模服务训练模型
            result = self.game_modeling_service.train_model(training_data, model_config)
            
            self.logger.info("小样本博弈模型训练完成")
            return result
            
        except Exception as e:
            self.logger.error(f"小样本博弈模型训练失败: {str(e)}")
            raise
    
    def train_imitation_model(self, expert_trajectories: Dict, model_config: Dict) -> Dict:
        """
        训练小样本专家轨迹模仿学习模型
        
        参数:
            expert_trajectories (Dict): 专家轨迹数据
            model_config (Dict): 模型配置
            
        返回:
            Dict: 训练结果，包含模型和训练指标
        """
        self.logger.info("开始训练小样本专家轨迹模仿学习模型")
        
        try:
            # 使用模仿学习服务训练模型
            result = self.imitation_learning_service.train_model(expert_trajectories, model_config)
            
            self.logger.info("小样本专家轨迹模仿学习模型训练完成")
            return result
            
        except Exception as e:
            self.logger.error(f"小样本专家轨迹模仿学习模型训练失败: {str(e)}")
            raise
    
    def evaluate_model(self, model: Any, test_data: Dict, eval_config: Dict = None) -> Dict:
        """
        评估模型
        
        参数:
            model (Any): 待评估的模型
            test_data (Dict): 测试数据
            eval_config (Dict): 评估配置
            
        返回:
            Dict: 评估结果
        """
        self.logger.info("开始模型评估")
        
        try:
            # 根据评估配置选择评估方法
            eval_method = eval_config.get('method', 'offline') if eval_config else 'offline'
            
            if eval_method == 'offline':
                # 使用离线评估器评估模型
                result = self.offline_evaluator.evaluate_policy(
                    model, 
                    test_data, 
                    method_type=eval_config.get('method_type', 'IS') if eval_config else 'IS'
                )
            elif eval_method == 'multi_metric':
                # 使用多指标评估器评估模型
                metric_config = eval_config.get('metric_config', {}) if eval_config else {}
                result = self.multi_metric_evaluator.calculate_metrics(
                    metric_config,
                    analysis_mode=eval_config.get('analysis_mode', 'micro') if eval_config else 'micro'
                )
                
                # 融合指标
                result.update(self.multi_metric_evaluator.fuse_metrics(
                    custom_weights=eval_config.get('weights', None) if eval_config else None
                ))
            else:
                raise ValueError(f"不支持的评估方法: {eval_method}")
            
            self.logger.info("模型评估完成")
            return result
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def visualize_results(self, data: Dict, vis_config: Dict = None) -> Dict:
        """
        可视化结果
        
        参数:
            data (Dict): 可视化数据
            vis_config (Dict): 可视化配置
            
        返回:
            Dict: 可视化结果，包含图表数据和保存路径
        """
        self.logger.info("开始结果可视化")
        
        try:
            # 使用可视化服务生成可视化结果
            result = self.visualization_service.visualize(data, vis_config)
            
            self.logger.info("结果可视化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"结果可视化失败: {str(e)}")
            raise
    
    def save_model(self, model: Any, save_path: str, model_format: str = "pytorch") -> str:
        """
        保存模型
        
        参数:
            model (Any): 待保存的模型
            save_path (str): 保存路径
            model_format (str): 模型格式，支持 "pytorch", "onnx"
            
        返回:
            str: 模型保存路径
        """
        self.logger.info(f"开始保存模型: {save_path}, 格式: {model_format}")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 根据模型格式选择不同的保存方法
            if model_format == "pytorch":
                torch.save(model.state_dict(), save_path)
            elif model_format == "onnx":
                # 需要模型实现了onnx导出方法
                if hasattr(model, 'export_onnx'):
                    model.export_onnx(save_path)
                else:
                    raise NotImplementedError("模型未实现ONNX导出方法")
            else:
                raise ValueError(f"不支持的模型格式: {model_format}")
            
            self.logger.info(f"模型保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def load_model(self, model_class: Any, model_path: str, model_config: Dict = None) -> Any:
        """
        加载模型
        
        参数:
            model_class (Any): 模型类
            model_path (str): 模型路径
            model_config (Dict): 模型配置
            
        返回:
            Any: 加载的模型
        """
        self.logger.info(f"开始加载模型: {model_path}")
        
        try:
            # 初始化模型
            model = model_class(**(model_config or {}))
            
            # 加载模型参数
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            self.logger.info(f"模型加载成功: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def get_available_models(self) -> Dict:
        """
        获取可用模型列表
        
        返回:
            Dict: 可用模型信息
        """
        models = {
            "基础算法模型": {
                "BehaviorCloner": BehaviorCloner,
                "AdversarialImitationLearner": AdversarialImitationLearner,
                "OfflineRLearner": OfflineRLearner,
                "OfflineFSPLearner": OfflineFSPLearner
            },
            "轨迹建模与生成模型": {
                "AutoencoderModel": AutoencoderModel,
                "TransformerModel": TransformerModel,
                "DiffusionTrajGenerator": DiffusionTrajGenerator,
                "VAETrajGenerator": VAETrajGenerator
            },
            "训练方法模型": {
                "EvolutionaryLearner": EvolutionaryLearner,
                "IncrementalLearner": IncrementalLearner,
                "FineTuneManager": FineTuneManager
            }
        }
        
        return models