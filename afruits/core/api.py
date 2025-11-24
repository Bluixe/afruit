import os
import sys
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具模块
from afruits.utils.DataPreprocessor import DataPreprocessor
from afruits.utils.TrajectoryPreprocessor import TrajectoryPreprocessor
from afruits.utils.OflineEvaluator import OfflineEvaluator
from afruits.utils.MultiMetricEvaluator import MultiMetricEvaluator

# 导入基础算法模块
from afruits.utils.BehaviorCloner import BehaviorCloner
from afruits.utils.AdversarialImitationLearner import AdversarialImitationLearner
from afruits.utils.OfflineRLearner import OfflineRLearner
from afruits.utils.OfflineFSPLearner import OfflineFSPLearner

# 导入轨迹建模与生成模型
from afruits.utils.AutoencoderModel import AutoencoderModel, AutoencoderTrainer
from afruits.utils.TransformerModel import TransformerModel, TransformerTrainer
from afruits.utils.DiffusionTrajGenerator import DiffusionTrajGenerator
from afruits.utils.VAETrajGenerator import VAETrajGenerator

# 导入训练方法模块
from afruits.utils.EvolutionaryLearner import EvolutionaryLearner
from afruits.utils.IncrementalLearner import IncrementalLearner
from afruits.utils.FineTuneManager import FineTuneManager

# 导入服务层模块
from afruits.core.services.game_modeling_service import GameModelingService
from afruits.core.services.imitation_learning_service import ImitationLearningService
from afruits.core.services.visualization_service import VisualizationService
from afruits.core.services.logging_service import LoggingService

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
        
        # 轨迹建模和生成服务
        self.trajectory_modeling_service = self.imitation_learning_service
        self.trajectory_generation_service = self.imitation_learning_service
        
        # 初始化数据预处理器
        self.data_preprocessor = DataPreprocessor()
        self.trajectory_preprocessor = TrajectoryPreprocessor()
        
        # 初始化评估器
        self.offline_evaluator = OfflineEvaluator()
        self.multi_metric_evaluator = MultiMetricEvaluator()
        
        self.logger.info("API初始化完成")
    
    def load_data(self, data_path: str, data_format: str = "json") -> Dict:
        """
        加载数据并补齐state_dim/action_dim信息
        - 支持json/csv/npy
        - 优先读取同名_meta_data.json侧车文件中的state_dim/action_dim
        - 若无侧车文件，则从内容中推断state_dim/action_dim与轨迹列表
        返回统一结构:
            {
               "state_dim": (D,),
               "action_dim": K,
               "trajectories": [...]
            }
        """
        self.logger.info(f"加载数据: {data_path}, 格式: {data_format}")
        try:
            import json
            import os

            # 1) 读取原始数据
            if data_format == "json":
                with open(data_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
            elif data_format == "csv":
                import pandas as pd
                loaded = pd.read_csv(data_path).to_dict('records')
            elif data_format == "npy":
                loaded = np.load(data_path, allow_pickle=True)
            else:
                raise ValueError(f"不支持的数据格式: {data_format}")

            # 2) 读取/推断维度
            state_dim = None
            action_dim = None
            traj_length = None
            trajectories = None

            meta_path = f"{os.path.splitext(data_path)[0]}_meta_data.json"
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                state_dim = meta_data.get("state_dim", None)
                action_dim = meta_data.get("action_dim", None)
                traj_length = meta_data.get("traj_length", None)

            # 3) 统一为标准结构
            if isinstance(loaded, dict):
                if 'state_dim' in loaded and 'action_dim' in loaded and 'traj_length' in loaded:
                    state_dim = state_dim or loaded.get('state_dim')
                    action_dim = action_dim or loaded.get('action_dim')
                    traj_length = traj_length or loaded.get('traj_length')
                if 'trajectories' in loaded:
                    trajectories = loaded['trajectories']
                else:
                    # 兼容直接以dict存轨迹的情况
                    trajectories = loaded
            else:
                # list/其它
                trajectories = loaded

            # 4) 尝试从内容推断维度
            if state_dim is None or action_dim is None:
                try:
                    tr_list = trajectories if isinstance(trajectories, list) else list(trajectories.values())  # dict -> values
                except Exception:
                    tr_list = []

                # 从首条轨迹推断state_dim
                if state_dim is None:
                    D = 0
                    if tr_list:
                        first = tr_list[0]
                        if isinstance(first, dict) and 'states' in first:
                            st = np.asarray(first['states'])
                            if st.ndim >= 2:
                                D = int(st.shape[-1])
                    state_dim = (D,)

                # 推断action_dim(离散类别数)
                if action_dim is None:
                    max_a = -1
                    for tr in tr_list:
                        if isinstance(tr, dict):
                            if 'actions' in tr and len(tr['actions']) > 0:
                                arr = np.asarray(tr['actions'])
                                max_a = max(max_a, int(np.max(arr)))
                            if 'opponent_actions' in tr and len(tr['opponent_actions']) > 0:
                                arr = np.asarray(tr['opponent_actions'])
                                max_a = max(max_a, int(np.max(arr)))
                    action_dim = int(max_a + 1) if max_a >= 0 else 0

            # 5) 统计数量
            num_traj = 0
            if isinstance(trajectories, list):
                num_traj = len(trajectories)
            elif isinstance(trajectories, dict):
                num_traj = len(trajectories)
            elif isinstance(trajectories, np.ndarray):
                num_traj = trajectories.shape[0]

            self.logger.info(f"数据加载成功，轨迹数: {num_traj}, state_dim={state_dim}, action_dim={action_dim}")
            return {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "traj_length": traj_length,
                "trajectories": trajectories
            }

        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise

    def load_processed_data(self, data_path: str, data_format: str = "json", model_type = None) -> Dict:
        """
        加载数据并补齐state_dim/action_dim信息
        - 支持json/csv/npy
        - 优先读取同名_meta_data.json侧车文件中的state_dim/action_dim
        - 若无侧车文件，则从内容中推断state_dim/action_dim与轨迹列表
        返回统一结构:
            {
               "state_dim": (D,),
               "action_dim": K,
               "trajectories": [...]
            }
        """
        self.logger.info(f"加载数据: {data_path}, 格式: {data_format}")
        try:
            import json
            import os

            # 1) 读取原始数据
            if data_format == "json":
                with open(data_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
            elif data_format == "csv":
                import pandas as pd
                loaded = pd.read_csv(data_path).to_dict('records')
            elif data_format == "npy":
                loaded = np.load(data_path, allow_pickle=True)
            else:
                raise ValueError(f"不支持的数据格式: {data_format}")

            # 2) 读取/推断维度
            state_dim = None
            action_dim = None
            traj_length = None
            data = None

            meta_path = f"{os.path.splitext(data_path)[0]}_meta_data.json"
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                state_dim = meta_data.get("state_dim", None)
                action_dim = meta_data.get("action_dim", None)
                traj_length = meta_data.get("traj_length", None)

            # 3) 统一为标准结构
            if isinstance(loaded, dict):
                if 'state_dim' in loaded and 'action_dim' in loaded and 'traj_length' in loaded:
                    state_dim = state_dim or loaded.get('state_dim')
                    action_dim = action_dim or loaded.get('action_dim')
                    traj_length = traj_length or loaded.get('traj_length')
                if 'data' in loaded:
                    trajectories = loaded['data']
                else:
                    # 兼容直接以dict存轨迹的情况
                    trajectories = loaded
            else:
                # list/其它
                trajectories = loaded

            # 5) 统计数量
            num_traj = 0
            if isinstance(trajectories, list):
                num_traj = len(trajectories)
            elif isinstance(trajectories, dict):
                num_traj = len(trajectories)
            elif isinstance(trajectories, np.ndarray):
                num_traj = trajectories.shape[0]

            self.logger.info(f"数据加载成功，轨迹数: {num_traj}, state_dim={state_dim}, action_dim={action_dim}")
            return {
                "state_dim": tuple(state_dim),
                "action_dim": action_dim,
                "traj_length": traj_length,
                "data": trajectories
            }

        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def preprocess_data(self, raw_data: Dict, preprocess_config: Dict = None) -> Dict:
        """
        预处理数据
        - 若输入为轨迹数据字典(含trajectories)，则按轨迹路径进行处理(更贴近训练数据管线)
        - 否则走传统key-array路径
        """
        self.logger.info("开始数据预处理")
        preprocess_config = preprocess_config or {}
        try:
            # 轨迹结构路径
            if isinstance(raw_data, dict) and 'trajectories' in raw_data:
                # 标准化轨迹数据形状/类型
                processed = self.data_preprocessor.load_data(raw_data)

                # 可选异常值/对齐：当前主要针对key-array数据，轨迹路径先跳过这两步
                # 根据配置进行标准化
                if preprocess_config.get('normalize', True):
                    norm_method = 'zscore'
                    processed = self.data_preprocessor.normalize_data(
                        processed,
                        norm_method=norm_method,
                        feature_ranges=preprocess_config.get('feature_ranges', None)
                    )
                    self.logger.info("轨迹数据标准化完成")

                self.logger.info("轨迹数据预处理完成")
                return processed

            # 非轨迹路径(保留旧流程)
            processed_data = self.data_preprocessor.load_data(raw_data)

            processed_data, outliers = self.data_preprocessor.outlier_processing(
                processed_data,
                threshold=preprocess_config.get('outlier_threshold', 3.0),
                strategy=preprocess_config.get('outlier_strategy', 'remove')
            )
            self.logger.info("异常值处理完成")

            processed_data = self.data_preprocessor.time_alignment(
                processed_data.get('ref_timestamps', None),
                processed_data,
                alignment_mode=preprocess_config.get('alignment_mode', 'linear')
            )
            self.logger.info("时间序列对齐完成")

            if preprocess_config.get('sensor_list', None):
                processed_data = self.data_preprocessor.sensor_fusion(
                    preprocess_config.get('sensor_list', []),
                    processed_data,
                )
                self.logger.info("传感器数据融合完成")

            processed_data = self.data_preprocessor.normalize_data(
                processed_data,
                norm_method='zscore',
                feature_ranges=preprocess_config.get('feature_ranges', None)
            )
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
            # 使用数据预处理器处理数据
            processed_data = self.trajectory_preprocessor.load_data(trajectories)

            processed_data = self.trajectory_preprocessor.format_unification(processed_data)
            self.logger.info("轨迹数据格式统一完成")
            
            # 如果有异常值处理配置
            if preprocess_config and 'min_length' in preprocess_config:
                processed_data = self.trajectory_preprocessor.segment_trajs(
                    processed_data, 
                    min_length=preprocess_config.get('min_length', 10),
                )
                self.logger.info("轨迹分段完成")
            
            # 如果有时间对齐配置
            if preprocess_config and 'reference_clock' in preprocess_config:
                if 'reference_clock' in processed_data:
                    processed_data = self.trajectory_preprocessor.time_align(
                        processed_data,
                        alignment_mode=preprocess_config.get('reference_clock', 'host')
                    )
                    self.logger.info("时间序列对齐完成")

            # 如果有噪声处理
            if preprocess_config and 'filter_type' in preprocess_config:
                if 'filter_type' in processed_data:
                    processed_data = self.trajectory_preprocessor.denoise(
                        processed_data,
                        filter_type=preprocess_config.get('filter_type', 'kalman')
                    )
                    self.logger.info("轨迹去噪完成")
            
            # 如果有数据增强
            if preprocess_config and 'method' in preprocess_config:
                if 'method' in processed_data:
                    processed_data = self.trajectory_preprocessor.data_enhance(
                        processed_data,
                        data_num=preprocess_config.get('data_num', 0),
                        method=preprocess_config.get('method', 'simple')
                    )
                    self.logger.info("轨迹去噪完成")
            
            self.logger.info("轨迹数据预处理完成")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"轨迹数据预处理失败: {str(e)}")
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
        评估模型（增强版）
        支持：
        - AutoencoderTrainer/TransformerTrainer：基于其evaluate接口评估
        - BehaviorCloner：使用其evaluate_policy对轨迹进行评估
        - VAETrajGenerator：计算重构误差与KL散度（无梯度）
        - DiffusionTrajGenerator：按训练同式计算噪声预测MSE（无梯度）
        - 回退：构造数据行为策略，使用OfflineEvaluator进行OPE（IS/DR/CIS）
        """
        self.logger.info("开始模型评估")
        try:
            import json
            import os
            import numpy as np
            import torch
            import torch.nn.functional as F  # 局部导入以避免全局污染
    
            # 统一提取原始样本数据（兼容不同结构）
            raw = None
            if isinstance(test_data, dict):
                raw = test_data.get('data') or test_data.get('trajectories') or test_data
            else:
                raw = test_data
    
            # 评估方法
            eval_method = (eval_config or {}).get('method', 'offline')
    
            # 1) 优先使用训练器的evaluate接口
            try:
                from afruits.utils.AutoencoderModel import AutoencoderTrainer
                from afruits.utils.TransformerModel import TransformerTrainer
                from afruits.utils.BehaviorCloner import BehaviorCloner
                from afruits.utils.VAETrajGenerator import VAETrajGenerator
                from afruits.utils.DiffusionTrajGenerator import DiffusionTrajGenerator
            except Exception:
                # 已在文件顶部静态导入，此处仅兜底
                pass
    
            # 构造加载序列的DataLoader（若支持）
            def _build_dataloader_for_trainer(trainer_obj, raw_data, batch_size=32):
                if hasattr(trainer_obj, 'load_sequences'):
                    data = trainer_obj.load_sequences(raw_data, batch_size=batch_size)
                    return data
                # 兼容VAETrajGenerator/DiffusionTrajGenerator
                if hasattr(trainer_obj, 'load_dataset'):
                    return trainer_obj.load_dataset(raw_data, batch_size=batch_size)
                return None
    
            # AutoencoderTrainer / TransformerTrainer
            if hasattr(model, 'evaluate') and (hasattr(model, 'load_sequences') or hasattr(model, 'load_dataset')):
                dl = _build_dataloader_for_trainer(model, raw, batch_size=32)
                if dl is None:
                    raise ValueError("无法为训练器构建评估数据加载器")
                loss_or_dict = model.evaluate(dl)
                # 标准化输出
                if isinstance(loss_or_dict, dict):
                    result = {'method': 'trainer_evaluate', **loss_or_dict}
                else:
                    result = {'method': 'trainer_evaluate', 'eval_loss': float(loss_or_dict)}
                self.logger.info("训练器评估完成")
                return result
    
            # BehaviorCloner 直接评估轨迹列表
            if hasattr(model, 'evaluate_policy'):
                # 期望 raw 为 List[trajectory], 每个trajectory包含 states/actions
                # 若为dict结构，尝试取其中的列表值
                trajectories = None
                if isinstance(raw, list):
                    trajectories = raw
                elif isinstance(raw, dict):
                    trajectories = list(raw.values())
                else:
                    trajectories = raw
                metrics = model.evaluate_policy(trajectories)
                
                # 增强：融合多样性指标（基于states）
                try:
                    states_list = []
                    if isinstance(trajectories, list):
                        for tr in trajectories:
                            if isinstance(tr, dict) and 'states' in tr:
                                st = np.asarray(tr['states'])
                                if st.ndim >= 2:
                                    states_list.append(st)
                    elif isinstance(trajectories, dict):
                        for tr in trajectories.values():
                            if isinstance(tr, dict) and 'states' in tr:
                                st = np.asarray(tr['states'])
                                if st.ndim >= 2:
                                    states_list.append(st)
                    if states_list:
                        states_all = np.concatenate(states_list, axis=0)
                        div_res = self.multi_metric_evaluator.eval_diversity({'states': states_all})
                        d_m = div_res.get('diversity_metrics', {})
                        # 将多样性指标扁平合并进metrics，便于雷达图/柱状图可视化
                        if isinstance(metrics, dict):
                            if 'inter_cluster_distance' in d_m:
                                metrics['diversity_inter_cluster_distance'] = float(d_m['inter_cluster_distance'])
                            if 'intra_cluster_variance' in d_m:
                                metrics['diversity_intra_cluster_variance'] = float(d_m['intra_cluster_variance'])
                            if 'cluster_count' in d_m:
                                metrics['diversity_cluster_count'] = int(d_m['cluster_count'])
                        # 可选：暴露降维可视化数据，便于embedding可视化（不影响原有逻辑）
                        vis_data = div_res.get('visualization_data', {})
                        if isinstance(metrics, dict) and 'diversity_points' not in metrics and 'points' in vis_data and 'labels' in vis_data:
                            metrics['diversity_points'] = vis_data['points']
                            metrics['diversity_labels'] = vis_data['labels']
                except Exception as _e:
                    self.logger.warning(f"融合多样性指标失败: {_e}")
                
                self.logger.info("BehaviorCloner评估完成")
                return {'method': 'bc_evaluate', **(metrics if isinstance(metrics, dict) else {'metrics': metrics})}
    
            # VAETrajGenerator（无梯度评估：重构误差 + KL散度）
            if isinstance(model, VAETrajGenerator):
                dl = _build_dataloader_for_trainer(model, raw, batch_size=32)
                if dl is None:
                    raise ValueError("无法为VAE评估构建数据加载器")
                model.encoder.eval()
                model.decoder.eval()
    
                total_loss, total_kl, total_recon, n_batches = 0.0, 0.0, 0.0, 0
                with torch.no_grad():
                    for batch in dl:
                        if len(batch) == 2 and getattr(model, 'has_separate_action', False):
                            states = batch[0].to(model.device)
                            actions = batch[1].to(model.device)
                            mu, logvar = model.encoder(states, actions)
                            # 重参数化
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z = mu + eps * std
                            reconstructed_states, reconstructed_actions = model.decoder(z)
    
                            # 状态重构损失
                            state_recon = F.mse_loss(reconstructed_states, states) if model.recon_loss_type == "mse" else F.l1_loss(reconstructed_states, states)
                            # 动作重构损失
                            if getattr(model, 'discrete_action', True):
                                bs, seq_len = actions.shape[0], actions.shape[1]
                                action_recon = F.cross_entropy(reconstructed_actions.reshape(-1, model.action_dim), actions.reshape(-1).long())
                            else:
                                action_recon = F.mse_loss(reconstructed_actions, actions) if model.recon_loss_type == "mse" else F.l1_loss(reconstructed_actions, actions)
                            recon = state_recon + action_recon
                        else:
                            traj = batch[0].to(model.device)
                            mu, logvar = model.encoder(traj)
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z = mu + eps * std
                            reconstructed = model.decoder(z)
                            recon = F.mse_loss(reconstructed, traj) if model.recon_loss_type == "mse" else F.l1_loss(reconstructed, traj)
    
                        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (states.shape[0] if 'states' in locals() else traj.shape[0])
                        loss = recon + model.kl_weight * kl
    
                        total_loss += float(loss.item())
                        total_kl += float(kl.item())
                        total_recon += float(recon.item())
                        n_batches += 1
    
                result = {
                    'method': 'vae_eval',
                    'avg_total_loss': total_loss / max(n_batches, 1),
                    'avg_kl_divergence': total_kl / max(n_batches, 1),
                    'avg_recon_error': total_recon / max(n_batches, 1),
                    'batches': n_batches
                }
                self.logger.info("VAE评估完成")
                return result
    
            # DiffusionTrajGenerator（无梯度评估：噪声预测MSE）
            if isinstance(model, DiffusionTrajGenerator):
                dl = _build_dataloader_for_trainer(model, raw, batch_size=32)
                if dl is None:
                    raise ValueError("无法为Diffusion评估构建数据加载器")
                if model.model is None:
                    # 必须先build_model
                    state_dim = (test_data.get('state_dim') if isinstance(test_data, dict) else None) or getattr(model, 'state_dim', None)
                    if state_dim is None:
                        # 从数据推断
                        first = None
                        if isinstance(raw, list) and raw:
                            first = raw[0]['states'] if isinstance(raw[0], dict) and 'states' in raw[0] else np.asarray(raw[0])
                        if first is not None:
                            state_dim = int(np.asarray(first).shape[-1])
                        else:
                            state_dim = 1
                    model.build_model(state_dim)
    
                model.model.eval()
                mse_sum, n_batches = 0.0, 0
                with torch.no_grad():
                    for batch in dl:
                        trajectories = batch[0].to(model.device)
                        bs = trajectories.shape[0]
                        t = torch.randint(0, model.diffusion_steps, (bs,), device=model.device)
                        noise = torch.randn_like(trajectories)
                        alphas_cumprod_t = torch.tensor(model.alphas_cumprod, device=model.device)[t]
                        if len(trajectories.shape) == 3:
                            alphas_cumprod_t = alphas_cumprod_t.view(-1, 1, 1)
                        else:
                            alphas_cumprod_t = alphas_cumprod_t.view(-1, 1)
                        noisy = torch.sqrt(alphas_cumprod_t) * trajectories + torch.sqrt(1 - alphas_cumprod_t) * noise
                        t_norm = t / model.diffusion_steps
                        pred_noise = model.model(x_state=noisy, t=t_norm)
                        mse = F.mse_loss(pred_noise, noise)
                        mse_sum += float(mse.item())
                        n_batches += 1
    
                result = {
                    'method': 'diffusion_eval',
                    'avg_noise_pred_mse': mse_sum / max(n_batches, 1),
                    'batches': n_batches
                }
                self.logger.info("扩散模型评估完成")
                return result
    
            # 2) multi_metric 原流程
            if eval_method == 'multi_metric':
                metric_config = (eval_config or {}).get('metric_config', {})
                result = self.multi_metric_evaluator.calculate_metrics(
                    metric_config,
                    analysis_mode=(eval_config or {}).get('analysis_mode', 'micro')
                )
                result.update(self.multi_metric_evaluator.fuse_metrics(
                    custom_weights=(eval_config or {}).get('weights', None)
                ))
                self.logger.info("多指标评估完成")
                return result
    
            # 3) 回退：OfflineEvaluator（IS/DR/CIS），构造数据行为策略
            method_type = (eval_config or {}).get('method_type', 'IS')
            action_dim = (test_data.get('action_dim') if isinstance(test_data, dict) else None) or 0
    
            class _DatasetBehaviorPolicy:
                def __init__(self, action_dim, num_steps):
                    self._action_dim = int(action_dim) if action_dim else 1
                    self._num_steps = int(num_steps) if num_steps else 10
                    self.rewards = np.zeros(self._num_steps, dtype=np.float32)
                def get_action_probs(self):
                    # 均匀分布概率矩阵 [num_steps, action_dim]
                    return np.ones((self._num_steps, self._action_dim), dtype=np.float32) / max(self._action_dim, 1)
                def get_rewards(self):
                    return self.rewards
                def estimate_values(self):
                    return np.zeros(self._num_steps, dtype=np.float32)
    
            # 从数据估计步数
            num_steps = 10
            try:
                if isinstance(raw, list) and raw:
                    first = raw[0]
                    if isinstance(first, dict) and 'actions' in first:
                        num_steps = len(first['actions'])
            except Exception:
                pass
    
            behavior_policy = _DatasetBehaviorPolicy(action_dim=action_dim, num_steps=num_steps)
            result = self.offline_evaluator.evaluate_policy(model, behavior_policy, method_type=method_type)
            result['method'] = f'ope_{method_type}'
            self.logger.info("离线OPE评估完成")
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
        加载模型（优先根据 model_class 名称进行明确分派；若未提供，则保留原先的识别式加载逻辑）
        
        兼容以下保存格式：
        - AutoencoderTrainer.save_model: {'state_dict', 'config'}
        - TransformerTrainer.save_model: {'state_dict', 'config'}
        - VAETrajGenerator.save_model: {'encoder_state_dict', 'decoder_state_dict', 'config', ...}
        - DiffusionTrajGenerator.save_model: {'encoder_state_dict', 'config', ...}
        - 纯state_dict: 直接加载到提供的model_class
        """
        self.logger.info(f"开始加载模型: {model_path}")
        try:
            # 1) 优先：根据 model_class 的名称显式调用对应类的 load_model
            if model_class is not None:
                # 解析名称（支持传入类、实例或字符串）
                if isinstance(model_class, str):
                    name = model_class
                elif isinstance(model_class, type):
                    name = model_class.__name__
                else:
                    name = model_class.__class__.__name__
                name = str(name)

                # 明确的名称分支
                if name == 'BehaviorCloner':
                    self.logger.info("根据model_class=BehaviorCloner，使用静态加载器")
                    try:
                        return BehaviorCloner.load_model(model_path, device=self.device)
                    except TypeError:
                        return BehaviorCloner.load_model(model_path)
                    
                elif name == 'AdversarialImitationLearner':
                    self.logger.info("根据model_class=AdversarialImitationLearner，使用静态加载器")
                    return AdversarialImitationLearner.load_model(model_path)
                elif name == 'OfflineRLearner':
                    self.logger.info("根据model_class=OfflineRLearner，使用静态加载器")
                    return OfflineRLearner.load_model(model_path)
                elif name == 'OfflineFSPLearner':
                    self.logger.info("根据model_class=OfflineFSPLearner，使用静态加载器")
                    return OfflineFSPLearner.load_model(model_path)

                elif name == 'AutoencoderModel':
                    self.logger.info("根据model_class=AutoencoderTrainer，使用静态加载器")
                    return AutoencoderTrainer.load_model(model_path)

                elif name == 'TransformerModel':
                    self.logger.info("根据model_class=TransformerTrainer，使用静态加载器")
                    try:
                        return TransformerTrainer.load_model(model_path, device=self.device)
                    except TypeError:
                        return TransformerTrainer.load_model(model_path)

                elif name == 'VAETrajGenerator':
                    self.logger.info("根据model_class=VAETrajGenerator，使用静态加载器")
                    try:
                        return VAETrajGenerator.load_model(model_path, device=self.device)
                    except TypeError:
                        return VAETrajGenerator.load_model(model_path)

                elif name == 'DiffusionTrajGenerator':
                    self.logger.info("根据model_class=DiffusionTrajGenerator，使用静态加载器")
                    return DiffusionTrajGenerator.load_model(model_path)

                # 泛化：若传入类本身实现了 load_model，则直接调用
                if hasattr(model_class, 'load_model'):
                    self.logger.info(f"检测到{name}.load_model，尝试调用")
                    try:
                        return model_class.load_model(model_path, device=self.device)
                    except TypeError:
                        return model_class.load_model(model_path)

                # 兜底：按提供的类实例化并加载 state_dict
                self.logger.info("未匹配到已知类，尝试将state_dict加载入提供的model_class实例")
                checkpoint = torch.load(model_path, map_location=self.device)
                model = model_class(**(model_config or {}))
                state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict)
                return model

            # 2) model_class 为 None：使用原先的检查点识别逻辑
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                # 识别 AutoencoderTrainer
                if 'state_dict' in checkpoint and 'config' in checkpoint:
                    cfg = checkpoint.get('config', {})
                    if 'encoder_type' in cfg or {'latent_dim', 'kl_weight', 'dropout_rate'} <= set(cfg.keys()):
                        self.logger.info("识别到AutoencoderTrainer格式，使用静态加载器")
                        trainer = AutoencoderTrainer.load_model(model_path)
                        return trainer

                    # 识别 TransformerTrainer
                    if {'d_model', 'num_heads', 'num_layers'} <= set(cfg.keys()):
                        self.logger.info("识别到TransformerTrainer格式，使用静态加载器")
                        trainer = TransformerTrainer.load_model(model_path, device=self.device)
                        return trainer

                # 识别 VAETrajGenerator
                if 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
                    self.logger.info("识别到VAETrajGenerator格式，使用静态加载器")
                    try:
                        generator = VAETrajGenerator.load_model(model_path, device=self.device)
                    except TypeError:
                        generator = VAETrajGenerator.load_model(model_path)
                    return generator

                # 识别 DiffusionTrajGenerator
                if 'encoder_state_dict' in checkpoint and 'decoder_state_dict' not in checkpoint:
                    self.logger.info("识别到DiffusionTrajGenerator格式，使用静态加载器")
                    generator = DiffusionTrajGenerator.load_model(model_path)
                    return generator

                # 未识别且未提供model_class
                raise ValueError("未提供model_class且无法识别检查点结构，无法加载模型")

            # 非字典检查点且未提供model_class
            raise ValueError("未提供model_class且检查点为非字典，无法加载模型")

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
    
    def train_trajectory_model(self, training_data: Dict, model_config: Dict) -> Dict:
        """
        训练轨迹模型
        
        参数:
            training_data (Dict): 训练数据
            model_config (Dict): 模型配置
            
        返回:
            Dict: 训练结果，包含模型和训练指标
        """
        self.logger.info("开始训练轨迹模型")
        
        try:
            # 确保model_config中包含training_method
            if 'training_method' not in model_config:
                model_config['training_method'] = 'standard'
                
            # 使用模仿学习服务训练模型
            result = self.imitation_learning_service.train_model(training_data, model_config)
            
            self.logger.info("轨迹模型训练完成")
            return result
            
        except Exception as e:
            self.logger.error(f"轨迹模型训练失败: {str(e)}")
            raise
    
    def train_trajectory_generator(self, training_data: Dict, model_config: Dict) -> Dict:
        """
        训练轨迹生成器
        
        参数:
            training_data (Dict): 训练数据
            model_config (Dict): 模型配置
            
        返回:
            Dict: 训练结果，包含模型和训练指标
        """
        self.logger.info("开始训练轨迹生成器")
        
        try:
            # 确保model_config中包含training_method
            if 'training_method' not in model_config:
                model_config['training_method'] = 'standard'
                
            # 使用模仿学习服务训练模型
            result = self.imitation_learning_service.train_model(training_data, model_config)
            
            self.logger.info("轨迹生成器训练完成")
            return result
            
        except Exception as e:
            self.logger.error(f"轨迹生成器训练失败: {str(e)}")
            raise

    def train_advanced_algorithm(self, training_data: Dict, model_config: Dict) -> Dict:
        """
        进化学习/增量学习/小样本微调
        
        参数:
            training_data (Dict): 训练数据
            model_config (Dict): 模型配置
            
        返回:
            Dict: 训练结果，包含模型和训练指标
        """
        self.logger.info("开始训练")
        
        try:
            # 确保model_config中包含training_method
            if 'training_method' not in model_config:
                model_config['training_method'] = 'standard'
                
            # 使用模仿学习服务训练模型
            result = self.imitation_learning_service.train_model(training_data, model_config)
            
            self.logger.info("轨迹生成器训练完成")
            return result
            
        except Exception as e:
            self.logger.error(f"轨迹生成器训练失败: {str(e)}")
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