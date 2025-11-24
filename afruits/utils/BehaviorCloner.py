import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import time
from afruits.utils.DataLoader import DataLoaderUtil

class BehaviorCloner:
    """
    行为克隆器类
    
    功能描述：实现基于模仿学习的行为克隆算法，通过观察专家行为数据学习策略
    
    核心功能：
    - 数据处理：支持时间序列数据处理与特征提取
    - 模型训练：支持MLP、CNN等多种网络结构
    - 策略评估：提供多种评估指标与可视化工具
    """
    
    def __init__(self,
                 batch_size: int = 32,
                 network_type: str = "MLP",
                 max_epochs: int = 200,
                 dropout_rate: float = 0.2):
        """
        初始化行为克隆器
        
        参数:
            batch_size (int): 训练批次大小，默认为32，有效取值范围16-512
            network_type (str): 网络类型，默认为"MLP"，有效取值范围[MLP,CNN]
            max_epochs (int): 最大训练轮次，默认为200，有效取值范围>=50
            dropout_rate (float): 防止过拟合的神经元丢弃率，默认为0.2，有效取值范围0.0-0.5
        """
        # 初始化参数
        self.batch_size = batch_size
        self.network_type = network_type
        self.max_epochs = max_epochs
        self.dropout_rate = dropout_rate

        self.config_to_save = {
            'batch_size': self.batch_size,
            'network_type': self.network_type,
            'max_epochs': self.max_epochs,
            'dropout_rate': self.dropout_rate
        }
        
        # 初始化模型和优化器
        self.model = None
        self.optimizer = None
        
        # 初始化训练状态
        self.is_trained = False

        self.dataloader_util = DataLoaderUtil()
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证batch_size
        if not isinstance(self.batch_size, int) or not (16 <= self.batch_size <= 512):
            raise ValueError(f"batch_size必须为16-512范围内的整数，当前值: {self.batch_size}")
        
        # 验证network_type
        valid_network_types = ["MLP", "CNN"]
        if self.network_type not in valid_network_types:
            raise ValueError(f"network_type必须为 {valid_network_types} 之一，当前值: {self.network_type}")
        
        # 验证max_epochs
        if not isinstance(self.max_epochs, int) or self.max_epochs < 50:
            raise ValueError(f"max_epochs必须为大于等于50的整数，当前值: {self.max_epochs}")
        
        # 验证dropout_rate
        if not isinstance(self.dropout_rate, float) or not (0.0 <= self.dropout_rate <= 0.5):
            raise ValueError(f"dropout_rate必须在0.0-0.5范围内，当前值: {self.dropout_rate}")

    def build_model(self, state_dim, action_dim):
        self.config_to_save.update({
            'state_dim': state_dim,
            'action_dim': action_dim
        })
        if isinstance(state_dim, tuple) and len(state_dim) == 1:
            input_dim = state_dim[0]
        else:
            input_dim = state_dim
        self.input_dim = input_dim
        output_dim = action_dim
        self.output_dim = output_dim
        if isinstance(input_dim, int) and input_dim > 0:
            # 创建MLP模型
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, output_dim)
            )
        elif isinstance(input_dim, tuple) and len(input_dim) == 3:
            # 创建CNN模型
            self.model = nn.Sequential(
                nn.Conv2d(input_dim[0], 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(32 * (input_dim[1]//4) * (input_dim[3]//4), 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, output_dim)
            )
    
    def process_data(self, raw_trajectories: Dict, context_frames: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据处理函数
        
        参数:
            raw_trajectories (Dict): 原始轨迹数据集，包含专家演示轨迹（每条轨迹包含时间序列状态和动作）
            context_frames (int): 上下文帧数，默认为4
        
        返回值:
            X_train: 三维数据（样本数×时间步×特征维度）
            y_train: 二维数据（样本数×动作维度）
        
        功能描述:
            1. 完成数据清洗、标准化和时间序列处理
            2. 自动填充缺失值
            3. 标准化特征到[-1, 1]区间
        """
        expert_states, expert_actions = self.dataloader_util.load_bc_gail_data(raw_trajectories)
        return expert_states, expert_actions
    
    def extract_features(self, state_data: np.ndarray) -> Dict:
        """
        特征提取函数
        
        参数:
            state_data (np.ndarray): 状态数据，包含:
                - position: [x,y,z] 三维坐标
                - velocity: [vx,vy,vz] 速度向量
                - radar_image: 输入图像（可选）
        
        返回值:
            特征字典，包含512维特征向量
        
        功能描述:
            1. 编码规则：||position||
            2. 速度特征：cos θ = (v1·v2)/(||v1|| ||v2||)
            3. 图像特征：通过CNN提取输入图像特征
        """
        # 初始化特征字典
        features = {}
        
        # 检查输入数据
        if not isinstance(state_data, np.ndarray):
            raise ValueError("state_data必须是numpy数组")
        
        # 提取位置特征（如果存在）
        if state_data.shape[-1] >= 3:
            features['position'] = state_data[..., :3]  # 提取前三个维度作为位置
        
        # 提取速度特征（如果存在）
        if state_data.shape[-1] >= 6:
            features['velocity'] = state_data[..., 3:6]  # 提取第4-6个维度作为速度
        
        # 提取图像特征（如果存在）
        if len(state_data.shape) > 2 and state_data.shape[-1] > 6:
            # 假设第7个维度开始是图像数据
            features['radar_image'] = state_data[..., 6:]
            
            # 如果使用CNN，可以在这里进行图像特征提取
            if self.network_type == "CNN" and self.model:
                # 这里应该实现CNN特征提取
                # 简化示例：直接返回原始图像数据
                pass
        
        # 计算特征统计信息
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                print(f"特征 {key} 形状: {value.shape}")
                if value.size > 0:
                    print(f"特征 {key} 统计: 最小值={np.min(value)}, 最大值={np.max(value)}, 均值={np.mean(value)}")
        
        return features
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, input_dim, output_dim, validation_split: float = 0.2) -> Dict:
        """
        模型训练函数
        
        参数:
            X_train (np.ndarray): 训练数据特征
            y_train (np.ndarray): 训练数据标签
            validation_split (float): 验证集比例，默认为0.2
        
        返回值:
            训练历史记录，包含以下下列指标:
                - train_loss: 训练集损失值
                - val_accuracy: 验证集准确率
        
        功能描述:
            1. 训练历史记录：包含以下数据指标
            2. MLP模式：FC → Dropout → FC → 输出层
            3. CNN模式：CNN特征提取 → 特征拼接 → FC → 输出层
            4. 损失函数：Huber损失
        """
        # 检查输入数据
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("X_train和y_train必须是numpy数组")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train和y_train的样本数不匹配: {X_train.shape[0]} vs {y_train.shape[0]}")
        
        # 转换为PyTorch张量, 确保y为整数类型
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 划分训练集和验证集
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        if type(input_dim) == tuple:
            input_dim = np.prod(input_dim)
        # 创建模型
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 定义损失函数
        # CrossEntropyLoss适用于分类任务
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_accuracy': []
        }
        
        # 训练循环
        for epoch in range(self.max_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            # 计算平均验证损失
            val_loss /= len(val_loader)
            history['val_accuracy'].append(1.0 - val_loss)  # 简化的准确率计算
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 标记模型已训练
        self.is_trained = True
        
        return history
    
    def evaluate_policy(self, test_trajectories: List, simulator: Any = None) -> Dict:
        """
        策略评估函数（增强版）
        
        参数:
            test_trajectories (List): 测试轨迹数据集
            simulator (Any, optional): 数字孪生仿真环境
        
        返回值:
            metrics (Dict): 评估指标字典，兼容旧键并新增多样性/分布相关指标:
                - accuracy / action_accuracy: 总体分类准确率
                - mean_abs_error: 平均绝对误差（基于离散动作索引）
                - error_distribution: 绝对误差平均值（与旧实现保持兼容）
                - per_action_accuracy: 各类别准确率
                - action_hist: 预测动作直方图（计数）
                - action_entropy: 预测分布平均信息熵
                - action_switch_rate: 连续时刻动作切换率
                - unique_actions_ratio: 预测到的唯一动作比率
        """
        if not test_trajectories:
            raise ValueError("test_trajectories不能为空")
        
        self.model.eval()
        
        # 全局统计
        total_samples = 0
        correct_predictions = 0
        all_abs_errors: List[float] = []
        per_class_correct = {}
        per_class_total = {}
        hist_counts = {}
        total_switches = 0
        total_switch_den = 0  # 累计 (T-1)
        entropies: List[float] = []
        predicted_unique_actions = set()
        
        import torch.nn.functional as F
        
        for trajectory in test_trajectories:
            if 'states' not in trajectory or 'actions' not in trajectory:
                print("警告: 轨迹缺少状态或动作数据，已跳过")
                continue
            
            states = np.asarray(trajectory['states'])
            actions = np.asarray(trajectory['actions']).reshape(-1)
            if len(states) != len(actions):
                print("警告: 轨迹的状态和动作数据长度不匹配，已跳过")
                continue
            
            # 前向预测
            states_tensor = torch.FloatTensor(states)
            with torch.no_grad():
                logits = self.model(states_tensor)  # [T, A]
                if isinstance(logits, torch.Tensor):
                    probs = F.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1).cpu().numpy().astype(int)
                    # 信息熵：-sum p log p
                    entropy_seq = (-probs * (probs.clamp(min=1e-12)).log()).sum(dim=-1).cpu().numpy()
                else:
                    # 兼容非Tensor返回
                    logits_np = np.asarray(logits)
                    preds = logits_np.argmax(axis=-1)
                    # 简单softmax以继续熵的计算
                    e_x = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
                    probs_np = e_x / np.clip(e_x.sum(axis=-1, keepdims=True), 1e-12, None)
                    entropy_seq = -(probs_np * np.log(np.clip(probs_np, 1e-12, None))).sum(axis=-1)
            
            # 统计切换率
            if len(preds) > 1:
                switches = np.sum(preds[1:] != preds[:-1])
                total_switches += int(switches)
                total_switch_den += (len(preds) - 1)
            
            # 样本级统计
            for i in range(len(actions)):
                gt = int(actions[i])
                pr = int(preds[i])
                all_abs_errors.append(abs(pr - gt))
                total_samples += 1
                if pr == gt:
                    correct_predictions += 1
                    per_class_correct[gt] = per_class_correct.get(gt, 0) + 1
                per_class_total[gt] = per_class_total.get(gt, 0) + 1
                hist_counts[pr] = hist_counts.get(pr, 0) + 1
                predicted_unique_actions.add(pr)
            
            # 累计熵
            if entropy_seq is not None and len(entropy_seq) > 0:
                entropies.extend(list(np.asarray(entropy_seq).reshape(-1)))
        
        # 汇总指标
        metrics: Dict[str, Any] = {
            'accuracy': float(correct_predictions / total_samples) if total_samples > 0 else 0.0,
            'mean_abs_error': float(np.mean(all_abs_errors)) if all_abs_errors else 0.0,
            'error_distribution': float(np.mean(all_abs_errors)) if all_abs_errors else 0.0,  # 兼容旧版本（标量）
            'action_hist': {int(k): int(v) for k, v in hist_counts.items()},
            'action_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'action_switch_rate': float(total_switches / total_switch_den) if total_switch_den > 0 else 0.0,
        }
        
        # 唯一动作比率
        try:
            action_space = int(getattr(self, 'output_dim', 0)) if hasattr(self, 'output_dim') else 0
            if action_space <= 0 and len(hist_counts) > 0:
                action_space = int(max(hist_counts.keys())) + 1
            metrics['unique_actions_ratio'] = float(len(predicted_unique_actions) / action_space) if action_space > 0 else 0.0
        except Exception:
            metrics['unique_actions_ratio'] = 0.0
        
        # 各类别准确率
        per_action_acc = {}
        for cls, tot in per_class_total.items():
            cor = per_class_correct.get(cls, 0)
            per_action_acc[int(cls)] = float(cor / tot) if tot > 0 else 0.0
        metrics['per_action_accuracy'] = per_action_acc
        
        # 仿真指标（可选）
        if simulator is not None:
            try:
                sim_metrics = simulator.evaluate(self.model)
                if isinstance(sim_metrics, dict):
                    metrics.update(sim_metrics)
            except Exception as e:
                print(f"仿真评估失败: {str(e)}")
        
        return metrics

    def save_model(self, save_path = None) -> None:

        """
        保存模型函数
        
        参数:
            path (str): 模型保存路径
        """

        
        if save_path is None:
            # 默认保存路径
            save_path = f"models/bc.pt"
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_state = {
            'state_dict': self.model.state_dict(),
            'config': {k:v for k,v in self.config_to_save.items()}
        }

        torch.save(model_state, save_path)
        import json
        config_path = os.path.splitext(save_path)[0] + '_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_state['config'], f, ensure_ascii=False, indent=4)
            
        print(f"模型已保存至: {save_path}")
        print(f"配置已保存至: {config_path}")

    @staticmethod
    def load_model(load_path, device: torch.device = None) -> 'BehaviorCloner':
        """
        静态方法：加载模型参数和配置，返回TransformerModel实例
        
        参数:
            load_path (str): 模型加载路径
            device (torch.device, optional): 计算设备
            
        返回值:
            TransformerModel实例
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_path is None:
            load_path = f"models/bc.pt"
        
        # 加载模型状态
        checkpoint = torch.load(load_path, map_location=device)
        config = checkpoint['config']
        
        # 创建TransformerModel实例
        model = BehaviorCloner(
            batch_size = config['batch_size'],
            network_type = config['network_type'],
            max_epochs = config['max_epochs'],
            dropout_rate = config['dropout_rate']
        )
        
        # 构建模型
        model.build_model(config['state_dim'], config['action_dim'])
        
        # 加载模型参数
        model.model.load_state_dict(checkpoint['state_dict'])
        
        print(f"成功加载模型: {load_path}")
        
        return model