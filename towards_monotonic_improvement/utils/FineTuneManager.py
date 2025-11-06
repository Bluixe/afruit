import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import copy

class FineTuneManager:
    """
    微调管理器类
    
    功能定位：实现预训练模型的快速迁移与高效参数优化
    
    核心特性：
    - 分层解冻：支持网络层级参数微调策略
    - 动态正则化：自适应权重衰减与Dropout
    - 早停机制：多指标过拟合监控与干预
    - 异构聚合：适配Transformer/VAE/FF网络模型
    """
    
    def __init__(self, 
                 base_model: Optional[nn.Module] = None,
                 trainable_layers: List[str] = ["*last*"],
                 freeze_strategy: str = "selective",
                 optimizer_config: Dict = {},
                 regularization_mode: str = "adaptive"):
        """
        初始化微调管理器
        
        参数:
            base_model (nn.Module, optional): 预训练模型对象
            trainable_layers (List[str]): 可微调层标识列表，默认为["*last*"]
            freeze_strategy (str): 冻结策略，默认为"selective"
            optimizer_config (Dict): 优化器参数配置
            regularization_mode (str): 正则化模式，默认为"adaptive"
        """
        # 初始化参数
        self.base_model = base_model
        self.trainable_layers = trainable_layers
        self.freeze_strategy = freeze_strategy
        self.optimizer_config = optimizer_config
        self.regularization_mode = regularization_mode
        
        # 初始化模型和优化器
        self.model = None
        self.optimizer = None
        
        # 训练状态跟踪
        self.is_trained = False
        self.training_history = {}
        
        # 验证参数
        self._validate_params()
    
    def _validate_params(self):
        """验证初始化参数是否合法"""
        # 验证trainable_layers
        if not isinstance(self.trainable_layers, list):
            raise ValueError(f"trainable_layers必须为列表，当前值: {self.trainable_layers}")
        
        # 验证freeze_strategy
        valid_freeze_strategies = ["selective", "progressive", "none"]
        if self.freeze_strategy not in valid_freeze_strategies:
            raise ValueError(f"freeze_strategy必须为 {valid_freeze_strategies} 之一，当前值: {self.freeze_strategy}")
        
        # 验证regularization_mode
        valid_regularization_modes = ["adaptive", "fixed", "none"]
        if self.regularization_mode not in valid_regularization_modes:
            raise ValueError(f"regularization_mode必须为 {valid_regularization_modes} 之一，当前值: {self.regularization_mode}")
    
    def setup_model(self, pretrained_weights: Optional[str] = None, model_config: Dict = {}):
        """
        模型加载与配置
        
        参数:
            pretrained_weights (str, optional): 预训练权重路径
            model_config (Dict): 模型配置参数
        
        处理流程:
            加载预训练模型权重点
            根据freeze_strategy冻结指定层
            初始化优化器与分层学习率
        """
        # 检查基础模型是否存在
        if self.base_model is None:
            raise ValueError("基础模型未设置，请先设置base_model")
        
        # 创建模型副本以避免修改原始模型
        self.model = copy.deepcopy(self.base_model)
        
        # 加载预训练权重（如果提供）
        if pretrained_weights:
            try:
                # 尝试加载权重
                state_dict = torch.load(pretrained_weights)
                self.model.load_state_dict(state_dict)
                print(f"成功加载预训练权重: {pretrained_weights}")
            except Exception as e:
                print(f"加载预训练权重失败: {str(e)}")
        
        # 应用冻结策略
        self._apply_freeze_strategy()
        
        # 初始化优化器
        self._setup_optimizer()
        
        return self.model
    
    def _apply_freeze_strategy(self):
        """应用冻结策略，根据trainable_layers和freeze_strategy设置层的可训练状态"""
        # 获取所有命名参数
        named_params = list(self.model.named_parameters())
        
        # 默认冻结所有层
        for name, param in named_params:
            param.requires_grad = False
        
        # 根据不同的冻结策略应用不同的处理
        if self.freeze_strategy == "none":
            # 不冻结任何层
            for name, param in named_params:
                param.requires_grad = True
                
        elif self.freeze_strategy == "selective":
            # 选择性冻结 - 只解冻指定的层
            for layer_pattern in self.trainable_layers:
                if layer_pattern == "*last*":
                    # 特殊情况：解冻最后几层
                    # 假设最后的层通常包含"fc"、"linear"、"output"等关键字
                    for name, param in named_params:
                        if any(keyword in name.lower() for keyword in ["fc", "linear", "output", "classifier"]):
                            param.requires_grad = True
                else:
                    # 解冻匹配模式的层
                    for name, param in named_params:
                        if layer_pattern in name:
                            param.requires_grad = True
                            
        elif self.freeze_strategy == "progressive":
            # 渐进式冻结 - 从底层到顶层逐渐解冻
            # 首先确定层的深度顺序
            layer_depths = {}
            for i, (name, _) in enumerate(named_params):
                # 提取层深度（假设命名格式为layer1.xxx, layer2.xxx等）
                parts = name.split('.')
                if len(parts) > 1:
                    try:
                        depth = int(''.join(filter(str.isdigit, parts[0])))
                        layer_depths[name] = depth
                    except:
                        # 如果无法提取深度，则使用参数索引作为深度
                        layer_depths[name] = i
                else:
                    layer_depths[name] = i
            
            # 计算要解冻的层数
            if "*last*" in self.trainable_layers:
                # 如果指定了*last*，则解冻最后30%的层
                threshold = max(layer_depths.values()) * 0.7
                for name, param in named_params:
                    if layer_depths.get(name, 0) >= threshold:
                        param.requires_grad = True
            else:
                # 否则，解冻指定的层
                for layer_pattern in self.trainable_layers:
                    for name, param in named_params:
                        if layer_pattern in name:
                            param.requires_grad = True
        
        # 打印可训练层信息
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        print(f"可训练层: {trainable_params}")
        print(f"可训练参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def _setup_optimizer(self):
        """设置优化器，支持分层学习率"""
        # 获取优化器配置
        lr = self.optimizer_config.get('lr', 0.001)
        weight_decay = self.optimizer_config.get('weight_decay', 0.0)
        optimizer_type = self.optimizer_config.get('type', 'Adam')
        
        # 检查是否使用分层学习率
        use_layer_lr = self.optimizer_config.get('use_layer_lr', False)
        
        if use_layer_lr:
            # 分层学习率设置
            # 将参数分组，不同层使用不同的学习率
            param_groups = []
            
            # 获取所有命名参数
            named_params = list(self.model.named_parameters())
            
            # 为不同层设置不同的学习率
            for i, (name, param) in enumerate(named_params):
                if not param.requires_grad:
                    continue
                
                # 根据层的深度或名称设置学习率
                # 这里使用一个简单的策略：越靠近输出层的学习率越高
                layer_depth = len(named_params) - i
                layer_lr = lr * (1.0 + 0.1 * layer_depth)
                
                param_groups.append({
                    'params': param,
                    'lr': layer_lr,
                    'weight_decay': weight_decay
                })
        else:
            # 使用统一的学习率
            param_groups = [{'params': [p for p in self.model.parameters() if p.requires_grad],
                            'lr': lr,
                            'weight_decay': weight_decay}]
        
        # 创建优化器
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(param_groups)
        elif optimizer_type == 'SGD':
            momentum = self.optimizer_config.get('momentum', 0.9)
            self.optimizer = optim.SGD(param_groups, momentum=momentum)
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(param_groups)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        return self.optimizer
    
    def prepare_data(self, raw_data, augment: bool = False, batch_size: int = 32, val_split: float = 0.2):
        """
        数据预处理
        
        参数:
            raw_data (Dataset): 原始微调数据集
            augment (bool): 是否启用数据增强
            batch_size (int): 批次大小，默认为32
            val_split (float): 验证集比例，默认为0.2
        
        返回值:
            Tuple[DataLoader, DataLoader]: 训练数据加载器和验证数据加载器
        """
        # 检查输入数据
        if raw_data is None or not isinstance(raw_data, (dict, np.ndarray, torch.Tensor)):
            raise ValueError("raw_data必须是字典、numpy数组或PyTorch张量")
        
        # 处理不同类型的输入数据
        if isinstance(raw_data, dict):
            # 字典类型数据，假设包含'x'和'y'键
            if 'x' not in raw_data or 'y' not in raw_data:
                raise ValueError("字典类型的raw_data必须包含'x'和'y'键")
            
            x_data = raw_data['x']
            y_data = raw_data['y']
            
            # 转换为numpy数组（如果不是）
            if not isinstance(x_data, np.ndarray):
                x_data = np.array(x_data)
            if not isinstance(y_data, np.ndarray):
                y_data = np.array(y_data)
        
        elif isinstance(raw_data, np.ndarray):
            # numpy数组类型数据，假设最后一列是标签
            if raw_data.shape[1] < 2:
                raise ValueError("numpy数组类型的raw_data至少需要2列（特征和标签）")
            
            x_data = raw_data[:, :-1]
            y_data = raw_data[:, -1]
        
        elif isinstance(raw_data, torch.Tensor):
            # PyTorch张量类型数据，假设最后一列是标签
            if raw_data.shape[1] < 2:
                raise ValueError("PyTorch张量类型的raw_data至少需要2列（特征和标签）")
            
            x_data = raw_data[:, :-1].numpy()
            y_data = raw_data[:, -1].numpy()
        
        # 数据增强（如果启用）
        if augment:
            x_data, y_data = self._augment_data(x_data, y_data)
        
        # 转换为PyTorch张量
        x_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # 创建数据集
        dataset = TensorDataset(x_tensor, y_tensor)
        
        # 划分训练集和验证集
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            print(f"数据准备完成: 训练集大小={train_size}, 验证集大小={val_size}")
            return train_loader, val_loader
        else:
            # 如果没有验证集，则返回None作为验证加载器
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print(f"数据准备完成: 训练集大小={len(dataset)}, 无验证集")
            return train_loader, None
    
    def _augment_data(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强函数
        
        参数:
            x_data (np.ndarray): 特征数据
            y_data (np.ndarray): 标签数据
            
        返回值:
            Tuple[np.ndarray, np.ndarray]: 增强后的特征数据和标签数据
        """
        # 检查输入数据
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError("特征数据和标签数据的样本数不匹配")
        
        # 原始数据
        augmented_x = [x_data]
        augmented_y = [y_data]
        
        # 数据增强技术1: 添加高斯噪声
        noise_scale = 0.05  # 噪声幅度
        noisy_x = x_data + np.random.normal(0, noise_scale, x_data.shape)
        augmented_x.append(noisy_x)
        augmented_y.append(y_data)
        
        # 数据增强技术2: 特征缩放
        scale_factor = 1.1  # 缩放因子
        scaled_x = x_data * scale_factor
        augmented_x.append(scaled_x)
        augmented_y.append(y_data)
        
        # 数据增强技术3: 特征移位
        shift_amount = 0.1  # 移位幅度
        shifted_x = x_data + shift_amount
        augmented_x.append(shifted_x)
        augmented_y.append(y_data)
        
        # 合并增强数据
        augmented_x = np.vstack(augmented_x)
        augmented_y = np.concatenate(augmented_y)
        
        # 打乱数据
        indices = np.arange(augmented_x.shape[0])
        np.random.shuffle(indices)
        augmented_x = augmented_x[indices]
        augmented_y = augmented_y[indices]
        
        print(f"数据增强完成: 原始样本数={x_data.shape[0]}, 增强后样本数={augmented_x.shape[0]}")
        return augmented_x, augmented_y
    
    def execute_finetuning(self, train_loader, val_loader=None, max_epochs: int = 10,
                          patience: int = 5, criterion=None, device: str = None):
        """
        微调训练
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器
            max_epochs (int): 最大训练轮次
            patience (int): 早停耐心值，默认为5
            criterion: 损失函数，默认为None（使用MSELoss）
            device (str): 设备，默认为None（自动选择）
        
        返回值:
            Dict: 训练报告，包含以下指标:
                - best_val_loss: 最佳验证损失
                - training_time: 总时间(分钟)
                - convergence_epoch: 收敛轮次
        """
        # 检查模型和优化器是否已设置
        if self.model is None:
            raise ValueError("模型未设置，请先调用setup_model方法")
        if self.optimizer is None:
            raise ValueError("优化器未设置，请先调用setup_model方法")
        
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 设置损失函数
        if criterion is None:
            criterion = nn.MSELoss()
        
        # 初始化训练状态
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # 记录开始时间
        start_time = time.time()
        convergence_epoch = max_epochs
        
        # 训练循环
        for epoch in range(max_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                # 将数据移动到设备
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # 前向传播
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            training_history['train_loss'].append(train_loss)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            training_history['learning_rates'].append(current_lr)
            
            # 验证阶段（如果有验证集）
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        # 将数据移动到设备
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        
                        # 前向传播
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                
                # 计算平均验证损失
                val_loss /= len(val_loader)
                training_history['val_loss'].append(val_loss)
                
                # 检查是否是最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # 早停检查
                if epochs_no_improve >= patience:
                    print(f"早停触发！{patience}轮未改善")
                    convergence_epoch = epoch + 1 - patience
                    break
            
            # 打印训练进度
            if val_loader is not None:
                print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
            else:
                print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.6f}, LR: {current_lr:.6f}")
            
            # 动态调整正则化（如果启用）
            if self.regularization_mode == "adaptive":
                self._adjust_regularization(epoch, train_loss, val_loss if val_loader else None)
        
        # 计算训练时间
        training_time = (time.time() - start_time) / 60.0  # 转换为分钟
        
        # 恢复最佳模型（如果有验证集）
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 标记模型已训练
        self.is_trained = True
        
        # 保存训练历史
        self.training_history = training_history
        
        # 生成训练报告
        training_report = {
            'best_val_loss': best_val_loss if val_loader else None,
            'final_train_loss': train_loss,
            'training_time': training_time,
            'convergence_epoch': convergence_epoch,
            'early_stopped': epochs_no_improve >= patience if val_loader else False
        }
        
        return training_report
    
    def _adjust_regularization(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """
        动态调整正则化参数
        
        参数:
            epoch (int): 当前训练轮次
            train_loss (float): 当前训练损失
            val_loss (float, optional): 当前验证损失
        """
        # 检查是否有足够的历史数据
        if len(self.training_history.get('train_loss', [])) < 2:
            return
        
        # 计算训练损失变化
        prev_train_loss = self.training_history['train_loss'][-2]
        train_loss_change = train_loss - prev_train_loss
        
        # 检查是否有验证损失
        if val_loss is not None and len(self.training_history.get('val_loss', [])) >= 2:
            prev_val_loss = self.training_history['val_loss'][-2]
            val_loss_change = val_loss - prev_val_loss
            
            # 检测过拟合迹象：训练损失下降但验证损失上升
            if train_loss_change < 0 and val_loss_change > 0:
                # 增加权重衰减以减轻过拟合
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] = min(param_group['weight_decay'] * 1.2, 0.1)
                print(f"检测到过拟合迹象，增加权重衰减至 {self.optimizer.param_groups[0]['weight_decay']:.6f}")
        
        # 检测欠拟合迹象：训练损失下降缓慢
        if train_loss_change > -0.001 and epoch > 5:
            # 减少权重衰减以增强拟合能力
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = max(param_group['weight_decay'] * 0.8, 1e-6)
            print(f"检测到欠拟合迹象，减少权重衰减至 {self.optimizer.param_groups[0]['weight_decay']:.6f}")
    
    def evaluate_model(self, test_loader, metrics: List = [], device: str = None):
        """
        模型评估
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            metrics (List): 评估指标列表，可以包含 'mse', 'mae', 'rmse', 'r2', 'accuracy'
            device (str): 设备，默认为None（自动选择）
        
        返回值:
            Dict: 指标字典
        """
        # 检查模型是否已训练
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用execute_finetuning方法")
        
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 初始化结果
        all_predictions = []
        all_targets = []
        
        # 收集预测结果和目标值
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # 将数据移动到设备
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # 前向传播
                outputs = self.model(batch_x)
                
                # 收集结果
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # 合并批次结果
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 计算评估指标
        metrics_dict = {}
        
        # 如果未指定指标，则使用默认指标
        if not metrics:
            metrics = ['mse', 'mae', 'rmse']
        
        # 计算各种指标
        for metric in metrics:
            if metric.lower() == 'mse':
                # 均方误差
                mse = np.mean((all_predictions - all_targets) ** 2)
                metrics_dict['mse'] = mse
            
            elif metric.lower() == 'mae':
                # 平均绝对误差
                mae = np.mean(np.abs(all_predictions - all_targets))
                metrics_dict['mae'] = mae
            
            elif metric.lower() == 'rmse':
                # 均方根误差
                rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
                metrics_dict['rmse'] = rmse
            
            elif metric.lower() == 'r2':
                # R方
                from sklearn.metrics import r2_score
                r2 = r2_score(all_targets, all_predictions)
                metrics_dict['r2'] = r2
            
            elif metric.lower() == 'accuracy':
                # 准确率（对于分类任务）
                # 将预测值转换为类别
                pred_classes = np.argmax(all_predictions, axis=1)
                target_classes = np.argmax(all_targets, axis=1)
                accuracy = np.mean(pred_classes == target_classes)
                metrics_dict['accuracy'] = accuracy
        
        # 打印评估结果
        print("模型评估结果:")
        for metric_name, metric_value in metrics_dict.items():
            print(f"  {metric_name}: {metric_value:.6f}")
        
        return metrics_dict