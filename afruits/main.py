
import os
import sys
import numpy as np
import torch
import logging
import datetime
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, 
                            QTextEdit, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, 
                            QCheckBox, QTableWidget, QTableWidgetItem, QSplitter, 
                            QMessageBox, QProgressBar, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入API
from afruits.core.api import AlgorithmAPI

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('afruits_app.log')
    ]
)

logger = logging.getLogger('afruits_app')

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib画布类，用于在Qt界面中嵌入matplotlib图表"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)

class TrainingThread(QThread):
    """训练线程类，用于在后台执行模型训练"""
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    training_finished = pyqtSignal(dict)
    
    def __init__(self, api, training_data, model_config):
        super().__init__()
        self.api = api
        self.training_data = training_data
        self.model_config = model_config
        
    def run(self):
        try:
            self.update_status.emit("开始训练模型...")
            self.update_progress.emit(10)
            
            # 根据模型类型选择不同的训练方法
            model_type = self.model_config.get('model_type', '')
            
            if model_type in ['BehaviorCloner', 'AdversarialImitationLearner', 'OfflineRLearner', 'OfflineFSPLearner']:
                # 博弈建模模型
                result = self.api.train_game_model(self.training_data, self.model_config)
                self.update_progress.emit(80)
                self.update_status.emit(f"模型 {model_type} 训练完成")
            elif model_type in ['AutoencoderModel', 'TransformerModel']:
                # 轨迹建模模型
                result = self.api.train_trajectory_model(self.training_data, self.model_config)
                self.update_progress.emit(80)
                self.update_status.emit(f"模型 {model_type} 训练完成")
            elif model_type in ['DiffusionTrajGenerator', 'VAETrajGenerator']:
                # 轨迹生成模型
                result = self.api.train_trajectory_generator(self.training_data, self.model_config)
                self.update_progress.emit(80)
                self.update_status.emit(f"模型 {model_type} 训练完成")
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            self.update_progress.emit(100)
            self.training_finished.emit(result)
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.update_status.emit(f"训练失败: {str(e)}")
            logger.error(f"训练失败: {str(e)}\n{error_traceback}")

class MainWindow(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        
        # 初始化API
        self.api = AlgorithmAPI(log_level="INFO")
        
        # 初始化UI
        self.init_ui()

        self.keys = ["AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator"]
        
        # 初始化数据
        self.processed_data = None
        self.training_data = None
        self.test_data = None
        self.current_model = None
        self.current_model_id = None
        self.training_result = None  # 用于存储训练结果，供可视化使用
        self.available_models = self.api.get_available_models()
        
        # 更新模型下拉框
        self.update_model_combobox()
        
        logger.info("应用程序初始化完成")
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("小样本博弈建模与专家轨迹模仿学习软件")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡部件
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # 创建各个选项卡
        self.create_data_tab()
        self.create_training_tab()
        self.create_evaluation_tab()
        self.create_visualization_tab()
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
        
        # 显示窗口
        self.show()

    def create_trajectory_data(self, model_type):

        if model_type in ["AutoencoderModel", "TransformerModel", "DiffusionTrajGenerator", "VAETrajGenerator"]:
            trajectories = []
            for i in range(200):
                # 每条轨迹包含40个时间步
                states = np.random.rand(80, 10)  # 10维状态空间
                # 离散动作空间，动作维度为5
                actions = np.random.randint(0, 5, size=(80,))  # 离散动作空间
                
                trajectories.append({
                    'states': states,
                    'actions': actions,
                })
            
            return {"data": trajectories,
                    "state_dim": (10,),
                    "action_dim": 5,
                    "traj_length": 80}
        elif model_type in ["OfflineRLearner", "OfflineFSPLearner", "BehaviorCloner", "AdversarialImitationLearner"]:
            trajectories = {}
        
            # 创建10条轨迹
            for i in range(10):
                # 每条轨迹包含20个时间步
                states = np.random.rand(20, 10)  # 10维状态空间
                actions = np.random.randint(0, 5, size=(20,))  # 离散动作空间
                opponent_actions = np.random.randint(0, 5, size=(20,))  # 对手离散动作
                next_states = np.random.rand(20, 10)  # 下一个状态

                if model_type == 'OfflineRLearner':
                    # 离线强化学习需要单独的轨迹格式
                    trajectories[f'traj_{i}'] = {
                        'states': states,
                        'actions': actions,
                        'rewards': np.random.rand(20),  # 随机奖励
                        'next_states': next_states,
                        'dones': np.zeros(20),  # 完成标志
                        'infos': [{} for _ in range(20)]  # 额外信息
                    }
                elif model_type == 'OfflineFSPLearner':
                    # 离线自对弈需要单独的轨迹格式
                    trajectories[f'traj_{i}'] = {
                        'states': states,
                        'actions': actions,
                        'opponent_actions': opponent_actions,
                        'next_states': next_states,
                        'rewards': np.random.rand(20),  # 随机奖励
                        'dones': np.zeros(20),  # 完成标志
                        'infos': [{} for _ in range(20)]  # 额外信息
                    }
                else:
                    trajectories[f'traj_{i}'] = {
                        'states': states,
                        'actions': actions,
                        'rewards': np.random.rand(20),  # 随机奖励
                        'dones': np.zeros(20),  # 完成标志
                        'infos': [{} for _ in range(20)]  # 额外信息
                    }
            return {"data": trajectories,
                    "state_dim": (10,),
                    "action_dim": 5}
    
        
    def create_data_tab(self):
        """创建数据选项卡"""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # 数据加载部分
        data_group = QGroupBox("数据加载")
        data_layout = QVBoxLayout()
        
        # 训练数据加载
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("训练数据:"))
        self.train_data_path = QLineEdit()
        train_layout.addWidget(self.train_data_path)
        train_btn = QPushButton("浏览...")
        train_btn.clicked.connect(lambda: self.load_data('train'))
        train_layout.addWidget(train_btn)
        data_layout.addLayout(train_layout)
        
        # 测试数据加载
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("测试数据:"))
        self.test_data_path = QLineEdit()
        test_layout.addWidget(self.test_data_path)
        test_btn = QPushButton("浏览...")
        test_btn.clicked.connect(lambda: self.load_data('test'))
        test_layout.addWidget(test_btn)
        data_layout.addLayout(test_layout)
        
        # 数据格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("数据格式:"))
        self.data_format_combo = QComboBox()
        self.data_format_combo.addItems(["json", "csv", "npy"])
        format_layout.addWidget(self.data_format_combo)
        data_layout.addLayout(format_layout)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 数据预处理部分
        preprocess_group = QGroupBox("数据预处理")
        preprocess_layout = QFormLayout()
        
        # 异常值处理
        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setRange(0.1, 10.0)
        self.outlier_threshold.setValue(3.0)
        preprocess_layout.addRow("异常值阈值:", self.outlier_threshold)
        
        # 对齐模式
        self.alignment_mode = QComboBox()
        self.alignment_mode.addItems(["linear", "nearest", "cubic"])
        preprocess_layout.addRow("对齐模式:", self.alignment_mode)
        
        # 标准化选项
        self.normalize_check = QCheckBox("启用标准化")
        self.normalize_check.setChecked(True)
        preprocess_layout.addRow("", self.normalize_check)
        
        # 预处理按钮
        preprocess_btn = QPushButton("执行预处理")
        preprocess_btn.clicked.connect(self.preprocess_data)
        preprocess_layout.addRow("", preprocess_btn)
        
        # 预处理状态
        self.preprocess_status = QLabel("未处理")
        preprocess_layout.addRow("状态:", self.preprocess_status)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # 数据信息显示
        info_group = QGroupBox("数据信息")
        info_layout = QVBoxLayout()
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        info_layout.addWidget(self.data_info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        self.tabs.addTab(data_tab, "数据管理")
        
    def create_training_tab(self):
        """创建训练选项卡"""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # 模型选择部分
        model_group = QGroupBox("模型选择")
        model_layout = QFormLayout()
        
        # 模型类型选择
        self.model_category_combo = QComboBox()
        self.model_category_combo.addItems(["基础算法模型", "轨迹建模与生成模型", "训练方法模型"])
        self.model_category_combo.currentIndexChanged.connect(self.update_model_combobox)
        model_layout.addRow("模型类别:", self.model_category_combo)
        
        # 具体模型选择
        self.model_type_combo = QComboBox()
        model_layout.addRow("模型类型:", self.model_type_combo)
        
        # 训练方法选择
        self.training_method_combo = QComboBox()
        self.training_method_combo.addItems(["standard", "evolutionary", "incremental", "fine_tune"])
        model_layout.addRow("训练方法:", self.training_method_combo)
        
        # 添加数据加载按钮
        data_load_layout = QHBoxLayout()
        self.train_data_path_label = QLineEdit()
        self.train_data_path_label.setReadOnly(True)
        self.train_data_path_label.setPlaceholderText("未选择训练数据")
        data_load_layout.addWidget(self.train_data_path_label)
        
        load_data_btn = QPushButton("加载数据")
        load_data_btn.clicked.connect(self.load_training_data)
        data_load_layout.addWidget(load_data_btn)
        
        model_layout.addRow("训练数据:", data_load_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 模型参数部分
        params_group = QGroupBox("模型参数")
        params_layout = QVBoxLayout()
        
        # 通用参数
        common_form = QFormLayout()
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(32)
        common_form.addRow("批次大小:", self.batch_size)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        common_form.addRow("训练轮数:", self.epochs)
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 0.1)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setDecimals(5)
        self.learning_rate.setSingleStep(0.0001)
        common_form.addRow("学习率:", self.learning_rate)
        
        self.validation_split = QDoubleSpinBox()
        self.validation_split.setRange(0.0, 0.5)
        self.validation_split.setValue(0.2)
        self.validation_split.setSingleStep(0.05)
        common_form.addRow("验证集比例:", self.validation_split)
        
        params_layout.addLayout(common_form)
        
        # 高级参数（可展开）
        self.advanced_params_text = QTextEdit()
        self.advanced_params_text.setPlaceholderText("在此输入高级参数，格式为JSON，例如：\n{\n  \"hidden_dim\": 64,\n  \"dropout_rate\": 0.2\n}")
        params_layout.addWidget(QLabel("高级参数:"))
        params_layout.addWidget(self.advanced_params_text)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 训练控制部分
        train_control_group = QGroupBox("训练控制")
        train_control_layout = QVBoxLayout()
        
        # 进度条
        self.train_progress = QProgressBar()
        train_control_layout.addWidget(self.train_progress)
        
        # 状态标签
        self.train_status = QLabel("就绪")
        train_control_layout.addWidget(self.train_status)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.stop_btn)
        
        train_control_layout.addLayout(btn_layout)
        
        train_control_group.setLayout(train_control_layout)
        layout.addWidget(train_control_group)
        
        self.tabs.addTab(training_tab, "模型训练")
        
    def create_evaluation_tab(self):
        """创建评估选项卡"""
        evaluation_tab = QWidget()
        layout = QVBoxLayout(evaluation_tab)
        
        # 模型选择部分
        model_group = QGroupBox("模型选择")
        model_layout = QHBoxLayout()
        
        model_layout.addWidget(QLabel("选择模型:"))
        self.eval_model_combo = QComboBox()
        model_layout.addWidget(self.eval_model_combo)
        
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 评估配置部分
        eval_config_group = QGroupBox("评估配置")
        eval_config_layout = QFormLayout()
        
        self.eval_method_combo = QComboBox()
        self.eval_method_combo.addItems(["offline", "multi_metric"])
        eval_config_layout.addRow("评估方法:", self.eval_method_combo)
        
        self.eval_metric_combo = QComboBox()
        self.eval_metric_combo.addItems(["IS", "WIS", "FQE", "MB"])
        eval_config_layout.addRow("评估指标:", self.eval_metric_combo)
        
        eval_btn = QPushButton("开始评估")
        eval_btn.clicked.connect(self.evaluate_model)
        eval_config_layout.addRow("", eval_btn)
        
        eval_config_group.setLayout(eval_config_layout)
        layout.addWidget(eval_config_group)
        
        # 评估结果部分
        results_group = QGroupBox("评估结果")
        results_layout = QVBoxLayout()
        
        self.eval_results_text = QTextEdit()
        self.eval_results_text.setReadOnly(True)
        results_layout.addWidget(self.eval_results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.tabs.addTab(evaluation_tab, "模型评估")
        
    def create_visualization_tab(self):
        """创建可视化选项卡"""
        visualization_tab = QWidget()
        layout = QVBoxLayout(visualization_tab)
        
        # 可视化类型选择
        vis_type_group = QGroupBox("可视化类型")
        vis_type_layout = QHBoxLayout()
        
        vis_type_layout.addWidget(QLabel("选择类型:"))
        self.vis_type_combo = QComboBox()
        self.vis_type_combo.addItems(["line", "bar", "scatter", "heatmap", "3d", 
                                     "trajectory", "distribution", "comparison", "embedding"])
        vis_type_layout.addWidget(self.vis_type_combo)
        
        vis_type_group.setLayout(vis_type_layout)
        layout.addWidget(vis_type_group)
        
        # 可视化配置
        vis_config_group = QGroupBox("可视化配置")
        vis_config_layout = QFormLayout()
        
        self.vis_title = QLineEdit("可视化图表")
        vis_config_layout.addRow("标题:", self.vis_title)
        
        self.vis_xlabel = QLineEdit("X轴")
        vis_config_layout.addRow("X轴标签:", self.vis_xlabel)
        
        self.vis_ylabel = QLineEdit("Y轴")
        vis_config_layout.addRow("Y轴标签:", self.vis_ylabel)
        
        self.vis_grid = QCheckBox()
        self.vis_grid.setChecked(True)
        vis_config_layout.addRow("显示网格:", self.vis_grid)
        
        vis_config_group.setLayout(vis_config_layout)
        layout.addWidget(vis_config_group)
        
        # 可视化画布
        canvas_group = QGroupBox("可视化结果")
        canvas_layout = QVBoxLayout()
        
        self.vis_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        canvas_layout.addWidget(self.vis_canvas)
        
        # 可视化按钮
        vis_btn_layout = QHBoxLayout()
        
        vis_btn = QPushButton("生成可视化")
        vis_btn.clicked.connect(self.generate_visualization)
        vis_btn_layout.addWidget(vis_btn)
        
        save_vis_btn = QPushButton("保存图表")
        save_vis_btn.clicked.connect(self.save_visualization)
        vis_btn_layout.addWidget(save_vis_btn)
        
        canvas_layout.addLayout(vis_btn_layout)
        
        canvas_group.setLayout(canvas_layout)
        layout.addWidget(canvas_group)
        
        self.tabs.addTab(visualization_tab, "可视化")
        
    def update_model_combobox(self):
        """更新模型下拉框"""
        self.model_type_combo.clear()
        
        category = self.model_category_combo.currentText()
        if category in self.available_models:
            models = self.available_models[category]
            self.model_type_combo.addItems(models.keys())
    
    def load_data(self, data_type):
        """加载数据"""
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, f"选择{data_type}数据文件", "", 
                "数据文件 (*.json *.csv *.npy);;所有文件 (*)"
            )
            
            if not file_path:
                return
            
            # 更新文件路径显示
            if data_type == 'train':
                self.train_data_path.setText(file_path)
            else:
                self.test_data_path.setText(file_path)
            
            # 获取数据格式
            data_format = self.data_format_combo.currentText()
            
            # 加载数据
            data = self.api.load_data(file_path, data_format)
            
            # 更新数据信息
            if data_type == 'train':
                self.training_data = data
                self.data_info_text.append(f"训练数据加载成功: {file_path}")
                self.data_info_text.append(f"数据条目数: {len(data) if isinstance(data, list) else '未知'}")
            else:
                self.test_data = data
                self.data_info_text.append(f"测试数据加载成功: {file_path}")
                self.data_info_text.append(f"数据条目数: {len(data) if isinstance(data, list) else '未知'}")
            
            self.statusBar().showMessage(f"{data_type}数据加载成功")
            logger.info(f"{data_type}数据加载成功: {file_path}")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")
            logger.error(f"数据加载失败: {str(e)}\n{error_traceback}")
    
    def preprocess_data(self):
        """预处理数据"""
        try:
            if self.training_data is None:
                QMessageBox.warning(self, "警告", "请先加载训练数据")
                return
            
            # 获取预处理配置
            preprocess_config = {
                'outlier_threshold': self.outlier_threshold.value(),
                'alignment_mode': self.alignment_mode.currentText(),
                'normalize': self.normalize_check.isChecked()
            }
            
            # 预处理数据
            self.processed_data = self.api.preprocess_data(self.training_data, preprocess_config)
            
            # 更新状态
            self.preprocess_status.setText("已处理")
            self.data_info_text.append("数据预处理完成")
            self.statusBar().showMessage("数据预处理完成")
            logger.info("数据预处理完成")
            
            # 分割数据为训练集和测试集 (80%训练, 20%测试)
            if self.processed_data and "trajectories" in self.processed_data:
                import json
                import os
                import numpy as np
                
                trajectories = self.processed_data["trajectories"]
                np.random.shuffle(trajectories)
                split_index = int(0.8 * len(trajectories))
                train_trajectories = trajectories[:split_index]
                test_trajectories = trajectories[split_index:]
                
                # 保存训练集和测试集
                data_dir = "data"
                os.makedirs(data_dir, exist_ok=True)
                
                train_path = os.path.join(data_dir, "train_data.json")
                test_path = os.path.join(data_dir, "test_data.json")
                
                # 保存为JSON文件
                with open(train_path, 'w') as f:
                    json.dump({
                        "trajectories": train_trajectories,
                        "state_dim": self.processed_data.get("state_dim", ()),
                        "action_dim": self.processed_data.get("action_dim", 0)
                    }, f, indent=2)
                    
                with open(test_path, 'w') as f:
                    json.dump({
                        "trajectories": test_trajectories,
                        "state_dim": self.processed_data.get("state_dim", ()),
                        "action_dim": self.processed_data.get("action_dim", 0)
                    }, f, indent=2)
                    
                logger.info(f"训练集和测试集已保存至: {data_dir}")
                self.statusBar().showMessage(f"数据已分割并保存至{data_dir}")
                self.data_info_text.append(f"训练集大小: {len(train_trajectories)}条轨迹")
                self.data_info_text.append(f"测试集大小: {len(test_trajectories)}条轨迹")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"数据预处理失败: {str(e)}")
            logger.error(f"数据预处理失败: {str(e)}\n{error_traceback}")
    
    def load_training_data(self):
        """加载训练数据"""
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, "选择训练数据文件", "",
                "数据文件 (*.json *.csv *.npy *.pt);;所有文件 (*)"
            )
            
            if not file_path:
                return
            
            # 更新文件路径显示
            self.train_data_path_label.setText(file_path)
            
            # 获取数据格式
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 加载数据
            if file_ext == '.npy':
                import numpy as np
                self.training_data = np.load(file_path, allow_pickle=True).item()
            elif file_ext == '.pt':
                self.training_data = torch.load(file_path)
            else:
                # 使用API加载其他格式
                data_format = 'json' if file_ext == '.json' else 'csv'
                self.training_data = self.api.load_data(file_path, data_format)
            
            # 显示数据信息
            self.statusBar().showMessage(f"训练数据加载成功: {file_path}")
            QMessageBox.information(self, "成功", f"训练数据加载成功: {file_path}")
            logger.info(f"训练数据加载成功: {file_path}")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"训练数据加载失败: {str(e)}")
            logger.error(f"训练数据加载失败: {str(e)}\n{error_traceback}")
    
    def start_training(self):
        """开始训练模型"""
        try:
            # 获取模型配置
            model_type = self.model_type_combo.currentText()
            training_method = self.training_method_combo.currentText()

            # 检查是否已加载训练数据
            if self.training_data is None:
                # 如果没有加载数据，询问用户是否使用模拟数据
                reply = QMessageBox.question(
                    self, "未加载数据",
                    "未检测到训练数据，是否使用模拟数据进行训练？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    training_data = self.create_trajectory_data(model_type=model_type)
                    self.statusBar().showMessage("使用模拟数据进行训练")
                else:
                    QMessageBox.warning(self, "警告", "请先加载训练数据")
                    return
            else:
                training_data = self.training_data
            
            # 基本配置
            model_config = {
                'model_type': model_type,
                'training_method': training_method,
                'batch_size': self.batch_size.value(),
                'max_epochs': self.epochs.value(),
                'learning_rate': self.learning_rate.value(),
                'validation_split': self.validation_split.value()
            }
            
            # 添加高级参数
            advanced_params_text = self.advanced_params_text.toPlainText()
            if advanced_params_text:
                try:
                    import json
                    advanced_params = json.loads(advanced_params_text)
                    model_config.update(advanced_params)
                except json.JSONDecodeError:
                    QMessageBox.warning(self, "警告", "高级参数格式不正确，请使用有效的JSON格式")
                    return
            
            # 创建并启动训练线程
            self.training_thread = TrainingThread(self.api, training_data, model_config)
            self.training_thread.update_progress.connect(self.train_progress.setValue)
            self.training_thread.update_status.connect(self.train_status.setText)
            self.training_thread.training_finished.connect(self.on_training_finished)
            
            # 更新UI状态
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.train_status.setText("准备训练...")
            self.train_progress.setValue(0)
            
            # 启动线程
            self.training_thread.start()
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"训练启动失败: {str(e)}")
            logger.error(f"训练启动失败: {str(e)}\n{error_traceback}")
    
    def stop_training(self):
        """停止训练"""
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.train_status.setText("训练已停止")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            logger.info("训练已手动停止")
    
    def on_training_finished(self, result):
        """训练完成回调"""
        # 更新UI状态
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 保存模型结果
        self.current_model = result.get('model')
        self.current_model_id = result.get('model_id')
        
        # 保存训练结果用于可视化
        self.training_result = result
        
        # 更新评估模型下拉框
        if self.current_model_id:
            self.eval_model_combo.addItem(self.current_model_id)
            self.eval_model_combo.setCurrentText(self.current_model_id)
        
        # 显示训练指标
        metrics = result.get('training_metrics', {})
        metrics_text = "训练指标:\n"
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_text += f"{key}: {value:.4f}\n"
        
        self.train_status.setText(f"训练完成: {self.current_model_id}")
        QMessageBox.information(self, "训练完成", f"模型 {self.current_model_id} 训练完成\n\n{metrics_text}")
        logger.info(f"模型 {self.current_model_id} 训练完成")
    
    def load_model(self):
        """加载模型"""
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, "选择模型文件", "", 
                "模型文件 (*.pt *.pth);;所有文件 (*)"
            )
            
            if not file_path:
                return
            
            # 获取模型类型
            model_type, ok = QInputDialog.getItem(
                self, "选择模型类型", "模型类型:", 
                list(self.available_models['基础算法模型'].keys()) + 
                list(self.available_models['轨迹建模与生成模型'].keys()) + 
                list(self.available_models['训练方法模型'].keys()),
                0, False
            )
            
            if not ok:
                return
            
            # 加载模型
            model_class = None
            for category in self.available_models:
                if model_type in self.available_models[category]:
                    model_class = self.available_models[category][model_type]
                    break
            
            if model_class is None:
                QMessageBox.warning(self, "警告", f"未找到模型类型: {model_type}")
                return
            
            # 加载模型
            self.current_model = self.api.load_model(model_class, file_path)
            self.current_model_id = f"{model_type}_{os.path.basename(file_path)}"
            
            # 更新评估模型下拉框
            self.eval_model_combo.addItem(self.current_model_id)
            self.eval_model_combo.setCurrentText(self.current_model_id)
            
            self.statusBar().showMessage(f"模型加载成功: {file_path}")
            logger.info(f"模型加载成功: {file_path}")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            logger.error(f"模型加载失败: {str(e)}\n{error_traceback}")
    
    def evaluate_model(self):
        """评估模型"""
        try:
            if self.current_model is None:
                QMessageBox.warning(self, "警告", "请先训练或加载模型")
                return
            
            if self.test_data is None:
                QMessageBox.warning(self, "警告", "请先加载测试数据")
                return
            
            # 获取评估配置
            eval_config = {
                'method': self.eval_method_combo.currentText(),
                'method_type': self.eval_metric_combo.currentText()
            }
            
            # 评估模型
            eval_result = self.api.evaluate_model(self.current_model, self.test_data, eval_config)
            
            # 显示评估结果
            result_text = "评估结果:\n"
            for key, value in eval_result.items():
                if isinstance(value, (int, float)):
                    result_text += f"{key}: {value:.4f}\n"
                else:
                    result_text += f"{key}: {value}\n"
            
            self.eval_results_text.setText(result_text)
            self.statusBar().showMessage("模型评估完成")
            logger.info("模型评估完成")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"模型评估失败: {str(e)}")
            logger.error(f"模型评估失败: {str(e)}\n{error_traceback}")
    
    def generate_visualization(self):
        """生成可视化"""
        try:
            if self.current_model is None and self.eval_results_text.toPlainText() == "":
                QMessageBox.warning(self, "警告", "请先训练或评估模型以获取可视化数据")
                return
            
            # 获取可视化类型
            vis_type = self.vis_type_combo.currentText()
            
            # 准备可视化数据
            data = {}

            print(self.eval_results_text.toPlainText())
            
            # 根据可视化类型准备不同的数据
            if vis_type == 'line':
                # 优先从训练结果中获取loss曲线数据
                if hasattr(self, 'training_result') and self.training_result:
                    training_metrics = self.training_result.get('training_metrics', {})
                    train_loss = training_metrics.get('train_loss', [])
                    val_loss = training_metrics.get('val_loss', [])
                    
                    if train_loss:
                        # 使用训练loss历史数据
                        data['x'] = list(range(len(train_loss)))
                        data['y'] = {}
                        
                        if train_loss:
                            data['y']['训练loss'] = train_loss
                        if val_loss:
                            data['y']['验证loss'] = val_loss
                            
                        # 设置默认标题和标签
                        if not self.vis_title.text():
                            self.vis_title.setText("Training Loss Curve")
                        if not self.vis_xlabel.text():
                            self.vis_xlabel.setText("Epoch")
                        if not self.vis_ylabel.text():
                            self.vis_ylabel.setText("Loss")
                    else:
                        # 如果没有loss历史，尝试从评估结果中提取数据
                        eval_text = self.eval_results_text.toPlainText()
                        if eval_text:
                            lines = eval_text.strip().split('\n')
                            metrics = {}
                            for line in lines:
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    try:
                                        metrics[key.strip()] = float(value.strip())
                                    except ValueError:
                                        continue
                            
                            if metrics:
                                data['x'] = list(range(len(metrics)))
                                data['y'] = metrics
                else:
                    # 如果没有训练结果，从评估结果中提取数据
                    eval_text = self.eval_results_text.toPlainText()
                    if eval_text:
                        lines = eval_text.strip().split('\n')
                        metrics = {}
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                try:
                                    metrics[key.strip()] = float(value.strip())
                                except ValueError:
                                    continue
                        
                        if metrics:
                            data['x'] = list(range(len(metrics)))
                            data['y'] = metrics
            
            elif vis_type == 'bar':
                # 从评估结果中提取数据
                eval_text = self.eval_results_text.toPlainText()
                if eval_text:
                    lines = eval_text.strip().split('\n')
                    metrics = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            try:
                                metrics[key.strip()] = float(value.strip())
                            except ValueError:
                                continue
                    
                    if metrics:
                        data['x'] = list(metrics.keys())
                        data['y'] = list(metrics.values())
            
            elif vis_type == 'scatter':
                n_points = 100
                data['x'] = np.random.rand(n_points) * 10
                data['y'] = np.random.rand(n_points) * 10
                data['colors'] = np.random.rand(n_points)
                data['sizes'] = np.random.rand(n_points) * 100 + 20
            
            elif vis_type == 'heatmap':
                data['matrix'] = np.random.rand(8, 10)
                data['xlabels'] = [f'特征 {i+1}' for i in range(10)]
                data['ylabels'] = [f'样本 {i+1}' for i in range(8)]
            
            elif vis_type == '3d':
                n_points = 100
                x = np.random.rand(n_points) * 10
                y = np.random.rand(n_points) * 10
                z = np.sin(x) * np.cos(y)
                data['x'] = x
                data['y'] = y
                data['z'] = z
                data['colors'] = z
            
            elif vis_type == 'trajectory':
                t1 = np.linspace(0, 2*np.pi, 100)
                x1 = np.cos(t1) * t1/3
                y1 = np.sin(t1) * t1/3
                
                t2 = np.linspace(0, 4*np.pi, 100)
                x2 = np.cos(t2) * (4 - t2/2)
                y2 = np.sin(t2) * (4 - t2/2)
                
                data['trajectories'] = {
                    '轨迹 1': list(zip(x1, y1)),
                    '轨迹 2': list(zip(x2, y2))
                }
            
            elif vis_type == 'distribution':
                data['distributions'] = {
                    'N(0,1)': np.random.normal(0, 1, 1000),
                    'N(3,1.5)': np.random.normal(3, 1.5, 1000),
                    'X(2)': np.random.exponential(2, 1000)
                }
            
            elif vis_type == 'comparison':
                # 从评估结果中提取数据或生成比较数据
                models = ['模型A', '模型B', '模型C']
                metrics = {
                    '准确率': {
                        '模型A': 0.85,
                        '模型B': 0.82,
                        '模型C': 0.88
                    },
                    '召回率': {
                        '模型A': 0.76,
                        '模型B': 0.81,
                        '模型C': 0.79
                    },
                    'F1分数': {
                        '模型A': 0.80,
                        '模型B': 0.81,
                        '模型C': 0.83
                    }
                }
                data['models'] = models
                data['metrics'] = metrics
            
            elif vis_type == 'embedding':
                from sklearn.datasets import make_blobs
                n_samples = 300
                n_features = 10
                n_clusters = 3
                
                X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
                data['features'] = X
                data['labels'] = y
            
            # 准备可视化配置
            vis_config = {
                'type': vis_type,
                'title': self.vis_title.text(),
                'xlabel': self.vis_xlabel.text(),
                'ylabel': self.vis_ylabel.text(),
                'grid': self.vis_grid.isChecked(),
                'figsize': (8, 6),
                'dpi': 100,
                'save_dir': 'visualizations'
            }
            
            # 确保保存目录存在
            os.makedirs(vis_config['save_dir'], exist_ok=True)
            
            # 执行可视化
            result = self.api.visualize_results(data, vis_config)
            
            # 显示可视化结果
            if result and 'figures' in result and len(result['figures']) > 0:
                # 清除当前画布
                self.vis_canvas.axes.clear()
                
                # 将图表复制到画布
                fig = result['figures'][0]
                for ax in fig.get_axes():
                    # 复制轴的内容到画布
                    self.vis_canvas.axes.set_title(ax.get_title())
                    self.vis_canvas.axes.set_xlabel(ax.get_xlabel())
                    self.vis_canvas.axes.set_ylabel(ax.get_ylabel())
                    
                    # 复制数据
                    for line in ax.get_lines():
                        self.vis_canvas.axes.plot(line.get_xdata(), line.get_ydata(),
                                                 color=line.get_color(),
                                                 linestyle=line.get_linestyle(),
                                                 marker=line.get_marker(),
                                                 label=line.get_label())
                    
                    # 复制散点图
                    for collection in ax.collections:
                        if hasattr(collection, 'get_offsets'):
                            offsets = collection.get_offsets()
                            if len(offsets) > 0:
                                x = offsets[:, 0]
                                y = offsets[:, 1]
                                self.vis_canvas.axes.scatter(x, y,
                                                           c=collection.get_facecolors() if hasattr(collection, 'get_facecolors') else None,
                                                           s=collection.get_sizes() if hasattr(collection, 'get_sizes') else None)
                
                # 添加图例
                if self.vis_canvas.axes.get_legend_handles_labels()[0]:
                    self.vis_canvas.axes.legend()
                
                # 设置网格
                if self.vis_grid.isChecked():
                    self.vis_canvas.axes.grid(True, linestyle='--', alpha=0.7)
                
                # 更新画布
                self.vis_canvas.draw()
                
                # 保存可视化路径
                self.current_vis_path = result['save_paths'][0] if 'save_paths' in result and len(result['save_paths']) > 0 else None
                
                self.statusBar().showMessage(f"可视化生成成功: {vis_type}")
                logger.info(f"可视化生成成功: {vis_type}")
            else:
                QMessageBox.warning(self, "警告", "可视化生成失败，请检查数据")
                logger.warning("可视化生成失败，请检查数据")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"可视化生成失败: {str(e)}")
            logger.error(f"可视化生成失败: {str(e)}\n{error_traceback}")
    def save_visualization(self):
        """保存可视化图表"""
        try:
            if not hasattr(self, 'current_vis_path') or not self.current_vis_path:
                QMessageBox.warning(self, "警告", "请先生成可视化")
                return
            
            # 打开文件对话框选择保存路径
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self, "保存可视化图表", "", 
                "图像文件 (*.png *.jpg *.pdf);;所有文件 (*)"
            )
            
            if not file_path:
                return
            
            # 确保文件有正确的扩展名
            if not any(file_path.endswith(ext) for ext in ['.png', '.jpg', '.pdf']):
                file_path += '.png'
            
            # 保存当前图表
            self.vis_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            self.statusBar().showMessage(f"图表已保存: {file_path}")
            logger.info(f"图表已保存: {file_path}")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"图表保存失败: {str(e)}")
            logger.error(f"图表保存失败: {str(e)}\n{error_traceback}")

if __name__ == "__main__":
    try:
        print("开始启动应用程序...")
        
        # 创建应用程序
        print("创建QApplication...")
        app = QApplication(sys.argv)
        
        # 设置应用程序样式
        print("设置应用程序样式...")
        app.setStyle("Fusion")
        
        # 创建主窗口
        print("创建主窗口...")
        window = MainWindow()
        
        # 显示主窗口
        print("显示主窗口...")
        window.show()
        
        # 运行应用程序
        print("启动应用程序主循环...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用程序启动失败: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc(file=sys.stdout)
        
        # 同时记录到日志文件
        if 'logger' in locals():
            logger.critical(f"应用程序启动失败: {str(e)}\n{traceback.format_exc()}")
        else:
            # 如果logger还未初始化，则直接写入日志文件
            with open('afruits_app_error.log', 'a') as f:
                f.write(f"\n[{datetime.datetime.now()}] 严重错误: 应用程序启动失败\n")
                f.write(f"错误信息: {str(e)}\n")
                f.write(traceback.format_exc())