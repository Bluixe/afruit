# 软件研制阶段说明文档

## 接口层
### afruits/core/api.py
AlgorithmAPI类提供统一的接口，包含以下方法：
- __init__: 初始化API，加载配置和模型
- load_data: 支持json/csv/npy格式的数据加载
- preprocess_data: 实现数据预处理（异常值处理/时间对齐/标准化）
- train_game_model: 执行小样本博弈模型训练
- train_imitation_model: 执行专家轨迹模仿学习模型训练
- evaluate_model: 支持离线评估和多指标评估
- visualize_results: 生成可视化结果
- save_model: 支持PyTorch/ONNX格式模型保存
- load_model: 加载预训练模型
- get_available_models: 获取所有可用模型列表

### afruits/main.py
主程序实现图形界面，包含：
- MatplotlibCanvas: 嵌入式Matplotlib画布
- TrainingThread: 多线程训练管理
- MainWindow: 主窗口包含数据管理、模型训练、评估、可视化四个选项卡
- create_trajectory_data: 生成示例轨迹数据
- 数据加载和预处理回调函数
- 模型训练控制和结果可视化

## 服务层
### afruits/core/services/game_modeling_service.py
GameModelingService实现博弈建模服务：
- train_model: 根据模型类型分发训练请求
- _train_behavior_cloner: 行为克隆模型训练
- _train_offline_rl: 离线强化学习训练
- _train_offline_fsp: 自对弈训练
- _train_adversial_imitation_learner: 对抗模仿学习

### afruits/core/services/imitation_learning_service.py
ImitationLearningService处理专家轨迹模仿：
- train_model: 根据模型类型分发训练任务
- _train_standard: 标准训练方法
- _train_autoencoder: 自编码器训练
- _train_transformer: Transformer模型训练
- _train_diffusion: 扩散模型训练
- _train_vae: 变分自编码器训练
- _train_evolutionary: 进化学习方法
- _train_incremental: 墽드