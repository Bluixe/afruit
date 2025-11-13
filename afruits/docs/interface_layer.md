# 接口层设计文档

## 1. 核心文件
`afruits/main.py` - 系统主入口，实现GUI界面和功能调度

## 2. 功能接口

### 2.1 数据管理接口
```python
class MainWindow:
    def create_data_tab(self):
        # 功能点：
        # - 训练/测试数据加载 (load_data)
        # - 数据格式选择 (json/csv/npy)
        # - 异常值处理 (outlier_threshold)
        # - 轨迹对齐 (alignment_mode)
        # - 数据标准化 (normalize_check)
        # - 数据预处理执行 (preprocess_data)
```

### 2.2 模型训练接口
```python
    def create_training_tab(self):
        # 功能点：
        # - 模型类别选择 (model_category_combo)
        # - 模型类型选择 (model_type_combo)
        # - 训练方法选择 (training_method_combo)
        # - 训练数据加载 (load_training_data)
        # - 训练参数配置 (batch_size/epochs等)
        # - 训练控制 (start_training/stop_training)
```

### 2.3 模型评估接口
```python
    def create_evaluation_tab(self):
        # 功能点：
        # - 模型选择 (eval_model_combo)
        # - 评估方法选择 (eval_method_combo)
        # - 评估指标选择 (eval_metric_combo)
        # - 评估执行 (evaluate_model)
```

### 2.4 可视化接口
```python
    def create_visualization_tab(self):
        # 功能点：
        # - 可视化类型选择 (vis_type_combo)
        # - 图表配置 (title/xlabel/ylabel)
        # - 可视化生成 (generate_visualization)
        # - 可视化保存 (save_visualization)
```

## 3. 接口调用流程
```mermaid
graph TD
    A[用户操作] --> B{功能类型}
    B -->|数据管理| C[create_data_tab]
    B -->|模型训练| D[create_training_tab]
    B -->|模型评估| E[create_evaluation_tab]
    B -->|可视化| F[create_visualization_tab]
    C --> G[调用服务层API]
    D --> G
    E --> G
    F --> G