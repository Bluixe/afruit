# 数据加载器文档

## 1. 核心文件
`afruits/utils/DataLoader.py`

## 2. 类结构
```python
class DataLoaderUtil:
    def __init__(self)  # 初始化
    def load_expert_data(data, batch_size)  # 加载专家轨迹数据
    def preprocess_data(data, config)  # 数据预处理
    def split_dataset(data, ratios)  # 数据集划分
    def augment_data(data, augmentation_factor)  # 数据增强
```

## 3. 核心方法详解

### 3.1 专家数据加载
```python
def load_expert_data(data, batch_size):
    """
    功能：加载并处理专家轨迹数据
    输入参数：
        data: 原始轨迹数据，格式为字典:
            {
                'states': 状态序列,
                'actions': 动作序列,
                'rewards': 奖励序列 (可选)
            }
        batch_size: 批次大小
    输出：
        dataloader: PyTorch DataLoader对象
        dataset_info: 数据集元信息
    处理流程：
        1. 数据标准化
        2. 序列填充对齐
        3. 构建TensorDataset
        4. 创建DataLoader
    """
```

### 3.2 数据预处理
```python
def preprocess_data(data, config):
    """
    功能：执行数据预处理流水线
    支持操作：
        - 异常值处理 (outlier_threshold)
        - 序列对齐 (alignment_mode)
        - 标准化/归一化 (normalize)
        - 特征工程
    配置参数：
        config = {
            'outlier_threshold': 3.0,  # 异常值阈值
            'alignment_mode': 'linear',  # 对齐方式
            'normalize': True  # 是否标准化
        }
    """
```

### 3.3 数据增强
```python
def augment_data(data, augmentation_factor):
    """
    功能：通过变换扩增数据集
    增强技术：
        - 随机缩放 (0.9-1.1倍)
        - 随机旋转 (±5度)
        - 添加高斯噪声 (std=0.01)
        - 时间扭曲 (动态时间规整)
    输出：
        扩增后的数据集 (原始大小的augmentation_factor倍)
    """
```

## 4. 使用示例
```python
# 初始化数据加载器
dataloader_util = DataLoaderUtil()

# 加载原始数据
raw_data = {
    'states': np.load('states.npy'),
    'actions': np.load('actions.npy')
}

# 数据预处理
preprocessed_data = dataloader_util.preprocess_data(
    raw_data,
    config={
        'outlier_threshold': 3.0,
        'alignment_mode': 'linear',
        'normalize': True
    }
)

# 数据增强
augmented_data = dataloader_util.augment_data(
    preprocessed_data,
    augmentation_factor=5
)

# 创建数据加载器
dataloader = dataloader_util.load_expert_data(
    augmented_data,
    batch_size=32
)['dataloader']

# 数据集划分
train_data, val_data, test_data = dataloader_util.split_dataset(
    augmented_data,
    ratios=[0.7, 0.2, 0.1]
)
```

## 5. 关键参数说明
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| batch_size | int | 32 | 数据加载批次大小 |
| outlier_threshold | float | 3.0 | 异常值检测阈值 (标准差倍数) |
| alignment_mode | str | 'linear' | 序列对齐方式 (linear/nearest/cubic) |
| normalize | bool | True | 是否标准化数据 |
| augmentation_factor | int | 5 | 数据增强倍数 |
| ratios | list | [0.7,0.2,0.1] | 数据集划分比例 [训练,验证,测试] |