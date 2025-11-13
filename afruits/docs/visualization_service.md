# 可视化服务文档

## 1. 核心文件
`afruits/core/services/visualization_service.py`

## 2. 类结构
```python
class VisualizationService:
    def __init__(config, logger)  # 初始化服务
    def visualize(data, vis_config)  # 可视化入口
    def _plot_line(data, config)  # 折线图
    def _plot_trajectory(data, config)  # 轨迹图
    def _plot_embedding(data, config)  # 嵌入图
    ... # 其他8种可视化方法
```

## 3. 可视化类型详解

### 3.1 轨迹图
```python
def _plot_trajectory(data, config):
    """
    功能：绘制多条轨迹的起点、终点和路径
    输入参数：
        data: {
            'trajectories': {
                'label1': [[x1,y1], [x2,y2], ...],
                'label2': [[x1,y1], [x2,y2], ...]
            }
        }
    输出：matplotlib图表对象
    关键技术：
        - 起点/终点标记
        - 多轨迹颜色区分
        - 路径平滑处理
    """
```

### 3.2 嵌入图
```python
def _plot_embedding(data, config):
    """
    功能：对高维特征进行降维可视化
    输入参数：
        data: {
            'features': n维特征数组,
            'labels': 类别标签
        }
        config: {
            'method': 'tsne'/'pca', # 降维方法
            'n_components': 2 # 降维维度
        }
    输出：散点图
    关键技术：
        - t-SNE非线性降维
        - PCA线性降维
        - 类别颜色编码
    """
```

### 3.3 3D图
```python
def _plot_3d(data, config):
    """
    功能：三维空间数据可视化
    输入参数：
        data: {
            'x': X轴数据,
            'y': Y轴数据,
            'z': Z轴数据,
            'colors': 颜色数据
        }
    输出：3D散点图/线图
    关键技术：
        - 三维坐标轴设置
        - 深度感知渲染
        - 交互式视角调整
    """
```

## 4. 可视化配置参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| type | str | 'line' | 可视化类型 |
| figsize | tuple | (10,6) | 图表尺寸 |
| dpi | int | 100 | 分辨率 |
| cmap | str | 'viridis' | 颜色映射 |
| title | str | '' | 图表标题 |
| grid | bool | True | 是否显示网格 |
| legend | bool | True | 是否显示图例 |
| save_format | str | 'png' | 保存格式 |
| save_dir | str | 'visualizations' | 保存目录 |

## 5. 使用示例
```python
# 创建可视化服务
vis_service = VisualizationService()

# 轨迹可视化
traj_data = {
    'trajectories': {
        '专家轨迹': [[0,0], [1,1], [2,2]],
        '生成轨迹': [[0,0], [1,0.5], [2,1]]
    }
}
result = vis_service.visualize(traj_data, {'type': 'trajectory', 'title': '轨迹对比'})

# 特征嵌入可视化
embed_data = {
    'features': np.random.rand(100, 10),
    'labels': np.random.randint(0,3,100)
}
result = vis_service.visualize(embed_data, {'type': 'embedding', 'method': 'tsne'})