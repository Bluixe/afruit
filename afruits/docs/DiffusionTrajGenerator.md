# 扩散轨迹生成器文档

## 1. 核心文件
`afruits/utils/DiffusionTrajGenerator.py`

## 2. 类结构
```python
class DiffusionTrajGenerator:
    def __init__(diffusion_steps, noise_schedule, dropout, im_embd)  # 初始化
    def _get_noise_schedule()  # 噪声调度算法
    def load_dataset(data, batch_size)  # 数据加载
    def build_model(state_dim, cond_dim)  # 模型构建
    def train(dataloader, epochs)  # 模型训练
    def generate(batch_size, seq_len, cond_data)  # 轨迹生成
    def save_model(save_path)  # 模型保存
    @staticmethod
    def load_model(load_path)  # 模型加载
```

## 3. 核心方法详解

### 3.1 噪声调度
```python
def _get_noise_schedule():
    """
    功能：生成扩散过程的噪声调度
    支持类型：
        - linear: 线性噪声增加
        - cosine: 余弦噪声增加
    数学原理：
        beta_t = 0.0001 + (0.02-0.0001) * (t/T)  # 线性
        alpha_t = 1 - beta_t
        alpha_cumprod_t = ∏_{i=1}^t alpha_i
    """
```

### 3.2 模型构建
```python
def build_model(state_dim, cond_dim):
    """
    功能：构建扩散模型网络
    网络结构：
        - 时间嵌入层 (time_embed)
        - 条件嵌入层 (cond_embed)
        - U-Net结构MLP：
          * 输入: [状态, 时间嵌入]
          * 隐藏层: 256->512->512->256
          * 输出: 预测噪声
    技术特点：
        - 支持3D轨迹输入 (batch, seq_len, state_dim)
        - 支持条件控制生成
    """
```

### 3.3 训练过程
```python
def train(dataloader, epochs):
    """
    功能：训练扩散模型
    流程：
        1. 对轨迹数据添加随机噪声
        2. 预测添加的噪声
        3. 计算预测噪声与实际噪声的MSE损失
        4. 反向传播更新权重
    关键参数：
        - t: 随机时间步 (0~1)
        - 噪声调度: alpha_cumprod控制噪声强度
    """
```

### 3.4 轨迹生成
```python
def generate(batch_size, seq_len, cond_data):
    """
    功能：生成新轨迹
    流程：
        1. 初始化随机噪声
        2. 从T步到0步迭代去噪：
           x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t)*ε_θ) + sqrt(beta_t)*z
        3. 物理约束检查 (可选)
    输出：
        - trajectories: 生成轨迹 (numpy数组)
        - validity: 物理有效性标志
    """
```

## 4. 使用示例
```python
# 初始化生成器
generator = DiffusionTrajGenerator(
    diffusion_steps=1000,
    noise_schedule='cosine',
    dropout=0.2,
    im_embd=128
)

# 构建模型
generator.build_model(state_dim=10, cond_dim=5)

# 训练模型
dataloader = generator.load_dataset(trajectory_data, batch_size=32)
training_history = generator.train(dataloader, epochs=100)

# 生成轨迹
cond_data = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])  # 条件信息
result = generator.generate(
    batch_size=5,
    seq_len=20,
    cond_data=cond_data
)

# 保存模型
generator.save_model('models/diffusion_generator.pt')

# 加载模型
loaded_generator = DiffusionTrajGenerator.load_model('models/diffusion_generator.pt')
```

## 5. 关键参数说明
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| diffusion_steps | int | 1000 | 扩散过程总步数 |
| noise_schedule | str | 'cosine' | 噪声调度类型 |
| dropout | float | 0.2 | 网络dropout比例 |
| im_embd | int | 128 | 图像嵌入维度 |
| state_dim | int | - | 状态空间维度 |
| cond_dim | int | 0 | 条件信息维度 |
| batch_size | int | 32 | 训练批次大小 |
| epochs | int | 100 | 训练轮数 |