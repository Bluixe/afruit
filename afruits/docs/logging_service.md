# 日志服务文档

## 1. 实现位置
`afruits/main.py` - 日志服务集成在系统主模块中

## 2. 核心功能
```python
# 日志初始化 (main.py 第26-36行)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),        # 控制台输出
        logging.FileHandler('afruits_app.log')  # 文件记录
    ]
)
logger = logging.getLogger('afruits_app')  # 全局日志对象
```

## 3. 日志级别管理
| 级别 | 使用场景 | 示例 |
|------|----------|------|
| DEBUG | 调试信息 | `logger.debug("参数详情: %s", params)` |
| INFO | 正常操作 | `logger.info("训练开始: %s", model_id)` |
| WARNING | 潜在问题 | `logger.warning("内存使用超过80%")` |
| ERROR | 操作失败 | `logger.error("模型加载失败: %s", str(e))` |
| CRITICAL | 系统错误 | `logger.critical("服务崩溃: %s", traceback)` |

## 4. 日志格式说明
```text
2025-11-13 18:28:21,432 - afruits_app - INFO - 应用程序初始化完成
[时间] - [日志器名称] - [级别] - [消息]
```

## 5. 关键日志点
```python
# 系统启动日志
logger.info("应用程序初始化完成")  # 主窗口初始化后

# 训练过程日志
logger.info(f"开始训练模型: {model_id}")  # 训练开始时
logger.error(f"训练失败: {str(e)}")  # 训练异常时

# 数据操作日志
logger.info(f"加载数据: {file_path}")  # 数据加载时
logger.warning("数据预处理异常值比例过高")  # 数据异常时

# 可视化日志
logger.debug(f"生成{vis_type}图表")  # 图表生成时
```

## 6. 日志文件管理
- 文件名：`afruits_app.log`
- 存储位置：当前工作目录
- 滚动策略：无大小限制（生产环境应改为RotatingFileHandler）
- 查看方式：
  ```bash
  tail -f afruits_app.log  # 实时监控
  grep "ERROR" afruits_app.log  # 筛选错误
  ```

## 7. 最佳实践
```python
# 带异常堆栈的日志记录
try:
    risky_operation()
except Exception as e:
    logger.error(f"操作失败: {str(e)}", exc_info=True)  # 包含堆栈信息

# 性能监控日志
start_time = time.time()
process_data()
logger.info(f"数据处理耗时: {time.time()-start_time:.2f}s")