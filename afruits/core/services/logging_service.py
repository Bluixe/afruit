import os
import logging
import datetime
from typing import Dict, Optional

class LoggingService:
    """
    日志服务类
    
    负责系统的日志记录功能，支持控制台输出和文件输出
    """
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        """
        初始化日志服务
        
        参数:
            log_level (str): 日志级别，可选值为 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            log_dir (str): 日志文件目录
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志级别
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.log_level = level_map.get(log_level.upper(), logging.INFO)
        
        # 生成日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"algorithm_{timestamp}.log")
        
        # 配置根日志记录器
        self._setup_root_logger()
        
        # 创建应用日志记录器
        self.logger = self._create_logger("algorithm")
        
        self.logger.info(f"日志服务初始化完成，日志文件: {self.log_file}")
    
    def _setup_root_logger(self):
        """配置根日志记录器"""
        # 重置根日志记录器
        root_logger = logging.getLogger()
        root_logger.handlers = []
        root_logger.setLevel(self.log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器到根日志记录器
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def _create_logger(self, name: str) -> logging.Logger:
        """
        创建日志记录器
        
        参数:
            name (str): 日志记录器名称
            
        返回:
            logging.Logger: 日志记录器
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        return logger
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        获取日志记录器
        
        参数:
            name (str, optional): 日志记录器名称，默认为None，表示使用默认日志记录器
            
        返回:
            logging.Logger: 日志记录器
        """
        if name:
            return self._create_logger(name)
        return self.logger
    
    def set_log_level(self, log_level: str):
        """
        设置日志级别
        
        参数:
            log_level (str): 日志级别，可选值为 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.log_level = level_map.get(log_level.upper(), logging.INFO)
        
        # 更新根日志记录器的级别
        logging.getLogger().setLevel(self.log_level)
        
        # 更新处理器的级别
        for handler in logging.getLogger().handlers:
            handler.setLevel(self.log_level)
        
        self.logger.info(f"日志级别已更新为: {log_level}")
    
    def log_config(self, config: Dict):
        """
        记录配置信息
        
        参数:
            config (Dict): 配置字典
        """
        self.logger.info("配置信息:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict, prefix: str = ""):
        """
        记录指标信息
        
        参数:
            metrics (Dict): 指标字典
            prefix (str, optional): 指标前缀
        """
        self.logger.info(f"{prefix}指标信息:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")