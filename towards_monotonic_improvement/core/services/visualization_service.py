import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class VisualizationService:
    """
    可视化服务类
    
    负责系统的可视化输出功能，支持多种可视化方式
    """
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        初始化可视化服务
        
        参数:
            config (Dict): 配置参数字典
            logger (logging.Logger): 日志记录器
        """
        # 初始化配置
        self.config = config or {}
        
        # 设置日志记录器
        self.logger = logger or logging.getLogger(__name__)
        
        # 设置默认可视化参数
        self.default_params = {
            'figsize': (10, 6),
            'dpi': 100,
            'cmap': 'viridis',
            'save_format': 'png',
            'save_dir': 'visualizations'
        }
        
        # 更新配置
        if config:
            self.default_params.update(config)
        
        # 创建保存目录
        os.makedirs(self.default_params['save_dir'], exist_ok=True)
        
        self.logger.info(f"可视化服务初始化完成，保存目录: {self.default_params['save_dir']}")
    
    def visualize(self, data: Dict, vis_config: Dict = None) -> Dict:
        """
        可视化数据
        
        参数:
            data (Dict): 可视化数据
            vis_config (Dict): 可视化配置
            
        返回:
            Dict: 可视化结果，包含图表数据和保存路径
        """
        # 合并配置
        config = self.default_params.copy()
        if vis_config:
            config.update(vis_config)
        
        # 初始化结果字典
        result = {
            'figures': [],
            'save_paths': []
        }
        
        # 获取可视化类型
        vis_type = config.get('type', 'line')
        
        # 根据可视化类型选择不同的可视化方法
        if vis_type == 'line':
            fig, save_path = self._plot_line(data, config)
        elif vis_type == 'bar':
            fig, save_path = self._plot_bar(data, config)
        elif vis_type == 'scatter':
            fig, save_path = self._plot_scatter(data, config)
        elif vis_type == 'heatmap':
            fig, save_path = self._plot_heatmap(data, config)
        elif vis_type == '3d':
            fig, save_path = self._plot_3d(data, config)
        elif vis_type == 'trajectory':
            fig, save_path = self._plot_trajectory(data, config)
        elif vis_type == 'distribution':
            fig, save_path = self._plot_distribution(data, config)
        elif vis_type == 'comparison':
            fig, save_path = self._plot_comparison(data, config)
        elif vis_type == 'embedding':
            fig, save_path = self._plot_embedding(data, config)
        else:
            self.logger.warning(f"不支持的可视化类型: {vis_type}，使用默认的折线图")
            fig, save_path = self._plot_line(data, config)
        
        # 添加到结果
        if fig:
            result['figures'].append(fig)
            result['save_paths'].append(save_path)
        
        return result
    
    def _plot_line(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制折线图"""
        self.logger.info("绘制折线图")
        
        # 提取数据
        x_data = data.get('x', range(len(next(iter(data['y'].values())) if isinstance(data.get('y', {}), dict) else data.get('y', []))))
        y_data = data.get('y', {})
        title = config.get('title', '折线图')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制折线图
        if isinstance(y_data, dict):
            for label, values in y_data.items():
                ax.plot(x_data, values, label=label)
            ax.legend()
        else:
            ax.plot(x_data, y_data)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 设置网格
        if config.get('grid', True):
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"折线图已保存: {save_path}")
        return fig, save_path
    
    def _plot_bar(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制柱状图"""
        self.logger.info("绘制柱状图")
        
        # 提取数据
        x_data = data.get('x', range(len(data.get('y', []))))
        y_data = data.get('y', [])
        title = config.get('title', '柱状图')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制柱状图
        if isinstance(y_data, dict):
            # 分组柱状图
            x = np.arange(len(x_data))
            width = 0.8 / len(y_data)
            
            for i, (label, values) in enumerate(y_data.items()):
                ax.bar(x + i * width - 0.4 + width/2, values, width, label=label)
            
            ax.set_xticks(x)
            ax.set_xticklabels(x_data)
            ax.legend()
        else:
            # 普通柱状图
            ax.bar(x_data, y_data)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"柱状图已保存: {save_path}")
        return fig, save_path
    
    def _plot_scatter(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制散点图"""
        self.logger.info("绘制散点图")
        
        # 提取数据
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        colors = data.get('colors', None)
        sizes = data.get('sizes', None)
        labels = data.get('labels', None)
        title = config.get('title', '散点图')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制散点图
        scatter = ax.scatter(x_data, y_data, c=colors, s=sizes, cmap=config['cmap'], alpha=0.7)
        
        # 添加颜色条
        if colors is not None:
            plt.colorbar(scatter, ax=ax, label=config.get('color_label', '值'))
        
        # 添加标签
        if labels is not None:
            for i, label in enumerate(labels):
                ax.annotate(label, (x_data[i], y_data[i]), fontsize=8)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 设置网格
        if config.get('grid', True):
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"散点图已保存: {save_path}")
        return fig, save_path
    
    def _plot_heatmap(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制热力图"""
        self.logger.info("绘制热力图")
        
        # 提取数据
        matrix = data.get('matrix', [])
        xlabels = data.get('xlabels', None)
        ylabels = data.get('ylabels', None)
        title = config.get('title', '热力图')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制热力图
        sns.heatmap(matrix, annot=config.get('annot', True), cmap=config['cmap'],
                   xticklabels=xlabels, yticklabels=ylabels, ax=ax)
        
        # 设置标题
        ax.set_title(title)
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"热力图已保存: {save_path}")
        return fig, save_path
    
    def _plot_3d(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制3D图"""
        self.logger.info("绘制3D图")
        
        # 提取数据
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        z_data = data.get('z', [])
        colors = data.get('colors', None)
        title = config.get('title', '3D图')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        zlabel = config.get('zlabel', 'Z')
        
        # 创建图表
        fig = plt.figure(figsize=config['figsize'], dpi=config['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D图
        if config.get('plot_type', 'scatter') == 'scatter':
            scatter = ax.scatter(x_data, y_data, z_data, c=colors, cmap=config['cmap'], alpha=0.7)
            
            # 添加颜色条
            if colors is not None:
                plt.colorbar(scatter, ax=ax, label=config.get('color_label', '值'))
        else:
            ax.plot3D(x_data, y_data, z_data)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"3D图已保存: {save_path}")
        return fig, save_path
    
    def _plot_trajectory(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制轨迹图"""
        self.logger.info("绘制轨迹图")
        
        # 提取数据
        trajectories = data.get('trajectories', {})
        title = config.get('title', '轨迹图')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制轨迹
        for label, traj in trajectories.items():
            x = [point[0] for point in traj]
            y = [point[1] for point in traj]
            ax.plot(x, y, label=label)
            
            # 标记起点和终点
            ax.scatter(x[0], y[0], marker='o', s=50, label=f"{label}_起点")
            ax.scatter(x[-1], y[-1], marker='x', s=50, label=f"{label}_终点")
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 添加图例
        if config.get('legend', True):
            ax.legend()
        
        # 设置网格
        if config.get('grid', True):
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"轨迹图已保存: {save_path}")
        return fig, save_path
    
    def _plot_distribution(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制分布图"""
        self.logger.info("绘制分布图")
        
        # 提取数据
        distributions = data.get('distributions', {})
        title = config.get('title', '分布图')
        xlabel = config.get('xlabel', '值')
        ylabel = config.get('ylabel', '频率')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制分布图
        if isinstance(distributions, dict):
            for label, values in distributions.items():
                sns.histplot(values, label=label, kde=True, ax=ax)
            ax.legend()
        else:
            sns.histplot(distributions, kde=True, ax=ax)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"分布图已保存: {save_path}")
        return fig, save_path
    
    def _plot_comparison(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制比较图"""
        self.logger.info("绘制比较图")
        
        # 提取数据
        metrics = data.get('metrics', {})
        models = data.get('models', [])
        title = config.get('title', '模型比较')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 设置柱状图参数
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        # 绘制柱状图
        for i, model in enumerate(models):
            values = [metrics[metric][model] for metric in metrics]
            ax.bar(x + i * width - 0.4 + width/2, values, width, label=model)
        
        # 设置刻度和标签
        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics.keys()))
        
        # 设置标题
        ax.set_title(title)
        
        # 添加图例
        ax.legend()
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"比较图已保存: {save_path}")
        return fig, save_path
    
    def _plot_embedding(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """绘制嵌入图"""
        self.logger.info("绘制嵌入图")
        
        # 提取数据
        features = data.get('features', [])
        labels = data.get('labels', None)
        title = config.get('title', '特征嵌入图')
        
        # 降维方法
        method = config.get('method', 'tsne')
        n_components = config.get('n_components', 2)
        
        # 降维
        if method == 'tsne':
            embedding = TSNE(n_components=n_components, random_state=42).fit_transform(features)
        elif method == 'pca':
            embedding = PCA(n_components=n_components, random_state=42).fit_transform(features)
        else:
            self.logger.warning(f"不支持的降维方法: {method}，使用PCA")
            embedding = PCA(n_components=n_components, random_state=42).fit_transform(features)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制散点图
        if labels is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=config['cmap'], alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='类别')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(f"维度1")
        ax.set_ylabel(f"维度2")
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"嵌入图已保存: {save_path}")
        return fig, save_path