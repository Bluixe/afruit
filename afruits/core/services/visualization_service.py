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
        
        self.logger.info(f"Visualization service initialized. Save dir: {self.default_params['save_dir']}")
    
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
        elif vis_type == 'traj_heatmap':
            fig, save_path = self._plot_traj_heatmap(data, config)
        elif vis_type == 'action_hist':
            fig, save_path = self._plot_action_hist(data, config)
        elif vis_type == 'radar':
            fig, save_path = self._plot_radar(data, config)
        else:
            self.logger.warning(f"Unsupported visualization type: {vis_type}, falling back to line plot")
            fig, save_path = self._plot_line(data, config)
        
        # 添加到结果
        if fig:
            result['figures'].append(fig)
            result['save_paths'].append(save_path)
        
        return result
    
    def _plot_line(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot line chart"""
        self.logger.info("Plotting line chart")
        
        # 提取数据（健壮处理，避免KeyError/StopIteration）
        y_data = data.get('y', None)

        # 推断x长度
        def infer_length(y):
            if isinstance(y, dict):
                if len(y) == 0:
                    return 0
                first_series = next(iter(y.values()))
                try:
                    return len(first_series)
                except TypeError:
                    return 0
            elif isinstance(y, (list, tuple, np.ndarray)):
                return len(y)
            return 0

        n = infer_length(y_data)
        if n == 0:
            raise ValueError("Invalid y for line plot. Expect y to be list/array or dict(label->list/array).")

        x_data = data.get('x', list(range(n)))

        title = config.get('title', 'Line Chart')
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

        self.logger.info(f"Line chart saved: {save_path}")
        return fig, save_path
    
    def _plot_bar(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot bar chart (supports grouped bars and label-flattened bars)"""
        self.logger.info("Plotting bar chart")
    
        # Extract data
        x_data = data.get('x', range(len(data.get('y', []))))
        y_data = data.get('y', [])
        title = config.get('title', 'Bar Chart')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
    
        # Style controls
        single_bar_width = float(config.get('bar_width', 0.35))           # width for single-series bars (narrower to create gaps)
        group_total_width = float(config.get('group_total_width', 0.7))   # total width used by a group in grouped bars
        tick_rotation = int(config.get('tick_rotation', 45))              # rotate for readability of long labels
    
        # Create figure
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
    
        # Render
        if isinstance(y_data, dict):
            # If x_data has only ONE category (e.g., ['accuracy']), flatten by label so each label becomes a bar
            try:
                x_len = len(x_data)
            except Exception:
                x_len = 1
            if x_len == 1:
                labels = list(y_data.keys())
                # values assumed to be list-like with length==1 per label
                values = []
                for v in y_data.values():
                    try:
                        # take first value if list-like, else cast to float
                        values.append(float(v[0] if isinstance(v, (list, tuple, np.ndarray)) else float(v)))
                    except Exception:
                        values.append(np.nan)
                positions = np.arange(len(labels))
                ax.bar(positions, values, width=single_bar_width, align='center')
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=tick_rotation, ha='right')
            else:
                # Grouped bars: each x category has multiple labels (series)
                x = np.arange(len(x_data))
                n_series = max(1, len(y_data))
                total_w = min(0.95, max(0.1, group_total_width))
                bar_w = total_w / n_series
                left = x - total_w / 2.0
    
                for i, (series_label, values) in enumerate(y_data.items()):
                    xi = left + i * bar_w + bar_w / 2.0
                    ax.bar(xi, values, bar_w, label=series_label)
    
                ax.set_xticks(x)
                ax.set_xticklabels(x_data)
                ax.legend()
        else:
            # Simple single-series bars
            try:
                positions = np.arange(len(x_data))
                ax.bar(positions, y_data, width=single_bar_width, align='center')
                ax.set_xticks(positions)
                ax.set_xticklabels(x_data, rotation=tick_rotation, ha='right')
            except Exception:
                # Fallback to default behavior if categorical fails
                ax.bar(x_data, y_data, width=single_bar_width)
    
        # Titles and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Add a little horizontal margin so bars have visible gaps at both ends
        try:
            ax.margins(x=0.1)
        except Exception:
            pass
        
        # Save
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
    
        self.logger.info(f"Bar chart saved: {save_path}")
        return fig, save_path
    
    def _plot_scatter(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot scatter"""
        self.logger.info("Plotting scatter")
        
        # 提取数据
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        colors = data.get('colors', None)
        sizes = data.get('sizes', None)
        labels = data.get('labels', None)
        title = config.get('title', 'Scatter Plot')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制散点图
        scatter = ax.scatter(x_data, y_data, c=colors, s=sizes, cmap=config['cmap'], alpha=0.7)
        
        # 添加颜色条
        if colors is not None:
            plt.colorbar(scatter, ax=ax, label=config.get('color_label', 'Value'))
        
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
        
        self.logger.info(f"Scatter plot saved: {save_path}")
        return fig, save_path
    
    def _plot_heatmap(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot heatmap"""
        self.logger.info("Plotting heatmap")
        
        # 提取数据
        matrix = data.get('matrix', [])
        xlabels = data.get('xlabels', None)
        ylabels = data.get('ylabels', None)
        title = config.get('title', 'Heatmap')
        
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
        
        self.logger.info(f"Heatmap saved: {save_path}")
        return fig, save_path
    
    def _plot_3d(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot 3D figure (supports sampling a random trajectory from trajectory.npy)"""
        self.logger.info("Plotting 3D figure")
    
        title = config.get('title', '3D Plot')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        zlabel = config.get('zlabel', 'Z')
        plot_type = config.get('plot_type', 'scatter')
    
        # 优先使用直接提供的x/y/z数据
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        z_data = data.get('z', [])
        colors = data.get('colors', None)
    
        def _try_load_random_traj():
            # 允许外部传入路径；否则按常见位置回退
            candidate_paths = []
            if 'traj_path' in config:
                candidate_paths.append(config['traj_path'])
            candidate_paths.extend([
                'afruits/data/trajectory.npy',
                'data/trajectory.npy',
                'trajectory.npy',
            ])
            for p in candidate_paths:
                try:
                    if os.path.exists(p):
                        arr = np.load(p, allow_pickle=True)
                        if len(arr) == 0:
                            continue
                        idx = np.random.randint(0, len(arr))
                        return arr[idx], p, idx
                except Exception as e:
                    self.logger.warning(f"加载轨迹文件失败 {p}: {e}")
            return None, None, None
    
        # 判断是否需要从轨迹文件中加载
        use_traj_file = (len(x_data) == 0 or len(y_data) == 0 or len(z_data) == 0) and ('states' not in data)
    
        # 创建图表
        fig = plt.figure(figsize=config['figsize'], dpi=config['dpi'])
        ax = fig.add_subplot(111, projection='3d')
    
        if not use_traj_file and len(x_data) > 0 and len(y_data) > 0 and len(z_data) > 0:
            # 兼容旧接口：x/y/z直接绘制
            if plot_type == 'scatter':
                scatter = ax.scatter(x_data, y_data, z_data, c=colors, cmap=config['cmap'], alpha=0.7)
                if colors is not None:
                    plt.colorbar(scatter, ax=ax, label=config.get('color_label', 'Value'))
            else:
                ax.plot3D(x_data, y_data, z_data)
        else:
            # 优先从data['states']中解析，否则从npy随机抽取
            traj_sample = None
            src_path, idx = None, None
    
            if 'states' in data and isinstance(data['states'], (list, tuple, np.ndarray)):
                traj_sample = {'states': np.asarray(data['states'])}
            else:
                traj_sample, src_path, idx = _try_load_random_traj()
    
            if traj_sample is None or not isinstance(traj_sample, dict) or 'states' not in traj_sample:
                raise ValueError("No valid 3D plot data provided, and failed to load a valid trajectory from trajectory.npy (requires a dict with 'states').")
    
            states = np.asarray(traj_sample['states'])
            if states.ndim < 2 or states.shape[1] < 9:
                raise ValueError(f"Invalid 'states' shape, expected [T, >=9], got {states.shape}")
    
            own_x, own_y, own_z = states[:, 0], states[:, 1], states[:, 2]
            bdt_x, bdt_y, bdt_z = states[:, 6], states[:, 7], states[:, 8]
    
            # 参考 generate_traj_data 的绘制风格
            ax.plot(own_x, own_y, own_z, label='Ownship', color='C0')
            ax.plot(bdt_x, bdt_y, bdt_z, label='Bandit', color='C1')
    
            ax.scatter(own_x[0], own_y[0], own_z[0], c='C0', marker='o', s=30)
            ax.scatter(own_x[-1], own_y[-1], own_z[-1], c='C0', marker='x', s=50)
            ax.scatter(bdt_x[0], bdt_y[0], bdt_z[0], c='C1', marker='o', s=30)
            ax.scatter(bdt_x[-1], bdt_y[-1], bdt_z[-1], c='C1', marker='x', s=50)
    
            ax.legend()
            if src_path is not None:
                self.logger.info(f"3D trajectory from: {src_path} [sample_index={idx}]")
    
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
    
        self.logger.info(f"3D plot saved: {save_path}")
        return fig, save_path
    
    def _plot_trajectory(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot trajectory (2D XY; supports sampling a random one from trajectory.npy)"""
        self.logger.info("Plotting trajectory")
    
        title = config.get('title', 'Trajectory Plot')
        xlabel = config.get('xlabel', 'X')
        ylabel = config.get('ylabel', 'Y')
        show_legend = config.get('legend', True)
        show_grid = config.get('grid', True)
    
        def _try_load_random_traj():
            candidate_paths = []
            if 'traj_path' in config:
                candidate_paths.append(config['traj_path'])
            candidate_paths.extend([
                'afruits/data/trajectory.npy',
                'data/trajectory.npy',
                'trajectory.npy',
            ])
            for p in candidate_paths:
                try:
                    if os.path.exists(p):
                        arr = np.load(p, allow_pickle=True)
                        if len(arr) == 0:
                            continue
                        idx = np.random.randint(0, len(arr))
                        return arr[idx], p, idx
                except Exception as e:
                    self.logger.warning(f"加载轨迹文件失败 {p}: {e}")
            return None, None, None
    
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
    
        trajectories = data.get('trajectories', None)
        plotted = False
    
        # 1) 兼容旧接口：dict(label -> [(x,y), ...])
        if isinstance(trajectories, dict) and len(trajectories) > 0:
            for label, traj in trajectories.items():
                x = [pt[0] for pt in traj]
                y = [pt[1] for pt in traj]
                ax.plot(x, y, label=label)
                ax.scatter(x[0], y[0], marker='o', s=50)
                ax.scatter(x[-1], y[-1], marker='x', s=50)
            plotted = True
    
        # 2) 若直接提供了'states'
        if not plotted and 'states' in data and isinstance(data['states'], (list, tuple, np.ndarray)):
            states = np.asarray(data['states'])
            if states.ndim >= 2 and states.shape[1] >= 8:
                own_x, own_y = states[:, 0], states[:, 1]
                bdt_x, bdt_y = states[:, 6], states[:, 7]
                ax.plot(own_x, own_y, label='Ownship', color='C0')
                ax.plot(bdt_x, bdt_y, label='Bandit', color='C1')
                ax.scatter(own_x[0], own_y[0], c='C0', marker='o', s=40)
                ax.scatter(own_x[-1], own_y[-1], c='C0', marker='x', s=60)
                ax.scatter(bdt_x[0], bdt_y[0], c='C1', marker='o', s=40)
                ax.scatter(bdt_x[-1], bdt_y[-1], c='C1', marker='x', s=60)
                plotted = True
    
        # 3) 默认：从trajectory.npy随机采样
        if not plotted:
            traj_sample, src_path, idx = _try_load_random_traj()
            if traj_sample is None or not isinstance(traj_sample, dict) or 'states' not in traj_sample:
                raise ValueError("No valid trajectory provided, and failed to load a valid trajectory from trajectory.npy (requires a dict with 'states').")
    
            states = np.asarray(traj_sample['states'])
            if states.ndim < 2 or states.shape[1] < 8:
                raise ValueError(f"Invalid 'states' shape, expected [T, >=8], got {states.shape}")
    
            own_x, own_y = states[:, 0], states[:, 1]
            bdt_x, bdt_y = states[:, 6], states[:, 7]
    
            ax.plot(own_x, own_y, label='Ownship', color='C0')
            ax.plot(bdt_x, bdt_y, label='Bandit', color='C1')
            ax.scatter(own_x[0], own_y[0], c='C0', marker='o', s=40)
            ax.scatter(own_x[-1], own_y[-1], c='C0', marker='x', s=60)
            ax.scatter(bdt_x[0], bdt_y[0], c='C1', marker='o', s=40)
            ax.scatter(bdt_x[-1], bdt_y[-1], c='C1', marker='x', s=60)
            self.logger.info(f"Trajectory from: {src_path} [sample_index={idx}]")
    
        # 标注标题/坐标轴
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
        # 更贴近飞行轨迹的比例
        try:
            ax.set_aspect('equal', adjustable='box')
        except Exception:
            pass
    
        if show_legend:
            ax.legend()
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
    
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
    
        self.logger.info(f"Trajectory plot saved: {save_path}")
        return fig, save_path
    
    def _plot_distribution(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot distribution"""
        self.logger.info("Plotting distribution")
        
        # 提取数据
        distributions = data.get('distributions', {})
        title = config.get('title', 'Distribution')
        xlabel = config.get('xlabel', 'Value')
        ylabel = config.get('ylabel', 'Frequency')
        
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
        
        self.logger.info(f"Distribution plot saved: {save_path}")
        return fig, save_path
    
    def _plot_comparison(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot comparison"""
        self.logger.info("Plotting comparison")
        
        # 提取数据
        metrics = data.get('metrics', {})
        models = data.get('models', [])
        title = config.get('title', 'Model Comparison')
        
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
        
        self.logger.info(f"Comparison plot saved: {save_path}")
        return fig, save_path
    
    def _plot_embedding(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot embedding"""
        self.logger.info("Plotting embedding")
        
        # 提取数据
        features = data.get('features', [])
        labels = data.get('labels', None)
        title = config.get('title', 'Feature Embedding')
        
        # 降维方法
        method = config.get('method', 'tsne')
        n_components = config.get('n_components', 2)
        
        # 降维
        if method == 'tsne':
            embedding = TSNE(n_components=n_components, random_state=42).fit_transform(features)
        elif method == 'pca':
            embedding = PCA(n_components=n_components, random_state=42).fit_transform(features)
        else:
            self.logger.warning(f"Unsupported dimensionality reduction method: {method}, falling back to PCA")
            embedding = PCA(n_components=n_components, random_state=42).fit_transform(features)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 绘制散点图
        if labels is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=config['cmap'], alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Class')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(f"Dim 1")
        ax.set_ylabel(f"Dim 2")
        
        # 保存图表
        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        
        self.logger.info(f"Embedding plot saved: {save_path}")
        return fig, save_path
    def _plot_traj_heatmap(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot 2D trajectory heatmap (XY density); auto-load from trajectory.npy if not provided"""
        self.logger.info("Plotting traj_heatmap")

        # 1) Collect XY from provided data or npy
        def _collect_xy_from_input(d: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            # states: [T, D] with D&gt;=2
            if isinstance(d, dict):
                if 'states' in d and isinstance(d['states'], (list, tuple, np.ndarray)):
                    st = np.asarray(d['states'])
                    if st.ndim >= 2 and st.shape[1] >= 2:
                        return st[:, 0], st[:, 1]
                # trajectories: dict(label -&gt; [(x,y), ...]) or list of dicts with 'states'
                if 'trajectories' in d and d['trajectories'] is not None:
                    tr = d['trajectories']
                    xs, ys = [], []
                    if isinstance(tr, dict):
                        for traj in tr.values():
                            if isinstance(traj, (list, tuple)) and len(traj) >0 and isinstance(traj[0], (list, tuple, np.ndarray)) and len(traj[0]) >= 2:
                                xy = np.asarray(traj)
                                xs.append(xy[:, 0])
                                ys.append(xy[:, 1])
                            elif isinstance(traj, dict) and 'states' in traj:
                                st = np.asarray(traj['states'])
                                if st.ndim >= 2 and st.shape[1] >= 2:
                                    xs.append(st[:, 0]); ys.append(st[:, 1])
                    elif isinstance(tr, list):
                        for traj in tr:
                            if isinstance(traj, dict) and 'states' in traj:
                                st = np.asarray(traj['states'])
                                if st.ndim >= 2 and st.shape[1] >= 2:
                                    xs.append(st[:, 0]); ys.append(st[:, 1])
                    if xs and ys:
                        return np.concatenate(xs), np.concatenate(ys)
            return None

        xy = _collect_xy_from_input(data)

        # 2) If not available, sample from trajectory.npy
        def _try_collect_from_npy() -> Optional[Tuple[np.ndarray, np.ndarray]]:
            candidate_paths = []
            if 'traj_path' in config:
                candidate_paths.append(config['traj_path'])
            candidate_paths.extend([
                'afruits/data/trajectory.npy',
                'data/trajectory.npy',
                'trajectory.npy',
            ])
            for p in candidate_paths:
                try:
                    if os.path.exists(p):
                        arr = np.load(p, allow_pickle=True)
                        if len(arr) == 0:
                            continue
                        # Aggregate multiple samples to form a denser heatmap
                        xs, ys = [], []
                        sample_indices = np.arange(len(arr))
                        # Limit number of samples to avoid too heavy memory; use up to 50
                        for idx in sample_indices[:min(50, len(arr))]:
                            sample = arr[idx]
                            if isinstance(sample, dict) and 'states' in sample:
                                st = np.asarray(sample['states'])
                                if st.ndim >= 2 and st.shape[1] >= 2:
                                    xs.append(st[:, 0]); ys.append(st[:, 1])
                        if xs and ys:
                            self.logger.info(f"traj_heatmap aggregated from: {p}, samples={len(xs)}")
                            return np.concatenate(xs), np.concatenate(ys)
                except Exception as e:
                    self.logger.warning(f"加载轨迹文件失败 {p}: {e}")
            return None

        if xy is None:
            xy = _try_collect_from_npy()

        if xy is None:
            raise ValueError("traj_heatmap requires ('states' or 'trajectories') with XY or a valid trajectory.npy")

        x_all, y_all = xy
        x_all = np.asarray(x_all).astype(float)
        y_all = np.asarray(y_all).astype(float)

        # 3) Compute 2D histogram
        bins = int(config.get('bins', 100))
        title = config.get('title', 'Trajectory Heatmap')
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])

        # Robust bounds
        x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
        y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
        if not np.isfinite([x_min, x_max, y_min, y_max]).all() or x_min == x_max or y_min == y_max:
            # Fallback to default ranges
            x_min, x_max = 0.0, 1.0
            y_min, y_max = 0.0, 1.0

        H, xedges, yedges = np.histogram2d(x_all, y_all, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
        # Plot with imshow
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(H.T, origin='lower', extent=extent, aspect='auto', cmap=config['cmap'])
        plt.colorbar(im, ax=ax, label='Density')

        ax.set_title(title)
        ax.set_xlabel(config.get('xlabel', 'X'))
        ax.set_ylabel(config.get('ylabel', 'Y'))
        try:
            ax.set_aspect('equal', adjustable='box')
        except Exception:
            pass

        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        self.logger.info(f"Trajectory heatmap saved: {save_path}")
        return fig, save_path

    def _plot_action_hist(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot histogram of discrete actions; can auto-load actions from trajectory.npy when missing"""
        self.logger.info("Plotting action_hist")

        def _collect_actions_from_input(d: Dict) -> Optional[np.ndarray]:
            if isinstance(d, dict):
                if 'actions' in d and isinstance(d['actions'], (list, tuple, np.ndarray)):
                    return np.asarray(d['actions']).reshape(-1)
                if 'trajectories' in d and d['trajectories'] is not None:
                    tr = d['trajectories']
                    acts = []
                    if isinstance(tr, dict):
                        for traj in tr.values():
                            if isinstance(traj, dict) and 'actions' in traj:
                                acts.append(np.asarray(traj['actions']).reshape(-1))
                    elif isinstance(tr, list):
                        for traj in tr:
                            if isinstance(traj, dict) and 'actions' in traj:
                                acts.append(np.asarray(traj['actions']).reshape(-1))
                    if acts:
                        return np.concatenate(acts)
            return None

        actions = _collect_actions_from_input(data)

        def _try_collect_actions_from_npy() -> Optional[np.ndarray]:
            candidate_paths = []
            if 'traj_path' in config:
                candidate_paths.append(config['traj_path'])
            candidate_paths.extend([
                'afruits/data/trajectory.npy',
                'data/trajectory.npy',
                'trajectory.npy',
            ])
            for p in candidate_paths:
                try:
                    if os.path.exists(p):
                        arr = np.load(p, allow_pickle=True)
                        if len(arr) == 0:
                            continue
                        acts = []
                        for idx in range(len(arr)):
                            sample = arr[idx]
                            if isinstance(sample, dict) and 'actions' in sample:
                                acts.append(np.asarray(sample['actions']).reshape(-1))
                        if acts:
                            self.logger.info(f"action_hist collected from: {p}, trajs={len(acts)}")
                            return np.concatenate(acts)
                except Exception as e:
                    self.logger.warning(f"加载轨迹文件失败 {p}: {e}")
            return None

        if actions is None:
            actions = _try_collect_actions_from_npy()

        if actions is None or actions.size == 0:
            raise ValueError("action_hist requires 'actions' (or trajectories with actions) or a valid trajectory.npy")

        # Ensure integer discrete actions
        try:
            actions = actions.astype(int)
        except Exception:
            # Try rounding floats
            actions = np.rint(actions).astype(int)

        title = config.get('title', 'Action Histogram')
        xlabel = config.get('xlabel', 'Action')
        ylabel = config.get('ylabel', 'Count')

        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        unique, counts = np.unique(actions, axis=None, return_counts=True)
        order = np.argsort(unique)
        unique = unique[order]; counts = counts[order]

        # Use bar plot for discrete values
        ax.bar(unique, counts, width=float(config.get('bar_width', 0.8)))
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        try:
            ax.set_xticks(unique)
        except Exception:
            pass

        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        self.logger.info(f"Action histogram saved: {save_path}")
        return fig, save_path

    def _plot_radar(self, data: Dict, config: Dict) -> Tuple[plt.Figure, str]:
        """Plot radar (spider) chart for a single metrics dict"""
        self.logger.info("Plotting radar")

        metrics = data.get('metrics', None)
        if not isinstance(metrics, dict) or len(metrics) == 0:
            raise ValueError("radar plot requires data['metrics'] as a non-empty dict of metric_name -> value")

        # Prepare values and labels
        labels = list(metrics.keys())
        values = [float(metrics[k]) for k in labels]

        # Close the circle
        angles = np.linspace(0, 2 * np.pi, num=len(labels), endpoint=False).tolist()
        angles += angles[:1]
        values += values[:1]

        title = config.get('title', 'Metrics Radar')
        fig = plt.figure(figsize=config['figsize'], dpi=config['dpi'])
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2, linestyle='-', label='Metrics')
        ax.fill(angles, values, alpha=0.25)

        # Setup labels around the circle
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        # Auto scale radial limit
        try:
            vmax = float(np.nanmax(values)) if np.isfinite(values).all() else 1.0
            vmin = float(np.nanmin(values)) if np.isfinite(values).all() else 0.0
            if vmax == vmin:
                vmax = vmin + 1.0
            ax.set_ylim(vmin, vmax)
        except Exception:
            pass

        ax.set_title(title)
        if config.get('legend', False):
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

        save_path = os.path.join(config['save_dir'], f"{title.replace(' ', '_').lower()}.{config['save_format']}")
        plt.tight_layout()
        plt.savefig(save_path)
        self.logger.info(f"Radar plot saved: {save_path}")
        return fig, save_path