import os
import sys
import numpy as np
import unittest
import shutil
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入可视化服务
from afruits.core.services.visualization_service import VisualizationService
from afruits.utils.enhanced_test_runner import EnhancedTestRunner

class TestVisualization(unittest.TestCase):
    """
    可视化服务测试类
    
    测试可视化服务的各种可视化功能
    """
    
    def setUp(self):
        """测试前的准备工作"""
        # 设置测试保存目录
        self.test_save_dir = 'test_visualizations'
        
        # 配置可视化服务
        config = {
            'save_dir': self.test_save_dir,
            'dpi': 80  # 降低DPI以加快测试
        }
        
        # 初始化日志
        self.logger = logging.getLogger('test_visualization')
        self.logger.setLevel(logging.INFO)
        
        # 初始化可视化服务
        self.vis_service = VisualizationService(config=config, logger=self.logger)
    
    def tearDown(self):
        """测试后的清理工作"""
        # 不删除生成的可视化文件，以便查看测试结果
        pass
    
    #---------- 辅助方法 ----------#
    
    def create_line_data(self):
        """创建折线图测试数据"""
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.sin(x) * np.cos(x)
        
        return {
            'x': x,
            'y': {
                'Sine': y1,
                'Cosine': y2,
                'Sine*Cosine': y3
            }
        }
    
    def create_bar_data(self):
        """创建柱状图测试数据"""
        categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        values1 = np.random.rand(5) * 10
        values2 = np.random.rand(5) * 10
        
        return {
            'x': categories,
            'y': {
                'Group 1': values1,
                'Group 2': values2
            }
        }
    
    def create_scatter_data(self):
        """创建散点图测试数据"""
        n_samples = 100
        x, y = np.random.rand(2, n_samples) * 10
        colors = np.random.rand(n_samples)
        sizes = np.random.rand(n_samples) * 100 + 20
        
        return {
            'x': x,
            'y': y,
            'colors': colors,
            'sizes': sizes,
            'labels': [f'Point {i}' for i in range(10)] + [None] * (n_samples - 10)
        }
    
    def create_heatmap_data(self):
        """创建热力图测试数据"""
        matrix = np.random.rand(8, 10)
        xlabels = [f'Feature {i}' for i in range(10)]
        ylabels = [f'Sample {i}' for i in range(8)]
        
        return {
            'matrix': matrix,
            'xlabels': xlabels,
            'ylabels': ylabels
        }
    
    def create_3d_data(self):
        """创建3D图测试数据"""
        n_samples = 100
        x = np.random.rand(n_samples) * 10
        y = np.random.rand(n_samples) * 10
        z = np.sin(x) * np.cos(y)
        colors = z
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'colors': colors
        }
    
    def create_trajectory_data(self):
        """创建轨迹图测试数据"""
        # 创建两条轨迹
        t1 = np.linspace(0, 2*np.pi, 100)
        x1 = np.cos(t1) * t1/3
        y1 = np.sin(t1) * t1/3
        
        t2 = np.linspace(0, 4*np.pi, 100)
        x2 = np.cos(t2) * (4 - t2/2)
        y2 = np.sin(t2) * (4 - t2/2)
        
        return {
            'trajectories': {
                'Trajectory 1': list(zip(x1, y1)),
                'Trajectory 2': list(zip(x2, y2))
            }
        }
    
    def create_distribution_data(self):
        """创建分布图测试数据"""
        # 创建三个不同的分布
        dist1 = np.random.normal(0, 1, 1000)
        dist2 = np.random.normal(3, 1.5, 1000)
        dist3 = np.random.exponential(2, 1000)
        
        return {
            'distributions': {
                'Normal(0,1)': dist1,
                'Normal(3,1.5)': dist2,
                'Exponential(2)': dist3
            }
        }
    
    def create_comparison_data(self):
        """创建比较图测试数据"""
        models = ['Model A', 'Model B', 'Model C']
        metrics = {
            'Accuracy': {
                'Model A': 0.85,
                'Model B': 0.82,
                'Model C': 0.88
            },
            'Recall': {
                'Model A': 0.76,
                'Model B': 0.81,
                'Model C': 0.79
            },
            'F1 Score': {
                'Model A': 0.80,
                'Model B': 0.81,
                'Model C': 0.83
            }
        }
        
        return {
            'models': models,
            'metrics': metrics
        }
    
    def create_embedding_data(self):
        """创建嵌入图测试数据"""
        # 创建聚类数据
        n_samples = 300
        n_features = 10
        n_clusters = 3
        
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
        
        return {
            'features': X,
            'labels': y
        }
    
    #---------- 测试方法 ----------#
    
    def test_line_plot(self):
        """测试折线图功能"""
        print("\nTesting Line Plot")
        
        # 创建测试数据
        data = self.create_line_data()
        
        # 配置可视化
        vis_config = {
            'type': 'line',
            'title': 'Test Line Plot',
            'xlabel': 'X Axis',
            'ylabel': 'Y Axis',
            'grid': True
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Line plot saved: {save_path}")
    
    def test_bar_plot(self):
        """测试柱状图功能"""
        print("\nTesting Bar Plot")
        
        # 创建测试数据
        data = self.create_bar_data()
        
        # 配置可视化
        vis_config = {
            'type': 'bar',
            'title': 'Test Bar Plot',
            'xlabel': 'Category',
            'ylabel': 'Value'
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Bar plot saved: {save_path}")
    
    def test_scatter_plot(self):
        """测试散点图功能"""
        print("\nTesting Scatter Plot")
        
        # 创建测试数据
        data = self.create_scatter_data()
        
        # 配置可视化
        vis_config = {
            'type': 'scatter',
            'title': 'Test Scatter Plot',
            'xlabel': 'X Axis',
            'ylabel': 'Y Axis',
            'color_label': 'Color Value'
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Scatter plot saved: {save_path}")
    
    def test_heatmap_plot(self):
        """测试热力图功能"""
        print("\nTesting Heatmap")
        
        # 创建测试数据
        data = self.create_heatmap_data()
        
        # 配置可视化
        vis_config = {
            'type': 'heatmap',
            'title': 'Test Heatmap',
            'annot': True,
            'cmap': 'coolwarm'
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Heatmap saved: {save_path}")
    
    def test_3d_plot(self):
        """测试3D图功能"""
        print("\nTesting 3D Plot")
        
        # 创建测试数据
        data = self.create_3d_data()
        
        # 配置可视化
        vis_config = {
            'type': '3d',
            'title': 'Test 3D Plot',
            'xlabel': 'X Axis',
            'ylabel': 'Y Axis',
            'zlabel': 'Z Axis',
            'plot_type': 'scatter'
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"3D plot saved: {save_path}")
    
    def test_trajectory_plot(self):
        """测试轨迹图功能"""
        print("\nTesting Trajectory Plot")
        
        # 创建测试数据
        data = self.create_trajectory_data()
        
        # 配置可视化
        vis_config = {
            'type': 'trajectory',
            'title': 'Test Trajectory Plot',
            'xlabel': 'X Coordinate',
            'ylabel': 'Y Coordinate',
            'legend': True,
            'grid': True
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Trajectory plot saved: {save_path}")
    
    def test_distribution_plot(self):
        """测试分布图功能"""
        print("\nTesting Distribution Plot")
        
        # 创建测试数据
        data = self.create_distribution_data()
        
        # 配置可视化
        vis_config = {
            'type': 'distribution',
            'title': 'Test Distribution Plot',
            'xlabel': 'Value',
            'ylabel': 'Frequency'
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Distribution plot saved: {save_path}")
    
    def test_comparison_plot(self):
        """测试比较图功能"""
        print("\nTesting Comparison Plot")
        
        # 创建测试数据
        data = self.create_comparison_data()
        
        # 配置可视化
        vis_config = {
            'type': 'comparison',
            'title': 'Model Performance Comparison'
        }
        
        # 执行可视化
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Comparison plot saved: {save_path}")
    
    def test_embedding_plot(self):
        """测试嵌入图功能"""
        print("\nTesting Embedding Plot")
        
        # 创建测试数据
        data = self.create_embedding_data()
        
        # 配置可视化 - 使用PCA
        vis_config_pca = {
            'type': 'embedding',
            'title': 'Test PCA Embedding',
            'method': 'pca'
        }
        
        # 执行可视化 - PCA
        result_pca = self.vis_service.visualize(data, vis_config_pca)
        
        # 验证结果 - PCA
        self.assertIn('figures', result_pca)
        self.assertIn('save_paths', result_pca)
        self.assertTrue(len(result_pca['figures']) > 0)
        self.assertTrue(len(result_pca['save_paths']) > 0)
        
        # 验证文件是否生成 - PCA
        save_path_pca = result_pca['save_paths'][0]
        self.assertTrue(os.path.exists(save_path_pca))
        
        print(f"PCA embedding plot saved: {save_path_pca}")
        
        # 配置可视化 - 使用t-SNE
        vis_config_tsne = {
            'type': 'embedding',
            'title': 'Test t-SNE Embedding',
            'method': 'tsne'
        }
        
        # 执行可视化 - t-SNE
        result_tsne = self.vis_service.visualize(data, vis_config_tsne)
        
        # 验证结果 - t-SNE
        self.assertIn('figures', result_tsne)
        self.assertIn('save_paths', result_tsne)
        self.assertTrue(len(result_tsne['figures']) > 0)
        self.assertTrue(len(result_tsne['save_paths']) > 0)
        
        # 验证文件是否生成 - t-SNE
        save_path_tsne = result_tsne['save_paths'][0]
        self.assertTrue(os.path.exists(save_path_tsne))
        
        print(f"t-SNE embedding plot saved: {save_path_tsne}")
    
    def test_invalid_type(self):
        """测试无效的可视化类型"""
        print("\nTesting Invalid Visualization Type")
        
        # 创建测试数据
        data = self.create_line_data()
        
        # 配置可视化
        vis_config = {
            'type': 'invalid_type',  # 无效的类型
            'title': 'Test Invalid Type'
        }
        
        # 执行可视化 - 应该默认使用折线图
        result = self.vis_service.visualize(data, vis_config)
        
        # 验证结果
        self.assertIn('figures', result)
        self.assertIn('save_paths', result)
        self.assertTrue(len(result['figures']) > 0)
        self.assertTrue(len(result['save_paths']) > 0)
        
        # 验证文件是否生成
        save_path = result['save_paths'][0]
        self.assertTrue(os.path.exists(save_path))
        
        print(f"Default line plot saved: {save_path}")


if __name__ == "__main__":
    # 使用增强的测试运行器
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVisualization)
    runner = EnhancedTestRunner(verbosity=2)
    runner.run(test_suite)