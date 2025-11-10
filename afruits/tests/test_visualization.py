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
                '正弦': y1,
                '余弦': y2,
                '正余弦乘积': y3
            }
        }
    
    def create_bar_data(self):
        """创建柱状图测试数据"""
        categories = ['类别A', '类别B', '类别C', '类别D', '类别E']
        values1 = np.random.rand(5) * 10
        values2 = np.random.rand(5) * 10
        
        return {
            'x': categories,
            'y': {
                '组1': values1,
                '组2': values2
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
            'labels': [f'点{i}' for i in range(10)] + [None] * (n_samples - 10)
        }
    
    def create_heatmap_data(self):
        """创建热力图测试数据"""
        matrix = np.random.rand(8, 10)
        xlabels = [f'特征{i}' for i in range(10)]
        ylabels = [f'样本{i}' for i in range(8)]
        
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
                '轨迹1': list(zip(x1, y1)),
                '轨迹2': list(zip(x2, y2))
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
                '正态分布(0,1)': dist1,
                '正态分布(3,1.5)': dist2,
                '指数分布(2)': dist3
            }
        }
    
    def create_comparison_data(self):
        """创建比较图测试数据"""
        models = ['模型A', '模型B', '模型C']
        metrics = {
            '准确率': {
                '模型A': 0.85,
                '模型B': 0.82,
                '模型C': 0.88
            },
            '召回率': {
                '模型A': 0.76,
                '模型B': 0.81,
                '模型C': 0.79
            },
            'F1分数': {
                '模型A': 0.80,
                '模型B': 0.81,
                '模型C': 0.83
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
        print("\n测试折线图功能")
        
        # 创建测试数据
        data = self.create_line_data()
        
        # 配置可视化
        vis_config = {
            'type': 'line',
            'title': '测试折线图',
            'xlabel': 'X轴',
            'ylabel': 'Y轴',
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
        
        print(f"折线图已保存: {save_path}")
    
    def test_bar_plot(self):
        """测试柱状图功能"""
        print("\n测试柱状图功能")
        
        # 创建测试数据
        data = self.create_bar_data()
        
        # 配置可视化
        vis_config = {
            'type': 'bar',
            'title': '测试柱状图',
            'xlabel': '类别',
            'ylabel': '数值'
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
        
        print(f"柱状图已保存: {save_path}")
    
    def test_scatter_plot(self):
        """测试散点图功能"""
        print("\n测试散点图功能")
        
        # 创建测试数据
        data = self.create_scatter_data()
        
        # 配置可视化
        vis_config = {
            'type': 'scatter',
            'title': '测试散点图',
            'xlabel': 'X轴',
            'ylabel': 'Y轴',
            'color_label': '颜色值'
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
        
        print(f"散点图已保存: {save_path}")
    
    def test_heatmap_plot(self):
        """测试热力图功能"""
        print("\n测试热力图功能")
        
        # 创建测试数据
        data = self.create_heatmap_data()
        
        # 配置可视化
        vis_config = {
            'type': 'heatmap',
            'title': '测试热力图',
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
        
        print(f"热力图已保存: {save_path}")
    
    def test_3d_plot(self):
        """测试3D图功能"""
        print("\n测试3D图功能")
        
        # 创建测试数据
        data = self.create_3d_data()
        
        # 配置可视化
        vis_config = {
            'type': '3d',
            'title': '测试3D图',
            'xlabel': 'X轴',
            'ylabel': 'Y轴',
            'zlabel': 'Z轴',
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
        
        print(f"3D图已保存: {save_path}")
    
    def test_trajectory_plot(self):
        """测试轨迹图功能"""
        print("\n测试轨迹图功能")
        
        # 创建测试数据
        data = self.create_trajectory_data()
        
        # 配置可视化
        vis_config = {
            'type': 'trajectory',
            'title': '测试轨迹图',
            'xlabel': 'X坐标',
            'ylabel': 'Y坐标',
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
        
        print(f"轨迹图已保存: {save_path}")
    
    def test_distribution_plot(self):
        """测试分布图功能"""
        print("\n测试分布图功能")
        
        # 创建测试数据
        data = self.create_distribution_data()
        
        # 配置可视化
        vis_config = {
            'type': 'distribution',
            'title': '测试分布图',
            'xlabel': '数值',
            'ylabel': '频率'
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
        
        print(f"分布图已保存: {save_path}")
    
    def test_comparison_plot(self):
        """测试比较图功能"""
        print("\n测试比较图功能")
        
        # 创建测试数据
        data = self.create_comparison_data()
        
        # 配置可视化
        vis_config = {
            'type': 'comparison',
            'title': '模型性能比较'
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
        
        print(f"比较图已保存: {save_path}")
    
    def test_embedding_plot(self):
        """测试嵌入图功能"""
        print("\n测试嵌入图功能")
        
        # 创建测试数据
        data = self.create_embedding_data()
        
        # 配置可视化 - 使用PCA
        vis_config_pca = {
            'type': 'embedding',
            'title': '测试PCA嵌入图',
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
        
        print(f"PCA嵌入图已保存: {save_path_pca}")
        
        # 配置可视化 - 使用t-SNE
        vis_config_tsne = {
            'type': 'embedding',
            'title': '测试t-SNE嵌入图',
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
        
        print(f"t-SNE嵌入图已保存: {save_path_tsne}")
    
    def test_invalid_type(self):
        """测试无效的可视化类型"""
        print("\n测试无效的可视化类型")
        
        # 创建测试数据
        data = self.create_line_data()
        
        # 配置可视化
        vis_config = {
            'type': 'invalid_type',  # 无效的类型
            'title': '测试无效类型'
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
        
        print(f"默认折线图已保存: {save_path}")


if __name__ == "__main__":
    # 使用增强的测试运行器
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVisualization)
    runner = EnhancedTestRunner(verbosity=2)
    runner.run(test_suite)