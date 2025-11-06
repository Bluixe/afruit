import numpy as np
import os
import torch
import sys
import unittest
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from afruits.utils.TrajectoryPreprocessor import TrajectoryPreprocessor
from afruits.utils.VAETrajGenerator import VAETrajGenerator

class TestVAEDataEnhance(unittest.TestCase):
    """测试VAE数据增强功能"""
    
    def setUp(self):
        """准备测试环境"""
        # 创建临时目录用于保存模型
        self.save_path = Path("./test_models")
        self.save_path.mkdir(exist_ok=True)
        
        # 创建轨迹预处理器
        self.preprocessor = TrajectoryPreprocessor()
        
        # 创建模拟轨迹数据
        self.seq_length = 100
        self.feature_dim = 6  # 状态+动作维度
        self.num_trajectories = 10
        
        # 生成随机轨迹
        self.trajectories = [
            np.random.randn(self.seq_length, self.feature_dim) 
            for _ in range(self.num_trajectories)
        ]
        
        # 创建并训练VAE模型
        self._train_vae_model()
    
    def _train_vae_model(self):
        """训练并保存VAE模型"""
        # 创建VAE生成器
        vae = VAETrajGenerator(
            latent_dim=32,
            seq_length=self.seq_length,
            kl_weight=0.001
        )
        
        # 准备数据
        trajectories_array = np.array(self.trajectories)
        
        # 保存为临时文件
        temp_data_path = self.save_path / "temp_data.npz"
        np.savez(temp_data_path, trajectories=trajectories_array)
        
        # 加载数据
        data_info = vae.load_dataset(str(temp_data_path), batch_size=4)
        
        # 构建模型
        vae.build_model(self.feature_dim)
        
        # 简单训练几个epoch
        vae.train(data_info['dataloader'], epochs=2)
        
        # 保存模型
        model_path = self.save_path / "vae_model.pt"
        torch.save({
            'encoder_state_dict': vae.encoder.state_dict(),
            'decoder_state_dict': vae.decoder.state_dict(),
        }, model_path)
        
        print(f"VAE模型已保存到: {model_path}")
    
    def test_latent_data_enhance(self):
        """测试隐空间数据增强方法"""
        # 使用data_enhance方法进行隐空间数据增强
        data_num = 5
        enhanced_trajectories = self.preprocessor.data_enhance(
            self.trajectories,
            data_num=data_num,
            method="latent",
            save_path=str(self.save_path)
        )
        
        # 验证结果
        # 1. 检查增强后的轨迹数量
        expected_count = len(self.trajectories) + data_num
        self.assertEqual(len(enhanced_trajectories), expected_count, 
                         f"增强后轨迹数量应为{expected_count}，实际为{len(enhanced_trajectories)}")
        
        # 2. 检查生成的轨迹形状
        for i in range(len(self.trajectories), len(enhanced_trajectories)):
            traj = enhanced_trajectories[i]
            self.assertEqual(traj.shape, (self.seq_length, self.feature_dim),
                            f"生成轨迹形状应为{(self.seq_length, self.feature_dim)}，实际为{traj.shape}")
        
        print("隐空间数据增强测试通过!")
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时文件
        import shutil
        if self.save_path.exists():
            shutil.rmtree(self.save_path)

if __name__ == "__main__":
    unittest.main()