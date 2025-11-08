import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afruits.utils.VAETrajGenerator import VAETrajGenerator

def test_vae_with_image_state():
    """
    测试VAE轨迹生成器对图像状态的支持
    """
    print("测试VAE轨迹生成器对图像状态的支持...")
    
    # 创建模拟的图像状态数据
    batch_size = 8
    seq_length = 100
    channels = 3
    height = 32
    width = 32
    action_dim = 4
    
    # 创建随机图像状态序列 [batch_size, seq_length, channels, height, width]
    states = torch.rand(batch_size, seq_length, channels, height, width)
    
    # 创建随机动作序列 [batch_size, seq_length, action_dim]
    actions = torch.rand(batch_size, seq_length, action_dim)
    
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(states, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建VAE轨迹生成器
    vae = VAETrajGenerator(
        latent_dim=32,
        seq_length=seq_length,
        kl_weight=0.001,
        dropout=0.2,
        im_embd=64
    )
    
    # 构建模型
    input_dim = {
        'state_dim': (channels, height, width),  # 图像状态维度
        'action_dim': action_dim,
        'total_dim': action_dim + 64  # action_dim + im_embd
    }
    
    # 构建模型
    encoder, decoder = vae.build_model(input_dim)
    
    # 验证模型是否正确构建
    print(f"是否为图像状态: {vae.is_image_state}")
    print(f"图像形状: {vae.image_shape}")
    print(f"状态维度: {vae.state_dim}")
    print(f"动作维度: {vae.action_dim}")
    
    # 测试前向传播
    print("\n测试单批次前向传播...")
    for batch in dataloader:
        states_batch, actions_batch = batch
        
        # 前向传播
        mu, logvar = encoder(states_batch, actions_batch)
        z = vae.reparameterize(mu, logvar)
        reconstructed_states, reconstructed_actions = decoder(z)
        
        # 打印形状
        print(f"输入状态形状: {states_batch.shape}")
        print(f"输入动作形状: {actions_batch.shape}")
        print(f"潜在向量形状: {z.shape}")
        print(f"重构状态形状: {reconstructed_states.shape}")
        print(f"重构动作形状: {reconstructed_actions.shape}")
        
        # 只测试一个批次
        break
    
    # 测试训练过程
    print("\n测试训练过程（仅1个epoch）...")
    history = vae.train(dataloader, epochs=1)
    
    # 测试生成
    print("\n测试轨迹生成...")
    generated = vae.generate(num_samples=2)
    
    print(f"生成状态形状: {generated['state'].shape}")
    print(f"生成动作形状: {generated['action'].shape}")
    
    print("\n测试完成!")
    return True

if __name__ == "__main__":
    test_vae_with_image_state()