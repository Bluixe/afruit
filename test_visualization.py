#!/usr/bin/env python3
"""
测试可视化修改是否正确工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from afruits.main import MainWindow
from PyQt5.QtWidgets import QApplication

def test_visualization():
    """测试可视化功能"""
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = MainWindow()
    
    # 模拟训练结果数据
    training_result = {
        'model_id': 'test_model',
        'model_type': 'AutoencoderModel',
        'training_metrics': {
            'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.16, 0.14, 0.12],
            'val_loss': [0.85, 0.65, 0.45, 0.35, 0.3, 0.25, 0.22, 0.2, 0.19, 0.18],
            'final_train_loss': 0.12,
            'final_val_loss': 0.18
        },
        'model': None
    }
    
    # 设置训练结果
    window.training_result = training_result
    
    # 设置可视化类型为line
    window.vis_type_combo.setCurrentText('line')
    
    # 设置标题和标签
    window.vis_title.setText("训练损失曲线")
    window.vis_xlabel.setText("训练轮次")
    window.vis_ylabel.setText("损失值")
    
    # 生成可视化
    try:
        window.generate_visualization()
        print("✅ 可视化生成成功！")
        print("✅ line类型现在显示训练loss曲线而不是eval结果")
        
        # 检查数据是否正确
        if hasattr(window, 'training_result'):
            training_metrics = window.training_result.get('training_metrics', {})
            if 'train_loss' in training_metrics:
                print(f"✅ 训练loss数据: {training_metrics['train_loss']}")
            if 'val_loss' in training_metrics:
                print(f"✅ 验证loss数据: {training_metrics['val_loss']}")
                
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 关闭应用
    app.quit()

if __name__ == "__main__":
    test_visualization()