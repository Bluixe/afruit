import torch
import torch.nn as nn
import numpy as np
from FineTuneManager import FineTuneManager

# 创建一个简单的测试模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建一个简单的模型
    input_dim = 10
    output_dim = 1
    model = SimpleModel(input_dim=input_dim, output_dim=output_dim)
    
    # 创建一些合成数据
    num_samples = 1000
    X = np.random.randn(num_samples, input_dim)
    # 创建一个简单的线性关系加上一些噪声
    w = np.random.randn(input_dim, output_dim)
    y = np.dot(X, w) + 0.1 * np.random.randn(num_samples, output_dim)
    
    # 将数据打包成字典
    data = {
        'x': X,
        'y': y
    }
    
    # 创建FineTuneManager实例
    fine_tune_manager = FineTuneManager(
        base_model=model,
        trainable_layers=["fc2", "fc3"],  # 只微调fc2和fc3层
        freeze_strategy="selective",
        optimizer_config={
            'lr': 0.01,
            'weight_decay': 1e-4,
            'type': 'Adam'
        },
        regularization_mode="adaptive"
    )
    
    # 设置模型
    fine_tune_manager.setup_model()
    
    # 准备数据
    train_loader, val_loader = fine_tune_manager.prepare_data(
        raw_data=data,
        augment=True,
        batch_size=32,
        val_split=0.2
    )
    
    # 执行微调
    training_report = fine_tune_manager.execute_finetuning(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=50,
        patience=5
    )
    
    print("\n训练报告:")
    for key, value in training_report.items():
        print(f"  {key}: {value}")
    
    # 创建测试数据
    num_test_samples = 200
    X_test = np.random.randn(num_test_samples, input_dim)
    y_test = np.dot(X_test, w) + 0.1 * np.random.randn(num_test_samples, output_dim)
    
    test_data = {
        'x': X_test,
        'y': y_test
    }
    
    # 准备测试数据
    test_loader, _ = fine_tune_manager.prepare_data(
        raw_data=test_data,
        augment=False,
        batch_size=32,
        val_split=0.0
    )
    
    # 评估模型
    metrics = fine_tune_manager.evaluate_model(
        test_loader=test_loader,
        metrics=['mse', 'mae', 'rmse', 'r2']
    )
    
    print("\n测试结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()