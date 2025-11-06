import os
import sys
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入测试模块
from tests.test_algorithm_framework import TestAlgorithmFramework
from tests.test_api_basic import TestAPIBasic

def run_tests():
    """运行所有测试"""
    print("开始运行算法框架测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加基本API测试用例
    test_suite.addTest(TestAPIBasic('test_api_initialization'))
    test_suite.addTest(TestAPIBasic('test_data_preprocessing'))
    test_suite.addTest(TestAPIBasic('test_trajectory_preprocessing'))
    
    # 添加算法框架测试用例
    test_suite.addTest(TestAlgorithmFramework('test_game_modeling'))
    test_suite.addTest(TestAlgorithmFramework('test_imitation_learning'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印测试结果摘要
    print("\n测试结果摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)