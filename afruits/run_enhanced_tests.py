import os
import sys
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入增强的测试运行器
from afruits.utils.enhanced_test_runner import run_tests_with_enhanced_reporting

def run_all_tests():
    """运行所有测试，使用增强的错误报告"""
    print("开始运行测试，使用增强的错误报告...")
    
    # 发现所有测试
    test_suite = unittest.defaultTestLoader.discover('afruits/tests')
    
    # 使用增强的测试运行器运行测试
    success = run_tests_with_enhanced_reporting(test_suite)
    
    return success

def run_specific_test(test_module_name):
    """运行指定的测试模块，使用增强的错误报告"""
    print(f"开始运行测试模块 {test_module_name}，使用增强的错误报告...")
    
    # 导入指定的测试模块
    try:
        test_module = __import__(f"afruits.tests.{test_module_name}", fromlist=['*'])
    except ImportError:
        print(f"错误：找不到测试模块 {test_module_name}")
        return False
    
    # 加载测试模块中的所有测试
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
    
    # 使用增强的测试运行器运行测试
    success = run_tests_with_enhanced_reporting(test_suite)
    
    return success

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果提供了测试模块名称，则运行指定的测试
        test_module_name = sys.argv[1]
        success = run_specific_test(test_module_name)
    else:
        # 否则运行所有测试
        success = run_all_tests()
    
    # 设置退出码
    sys.exit(0 if success else 1)