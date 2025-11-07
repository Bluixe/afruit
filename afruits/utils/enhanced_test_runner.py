import sys
import traceback
import unittest
from unittest.runner import TextTestResult, TextTestRunner

class EnhancedTestResult(TextTestResult):
    """增强的测试结果类，提供更详细的错误报告，包括文件名和行号"""
    
    def addError(self, test, err):
        """添加错误信息，增强错误报告"""
        super().addError(test, err)
        self._print_error_details("ERROR", test, err)
    
    def addFailure(self, test, err):
        """添加失败信息，增强错误报告"""
        super().addFailure(test, err)
        self._print_error_details("FAIL", test, err)
    
    def _print_error_details(self, flavor, test, err):
        """打印详细的错误信息，包括文件名和行号"""
        self.stream.writeln(self.separator1)
        self.stream.writeln(f"{flavor}: {self.getDescription(test)}")
        self.stream.writeln(self.separator2)
        
        # 获取完整的错误追踪信息
        tb_lines = traceback.format_exception(*err)
        
        # 打印错误追踪信息，突出显示文件名和行号
        for line in tb_lines:
            # 高亮显示文件名和行号
            if line.lstrip().startswith("File "):
                parts = line.split(", ")
                if len(parts) >= 2:
                    file_part = parts[0].strip()
                    line_part = parts[1].strip()
                    self.stream.write(f"\033[1;36m{file_part}, {line_part}\033[0m\n")
                    if len(parts) > 2:
                        self.stream.write("".join(parts[2:]) + "\n")
                else:
                    self.stream.write(line)
            else:
                self.stream.write(line)

class EnhancedTestRunner(TextTestRunner):
    """增强的测试运行器，使用增强的测试结果类"""
    
    def _makeResult(self):
        """创建增强的测试结果对象"""
        return EnhancedTestResult(self.stream, self.descriptions, self.verbosity)

def run_tests_with_enhanced_reporting(test_suite=None):
    """使用增强的错误报告运行测试"""
    if test_suite is None:
        # 如果没有提供测试套件，则发现所有测试
        test_suite = unittest.defaultTestLoader.discover('tests')
    
    # 使用增强的测试运行器
    runner = EnhancedTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印测试结果摘要
    print("\n测试结果摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    return result.wasSuccessful()