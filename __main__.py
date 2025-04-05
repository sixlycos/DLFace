"""
主入口模块 - 允许通过 python -m 命令运行应用程序
"""

import os
import sys

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入并运行启动脚本
from run import main

if __name__ == "__main__":
    main() 