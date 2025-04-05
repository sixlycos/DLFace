#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DLface项目清理脚本 - 简化版

该脚本用于删除项目中不必要的临时文件、冗余下载脚本和文档文件，
保留核心功能所需的文件。
"""

import os
import sys
import shutil
import glob

# 定义要保留的核心文件
CORE_FILES = [
    '__main__.py',              # 模块入口点
    'run.py',                   # 主启动脚本
    'auto_gpu_detection.py',    # GPU检测工具
    'download_model.py',        # 整合版模型下载工具
    'requirements.txt',         # 项目依赖
    'README.md',                # 项目说明
    'QUICKSTART.md',            # 快速入门指南
]

# 定义要保留的核心目录
CORE_DIRS = [
    'app',                      # 应用程序代码
    'core',                     # 核心功能模块
    'data',                     # 数据和模型目录
    'examples',                 # 示例代码
    'tests',                    # 测试代码
    'temp',                     # 临时文件目录
    'models',                   # 模型文件目录
]

# 定义要删除的冗余文件列表
REDUNDANT_FILES = [
    # 旧版模型下载脚本
    'download_bisenet.py',
    'download_dnn_models.py',
    'download_mediapipe_models.py',
    'download_opencv_models.py',
    
    # 临时修复和文档文件
    'MODEL_FIX_README.md',
    'MODEL_FIX_SOLUTION.md',
    'CLEANUP_README.md',
    
    # 旧的初始化和清理脚本
    'init_system.py',
    'cleanup.py',
    
    # 批处理文件
    '*.bat',
    '*.ps1',
    
    # 日志文件
    '*.log',
]

def print_header(text):
    """打印带格式的标题"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def confirm_action(message):
    """确认用户操作"""
    response = input(f"{message} (y/n): ").strip().lower()
    return response == 'y'

def clean_project():
    """执行项目清理过程"""
    print_header("开始清理DLface项目")
    
    # 确认删除
    if not confirm_action("此操作将删除项目中的冗余文件。确认继续吗？"):
        print("已取消操作")
        return
    
    # 删除冗余文件
    deleted_files = []
    for pattern in REDUNDANT_FILES:
        if '*' in pattern:
            # 使用glob模式匹配
            matching_files = glob.glob(pattern)
            for file in matching_files:
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                        deleted_files.append(file)
                except Exception as e:
                    print(f"删除 {file} 时出错: {e}")
        else:
            # 直接匹配文件名
            if os.path.exists(pattern) and os.path.isfile(pattern):
                try:
                    os.remove(pattern)
                    deleted_files.append(pattern)
                except Exception as e:
                    print(f"删除 {pattern} 时出错: {e}")
    
    # 清理Python缓存文件
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                try:
                    cache_dir = os.path.join(root, dir_name)
                    shutil.rmtree(cache_dir)
                    print(f"已删除Python缓存: {cache_dir}")
                except Exception as e:
                    print(f"删除 {cache_dir} 时出错: {e}")
    
    # 显示删除的文件
    if deleted_files:
        print_header("已删除以下文件")
        for file in deleted_files:
            print(f" - {file}")
    else:
        print("没有找到冗余文件需要删除")
    
    print_header("清理完成")
    print("项目中现在只保留了核心功能所需的文件和目录。")
    print("\n核心文件:")
    for file in CORE_FILES:
        if os.path.exists(file):
            print(f" - {file}")
    
    print("\n核心目录:")
    for directory in CORE_DIRS:
        if os.path.exists(directory):
            print(f" - {directory}/")
    
    # 确认删除清理脚本本身
    if confirm_action("\n是否删除清理脚本本身？"):
        try:
            os.remove(__file__)
            print(f"已删除清理脚本: {__file__}")
        except Exception as e:
            print(f"删除清理脚本时出错: {e}")

if __name__ == "__main__":
    clean_project() 