"""
日志配置模块 - 提供统一的日志配置
确保所有模块使用相同的日志格式和编码设置
"""

import logging
import sys
import os

def setup_logger():
    """设置全局日志配置，使用UTF-8编码"""
    # 确保输出使用UTF-8编码
    if sys.stdout.encoding != 'utf-8':
        # 尝试修复Windows控制台的编码问题
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        elif hasattr(sys, 'stdout'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(stream=sys.stdout)
        ]
    )
    
    # 设置日志级别
    logging.getLogger().setLevel(logging.INFO)
    
    # 获取环境变量中的GPU设置，默认为启用GPU
    use_gpu = os.environ.get("DL_FACE_USE_GPU", "1") == "1"
    
    # 记录GPU状态
    if use_gpu:
        logging.info("GPU加速已启用 (DL_FACE_USE_GPU=1)")
    else:
        logging.info("GPU加速未启用 (DL_FACE_USE_GPU=0)")
        
    return logging.getLogger() 