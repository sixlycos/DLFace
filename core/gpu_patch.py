"""
GPU补丁模块 - 修复GPU检测和使用问题

修复OpenCV的getBuildInformation问题
确保PyTorch和TensorFlow正确使用GPU
"""

import os
import logging
import sys

# 导入日志配置
from .logger_config import setup_logger
logger = setup_logger()

# 设置环境变量
def setup_gpu_env():
    """检查和设置GPU环境变量"""
    # 获取环境变量中的GPU设置
    use_gpu = os.environ.get("DL_FACE_USE_GPU", "0") == "1"
    
    if use_gpu:
        logger.info("根据环境变量配置GPU加速...")
        # 确保CUDA_VISIBLE_DEVICES正确设置
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.info("已设置CUDA_VISIBLE_DEVICES=0")
        
        # 设置TensorFlow内存增长
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        
        # 设置PyTorch内存分配
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    else:
        logger.info("根据环境变量禁用GPU加速...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    return use_gpu

# 检查CUDA是否可用的替代方法
def check_cuda_availability():
    """检查CUDA是否可用 - 替代cv2.getBuildInformation问题"""
    try:
        import cv2
        # 首先尝试经典方法
        try:
            # 安全地检查getBuildInformation方法是否存在
            if hasattr(cv2, 'getBuildInformation'):
                cv_build_info = cv2.getBuildInformation()
                cuda_support = "CUDA" in cv_build_info and "YES" in cv_build_info[cv_build_info.find("CUDA"):]
                if cuda_support:
                    logger.info("OpenCV已编译支持CUDA加速")
                    return True
            else:
                logger.warning("OpenCV没有getBuildInformation方法")
        except Exception as e:
            logger.warning(f"使用OpenCV getBuildInformation方法时出错: {e}")
        
        # 使用替代检测方法
        # 尝试创建CUDA-enabled函数
        try:
            if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
                if cuda_device_count > 0:
                    logger.info(f"检测到{cuda_device_count}个支持CUDA的设备")
                    return True
        except Exception as e:
            logger.warning(f"OpenCV CUDA设备检测失败: {e}")
        
        # 尝试PyTorch检测
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"PyTorch检测到支持CUDA: {torch.cuda.get_device_name(0)}")
                return True
        except Exception as e:
            logger.warning(f"PyTorch CUDA检测失败: {e}")
            
        # 尝试TensorFlow检测  
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"TensorFlow检测到{len(gpus)}个GPU")
                return True
        except Exception as e:
            logger.warning(f"TensorFlow GPU检测失败: {e}")
        
        # 尝试读取环境变量
        if os.environ.get("DL_FACE_USE_GPU", "0") == "1":
            logger.info("环境变量DL_FACE_USE_GPU=1，强制启用GPU")
            return True
            
        logger.warning("未检测到支持CUDA的GPU或环境配置")
        return False
    except Exception as e:
        logger.error(f"CUDA检测过程失败: {e}")
        return False

# 初始化和应用GPU设置
def apply_gpu_patches():
    """应用所有GPU相关补丁和优化设置"""
    # 设置环境变量
    use_gpu = setup_gpu_env()
    
    # 如果需要使用GPU，进行额外配置
    if use_gpu:
        # 配置PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"PyTorch检测到{device_count}个GPU: {device_name}")
                
                # 设置默认设备
                if not hasattr(torch, "_patched_cuda"):
                    original_device = torch.device
                    def patched_device(device_type, index=None):
                        if device_type == "cuda" and index is None:
                            return original_device("cuda", 0)
                        return original_device(device_type, index)
                    torch.device = patched_device
                    torch._patched_cuda = True
                    logger.info("已为PyTorch应用CUDA设备补丁")
            else:
                logger.warning("PyTorch未检测到可用的CUDA设备")
        except Exception as e:
            logger.error(f"配置PyTorch CUDA时出错: {e}")
        
        # 配置TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        logger.warning(f"无法为GPU {gpu} 设置内存增长: {str(e)}")
                logger.info(f"已配置TensorFlow使用{len(gpus)}个GPU")
            else:
                logger.warning("TensorFlow未检测到GPU设备")
        except Exception as e:
            logger.error(f"配置TensorFlow GPU时出错: {e}")
    
    # 返回GPU可用性
    return use_gpu and check_cuda_availability()

# 如果直接运行此文件，执行检测
if __name__ == "__main__":
    gpu_available = apply_gpu_patches()
    print(f"GPU可用性检测结果: {'可用' if gpu_available else '不可用'}") 