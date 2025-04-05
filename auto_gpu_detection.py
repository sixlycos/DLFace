"""
自动GPU检测脚本

这个脚本会在启动时自动检测系统是否有可用的GPU，并设置相应的环境变量：
1. 如果检测到GPU并可用，设置 DL_FACE_USE_GPU=1
2. 如果没有检测到GPU或GPU不可用，设置 DL_FACE_USE_GPU=0
"""

import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

def detect_gpu():
    """检测系统是否有可用的GPU"""
    has_gpu = False
    
    # 检查PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            logging.info("✅ PyTorch检测到可用GPU: %s", torch.cuda.get_device_name(0))
            has_gpu = True
        else:
            logging.info("❌ PyTorch未检测到可用GPU")
    except ImportError:
        logging.info("❌ PyTorch未安装，跳过PyTorch GPU检测")
    except Exception as e:
        logging.error("PyTorch GPU检测出错: %s", str(e))
    
    # 检查TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info("✅ TensorFlow检测到%d个可用GPU", len(gpus))
            has_gpu = True
        else:
            logging.info("❌ TensorFlow未检测到可用GPU")
    except ImportError:
        logging.info("❌ TensorFlow未安装，跳过TensorFlow GPU检测")
    except Exception as e:
        logging.error("TensorFlow GPU检测出错: %s", str(e))
    
    # 检查OpenCV
    try:
        import cv2
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logging.info("✅ OpenCV检测到%d个支持CUDA的设备", cv2.cuda.getCudaEnabledDeviceCount())
            has_gpu = True
        else:
            logging.info("❌ OpenCV未检测到支持CUDA的设备")
    except (ImportError, AttributeError):
        logging.info("❌ OpenCV未安装或不支持CUDA，跳过OpenCV GPU检测")
    except Exception as e:
        logging.error("OpenCV GPU检测出错: %s", str(e))
    
    return has_gpu

def set_gpu_environment():
    """根据GPU检测结果设置环境变量"""
    has_gpu = detect_gpu()
    
    if has_gpu:
        os.environ["DL_FACE_USE_GPU"] = "1"
        logging.info("✅ 检测到可用GPU，已设置 DL_FACE_USE_GPU=1")
    else:
        os.environ["DL_FACE_USE_GPU"] = "0"
        logging.info("❌ 未检测到可用GPU，已设置 DL_FACE_USE_GPU=0")
    
    return has_gpu

if __name__ == "__main__":
    # 检测GPU并设置环境变量
    has_gpu = set_gpu_environment()
    
    # 运行主程序（如果命令行参数中包含--run）
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        import subprocess
        
        try:
            # 使用当前设置的环境变量启动应用
            cmd = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
            logging.info("正在启动应用: %s", " ".join(cmd))
            subprocess.run(cmd, env=os.environ)
        except Exception as e:
            logging.error("启动应用失败: %s", str(e))
    else:
        # 打印使用说明
        print("\n要启动应用，请运行:")
        print(f"python {__file__} --run") 