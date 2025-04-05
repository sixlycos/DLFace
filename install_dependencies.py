#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DLface依赖安装脚本

该脚本检测当前Python版本，并安装合适的依赖项。
它会根据不同的Python版本选择合适的TensorFlow版本，
并安装其他所有必需的依赖。

使用方法:
    python install_dependencies.py [--gpu] [--cpu-only] [--no-verify]
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
from importlib import import_module
from pkg_resources import parse_version

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logger = logging.getLogger("DependencyInstaller")

# 定义Python版本范围及对应的TensorFlow版本
PY_VERSION_MAPPING = {
    "3.8": "tensorflow==2.12.0",
    "3.9": "tensorflow==2.15.0",
    "3.10": "tensorflow==2.15.0",
    "3.11": "tensorflow==2.15.0",
    "default": "tensorflow>=2.12.0,<2.16.0"
}

# 定义GPU版本的TensorFlow（如有必要）
PY_VERSION_MAPPING_GPU = {
    "3.8": "tensorflow==2.12.0",  # TF 2.12+ 合并了GPU支持，不再需要tensorflow-gpu
    "3.9": "tensorflow==2.15.0",
    "3.10": "tensorflow==2.15.0",
    "3.11": "tensorflow==2.15.0",
    "default": "tensorflow>=2.12.0,<2.16.0"
}

# 关键依赖项，这些依赖项可能需要特殊处理
CRITICAL_DEPS = [
    "dlib",
    "face-alignment",
    "mediapipe",
    "h5py",
]

def get_python_version():
    """获取当前Python版本"""
    ver = sys.version_info
    return f"{ver.major}.{ver.minor}"

def check_pip():
    """检查pip是否已安装并是最新版本"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        # 更新pip
        logger.info("更新pip到最新版本...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        return True
    except subprocess.CalledProcessError:
        logger.error("无法找到pip。请先安装pip: https://pip.pypa.io/en/stable/installation/")
        return False

def install_package(package, upgrade=False):
    """安装单个包"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    
    logger.info(f"正在安装: {package}")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"安装 {package} 失败: {e}")
        return False

def verify_package(package_name):
    """验证包是否已正确安装"""
    package_base = package_name.split("==")[0].split(">=")[0].split("<")[0].strip()
    try:
        module = import_module(package_base.replace("-", "_"))
        if hasattr(module, "__version__"):
            logger.info(f"已验证: {package_base} {module.__version__}")
        else:
            logger.info(f"已验证: {package_base} (无版本信息)")
        return True
    except ImportError:
        logger.error(f"无法导入 {package_base}，安装可能失败")
        return False

def install_from_requirements(requirements_file="requirements.txt", use_gpu=False, py_version=None):
    """从requirements.txt安装依赖"""
    if py_version is None:
        py_version = get_python_version()
    
    logger.info(f"检测到Python版本: {py_version}")
    
    # 读取requirements.txt
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = f.readlines()
    
    # 过滤掉注释和空行，并正确处理行内注释
    cleaned_requirements = []
    for line in requirements:
        line = line.strip()
        if line and not line.startswith("#"):
            # 处理行内注释：截取注释符号前的部分
            if "#" in line:
                line = line.split("#")[0].strip()
            cleaned_requirements.append(line)
    
    requirements = cleaned_requirements
    
    # 特殊处理TensorFlow
    if use_gpu:
        tf_version = PY_VERSION_MAPPING_GPU.get(py_version, PY_VERSION_MAPPING_GPU["default"])
    else:
        tf_version = PY_VERSION_MAPPING.get(py_version, PY_VERSION_MAPPING["default"])
    
    # 替换requirements中的tensorflow版本
    requirements = [req if not req.startswith("tensorflow") else tf_version for req in requirements]
    
    # 如果需要特定的ml-dtypes版本（解决mediapipe兼容性问题）
    if use_gpu and float(py_version) < 3.9:
        # 对于较旧的Python版本，确保ml-dtypes兼容
        requirements.append("ml-dtypes==0.2.0")
    
    # 首先安装关键依赖项
    critical_deps = [dep for dep in requirements if any(critical in dep for critical in CRITICAL_DEPS)]
    other_deps = [dep for dep in requirements if not any(critical in dep for critical in CRITICAL_DEPS)]
    
    # 安装非关键依赖
    success = True
    for dep in other_deps:
        if not install_package(dep):
            success = False
    
    # 安装关键依赖
    for dep in critical_deps:
        if not install_package(dep):
            # 对于face-alignment，尝试安装特定版本
            if "face-alignment" in dep:
                logger.warning("尝试安装特定版本的face-alignment...")
                if not install_package("face-alignment==1.3.5"):
                    success = False
            else:
                success = False
    
    return success

def check_gpu():
    """检查是否有可用的GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"检测到 {len(gpus)} 个GPU设备:")
            for gpu in gpus:
                logger.info(f"  - {gpu}")
            return True
        else:
            logger.warning("未检测到可用的GPU")
            return False
    except ImportError:
        logger.warning("无法导入TensorFlow来检测GPU，将尝试安装兼容的TensorFlow版本")
        return False
    except Exception as e:
        logger.warning(f"检查GPU时出错: {e}")
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DLface依赖安装脚本")
    parser.add_argument("--gpu", action="store_true", help="安装GPU版本的TensorFlow")
    parser.add_argument("--cpu-only", action="store_true", help="强制安装CPU版本的TensorFlow")
    parser.add_argument("--no-verify", action="store_true", help="跳过依赖验证")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("DLface依赖安装脚本")
    logger.info("=" * 50)
    
    # 检查pip
    if not check_pip():
        return False
    
    # 检查Python版本
    py_version = get_python_version()
    if parse_version(py_version) < parse_version("3.8") or parse_version(py_version) >= parse_version("3.12"):
        logger.warning(f"Python {py_version} 可能与DLface不兼容")
        logger.warning("推荐使用Python 3.8-3.11")
        if not input("是否继续安装？(y/n): ").lower().startswith('y'):
            return False
    
    # 决定使用GPU还是CPU版本
    use_gpu = False
    if args.cpu_only:
        logger.info("将安装CPU版本的TensorFlow")
    elif args.gpu:
        logger.info("将安装GPU版本的TensorFlow")
        use_gpu = True
    else:
        # 自动检测
        logger.info("检测是否有可用的GPU...")
        if check_gpu():
            logger.info("检测到GPU，将安装GPU版本的TensorFlow")
            use_gpu = True
        else:
            logger.info("未检测到GPU或GPU不可用，将安装CPU版本的TensorFlow")
    
    # 安装依赖
    logger.info("开始安装依赖...")
    success = install_from_requirements(use_gpu=use_gpu, py_version=py_version)
    
    if success:
        logger.info("依赖安装完成！")
        
        # 验证关键依赖
        if not args.no_verify:
            logger.info("验证关键依赖...")
            verify_package("tensorflow")
            verify_package("opencv-python")
            verify_package("mediapipe")
            verify_package("dlib")
            verify_package("face-alignment")
            
        logger.info("\n接下来您可以:")
        logger.info("1. 下载模型: python download_model.py --all")
        logger.info("2. 启动应用: python run.py")
    else:
        logger.error("部分依赖安装失败，请查看日志并手动解决问题")
    
    return success

if __name__ == "__main__":
    main() 