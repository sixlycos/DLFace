#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DLface依赖安装脚本

该脚本检测当前Python版本，并安装合适的依赖项。
它会创建虚拟环境，根据不同的Python版本选择合适的TensorFlow版本，
并安装其他所有必需的依赖。

使用方法:
    python install_dependencies.py [--gpu] [--cpu-only] [--no-verify] [--venv VENV_PATH] [--no-venv]
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
import venv
import shutil
from pathlib import Path
from importlib import import_module

# 工作目录
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# 尝试使用pkg_resources，但在缺失时不报错
try:
    from pkg_resources import parse_version
except ImportError:
    # 简单版本解析器
    def parse_version(version_string):
        return tuple(map(int, version_string.split('.')))

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

# 额外必需的依赖项，以确保主要功能正常工作
ESSENTIAL_DEPS = [
    "python-dateutil>=2.8.2",
]

def get_python_version():
    """获取当前Python版本"""
    ver = sys.version_info
    return f"{ver.major}.{ver.minor}"

def check_pip(python_exe=None):
    """检查pip是否已安装并是最新版本"""
    if python_exe is None:
        python_exe = sys.executable
        
    try:
        subprocess.check_call([python_exe, "-m", "pip", "--version"], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 更新pip
        logger.info("更新pip到最新版本...")
        subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        logger.error("无法找到pip。请先安装pip: https://pip.pypa.io/en/stable/installation/")
        return False

def install_package(package, upgrade=False, python_exe=None):
    """安装单个包"""
    if python_exe is None:
        python_exe = sys.executable
        
    cmd = [python_exe, "-m", "pip", "install"]
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

def verify_package(package_name, python_exe=None):
    """验证包是否已正确安装"""
    if python_exe is None:
        python_exe = sys.executable
        
    package_base = package_name.split("==")[0].split(">=")[0].split("<")[0].strip()
    try:
        result = subprocess.run(
            [python_exe, "-c", f"import {package_base.replace('-', '_')}; print('已安装')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "已安装" in result.stdout:
            logger.info(f"已验证: {package_base}")
            return True
        else:
            logger.error(f"无法导入 {package_base}，安装可能失败")
            return False
    except subprocess.CalledProcessError:
        logger.error(f"无法验证 {package_base}")
        return False

def install_from_requirements(requirements_file="requirements.txt", use_gpu=False, py_version=None, python_exe=None):
    """从requirements.txt安装依赖"""
    if python_exe is None:
        python_exe = sys.executable
        
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
    
    # 添加必要的依赖
    for dep in ESSENTIAL_DEPS:
        if dep not in cleaned_requirements:
            cleaned_requirements.append(dep)
    
    requirements = cleaned_requirements
    
    # 特殊处理TensorFlow
    if use_gpu:
        # 先卸载可能存在的CPU版本
        subprocess.run([python_exe, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-cpu", "tensorflow-intel"], 
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        tf_version = PY_VERSION_MAPPING_GPU.get(py_version, PY_VERSION_MAPPING_GPU["default"])
        # 安装CUDA相关依赖
        logger.info("安装CUDA支持的TensorFlow依赖...")
        
        # 确保安装最新的tensorboard
        install_package("tensorboard>=2.12.0", True, python_exe)
        
        # 安装特定版本的TensorFlow
        logger.info(f"安装GPU版本TensorFlow: {tf_version}")
        if not install_package(tf_version, False, python_exe):
            logger.warning("GPU版本TensorFlow安装失败，尝试安装CPU版本...")
            tf_version = PY_VERSION_MAPPING.get(py_version, PY_VERSION_MAPPING["default"])
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
        if not install_package(dep, False, python_exe):
            success = False
    
    # 安装关键依赖
    for dep in critical_deps:
        if not install_package(dep, False, python_exe):
            # 对于face-alignment，尝试安装特定版本
            if "face-alignment" in dep:
                logger.warning("尝试安装特定版本的face-alignment...")
                if not install_package("face-alignment==1.3.5", False, python_exe):
                    success = False
            # 对于dlib，尝试安装预编译版本
            elif "dlib" in dep:
                logger.warning("尝试安装预编译版本的dlib...")
                py_ver = py_version.replace(".", "")
                dlib_url = f"https://github.com/jloh02/dlib/releases/download/v19.24/dlib-19.24.0-cp{py_ver}-cp{py_ver}-win_amd64.whl"
                if not install_package(dlib_url, False, python_exe):
                    success = False
            else:
                success = False
    
    # 确保安装了python-dateutil
    install_package("python-dateutil>=2.8.2", True, python_exe)
    
    return success

def check_gpu(python_exe=None):
    """检查是否有可用的GPU"""
    if python_exe is None:
        python_exe = sys.executable
        
    # 检查PyTorch是否能识别GPU
    try:
        result = subprocess.run(
            [python_exe, "-c", "import torch; print(f'PYTORCH_GPU_AVAILABLE={torch.cuda.is_available()}')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "PYTORCH_GPU_AVAILABLE=True" in result.stdout:
            logger.info("PyTorch检测到可用GPU")
            return True
    except:
        logger.warning("PyTorch未检测到GPU或未安装PyTorch")
    
    # 检查TensorFlow是否能识别GPU
    try:
        result = subprocess.run(
            [python_exe, "-c", "import tensorflow as tf; print(f'TF_GPU_AVAILABLE={len(tf.config.list_physical_devices(\"GPU\"))>0}')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "TF_GPU_AVAILABLE=True" in result.stdout:
            logger.info("TensorFlow检测到可用GPU")
            return True
        else:
            logger.warning("TensorFlow未检测到可用GPU")
    except:
        logger.warning("TensorFlow未检测到GPU或未安装TensorFlow")
    
    # 检查NVIDIA驱动
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            logger.info("检测到NVIDIA驱动，但TensorFlow/PyTorch未能识别GPU")
            
            # 确认CUDA版本
            cuda_version = None
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    cuda_version = line.split(': ')[1].strip()
                    logger.info(f"检测到CUDA版本: {cuda_version}")
                    
            logger.info("尝试使用GPU版本...")
            return True
    except:
        logger.warning("未检测到NVIDIA驱动或无法运行nvidia-smi")
    
    return False

def create_virtual_env(venv_path):
    """创建虚拟环境"""
    venv_path = os.path.abspath(venv_path)
    logger.info(f"创建虚拟环境: {venv_path}")
    
    if os.path.exists(venv_path):
        logger.warning(f"虚拟环境已存在: {venv_path}")
        if input("是否重建虚拟环境? (y/n): ").lower().startswith('y'):
            logger.info("删除旧虚拟环境...")
            shutil.rmtree(venv_path, ignore_errors=True)
        else:
            logger.info("使用现有虚拟环境...")
            return venv_path
    
    # 创建虚拟环境
    try:
        venv.create(venv_path, with_pip=True)
        logger.info(f"虚拟环境创建成功: {venv_path}")
    except Exception as e:
        logger.error(f"创建虚拟环境失败: {e}")
        return None
    
    return venv_path

def get_venv_python(venv_path):
    """获取虚拟环境中的Python可执行文件路径"""
    if platform.system() == "Windows":
        python_exe = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_exe = os.path.join(venv_path, "bin", "python")
        
    if not os.path.exists(python_exe):
        logger.error(f"未找到虚拟环境中的Python可执行文件: {python_exe}")
        return None
        
    return python_exe

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DLface依赖安装脚本")
    parser.add_argument("--gpu", action="store_true", help="安装GPU版本的TensorFlow")
    parser.add_argument("--cpu-only", action="store_true", help="强制安装CPU版本的TensorFlow")
    parser.add_argument("--no-verify", action="store_true", help="跳过依赖验证")
    parser.add_argument("--venv", type=str, default=".venv", help="虚拟环境路径 (默认: .venv)")
    parser.add_argument("--no-venv", action="store_true", help="不创建虚拟环境，直接使用当前Python环境")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("DLface依赖安装脚本 - 增强版")
    logger.info("=" * 50)
    
    # 决定是否使用虚拟环境
    python_exe = sys.executable
    if not args.no_venv:
        venv_path = create_virtual_env(args.venv)
        if venv_path:
            python_exe = get_venv_python(venv_path)
            if not python_exe:
                logger.error("无法获取虚拟环境中的Python可执行文件，将使用当前Python环境")
                python_exe = sys.executable
    else:
        logger.info("使用当前Python环境")
    
    # 检查pip
    if not check_pip(python_exe):
        return False
    
    # 检查Python版本
    result = subprocess.run([python_exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                         stdout=subprocess.PIPE, text=True)
    py_version = result.stdout.strip()
    
    if not (py_version.startswith("3.8") or py_version.startswith("3.9") or 
            py_version.startswith("3.10") or py_version.startswith("3.11")):
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
        if check_gpu(python_exe):
            logger.info("检测到GPU，将安装GPU版本的TensorFlow")
            use_gpu = True
        else:
            logger.info("未检测到GPU或GPU不可用，将安装CPU版本的TensorFlow")
    
    # 安装依赖
    logger.info("开始安装依赖...")
    success = install_from_requirements(use_gpu=use_gpu, py_version=py_version, python_exe=python_exe)
    
    if success:
        logger.info("依赖安装完成！")
        
        # 验证关键依赖
        if not args.no_verify:
            logger.info("验证关键依赖...")
            verify_package("tensorflow", python_exe)
            verify_package("opencv-python", python_exe)
            verify_package("mediapipe", python_exe)
            verify_package("dlib", python_exe)
            verify_package("face-alignment", python_exe)
            verify_package("python-dateutil", python_exe)
            
        logger.info("\n接下来您可以:")
        if not args.no_venv:
            if platform.system() == "Windows":
                activate_cmd = f"{args.venv}\\Scripts\\activate"
            else:
                activate_cmd = f"source {args.venv}/bin/activate"
            logger.info(f"1. 激活虚拟环境: {activate_cmd}")
            
        logger.info("2. 下载模型: python download_model.py --all")
        logger.info("3. 启动应用: python run.py")
    else:
        logger.error("部分依赖安装失败，请查看日志并手动解决问题")
    
    return success

if __name__ == "__main__":
    main() 