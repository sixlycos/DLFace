"""
启动脚本 - 系统入口点，用于启动应用程序
"""

import os
import subprocess
import argparse
import sys
import importlib.util

# 尝试导入自动GPU检测模块，如果存在的话
AUTO_GPU_DETECTION = True
try:
    spec = importlib.util.find_spec('auto_gpu_detection')
    if spec is not None:
        from auto_gpu_detection import set_gpu_environment
    else:
        AUTO_GPU_DETECTION = False
except ImportError:
    AUTO_GPU_DETECTION = False

def download_model():
    """下载预训练模型"""
    print("正在下载预训练模型...")
    
    # 调用整合版的模型下载脚本
    try:
        # 直接导入并运行download_model.py中的main函数
        from download_model import main as download_model_main
        download_model_main()
        return True
    except ImportError:
        # 如果无法导入，尝试使用subprocess运行脚本
        try:
            subprocess.run([sys.executable, "download_model.py", "--all"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"下载失败: {e}")
            print("请手动运行: python download_model.py --all")
            return False
        except FileNotFoundError:
            print("未找到下载脚本: download_model.py")
            print("请确保此文件存在并位于正确的路径")
            return False


def create_dataset_dir():
    """创建数据集目录"""
    dataset_dir = os.path.join("data", "dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"创建数据集目录: {dataset_dir}")
        print("请将训练图像放入此目录以训练模型")


def create_samples_dir():
    """创建示例目录"""
    samples_dir = os.path.join("data", "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir, exist_ok=True)
        print(f"创建示例目录: {samples_dir}")


def run_app():
    """启动Streamlit应用"""
    streamlit_app_path = os.path.join("app", "streamlit_app.py")
    
    print("正在启动应用...")
    try:
        subprocess.run(["streamlit", "run", streamlit_app_path])
    except Exception as e:
        print(f"启动失败: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频人物面部替换系统")
    parser.add_argument("--download-model", action="store_true", help="下载预训练模型")
    parser.add_argument("--no-run", action="store_true", help="不启动应用")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用CPU模式")
    parser.add_argument("--force-gpu", action="store_true", help="强制使用GPU模式")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 处理GPU/CPU模式选择
    if args.force_gpu:
        os.environ["DL_FACE_USE_GPU"] = "1"
        print("已强制启用GPU模式")
    elif args.force_cpu:
        os.environ["DL_FACE_USE_GPU"] = "0"
        print("已强制启用CPU模式")
    elif AUTO_GPU_DETECTION:
        # 执行自动GPU检测
        print("正在进行自动GPU检测...")
        has_gpu = set_gpu_environment()
        if has_gpu:
            print("检测到GPU，将使用GPU模式运行")
        else:
            print("未检测到可用GPU，将使用CPU模式运行")
    
    # 创建目录
    create_dataset_dir()
    create_samples_dir()
    
    # 下载模型
    if args.download_model:
        download_model()
    
    # 启动应用
    if not args.no_run:
        # 检查依赖
        try:
            import streamlit
        except ImportError:
            print("未安装Streamlit，正在尝试安装...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
            except Exception as e:
                print(f"安装失败: {e}")
                print("请手动安装Streamlit: pip install streamlit")
                return
        
        run_app()


if __name__ == "__main__":
    main() 