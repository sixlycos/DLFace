"""
工具函数 - 为前端应用提供辅助功能
"""

import os
import cv2
import numpy as np
import tempfile
from typing import Optional, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# 导入我们的自定义路径修复函数
from core.utils import read_image_with_path_fix


def read_image(file_path: str) -> np.ndarray:
    """
    读取图像文件，支持中文路径
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        图像数组
    """
    try:
        # 首先尝试直接读取
        img = cv2.imread(file_path)
        if img is not None:
            return img
        
        # 如果失败，尝试使用np.fromfile方法读取
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        st.error(f"读取图像文件失败: {file_path}, 错误: {str(e)}")
        return None


def save_image(image: np.ndarray, file_path: str) -> str:
    """
    保存图像到文件
    
    Args:
        image: 图像数组
        file_path: 保存路径
        
    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    cv2.imwrite(file_path, image)
    return file_path


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    加载图像文件
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        加载的图像，如果失败则返回None
    """
    return read_image_with_path_fix(file_path)


def save_uploaded_file(uploaded_file, save_dir: str = None) -> Optional[str]:
    """
    保存上传的文件
    
    Args:
        uploaded_file: 上传的文件对象
        save_dir: 保存目录，如果为None则使用临时目录
        
    Returns:
        保存的文件路径，如果失败则返回None
    """
    try:
        # 如果未指定保存目录，使用临时目录
        if save_dir is None:
            temp_dir = tempfile.mkdtemp()
            save_dir = temp_dir
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"保存文件失败: {str(e)}")
        return None


def create_thumbnail(image: np.ndarray, max_size: int = 300) -> np.ndarray:
    """
    创建缩略图
    
    Args:
        image: 原始图像
        max_size: 最大尺寸
        
    Returns:
        缩略图
    """
    height, width = image.shape[:2]
    
    # 计算缩放比例
    if height > width:
        ratio = max_size / height
    else:
        ratio = max_size / width
    
    # 调整大小
    new_size = (int(width * ratio), int(height * ratio))
    thumbnail = cv2.resize(image, new_size)
    
    return thumbnail


def plot_side_by_side(image1: np.ndarray, image2: np.ndarray, 
                     title1: str = "原始图像", title2: str = "处理后图像",
                     figsize: Tuple[int, int] = (10, 5)) -> np.ndarray:
    """
    创建两张图像的并排对比图
    
    Args:
        image1: 第一张图像
        image2: 第二张图像
        title1: 第一张图像的标题
        title2: 第二张图像的标题
        figsize: 图形大小
        
    Returns:
        对比图像
    """
    # 转换为RGB格式（matplotlib需要RGB格式）
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 显示图像
    axes[0].imshow(image1_rgb)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(image2_rgb)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 将图形转换为图像
    fig.canvas.draw()
    comparison_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison_image = comparison_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # 关闭图形
    plt.close(fig)
    
    # 转换回BGR格式
    comparison_image = cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR)
    
    return comparison_image


def generate_output_filename(prefix: str = "output", extension: str = ".mp4") -> str:
    """
    生成输出文件名
    
    Args:
        prefix: 文件名前缀
        extension: 文件扩展名
        
    Returns:
        生成的文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        包含视频信息的字典
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    # 释放资源
    cap.release()
    
    return info


def extract_video_preview(video_path: str, frame_count: int = 5) -> List[np.ndarray]:
    """
    从视频中提取预览帧
    
    Args:
        video_path: 视频文件路径
        frame_count: 提取的帧数
        
    Returns:
        预览帧列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算步长
    step = total_frames / (frame_count + 1)
    
    # 提取帧
    frames = []
    for i in range(1, frame_count + 1):
        # 设置位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
        
        # 读取帧
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    # 释放资源
    cap.release()
    
    return frames 