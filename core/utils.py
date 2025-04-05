"""
工具函数模块 - 提供各种实用功能
"""

import os
import cv2
import numpy as np
from typing import Optional

def read_image_with_path_fix(file_path: str) -> Optional[np.ndarray]:
    """
    使用路径修复方法读取图像，解决中文路径问题
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        读取的图像，如果失败则返回None
    """
    try:
        # 尝试直接读取
        img = cv2.imread(file_path)
        if img is not None:
            return img
        
        # 如果直接读取失败，尝试修复路径
        norm_path = os.path.normpath(file_path)
        img = cv2.imread(norm_path)
        if img is not None:
            return img
        
        # 如果仍然失败，尝试使用np.fromfile方法读取
        with open(file_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"读取图像文件失败: {file_path}, 错误: {str(e)}")
        return None 