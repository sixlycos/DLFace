"""
基本用法示例 - 展示系统的基本使用方法

本示例展示如何使用视频人物面部替换系统的核心功能
"""

import os
import sys
import cv2
import numpy as np

# 将项目根目录添加到路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_processor import VideoProcessor
from core.face_detector import FaceDetector


def basic_face_swap_example():
    """基本面部替换示例"""
    print("基本面部替换示例")
    
    # 初始化处理器
    processor = VideoProcessor(use_cuda=False)  # 设置为False使用CPU，或True使用GPU
    
    # 加载模型
    model_path = os.path.join("data", "models", "face_swap_model.h5")
    if os.path.exists(model_path):
        processor.load_model(model_path)
        print(f"已加载模型: {model_path}")
    else:
        print(f"警告: 模型文件不存在 {model_path}")
        print("将使用未训练的模型，效果可能不佳")
    
    # 设置源人脸和目标视频路径
    # 可以替换为您自己的图像和视频
    source_face_path = os.path.join("data", "samples", "source_face.jpg")
    target_video_path = os.path.join("data", "samples", "target_video.mp4")
    
    # 检查文件是否存在
    if not os.path.exists(source_face_path):
        print(f"源人脸文件不存在: {source_face_path}")
        return
    
    if not os.path.exists(target_video_path):
        print(f"目标视频文件不存在: {target_video_path}")
        return
    
    # 设置输出路径
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "output_video.mp4")
    
    # 处理视频
    try:
        print(f"开始处理视频: {target_video_path}")
        print(f"使用源人脸: {source_face_path}")
        print(f"输出视频: {output_video_path}")
        
        # 限制处理帧数，加快示例运行速度
        frame_limit = 30
        
        # 处理视频
        processor.process_video(
            source_face_path, target_video_path, output_video_path, frame_limit=frame_limit
        )
        
        print(f"视频处理完成: {output_video_path}")
    except Exception as e:
        print(f"处理视频时出错: {e}")
    finally:
        # 清理临时文件
        processor.cleanup()


def detect_face_example():
    """人脸检测示例"""
    print("\n人脸检测示例")
    
    # 初始化人脸检测器
    detector = FaceDetector(use_cuda=False)
    
    # 设置图像路径
    image_path = os.path.join("data", "samples", "source_face.jpg")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 检测人脸
    faces = detector.detect_faces_mtcnn(image)
    
    # 显示结果
    if faces:
        print(f"检测到 {len(faces)} 个人脸")
        
        # 绘制检测结果
        for face in faces:
            # 获取人脸框
            x, y, width, height = face['box']
            
            # 绘制人脸框
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # 绘制关键点
            keypoints = face['keypoints']
            for point in keypoints.values():
                cv2.circle(image, point, 2, (0, 0, 255), -1)
        
        # 保存结果
        output_dir = os.path.join("data", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detected_face.jpg")
        cv2.imwrite(output_path, image)
        
        print(f"检测结果已保存: {output_path}")
    else:
        print("未检测到人脸")


if __name__ == "__main__":
    # 创建示例输出目录
    os.makedirs(os.path.join("data", "output"), exist_ok=True)
    
    # 运行示例
    detect_face_example()
    basic_face_swap_example() 