"""
测试脚本 - 测试系统各个组件的功能
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import unittest

# 将项目根目录添加到路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_detector import FaceDetector
from core.eye_tracker import EyeTracker
from core.autoencoder import FaceSwapAutoEncoder
from core.postprocessor import FacePostprocessor
from core.video_processor import VideoProcessor


class TestFaceDetector(unittest.TestCase):
    """测试人脸检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = FaceDetector(use_cuda=False)
        
        # 创建测试图像（蓝色方块，200x200）
        self.test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.test_image[:, :, 0] = 255  # 蓝色
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.mtcnn_detector)
        self.assertIsNotNone(self.detector.dlib_detector)
    
    def test_extract_face(self):
        """测试人脸提取"""
        # 使用测试图像创建假人脸框
        face_box = (50, 50, 100, 100)
        
        # 提取人脸
        face = self.detector.extract_face(self.test_image, face_box)
        
        # 验证结果
        self.assertIsNotNone(face)
        self.assertEqual(face.shape[:2], (256, 256))  # 默认调整大小为256x256


class TestEyeTracker(unittest.TestCase):
    """测试眼睛跟踪器"""
    
    def setUp(self):
        """设置测试环境"""
        self.eye_tracker = EyeTracker(use_cuda=False)
        
        # 创建测试图像（蓝色方块，200x200）
        self.test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.test_image[:, :, 0] = 255  # 蓝色
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.eye_tracker)
        self.assertIsNotNone(self.eye_tracker.face_alignment_net)
    
    def test_get_eye_heatmap_with_no_eyes(self):
        """测试在没有眼睛的图像上生成热力图"""
        # 对于没有人脸的图像，应该返回全零热力图
        heatmap = self.eye_tracker.get_eye_heatmap(self.test_image)
        
        # 验证结果
        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.shape, self.test_image.shape[:2])
        self.assertEqual(heatmap.max(), 0)  # 应该是全零热力图


class TestAutoEncoder(unittest.TestCase):
    """测试自编码器"""
    
    def setUp(self):
        """设置测试环境"""
        self.autoencoder = FaceSwapAutoEncoder(input_shape=(64, 64, 3))
        
        # 创建测试图像（随机图像，64x64x3）
        self.test_image = np.random.random((64, 64, 3)).astype(np.float32)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.autoencoder)
        self.assertIsNotNone(self.autoencoder.encoder)
        self.assertIsNotNone(self.autoencoder.decoder)
    
    def test_build_model(self):
        """测试构建模型"""
        model = self.autoencoder.build_model()
        self.assertIsNotNone(model)


class TestPostProcessor(unittest.TestCase):
    """测试后处理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.postprocessor = FacePostprocessor()
        
        # 创建测试图像（蓝色方块，200x200）
        self.test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.test_image[:, :, 0] = 255  # 蓝色
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.postprocessor)
        self.assertIsNotNone(self.postprocessor.face_detector)
    
    def test_create_face_mask(self):
        """测试创建人脸掩码"""
        # 创建假人脸框
        face_box = (50, 50, 100, 100)
        
        # 创建掩码
        mask = self.postprocessor.create_face_mask(self.test_image, face_box)
        
        # 验证结果
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, self.test_image.shape[:2])
        self.assertGreater(mask.max(), 0)  # 应该有非零值


class TestVideoProcessor(unittest.TestCase):
    """测试视频处理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = VideoProcessor(use_cuda=False)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.processor)
        self.assertIsNotNone(self.processor.face_detector)
        self.assertIsNotNone(self.processor.eye_tracker)
        self.assertIsNotNone(self.processor.postprocessor)
        self.assertIsNone(self.processor.model)


if __name__ == "__main__":
    unittest.main() 