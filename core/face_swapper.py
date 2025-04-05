"""
面部替换模块 - 负责执行面部替换操作

实现基于自编码器的面部替换功能，支持模型加载和推理
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, Union
import time

# 导入统一日志配置和GPU补丁
from .logger_config import setup_logger
from .gpu_patch import check_cuda_availability, apply_gpu_patches

# 设置日志
logger = setup_logger()

# 确保GPU设置应用
gpu_available = apply_gpu_patches()

# 尝试导入TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("TensorFlow已成功导入")
    
    # GPU信息已经在apply_gpu_patches中处理，这里不再重复
        
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow导入失败，将使用简单替代方法")

# 尝试导入PyTorch作为备选
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch已成功导入")
    
    # GPU信息已经在apply_gpu_patches中处理，这里不再重复
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch导入失败")


# 导入面部增强器
try:
    from improve_face_swap import FaceEnhancer
    FACE_ENHANCER_AVAILABLE = True
except ImportError:
    FACE_ENHANCER_AVAILABLE = False
    logger.warning("无法导入FaceEnhancer，面部增强功能将不可用")

class FaceSwapper:
    """面部替换类，实现面部替换功能"""
    
    def __init__(self, use_cuda: bool = True):
        """
        初始化面部替换器
        
        Args:
            use_cuda: 是否使用CUDA加速
        """
        self.use_cuda = use_cuda
        self.model = None
        self.model_path = None
        self.device = None
        
        # 初始化面部增强器
        if FACE_ENHANCER_AVAILABLE:
            self.enhancer = FaceEnhancer(use_gpu=use_cuda)
            logger.info("面部增强器已初始化")
        else:
            self.enhancer = None
            logger.warning("面部增强器不可用")
        
        # 检查是否可以使用GPU
        if self.use_cuda:
            # 首先检查TensorFlow
            if TF_AVAILABLE:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"TensorFlow检测到 {len(gpus)} 个GPU:")
                    for gpu in gpus:
                        logger.info(f"  - {gpu.name}")
                if gpus:
                    logger.info(f"FaceSwapper将使用TensorFlow + GPU加速")
                else:
                    # 尝试检查PyTorch
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        logger.info(f"FaceSwapper将使用PyTorch + GPU加速")
                        self.use_cuda = True
                        self.device = torch.device("cuda")
                    else:
                        logger.warning("未检测到可用的GPU，将使用CPU")
                        self.use_cuda = False
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                logger.info(f"FaceSwapper将使用PyTorch + GPU加速")
                self.use_cuda = True
                self.device = torch.device("cuda")
            else:
                logger.warning("未检测到可用的深度学习框架或GPU，将使用CPU")
                self.use_cuda = False
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        加载面部替换模型
        
        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
            
        Returns:
            是否成功加载模型
        """
        # 检查TensorFlow是否可用
        if not TF_AVAILABLE:
            logger.warning("TensorFlow不可用，将使用简单替代方法")
            self.model = "placeholder"
            return True
        
        # 设置模型路径
        if model_path is None:
            # 使用默认模型路径
            model_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models")
            model_path = os.path.join(model_dir, "face_swap_model.h5")
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.warning(f"默认模型文件不存在: {model_path}")
                model_path = os.path.join(model_dir, "face_swap_placeholder.h5")
                
                # 如果占位符也不存在，创建一个
                if not os.path.exists(model_path):
                    logger.warning("创建模型占位符文件")
                    os.makedirs(model_dir, exist_ok=True)
                    with open(model_path, 'wb') as f:
                        f.write(b'FACESWAP_MODEL_PLACEHOLDER')
                
                logger.info(f"将使用占位符模型: {model_path}")
                self.model = "placeholder"
                self.model_path = model_path
                return True
        
        self.model_path = model_path
        logger.info(f"正在加载模型: {model_path}")
        
        # 使用TensorFlow加载模型
        try:
            # 检查是否为占位符模型
            with open(model_path, 'rb') as f:
                header = f.read(32)
                if b'FACESWAP_MODEL_PLACEHOLDER' in header or b'FSMP' in header:
                    logger.info("检测到模型占位符，将使用简单替代方法")
                    self.model = "placeholder"
                    return True
            
            # 尝试加载模型
            try:
                # 在GPU上创建模型（如果可用）
                if self.use_cuda and TF_AVAILABLE:
                    with tf.device('/GPU:0'):
                        self._create_model()
                        
                    # 加载权重
                    try:
                        self.model.load_weights(model_path)
                        logger.info(f"模型权重已成功加载到GPU: {model_path}")
                    except Exception as e:
                        logger.warning(f"加载模型权重到GPU失败: {str(e)}")
                        logger.warning("将使用未训练的模型")
                else:
                    # 在CPU上创建模型
                    self._create_model()
                    
                    # 加载权重
                    try:
                        self.model.load_weights(model_path)
                        logger.info(f"模型权重已成功加载到CPU: {model_path}")
                    except Exception as e:
                        logger.warning(f"加载模型权重到CPU失败: {str(e)}")
                        logger.warning("将使用未训练的模型")
                
                return True
            except Exception as e:
                logger.error(f"创建模型时出错: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return False
            
    def _create_model(self):
        """创建模型架构"""
        # 创建简单的自编码器模型
        input_shape = (256, 256, 3)
        
        # 输入层
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # 编码器部分
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # 解码器部分
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        # 创建模型
        self.model = tf.keras.models.Model(inputs, decoded)
    
    def swap_face(self, source_face: np.ndarray, target_face: np.ndarray) -> Optional[np.ndarray]:
        """
        执行面部替换
        
        Args:
            source_face: 源人脸图像
            target_face: 目标人脸图像
            
        Returns:
            替换后的人脸图像，如果失败则返回None
        """
        # 检查输入
        if source_face is None or target_face is None:
            logger.error("源人脸或目标人脸为空")
            return None
        
        # 检查图像大小是否异常大
        if source_face.shape[0] > 1000 or source_face.shape[1] > 1000:
            logger.warning(f"源人脸大小异常: {source_face.shape}，将自动调整大小")
            # 保持宽高比的情况下调整大小
            ratio = min(1000 / source_face.shape[0], 1000 / source_face.shape[1])
            new_size = (int(source_face.shape[1] * ratio), int(source_face.shape[0] * ratio))
            source_face = cv2.resize(source_face, new_size)
        
        if target_face.shape[0] > 1000 or target_face.shape[1] > 1000:
            logger.warning(f"目标人脸大小异常: {target_face.shape}，将自动调整大小")
            # 保持宽高比的情况下调整大小
            ratio = min(1000 / target_face.shape[0], 1000 / target_face.shape[1])
            new_size = (int(target_face.shape[1] * ratio), int(target_face.shape[0] * ratio))
            target_face = cv2.resize(target_face, new_size)
            
        # 如果源人脸和目标人脸大小不一致，发出警告
        if source_face.shape != target_face.shape:
            logger.warning(f"源人脸和目标人脸大小不一致，调整大小")
            # 调整源人脸大小以匹配目标人脸
            source_face = cv2.resize(source_face, (target_face.shape[1], target_face.shape[0]))
        
        # 如果模型未加载或是占位符模型，使用基于关键点的面部替换方法
        if self.model is None or self.model == "placeholder":
            logger.info("模型未加载或是占位符，执行基于关键点的面部替换")
            return self._facial_landmarks_swap(source_face, target_face)
        
        # 使用TensorFlow模型执行面部替换
        try:
            # 准备模型输入 - 调整到正确的大小
            model_input_size = (256, 256)
            
            # 检查源人脸是否已经是正确尺寸，否则调整
            if source_face.shape[0] != model_input_size[1] or source_face.shape[1] != model_input_size[0]:
                logger.info(f"调整源人脸以适应模型输入: {source_face.shape} -> {model_input_size}")
                source_face_resized = cv2.resize(source_face, model_input_size)
            else:
                source_face_resized = source_face
                
            # 保存原始目标人脸的尺寸，用于稍后恢复
            target_original_shape = target_face.shape
            
            # 归一化图像
            source_face_norm = source_face_resized.astype(np.float32) / 255.0
            
            # GPU加速处理
            if self.use_cuda and TF_AVAILABLE:
                try:
                    with tf.device('/GPU:0'):
                        # 扩展维度以匹配模型输入
                        source_face_batch = np.expand_dims(source_face_norm, axis=0)
                        
                        # 执行推理
                        start_time = time.time()
                        swapped_face_batch = self.model.predict(source_face_batch)
                        inference_time = time.time() - start_time
                        
                        logger.debug(f"GPU推理耗时: {inference_time:.4f}秒")
                except Exception as gpu_error:
                    logger.warning(f"GPU推理失败，回退到CPU: {str(gpu_error)}")
                    # 扩展维度以匹配模型输入
                    source_face_batch = np.expand_dims(source_face_norm, axis=0)
                    
                    # 执行推理
                    start_time = time.time()
                    swapped_face_batch = self.model.predict(source_face_batch)
                    inference_time = time.time() - start_time
                    
                    logger.debug(f"CPU推理耗时: {inference_time:.4f}秒")
            else:
                # CPU处理
                # 扩展维度以匹配模型输入
                source_face_batch = np.expand_dims(source_face_norm, axis=0)
                
                # 执行推理
                start_time = time.time()
                swapped_face_batch = self.model.predict(source_face_batch)
                inference_time = time.time() - start_time
                
                logger.debug(f"CPU推理耗时: {inference_time:.4f}秒")
            
            # 取出结果并转换回原始值域
            swapped_face = (swapped_face_batch[0] * 255.0).astype(np.uint8)
            
            # 将结果调整回目标图像尺寸
            swapped_face = cv2.resize(swapped_face, (target_original_shape[1], target_original_shape[0]))
            
            # 进行颜色校正，使结果更接近目标人脸的颜色分布
            swapped_face = self._color_transfer(target_face, swapped_face)
            
            return swapped_face
        except Exception as e:
            logger.error(f"执行面部替换时出错: {str(e)}")
            logger.debug(f"源人脸尺寸: {source_face.shape}, 目标人脸尺寸: {target_face.shape}")
            # 如果模型推理失败，回退到基于关键点的方法
            return self._facial_landmarks_swap(source_face, target_face)
    
    def _facial_landmarks_swap(self, source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        """
        基于面部关键点的面部替换方法 - 简化且强健的版本
        
        Args:
            source_face: 源人脸图像
            target_face: 目标人脸图像
            
        Returns:
            替换后的人脸图像
        """
        try:
            # 确保图像尺寸一致 - 调整源人脸大小以匹配目标人脸
            if source_face.shape != target_face.shape:
                logger.info(f"调整源人脸大小以匹配目标人脸: {source_face.shape} -> {target_face.shape}")
                source_face = cv2.resize(source_face, (target_face.shape[1], target_face.shape[0]))
            
            # 检测源人脸和目标人脸的特征点
            src_landmarks = self._detect_landmarks(source_face)
            dst_landmarks = self._detect_landmarks(target_face)
            
            # 如果没有足够的特征点，回退到简单混合
            if src_landmarks is None or dst_landmarks is None or len(src_landmarks) < 5 or len(dst_landmarks) < 5:
                logger.warning("无法检测到足够的特征点，使用基本的Alpha混合")
                return self._basic_alpha_blend(source_face, target_face)
            
            # 创建结果图像副本
            result = target_face.copy()
            
            # 计算特征点的边界框，用于确定面部区域
            src_face_rect = self._get_face_rect(src_landmarks, source_face.shape)
            dst_face_rect = self._get_face_rect(dst_landmarks, target_face.shape)
            
            # 裁剪源人脸和目标人脸的面部区域
            x1, y1, w1, h1 = src_face_rect
            x2, y2, w2, h2 = dst_face_rect
            
            src_face_only = source_face[y1:y1+h1, x1:x1+w1]
            dst_face_only = target_face[y2:y2+h2, x2:x2+w2]
            
            # 调整源人脸区域大小以匹配目标人脸区域
            src_face_resized = cv2.resize(src_face_only, (w2, h2))
            
            # 创建掩码 - 使用人脸区域的椭圆形掩码
            mask = np.zeros((h2, w2), dtype=np.uint8)
            center = (w2 // 2, h2 // 2)
            axes = (w2 // 2 - w2 // 10, h2 // 2 - h2 // 10)  # 稍微缩小一点以确保不会超出脸部边界
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
            # 平滑掩码边缘以获得更自然的融合
            mask = cv2.GaussianBlur(mask, (21, 21), 11)
            
            # 在色彩空间中进行混合以获得更好的效果
            # 转换到LAB色彩空间
            src_lab = cv2.cvtColor(src_face_resized, cv2.COLOR_BGR2LAB)
            dst_lab = cv2.cvtColor(dst_face_only, cv2.COLOR_BGR2LAB)
            
            # 分离LAB通道
            src_l, src_a, src_b = cv2.split(src_lab)
            dst_l, dst_a, dst_b = cv2.split(dst_lab)
            
            # 使用目标图像的亮度，但使用源图像的颜色（混合）
            mask_float = mask.astype(np.float32) / 255.0
            
            # 混合亮度通道 - 使用50%源图像亮度和50%目标图像亮度
            l_blend = cv2.addWeighted(src_l, 0.5, dst_l, 0.5, 0)
            
            # 使用掩码混合颜色通道
            mask_2d = mask_float
            a_blend = dst_a * (1 - mask_2d) + src_a * mask_2d
            b_blend = dst_b * (1 - mask_2d) + src_b * mask_2d
            
            # 合并通道
            blend_lab = cv2.merge([l_blend.astype(np.uint8), a_blend.astype(np.uint8), b_blend.astype(np.uint8)])
            
            # 转换回BGR
            blend_bgr = cv2.cvtColor(blend_lab, cv2.COLOR_LAB2BGR)
            
            # 应用高质量的边缘平滑
            alpha = np.expand_dims(mask_float, axis=2)
            alpha = np.repeat(alpha, 3, axis=2)
            blended_face = (dst_face_only * (1 - alpha) + blend_bgr * alpha).astype(np.uint8)
            
            # 将混合后的脸部放回原图
            result[y2:y2+h2, x2:x2+w2] = blended_face
            
            # 应用面部增强
            if self.enhancer is not None and face_mask is not None and face_mask.shape[0] > 0 and face_mask.shape[1] > 0:
                logger.info("应用面部增强...")
                try:
                    result = self.enhancer.enhance(result, face_mask, image)
                    logger.info("面部增强完成")
                except Exception as e:
                    logger.error(f"面部增强失败: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"基于关键点的面部替换失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._basic_alpha_blend(source_face, target_face)
    
    def _basic_alpha_blend(self, source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        """
        基本的Alpha混合方法，在其他方法失败时作为备选
        
        Args:
            source_face: 源人脸图像
            target_face: 目标人脸图像
            
        Returns:
            混合后的人脸图像
        """
        # 确保图像尺寸一致
        if source_face.shape != target_face.shape:
            source_face = cv2.resize(source_face, (target_face.shape[1], target_face.shape[0]))
        
        # 创建掩码 - 简单的椭圆形
        mask = np.zeros(target_face.shape[:2], dtype=np.uint8)
        height, width = mask.shape
        center = (width // 2, height // 2)
        axes = (width // 2 - width // 10, height // 2 - height // 10)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # 平滑掩码边缘
        mask = cv2.GaussianBlur(mask, (31, 31), 11)
        
        # 转换为3通道掩码，用于混合
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        
        # 简单的Alpha混合
        alpha = 0.85  # 控制混合强度
        blend = (source_face * alpha + target_face * (1 - alpha)).astype(np.uint8)
        
        # 使用掩码混合
        result = (target_face * (1 - mask_3d) + blend * mask_3d).astype(np.uint8)
        
        return result
    
    def _get_face_rect(self, landmarks: np.ndarray, image_shape: tuple) -> tuple:
        """
        根据特征点获取人脸矩形区域，并确保不会超出图像边界
        
        Args:
            landmarks: 人脸特征点
            image_shape: 图像形状 (高度, 宽度)
            
        Returns:
            人脸矩形 (x, y, 宽度, 高度)
        """
        height, width = image_shape[:2]
        
        # 设置边界
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        
        # 查找边界
        for (x, y) in landmarks:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        
        # 扩大边界区域以包含整个脸部
        padding_x = int((max_x - min_x) * 0.3)
        padding_y = int((max_y - min_y) * 0.3)
        
        # 确保矩形在图像内部
        x = max(0, int(min_x) - padding_x)
        y = max(0, int(min_y) - padding_y)
        w = min(width - x, int(max_x - min_x) + 2 * padding_x)
        h = min(height - y, int(max_y - min_y) + 2 * padding_y)
        
        return (x, y, w, h)
    
    def _detect_landmarks(self, face_image: np.ndarray) -> np.ndarray:
        """
        检测面部特征点
        
        Args:
            face_image: 人脸图像
            
        Returns:
            面部特征点数组
        """
        try:
            # 尝试使用dlib检测特征点
            import dlib
            predictor_path = os.path.join("data", "models", "shape_predictor_68_face_landmarks.dat")
            if os.path.exists(predictor_path):
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor(predictor_path)
                
                # 转换为灰度图
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                
                # 检测面部
                faces = detector(gray)
                if len(faces) > 0:
                    # 获取特征点
                    landmarks = predictor(gray, faces[0])
                    points = []
                    for i in range(68):
                        x = landmarks.part(i).x
                        y = landmarks.part(i).y
                        points.append((x, y))
                    return np.array(points)
            
            # 如果dlib不可用或检测失败，尝试使用OpenCV
            import cv2
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # 如果没有特征点检测器，创建一些基本点
                points = []
                # 眼睛
                points.append((x + w//4, y + h//3))
                points.append((x + 3*w//4, y + h//3))
                # 鼻子
                points.append((x + w//2, y + h//2))
                # 嘴巴
                points.append((x + w//3, y + 2*h//3))
                points.append((x + 2*w//3, y + 2*h//3))
                return np.array(points)
        except Exception as e:
            logger.warning(f"特征点检测失败: {str(e)}")
        
        return None
    
    def _color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        执行颜色转移，使目标图像的颜色分布匹配源图像
        
        Args:
            source: 源图像
            target: 目标图像
            
        Returns:
            颜色校正后的图像
        """
        # 检查图像大小是否匹配
        if source.shape != target.shape:
            target = cv2.resize(target, (source.shape[1], source.shape[0]))
        
        # 转换到LAB颜色空间
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 分离通道
        (src_l, src_a, src_b) = cv2.split(source_lab)
        (tar_l, tar_a, tar_b) = cv2.split(target_lab)
        
        # 计算每个通道的均值和标准差
        src_l_mean, src_l_std = np.mean(src_l), np.std(src_l)
        src_a_mean, src_a_std = np.mean(src_a), np.std(src_a)
        src_b_mean, src_b_std = np.mean(src_b), np.std(src_b)
        
        tar_l_mean, tar_l_std = np.mean(tar_l), np.std(tar_l)
        tar_a_mean, tar_a_std = np.mean(tar_a), np.std(tar_a)
        tar_b_mean, tar_b_std = np.mean(tar_b), np.std(tar_b)
        
        # 应用统计迁移
        # 避免除以0
        tar_l_std = max(tar_l_std, 1e-5)
        tar_a_std = max(tar_a_std, 1e-5)
        tar_b_std = max(tar_b_std, 1e-5)
        
        tar_l = ((tar_l - tar_l_mean) / tar_l_std) * src_l_std + src_l_mean
        tar_a = ((tar_a - tar_a_mean) / tar_a_std) * src_a_std + src_a_mean
        tar_b = ((tar_b - tar_b_mean) / tar_b_std) * src_b_std + src_b_mean
        
        # 将结果剪切到有效范围
        tar_l = np.clip(tar_l, 0, 255)
        tar_a = np.clip(tar_a, 0, 255)
        tar_b = np.clip(tar_b, 0, 255)
        
        # 合并通道
        transfer = cv2.merge([tar_l.astype(np.uint8), tar_a.astype(np.uint8), tar_b.astype(np.uint8)])
        
        # 转换回BGR颜色空间
        transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
        
        return transfer 