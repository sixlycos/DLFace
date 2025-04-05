"""
自编码器模型 - 用于面部特征提取和替换

包含编码器和解码器，实现面部特征提取与重构功能
支持眼睛注视方向优化
支持DeepFakes风格的自编码器网络结构和预训练模型
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, List, Union, Dict, Any
import os
import cv2
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU设置优化
def configure_gpu():
    """配置GPU设置以优化性能"""
    try:
        # 检查GPU是否可用
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("未检测到GPU硬件，将使用CPU进行计算")
            return False
        
        logger.info(f"检测到 {len(gpus)} 个GPU设备")
        
        # 为每个GPU设置内存增长选项，避免一次性分配所有内存
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"已为GPU设备设置内存增长: {gpu}")
            except RuntimeError as e:
                logger.error(f"无法为GPU {gpu} 设置内存增长: {str(e)}")
        
        # 设置GPU选项
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.85,  # 限制GPU内存使用率，避免内存溢出
            allow_growth=True
        )
        
        # 设置运行选项
        config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,  # 如果操作无法在GPU上运行，自动转移到CPU
            log_device_placement=False  # 不在终端输出设备分配日志
        )
        
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)
        
        logger.info("GPU配置完成，已优化设置")
        return True
    except Exception as e:
        logger.error(f"配置GPU时出错: {str(e)}")
        return False


class ConvBlock(tf.keras.layers.Layer):
    """卷积块: 卷积+实例归一化+LeakyReLU"""
    
    def __init__(self, filters: int, kernel_size: int = 3, strides: int = 1, 
                 padding: str = 'same', use_norm: bool = True):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding,
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )
        self.use_norm = use_norm
        if use_norm:
            # 使用实例归一化而不是批归一化，因为批归一化在推理时可能会导致问题
            self.norm = tf.keras.layers.LayerNormalization()
        self.lrelu = tf.keras.layers.LeakyReLU(0.2)
        
    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.lrelu(x)
        return x


class UpSampleBlock(tf.keras.layers.Layer):
    """上采样块: 上采样+卷积+实例归一化+LeakyReLU"""
    
    def __init__(self, filters: int, kernel_size: int = 3, 
                 use_norm: bool = True, use_dropout: bool = False):
        super(UpSampleBlock, self).__init__()
        # 使用反卷积进行上采样，同时完成卷积操作
        self.up_conv = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )
        self.use_norm = use_norm
        if use_norm:
            self.norm = tf.keras.layers.LayerNormalization()
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)
        self.lrelu = tf.keras.layers.LeakyReLU(0.2)
        
    def call(self, x, training=None):
        x = self.up_conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_dropout and training:
            x = self.dropout(x)
        x = self.lrelu(x)
        return x


class DeepFakesEncoder(tf.keras.Model):
    """DeepFakes风格的编码器 - 使用更多卷积层和残差连接"""
    
    def __init__(self, latent_dim: int = 1024):
        super(DeepFakesEncoder, self).__init__()
        
        # 编码器网络
        self.conv1 = ConvBlock(64, 5, 2, use_norm=False)  # 128x128
        self.conv2 = ConvBlock(128, 5, 2)  # 64x64
        self.conv3 = ConvBlock(256, 5, 2)  # 32x32
        self.conv4 = ConvBlock(512, 5, 2)  # 16x16
        self.conv5 = ConvBlock(1024, 5, 2)  # 8x8
        
        # 残差块
        self.res1 = self._make_res_block(1024)
        self.res2 = self._make_res_block(1024)
        
        # 输出潜在特征向量
        self.latent_conv = tf.keras.layers.Conv2D(
            latent_dim, 4, padding='valid',
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )  # 5x5
    
    def _make_res_block(self, filters: int):
        """创建残差块"""
        return [
            tf.keras.layers.Conv2D(
                filters, 3, padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(
                filters, 3, padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
            ),
            tf.keras.layers.LayerNormalization()
        ]
    
    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        
        # 应用残差块
        identity = x
        for layer in self.res1:
            x = layer(x, training=training)
        x = x + identity
        
        identity = x
        for layer in self.res2:
            x = layer(x, training=training)
        x = x + identity
        
        x = self.latent_conv(x, training=training)
        return x


class DeepFakesDecoder(tf.keras.Model):
    """DeepFakes风格的解码器 - 使用更多上采样层和残差连接"""
    
    def __init__(self, output_channels: int = 3):
        super(DeepFakesDecoder, self).__init__()
        
        # 解码器网络
        self.up_conv1 = tf.keras.layers.Conv2DTranspose(
            1024, 4, padding='valid',
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )  # 8x8
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.lrelu1 = tf.keras.layers.LeakyReLU(0.2)
        
        # 残差块
        self.res1 = self._make_res_block(1024)
        self.res2 = self._make_res_block(1024)
        
        self.up2 = UpSampleBlock(512)  # 16x16
        self.up3 = UpSampleBlock(256)  # 32x32
        self.up4 = UpSampleBlock(128)  # 64x64
        self.up5 = UpSampleBlock(64)   # 128x128
        
        # 输出图像
        self.output_conv = tf.keras.layers.Conv2D(
            output_channels, 5, padding='same', activation='tanh',
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )
    
    def _make_res_block(self, filters: int):
        """创建残差块"""
        return [
            tf.keras.layers.Conv2D(
                filters, 3, padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(
                filters, 3, padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
            ),
            tf.keras.layers.LayerNormalization()
        ]
    
    def call(self, x, training=None, eye_heatmap=None):
        x = self.up_conv1(x, training=training)
        x = self.norm1(x, training=training)
        x = self.lrelu1(x, training=training)
        
        # 应用残差块
        identity = x
        for layer in self.res1:
            x = layer(x, training=training)
        x = x + identity
        
        identity = x
        for layer in self.res2:
            x = layer(x, training=training)
        x = x + identity
        
        x = self.up2(x, training=training)
        x = self.up3(x, training=training)
        x = self.up4(x, training=training)
        
        # 如果有眼睛热力图，则在最后的上采样层应用
        if eye_heatmap is not None:
            # 将热力图调整到与x相同的大小
            eye_heatmap_resized = tf.image.resize(
                eye_heatmap, tf.shape(x)[1:3],
                method=tf.image.ResizeMethod.BILINEAR
            )
            # 扩展维度以匹配特征图
            eye_heatmap_expanded = tf.expand_dims(eye_heatmap_resized, axis=-1)
            eye_heatmap_expanded = tf.cast(eye_heatmap_expanded, dtype=x.dtype)
            
            # 应用眼睛热力图注意力
            attention = 1.0 + 0.5 * eye_heatmap_expanded  # 增强眼睛区域特征
            x = x * attention
        
        x = self.up5(x, training=training)
        x = self.output_conv(x, training=training)
        
        return x


class Encoder(tf.keras.Model):
    """编码器 - 将人脸图像编码为潜在特征向量"""
    
    def __init__(self, latent_dim: int = 1024):
        super(Encoder, self).__init__()
        
        # 编码器网络
        self.conv1 = ConvBlock(64, 5, 2, use_norm=False)  # 128x128
        self.conv2 = ConvBlock(128, 5, 2)  # 64x64
        self.conv3 = ConvBlock(256, 5, 2)  # 32x32
        self.conv4 = ConvBlock(512, 5, 2)  # 16x16
        self.conv5 = ConvBlock(1024, 5, 2)  # 8x8
        
        # 输出潜在特征向量
        self.latent_conv = tf.keras.layers.Conv2D(
            latent_dim, 4, padding='valid',
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )  # 5x5
        
    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.latent_conv(x, training=training)
        return x


class Decoder(tf.keras.Model):
    """解码器 - 将潜在特征向量解码为人脸图像"""
    
    def __init__(self, output_channels: int = 3):
        super(Decoder, self).__init__()
        
        # 解码器网络
        self.up_conv1 = tf.keras.layers.Conv2DTranspose(
            1024, 4, padding='valid',
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )  # 8x8
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.lrelu1 = tf.keras.layers.LeakyReLU(0.2)
        
        self.up2 = UpSampleBlock(512)  # 16x16
        self.up3 = UpSampleBlock(256)  # 32x32
        self.up4 = UpSampleBlock(128)  # 64x64
        self.up5 = UpSampleBlock(64)   # 128x128
        
        # 输出图像
        self.output_conv = tf.keras.layers.Conv2D(
            output_channels, 5, padding='same', activation='tanh',
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)
        )
        
    def call(self, x, training=None, eye_heatmap=None):
        x = self.up_conv1(x, training=training)
        x = self.norm1(x, training=training)
        x = self.lrelu1(x, training=training)
        
        x = self.up2(x, training=training)
        x = self.up3(x, training=training)
        x = self.up4(x, training=training)
        
        # 如果有眼睛热力图，则在最后的上采样层应用
        if eye_heatmap is not None:
            # 将热力图调整到与x相同的大小
            eye_heatmap_resized = tf.image.resize(
                eye_heatmap, tf.shape(x)[1:3],
                method=tf.image.ResizeMethod.BILINEAR
            )
            # 扩展维度以匹配特征图
            eye_heatmap_expanded = tf.expand_dims(eye_heatmap_resized, axis=-1)
            eye_heatmap_expanded = tf.cast(eye_heatmap_expanded, dtype=x.dtype)
            
            # 应用眼睛热力图注意力
            attention = 1.0 + 0.5 * eye_heatmap_expanded  # 增强眼睛区域特征
            x = x * attention
        
        x = self.up5(x, training=training)
        x = self.output_conv(x, training=training)
        
        return x


class FaceSwapAutoEncoder(tf.keras.Model):
    """面部替换自编码器"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), 
                 latent_dim: int = 1024, use_deepfakes: bool = True,
                 use_gpu: bool = True):
        super(FaceSwapAutoEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.use_deepfakes = use_deepfakes
        
        # 检查是否需要配置GPU
        if use_gpu:
            self.gpu_available = configure_gpu()
        else:
            self.gpu_available = False
            logger.info("未启用GPU加速")
        
        # 选择编码器和解码器
        if use_deepfakes:
            self.encoder = DeepFakesEncoder(latent_dim)
            self.decoder = DeepFakesDecoder()
            logger.info("使用DeepFakes风格的自编码器网络")
        else:
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder()
            logger.info("使用基础自编码器网络")
    
    def call(self, x, training=None, eye_heatmap=None):
        # 编码
        latent = self.encoder(x, training=training)
        
        # 解码
        output = self.decoder(latent, training=training, eye_heatmap=eye_heatmap)
        
        return output
    
    def build_model(self):
        """构建完整模型以支持保存和加载"""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        eye_heatmap_input = tf.keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1], 1))
        
        # 编码
        latent = self.encoder(inputs)
        
        # 解码
        outputs = self.decoder(latent, eye_heatmap=eye_heatmap_input)
        
        # 创建模型
        model = tf.keras.Model(inputs=[inputs, eye_heatmap_input], outputs=outputs)
        
        return model
    
    def swap_faces(self, source_face: np.ndarray, target_face: np.ndarray, 
                   source_eye_heatmap: Optional[np.ndarray] = None, 
                   target_eye_heatmap: Optional[np.ndarray] = None,
                   blend_factor: float = 0.7) -> np.ndarray:
        """执行面部替换
        
        Args:
            source_face: 源人脸图像，形状为(256, 256, 3)，值域为[0, 1]
            target_face: 目标人脸图像，形状为(256, 256, 3)，值域为[0, 1]
            source_eye_heatmap: 源图像眼睛热力图，形状为(256, 256)，值域为[0, 1]
            target_eye_heatmap: 目标图像眼睛热力图，形状为(256, 256)，值域为[0, 1]
            blend_factor: 混合因子，控制源脸和目标脸的混合比例，范围为[0, 1]，值越大源脸特征越明显
            
        Returns:
            替换后的人脸图像，形状为(256, 256, 3)，值域为[0, 1]
        """
        # 检查输入
        if source_face is None or target_face is None:
            logger.error("输入人脸为空")
            return np.zeros_like(target_face)
        
        # 确保人脸大小一致
        if source_face.shape != target_face.shape:
            logger.warning("源人脸和目标人脸大小不一致，调整大小")
            source_face = cv2.resize(source_face, (target_face.shape[1], target_face.shape[0]))
        
        # 记录处理时间
        start_time = time.time()
        
        # 预处理图像
        source_face_norm = source_face.astype(np.float32) / 255.0
        source_face_norm = source_face_norm * 2.0 - 1.0  # 转换为[-1, 1]范围
        
        target_face_norm = target_face.astype(np.float32) / 255.0
        target_face_norm = target_face_norm * 2.0 - 1.0  # 转换为[-1, 1]范围
        
        # 添加批次维度
        source_face_tensor = tf.expand_dims(source_face_norm, axis=0)
        target_face_tensor = tf.expand_dims(target_face_norm, axis=0)
        
        # 处理眼睛热力图
        if source_eye_heatmap is None:
            source_eye_heatmap = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.float32)
        
        if target_eye_heatmap is None:
            target_eye_heatmap = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.float32)
        
        # 添加批次维度和通道维度
        source_eye_tensor = tf.expand_dims(tf.expand_dims(source_eye_heatmap, axis=0), axis=-1)
        target_eye_tensor = tf.expand_dims(tf.expand_dims(target_eye_heatmap, axis=0), axis=-1)
        
        try:
            # 如果有GPU可用，确保张量在GPU上
            if self.gpu_available:
                with tf.device('/GPU:0'):
                    # 使用源人脸编码，目标人脸解码
                    source_latent = self.encoder(source_face_tensor, training=False)
                    
                    # 如果需要混合源脸和目标脸的特征
                    if blend_factor < 1.0:
                        # 编码目标人脸
                        target_latent = self.encoder(target_face_tensor, training=False)
                        
                        # 混合潜在特征
                        blended_latent = source_latent * blend_factor + target_latent * (1.0 - blend_factor)
                        
                        # 解码混合特征
                        result_tensor = self.decoder(blended_latent, training=False, eye_heatmap=target_eye_tensor)
                    else:
                        # 直接使用源脸特征解码
                        result_tensor = self.decoder(source_latent, training=False, eye_heatmap=target_eye_tensor)
            else:
                # 使用源人脸编码，目标人脸解码
                source_latent = self.encoder(source_face_tensor, training=False)
                
                # 如果需要混合源脸和目标脸的特征
                if blend_factor < 1.0:
                    # 编码目标人脸
                    target_latent = self.encoder(target_face_tensor, training=False)
                    
                    # 混合潜在特征
                    blended_latent = source_latent * blend_factor + target_latent * (1.0 - blend_factor)
                    
                    # 解码混合特征
                    result_tensor = self.decoder(blended_latent, training=False, eye_heatmap=target_eye_tensor)
                else:
                    # 直接使用源脸特征解码
                    result_tensor = self.decoder(source_latent, training=False, eye_heatmap=target_eye_tensor)
            
            # 转换回NumPy数组
            result = result_tensor.numpy()[0]
            result = (result + 1.0) / 2.0  # 转换回[0, 1]范围
            result = np.clip(result, 0.0, 1.0)
            result = (result * 255.0).astype(np.uint8)
            
            # 记录处理时间
            process_time = time.time() - start_time
            logger.debug(f"面部替换完成，处理时间: {process_time:.4f}秒")
            
            return result
        except Exception as e:
            logger.error(f"面部替换出错: {str(e)}")
            return target_face
    
    @tf.function
    def predict_optimized(self, source_tensor, target_eye_tensor=None):
        """使用@tf.function进行图优化加速推理"""
        source_latent = self.encoder(source_tensor, training=False)
        return self.decoder(source_latent, training=False, eye_heatmap=target_eye_tensor)
    
    def save_weights(self, filepath: str):
        """保存模型权重"""
        self.build_model().save_weights(filepath)
        logger.info(f"模型权重已保存到: {filepath}")
    
    def load_weights(self, filepath: str):
        """加载模型权重"""
        try:
            self.build_model().load_weights(filepath)
            logger.info(f"模型权重已从以下位置加载: {filepath}")
            return True
        except Exception as e:
            logger.error(f"加载模型权重时出错: {str(e)}")
            return False
    
    def load_pretrained_model(self, model_dir: str, model_name: str = "deepfakes_model") -> bool:
        """
        加载预训练模型
        
        Args:
            model_dir: 模型目录
            model_name: 模型名称
        
        Returns:
            是否成功加载
        """
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        
        if not os.path.exists(model_path):
            logger.warning(f"预训练模型不存在: {model_path}")
            return False
        
        try:
            self.load_weights(model_path)
            logger.info(f"成功加载预训练模型: {model_path}")
            return True
        except Exception as e:
            logger.error(f"加载预训练模型时出错: {str(e)}")
            return False
    
    def _create_face_mask(self, face_image: np.ndarray) -> np.ndarray:
        """
        创建面部掩码
        
        Args:
            face_image: 人脸图像
            
        Returns:
            面部掩码
        """
        height, width = face_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 创建椭圆掩码
        center = (width // 2, height // 2)
        axes = (width // 2 - width // 8, height // 2 - height // 8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # 平滑掩码边缘
        mask = cv2.GaussianBlur(mask, (51, 51), 30)
        
        return mask

# 在模块加载时调用配置函数
configure_gpu() 