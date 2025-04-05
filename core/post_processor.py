"""
后处理模块 - 负责面部替换后的图像优化

实现面部边缘平滑、颜色校正和图像增强等功能
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, Union, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostProcessor:
    """后处理类，实现面部替换后的图像优化功能"""
    
    def __init__(self, smooth_factor: float = 0.7, color_correction: bool = True, 
                 enhance_face: bool = True):
        """
        初始化后处理器
        
        Args:
            smooth_factor: 平滑因子，值域[0.0-1.0]，0表示无平滑
            color_correction: 是否进行颜色校正
            enhance_face: 是否增强面部细节
        """
        self.smooth_factor = smooth_factor
        self.color_correction = color_correction
        self.enhance_face = enhance_face
        logger.info(f"后处理器初始化完成")
    
    def process(self, original_face: np.ndarray, swapped_face: np.ndarray, 
                 face_mask: Optional[np.ndarray] = None, 
                 landmarks: Optional[List] = None) -> np.ndarray:
        """
        对替换后的面部进行后处理，包括边缘平滑、颜色校正和质量增强
        
        Args:
            original_face: 原始面部图像
            swapped_face: 替换后的面部图像
            face_mask: 面部掩码
            landmarks: 面部特征点
            
        Returns:
            后处理后的面部图像
        """
        # 检查图像尺寸
        if original_face.shape != swapped_face.shape:
            logger.warning(f"原始图像和替换图像大小不一致，调整大小: {original_face.shape} -> {swapped_face.shape}")
            original_face = cv2.resize(original_face, (swapped_face.shape[1], swapped_face.shape[0]))
        
        # 如果没有提供面部掩码，创建一个
        if face_mask is None:
            face_mask = self._create_face_mask(original_face)
        else:
            # 确保掩码尺寸与图像一致
            if face_mask.shape[:2] != original_face.shape[:2]:
                logger.warning(f"掩码尺寸与图像不一致，调整大小: {face_mask.shape} -> {original_face.shape[:2]}")
                face_mask = cv2.resize(face_mask, (original_face.shape[1], original_face.shape[0]))
        
        try:
            # 进行颜色校正
            color_corrected = self._color_correction(original_face, swapped_face)
            
            # 平滑边缘
            smoothed = self._smooth_edges(color_corrected, original_face, face_mask)
            
            # 增强面部
            enhanced = self._enhance_face(smoothed, original_face, face_mask)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"后处理失败: {str(e)}，返回基础混合")
            # 如果处理失败，回退到基本的Alpha混合
            return self._basic_blend(original_face, swapped_face, face_mask)
    
    def _basic_blend(self, original_face: np.ndarray, swapped_face: np.ndarray, 
                   face_mask: np.ndarray) -> np.ndarray:
        """
        当更复杂的处理失败时，执行基本的Alpha混合
        
        Args:
            original_face: 原始面部图像
            swapped_face: 替换后的面部图像
            face_mask: 面部掩码
            
        Returns:
            混合后的图像
        """
        # 确保所有输入尺寸一致
        if original_face.shape != swapped_face.shape:
            swapped_face = cv2.resize(swapped_face, (original_face.shape[1], original_face.shape[0]))
        
        if face_mask.shape[:2] != original_face.shape[:2]:
            face_mask = cv2.resize(face_mask, (original_face.shape[1], original_face.shape[0]))
        
        # 将掩码转换为三通道
        if len(face_mask.shape) == 2:
            mask_3d = np.repeat(face_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        else:
            mask_3d = face_mask.astype(np.float32) / 255.0
        
        # 简单的Alpha混合
        result = (swapped_face * mask_3d + original_face * (1 - mask_3d)).astype(np.uint8)
        
        return result
    
    def _color_correction(self, target_face: np.ndarray, source_face: np.ndarray) -> np.ndarray:
        """
        执行颜色校正，使源图像颜色分布匹配目标图像
        
        Args:
            target_face: 目标面部图像
            source_face: 源面部图像
            
        Returns:
            颜色校正后的源图像
        """
        # 确保尺寸一致
        if target_face.shape != source_face.shape:
            source_face = cv2.resize(source_face, (target_face.shape[1], target_face.shape[0]))
        
        # 转换到LAB颜色空间
        target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
        source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
        
        # 分离通道
        target_l, target_a, target_b = cv2.split(target_lab)
        source_l, source_a, source_b = cv2.split(source_lab)
        
        # 计算每个通道的均值和标准差
        # 使用numpy的统计函数快速计算
        target_l_mean, target_l_std = np.mean(target_l), np.std(target_l)
        target_a_mean, target_a_std = np.mean(target_a), np.std(target_a)
        target_b_mean, target_b_std = np.mean(target_b), np.std(target_b)
        
        source_l_mean, source_l_std = np.mean(source_l), np.std(source_l)
        source_a_mean, source_a_std = np.mean(source_a), np.std(source_a)
        source_b_mean, source_b_std = np.mean(source_b), np.std(source_b)
        
        # 防止除以零
        source_l_std = max(source_l_std, 1e-5)
        source_a_std = max(source_a_std, 1e-5)
        source_b_std = max(source_b_std, 1e-5)
        
        # 修正L通道（亮度）
        corrected_l = ((source_l - source_l_mean) * (target_l_std / source_l_std)) + target_l_mean
        # 修正A和B通道（颜色）
        corrected_a = ((source_a - source_a_mean) * (target_a_std / source_a_std)) + target_a_mean
        corrected_b = ((source_b - source_b_mean) * (target_b_std / source_b_std)) + target_b_mean
        
        # 剪切值到有效范围
        corrected_l = np.clip(corrected_l, 0, 255)
        corrected_a = np.clip(corrected_a, 0, 255)
        corrected_b = np.clip(corrected_b, 0, 255)
        
        # 合并通道
        corrected_lab = cv2.merge([corrected_l.astype(np.uint8), 
                                    corrected_a.astype(np.uint8), 
                                    corrected_b.astype(np.uint8)])
        
        # 转换回BGR颜色空间
        corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        return corrected_bgr
    
    def _smooth_edges(self, swapped_face: np.ndarray, original_face: np.ndarray, 
                      face_mask: np.ndarray) -> np.ndarray:
        """
        平滑面部边缘
        
        Args:
            swapped_face: 替换后的面部图像
            original_face: 原始面部图像
            face_mask: 面部掩码
            
        Returns:
            边缘平滑后的图像
        """
        # 确保尺寸一致
        if swapped_face.shape != original_face.shape:
            swapped_face = cv2.resize(swapped_face, (original_face.shape[1], original_face.shape[0]))
        
        if face_mask.shape[:2] != original_face.shape[:2]:
            face_mask = cv2.resize(face_mask, (original_face.shape[1], original_face.shape[0]))
            
        # 创建边缘区域掩码 - 腐蚀原掩码以获取边缘区域
        kernel = np.ones((9, 9), np.uint8)
        eroded_mask = cv2.erode(face_mask, kernel, iterations=3)
        edge_mask = face_mask - eroded_mask
        
        # 对边缘区域应用高斯模糊
        blurred_swapped = cv2.GaussianBlur(swapped_face, (9, 9), 0)
        
        # 将边缘掩码转换为3通道
        edge_mask_3d = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        
        # 在边缘区域混合原始图像和模糊图像
        result = swapped_face.copy()
        result = (result * (1 - edge_mask_3d) + blurred_swapped * edge_mask_3d).astype(np.uint8)
        
        # 使用原始掩码混合结果和原始图像
        mask_3d = np.repeat(face_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        result = (result * mask_3d + original_face * (1 - mask_3d)).astype(np.uint8)
        
        return result
    
    def _enhance_face(self, face_image: np.ndarray, original_face: np.ndarray, 
                      face_mask: np.ndarray) -> np.ndarray:
        """
        增强面部图像质量
        
        Args:
            face_image: 需要增强的面部图像
            original_face: 原始面部图像
            face_mask: 面部掩码
            
        Returns:
            增强后的图像
        """
        # 确保尺寸一致
        if face_image.shape != original_face.shape:
            logger.warning(f"增强前图像尺寸不一致: {face_image.shape} vs {original_face.shape}")
            face_image = cv2.resize(face_image, (original_face.shape[1], original_face.shape[0]))
        
        if face_mask.shape[:2] != original_face.shape[:2]:
            logger.warning(f"掩码尺寸与图像不一致: {face_mask.shape[:2]} vs {original_face.shape[:2]}")
            face_mask = cv2.resize(face_mask, (original_face.shape[1], original_face.shape[0]))
            
        try:
            # 检查图像是否有3个通道
            if len(face_image.shape) != 3 or face_image.shape[2] != 3:
                logger.warning(f"面部图像通道数不正确: {face_image.shape}")
                return face_image
                
            if len(original_face.shape) != 3 or original_face.shape[2] != 3:
                logger.warning(f"原始图像通道数不正确: {original_face.shape}")
                return face_image
            
            # 分离通道并检查每个通道的尺寸
            try:
                b, g, r = cv2.split(face_image)
                orig_b, orig_g, orig_r = cv2.split(original_face)
                
                # 检查通道尺寸
                if b.shape != g.shape or b.shape != r.shape:
                    logger.warning(f"面部图像通道尺寸不一致: b{b.shape}, g{g.shape}, r{r.shape}")
                    # 重新调整所有通道到相同尺寸
                    target_shape = face_image.shape[:2]
                    b = cv2.resize(b, (target_shape[1], target_shape[0]))
                    g = cv2.resize(g, (target_shape[1], target_shape[0]))
                    r = cv2.resize(r, (target_shape[1], target_shape[0]))
                
                # 对每个通道单独进行增强
                r_enhanced = self._enhance_channel(r, orig_r)
                g_enhanced = self._enhance_channel(g, orig_g)
                b_enhanced = self._enhance_channel(b, orig_b)
                
                # 再次检查通道尺寸
                if b_enhanced.shape != g_enhanced.shape or b_enhanced.shape != r_enhanced.shape:
                    logger.warning(f"增强后通道尺寸不一致，调整到相同尺寸")
                    target_shape = (face_image.shape[0], face_image.shape[1])
                    b_enhanced = cv2.resize(b_enhanced, (target_shape[1], target_shape[0]))
                    g_enhanced = cv2.resize(g_enhanced, (target_shape[1], target_shape[0]))
                    r_enhanced = cv2.resize(r_enhanced, (target_shape[1], target_shape[0]))
                
                # 对每个通道进行额外检查，确保数据类型和数值范围正确
                b_enhanced = np.clip(b_enhanced, 0, 255).astype(np.uint8)
                g_enhanced = np.clip(g_enhanced, 0, 255).astype(np.uint8)
                r_enhanced = np.clip(r_enhanced, 0, 255).astype(np.uint8)
                
                # 合并通道
                try:
                    enhanced = cv2.merge([b_enhanced, g_enhanced, r_enhanced])
                except Exception as merge_error:
                    logger.error(f"通道合并失败: {str(merge_error)}")
                    logger.debug(f"通道形状: b{b_enhanced.shape}, g{g_enhanced.shape}, r{r_enhanced.shape}")
                    # 如果合并失败，返回原始图像
                    return face_image
            except Exception as channel_error:
                logger.error(f"通道处理失败: {str(channel_error)}")
                return face_image
            
            # 应用轻微锐化
            try:
                sharpen_kernel = np.array([[-0.1, -0.1, -0.1],
                                            [-0.1,  2.0, -0.1],
                                            [-0.1, -0.1, -0.1]])
                enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
            except Exception as sharpen_error:
                logger.warning(f"锐化失败: {str(sharpen_error)}")
                # 继续处理而不应用锐化
            
            # 与原始图像混合，使用原始面部掩码
            try:
                # 确保掩码形状正确
                if len(face_mask.shape) == 3 and face_mask.shape[2] > 1:
                    # 如果掩码是多通道的，取第一个通道
                    face_mask = face_mask[:,:,0]
                
                mask_3d = np.repeat(face_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
                
                # 确保增强图像和原始图像尺寸一致
                if enhanced.shape != original_face.shape:
                    enhanced = cv2.resize(enhanced, (original_face.shape[1], original_face.shape[0]))
                
                result = (enhanced * mask_3d + original_face * (1 - mask_3d)).astype(np.uint8)
                return result
            except Exception as blend_error:
                logger.error(f"图像混合失败: {str(blend_error)}")
                return face_image
            
        except Exception as e:
            logger.error(f"增强面部时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果增强失败，返回未增强的图像
            return face_image
    
    def _enhance_channel(self, channel: np.ndarray, original_channel: np.ndarray) -> np.ndarray:
        """
        增强单个颜色通道
        
        Args:
            channel: 需要增强的通道
            original_channel: 原始通道
            
        Returns:
            增强后的通道
        """
        # 确保尺寸一致
        if channel.shape != original_channel.shape:
            channel = cv2.resize(channel, (original_channel.shape[1], original_channel.shape[0]))
            
        # 应用CLAHE（对比度受限的自适应直方图均衡化）增强对比度
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(channel)
        
        # 混合原始通道和增强通道
        alpha = 0.7  # 控制增强的强度
        blended = cv2.addWeighted(enhanced, alpha, channel, 1 - alpha, 0)
        
        return blended
    
    def _create_face_mask(self, face_image: np.ndarray, face_landmarks: Optional[List] = None) -> np.ndarray:
        """
        创建面部掩码
        
        Args:
            face_image: 面部图像
            face_landmarks: 面部特征点，如果有则用于创建更精确的掩码
            
        Returns:
            面部掩码，单通道图像
        """
        # 获取图像尺寸
        height, width = face_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 如果有特征点，使用凸包
        if face_landmarks and len(face_landmarks) > 5:
            try:
                # 将特征点转换为适合凸包的格式
                points = np.array(face_landmarks, dtype=np.int32)
                
                # 计算凸包
                hull = cv2.convexHull(points)
                
                # 创建掩码
                cv2.fillConvexPoly(mask, hull, 255)
                
                # 平滑掩码边缘
                mask = cv2.GaussianBlur(mask, (31, 31), 11)
                
                return mask
            except Exception as e:
                logger.warning(f"使用特征点创建掩码失败: {str(e)}，回退到椭圆掩码")
        
        # 如果没有特征点或者凸包创建失败，使用椭圆掩码
        center = (width // 2, height // 2)
        axes = (width // 2 - width // 10, height // 2 - height // 10)  # 使椭圆略小于图像
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # 平滑边缘
        mask = cv2.GaussianBlur(mask, (31, 31), 11)
        
        return mask 