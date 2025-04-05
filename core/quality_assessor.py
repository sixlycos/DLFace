"""
质量评估模块 - 用于评估生成图像的质量

使用SSIM（结构相似性指数）和PSNR（峰值信噪比）等指标评估生成图像的质量
可视化评估结果，并提供质量报告功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Union
import os
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualityAssessor:
    """图像质量评估类，用于评估生成图像的质量"""
    
    def __init__(self):
        """初始化质量评估器"""
        logger.info("质量评估器初始化完成")
    
    def assess_quality(self, original_image: np.ndarray, generated_image: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估生成图像的质量
        
        Args:
            original_image: 原始图像
            generated_image: 生成的图像
            mask: 评估区域掩码，如果为None则评估整个图像
            
        Returns:
            包含评估指标的字典，如SSIM、PSNR等
        """
        # 检查输入
        if original_image is None or generated_image is None:
            logger.error("输入图像为空")
            return {'ssim': 0.0, 'psnr': 0.0}
        
        # 确保图像大小一致
        if original_image.shape != generated_image.shape:
            logger.warning("原始图像和生成图像大小不一致，调整大小")
            generated_image = cv2.resize(generated_image, 
                                        (original_image.shape[1], original_image.shape[0]))
        
        # 转换为灰度图像用于SSIM计算
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_image
            generated_gray = generated_image
        
        # 如果提供了掩码，应用掩码
        if mask is not None:
            # 确保掩码大小与图像一致
            if mask.shape[:2] != original_gray.shape[:2]:
                mask = cv2.resize(mask, (original_gray.shape[1], original_gray.shape[0]))
            
            # 确保掩码为单通道
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # 创建掩码的布尔版本
            mask_bool = mask > 128
            
            # 应用掩码
            original_masked = np.where(mask_bool, original_gray, 0)
            generated_masked = np.where(mask_bool, generated_gray, 0)
            
            # 计算SSIM
            try:
                ssim_value = ssim(original_masked, generated_masked)
            except Exception as e:
                logger.error(f"计算SSIM时出错: {str(e)}")
                ssim_value = 0.0
            
            # 计算PSNR
            try:
                psnr_value = psnr(original_masked, generated_masked)
            except Exception as e:
                logger.error(f"计算PSNR时出错: {str(e)}")
                psnr_value = 0.0
        else:
            # 计算SSIM
            try:
                ssim_value = ssim(original_gray, generated_gray)
            except Exception as e:
                logger.error(f"计算SSIM时出错: {str(e)}")
                ssim_value = 0.0
            
            # 计算PSNR
            try:
                psnr_value = psnr(original_gray, generated_gray)
            except Exception as e:
                logger.error(f"计算PSNR时出错: {str(e)}")
                psnr_value = 0.0
        
        # 返回评估结果
        return {
            'ssim': ssim_value,
            'psnr': psnr_value
        }
    
    def assess_batch(self, original_images: List[np.ndarray], generated_images: List[np.ndarray], 
                    masks: Optional[List[np.ndarray]] = None) -> List[Dict[str, float]]:
        """
        批量评估生成图像的质量
        
        Args:
            original_images: 原始图像列表
            generated_images: 生成的图像列表
            masks: 评估区域掩码列表，如果为None则评估整个图像
            
        Returns:
            包含评估指标的字典列表
        """
        # 检查输入
        if len(original_images) != len(generated_images):
            logger.error("原始图像和生成图像数量不一致")
            return []
        
        # 如果提供了掩码，检查数量
        if masks is not None and len(masks) != len(original_images):
            logger.error("掩码数量与图像数量不一致")
            return []
        
        # 批量评估
        results = []
        for i in range(len(original_images)):
            # 获取当前掩码
            mask = masks[i] if masks is not None else None
            
            # 评估当前图像
            result = self.assess_quality(original_images[i], generated_images[i], mask)
            results.append(result)
        
        return results
    
    def generate_quality_heatmap(self, original_image: np.ndarray, generated_image: np.ndarray,
                               window_size: int = 8) -> np.ndarray:
        """
        生成质量热力图
        
        Args:
            original_image: 原始图像
            generated_image: 生成的图像
            window_size: 滑动窗口大小
            
        Returns:
            SSIM热力图
        """
        # 检查输入
        if original_image is None or generated_image is None:
            logger.error("输入图像为空")
            return np.zeros((100, 100), dtype=np.float32)
        
        # 确保图像大小一致
        if original_image.shape != generated_image.shape:
            logger.warning("原始图像和生成图像大小不一致，调整大小")
            generated_image = cv2.resize(generated_image, 
                                        (original_image.shape[1], original_image.shape[0]))
        
        # 转换为灰度图像
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_image
            generated_gray = generated_image
        
        # 计算SSIM热力图
        try:
            # 调用SSIM函数，设置full=True以获取完整热力图
            ssim_value, ssim_map = ssim(original_gray, generated_gray, 
                                     win_size=window_size, full=True)
            
            # 将SSIM热力图转换为彩色图像
            ssim_map_normalized = (ssim_map - ssim_map.min()) / (ssim_map.max() - ssim_map.min())
            ssim_heatmap = cv2.applyColorMap((ssim_map_normalized * 255).astype(np.uint8), 
                                           cv2.COLORMAP_JET)
            
            return ssim_heatmap
        except Exception as e:
            logger.error(f"生成质量热力图时出错: {str(e)}")
            return np.zeros_like(original_image)
    
    def visualize_quality(self, original_image: np.ndarray, generated_image: np.ndarray,
                        output_path: Optional[str] = None) -> np.ndarray:
        """
        可视化质量评估结果
        
        Args:
            original_image: 原始图像
            generated_image: 生成的图像
            output_path: 输出路径，如果为None则不保存
            
        Returns:
            可视化结果图像
        """
        # 检查输入
        if original_image is None or generated_image is None:
            logger.error("输入图像为空")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 确保图像大小一致
        if original_image.shape != generated_image.shape:
            logger.warning("原始图像和生成图像大小不一致，调整大小")
            generated_image = cv2.resize(generated_image, 
                                        (original_image.shape[1], original_image.shape[0]))
        
        # 评估质量
        quality = self.assess_quality(original_image, generated_image)
        
        # 生成热力图
        heatmap = self.generate_quality_heatmap(original_image, generated_image)
        
        # 调整大小以匹配原始图像
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # 创建差异图像
        diff = cv2.absdiff(original_image, generated_image)
        
        # 创建可视化图像
        h, w = original_image.shape[:2]
        
        # 确保所有图像都是三通道
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        if len(generated_image.shape) == 2:
            generated_image = cv2.cvtColor(generated_image, cv2.COLOR_GRAY2BGR)
        if len(diff.shape) == 2:
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        # 创建画布
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # 放置四个图像
        canvas[:h, :w] = original_image
        canvas[:h, w:] = generated_image
        canvas[h:, :w] = diff
        canvas[h:, w:] = heatmap
        
        # 添加标题和评估结果
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Generated", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Difference", (10, h + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "SSIM Heatmap", (w + 10, h + 30), font, 1, (255, 255, 255), 2)
        
        # 添加评估结果
        cv2.putText(canvas, f"SSIM: {quality['ssim']:.4f}", (10, h + 70), font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"PSNR: {quality['psnr']:.2f} dB", (10, h + 100), font, 0.7, (255, 255, 255), 2)
        
        # 如果提供了输出路径，保存结果
        if output_path is not None:
            try:
                cv2.imwrite(output_path, canvas)
                logger.info(f"可视化结果已保存到: {output_path}")
            except Exception as e:
                logger.error(f"保存可视化结果时出错: {str(e)}")
        
        return canvas
    
    def generate_quality_report(self, original_images: List[np.ndarray], 
                              generated_images: List[np.ndarray],
                              output_dir: str) -> Dict[str, float]:
        """
        生成质量评估报告
        
        Args:
            original_images: 原始图像列表
            generated_images: 生成的图像列表
            output_dir: 输出目录
            
        Returns:
            平均质量指标字典
        """
        # 检查输入
        if len(original_images) != len(generated_images):
            logger.error("原始图像和生成图像数量不一致")
            return {'avg_ssim': 0.0, 'avg_psnr': 0.0}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 批量评估
        results = self.assess_batch(original_images, generated_images)
        
        # 计算平均指标
        avg_ssim = sum(result['ssim'] for result in results) / len(results)
        avg_psnr = sum(result['psnr'] for result in results) / len(results)
        
        # 可视化部分结果
        for i in range(min(5, len(original_images))):
            output_path = os.path.join(output_dir, f"quality_vis_{i}.png")
            self.visualize_quality(original_images[i], generated_images[i], output_path)
        
        # 生成报告文本
        report_path = os.path.join(output_dir, "quality_report.txt")
        try:
            with open(report_path, 'w') as f:
                f.write("图像质量评估报告\n")
                f.write("================\n\n")
                f.write(f"评估图像数量: {len(results)}\n\n")
                f.write(f"平均SSIM: {avg_ssim:.4f}\n")
                f.write(f"平均PSNR: {avg_psnr:.2f} dB\n\n")
                f.write("各图像评估结果:\n\n")
                
                for i, result in enumerate(results):
                    f.write(f"图像 {i+1}:\n")
                    f.write(f"  SSIM: {result['ssim']:.4f}\n")
                    f.write(f"  PSNR: {result['psnr']:.2f} dB\n\n")
            
            logger.info(f"质量评估报告已保存到: {report_path}")
        except Exception as e:
            logger.error(f"保存质量评估报告时出错: {str(e)}")
        
        # 生成可视化报告（图表）
        try:
            # 创建柱状图
            plt.figure(figsize=(12, 6))
            
            # SSIM图表
            plt.subplot(1, 2, 1)
            ssim_values = [result['ssim'] for result in results]
            plt.bar(range(len(ssim_values)), ssim_values)
            plt.axhline(y=avg_ssim, color='r', linestyle='--', label=f'Avg: {avg_ssim:.4f}')
            plt.title('SSIM Values')
            plt.xlabel('Image Index')
            plt.ylabel('SSIM')
            plt.legend()
            
            # PSNR图表
            plt.subplot(1, 2, 2)
            psnr_values = [result['psnr'] for result in results]
            plt.bar(range(len(psnr_values)), psnr_values)
            plt.axhline(y=avg_psnr, color='r', linestyle='--', label=f'Avg: {avg_psnr:.2f} dB')
            plt.title('PSNR Values')
            plt.xlabel('Image Index')
            plt.ylabel('PSNR (dB)')
            plt.legend()
            
            # 保存图表
            plt.tight_layout()
            vis_path = os.path.join(output_dir, "quality_metrics.png")
            plt.savefig(vis_path)
            plt.close()
            
            logger.info(f"质量指标可视化已保存到: {vis_path}")
        except Exception as e:
            logger.error(f"生成质量指标可视化时出错: {str(e)}")
        
        return {
            'avg_ssim': avg_ssim,
            'avg_psnr': avg_psnr
        } 