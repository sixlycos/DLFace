"""
视频处理模块 - 负责视频的加载、处理和保存

包含视频处理的核心功能，如面部检测、替换和后处理
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Optional, Any, Union
import logging
from tqdm import tqdm
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import glob

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入自定义模块
from .face_detector import FaceDetector
from .face_swapper import FaceSwapper
from .post_processor import PostProcessor
from .eye_tracker import EyeTracker

# 尝试导入可选模块
try:
    from .quality_assessor import QualityAssessor
    EYE_TRACKER_AVAILABLE = True
except (ImportError, AttributeError) as e:
    EYE_TRACKER_AVAILABLE = False
    logger.warning(f"眼部跟踪模块导入失败: {str(e)}")

# 导入工具函数
from .utils import read_image_with_path_fix


class VideoProcessor:
    """视频处理器，用于处理源人脸和目标视频的面部替换任务"""
    
    def __init__(self, use_cuda: bool = False, num_workers: int = 4, source_path: str = None, face_detector_cls=None):
        """
        初始化视频处理器
        
        参数:
            use_cuda: 是否使用CUDA加速
            num_workers: 多线程处理的工作线程数量
            source_path: 源人脸图像路径
            face_detector_cls: 人脸检测器类，如果为None则使用默认检测器
        """
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.face_detector = FaceDetector() if face_detector_cls is None else face_detector_cls()
        self.face_swapper = FaceSwapper(use_cuda=use_cuda)
        self.post_processor = PostProcessor()
        self.eye_tracker = EyeTracker(use_cuda=use_cuda)
        self.progress = 0.0
        self.frames_to_process = 0  # 添加处理总帧数属性
        self._lock = threading.Lock()  # 用于线程安全的进度更新
        logger.info(f"初始化VideoProcessor, CUDA加速: {'启用' if use_cuda else '禁用'}, 工作线程: {num_workers}")
        
        # 加载源人脸
        if source_path:
            self.load_source_face(source_path)
    
    def load_source_face(self, source_path: str) -> bool:
        """
        加载源人脸图像
        
        Args:
            source_path: 源人脸图像路径
            
        Returns:
            是否成功加载
        """
        # 检查路径是否存在
        if not os.path.exists(source_path):
            logger.error(f"源人脸图像不存在: {source_path}")
            return False
        
        # 读取图像
        source_face_img = read_image_with_path_fix(source_path)
        if source_face_img is None:
            logger.error(f"无法读取源人脸图像: {source_path}")
            return False
        
        # 检测人脸
        faces = self.face_detector.detect_faces(source_face_img)
        if not faces:
            logger.error(f"源人脸图像中未检测到人脸: {source_path}")
            return False
        
        # 获取第一个人脸
        face = faces[0]
        
        # 对齐人脸
        self.source_face = self.face_detector.align_face(source_face_img, face)
        if self.source_face is None:
            logger.error(f"无法对齐源人脸图像: {source_path}")
            return False
        
        # 保存源人脸特征点
        self.source_landmarks = face.get('landmarks', [])
        
        logger.info(f"成功加载源人脸图像: {source_path}")
        return True
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        参数:
            model_path: 模型文件路径
            
        返回:
            加载是否成功
        """
        try:
            logger.info(f"正在加载模型: {model_path}")
            if os.path.exists(model_path):
                self.face_swapper.load_model(model_path)
                logger.info("模型加载成功")
                return True
            else:
                logger.warning(f"模型文件不存在: {model_path}")
                # 使用未训练的模型
                logger.info("使用未训练的模型")
                return False
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def _process_frame(self, frame_data, source_face: np.ndarray) -> np.ndarray:
        """
        处理单帧图像 - 使用简化且更可靠的替换方法
        
        Args:
            frame_data: 帧数据，可以是图像路径(str)或直接的图像数据(np.ndarray)
            source_face: 源人脸图像
            
        Returns:
            处理后的帧图像，如果没有检测到人脸则返回None
        """
        # 读取帧
        if isinstance(frame_data, str):
            frame = read_image_with_path_fix(frame_data)
            if frame is None:
                logger.warning(f"无法读取帧: {frame_data}")
                return None
        else:
            frame = frame_data  # 直接使用传入的图像数据
        
        # 检测人脸
        faces = self.face_detector.detect_faces(frame)
        
        # 如果没有检测到人脸，返回None而不是原帧，表示这帧可以跳过
        if not faces:
            logger.info(f"当前帧未检测到人脸，跳过处理")
            return None
        
        # 处理每个检测到的人脸
        result_frame = frame.copy()
        face_processed = False
        
        for face in faces:
            try:
                # 获取人脸区域
                bbox = face.get('bbox')
                if not bbox:
                    logger.warning("未能获取有效的人脸边界框")
                    continue
                    
                x, y, w, h = bbox
                
                # 确保边界框在图像内部
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    logger.warning(f"人脸边界框超出图像范围: {x},{y},{w},{h}")
                    # 修正边界框
                    x = max(0, x)
                    y = max(0, y)
                    w = min(frame.shape[1] - x, w)
                    h = min(frame.shape[0] - y, h)
                
                # 提取目标人脸区域
                target_face_region = frame[y:y+h, x:x+w]
                
                # 调整源人脸大小以匹配目标人脸区域
                source_face_resized = cv2.resize(source_face, (w, h))
                
                # 直接使用face_swapper的swap_face方法
                swapped_face = self.face_swapper.swap_face(source_face_resized, target_face_region)
                
                if swapped_face is None:
                    logger.warning("面部替换失败，跳过当前人脸")
                    continue
                
                # 确保替换后的人脸尺寸正确
                if swapped_face.shape[:2] != (h, w):
                    logger.info(f"调整替换后的人脸尺寸: {swapped_face.shape[:2]} -> {(h, w)}")
                    swapped_face = cv2.resize(swapped_face, (w, h))
                
                # 创建简单椭圆掩码用于混合
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (w // 2, h // 2)
                axes = (w // 2 - w // 10, h // 2 - h // 10)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (21, 21), 11)  # 使用较大的高斯模糊
                
                # 将掩码转换为3通道并归一化
                mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
                
                # 混合
                roi = result_frame[y:y+h, x:x+w]
                blended = (roi * (1 - mask_3d) + swapped_face * mask_3d).astype(np.uint8)
                
                # 应用到结果帧
                result_frame[y:y+h, x:x+w] = blended
                face_processed = True
                
            except Exception as e:
                logger.error(f"处理人脸时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        if face_processed:
            return result_frame
        else:
            # 如果没有成功处理任何人脸，也返回None
            return None
    
    def process_video(
        self, 
        source_face_path: str, 
        target_video_path: str, 
        output_path: str, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None, 
        fps: Optional[float] = None, 
        resolution: Optional[Tuple[int, int]] = None, 
        show_progress: bool = True,
        roi: Optional[Tuple[int, int, int, int]] = None,
        enable_quality_assessment: bool = False
    ) -> str:
        """
        处理视频

        参数:
            source_face_path: 源人脸图像路径
            target_video_path: 目标视频路径
            output_path: 输出视频路径
            start_frame: 起始帧索引
            end_frame: 结束帧索引，如果为None则处理到最后一帧
            fps: 输出视频的帧率，如果为None则使用原视频帧率
            resolution: 输出视频的分辨率 (宽度, 高度)，如果为None则使用原视频分辨率
            show_progress: 是否显示进度条
            roi: 区域限定 (x, y, width, height)，如果指定则只在该区域内检测人脸
            enable_quality_assessment: 是否启用质量评估

        返回:
            输出视频路径
        """
        # 重置进度
        self.progress = 0.0
        logger.info(f"开始处理视频: {target_video_path}")
        
        # 检查目标视频和源人脸是否存在
        if not os.path.exists(target_video_path):
            error_msg = f"目标视频不存在: {target_video_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not os.path.exists(source_face_path):
            error_msg = f"源人脸图像不存在: {source_face_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 加载源人脸
        logger.info(f"加载源人脸: {source_face_path}")
        source_face_img = read_image_with_path_fix(source_face_path)
        if source_face_img is None:
            error_msg = f"无法读取源人脸图像: {source_face_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 打开视频
        logger.info(f"打开目标视频: {target_video_path}")
        cap = cv2.VideoCapture(target_video_path)
        if not cap.isOpened():
            error_msg = f"无法打开目标视频: {target_video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"视频信息 - 总帧数: {total_frames}, 帧率: {original_fps}, 分辨率: {width}x{height}")
        
        # 设置输出视频参数
        output_fps = fps if fps is not None else original_fps
        if resolution is not None:
            output_width, output_height = resolution
        else:
            output_width, output_height = width, height
        
        logger.info(f"输出视频参数 - 帧率: {output_fps}, 分辨率: {output_width}x{output_height}")
        
        # 设置开始和结束帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        
        # 计算需要处理的帧数
        frames_to_process = end_frame - start_frame
        # 设置要处理的总帧数
        self.frames_to_process = frames_to_process
        logger.info(f"需要处理的帧数: {frames_to_process} (从 {start_frame} 到 {end_frame})")
        
        # 创建临时目录存储处理后的帧
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 如果需要质量评估，创建质量评估目录和结果存储
        quality_dir = None
        quality_results = {}
        
        if enable_quality_assessment:
            quality_assessor = QualityAssessor()
            quality_dir = os.path.join(os.path.dirname(output_path), "quality_assessment")
            os.makedirs(quality_dir, exist_ok=True)
            logger.info(f"启用质量评估，结果将保存到: {quality_dir}")
            quality_results = {
                "frames": [],
                "ssim": [],
                "psnr": [],
                "ms_ssim": []
            }
        
        # 处理进度
        processed_frames = 0
        skipped_frames = 0
        success_frames = 0
        
        # 使用tqdm显示进度
        if show_progress:
            progress_bar = tqdm(total=frames_to_process, desc="处理视频", unit="帧")
        
        # 开始时间
        start_time = time.time()
        
        # 创建帧处理函数
        def process_frame_batch(frame_batch, batch_indices):
            """处理一批帧"""
            processed_batch = []
            for i, (frame, frame_idx) in enumerate(zip(frame_batch, batch_indices)):
                # 处理帧
                result_frame = self._process_frame(frame, source_face_img)
                
                # 如果未检测到人脸或处理失败，使用原始帧
                if result_frame is None:
                    logger.info(f"第 {start_frame + frame_idx} 帧未检测到人脸或处理失败，使用原始帧")
                    # 确保frame是一个ndarray类型的图像数据
                    result_frame = frame.copy() if isinstance(frame, np.ndarray) else cv2.imread(frame)
                    # 调整帧大小以匹配输出分辨率
                    if result_frame.shape[1] != output_width or result_frame.shape[0] != output_height:
                        result_frame = cv2.resize(result_frame, (output_width, output_height))
                    is_face_processed = False
                else:
                    # 调整帧大小以匹配输出分辨率
                    if result_frame.shape[1] != output_width or result_frame.shape[0] != output_height:
                        result_frame = cv2.resize(result_frame, (output_width, output_height))
                    is_face_processed = True
                
                # 如果启用质量评估且是处理后的帧
                if enable_quality_assessment and is_face_processed:
                    try:
                        # 评估图像质量
                        quality = quality_assessor.assess_quality(source_face_img, result_frame)
                        
                        # 存储结果
                        with self._lock:
                            quality_results["frames"].append(start_frame + frame_idx)
                            quality_results["ssim"].append(quality["ssim"])
                            quality_results["psnr"].append(quality["psnr"])
                            
                            # 每10帧保存一个质量可视化
                            if frame_idx % 10 == 0:
                                # 生成质量可视化
                                vis_path = os.path.join(quality_dir, f"quality_vis_{start_frame + frame_idx:04d}.png")
                                quality_assessor.visualize_quality(source_face_img, result_frame, vis_path)
                    except Exception as e:
                        logger.error(f"质量评估第 {start_frame + frame_idx} 帧时出错: {str(e)}")
                
                # 保存处理后的帧
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, result_frame)
                processed_batch.append((frame_path, is_face_processed))
            
            return processed_batch
        
        try:
            # 创建线程池
            executor = ThreadPoolExecutor(max_workers=self.num_workers)
            futures = []
            
            # 批量处理大小
            batch_size = min(10, max(1, frames_to_process // self.num_workers))
            
            # 收集要处理的帧
            frame_batches = []
            batch_indices = []
            current_batch = []
            current_indices = []
            
            for frame_idx in range(frames_to_process):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_batch.append(frame)
                current_indices.append(frame_idx)
                
                if len(current_batch) >= batch_size:
                    frame_batches.append(current_batch)
                    batch_indices.append(current_indices)
                    current_batch = []
                    current_indices = []
            
            # 添加最后一个批次
            if current_batch:
                frame_batches.append(current_batch)
                batch_indices.append(current_indices)
            
            # 提交批处理任务
            for batch, indices in zip(frame_batches, batch_indices):
                future = executor.submit(process_frame_batch, batch, indices)
                futures.append(future)
            
            # 处理结果
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    for frame_path, is_processed in batch_results:
                        processed_frames += 1
                        if is_processed:
                            success_frames += 1
                        else:
                            skipped_frames += 1
                            
                        # 更新进度
                        self.progress = processed_frames / frames_to_process
                        if show_progress:
                            progress_bar.update(1)
                            
                        # 如果视频比较长，每100帧记录一次日志
                        if processed_frames % 100 == 0:
                            elapsed = time.time() - start_time
                            fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                            logger.info(f"已处理 {processed_frames}/{frames_to_process} 帧 "
                                      f"({processed_frames/frames_to_process*100:.1f}%), "
                                      f"处理速度: {fps_processing:.2f} 帧/秒")
                except Exception as e:
                    logger.error(f"处理帧批次时出错: {str(e)}")
            
            # 关闭线程池
            executor.shutdown()
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
            
            # 按顺序写入帧
            logger.info("正在将处理后的帧合成为视频...")
            for i in range(processed_frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        out.write(frame)
                    # 删除临时帧文件
                    os.remove(frame_path)
            
            # 释放资源
            out.release()
            
        except Exception as e:
            logger.error(f"处理视频时发生异常: {str(e)}")
            raise e
        
        finally:
            # 关闭进度条
            if show_progress and 'progress_bar' in locals():
                progress_bar.close()
            
            # 释放资源
            cap.release()
            
            # 清理临时目录
            try:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"清理临时目录时出错: {str(e)}")
            
            # 计算总处理时间
            total_time = time.time() - start_time
            fps_avg = processed_frames / total_time if total_time > 0 else 0
            
            # 输出处理统计信息
            logger.info(f"视频处理完成! 总共处理了 {processed_frames} 帧")
            logger.info(f"成功处理人脸: {success_frames} 帧, 保留原帧(无人脸): {skipped_frames} 帧")
            logger.info(f"总处理时间: {total_time:.2f} 秒, 平均速度: {fps_avg:.2f} 帧/秒")
            logger.info(f"无人脸帧占比: {skipped_frames/processed_frames*100:.1f}% ({skipped_frames}/{processed_frames})")
            
            # 如果启用了质量评估，生成质量报告
            if enable_quality_assessment and quality_results["frames"]:
                try:
                    # 生成质量报告
                    report_path = os.path.join(quality_dir, "quality_report.txt")
                    with open(report_path, "w") as f:
                        f.write(f"视频质量评估报告\n")
                        f.write(f"=================\n\n")
                        f.write(f"源文件: {os.path.basename(target_video_path)}\n")
                        f.write(f"输出文件: {os.path.basename(output_path)}\n")
                        f.write(f"处理帧数: {processed_frames}\n")
                        f.write(f"成功处理人脸帧数: {success_frames}\n")
                        f.write(f"保留原帧(无人脸)数: {skipped_frames}\n")
                        f.write(f"无人脸帧占比: {skipped_frames/processed_frames*100:.1f}%\n\n")
                        
                        # 计算平均质量指标
                        avg_ssim = np.mean(quality_results["ssim"])
                        avg_psnr = np.mean(quality_results["psnr"])
                        
                        f.write(f"平均质量指标:\n")
                        f.write(f"  SSIM: {avg_ssim:.4f} (0-1, 越高越好)\n")
                        f.write(f"  PSNR: {avg_psnr:.2f} dB (通常 30-50dB 为可接受范围)\n\n")
                        
                        # 找出最好和最差的帧
                        best_idx = np.argmax(quality_results["ssim"])
                        worst_idx = np.argmin(quality_results["ssim"])
                        
                        f.write(f"最佳帧: 第 {quality_results['frames'][best_idx]} 帧\n")
                        f.write(f"  SSIM: {quality_results['ssim'][best_idx]:.4f}\n")
                        f.write(f"  PSNR: {quality_results['psnr'][best_idx]:.2f} dB\n\n")
                        
                        f.write(f"最差帧: 第 {quality_results['frames'][worst_idx]} 帧\n")
                        f.write(f"  SSIM: {quality_results['ssim'][worst_idx]:.4f}\n")
                        f.write(f"  PSNR: {quality_results['psnr'][worst_idx]:.2f} dB\n")
                    
                    # 生成质量指标图表
                    try:
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(12, 8))
                        
                        # SSIM曲线
                        ax1 = plt.subplot(211)
                        ax1.plot(quality_results["frames"], quality_results["ssim"], 'b-')
                        ax1.set_ylabel('SSIM')
                        ax1.set_title('质量评估指标')
                        ax1.grid(True)
                        
                        # PSNR曲线
                        ax2 = plt.subplot(212, sharex=ax1)
                        ax2.plot(quality_results["frames"], quality_results["psnr"], 'g-')
                        ax2.set_xlabel('帧')
                        ax2.set_ylabel('PSNR (dB)')
                        ax2.grid(True)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(quality_dir, "quality_metrics.png"))
                        plt.close()
                        
                    except Exception as e:
                        logger.error(f"生成质量指标图表失败: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"生成质量报告失败: {str(e)}")
            
            # 完成进度
            self.progress = 1.0
        
        return output_path
    
    def process_video_direct(self, source_face_path: str, target_video_path: str, output_video_path: str,
                            roi: Optional[Tuple[int, int, int, int]] = None, quality_assessment: bool = False) -> str:
        """
        直接处理视频 - 简化版本，用于前端调用
        
        Args:
            source_face_path: 源人脸图像路径
            target_video_path: 目标视频路径
            output_video_path: 输出视频路径
            roi: 感兴趣区域(x, y, w, h)，如果指定则只在此区域内检测人脸
            quality_assessment: 是否启用质量评估
            
        Returns:
            输出视频路径
        """
        return self.process_video(
            source_face_path, target_video_path, output_video_path,
            roi=roi, enable_quality_assessment=quality_assessment
        )
    
    def process_frames(self, source_face_path: str, frames_dir: str, output_dir: str,
                      pattern: str = "frame_*.jpg", roi: Optional[Tuple[int, int, int, int]] = None) -> int:
        """
        处理文件夹中的帧图像
        
        Args:
            source_face_path: 源人脸图像路径
            frames_dir: 帧图像目录
            output_dir: 输出目录
            pattern: 帧文件名模式
            roi: 感兴趣区域(x, y, w, h)，如果指定则只在此区域内检测人脸
            
        Returns:
            成功处理的帧数量
        """
        if not os.path.exists(frames_dir):
            logger.error(f"帧目录不存在: {frames_dir}")
            return 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载源人脸图像
        if not self.source_face is None and not self.load_source_face(source_face_path):
            logger.error(f"加载源人脸图像失败: {source_face_path}")
            return 0
        
        # 获取所有帧文件
        frame_files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
        
        if not frame_files:
            logger.error(f"未找到匹配的帧文件: {os.path.join(frames_dir, pattern)}")
            return 0
        
        # 处理每一帧
        processed_count = 0
        
        try:
            frame_iterator = tqdm(frame_files, desc="处理帧")
        except ImportError:
            frame_iterator = frame_files
            logger.info(f"正在处理帧: 总数 {len(frame_files)}")
        
        for frame_path in frame_iterator:
            # 读取帧
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"无法读取帧: {frame_path}")
                continue
            
            # 检测人脸
            faces = self.face_detector.detect_faces(frame, roi)
            
            # 如果检测到人脸
            if faces:
                # 处理每个检测到的人脸
                result_frame = frame.copy()
                
                for face in faces:
                    # 提取目标人脸
                    target_face = self.face_detector.align_face(frame, face, handle_occlusion=True)
                    
                    # 获取目标人脸特征点
                    target_landmarks = face.get('landmarks', [])
                    
                    # 如果有眼部跟踪器，优化眼神方向
                    if self.eye_tracker is not None and len(target_landmarks) > 0 and len(self.source_landmarks) > 0:
                        try:
                            # 检测眼睛位置
                            source_eyes = self.eye_tracker.detect_eyes(self.source_face)
                            target_eyes = self.eye_tracker.detect_eyes(target_face)
                            
                            # 优化眼神方向
                            if source_eyes and target_eyes:
                                optimized_source_face = self.eye_tracker.optimize_eye_direction(
                                    source_eyes, target_eyes, self.source_face, target_face
                                )
                                
                                # 增强眼睛
                                enhanced_source_face = self.eye_tracker.enhance_eyes(optimized_source_face, source_eyes)
                            else:
                                enhanced_source_face = self.source_face
                        except Exception as e:
                            logger.warning(f"眼部跟踪优化失败: {str(e)}")
                            enhanced_source_face = self.source_face
                    else:
                        enhanced_source_face = self.source_face
                    
                    # 执行面部替换
                    swapped_face = self.face_swapper.swap_face(enhanced_source_face, target_face)
                    
                    if swapped_face is None:
                        logger.warning(f"帧 {frame_path} 面部替换失败")
                        continue
                    
                    # 后处理
                    face_mask = None
                    if hasattr(self.face_swapper, '_create_face_mask'):
                        face_mask = self.face_swapper._create_face_mask(target_face)
                    
                    processed_face = self.post_processor.process(
                        target_face, swapped_face, face_mask, target_landmarks
                    )
                    
                    # 将替换后的人脸放回原图
                    bbox = face.get('bbox')
                    if bbox:
                        x, y, w, h = bbox
                        # 调整大小以匹配原始人脸区域
                        processed_face = cv2.resize(processed_face, (w, h))
                        # 创建掩码
                        mask = np.zeros((h, w), dtype=np.uint8)
                        center = (w // 2, h // 2)
                        axes = (w // 2 - w // 8, h // 2 - h // 8)
                        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                        mask = cv2.GaussianBlur(mask, (31, 31), 0)
                        # 混合
                        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
                        roi = result_frame[y:y+h, x:x+w]
                        roi = (roi * (1 - mask_3d) + processed_face * mask_3d).astype(np.uint8)
                        result_frame[y:y+h, x:x+w] = roi
                
                # 生成输出路径
                frame_filename = os.path.basename(frame_path)
                output_path = os.path.join(output_dir, frame_filename)
                
                # 保存结果
                cv2.imwrite(output_path, result_frame)
            else:
                # 如果没有检测到人脸，复制原始帧
                frame_filename = os.path.basename(frame_path)
                output_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(output_path, frame)
            
            processed_count += 1
        
        logger.info(f"帧处理完成，已处理 {processed_count} 帧，输出目录: {output_dir}")
        return processed_count
    
    def extract_frames_without_ffmpeg(self, video_path: str, output_dir: str, 
                                    interval: int = 1, max_frames: Optional[int] = None) -> int:
        """
        不使用FFmpeg提取视频帧
        
        Args:
            video_path: 视频路径
            output_dir: 输出目录
            interval: 帧间隔
            max_frames: 最大帧数，如果为None则无限制
            
        Returns:
            提取的帧数量
        """
        if not os.path.exists(video_path):
            logger.error(f"视频不存在: {video_path}")
            return 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return 0
        
        # 获取视频信息
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # 计算要提取的帧数
        to_extract = total_frames // interval
        if max_frames is not None:
            to_extract = min(to_extract, max_frames)
        
        # 提取帧
        extracted_count = 0
        frame_index = 0
        
        try:
            iterator = tqdm(range(to_extract), desc="提取帧")
        except ImportError:
            iterator = range(to_extract)
            logger.info(f"正在提取帧: 总数 {to_extract}")
        
        for _ in iterator:
            # 设置当前帧位置
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            # 读取帧
            ret, frame = video.read()
            if not ret:
                logger.warning(f"无法读取第 {frame_index} 帧，提前结束")
                break
            
            # 保存帧
            output_path = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            
            extracted_count += 1
            frame_index += interval
            
            # 检查是否达到最大帧数
            if max_frames is not None and extracted_count >= max_frames:
                break
        
        # 释放资源
        video.release()
        
        logger.info(f"帧提取完成，已提取 {extracted_count} 帧，输出目录: {output_dir}")
        return extracted_count
    
    def frames_to_video_without_ffmpeg(self, frames_dir: str, output_path: str, 
                                     fps: float = 30.0, pattern: str = "frame_*.jpg") -> bool:
        """
        不使用FFmpeg将帧转换为视频
        
        Args:
            frames_dir: 帧目录
            output_path: 输出视频路径
            fps: 帧率
            pattern: 帧文件名模式
            
        Returns:
            是否成功
        """
        if not os.path.exists(frames_dir):
            logger.error(f"帧目录不存在: {frames_dir}")
            return False
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有帧文件
        frame_files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
        
        if not frame_files:
            logger.error(f"未找到匹配的帧文件: {os.path.join(frames_dir, pattern)}")
            return False
        
        # 读取第一帧以获取尺寸
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            logger.error(f"无法读取第一帧: {frame_files[0]}")
            return False
        
        height, width = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4格式编码器
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height)
        )
        
        # 确保视频写入器正确创建
        if not video_writer.isOpened():
            logger.error(f"无法创建输出视频: {output_path}")
            return False
        
        # 写入帧
        try:
            iterator = tqdm(frame_files, desc="合成视频")
        except ImportError:
            iterator = frame_files
            logger.info(f"正在合成视频: 总帧数 {len(frame_files)}")
        
        for frame_path in iterator:
            # 读取帧
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"无法读取帧: {frame_path}")
                continue
            
            # 写入视频
            video_writer.write(frame)
        
        # 释放资源
        video_writer.release()
        
        logger.info(f"视频合成完成，共 {len(frame_files)} 帧，输出: {output_path}")
        return True
    
    def cleanup(self):
        """
        清理临时文件和资源
        """
        logger.info("清理临时文件和资源")
        # 释放可能的资源
        try:
            # 释放模型资源
            if hasattr(self.face_swapper, 'model') and self.face_swapper.model is not None:
                if hasattr(self.face_swapper.model, 'close'):
                    self.face_swapper.model.close()
            
            # 释放眼部跟踪器资源
            if self.eye_tracker is not None:
                if hasattr(self.eye_tracker, 'close'):
                    self.eye_tracker.close()
            
            # 清除缓存的图像
            self.source_face = None
            self.source_landmarks = None
            
            logger.info("资源清理完成")
        except Exception as e:
            logger.warning(f"清理资源时出错: {str(e)}")

    def process_image(self, source_face_path: str, target_image_path: str, output_path: str, 
                     enhance_face: bool = True) -> bool:
        """
        处理单张图像，替换人脸
        
        Args:
            source_face_path: 源人脸图像路径
            target_image_path: 目标图像路径
            output_path: 输出图像路径
            enhance_face: 是否增强人脸效果
            
        Returns:
            处理是否成功
        """
        try:
            # 加载图像
            source_face_img = read_image_with_path_fix(source_face_path)
            target_image = read_image_with_path_fix(target_image_path)
            
            # ... existing code ...
        
        except Exception as e:
            logging.error(f"处理图像失败: {str(e)}")
            return False

    def _extract_frames(self, video_path: str, output_dir: str, 
                      sample_rate: int = 1) -> Tuple[List[str], float]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            sample_rate: 采样率，每N帧提取一帧
            
        Returns:
            (帧文件路径列表, 视频帧率)
        """
        # ... existing code ...
        
        # 读取提取的第一帧以获取图像尺寸
        if frame_files:
            first_frame = read_image_with_path_fix(frame_files[0])
            # ... existing code ...
        
        # ... existing code ...
    
    def _process_frame(self, frame_path: str, source_face: np.ndarray) -> np.ndarray:
        """
        处理单帧图像
        
        Args:
            frame_path: 帧图像路径
            source_face: 源人脸图像
            
        Returns:
            处理后的帧图像
        """
        # 读取帧
        frame = read_image_with_path_fix(frame_path)
        
        # ... existing code ... 