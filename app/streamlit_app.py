"""
Streamlit前端应用 - 提供用户友好的界面，用于视频人物面部替换

使用Streamlit构建的Web应用，用户可上传源人脸和目标视频，配置替换参数，预览和下载结果
"""

import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
import sys
import glob
from PIL import Image
import shutil
import logging

# 将项目根目录添加到路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入日志配置和GPU补丁
from core.logger_config import setup_logger
from core.gpu_patch import apply_gpu_patches

# 设置日志和应用GPU补丁
logger = setup_logger()
gpu_available = apply_gpu_patches()
logger.info(f"GPU可用性: {'启用' if gpu_available else '禁用'}")

from core.video_processor import VideoProcessor
from core.face_detector import FaceDetector
from core.eye_tracker import EyeTracker
from app.utils import (
    save_uploaded_file, create_thumbnail, plot_side_by_side,
    generate_output_filename, get_video_info, extract_video_preview,
    load_image
)
from core.utils import read_image_with_path_fix


# 初始化会话状态
def init_session_state():
    """初始化Streamlit会话状态"""
    # 创建必要的目录结构
    os.makedirs("temp", exist_ok=True)
    os.makedirs("temp/faces", exist_ok=True)
    os.makedirs("temp/videos", exist_ok=True)
    os.makedirs("temp/output", exist_ok=True)
    
    if 'processor' not in st.session_state:
        # 使用共享的GPU检测结果
        use_cuda = gpu_available
        
        if use_cuda:
            # 尝试获取更详细的GPU信息
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
                    st.sidebar.success(f"✅ 已检测到GPU并启用加速\n{gpu_info}")
                    
                    # 输出详细GPU信息
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                    st.sidebar.info(f"GPU内存: 已分配 {memory_allocated:.1f}MB / 已预留 {memory_reserved:.1f}MB")
                else:
                    try:
                        import tensorflow as tf
                        gpus = tf.config.list_physical_devices('GPU')
                        if gpus:
                            gpu_info = f"GPU数量: {len(gpus)}"
                            st.sidebar.success(f"✅ 已检测到GPU并启用加速\n{gpu_info}")
                        else:
                            st.sidebar.warning("⚠️ TensorFlow未检测到GPU，但环境变量指示GPU可用")
                    except:
                        st.sidebar.warning("⚠️ 无法获取详细GPU信息，但已启用GPU加速")
            except:
                st.sidebar.warning("⚠️ 无法获取详细GPU信息，但已启用GPU加速")
        else:
            st.sidebar.warning("⚠️ 未检测到兼容的GPU，使用CPU模式")
        
        # 初始化视频处理器
        st.session_state.processor = VideoProcessor(use_cuda=use_cuda)
        st.session_state.face_detector = FaceDetector(use_gpu=use_cuda)
        st.session_state.eye_tracker = EyeTracker(use_cuda=use_cuda)
    
    # 其他会话状态变量
    if 'source_face_path' not in st.session_state:
        st.session_state.source_face_path = None
    
    if 'target_video_path' not in st.session_state:
        st.session_state.target_video_path = None
    
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'preview_image' not in st.session_state:
        st.session_state.preview_image = None


def load_model():
    """加载模型"""
    with st.spinner("正在加载模型..."):
        # 模型路径
        model_path = os.path.join("data", "models", "face_swap_model.h5")
        
        # 如果模型文件不存在，显示警告
        if not os.path.exists(model_path):
            st.warning(f"⚠️ 模型文件不存在: {model_path}")
            st.info("系统将使用未训练的模型，效果可能不佳。请先训练模型。")
        
        # 加载模型
        st.session_state.processor.load_model(model_path)
        st.session_state.model_loaded = True
        st.success("✅ 模型加载完成！")


def process_source_face(source_face_path):
    """处理源人脸"""
    if source_face_path and os.path.exists(source_face_path):
        # 读取源人脸图像
        source_face = load_image(source_face_path)
        
        if source_face is not None:
            # 在UI中显示源人脸
            st.image(cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB), 
                    caption="源人脸", use_column_width=True)
            return True
        else:
            st.error("无法读取源人脸图像")
    else:
        st.error("源人脸图像路径无效")
    
    return False


def upload_source_face():
    """上传源人脸"""
    st.write("请上传一张包含清晰人脸的照片，该人脸将被用于替换视频中的人脸。")
    st.write("建议使用正面、光线良好、表情自然的照片，以获得最佳效果。")
    
    uploaded_file = st.file_uploader("选择图片文件", type=["jpg", "jpeg", "png"], key="source_face_uploader")
    
    if uploaded_file is not None:
        # 显示上传信息
        st.success(f"已接收文件: {uploaded_file.name}")
        
        # 创建保存目录
        os.makedirs("temp/faces", exist_ok=True)
        
        # 保存上传的文件
        source_face_path = save_uploaded_file(uploaded_file, save_dir="temp/faces")
        
        if not source_face_path:
            st.error("图片保存失败，请重试")
            return False
        
        # 将路径保存到session state
        st.session_state.source_face_path = source_face_path
        
        # 读取源人脸图像
        source_face = load_image(source_face_path)
        
        if source_face is not None:
            # 检测人脸
            with st.spinner("正在分析人脸..."):
                faces = st.session_state.face_detector.detect_faces(source_face)
            
            # 在UI中显示源人脸
            st.image(cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB), 
                    caption="上传的源人脸", use_column_width=True)
            
            # 显示人脸检测结果
            if faces:
                st.success(f"检测到 {len(faces)} 个人脸")
                
                # 如果检测到多个人脸，显示警告
                if len(faces) > 1:
                    st.warning("图片中包含多个人脸。为获得最佳效果，建议使用只包含一个清晰人脸的照片。")
            else:
                st.warning("未检测到人脸，请上传包含清晰人脸的照片")
            
            # 设置成功上传标志
            st.session_state.source_face_uploaded = True
            
            # 初始化处理器
            if "processor" not in st.session_state:
                with st.spinner("初始化处理器..."):
                    processor = VideoProcessor()
                    st.session_state.processor = processor
                    
            # 加载源人脸
            st.session_state.processor.load_source_face(source_face_path)
            
            return True
        else:
            st.error("无法读取源人脸图像")
            st.info("请确保上传的是有效的图片文件。支持的格式包括JPG、JPEG和PNG。")
    
    return False


def upload_target_video():
    """上传目标视频"""
    st.write("请上传一个包含人脸的视频文件，该视频中的人脸将被替换为源人脸。")
    st.write("支持的格式: MP4, AVI, MOV")
    
    target_video_file = st.file_uploader("选择视频文件", type=["mp4", "avi", "mov"], key="target_video_uploader")
    
    if target_video_file is not None:
        # 显示上传信息
        st.success(f"已接收文件: {target_video_file.name}")
        
        # 创建保存目录
        os.makedirs("temp/videos", exist_ok=True)
        
        # 保存上传的文件
        target_video_path = save_uploaded_file(target_video_file, save_dir="temp/videos")
        if not target_video_path:
            st.error("视频保存失败，请重试")
            return False
            
        st.session_state.target_video_path = target_video_path
        
        # 获取视频信息
        try:
            video_info = get_video_info(target_video_path)
            
            # 显示视频信息
            col1, col2, col3 = st.columns(3)
            col1.metric("视频时长", f"{video_info['duration']}秒")
            col2.metric("帧数", f"{video_info['frame_count']}")
            col3.metric("分辨率", f"{video_info['width']}x{video_info['height']}")
            
            # 提取预览帧
            with st.spinner("正在处理视频预览..."):
                preview_frames = extract_video_preview(target_video_path, frame_count=3)
            
            if preview_frames:
                # 显示预览帧
                st.write("视频预览:")
                cols = st.columns(len(preview_frames))
                for i, frame in enumerate(preview_frames):
                    # 创建缩略图
                    thumbnail = create_thumbnail(frame, max_size=200)
                    
                    # 在对应的列中显示缩略图
                    cols[i].image(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
                    
                    # 尝试检测人脸
                    faces = st.session_state.face_detector.detect_faces(frame)
                    if faces:
                        cols[i].success(f"检测到 {len(faces)} 个人脸")
                    else:
                        cols[i].warning("未检测到人脸")
            
            return True
        except Exception as e:
            st.error(f"⚠️ 读取视频时出错: {str(e)}")
            st.info("请确保上传的视频文件完整且格式正确。如果问题持续，请尝试转换视频格式或使用另一个视频文件。")
            return False
    
    return False


def process_single_frame():
    """处理单帧预览"""
    if not st.session_state.model_loaded:
        st.warning("⚠️ 请先加载模型")
        return
    
    if st.session_state.source_face_path is None:
        st.warning("⚠️ 请先上传源人脸图像")
        return
    
    if st.session_state.target_video_path is None:
        st.warning("⚠️ 请先上传目标视频")
        return
    
    # 预览设置
    preview_col1, preview_col2 = st.columns(2)
    
    with preview_col1:
        # 自动寻找人脸帧选项
        auto_find_face = st.checkbox("自动定位到有人脸的帧", value=True)
        max_search_frames = st.slider("最大搜索帧数", min_value=10, max_value=500, value=100, step=10,
                                    help="自动搜索人脸时最多检查的帧数")
    
    with preview_col2:
        # 手动选择帧
        manual_frame = st.number_input("手动选择帧", value=0, min_value=0, step=1, 
                                     help="输入要查看的特定帧编号")
    
    # 提取视频的帧
    cap = cv2.VideoCapture(st.session_state.target_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # 显示视频信息
    st.info(f"视频总帧数: {total_frames}, 帧率: {frame_rate:.2f}fps, 时长: {total_frames/frame_rate:.2f}秒")
    
    # 获取要处理的帧
    selected_frame = manual_frame
    frame = None
    
    # 自动寻找包含人脸的帧
    if auto_find_face:
        with st.spinner("正在搜索包含人脸的帧..."):
            frame_with_face = None
            frame_idx = 0
            
            # 限制搜索范围
            search_limit = min(total_frames, max_search_frames)
            
            # 创建临时进度条
            progress = st.progress(0)
            
            for i in range(search_limit):
                # 更新进度
                progress.progress(i / search_limit)
                
                # 设置帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, current_frame = cap.read()
                
                if not ret:
                    break
                
                # 检测人脸
                faces = st.session_state.face_detector.detect_faces(current_frame)
                
                if faces:
                    frame_with_face = current_frame
                    frame_idx = i
                    selected_frame = i
                    break
            
            # 清除进度条
            progress.empty()
            
            if frame_with_face is not None:
                st.success(f"✅ 在第 {frame_idx} 帧找到人脸")
                frame = frame_with_face
            else:
                st.warning("⚠️ 在搜索范围内未找到包含人脸的帧")
                # 重置帧位置并读取第一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
                ret, frame = cap.read()
    else:
        # 手动选择帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        ret, frame = cap.read()
    
    cap.release()
    
    if not ret or frame is None:
        st.error("⚠️ 无法读取视频帧")
        return
    
    # 区域限定功能
    use_roi = st.checkbox("限定人脸检测区域（预览）", value=False)
    roi = None
    
    if use_roi:
        # 创建两列布局用于输入ROI
        height, width = frame.shape[:2]
        roi_col1, roi_col2 = st.columns(2)
        
        with roi_col1:
            roi_x = st.number_input("X坐标", value=width//4, min_value=0, max_value=width-1)
            roi_w = st.number_input("宽度", value=width//2, min_value=1, max_value=width)
        
        with roi_col2:
            roi_y = st.number_input("Y坐标", value=height//4, min_value=0, max_value=height-1)
            roi_h = st.number_input("高度", value=height//2, min_value=1, max_value=height)
        
        roi = (roi_x, roi_y, roi_w, roi_h)
        
        # 绘制ROI区域到帧上
        preview_frame = frame.copy()
        cv2.rectangle(preview_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        frame = preview_frame
    
    # 处理选项
    st.subheader("处理选项")
    
    col1, col2 = st.columns(2)
    
    with col1:
        smooth_factor = st.slider("边缘平滑程度(预览)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        use_eye_optimization = st.checkbox("启用眼部优化(预览)", value=True)
    
    with col2:
        color_correction = st.checkbox("颜色校正(预览)", value=True)
        face_enhancement = st.checkbox("面部增强(预览)", value=True)
    
    # 更新后处理器参数
    st.session_state.processor.post_processor.smooth_factor = smooth_factor
    st.session_state.processor.post_processor.color_correction = color_correction
    st.session_state.processor.post_processor.enhance_face = face_enhancement
    
    # 处理帧
    with st.spinner("处理中..."):
        # 检测人脸
        faces = st.session_state.face_detector.detect_faces(frame, roi)
        
        if not faces:
            st.error("⚠️ 未在视频帧中检测到人脸")
            # 显示原始帧
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="原始帧")
            return
        
        # 处理人脸
        result_frame = frame.copy()
        
        for face in faces:
            # 提取目标人脸
            target_face = st.session_state.face_detector.align_face(frame, face, handle_occlusion=True)
            
            # 获取目标人脸特征点
            target_landmarks = face.get('landmarks', [])
            
            # 如果有眼部跟踪器，优化眼神方向
            if use_eye_optimization and st.session_state.eye_tracker is not None and len(target_landmarks) > 0:
                try:
                    # 检测眼睛位置
                    source_face = cv2.imread(st.session_state.source_face_path)
                    source_eyes = st.session_state.eye_tracker.detect_eyes(source_face)
                    target_eyes = st.session_state.eye_tracker.detect_eyes(target_face)
                    
                    # 优化眼神方向
                    if source_eyes and target_eyes:
                        optimized_source_face = st.session_state.eye_tracker.optimize_eye_direction(
                            source_eyes, target_eyes, source_face, target_face
                        )
                        
                        # 增强眼睛
                        enhanced_source_face = st.session_state.eye_tracker.enhance_eyes(
                            optimized_source_face, source_eyes
                        )
                    else:
                        enhanced_source_face = source_face
                except Exception as e:
                    st.warning(f"眼部跟踪优化失败: {str(e)}")
                    enhanced_source_face = cv2.imread(st.session_state.source_face_path)
            else:
                # 直接加载源人脸图像
                enhanced_source_face = cv2.imread(st.session_state.source_face_path)
            
            # 执行面部替换
            swapped_face = st.session_state.processor.face_swapper.swap_face(
                enhanced_source_face, target_face
            )
            
            if swapped_face is None:
                st.warning("⚠️ 面部替换失败")
                continue
            
            # 后处理
            face_mask = None
            if hasattr(st.session_state.processor.face_swapper, '_create_face_mask'):
                face_mask = st.session_state.processor.face_swapper._create_face_mask(target_face)
            
            processed_face = st.session_state.processor.post_processor.process(
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
                
                # 确保roi和processed_face尺寸一致
                roi = result_frame[y:y+h, x:x+w].copy()
                if roi.shape != processed_face.shape:
                    processed_face = cv2.resize(processed_face, (roi.shape[1], roi.shape[0]))
                    # 重新创建掩码以匹配调整后的大小
                    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
                    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
                
                # 混合图像
                roi = (roi * (1 - mask_3d) + processed_face * mask_3d).astype(np.uint8)
                result_frame[y:y+h, x:x+w] = roi
        
        # 显示处理结果
        st.subheader("处理结果对比")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="原始帧")
        
        with col2:
            st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), caption="处理后帧")
        
        # 保存处理后的帧到会话状态中，以便后续使用
        st.session_state.preview_image = result_frame


def process_video():
    """处理视频"""
    if not st.session_state.model_loaded:
        st.warning("⚠️ 请先加载模型")
        return
    
    if st.session_state.source_face_path is None:
        st.warning("⚠️ 请先上传源人脸图像")
        return
    
    if st.session_state.target_video_path is None:
        st.warning("⚠️ 请先上传目标视频")
        return
    
    # 处理参数
    st.subheader("视频处理参数")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 起始帧和结束帧
        start_frame = st.number_input("起始帧", value=0, min_value=0, step=1)
        fps = st.number_input("输出帧率", value=0, min_value=0, help="0表示使用原始帧率")
    
    with col2:
        # 限制帧数
        frame_limit = st.number_input("处理帧数限制", value=0, min_value=0, help="0表示不限制")
        resolution = st.selectbox("输出分辨率", ["原始分辨率", "1080p", "720p", "480p"])
    
    # 区域限定功能
    st.subheader("区域限定（可选）")
    use_roi = st.checkbox("限定人脸检测区域", value=False)
    roi = None
    
    if use_roi:
        # 显示目标视频的第一帧，让用户选择区域
        cap = cv2.VideoCapture(st.session_state.target_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # 调整图像大小以适应界面
            max_width = 600
            height, width = frame.shape[:2]
            scale = max_width / width if width > max_width else 1.0
            if scale < 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
            # 显示第一帧
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="目标视频第一帧")
            
            # 创建两列布局用于输入ROI
            roi_col1, roi_col2 = st.columns(2)
            
            with roi_col1:
                roi_x = st.number_input("X坐标", value=width//4, min_value=0, max_value=width-1)
                roi_w = st.number_input("宽度", value=width//2, min_value=1, max_value=width)
            
            with roi_col2:
                roi_y = st.number_input("Y坐标", value=height//4, min_value=0, max_value=height-1)
                roi_h = st.number_input("高度", value=height//2, min_value=1, max_value=height)
            
            # 调整坐标到原始图像大小
            if scale < 1.0:
                roi_x = int(roi_x / scale)
                roi_y = int(roi_y / scale)
                roi_w = int(roi_w / scale)
                roi_h = int(roi_h / scale)
            
            roi = (roi_x, roi_y, roi_w, roi_h)
            
            # 绘制ROI区域
            preview_frame = frame.copy()
            cv2.rectangle(preview_frame, (int(roi_x*scale), int(roi_y*scale)), 
                         (int((roi_x+roi_w)*scale), int((roi_y+roi_h)*scale)), (0, 255, 0), 2)
            st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), caption="选定区域预览")
    
    # 眼部优化选项
    st.subheader("眼部优化")
    use_eye_optimization = st.checkbox("启用眼部优化", value=True)
    
    if use_eye_optimization:
        eye_smooth_factor = st.slider("眼部平滑程度", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        pupil_enhancement = st.slider("瞳孔增强程度", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # 质量评估选项
    st.subheader("质量评估")
    enable_quality_assessment = st.checkbox("启用质量评估", value=False)
    
    # 后处理选项
    st.subheader("后处理选项")
    smooth_factor = st.slider("边缘平滑程度", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    color_correction = st.checkbox("颜色校正", value=True)
    face_enhancement = st.checkbox("面部增强", value=True)
    
    # 确定输出分辨率
    output_resolution = None
    if resolution == "1080p":
        output_resolution = (1920, 1080)
    elif resolution == "720p":
        output_resolution = (1280, 720)
    elif resolution == "480p":
        output_resolution = (854, 480)
    
    # 进度状态容器
    progress_placeholder = st.empty()
    
    # 处理按钮
    if st.button("开始处理视频"):
        # 创建输出目录
        os.makedirs("temp/output", exist_ok=True)
        
        # 生成输出文件名
        output_filename = generate_output_filename(
            st.session_state.source_face_path, 
            st.session_state.target_video_path
        )
        output_path = os.path.join("temp/output", output_filename)
        
        # 更新后处理器参数
        st.session_state.processor.post_processor.smooth_factor = smooth_factor
        st.session_state.processor.post_processor.color_correction = color_correction
        st.session_state.processor.post_processor.enhance_face = face_enhancement
        
        # 创建进度条
        progress_bar = progress_placeholder.progress(0)
        status_text = st.empty()
        status_text.text("准备处理视频...")
        
        # 创建一个定时更新进度的辅助函数
        def update_progress():
            if hasattr(st.session_state.processor, 'progress'):
                progress = st.session_state.processor.progress
                progress_bar.progress(progress)
                status_text.text(f"处理中... {progress:.1%}")
                
                # 如果处理完成
                if progress >= 1.0:
                    status_text.text("处理完成！")
                    return True
            return False
        
        # 处理视频
        try:
            # 确定结束帧
            end_frame = None if frame_limit == 0 else start_frame + frame_limit
            output_fps = None if fps == 0 else fps
            
            # 添加进度显示变量
            st.session_state.processing_status = "处理中"
            st.session_state.processing_error = None
            st.session_state.output_video_path = None
            
            # 启动处理线程
            def process_thread():
                try:
                    output_path_result = st.session_state.processor.process_video(
                        st.session_state.source_face_path,
                        st.session_state.target_video_path,
                        output_path,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        fps=output_fps,
                        resolution=output_resolution,
                        show_progress=True,
                        roi=roi,
                        enable_quality_assessment=enable_quality_assessment
                    )
                    # 使用线程安全的方式更新结果
                    with st.session_state._lock if hasattr(st.session_state, '_lock') else threading.Lock():
                        st.session_state.output_video_path = output_path_result
                        st.session_state.processing_status = "完成"
                except Exception as e:
                    # 捕获并记录错误
                    error_msg = str(e)
                    with st.session_state._lock if hasattr(st.session_state, '_lock') else threading.Lock():
                        st.session_state.processing_error = error_msg
                        st.session_state.processing_status = "错误"
            
            # 启动线程
            import threading
            if not hasattr(st.session_state, '_lock'):
                st.session_state._lock = threading.Lock()
                
            processing_thread = threading.Thread(target=process_thread)
            processing_thread.daemon = True
            processing_thread.start()
            
            # 实时进度显示添加剩余时间估计
            start_time = time.time()
            last_progress = 0
            speeds = []  # 存储近期处理速度
            
            while st.session_state.processing_status == "处理中":
                current_progress = st.session_state.processor.progress
                
                # 计算处理速度和预计剩余时间
                if current_progress > last_progress:
                    elapsed = time.time() - start_time
                    progress_diff = current_progress - last_progress
                    if progress_diff > 0 and elapsed > 0:
                        speed = progress_diff / elapsed
                        speeds.append(speed)
                        # 只保留最近10个速度样本
                        if len(speeds) > 10:
                            speeds.pop(0)
                        
                        # 计算平均速度和预计剩余时间
                        avg_speed = sum(speeds) / len(speeds)
                        remaining_progress = 1.0 - current_progress
                        estimated_time = remaining_progress / avg_speed if avg_speed > 0 else 0
                        
                        # 更新进度显示
                        progress_bar.progress(current_progress)
                        status_text.text(f"处理中... {current_progress:.1%} 已完成 | 预计剩余时间: {estimated_time:.1f}秒 | 帧率: {int(avg_speed * 100 * st.session_state.processor.frames_to_process)}帧/秒")
                        
                        # 重置
                        last_progress = current_progress
                        start_time = time.time()
                    
                # 防止UI卡死
                time.sleep(0.5)
                
                # 检查是否完成
                if st.session_state.processing_status in ["完成", "错误"]:
                    break
            
            # 处理完成或出错
            if st.session_state.processing_status == "错误":
                st.error(f"处理视频时出错: {st.session_state.processing_error}")
                
            elif st.session_state.processing_status == "完成":
                progress_bar.progress(1.0)
                status_text.text("处理完成！")
                # 等待一小段时间确保文件写入完成
                time.sleep(1)
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"启动处理时出错: {str(e)}")
    
    # 如果处理完成，显示下载链接和结果
    if st.session_state.processing_status == "完成" and st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
        st.success("✅ 视频处理完成！")
        
        # 显示视频预览
        st.video(st.session_state.output_video_path)
        
        # 下载链接
        with open(st.session_state.output_video_path, "rb") as file:
            st.download_button(
                label="下载处理后的视频",
                data=file,
                file_name=os.path.basename(st.session_state.output_video_path),
                mime="video/mp4"
            )
        
        # 显示质量评估结果（如果有）
        if enable_quality_assessment:
            quality_dir = os.path.join(os.path.dirname(st.session_state.output_video_path), "quality_assessment")
            if os.path.exists(quality_dir):
                st.subheader("质量评估结果")
                
                # 显示质量评估报告
                report_path = os.path.join(quality_dir, "quality_report.txt")
                if os.path.exists(report_path):
                    with open(report_path, "r") as file:
                        report_content = file.read()
                        st.text_area("评估报告", report_content, height=200)
                
                # 显示质量指标图表
                metrics_path = os.path.join(quality_dir, "quality_metrics.png")
                if os.path.exists(metrics_path):
                    st.image(metrics_path, caption="质量指标图表")
                
                # 显示可视化结果
                st.subheader("质量可视化")
                vis_files = sorted(glob.glob(os.path.join(quality_dir, "quality_vis_*.png")))
                if vis_files:
                    for vis_file in vis_files[:3]:  # 只显示前3个
                        st.image(vis_file, caption=os.path.basename(vis_file))


def train_model_ui():
    """训练模型界面"""
    st.write("## 模型训练")
    st.write("使用自己的数据集训练面部替换模型。")
    
    # 设置训练参数
    dataset_dir = st.text_input("数据集目录", "data/dataset")
    epochs = st.slider("训练轮数", 10, 200, 50)
    
    # 训练按钮
    if st.button("开始训练"):
        # 检查数据集目录是否存在
        if not os.path.exists(dataset_dir):
            st.error(f"⚠️ 数据集目录不存在: {dataset_dir}")
            st.info("请先创建数据集目录，并放入人脸图像")
            return
        
        # 开始训练
        with st.spinner("正在训练模型..."):
            try:
                # 训练模型
                history = st.session_state.processor.train_model(dataset_dir, epochs=epochs)
                
                # 显示训练结果
                st.success("✅ 模型训练完成！")
                
                # 加载新模型
                st.session_state.model_loaded = False
                load_model()
            except Exception as e:
                st.error(f"⚠️ 训练过程中出错: {e}")


def main():
    """主函数"""
    # 设置页面标题
    st.set_page_config(
        page_title="视频人物面部替换系统",
        page_icon="🎭",
        layout="wide"
    )
    
    # 应用标题
    st.title("🎭 视频人物面部替换系统")
    
    # 初始化会话状态
    init_session_state()
    
    # 侧边栏 - 系统状态和控制
    with st.sidebar:
        st.header("系统控制")
        
        # 加载模型
        if st.button("加载/重置模型"):
            load_model()
        
        # 显示模型状态
        if st.session_state.model_loaded:
            st.success("✅ 模型已加载")
        else:
            st.warning("⚠️ 模型未加载")
        
        # 分隔线
        st.divider()
        
        # 关于信息
        st.subheader("关于")
        st.markdown("""
        本系统基于深度学习技术实现视频中的人物面部替换。
        
        主要功能:
        - 视频中人脸检测与跟踪
        - 基于深度学习的面部替换
        - 眼部优化与表情匹配
        - 高质量后处理与融合
        """)
    
    # 主要功能区
    tab1, tab2, tab3, tab4 = st.tabs(["源与目标", "单帧预览", "视频处理", "模型训练"])
    
    with tab1:
        # 创建两列布局，左边上传源人脸，右边上传目标视频
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("第一步：上传源人脸")
            upload_source_face()
        
        with col2:
            st.subheader("第二步：上传目标视频")
            video_uploaded = upload_target_video()
            
            if video_uploaded:
                st.success("✅ 源人脸和目标视频均已上传！")
                st.info("请点击上方的\"单帧预览\"或\"视频处理\"选项卡继续操作。")
    
    with tab2:
        process_single_frame()
    
    with tab3:
        process_video()
    
    with tab4:
        train_model_ui()


if __name__ == "__main__":
    main() 