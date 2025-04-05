"""
眼部跟踪模块 - 用于面部视频中的眼睛跟踪和眼神方向优化
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any

# 尝试导入可选依赖
try:
    import face_alignment
    from face_alignment import LandmarksType
    FACE_ALIGNMENT_AVAILABLE = True
    print("face_alignment库导入成功")
except ImportError as e:
    FACE_ALIGNMENT_AVAILABLE = False
    print(f"警告: face_alignment库导入失败: {str(e)}")
    # 创建备用LandmarksType
    class LandmarksType:
        TWO_D = 1
        THREE_D = 3
        TWO_HALF_D = 2

# 创建枚举类型的替代
class FallbackLandmarksType:
    """当face_alignment不可用时的备用类型"""
    TWO_D = 1
    THREE_D = 3
    TWO_HALF_D = 2

class EyeTracker:
    """眼部跟踪和注视方向优化类"""
    
    def __init__(self, use_cuda: bool = True):
        """
        初始化眼部跟踪器
        
        Args:
            use_cuda: 是否使用CUDA加速
        """
        # 初始化人脸对齐模型用于眼部追踪
        self.device = 'cuda' if use_cuda else 'cpu'
        
        # 如果face_alignment可用，使用标准库
        if FACE_ALIGNMENT_AVAILABLE:
            try:
                self.fa = face_alignment.FaceAlignment(
                    LandmarksType.TWO_D, 
                    device=self.device,
                    flip_input=False
                )
                print("成功初始化face_alignment库用于眼部跟踪")
            except Exception as e:
                print(f"警告: 初始化face_alignment失败: {str(e)}")
                # 尝试使用CPU模式初始化
                try:
                    self.fa = face_alignment.FaceAlignment(
                        LandmarksType.TWO_D, 
                        device='cpu',
                        flip_input=False
                    )
                    print("成功使用CPU模式初始化face_alignment库")
                except Exception as e2:
                    print(f"警告: 备用初始化face_alignment也失败: {str(e2)}")
                    self.fa = None
        else:
            # 如果不可用，使用替代方法
            print("警告: face_alignment库不可用，使用简单眼部跟踪替代")
            self.fa = None
        
        # 存储前N帧的眼部位置，用于平滑处理
        self.eye_history = []
        self.history_size = 5  # 保存5帧历史
        
        # 加载眼部角度映射表 (如果有)
        self.angle_map = self._load_angle_map()
    
    def _load_angle_map(self) -> Dict[str, Any]:
        """
        加载眼部角度映射表
        
        Returns:
            眼部角度映射字典
        """
        # 这里可以从文件加载预定义的眼部角度映射
        # 简单起见，返回一个空字典
        return {}
    
    def detect_eyes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的眼睛位置
        
        Args:
            image: BGR格式图像
            
        Returns:
            眼睛位置列表，每个元素为(x, y, w, h)
        """
        # 首先尝试使用face_alignment
        if self.fa is not None:
            try:
                # 使用face_alignment检测关键点
                landmarks = self.fa.get_landmarks(image)
                if landmarks and len(landmarks) > 0:
                    # 取第一个人脸
                    face_landmarks = landmarks[0]
                    
                    # 提取左眼和右眼的关键点 (按照68点模型)
                    left_eye = face_landmarks[36:42]
                    right_eye = face_landmarks[42:48]
                    
                    # 计算眼睛边界框
                    left_eye_box = self._points_to_box(left_eye)
                    right_eye_box = self._points_to_box(right_eye)
                    
                    return [left_eye_box, right_eye_box]
            except Exception as e:
                print(f"使用face_alignment检测眼睛失败: {str(e)}")
        
        # 如果face_alignment不可用或检测失败，使用Haar级联分类器
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 首先尝试使用本地级联分类器
            cascade_path = os.path.join(os.path.dirname(__file__), "..", "data", "cascades", "haarcascade_eye.xml")
            eye_cascade = cv2.CascadeClassifier(cascade_path)
            
            # 检查分类器是否加载成功
            if eye_cascade.empty():
                print(f"警告: 无法从本地路径加载级联分类器: {cascade_path}")
                # 尝试使用OpenCV内置路径
                try:
                    opencv_path = cv2.data.haarcascades + "haarcascade_eye.xml"
                    eye_cascade = cv2.CascadeClassifier(opencv_path)
                    if eye_cascade.empty():
                        print(f"警告: 无法从OpenCV路径加载级联分类器: {opencv_path}")
                        # 如果两种方法都失败，返回估计的眼睛位置
                        h, w = image.shape[:2]
                        left_eye = (int(w * 0.3), int(h * 0.4), int(w * 0.1), int(h * 0.05))
                        right_eye = (int(w * 0.6), int(h * 0.4), int(w * 0.1), int(h * 0.05))
                        return [left_eye, right_eye]
                except Exception as e:
                    print(f"加载OpenCV级联分类器时出错: {str(e)}")
                    # 返回估计的眼睛位置
                    h, w = image.shape[:2]
                    left_eye = (int(w * 0.3), int(h * 0.4), int(w * 0.1), int(h * 0.05))
                    right_eye = (int(w * 0.6), int(h * 0.4), int(w * 0.1), int(h * 0.05))
                    return [left_eye, right_eye]
            
            # 检测眼睛
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # 如果检测到眼睛，返回结果
            if len(eyes) > 0:
                return [(x, y, w, h) for (x, y, w, h) in eyes[:2]]  # 最多返回两个眼睛
            else:
                # 如果检测不到眼睛，返回估计的位置
                h, w = image.shape[:2]
                left_eye = (int(w * 0.3), int(h * 0.4), int(w * 0.1), int(h * 0.05))
                right_eye = (int(w * 0.6), int(h * 0.4), int(w * 0.1), int(h * 0.05))
                return [left_eye, right_eye]
                
        except Exception as e:
            print(f"使用Haar级联分类器检测眼睛失败: {str(e)}")
            
            # 所有方法都失败，返回估计的眼睛位置
            h, w = image.shape[:2]
            left_eye = (int(w * 0.3), int(h * 0.4), int(w * 0.1), int(h * 0.05))
            right_eye = (int(w * 0.6), int(h * 0.4), int(w * 0.1), int(h * 0.05))
            return [left_eye, right_eye]
    
    def _points_to_box(self, points: np.ndarray) -> Tuple[int, int, int, int]:
        """
        将关键点数组转换为边界框
        
        Args:
            points: 关键点数组
            
        Returns:
            边界框 (x, y, w, h)
        """
        min_x = int(np.min(points[:, 0]))
        min_y = int(np.min(points[:, 1]))
        max_x = int(np.max(points[:, 0]))
        max_y = int(np.max(points[:, 1]))
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def track_eyes(self, frames: List[np.ndarray]) -> List[List[Tuple[int, int, int, int]]]:
        """
        在连续帧中跟踪眼睛
        
        Args:
            frames: 帧列表
            
        Returns:
            每帧的眼睛位置列表
        """
        eye_positions = []
        
        for frame in frames:
            # 检测当前帧的眼睛
            current_eyes = self.detect_eyes(frame)
            
            # 添加到历史记录
            self.eye_history.append(current_eyes)
            if len(self.eye_history) > self.history_size:
                self.eye_history.pop(0)
            
            # 平滑处理
            smoothed_eyes = self._smooth_eye_positions(current_eyes)
            eye_positions.append(smoothed_eyes)
        
        return eye_positions
    
    def _smooth_eye_positions(self, current_eyes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        平滑眼睛位置，减少抖动
        
        Args:
            current_eyes: 当前帧的眼睛位置
            
        Returns:
            平滑后的眼睛位置
        """
        if len(self.eye_history) < 2:
            return current_eyes
        
        # 计算历史记录中所有眼睛的平均位置
        smoothed_eyes = []
        for i in range(len(current_eyes)):
            # 提取历史记录中第i个眼睛的位置
            eye_history_i = [eyes[i] for eyes in self.eye_history if i < len(eyes)]
            
            # 计算平均值
            avg_x = sum(eye[0] for eye in eye_history_i) / len(eye_history_i)
            avg_y = sum(eye[1] for eye in eye_history_i) / len(eye_history_i)
            avg_w = sum(eye[2] for eye in eye_history_i) / len(eye_history_i)
            avg_h = sum(eye[3] for eye in eye_history_i) / len(eye_history_i)
            
            # 使用加权平均，当前帧占较大权重
            weight_current = 0.6
            weight_history = 1.0 - weight_current
            
            smooth_x = int(current_eyes[i][0] * weight_current + avg_x * weight_history)
            smooth_y = int(current_eyes[i][1] * weight_current + avg_y * weight_history)
            smooth_w = int(current_eyes[i][2] * weight_current + avg_w * weight_history)
            smooth_h = int(current_eyes[i][3] * weight_current + avg_h * weight_history)
            
            smoothed_eyes.append((smooth_x, smooth_y, smooth_w, smooth_h))
        
        return smoothed_eyes
    
    def estimate_gaze_direction(self, eye_region: np.ndarray) -> Tuple[float, float]:
        """
        估算眼神注视方向
        
        Args:
            eye_region: 眼睛区域图像
            
        Returns:
            (水平角度, 垂直角度) 弧度制
        """
        if eye_region is None or eye_region.size == 0:
            return (0.0, 0.0)
        
        # 转为灰度图
        if len(eye_region.shape) == 3:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_region
        
        # 阈值处理以分离瞳孔
        _, thresh = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            # 找到最大的轮廓 (假设是瞳孔)
            pupil_contour = max(contours, key=cv2.contourArea)
            
            # 计算瞳孔中心
            M = cv2.moments(pupil_contour)
            if M["m00"] != 0:
                pupil_x = int(M["m10"] / M["m00"])
                pupil_y = int(M["m01"] / M["m00"])
            else:
                # 如果无法计算矩，使用眼睛区域中心
                pupil_x = eye_region.shape[1] // 2
                pupil_y = eye_region.shape[0] // 2
            
            # 计算眼睛区域中心
            eye_center_x = eye_region.shape[1] // 2
            eye_center_y = eye_region.shape[0] // 2
            
            # 计算相对位置
            rel_x = (pupil_x - eye_center_x) / (eye_region.shape[1] / 2)
            rel_y = (pupil_y - eye_center_y) / (eye_region.shape[0] / 2)
            
            # 转换为角度 (简单的线性映射)
            # 假设眼球可以旋转±30度
            max_angle = np.radians(30)
            horiz_angle = rel_x * max_angle
            vert_angle = rel_y * max_angle
            
            return (horiz_angle, vert_angle)
        
        # 如果无法找到瞳孔，返回默认值
        return (0.0, 0.0)
    
    def optimize_eye_direction(self, source_eyes: List[Tuple[int, int, int, int]], 
                             target_eyes: List[Tuple[int, int, int, int]], 
                             source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """
        优化眼睛方向，使源人脸的眼神方向与目标人脸匹配
        
        Args:
            source_eyes: 源图像中的眼睛位置列表
            target_eyes: 目标图像中的眼睛位置列表
            source_img: 源图像
            target_img: 目标图像
            
        Returns:
            优化后的源图像
        """
        if not source_eyes or not target_eyes or len(source_eyes) < 1 or len(target_eyes) < 1:
            print("眼睛位置无效")
            return source_img
        
        result_img = source_img.copy()
        
        # 处理每个眼睛
        for i in range(min(len(source_eyes), len(target_eyes))):
            source_eye = source_eyes[i]
            target_eye = target_eyes[i]
            
            # 获取眼睛区域
            sx, sy, sw, sh = source_eye
            tx, ty, tw, th = target_eye
            
            # 裁剪眼睛区域
            source_eye_region = source_img[sy:sy+sh, sx:sx+sw]
            target_eye_region = target_img[ty:ty+th, tx:tx+tw]
            
            if source_eye_region.size == 0 or target_eye_region.size == 0:
                continue
            
            # 调整大小以便比较
            source_eye_region_resized = cv2.resize(source_eye_region, (100, 50))
            target_eye_region_resized = cv2.resize(target_eye_region, (100, 50))
            
            # 估计视线方向
            source_horiz, source_vert = self.estimate_gaze_direction(source_eye_region_resized)
            target_horiz, target_vert = self.estimate_gaze_direction(target_eye_region_resized)
            
            # 计算差异
            horiz_diff = target_horiz - source_horiz
            vert_diff = target_vert - source_vert
            
            # 调整视线方向
            adjusted_eye = self._adjust_eye_direction(source_eye_region, horiz_diff, vert_diff)
            
            # 将调整后的眼睛区域放回图像
            if adjusted_eye.shape[:2] == (sh, sw):
                result_img[sy:sy+sh, sx:sx+sw] = adjusted_eye
        
        return result_img
    
    def _adjust_eye_direction(self, eye_img: np.ndarray, 
                           horiz_angle: float, vert_angle: float) -> np.ndarray:
        """
        调整眼睛图像的注视方向
        
        Args:
            eye_img: 眼睛区域图像
            horiz_angle: 水平方向角度
            vert_angle: 垂直方向角度
            
        Returns:
            调整后的眼睛图像
        """
        if eye_img.size == 0:
            return eye_img
        
        # 转为灰度图以检测瞳孔
        if len(eye_img.shape) == 3:
            gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_img.copy()
            eye_img = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2BGR)
        
        # 阈值处理以分离瞳孔
        _, thresh = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        result = eye_img.copy()
        
        if contours:
            # 找到最大的轮廓 (假设是瞳孔)
            pupil_contour = max(contours, key=cv2.contourArea)
            
            # 创建瞳孔掩码
            pupil_mask = np.zeros_like(gray_eye)
            cv2.drawContours(pupil_mask, [pupil_contour], -1, 255, -1)
            
            # 计算瞳孔中心
            M = cv2.moments(pupil_contour)
            if M["m00"] != 0:
                pupil_x = int(M["m10"] / M["m00"])
                pupil_y = int(M["m01"] / M["m00"])
                
                # 计算偏移量 (将角度转换为像素偏移)
                eye_width = eye_img.shape[1]
                eye_height = eye_img.shape[0]
                
                offset_x = int(horiz_angle * eye_width / np.radians(60))
                offset_y = int(vert_angle * eye_height / np.radians(60))
                
                # 创建变换矩阵
                M = np.float32([
                    [1, 0, offset_x],
                    [0, 1, offset_y]
                ])
                
                # 只移动瞳孔区域
                pupil_region = cv2.bitwise_and(eye_img, eye_img, mask=pupil_mask)
                background = cv2.bitwise_and(eye_img, eye_img, mask=255-pupil_mask)
                
                # 平移瞳孔
                moved_pupil = cv2.warpAffine(pupil_region, M, (eye_img.shape[1], eye_img.shape[0]))
                
                # 合并回背景
                result = cv2.add(moved_pupil, background)
        
        return result
    
    def enhance_eyes(self, face_img: np.ndarray, eyes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        增强眼睛区域，平滑眼睛并调整瞳孔
        
        Args:
            face_img: 人脸图像
            eyes: 眼睛位置列表，每个元素为(x, y, w, h)
            
        Returns:
            增强后的人脸图像
        """
        if face_img is None or not eyes:
            return face_img
        
        result_img = face_img.copy()
        
        # 处理每个眼睛
        for eye in eyes:
            x, y, w, h = eye
            
            # 确保坐标在有效范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, face_img.shape[1] - x)
            h = min(h, face_img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # 提取眼睛区域
            eye_region = face_img[y:y+h, x:x+w]
            
            # 眼睛区域预处理 - 使用高级去噪并提高对比度
            eye_region_processed = self._refine_eye_region(eye_region)
            
            # 增强瞳孔
            enhanced_eye = self._enhance_pupil(eye_region_processed)
            
            # 平滑眼睛区域
            smoothed_eye = self._smooth_eye_region(enhanced_eye)
            
            # 为睫毛增强细节
            detailed_eye = self._enhance_eyelashes(smoothed_eye)
            
            # 将增强后的眼睛区域放回图像
            result_img[y:y+h, x:x+w] = detailed_eye
        
        return result_img
    
    def _smooth_eye_region(self, eye_region: np.ndarray) -> np.ndarray:
        """
        平滑眼睛区域，减少噪点和锐利边缘
        
        Args:
            eye_region: 眼睛区域图像
            
        Returns:
            平滑后的眼睛区域
        """
        if eye_region is None or eye_region.size == 0:
            return eye_region
        
        # 创建结果图像
        result = eye_region.copy()
        
        try:
            # 转换为LAB颜色空间
            if len(eye_region.shape) == 3 and eye_region.shape[2] == 3:
                lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # 对L通道应用双边滤波
                l_filtered = cv2.bilateralFilter(l, 9, 75, 75)
                
                # 合并通道
                lab_filtered = cv2.merge([l_filtered, a, b])
                
                # 转换回BGR
                result = cv2.cvtColor(lab_filtered, cv2.COLOR_LAB2BGR)
            else:
                # 对灰度图像应用双边滤波
                result = cv2.bilateralFilter(eye_region, 9, 75, 75)
        except Exception as e:
            print(f"平滑眼睛区域时出错: {str(e)}")
        
        return result
    
    def _enhance_pupil(self, eye_region: np.ndarray) -> np.ndarray:
        """
        增强瞳孔，使其更加清晰和自然
        
        Args:
            eye_region: 眼睛区域图像
            
        Returns:
            增强后的眼睛区域
        """
        if eye_region is None or eye_region.size == 0:
            return eye_region
        
        # 创建结果图像
        result = eye_region.copy()
        
        try:
            # 转换为灰度图
            if len(eye_region.shape) == 3:
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                is_color = True
            else:
                gray_eye = eye_region.copy()
                is_color = False
            
            # 使用自适应阈值处理，更好地处理不同光照条件
            thresh = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # 应用形态学操作来优化瞳孔区域
            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 查找轮廓
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                # 筛选可能的瞳孔轮廓
                pupil_candidates = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 10:  # 忽略太小的区域
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h if h > 0 else 0
                        # 瞳孔应该相对圆形
                        if 0.5 < aspect_ratio < 2.0:
                            # 计算轮廓的圆度
                            perimeter = cv2.arcLength(contour, True)
                            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                            if circularity > 0.5:  # 瞳孔应该相对圆形
                                pupil_candidates.append((contour, circularity, area))
                
                # 按圆度和面积排序选择最佳瞳孔轮廓
                if pupil_candidates:
                    pupil_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                    pupil_contour = pupil_candidates[0][0]
                    
                    # 创建瞳孔掩码
                    pupil_mask = np.zeros_like(gray_eye)
                    cv2.drawContours(pupil_mask, [pupil_contour], -1, 255, -1)
                    
                    # 平滑瞳孔掩码边缘
                    pupil_mask = cv2.GaussianBlur(pupil_mask, (5, 5), 0)
                    
                    # 如果是彩色图像，对瞳孔进行增强
                    if is_color:
                        # 增强瞳孔对比度
                        eye_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(eye_hsv)
                        
                        # 增加瞳孔区域的饱和度和对比度
                        pupil_mask_norm = pupil_mask.astype(np.float32) / 255.0
                        
                        # 增加瞳孔区域的对比度
                        v_enhanced = v.astype(np.float32)
                        v_enhanced = v_enhanced * 0.8  # 降低亮度使瞳孔更深邃
                        v_enhanced = np.clip(v_enhanced, 0, 255).astype(np.uint8)
                        
                        # 增加瞳孔区域的饱和度
                        s_enhanced = s.astype(np.float32)
                        s_enhanced = s_enhanced * 1.3  # 增加饱和度
                        s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
                        
                        # 应用掩码混合
                        v_final = v * (1 - pupil_mask_norm) + v_enhanced * pupil_mask_norm
                        s_final = s * (1 - pupil_mask_norm) + s_enhanced * pupil_mask_norm
                        
                        # 合并通道
                        hsv_enhanced = cv2.merge([h, s_final.astype(np.uint8), v_final.astype(np.uint8)])
                        
                        # 转换回BGR
                        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
                        
                        # 为瞳孔添加高光，模拟真实眼睛的反光
                        # 计算瞳孔中心
                        M = cv2.moments(pupil_contour)
                        if M["m00"] != 0:
                            pupil_cx = int(M["m10"] / M["m00"])
                            pupil_cy = int(M["m01"] / M["m00"])
                            
                            # 添加高光点
                            highlight_size = max(2, min(pupil_mask.shape) // 20)
                            highlight_x = pupil_cx - highlight_size // 2
                            highlight_y = pupil_cy - highlight_size // 2
                            
                            # 确保坐标在图像范围内
                            highlight_x = max(0, min(highlight_x, pupil_mask.shape[1] - highlight_size - 1))
                            highlight_y = max(0, min(highlight_y, pupil_mask.shape[0] - highlight_size - 1))
                            
                            # 添加高光
                            highlight_intensity = 200  # 高光亮度
                            highlight_mask = np.zeros_like(pupil_mask)
                            cv2.circle(highlight_mask, (pupil_cx, pupil_cy), highlight_size // 2, 255, -1)
                            highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 0)
                            highlight_mask_norm = highlight_mask.astype(np.float32) / 255.0
                            
                            # 应用高光
                            for c in range(3):
                                channel = result[:, :, c].astype(np.float32)
                                channel = channel * (1 - highlight_mask_norm * 0.7) + highlight_intensity * highlight_mask_norm * 0.7
                                result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
                    else:
                        # 如果是灰度图像，简单增强对比度
                        pupil_mask_norm = pupil_mask.astype(np.float32) / 255.0
                        gray_enhanced = gray_eye.astype(np.float32) * 0.8  # 降低亮度使瞳孔更深邃
                        gray_enhanced = np.clip(gray_enhanced, 0, 255).astype(np.uint8)
                        result = gray_eye * (1 - pupil_mask_norm) + gray_enhanced * pupil_mask_norm
                        result = result.astype(np.uint8)
            
            # 最后还可以应用一点锐化，增强瞳孔边缘
            if is_color:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5
                result = cv2.filter2D(result, -1, kernel)
        
        except Exception as e:
            print(f"增强瞳孔时出错: {str(e)}")
        
        return result
    
    def _refine_eye_region(self, face_img: np.ndarray, eye: Tuple[int, int, int, int]) -> np.ndarray:
        """
        精细处理眼睛区域，包括去除红眼、增加细节和自然度
        
        Args:
            face_img: 人脸图像
            eye: 眼睛位置(x, y, w, h)
            
        Returns:
            处理后的人脸图像
        """
        if face_img is None:
            return face_img
        
        x, y, w, h = eye
        
        # 确保坐标在有效范围内
        x = max(0, x)
        y = max(0, y)
        w = min(w, face_img.shape[1] - x)
        h = min(h, face_img.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return face_img
        
        # 创建结果图像
        result = face_img.copy()
        
        try:
            # 提取眼睛区域
            eye_region = face_img[y:y+h, x:x+w]
            
            # 转换为LAB颜色空间
            if len(eye_region.shape) == 3 and eye_region.shape[2] == 3:
                lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # 增强细节（锐化L通道）
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
                l_sharpened = cv2.filter2D(l, -1, kernel)
                
                # 调整锐化程度，避免过度锐化
                l_final = cv2.addWeighted(l, 0.7, l_sharpened, 0.3, 0)
                
                # 红眼检测和去除
                if np.mean(a) > 128:  # a通道过高可能表示偏红
                    a = np.clip(a * 0.8, 0, 255).astype(np.uint8)  # 降低红色
                
                # 合并通道
                lab_enhanced = cv2.merge([l_final, a, b])
                
                # 转换回BGR
                enhanced_eye = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                
                # 将增强后的眼睛区域放回图像
                result[y:y+h, x:x+w] = enhanced_eye
        except Exception as e:
            print(f"精细处理眼睛区域时出错: {str(e)}")
        
        return result
    
    def adjust_eyes_brightness(self, face_img: np.ndarray, eyes: List[Tuple[int, int, int, int]], 
                             target_brightness: float = 0.5) -> np.ndarray:
        """
        调整眼睛区域的亮度
        
        Args:
            face_img: 人脸图像
            eyes: 眼睛位置列表
            target_brightness: 目标亮度，0-1之间，0.5为原始亮度
            
        Returns:
            调整后的人脸图像
        """
        if face_img is None or not eyes:
            return face_img
        
        result = face_img.copy()
        
        for eye in eyes:
            x, y, w, h = eye
            
            # 确保坐标在有效范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, face_img.shape[1] - x)
            h = min(h, face_img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # 提取眼睛区域
            eye_region = face_img[y:y+h, x:x+w]
            
            # 转换为HSV颜色空间
            if len(eye_region.shape) == 3 and eye_region.shape[2] == 3:
                hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
                h_channel, s_channel, v_channel = cv2.split(hsv)
                
                # 计算当前亮度
                current_brightness = np.mean(v_channel) / 255.0
                
                # 计算亮度调整因子
                if current_brightness > 0:
                    brightness_factor = target_brightness / current_brightness
                else:
                    brightness_factor = target_brightness
                
                # 调整亮度
                v_adjusted = np.clip(v_channel * brightness_factor, 0, 255).astype(np.uint8)
                
                # 合并通道
                hsv_adjusted = cv2.merge([h_channel, s_channel, v_adjusted])
                
                # 转换回BGR
                adjusted_eye = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
                
                # 将调整后的眼睛区域放回图像
                result[y:y+h, x:x+w] = adjusted_eye
        
        return result
    
    def _enhance_eyelashes(self, eye_region: np.ndarray) -> np.ndarray:
        """
        增强睫毛细节
        
        Args:
            eye_region: 眼睛区域图像
            
        Returns:
            增强睫毛后的图像
        """
        if eye_region is None or eye_region.size == 0:
            return eye_region
        
        result = eye_region.copy()
        
        try:
            # 仅处理彩色图像
            if len(eye_region.shape) == 3:
                # 转为灰度图
                gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                
                # 提取边缘，用于找到可能的睫毛区域
                edges = cv2.Canny(gray, 50, 150)
                
                # 膨胀边缘以捕获更多的睫毛区域
                dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
                
                # 找到睫毛区域
                eyelash_mask = dilated.astype(np.float32) / 255.0
                
                # 转换到 LAB 颜色空间，增强色彩对比度
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # 增强亮度对比度
                l_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
                
                # 锐化边缘
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5
                l_enhanced = cv2.filter2D(l_enhanced, -1, kernel)
                
                # 只在睫毛区域应用增强
                l_final = l * (1 - eyelash_mask * 0.3) + l_enhanced * (eyelash_mask * 0.3)
                
                # 合并通道
                lab_enhanced = cv2.merge([l_final.astype(np.uint8), a, b])
                
                # 转换回 BGR
                result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"增强睫毛时出错: {str(e)}")
        
        return result 