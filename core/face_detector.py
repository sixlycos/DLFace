"""
面部检测模块 - 负责检测和对齐人脸

实现基于多种方法的人脸检测和特征点定位功能
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any, Union
import threading
import time
from .utils import read_image_with_path_fix

# 导入统一日志配置和GPU补丁
from .logger_config import setup_logger
from .gpu_patch import check_cuda_availability

# 设置日志
logger = setup_logger()

# 尝试导入dlib
try:
    import dlib
    DLIB_AVAILABLE = True
    logger.info("dlib已成功导入")
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib导入失败，将使用OpenCV替代方法")

# 尝试导入face_alignment
try:
    import face_alignment
    from face_alignment import LandmarksType
    FA_AVAILABLE = True
    logger.info("face_alignment已成功导入")
except ImportError:
    FA_AVAILABLE = False
    logger.warning("face_alignment导入失败，将使用替代方法")
    # 创建一个简单的LandmarksType枚举替代
    class LandmarksType:
        TWO_D = 1
        THREE_D = 2

# 尝试导入用于面部分割的库
try:
    import mediapipe as mp
    
    # 添加MediaPipe补丁
    try:
        import mediapipe_patch  # 自动修复MediaPipe初始化问题
        logger.info("已应用MediaPipe补丁")
    except ImportError:
        logger.warning("MediaPipe补丁不可用，尝试创建内联补丁")
        # 创建内联补丁
        try:
            # 设置环境变量
            mp_dir = os.path.dirname(mp.__file__)
            os.environ["MEDIAPIPE_MODEL_PATH"] = mp_dir
            os.environ["MEDIAPIPE_RESOURCE_DIR"] = mp_dir
            
            # 创建必要的目录
            modules_dir = os.path.join(mp_dir, "modules")
            os.makedirs(modules_dir, exist_ok=True)
            
            # 创建face_detection目录
            face_detection_dir = os.path.join(modules_dir, "face_detection")
            os.makedirs(face_detection_dir, exist_ok=True)
            
            # 创建模型文件
            short_range_model = os.path.join(face_detection_dir, "face_detection_short_range_cpu.binarypb")
            if not os.path.exists(short_range_model):
                with open(short_range_model, 'wb') as f:
                    f.write(b'placeholder')
                logger.info(f"创建了占位模型文件: {short_range_model}")
            
            logger.info("已创建MediaPipe内联补丁")
        except Exception as patch_error:
            logger.warning(f"创建MediaPipe内联补丁失败: {str(patch_error)}")
    
    MP_AVAILABLE = True
    logger.info("mediapipe已成功导入")
except ImportError:
    MP_AVAILABLE = False
    logger.warning("mediapipe导入失败，将使用替代方法进行面部分割")

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    logger.info("tensorflow已成功导入")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("tensorflow导入失败")


class FaceDetector:
    """面部检测类，实现面部检测和特征点定位功能"""
    
    def __init__(self, use_dlib: bool = True, predictor_path: Optional[str] = None,
                use_face_segmentation: bool = True, use_gpu: bool = True,
                detection_size: Tuple[int, int] = (640, 480)):
        """
        初始化人脸检测器
        
        参数:
            use_dlib: 是否使用dlib进行人脸检测和特征点检测
            predictor_path: dlib预测器模型路径，如果为None则使用默认路径
            use_face_segmentation: 是否使用面部分割
            use_gpu: 是否使用GPU加速
            detection_size: 用于检测的图像大小，可以调小以提高速度
        """
        # 保存参数
        self.use_dlib = use_dlib and DLIB_AVAILABLE
        self.use_face_segmentation = use_face_segmentation and MP_AVAILABLE
        self.use_gpu = use_gpu and check_cuda_availability()
        self.detection_size = detection_size
        
        # 线程锁，用于多线程访问
        self._lock = threading.Lock()
        
        # 初始化所有属性，避免AttributeError
        # dlib相关
        self.detector = None
        self.predictor = None
        
        # face_alignment相关
        self.fa = None
        
        # OpenCV相关
        self.face_cascade = None
        self.eye_cascade = None
        self.facemark = None
        self.dnn_face_detector = None
        
        # MediaPipe相关
        self.mp_face_detection = None
        self.mp_face_mesh = None
        self.mp_drawing = None
        self.face_detection = None
        self.face_mesh = None
        
        # 面部分割相关
        self.bisenet_model = None
        
        # 初始化检测器
        self._init_detector()
        
        # 如果使用面部分割，初始化分割器
        if self.use_face_segmentation:
            self._init_face_segmentation()
        
        # 记录初始化信息
        logger.info(f"人脸检测器初始化完成, 使用dlib: {self.use_dlib}, 使用面部分割: {self.use_face_segmentation}, GPU加速: {self.use_gpu}")
    
    def _load_bisenet_model(self) -> None:
        """加载BiSeNet面部分割模型（如果可用）"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow不可用，无法加载BiSeNet模型")
            return
        
        try:
            # 检查模型文件是否存在
            model_path = os.path.join("data", "models", "bisenet_face_parsing.h5")
            if os.path.exists(model_path):
                logger.info(f"正在加载BiSeNet面部分割模型: {model_path}")
                self.bisenet_model = load_model(model_path)
                logger.info("BiSeNet面部分割模型加载成功")
            else:
                logger.warning(f"BiSeNet模型文件不存在: {model_path}")
                self.bisenet_model = None
        except Exception as e:
            logger.error(f"加载BiSeNet模型时出错: {str(e)}")
            self.bisenet_model = None
            
    def _init_detector(self) -> None:
        """初始化检测器和特征点预测器"""
        # 使用dlib
        if self.use_dlib:
            try:
                # 初始化人脸检测器
                self.detector = dlib.get_frontal_face_detector()
                
                # 加载特征点预测器
                predictor_path = os.path.join("data", "models", "shape_predictor_68_face_landmarks.dat")
                if not os.path.exists(predictor_path):
                    logger.warning(f"特征点预测器模型不存在: {predictor_path}")
                    logger.warning("将使用OpenCV特征点检测")
                    self.predictor = None
                else:
                    self.predictor = dlib.shape_predictor(predictor_path)
                
                # 尝试初始化face_alignment模块
                if FA_AVAILABLE:
                    try:
                        self.fa = face_alignment.FaceAlignment(
                            LandmarksType.TWO_D, 
                            flip_input=False, 
                            device='gpu' if self.use_gpu else 'cpu'
                        )
                        logger.info("face_alignment初始化成功")
                    except Exception as fa_error:
                        logger.warning(f"face_alignment初始化失败: {str(fa_error)}")
                        logger.warning("尝试使用备用方法初始化face_alignment")
                        try:
                            # 使用更加兼容的方式初始化
                            self.fa = face_alignment.FaceAlignment(
                                LandmarksType.TWO_D, 
                                flip_input=False, 
                                device='cpu' # 强制使用CPU模式
                            )
                            logger.info("face_alignment备用初始化成功")
                        except Exception as e:
                            logger.warning(f"face_alignment备用初始化也失败: {str(e)}")
                            self.fa = None
                else:
                    self.fa = None
                    logger.warning("face_alignment不可用，将使用dlib特征点检测")
            except Exception as e:
                logger.error(f"初始化dlib检测器时出错: {str(e)}")
                self.use_dlib = False
                self.detector = None
                self.predictor = None
                self.fa = None
        
        # 使用OpenCV
        if not self.use_dlib:
            try:
                # 加载OpenCV的人脸检测器
                face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                
                # 加载眼睛检测器
                eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                
                # 初始化OpenCV的特征点检测器
                self.facemark = cv2.face.createFacemarkLBF()
                lbf_model_path = os.path.join("data", "models", "lbfmodel.yaml")
                if os.path.exists(lbf_model_path):
                    self.facemark.loadModel(lbf_model_path)
                    logger.info("已加载OpenCV特征点模型")
                else:
                    logger.warning(f"OpenCV特征点模型不存在: {lbf_model_path}")
                    self.facemark = None
                
                # 使用DNN模型进行人脸检测
                self.dnn_face_detector = None
                proto_path = os.path.join("data", "models", "deploy.prototxt")
                model_path = os.path.join("data", "models", "res10_300x300_ssd_iter_140000.caffemodel")
                if os.path.exists(proto_path) and os.path.exists(model_path):
                    self.dnn_face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                    
                    # 设置DNN的计算后端和目标
                    if self.use_gpu:
                        self.dnn_face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.dnn_face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        logger.info("DNN人脸检测器将使用CUDA加速")
                    else:
                        logger.info("DNN人脸检测器将使用CPU")
                    
                    logger.info("已加载DNN人脸检测器")
                else:
                    logger.warning("DNN人脸检测模型文件不存在，将使用Haar级联检测器")
                
                logger.info("已初始化OpenCV检测器")
            except Exception as e:
                logger.error(f"初始化OpenCV检测器时出错: {str(e)}")
                self.face_cascade = None
                self.eye_cascade = None
                self.facemark = None
                self.dnn_face_detector = None
        
        # 加载BiSeNet模型（如果TensorFlow可用）
        if TF_AVAILABLE:
            self._load_bisenet_model()
    
    def _init_face_segmentation(self) -> None:
        """初始化面部分割"""
        if MP_AVAILABLE:
            try:
                # 初始化MediaPipe的Face Detection
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_drawing = mp.solutions.drawing_utils
                
                # 创建检测器实例
                self.face_detection = self.mp_face_detection.FaceDetection(
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe人脸检测初始化成功")
                
                # 初始化Face Mesh
                try:
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        min_detection_confidence=0.5
                    )
                    logger.info("MediaPipe面部网格初始化成功")
                except Exception as e:
                    logger.warning(f"MediaPipe面部网格初始化失败: {str(e)}")
                    self.face_mesh = None
                    
                    # 尝试创建一个备用的Face Mesh
                    try:
                        from collections import namedtuple
                        
                        # 创建一个简单的替代类
                        class FallbackFaceMesh:
                            def __init__(self):
                                pass
                                
                            def process(self, image):
                                Result = namedtuple('Result', ['multi_face_landmarks'])
                                return Result(multi_face_landmarks=[])
                        
                        self.face_mesh = FallbackFaceMesh()
                        logger.info("已创建MediaPipe面部网格备用实现")
                    except Exception as fallback_error:
                        logger.warning(f"创建MediaPipe面部网格备用实现失败: {str(fallback_error)}")
                
                # 加载BiSeNet模型用于面部分割 (如果TensorFlow可用)
                self._load_bisenet_model()
                
                logger.info("Mediapipe面部分割模块初始化完成")
            except Exception as e:
                logger.warning(f"初始化MediaPipe时出错: {str(e)}")
                
                # 创建备用类
                try:
                    from collections import namedtuple
                    
                    # 创建一个简单的替代类
                    class FallbackDetection:
                        def __init__(self, min_detection_confidence=0.5):
                            self.min_detection_confidence = min_detection_confidence
                            
                        def process(self, image):
                            Result = namedtuple('Result', ['detections'])
                            return Result(detections=[])
                
                    self.face_detection = FallbackDetection()
                    logger.info("已创建MediaPipe人脸检测的替代实现")
                except Exception as fallback_error:
                    logger.warning(f"创建备用检测器失败: {str(fallback_error)}")
                    self.use_face_segmentation = False
    
    def detect_faces(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
        """
        检测图像中的人脸
        
        参数:
            image: 输入图像
            roi: 感兴趣区域 (x, y, width, height)，如果指定则只在该区域内检测
            
        返回:
            检测到的人脸列表，每个人脸是一个字典，包含bbox和landmarks
        """
        # 确保所有需要的属性都存在
        for attr in ['detector', 'predictor', 'fa', 'face_cascade', 'eye_cascade', 
                     'facemark', 'dnn_face_detector', 'mp_face_detection', 
                     'mp_face_mesh', 'face_detection', 'face_mesh']:
            if not hasattr(self, attr):
                setattr(self, attr, None)
                logger.warning(f"属性 {attr} 不存在，已设置为None")
        
        # 检查图像
        if image is None or image.size == 0:
            logger.error("输入图像为空")
            return []
        
        # 如果指定了ROI，裁剪图像
        roi_offset_x, roi_offset_y = 0, 0
        if roi is not None:
            x, y, w, h = roi
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                roi_image = image[y:y+h, x:x+w]
                roi_offset_x, roi_offset_y = x, y
            else:
                logger.warning(f"无效的ROI: {roi}，将使用整个图像")
                roi_image = image
        else:
            roi_image = image
        
        # 调整图像大小以加速处理
        orig_h, orig_w = roi_image.shape[:2]
        scale_factor = min(self.detection_size[0] / orig_w, self.detection_size[1] / orig_h)
        
        if scale_factor < 1.0:
            # 只有当图像过大时才调整大小
            resized_w = int(orig_w * scale_factor)
            resized_h = int(orig_h * scale_factor)
            resized_image = cv2.resize(roi_image, (resized_w, resized_h))
        else:
            resized_image = roi_image
            scale_factor = 1.0
        
        # 检测开始时间
        start_time = time.time()
        faces = []
        
        # 使用dlib检测
        if self.use_dlib:
            with self._lock:  # 加锁以确保线程安全
                # 转换为灰度图像
                if len(resized_image.shape) == 3:
                    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = resized_image
                
                # 使用dlib检测人脸
                try:
                    if hasattr(self, 'detector') and self.detector is not None:
                        dlib_faces = self.detector(gray, 1)
                        
                        # 处理每个检测到的人脸
                        for face in dlib_faces:
                            # 获取边界框
                            x = face.left()
                            y = face.top()
                            w = face.right() - face.left()
                            h = face.bottom() - face.top()
                            
                            # 调整回原始图像大小
                            x = int(x / scale_factor)
                            y = int(y / scale_factor)
                            w = int(w / scale_factor)
                            h = int(h / scale_factor)
                            
                            # 添加ROI偏移
                            x += roi_offset_x
                            y += roi_offset_y
                            
                            face_dict = {
                                'bbox': (x, y, w, h),
                                'confidence': None  # dlib没有置信度
                            }
                            
                            # 提取特征点
                            landmarks = []
                            
                            if hasattr(self, 'fa') and self.fa is not None:
                                # 使用face_alignment提取特征点
                                try:
                                    # 裁剪人脸区域
                                    face_img = image[max(0, y):min(image.shape[0], y+h), max(0, x):min(image.shape[1], x+w)]
                                    if face_img.size > 0:
                                        # 调整大小以加快处理
                                        face_img_resized = cv2.resize(face_img, (256, 256))
                                        # 预测特征点
                                        preds = self.fa.get_landmarks(face_img_resized)
                                        if preds:
                                            # 调整特征点坐标回原始图像
                                            scale_x = w / 256
                                            scale_y = h / 256
                                            landmarks = [(int(p[0] * scale_x) + x, int(p[1] * scale_y) + y) for p in preds[0]]
                                except Exception as e:
                                    logger.warning(f"使用face_alignment提取特征点失败: {str(e)}")
                            
                            if not landmarks and hasattr(self, 'predictor') and self.predictor is not None:
                                # 如果face_alignment失败，使用dlib提取特征点
                                try:
                                    # 将dlib矩形调整回缩放后的大小
                                    scaled_rect = dlib.rectangle(
                                        int(face.left()), 
                                        int(face.top()), 
                                        int(face.right()), 
                                        int(face.bottom())
                                    )
                                    shape = self.predictor(gray, scaled_rect)
                                    landmarks = [(int(shape.part(i).x / scale_factor) + roi_offset_x, 
                                                int(shape.part(i).y / scale_factor) + roi_offset_y) 
                                            for i in range(68)]
                                except Exception as e:
                                    logger.warning(f"使用dlib提取特征点失败: {str(e)}")
                            
                            face_dict['landmarks'] = landmarks
                            faces.append(face_dict)
                    else:
                        logger.warning("dlib检测器不可用，跳过dlib检测")
                except Exception as e:
                    logger.error(f"dlib人脸检测失败: {str(e)}")
        
        # 使用OpenCV检测
        if not self.use_dlib or not faces:
            with self._lock:  # 加锁以确保线程安全
                # 转换为灰度图像
                if len(resized_image.shape) == 3:
                    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = resized_image
                
                # 使用DNN模型检测（如果可用）
                if hasattr(self, 'dnn_face_detector') and self.dnn_face_detector is not None:
                    try:
                        h, w = resized_image.shape[:2]
                        blob = cv2.dnn.blobFromImage(cv2.resize(resized_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                        self.dnn_face_detector.setInput(blob)
                        detections = self.dnn_face_detector.forward()
                        
                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.5:  # 过滤低置信度检测
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (x1, y1, x2, y2) = box.astype("int")
                                
                                # 计算边界框
                                x = int(x1 / scale_factor)
                                y = int(y1 / scale_factor)
                                w = int((x2 - x1) / scale_factor)
                                h = int((y2 - y1) / scale_factor)
                                
                                # 添加ROI偏移
                                x += roi_offset_x
                                y += roi_offset_y
                                
                                face_dict = {
                                    'bbox': (x, y, w, h),
                                    'confidence': float(confidence)
                                }
                                
                                # 尝试使用OpenCV提取特征点
                                landmarks = []
                                if hasattr(self, 'facemark') and self.facemark is not None:
                                    try:
                                        # 创建OpenCV兼容的面部矩形
                                        faces_rect = np.array([[x1, y1, x2-x1, y2-y1]])
                                        success, landmarks_list = self.facemark.fit(gray, faces_rect)
                                        if success and len(landmarks_list) > 0:
                                            # 调整特征点坐标回原始图像
                                            landmarks = [(int(p[0] / scale_factor) + roi_offset_x, 
                                                         int(p[1] / scale_factor) + roi_offset_y) 
                                                       for p in landmarks_list[0][0]]
                                    except Exception as e:
                                        logger.warning(f"OpenCV特征点提取失败: {str(e)}")
                                
                                face_dict['landmarks'] = landmarks
                                faces.append(face_dict)
                    except Exception as e:
                        logger.error(f"DNN人脸检测失败: {str(e)}")
                        logger.info(f"跳过DNN检测，尝试使用级联分类器")
                else:
                    logger.info("DNN人脸检测器不可用，使用替代方法")
                
                # 如果DNN检测失败或不可用，使用级联分类器
                if not faces and hasattr(self, 'face_cascade') and self.face_cascade is not None:
                    try:
                        # 使用Haar级联分类器检测人脸
                        opencv_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        for (x, y, w, h) in opencv_faces:
                            # 调整回原始图像大小
                            x = int(x / scale_factor)
                            y = int(y / scale_factor)
                            w = int(w / scale_factor)
                            h = int(h / scale_factor)
                            
                            # 添加ROI偏移
                            x += roi_offset_x
                            y += roi_offset_y
                            
                            face_dict = {
                                'bbox': (x, y, w, h),
                                'confidence': None  # Haar级联分类器没有置信度
                            }
                            
                            # 尝试提取特征点
                            landmarks = []
                            if hasattr(self, 'facemark') and self.facemark is not None:
                                try:
                                    # 创建OpenCV兼容的面部矩形
                                    faces_rect = np.array([[int(x*scale_factor), int(y*scale_factor), 
                                                          int(w*scale_factor), int(h*scale_factor)]])
                                    success, landmarks_list = self.facemark.fit(gray, faces_rect)
                                    if success and len(landmarks_list) > 0:
                                        # 调整特征点坐标回原始图像
                                        landmarks = [(int(p[0] / scale_factor) + roi_offset_x, 
                                                     int(p[1] / scale_factor) + roi_offset_y) 
                                                   for p in landmarks_list[0][0]]
                                except Exception as e:
                                    logger.warning(f"OpenCV特征点提取失败: {str(e)}")
                            
                            face_dict['landmarks'] = landmarks
                            faces.append(face_dict)
                    except Exception as e:
                        logger.error(f"Haar级联分类器人脸检测失败: {str(e)}")
        
        # 记录检测时间
        detection_time = time.time() - start_time
        if faces:
            logger.debug(f"检测到 {len(faces)} 个人脸，耗时: {detection_time:.4f}秒")
        
        return faces
    
    def segment_face(self, image: np.ndarray, face: Dict[str, Any]) -> np.ndarray:
        """
        使用面部分割技术处理人脸，尤其是遮挡的人脸
        
        Args:
            image: 输入图像
            face: 人脸信息，包含边界框和特征点
            
        Returns:
            面部分割掩码，值域为[0, 255]，0表示背景，255表示前景
        """
        if image is None or face is None:
            logger.error("输入图像或人脸信息为空")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 获取人脸边界框
        bbox = face.get('bbox')
        if bbox is None:
            logger.error("人脸信息中没有边界框")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        x, y, w, h = bbox
        
        # 创建空白掩码
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 使用MediaPipe进行面部分割
        if self.use_face_segmentation and self.face_mesh is not None:
            try:
                # 裁剪人脸区域并调整大小
                face_img = image[y:y+h, x:x+w]
                if face_img.size == 0:
                    logger.warning("人脸区域为空")
                    return mask
                
                # 确保人脸图像有3个通道
                if len(face_img.shape) == 2:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                elif face_img.shape[2] == 4:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
                elif face_img.shape[2] == 3 and face_img.dtype == np.uint8:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # 处理图像
                results = self.face_mesh.process(face_img)
                
                # 如果检测到面部特征点
                if results.multi_face_landmarks:
                    # 获取第一个人脸的特征点
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    
                    # 创建面部区域的掩码
                    face_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 提取面部轮廓点
                    contour_points = []
                    for i in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]:
                        # 检查索引是否在范围内
                        if i < len(face_landmarks):
                            landmark = face_landmarks[i]
                            point = (int(landmark.x * w), int(landmark.y * h))
                            contour_points.append(point)
                    
                    if contour_points:
                        # 将轮廓点转换为numpy数组
                        contour_points = np.array(contour_points, dtype=np.int32)
                        
                        # 绘制填充轮廓
                        cv2.fillPoly(face_mask, [contour_points], 255)
                        
                        # 将面部掩码复制到完整掩码上
                        mask[y:y+h, x:x+w] = face_mask
            except Exception as e:
                logger.error(f"MediaPipe面部分割出错: {str(e)}")
        
        # 如果MediaPipe分割失败或不可用，尝试使用BiSeNet模型
        if np.sum(mask) == 0 and hasattr(self, 'bisenet_model') and self.bisenet_model is not None:
            try:
                # 裁剪人脸区域并调整大小
                face_img = image[y:y+h, x:x+w]
                if face_img.size == 0:
                    logger.warning("人脸区域为空")
                    return mask
                
                # 调整大小为模型输入尺寸
                face_img_resized = cv2.resize(face_img, (512, 512))
                
                # 确保图像格式正确
                if len(face_img_resized.shape) == 2:
                    face_img_resized = cv2.cvtColor(face_img_resized, cv2.COLOR_GRAY2RGB)
                elif face_img_resized.shape[2] == 4:
                    face_img_resized = cv2.cvtColor(face_img_resized, cv2.COLOR_BGRA2RGB)
                elif face_img_resized.shape[2] == 3 and face_img_resized.dtype == np.uint8:
                    face_img_resized = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
                
                # 预处理图像
                face_img_norm = face_img_resized.astype(np.float32) / 255.0
                face_img_norm = np.expand_dims(face_img_norm, axis=0)
                
                # 预测分割掩码
                pred_mask = self.bisenet_model.predict(face_img_norm)[0]
                
                # 处理预测结果
                if len(pred_mask.shape) == 3 and pred_mask.shape[2] > 1:
                    # 如果是多类别分割（例如19类），合并人脸部分的类别
                    face_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 人脸相关的类别
                    face_mask = np.zeros(pred_mask.shape[:2], dtype=np.uint8)
                    for cls in face_classes:
                        face_mask |= (np.argmax(pred_mask, axis=-1) == cls).astype(np.uint8)
                else:
                    # 如果是二元分割
                    face_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                
                # 调整大小为原始人脸尺寸
                face_mask = cv2.resize(face_mask, (w, h))
                
                # 将面部掩码复制到完整掩码上
                mask[y:y+h, x:x+w] = face_mask
            except Exception as e:
                logger.error(f"BiSeNet面部分割出错: {str(e)}")
                logger.info("BiSeNet模型不可用或出错，将使用基于特征点的简单分割")
        
        # 如果上述方法都失败，使用基于特征点的简单分割
        if np.sum(mask) == 0:
            try:
                # 获取特征点
                landmarks = face.get('landmarks', [])
                
                if landmarks and len(landmarks) > 0:
                    # 转换为numpy数组
                    points = np.array(landmarks, dtype=np.int32)
                    
                    # 创建凸包
                    hull = cv2.convexHull(points)
                    
                    # 填充凸包区域
                    cv2.fillConvexPoly(mask, hull, 255)
                else:
                    # 如果没有特征点，使用椭圆近似人脸
                    center = (x + w // 2, y + h // 2)
                    axes = (w // 2, h // 2)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            except Exception as e:
                logger.error(f"基于特征点的面部分割出错: {str(e)}")
                # 如果所有方法都失败，使用矩形区域
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        
        # 平滑掩码边缘
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def align_face(self, image: np.ndarray, face: Dict[str, Any], target_size: Tuple[int, int] = (256, 256),
                  handle_occlusion: bool = False) -> np.ndarray:
        """
        对齐人脸
        
        Args:
            image: 输入图像
            face: 人脸信息，包含边界框和特征点
            target_size: 目标大小
            handle_occlusion: 是否处理遮挡
            
        Returns:
            对齐后的人脸图像
        """
        # 检查输入
        if image is None or face is None:
            logger.error("输入图像或人脸信息为空")
            return None
        
        # 获取人脸边界框
        bbox = face.get('bbox')
        if bbox is None:
            logger.error("人脸信息中没有边界框")
            return None
        
        try:
            x, y, w, h = bbox
            
            # 确保坐标在有效范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                logger.error("人脸边界框无效")
                return None
            
            # 提取人脸区域
            face_img = image[y:y+h, x:x+w]
            
            # 处理遮挡
            if handle_occlusion and hasattr(self, 'use_face_segmentation') and self.use_face_segmentation:
                try:
                    # 获取面部分割掩码
                    face_mask = self.segment_face(image, face)
                    
                    # 在有掩码的情况下处理遮挡
                    if np.sum(face_mask) > 0:
                        # 裁剪掩码到人脸区域
                        face_mask_roi = face_mask[y:y+h, x:x+w]
                        
                        # 将遮挡区域替换为周围像素的平均值
                        occlusion = (face_mask_roi < 128)
                        if np.any(occlusion):
                            # 对于彩色图像
                            if len(face_img.shape) == 3:
                                for c in range(face_img.shape[2]):
                                    channel = face_img[:, :, c]
                                    # 计算非遮挡区域的平均值
                                    mean_value = channel[~occlusion].mean()
                                    # 将遮挡区域替换为平均值
                                    channel[occlusion] = mean_value
                                    face_img[:, :, c] = channel
                            # 对于灰度图像
                            else:
                                # 计算非遮挡区域的平均值
                                mean_value = face_img[~occlusion].mean()
                                # 将遮挡区域替换为平均值
                                face_img[occlusion] = mean_value
                except Exception as e:
                    logger.warning(f"处理遮挡时出错: {str(e)}")
                    # 出错时继续执行，不处理遮挡
            
            # 获取特征点
            landmarks = face.get('landmarks', [])
            
            # 如果有足够的特征点，使用特征点对齐
            if landmarks and len(landmarks) >= 5:
                # 对于68点模型，提取5个关键点（眼睛和嘴巴）
                if len(landmarks) == 68:
                    src_points = np.array([
                        landmarks[36],  # 左眼左角
                        landmarks[45],  # 右眼右角
                        landmarks[30],  # 鼻尖
                        landmarks[48],  # 嘴左角
                        landmarks[54]   # 嘴右角
                    ], dtype=np.float32)
                # 对于其他模型，使用前5个点
                else:
                    src_points = np.array(landmarks[:5], dtype=np.float32)
                
                # 调整特征点坐标，使其相对于人脸区域
                src_points[:, 0] -= x
                src_points[:, 1] -= y
                
                # 定义目标特征点位置（根据目标大小进行缩放）
                target_width, target_height = target_size
                dst_points = np.array([
                    [target_width * 0.3, target_height * 0.3],    # 左眼
                    [target_width * 0.7, target_height * 0.3],    # 右眼
                    [target_width * 0.5, target_height * 0.5],    # 鼻尖
                    [target_width * 0.35, target_height * 0.7],   # 嘴左角
                    [target_width * 0.65, target_height * 0.7]    # 嘴右角
                ], dtype=np.float32)
                
                # 计算变换矩阵
                M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
                
                # 应用变换
                aligned_face = cv2.warpAffine(face_img, M, target_size)
            else:
                # 如果没有足够的特征点，简单地调整大小
                aligned_face = cv2.resize(face_img, target_size)
            
            return aligned_face
        except Exception as e:
            logger.error(f"对齐人脸时出错: {str(e)}")
            
            # 如果对齐失败，简单地调整大小
            try:
                face_img = image[y:y+h, x:x+w]
                return cv2.resize(face_img, target_size)
            except:
                return None
    
    def extract_faces(self, image: np.ndarray, min_confidence: float = 0.8, target_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
        """
        从图像中提取人脸
        
        Args:
            image: 输入图像
            min_confidence: 最小置信度
            target_size: 目标大小
            
        Returns:
            人脸图像列表
        """
        # 检测人脸
        faces = self.detect_faces(image)
        
        # 过滤低置信度的人脸
        faces = [face for face in faces if face.get('confidence', 0) >= min_confidence]
        
        # 对齐人脸
        aligned_faces = []
        for face in faces:
            aligned_face = self.align_face(image, face, target_size)
            aligned_faces.append(aligned_face)
        
        return aligned_faces
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict[str, Any]], draw_landmarks: bool = True) -> np.ndarray:
        """
        在图像上绘制人脸边界框和特征点
        
        Args:
            image: 输入图像
            faces: 人脸信息列表
            draw_landmarks: 是否绘制特征点
            
        Returns:
            绘制后的图像
        """
        # 创建图像副本
        result = image.copy()
        
        # 绘制每个人脸
        for face in faces:
            # 获取边界框和特征点
            bbox = face.get('bbox', None)
            landmarks = face.get('landmarks', [])
            confidence = face.get('confidence', 0)
            
            # 绘制边界框
            if bbox is not None:
                x, y, w, h = bbox
                color = (0, 255, 0) if confidence >= 0.8 else (0, 165, 255)
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
                
                # 绘制置信度
                cv2.putText(result, f"{confidence:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制特征点
            if draw_landmarks and landmarks:
                for i, (px, py) in enumerate(landmarks):
                    cv2.circle(result, (px, py), 2, (0, 0, 255), -1)
                    
                    # 如果是68点模型，连接特定的点
                    if len(landmarks) == 68:
                        # 连接轮廓点
                        if i < 16 and i > 0:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        # 连接左眉毛
                        if 17 <= i < 21:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        # 连接右眉毛
                        if 22 <= i < 26:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        # 连接鼻梁
                        if 27 <= i < 30:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        # 连接鼻子底部
                        if 31 <= i < 35:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        # 连接左眼
                        if 36 <= i < 41:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        if i == 41:
                            cv2.line(result, landmarks[36], landmarks[41], (255, 0, 0), 1)
                        # 连接右眼
                        if 42 <= i < 47:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        if i == 47:
                            cv2.line(result, landmarks[42], landmarks[47], (255, 0, 0), 1)
                        # 连接嘴唇外部
                        if 48 <= i < 59:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        if i == 59:
                            cv2.line(result, landmarks[48], landmarks[59], (255, 0, 0), 1)
                        # 连接嘴唇内部
                        if 60 <= i < 67:
                            cv2.line(result, landmarks[i-1], landmarks[i], (255, 0, 0), 1)
                        if i == 67:
                            cv2.line(result, landmarks[60], landmarks[67], (255, 0, 0), 1)
        
        return result 