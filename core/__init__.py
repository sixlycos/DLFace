
# 导入补丁
try:
    import mediapipe_patch
    print("已应用MediaPipe补丁")
except ImportError:
    print("MediaPipe补丁不可用，可能会遇到初始化问题")

# 修复face_alignment枚举
try:
    import face_alignment
    # 确保使用正确的枚举
    from face_alignment import LandmarksType
    print(f"face_alignment初始化成功: {face_alignment.__version__}")
except ImportError:
    print("face_alignment未安装")

"""
视频人物面部替换系统核心模块包
""" 