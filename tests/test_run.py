"""
快速测试脚本 - 测试整个系统的工作流程

执行简单的面部替换测试，不需要图形界面，适合命令行环境
"""

import os
import sys
import argparse
import tempfile
import cv2
import numpy as np

# 将项目根目录添加到路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_processor import VideoProcessor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频人物面部替换系统测试")
    parser.add_argument("--source", type=str, help="源人脸图像路径")
    parser.add_argument("--target", type=str, help="目标视频路径")
    parser.add_argument("--output", type=str, help="输出视频路径")
    parser.add_argument("--frames", type=int, default=10, help="处理的帧数")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    
    return parser.parse_args()


def test_face_swap(source_path, target_path, output_path, frames=10, model_path=None, use_cpu=False):
    """
    测试面部替换功能
    
    Args:
        source_path: 源人脸图像路径
        target_path: 目标视频路径
        output_path: 输出视频路径
        frames: 处理的帧数
        model_path: 模型路径
        use_cpu: 是否强制使用CPU
    """
    print(f"测试参数: 源={source_path}, 目标={target_path}, 输出={output_path}, 帧数={frames}")
    
    # 初始化视频处理器
    processor = VideoProcessor(use_cuda=not use_cpu)
    
    # 加载模型
    if model_path is None:
        model_path = os.path.join("data", "models", "face_swap_model.h5")
    
    if os.path.exists(model_path):
        processor.load_model(model_path)
    else:
        print(f"警告: 模型文件不存在 {model_path}")
        print("将使用未训练的模型，效果可能不佳")
        # 创建并加载未训练的模型
        processor.model = processor.load_model(None)
    
    # 处理视频
    try:
        output = processor.process_video(source_path, target_path, output_path, frame_limit=frames)
        print(f"测试完成! 输出视频: {output}")
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        return False
    finally:
        # 清理临时文件
        processor.cleanup()


def generate_test_data():
    """生成测试数据"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 生成测试源人脸图像 (蓝色方块，200x200)
    source_path = os.path.join(temp_dir, "test_face.jpg")
    source_img = np.zeros((200, 200, 3), dtype=np.uint8)
    source_img[:, :, 0] = 255  # 蓝色
    # 画一个简单的人脸
    cv2.circle(source_img, (100, 100), 80, (0, 0, 200), -1)  # 脸
    cv2.circle(source_img, (70, 70), 10, (255, 255, 255), -1)  # 左眼
    cv2.circle(source_img, (130, 70), 10, (255, 255, 255), -1)  # 右眼
    cv2.ellipse(source_img, (100, 130), (30, 10), 0, 0, 180, (0, 255, 255), -1)  # 嘴
    cv2.imwrite(source_path, source_img)
    
    # 生成测试目标视频 (红色方块，10帧，200x200)
    target_path = os.path.join(temp_dir, "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, 30, (200, 200))
    
    for i in range(10):
        target_img = np.zeros((200, 200, 3), dtype=np.uint8)
        target_img[:, :, 2] = 255  # 红色
        # 画一个简单的人脸，随着帧数移动
        cv2.circle(target_img, (100 + i * 5, 100), 80, (0, 0, 200), -1)  # 脸
        cv2.circle(target_img, (70 + i * 5, 70), 10, (255, 255, 255), -1)  # 左眼
        cv2.circle(target_img, (130 + i * 5, 70), 10, (255, 255, 255), -1)  # 右眼
        cv2.ellipse(target_img, (100 + i * 5, 130), (30, 10), 0, 0, 180, (0, 255, 255), -1)  # 嘴
        out.write(target_img)
    
    out.release()
    
    # 输出视频路径
    output_path = os.path.join(temp_dir, "output.mp4")
    
    return source_path, target_path, output_path, temp_dir


def main():
    """主函数"""
    args = parse_args()
    
    # 如果未提供源和目标，生成测试数据
    if args.source is None or args.target is None:
        print("未提供源或目标，生成测试数据...")
        source_path, target_path, output_path, temp_dir = generate_test_data()
        
        if args.output is None:
            args.output = output_path
    else:
        source_path = args.source
        target_path = args.target
        
        if args.output is None:
            args.output = os.path.join(tempfile.mkdtemp(), "output.mp4")
        
        temp_dir = None
    
    # 测试面部替换
    success = test_face_swap(
        source_path, target_path, args.output,
        frames=args.frames, model_path=args.model, use_cpu=args.cpu
    )
    
    # 如果使用了临时数据，清理
    if temp_dir:
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    # 返回状态码
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 