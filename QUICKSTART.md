# 基于深度学习的视频人物面部替换系统 - 快速启动指南

## 系统环境要求

- Python 3.8-3.11（推荐3.9或3.10）
- CUDA支持（可选，用于GPU加速）
- 磁盘空间至少1GB（用于模型文件和依赖）

## 快速开始

1. **安装依赖**

使用自动安装脚本，它将检测您的Python版本并安装兼容的依赖：

```bash
# 自动检测并安装（推荐）
python install_dependencies.py

# 强制使用GPU版本
python install_dependencies.py --gpu

# 强制使用CPU版本
python install_dependencies.py --cpu-only
```

或者手动安装依赖：

```bash
pip install -r requirements.txt
```

2. **下载预训练模型**

下载所有必需的模型文件：

```bash
python download_model.py --all
```

您也可以选择性下载特定模型：

```bash
# 仅下载dlib面部特征点模型
python download_model.py --dlib

# 仅下载MediaPipe模型
python download_model.py --mediapipe
```

3. **启动应用**

```bash
python run.py
```

## 使用步骤

1. 在左侧菜单选择「视频处理」模式
2. 点击「加载模型」按钮加载预训练模型
3. 上传源人脸图像（待替换到视频中的人脸）
4. 上传目标视频（需要替换人脸的视频）
5. 点击「生成预览」查看效果
6. 点击「开始处理视频」，处理完整视频
7. 下载处理后的视频

## 模型训练（可选）

如果您希望训练自己的模型，请准备包含清晰人脸的高质量图像，放置在 `data/dataset` 目录下，然后：

1. 在左侧菜单选择「模型训练」模式
2. 设置训练参数
3. 点击「开始训练」按钮

## 目录结构

```
├── app                    # 应用程序代码
│   ├── streamlit_app.py   # Streamlit前端应用
│   └── utils.py           # 工具函数
├── core                   # 核心模块
│   ├── autoencoder.py     # 自编码器模型
│   ├── eye_tracker.py     # 眼睛注视方向优化
│   ├── face_detector.py   # 人脸检测模块
│   ├── postprocessor.py   # 后处理模块
│   └── video_processor.py # 视频处理模块
├── data                   # 数据目录
│   ├── models             # 预训练模型
│   ├── dataset            # 训练数据集
│   └── samples            # 示例图像和视频
├── tests                  # 测试代码
├── run.py                 # 启动脚本
└── requirements.txt       # 项目依赖
```

## 常见问题

- **问题**: 无法加载模型
  **解决方案**: 执行 `python download_model.py --all` 重新下载所有模型

- **问题**: 未检测到GPU
  **解决方案**: 执行 `python run.py --force-cpu` 强制使用CPU模式，或确认已正确安装CUDA和对应版本的TensorFlow/PyTorch

- **问题**: 视频处理速度慢
  **解决方案**: 减少处理帧数或使用GPU加速

- **问题**: 面部替换效果不理想
  **解决方案**: 尝试使用更高质量的源人脸图像，或训练自定义模型

- **问题**: MediaPipe模型初始化失败
  **解决方案**: 执行 `python download_model.py --mediapipe` 单独下载MediaPipe模型

- **问题**: 特征点检测失败
  **解决方案**: 执行 `python download_model.py --dlib` 重新下载dlib特征点模型

- **问题**: 安装dlib失败
  **解决方案**: 
    * **Windows**: 
      1. 从[cmake.org](https://cmake.org/download/)下载并安装CMake，安装时选择"Add CMake to the system PATH"
      2. 安装Visual Studio 2019或更高版本，或仅安装[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
      3. 安装完成后运行`pip install dlib`
      4. 或直接安装预编译版本: `pip install https://github.com/jloh02/dlib/releases/download/v19.24/dlib-19.24.0-cp311-cp311-win_amd64.whl` (注意选择与您Python版本匹配的wheel文件)
    * **Linux**: 安装build-essential和cmake: `sudo apt-get update && sudo apt-get install build-essential cmake`，然后运行`pip install dlib`
    * **macOS**: 安装XCode命令行工具和cmake: `xcode-select --install && brew install cmake`，然后运行`pip install dlib`
    * 或查看更多版本: [jloh02/dlib](https://github.com/jloh02/dlib/releases) 或 [datamagic2020/dlib_wheels](https://github.com/datamagic2020/dlib_wheels)

## Python版本兼容性

| Python版本 | TensorFlow版本 | 说明 |
|-----------|---------------|------|
| 3.8       | 2.12.0        | 稳定，但某些功能可能较旧 |
| 3.9       | 2.15.0        | 推荐，最佳兼容性 |
| 3.10      | 2.15.0        | 推荐，最佳兼容性 |
| 3.11      | 2.15.0        | 支持，但部分依赖可能有兼容性问题 |
| 3.12      | 不支持        | 尚不完全兼容 |

## 可用的命令行选项

```bash
# 下载所有模型
python download_model.py --all

# 启动应用
python run.py

# 使用CPU模式启动
python run.py --force-cpu

# 使用GPU模式启动
python run.py --force-gpu

# 下载模型但不启动应用
python run.py --download-model --no-run
```

## 核心文件说明

- `run.py` - 主启动脚本
- `download_model.py` - 模型下载工具
- `install_dependencies.py` - 依赖安装脚本 
- `auto_gpu_detection.py` - GPU自动检测工具
- `app/streamlit_app.py` - 前端界面应用

## 联系方式

如有任何问题或建议，请联系：刘丰瑞 (邮箱地址) 