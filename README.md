# DLface - 深度学习视频人脸替换系统

DLface是一个基于深度学习的视频人脸替换系统，提供高质量的面部替换效果，同时支持眼神方向优化和面部细节增强功能。

## 功能特点

- **高质量面部替换**: 基于深度学习自编码器网络架构，实现自然且高质量的面部替换
- **眼神方向优化**: 自动调整替换后的眼睛注视方向，使其与原视频中的眼神方向一致
- **区域限定检测**: 支持在视频的特定区域进行人脸检测，避免处理不需要的人脸
- **遮挡人脸处理**: 能够处理部分遮挡的人脸，提高系统鲁棒性
- **质量评估系统**: 评估替换效果的质量，生成详细报告
- **GPU加速**: 支持CUDA加速，提高处理速度
- **多线程处理**: 利用多线程并行处理视频帧，提高效率
- **用户友好界面**: 基于Streamlit的直观界面，支持所有功能的配置

## 系统要求

- Python 3.8-3.11（推荐3.9或3.10）
- CUDA 11.2+（可选，用于GPU加速）
- 4GB以上内存
- Windows 10/11, Linux, macOS

详细的兼容性信息请查看 [COMPATIBILITY.md](COMPATIBILITY.md) 文档。

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/DLface.git
cd DLface
```

2. 安装依赖（使用自动安装脚本）：
```bash
# 自动检测并安装依赖
python install_dependencies.py

# 或使用GPU版本
python install_dependencies.py --gpu

# 或使用CPU版本
python install_dependencies.py --cpu-only
```

> **重要提示**: dlib库安装需要CMake编译器。如果安装过程中出现`CMake is not installed on your system!`错误，请按照以下步骤解决：
> 
> **Windows用户**:
> - 从[cmake.org](https://cmake.org/download/)下载并安装CMake，安装时选择"Add CMake to the system PATH"
> - 安装Visual Studio 2019 或以上版本，或仅安装[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> - 安装完成后重新运行`python install_dependencies.py`
>
> **或者使用预编译版本**：
> ```bash
> pip install https://github.com/jloh02/dlib/releases/download/v19.24/dlib-19.24.0-cp311-cp311-win_amd64.whl
> ```
> (根据您的Python版本选择相应的wheel文件)

如果您遇到其他安装问题，请参考 [COMPATIBILITY.md](COMPATIBILITY.md) 中的兼容性指南和解决方案。

3. 下载预训练模型：
```bash
python download_model.py --all
```

## 使用方法

### 运行前端界面

```bash
python run.py
```

或者直接使用：

```bash
streamlit run app/streamlit_app.py
```

### 命令行选项

```bash
python run.py [选项]
```

可用选项：
- `--download-model`: 下载预训练模型
- `--no-run`: 仅下载模型但不启动应用
- `--force-cpu`: 强制使用CPU模式
- `--force-gpu`: 强制使用GPU模式

详细的使用说明请查看 [QUICKSTART.md](QUICKSTART.md) 文档。

### 界面操作步骤

1. **加载模型**: 点击侧边栏的"加载/重置模型"按钮
2. **源与目标**: 上传源人脸图像和目标视频
3. **单帧预览**: 测试效果，调整参数
4. **视频处理**: 设置处理参数并执行处理
5. **下载结果**: 处理完成后，下载处理后的视频

## 项目结构

整合和清理后的项目包含以下核心文件和目录：

```
├── app/                    # 应用程序代码
│   └── streamlit_app.py    # Streamlit前端应用
├── core/                   # 核心功能模块
├── data/                   # 数据和模型目录
│   ├── models/             # 模型存放目录
│   ├── dataset/            # 训练数据集目录
│   └── samples/            # 示例图像和视频
├── examples/               # 示例代码
├── tests/                  # 测试代码
├── temp/                   # 临时文件目录
├── __main__.py             # Python模块入口点
├── run.py                  # 主启动脚本
├── auto_gpu_detection.py   # GPU检测工具
├── download_model.py       # 整合版模型下载工具
├── install_dependencies.py # 依赖安装脚本
├── COMPATIBILITY.md        # 兼容性和环境要求指南
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明
└── QUICKSTART.md           # 快速入门指南
```

## 模型下载说明

`download_model.py`脚本整合了所有模型的下载功能，包括：

1. **dlib面部特征点检测模型**: 用于面部特征点定位
2. **MediaPipe模型**: 用于面部检测和网格构建
3. **OpenCV DNN和LBF模型**: 用于人脸检测和特征点检测
4. **BiSeNet面部解析模型**: 用于面部区域分割

使用方法：
```bash
# 下载所有模型
python download_model.py --all

# 仅下载特定模型
python download_model.py --dlib
python download_model.py --mediapipe
python download_model.py --opencv
python download_model.py --bisenet
python download_model.py --face-landmarker
```

## 性能优化

### GPU加速

系统自动检测是否有可用的GPU，并在可用时使用GPU加速。为获得最佳性能：

1. 确保已安装CUDA和cuDNN
2. 确保安装了GPU版本的TensorFlow和PyTorch
3. 在高端GPU上，可以增加`num_workers`参数以提高并行处理效率

### 性能提示

- **调整检测大小**: 减小`detection_size`可以加快人脸检测速度
- **限制处理区域**: 使用ROI功能限制人脸检测区域可以显著提高处理速度
- **多线程处理**: 系统默认使用多线程处理视频帧，可以根据CPU核心数调整`num_workers`参数
- **批量处理**: 处理大型视频时，考虑先提取帧再处理，或分段处理

## 故障排除

- **内存不足**: 减小批处理大小或处理的视频分辨率
- **GPU内存不足**: 启用显存增长选项或减小模型大小
- **检测失败**: 尝试调整照明条件或使用不同的人脸检测器
- **模型加载错误**: 运行`python download_model.py --all`重新下载所有模型
- **其他问题**: 查看 [COMPATIBILITY.md](COMPATIBILITY.md) 中的常见问题解决方案

## 文档索引

- [README.md](README.md): 项目概述和主要功能说明
- [QUICKSTART.md](QUICKSTART.md): 快速入门指南
- [COMPATIBILITY.md](COMPATIBILITY.md): 兼容性和环境要求详细说明

## 许可证

MIT License

## 致谢

- 感谢所有开源项目的贡献者
- 特别感谢DeepFakes社区的技术支持 