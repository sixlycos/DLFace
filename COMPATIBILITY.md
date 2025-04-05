# DLface 兼容性和环境要求指南

本文档详细说明了DLface项目的依赖需求、版本兼容性和测试环境，以帮助用户正确配置环境并解决可能的兼容性问题。

## Python版本兼容性

DLface项目在以下Python版本中进行了测试和验证：

| Python版本 | 兼容性 | 建议 |
|------------|------|------|
| 3.8        | ✅ 良好  | 稳定但对TensorFlow 2.15.0不兼容，使用2.12.0 |
| 3.9        | ✅ 最佳  | **推荐** - 所有组件兼容性最佳 |
| 3.10       | ✅ 最佳  | **推荐** - 所有组件兼容性最佳 |
| 3.11       | ✅ 良好  | 大多数功能正常，但部分依赖可能有兼容性问题 |
| 3.12       | ❌ 不支持 | TensorFlow等关键组件不支持Python 3.12 |
| <3.8       | ❌ 不支持 | 较老的Python版本不支持 |

## 关键依赖版本

| 依赖项 | 兼容版本 | 说明 |
|--------|---------|------|
| TensorFlow | 2.12.0 - 2.15.x | Python 3.8使用2.12.0，3.9-3.11使用2.15.0 |
| MediaPipe | 0.10.3 - 0.10.9 | 0.10.8版本与TensorFlow兼容性最佳 |
| face-alignment | 1.3.5 - 1.4.1 | 1.3.5版本在修复模型问题时兼容性最佳 |
| dlib | 19.22.1 - 19.24.2 | Windows用户可能需要预编译wheel |
| OpenCV | 4.7.0 - 4.8.1 | 与其他计算机视觉组件兼容 |
| numpy | 1.23.5 - 1.26.3 | 版本取决于Python和TensorFlow版本 |
| ml-dtypes | 0.2.0 | 对于Python 3.8和MediaPipe兼容性需要特定版本 |

## GPU支持

DLface支持GPU加速，但有以下要求：

| 组件 | 要求 |
|------|------|
| CUDA | 11.2 - 12.0 (取决于TensorFlow版本) |
| cuDNN | 8.1+ |
| GPU内存 | 至少2GB（推荐4GB+） |
| 支持的GPU | NVIDIA GeForce GTX 1060或更高版本 |

**注意**：GPU加速对视频处理速度有显著提升，但在某些系统上可能需要特定版本的CUDA和cuDNN。

## 操作系统兼容性

| 操作系统 | 状态 | 备注 |
|---------|------|------|
| Windows 10/11 | ✅ 完全支持 | 推荐使用WSL2以获得更好的兼容性 |
| Ubuntu 20.04+ | ✅ 完全支持 | 推荐的开发和部署环境 |
| macOS | ⚠️ 部分支持 | M1/M2芯片需要特殊配置，GPU加速有限 |

## 内存和存储要求

- **RAM**: 至少4GB，推荐8GB+
- **磁盘空间**: 
  - 基础安装: ~500MB
  - 模型文件: ~300MB
  - 处理视频所需临时空间: 取决于视频大小

## 常见兼容性问题和解决方案

### 1. dlib安装失败

**症状**: 安装时出现`CMake is not installed on your system!`错误，或编译失败。

**Windows解决方案**:
1. 安装CMake:
   - 从[cmake.org](https://cmake.org/download/)下载并安装
   - 安装时选择"Add CMake to the system PATH for all users"
2. 安装编译工具:
   - 安装Visual Studio 2019或更高版本，或仅安装[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. 重新安装dlib:
   ```bash
   pip install dlib
   ```
4. 或者使用预编译wheel (推荐简单方案):
   ```bash
   # 针对Python 3.11
   pip install https://github.com/jloh02/dlib/releases/download/v19.24/dlib-19.24.0-cp311-cp311-win_amd64.whl
   
   # 针对Python 3.9/3.10
   pip install https://github.com/jloh02/dlib/releases/download/v19.24/dlib-19.24.0-cp310-cp310-win_amd64.whl
   ```
   更多版本: [jloh02/dlib](https://github.com/jloh02/dlib/releases) 或 [datamagic2020/dlib_wheels](https://github.com/datamagic2020/dlib_wheels)

**Linux解决方案**:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
pip install dlib
```

**macOS解决方案**:
```bash
xcode-select --install
brew install cmake
pip install dlib
```

### 2. MediaPipe与TensorFlow版本冲突

**症状**: MediaPipe初始化失败，提示ml-dtypes版本错误

**解决方案**:
```bash
pip uninstall -y mediapipe ml-dtypes
pip install ml-dtypes==0.2.0
pip install mediapipe==0.10.8
```

### 3. 模型加载失败

**解决方案**:
```bash
# 重新下载所有模型
python download_model.py --all

# 检查模型文件权限
chmod -R 755 data/models/
```

### 4. GPU不被检测或使用

**解决方案**:
- 确认CUDA和cuDNN已正确安装
- 使用 `nvidia-smi` 命令验证GPU是否被系统识别
- 强制使用GPU: `python run.py --force-gpu`

## 测试环境配置

以下是我们开发和测试使用的环境配置，可作为参考：

### 开发环境

```
- Windows 11 Pro
- Python 3.9.13
- CUDA 11.7
- cuDNN 8.5
- NVIDIA GeForce RTX 3070
- TensorFlow 2.15.0
- MediaPipe 0.10.8
- 32GB RAM
```

### 测试环境 1 (CPU)

```
- Ubuntu 22.04 LTS
- Python 3.10.12
- TensorFlow 2.15.0 (CPU版本)
- MediaPipe 0.10.8
- 16GB RAM
```

### 测试环境 2 (Mac)

```
- macOS Monterey
- Python 3.9.7
- TensorFlow 2.12.0 (Mac优化版)
- MediaPipe 0.10.3
- Apple M1 Pro
- 16GB RAM
```

## 更新依赖

如果您需要升级或更新依赖，建议使用我们的安装脚本：

```bash
python install_dependencies.py --gpu  # 或 --cpu-only
```

它会自动检测您的Python版本并安装兼容的依赖项。 