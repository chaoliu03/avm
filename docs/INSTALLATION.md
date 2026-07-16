# 安装指南

## 系统要求

### 操作系统

- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS 10.14+
- Windows 10+ (推荐使用 WSL2)

### 硬件要求

- **内存**：最小需 4GB RAM，推荐 8GB+
- **存储空间**：1GB 空闲空间（用于构建和输出文件）
- **CPU**：推荐多核处理器以获得最佳性能

## 依赖项

### 必需依赖项

#### OpenCV (版本 3.0 及以上)

OpenCV 是图像处理的操作核心。

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install libopencv-dev libopencv-contrib-dev
```

**CentOS/RHEL:**

```bash
sudo yum install opencv-devel
# 或者针对较新版本：
sudo dnf install opencv-devel
```

**macOS (使用 Homebrew):**

```bash
brew install opencv
```

**手动安装：**
如果包管理器提供的版本不够新：

```bash
# 下载 OpenCV 源码
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x  # 或您想要的分支/版本

# 构建并安装
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      ..
make -j$(nproc)
sudo make install
```

#### CMake (版本 3.10 及以上)

用于构建项目。

**Ubuntu/Debian:**

```bash
sudo apt install cmake
```

**CentOS/RHEL:**

```bash
sudo yum install cmake
# 或者：sudo dnf install cmake
```

**macOS:**

```bash
brew install cmake
```

### 可选依赖项

#### Git (用于版本控制)

```bash
# Ubuntu/Debian
sudo apt install git

# CentOS/RHEL  
sudo yum install git

# macOS
git --version  # 通常已预装
```

## 安装步骤

### 方法 1：从 GitHub 克隆（推荐）

```bash
# 克隆仓库
git clone https://github.com/xixu-me/AVM.git
cd AVM

# 赋予脚本执行权限
chmod +x scripts/*.sh

# 构建项目
./scripts/build.sh
```

### 方法 2：下载并构建

```bash
# 下载并解压源码
wget https://github.com/xixu-me/AVM/archive/main.zip
unzip main.zip
cd AVM-main

# 赋予脚本执行权限
chmod +x scripts/*.sh

# 构建项目
./scripts/build.sh
```

### 手动构建过程

如果您更倾向于手动构建：

```bash
# 创建构建目录
mkdir -p build
cd build

# 使用 CMake 进行配置
cmake ..

# 构建（根据您的 CPU 核心数调整 -j 参数）
make -j$(nproc)

# 验证构建
ls bin/avm  # 如果构建成功，该文件应该存在
```

## 验证

### 测试安装

```bash
# 使用提供的示例图像进行快速测试
./scripts/run.sh

# 检查输出结果
ls build/stitched_result_with_su7.jpg  # 运行成功后，该文件应该存在
```

### 期望的输出文件

成功执行后，您应该在 `build/` 目录下看到：

```
build/
├── front_undis.jpg         # 前视去畸变图像
├── back_undis.jpg          # 后视去畸变图像  
├── left_undis.jpg          # 左视去畸变图像
├── right_undis.jpg         # 右视去畸变图像
├── bird_front_2.jpg        # 前向鸟瞰图
├── bird_back_2.jpg         # 后向鸟瞰图
├── bird_left_2.jpg         # 左向鸟瞰图
├── bird_right_2.jpg        # 右向鸟瞰图
└── stitched_result_with_su7.jpg  # 最终全景拼接结果
```

## 问题排查

### 未找到 OpenCV

```
CMake Error: Could not find OpenCV
```

**解决方案：**

- 验证 OpenCV 安装情况：`pkg-config --modversion opencv`
- 手动设置 OpenCV 路径：`export OpenCV_DIR=/path/to/opencv`
- 重新安装 OpenCV 开发包

### 构建错误

```
error: opencv2/opencv.hpp: No such file or directory
```

**解决方案：**

- 安装 OpenCV 开发头文件
- 检查 CMakeLists.txt 中的包含路径
- 验证编译器能找到 OpenCV 的包含路径

### 运行时错误

```
error while loading shared libraries: libopencv_core.so
```

**解决方案：**

- 更新共享库缓存：`sudo ldconfig`
- 添加到 LD_LIBRARY_PATH：`export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`

### 权限错误

```
Permission denied: ./scripts/build.sh
```

**解决方案：**

```bash
chmod +x scripts/*.sh
```

### 缺失输入图像

```
[ERROR] Unable to read input image files
```

**解决方案：**

- 确保输入图像位于 `assets/images/` 目录中
- 检查文件权限和格式（支持 PNG/JPG）
- 验证图像文件没有损坏

## 性能优化

### 构建优化

为了获得最佳性能，请使用 Release 模式构建：

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 运行优化

- 使用 SSD 存储以加快 I/O 读取/写入
- 确保有足够的 RAM 以避免磁盘虚拟内存交换（Swap）
- 在处理图像期间关闭不必要的应用程序

## 下一步

安装成功后：

1. **阅读文档**：查看 `README.md` 获取详细的使用说明。
2. **尝试自定义图像**：用您自己的鱼眼图像替换示例图像。
3. **探索参数配置**：阅读并调整源码中的相机参数。
4. **贡献**：参阅 `CONTRIBUTING.md` 了解开发指南。

## 获取帮助

如果您遇到问题：

1. 查看此问题排查部分。
2. 搜索已有的 GitHub Issue。
3. 创建一个新的 Issue 并附带详细的错误信息。
4. 包含您的系统环境信息和构建日志。
