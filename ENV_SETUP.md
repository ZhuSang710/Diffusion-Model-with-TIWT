# Fast-DDPM 训练/测试环境配置（Windows + 虚拟环境）

本文档以本仓库代码实际导入项与运行入口为准，提供一份可从零搭建的环境配置文本，用于：
- 训练：`fast_ddpm_main.py` / `ddpm_main.py`
- 采样：加 `--sample`（可选 `--fid` / `--interpolation`）
- 测试：加 `--test`（通常需要已有 checkpoint）

## 0. 前置条件
- Windows 10/11 64-bit
- Git（可选，但推荐）
- 建议安装 Anaconda/Miniconda（更容易复现）
- 若使用 GPU：已安装 NVIDIA 驱动（能运行 `nvidia-smi`）

## 1. 创建虚拟环境（推荐 Conda）
在仓库根目录（包含 environment.yml / requirements.txt 的目录）执行：

```powershell
conda env create -f environment.yml
conda activate fastddpm
```

如果你不想用 environment.yml，也可以：

```powershell
conda create -n fastddpm python=3.10.6 -y
conda activate fastddpm
```

## 1.1 创建虚拟环境（venv + pip）
如果你习惯使用 Python 自带的 venv（不依赖 Conda），在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

后续步骤与 Conda 相同（安装 PyTorch → 安装 requirements.txt）。

## 2. 安装 PyTorch（按 CPU/GPU 二选一）
本仓库代码只用到 PyTorch + torchvision 的常规能力（DataParallel、transforms、utils.save_image 等）。PyTorch 的安装与 CUDA 版本强相关，建议以 PyTorch 官网选择器为准；这里给出两套常用组合。

### 组合 A（默认推荐）：torch 2.0.1 + torchvision 0.15.2
#### GPU（CUDA 11.8 轮子）
```powershell
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

#### CPU-only
```powershell
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

### 组合 B（旧版本替代）：torch 1.12.1 + torchvision 0.13.1
当你的显卡/驱动环境不适配新版本时可用：

```powershell
pip install --upgrade pip
pip install torch==1.12.1 torchvision==0.13.1
```

说明：
- README 中同时出现过 `torch==1.12.1` 与 `torchvision==0.15.2`，两者并非标准匹配组合；本文提供“可用组合”以避免安装后运行时报错。

## 3. 安装其余依赖
```powershell
pip install -r requirements.txt
```

可选依赖（README 提及但代码当前未直接 import）：
- `opencv-python`（如果你后续加入了 OpenCV 预处理）
- `tensorboardX`（当前代码使用 `torch.utils.tensorboard`，一般不需要）

## 4. 最小导入自检（强烈建议先做）
```powershell
python -c "import torch, torchvision; import numpy, pandas, yaml; import PIL; import skimage; import medpy; import lmdb; import matplotlib; import requests; import pywt; print('OK'); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)"
```

GPU 自检（有 NVIDIA GPU 才有意义）：
```powershell
python -c "import torch; print('cuda_available:', torch.cuda.is_available()); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 5. 数据准备（目录结构必须匹配）
仓库默认期望的数据目录结构（与 README 一致）：

```text
Fast-DDPM-main/
  configs/
  data/
    LD_FD_CT_train/
    LD_FD_CT_test/
    PMUB-train/
    PMUB-test/
    Brats_train/
    Brats_test/
```

数据下载（作者处理后的数据集）：README 给出的链接
- https://drive.google.com/file/d/1kF0g8fMR5XPQ2FTbutfTQ-hwG_mTqerx/view?usp=drive_link

### BRATS 数据集格式（代码要求）
BRATS 数据读取逻辑见 `datasets/BRATS.py`：
- 需要 `data/Brats_train/A/*.npy` 与 `data/Brats_train/B/*.npy`
- 测试同理：`data/Brats_test/A/*.npy` 与 `data/Brats_test/B/*.npy`

## 6. 运行命令（训练 / 采样 / 测试）
### 6.1 Fast-DDPM 训练
```powershell
python fast_ddpm_main.py --config pmub_linear.yml --dataset PMUB --exp exp --doc fastddpm_pmub --scheduler_type uniform --timesteps 10 --ni
```

```powershell
python fast_ddpm_main.py --config ldfd_linear.yml --dataset LDFDCT --exp exp --doc fastddpm_ldfd --scheduler_type uniform --timesteps 10 --ni
```

```powershell
python fast_ddpm_main.py --config brats_linear.yml --dataset BRATS --exp exp --doc fastddpm_brats --scheduler_type uniform --timesteps 10 --ni
```

说明：
- `--config` 默认会从 `configs/` 下查找；也可传 `configs/xxx.yml`
- 日志：`exp/logs/<doc>/`
- TensorBoard：`exp/tensorboard/<doc>/`，启动命令示例：`tensorboard --logdir exp/tensorboard`

### 6.2 Fast-DDPM 采样（需要已有 checkpoint）
```powershell
python fast_ddpm_main.py --config pmub_linear.yml --dataset PMUB --exp exp --doc fastddpm_pmub --sample --scheduler_type uniform --timesteps 10 --ni
```

输出路径（由脚本创建）：
- `exp/image_samples/<doc>/...`

### 6.3 DDPM 训练/采样
```powershell
python ddpm_main.py --config pmub_linear.yml --dataset PMUB --exp exp --doc ddpm_pmub --timesteps 1000 --ni
```

```powershell
python ddpm_main.py --config pmub_linear.yml --dataset PMUB --exp exp --doc ddpm_pmub --sample --timesteps 1000 --ni
```

## 7. 小波增强版本（可选）
本仓库包含小波相关模型与配置（例如 `configs/ldfd_wavelet.yml`，并在 `fast_ddpm_main.py` 中根据 `model.use_wavelet` 分流）。

如需启用：
```powershell
python fast_ddpm_main.py --config ldfd_wavelet.yml --dataset LDFDCT --exp exp --doc fastddpm_ldfd_wavelet --scheduler_type uniform --timesteps 10 --ni
```

依赖要求：
- 必须安装 `PyWavelets`（本仓库已在 requirements.txt 中包含 `PyWavelets`）

## 8. 常见问题排查
### 8.1 CUDA 不可用
- 先运行本文件“GPU 自检”命令确认 `torch.cuda.is_available()`
- 确认 NVIDIA 驱动正常（`nvidia-smi`）
- 确认 PyTorch 安装的是 GPU 版本（通常来自 `https://download.pytorch.org/whl/cu118`）

### 8.2 GPU 显存不足（OutOfMemoryError）
可调整：
- `configs/*.yml` 里的 `training.batch_size`（优先减小）
- `model.ch` / `model.ch_mult`（减小模型规模）
- 选择更小的 `--timesteps`（Fast-DDPM 默认 10）

### 8.3 安装 medpy 失败
Windows 下建议：
- 确认 pip 已升级：`pip install -U pip`
- 若仍失败，可尝试先安装其依赖：`pip install -U numpy scipy SimpleITK`
