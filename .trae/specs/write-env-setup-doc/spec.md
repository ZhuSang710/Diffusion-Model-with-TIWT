# Fast-DDPM 训练/测试环境配置文档 Spec

## Why
当前仓库缺少可复现的环境配置清单（README 中提到 requirements.txt 但仓库未提供），且代码依赖较多（PyTorch、LMDB、MedPy、PyWavelets 等）。需要一份可以按步骤从零配置出可训练/可测试环境的“环境配置文本”，并与本仓库实际代码导入项保持一致。

## What Changes
- 新增一份环境配置文本（文档文件），覆盖 Windows 本地训练/采样/测试的完整环境准备流程
- 新增可直接安装依赖的依赖清单文件（pip requirements 与可选 conda environment.yml）
- 文档中明确：Python 版本、PyTorch+CUDA 选择、数据集目录结构、训练/采样/测试命令、常见报错排查
- **BREAKING**：无

## Impact
- Affected specs: 环境可复现性、训练/采样/测试可运行性、依赖版本一致性
- Affected code: 无（仅新增文档与依赖清单文件）

## ADDED Requirements
### Requirement: 环境配置文本
系统 SHALL 提供一份环境配置文本，使用户在全新机器/全新虚拟环境中，按步骤即可完成本仓库 Fast-DDPM/DDPM 的训练、采样与测试环境搭建。

#### Scenario: 全新环境搭建成功
- **WHEN** 用户按文档创建虚拟环境并安装依赖
- **AND WHEN** 用户按文档准备数据目录结构并执行示例命令
- **THEN** `python fast_ddpm_main.py ...` 与 `python ddpm_main.py ...` 能够正常启动并开始训练/采样（在 CPU 或 GPU 条件允许下）

### Requirement: 依赖清单
系统 SHALL 提供可直接用于安装的依赖清单文件，且其内容至少覆盖代码中出现的第三方导入项：
- torch / torchvision / tensorboard
- numpy / pandas / tqdm
- PyYAML / Pillow
- scikit-image / medpy
- lmdb / matplotlib / requests
- PyWavelets（对应新增的小波相关代码路径）

#### Scenario: 依赖覆盖校验
- **WHEN** 用户在新环境中安装依赖清单
- **THEN** 运行最小导入校验（文档内给出命令）不应出现 `ModuleNotFoundError`

### Requirement: PyTorch 版本兼容说明
系统 SHALL 在文档中给出 PyTorch/torchvision 的可用版本组合建议（至少包含与 README 提示的 Python 3.10.6 对齐的组合），并区分 CPU-only 与 CUDA 版本安装方式。

## MODIFIED Requirements
无

## REMOVED Requirements
无

