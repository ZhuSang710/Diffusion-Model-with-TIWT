# Tasks
- [x] Task 1: 盘点仓库真实依赖与运行入口
  - [x] SubTask 1.1: 基于所有 .py 文件的 import 列表与 README“Requirements”，整理第三方依赖集合与可选依赖
  - [x] SubTask 1.2: 明确训练/采样/测试入口与命令参数（fast_ddpm_main.py、ddpm_main.py、configs/*.yml、data 目录结构）
  - [x] SubTask 1.3: 梳理 GPU/CPU 运行差异与常见错误点（CUDA、显存、LMDB、MedPy 安装）

- [x] Task 2: 产出“完整环境配置文本”（面向 Windows + 虚拟环境）
  - [x] SubTask 2.1: 给出两套可选路径：Conda 环境（推荐）与 venv+pip 环境
  - [x] SubTask 2.2: 文档必须包含：Python 版本、CUDA/驱动前置、PyTorch 安装指令（CPU/GPU）、其余依赖安装指令
  - [x] SubTask 2.3: 文档必须包含：数据放置结构、训练/采样/测试示例命令、最小导入校验命令、日志/输出目录说明
  - [x] SubTask 2.4: 文档必须包含：与本仓库“新增小波支持”相关依赖说明（PyWavelets）与对应配置字段提示

- [x] Task 3: 增加可执行的依赖清单文件
  - [x] SubTask 3.1: 新增 requirements.txt（或等价文件），确保覆盖所有第三方依赖
  - [x] SubTask 3.2: 可选新增 environment.yml（conda），并在文档中说明如何使用
  - [x] SubTask 3.3: 针对 PyTorch/torchvision 版本组合不一致问题，选择一套默认组合并在文档中给出替代组合

- [x] Task 4: 验证文档可用性（最小可运行校验）
  - [x] SubTask 4.1: 在文档中提供最小自检步骤（如 `python -c "import ..."` 与一次 dry-run 命令）
  - [x] SubTask 4.2: 校验所有命令与路径在 Windows PowerShell 下可直接复制运行（必要时同时给出 cmd/PowerShell 写法差异说明）

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 2
