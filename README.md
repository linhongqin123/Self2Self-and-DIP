# Single-Image Self-Supervised Denoising: DIP vs Self2Self

本项目是一个关于单图像自监督去噪（Single-Image Self-Supervised Denoising）的研究课题，主要实现了 **Deep Image Prior (DIP)** 和 **Self2Self (S2S)** 算法，并将其与传统方法 **BM3D** 以及预训练方法 **Neighbor2Neighbor (N2N)** 在标准数据集（Set12）上进行了深度对比分析。

##  核心特性

  - **零样本学习 (Zero-shot Learning)**：无需成对的带噪/清晰图像数据集，仅依靠单张带噪图像即可完成推理。
  - **多种算法集成**：
      - **DIP**: 利用神经网络结构先验，配合早停法（Early Stopping）恢复图像结构。
      - **Self2Self**: 结合伯努利采样遮挡（Bernoulli Masking）、Dropout 和集成推理（Ensemble Inference）保护图像纹理。
  - **自动化评测引擎**：内置 `benchmark.py`，支持在 $\sigma=15, 25, 35, 50$ 噪声水平下自动运行所有算法并生成指标报表。
  - **学术级可视化**：自动生成 PSNR 迭代曲线图、遮挡蒙版图以及多算法横向对比图。

## 🛠️ 环境配置

建议使用 Anaconda 创建虚拟环境以确保依赖项兼容：

```bash
# 创建环境
conda create -n cv_hw10 python=3.9
conda activate cv_hw10

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pandas scikit-image tqdm bm3d tabulate
```

*注：本项目开发环境基于 NVIDIA GeForce RTX 5070 Laptop GPU，建议在支持 CUDA 的环境下运行以获得最佳性能。*

##  项目结构

```text
.
├── Set12/                  # 测试数据集
├── benchmark_results/      # 自动生成的实验结果与对比图
├── n2n_weights.pth         # 预训练的 N2N 模型权重
├── dip_denoise.py                  # DIP 算法核心实现
├── Self2Self.py            # Self2Self 算法核心实现
├── benchmark.py            # 自动化全矩阵评测脚本
└── README.md
```

##  快速开始

### 1\. 准备数据

将待测试的图像放入 `Set12` 文件夹中。

### 2\. 运行自动化评测

执行以下命令，脚本将遍历所有图像并在 4 个噪声水平下运行所有算法：

```bash
python benchmark.py
```

运行结束后，你可以在终端看到如下格式的 Markdown 数据表，并在 `benchmark_results/` 目录下找到对比组图：

| Algorithm | 15 | 25 | 35 | 50 |
| :--- | :---: | :---: | :---: | :---: |
| BM3D | 32.40 | 29.97 | 28.31 | 26.33 |
| DIP | 29.13 | 26.71 | 25.25 | 23.85 |
| Self2Self | 27.34 | 25.81 | 25.13 | 23.84 |

##  实验结论

通过对 Set12 数据集的广泛测试，我们得出以下结论：

  - **DIP** 对低频结构（平滑区域）恢复效果极佳，但由于其全局先验特性，对早停步数极其敏感。
  - **Self2Self** 在高噪声水平（$\sigma=50$）下展现了更强的稳定性，其 SSIM 指标通常优于 DIP，能够更好地保护边缘和细微纹理。
  - **N2N** 在面对跨域数据（Domain Shift）时表现较弱，验证了单图像自监督方法在零样本场景下的灵活性。
