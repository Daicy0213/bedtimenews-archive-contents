#!/bin/bash

# 安装 Pytorch (假设你已经安装了 CUDA 11.8 或 12.x)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 安装 Hugging Face 生态组件
# trl: 专用于SFT微调的库
# peft: 参数高效微调 (LoRA)
# bitsandbytes: 量化库
# accelerate: 多卡分布式训练
pip install transformers datasets peft bitsandbytes trl accelerate pyyaml

pip install nvidia-ml-py

# 安装 flash-attn (可选，但在4090上推荐，能显著加速)
pip install flash-attn --no-build-isolation