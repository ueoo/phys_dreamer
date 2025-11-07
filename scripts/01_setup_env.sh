#!/usr/bin/env bash
set -euo pipefail

# Create and activate environment (adjust Python/CUDA versions as needed)
conda create -y -n physdreamer python=3.10
echo "Run: conda activate physdreamer"

# Install PyTorch (pick the right CUDA wheel for your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core requirements
pip install -r "$(dirname "$0")/../requirements.txt"

# Install diff-gaussian-rasterization (follow upstream instructions if needed)
# Example if you have it locally:
# git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
# cd diff-gaussian-rasterization && pip install . && cd -

# NVIDIA Warp for MPM
pip install warp-lang

# IO / video / diffusion (optional if using external SVD)
pip install imageio[ffmpeg] decord diffusers transformers accelerate safetensors

echo "Environment bootstrap complete. Activate with: conda activate physdreamer"
