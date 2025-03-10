#!/bin/bash
#SBATCH --job-name=problem4
#SBATCH --output=out/problem4.out
#SBATCH --error=out/problem4.err
#SBATCH --partition=general
#SBATCH --time=1-12:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# Your job commands go here

# Load in cuda
source /etc/profile.d/modules.sh
module load cuda-12.4
nvcc --version
source ~/.bashrc

# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate minitorch
# Install pycuda
pip install pycuda

# compile_cuda.sh
mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
# verify installation
python3 -m install
# Problem 1
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC
python kernel_tests/test_softmax_fw.py