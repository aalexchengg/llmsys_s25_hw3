#!/bin/bash
#SBATCH --job-name=final
#SBATCH --output=out/final.out
#SBATCH --error=out/final.err
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=32G

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
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC
# Problem 1
python kernel_tests/test_softmax_fw.py
echo "Finished softmax fw test"
# Problem 2
python kernel_tests/test_softmax_bw.py
echo "Finished softmax bw test"
# Problem 3
python kernel_tests/test_layernorm_fw.py
echo "Finished layernorm fw test"
# Problem 4
python kernel_tests/test_layernorm_bw.py
echo "Finished layernorm bw test"
# Problem 5
# echo "Running without fused kernel..."
# python project/run_machine_translation.py --use-fused-kernel False
# echo "Running with fused kernel..."
# python project/run_machine_translation.py --use-fused-kernel True
# echo "All finished."
