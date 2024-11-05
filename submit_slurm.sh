#!/bin/bash

#SBATCH --job-name="PL-ABFE-BRD4"
#SBATCH --partition=rtx3090
#SBATCH --propagate=NONE
#SBATCH --cpus-per-gpu=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH -q hca-csd765
#SBATCH --gpus 16
#SBATCH -A ddp325

Folder=$(pwd)
source ~/.bashrc
cd ${Folder}/

conda init
conda activate openff-evaluator-6

export LD_LIBRARY_PATH="$HOME/miniconda3/envs/openff-evaluator-6/lib:${LD_LIBRARY_PATH}"
export OPENMM_PLUGIN_DIR="$HOME/miniconda3/envs/openff-evaluator-6/lib/plugins"
export OE_LICENSE="$HOME/oe_license.txt"
export JAX_ENABLE_X64=True

cd /tscc/nfs/home/jta002/workspace/PL-ABFE/PL-ABFE-BRD4

echo "Starting job at $(date)"
ray start --head --num-gpus=16 --port=8835
srun ray start --address=$(hostname):8835
python main.py
echo "Finished job at $(date)"