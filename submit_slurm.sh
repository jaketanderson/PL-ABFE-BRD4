#!/bin/bash

#SBATCH --job-name="PL-ABFE-BRD4"
#SBATCH --partition=rtx3090
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-00:10:00
#SBATCH -q hca-csd765
#SBATCH --gpus=8
#SBATCH --nodes=1-8
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

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # Get the first node as the head
HEAD_PORT=8836  # Port for Ray head node

if [[ "$SLURM_NODEID" == "0" ]]; then
    # Start Ray head node
    echo "Starting Ray head on $HEAD_NODE"
    ray start --head --port=$HEAD_PORT

    # Check every 10 seconds for connected workers
    for i in {1..18}; do
        sleep 10
        worker_count=$(ray status | grep -c "node")
        if [[ $worker_count -ge 8 ]]; then
            break
        fi
    done

    python main.py

    ray stop
else
    # Start Ray worker nodes
    echo "Starting Ray worker on $SLURM_NODEID, connecting to head node at $HEAD_NODE:$HEAD_PORT"
    ray start --address="$HEAD_NODE:$HEAD_PORT"
fi
