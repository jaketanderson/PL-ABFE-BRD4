#!/bin/bash

#SBATCH --job-name="PL-ABFE-BRD4"
#SBATCH --partition=168h
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=2-00:00:00
#SBATCH -q hca-csd765
#SBATCH --gpus-per-task=8
#SBATCH --nodes=7
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

cd /home/jta002/workspace/PL-ABFE/PL-ABFE-BRD4

echo "Starting job at $(date)"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=5379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
echo $SLURM_GPUS_PER_TASK
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

python -u main.py
