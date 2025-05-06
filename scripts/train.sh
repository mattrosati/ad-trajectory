#!/bin/bash
#SBATCH --job-name=ad-traj-array
#SBATCH --output=logs/traj_%A_%a.log
#SBATCH --array=0-2                        # Update this range to match the number of configs
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --constraint=a5000|rtx3090
#SBATCH --mem=16gb
#SBATCH --time=1-00:00:00

# Activate environment
ENVIRONMENT="/home/mr2238/project/conda_envs/ad_traj"
CONFIG_LIST="/home/mr2238/ad-trajectory/configs/config_list.txt"

module reset
module load miniconda
conda activate "$ENVIRONMENT"

cd /home/mr2238/ad-trajectory/

# Select config based on array ID
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")

echo "Running config: $CONFIG"
python src/train.py "$CONFIG"

conda deactivate