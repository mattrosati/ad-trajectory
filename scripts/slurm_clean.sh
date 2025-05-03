#!/bin/bash
#SBATCH --job-name=data_clean                        # Job name
#SBATCH --output data_clean.log                # Output log file
#SBATCH --mail-type=NONE                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<USER>@gmail.com               # Email address to send status updates to
#SBATCH --partition day                            # Train on private partition
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=10gb                                 # Job memory request
#SBATCH --time=05:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

PREFIX="/home/mr2238/palmer_scratch/adni" # path to data directory
ENVIRONMENT="/home/mr2238/project/conda_envs/ad_traj" # path to conda environment

module load miniconda
conda activate "$ENVIRONMENT"

cd /home/mr2238/ad-trajectory/

# Clean raw data
bash ./scripts/clean.sh "$PREFIX"

conda deactivate