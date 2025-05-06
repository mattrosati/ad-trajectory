#!/bin/bash
#SBATCH --job-name=ad-traj                         # Job name
#SBATCH --output traj_%J.log                # Output log file
#SBATCH --mail-type=NONE                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<USER>@gmail.com               # Email address to send status updates to
#SBATCH --partition pi_dijk                            # Train on private partition
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16gb                                 # Job memory request
#SBATCH --time=01:00:00                          # Time limit hrs:min:sec

PREFIX="/home/mr2238/palmer_scratch/adni" # path to data directory
ENVIRONMENT="/home/mr2238/project/conda_envs/ad_traj" # path to conda environment

date;hostname;pwd

module reset

module load miniconda
conda activate "$ENVIRONMENT"

cd /home/mr2238/ad-trajectory/

# Extract and save embeddings from TabPFN
python src/embeds_main.py \
    --load-dir $PREFIX/processed/ \
    --save-dir $PREFIX/model_data/

conda deactivate