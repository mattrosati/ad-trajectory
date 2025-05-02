#!/bin/bash
#SBATCH --job-name=env_install                        # Job name
#SBATCH --output env_install.log                # Output log file
#SBATCH --mail-type=NONE                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<USER>@gmail.com               # Email address to send status updates to
#SBATCH --partition day                            # Train on private partition
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=5
#SBATCH --mem=10gb                                 # Job memory request
#SBATCH --time=05:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
cd /home/mr2238/ad-trajectory/

# Install the required packages
mamba env create -f ~/ad-trajectory/environment.yml

conda activate ad_traj
bash ./scripts/clean.sh
conda deactivate