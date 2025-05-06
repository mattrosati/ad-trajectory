#!/bin/bash
#SBATCH --job-name=env_install                        # Job name
#SBATCH --output env_install.log                # Output log file
#SBATCH --mail-type=NONE                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<USER>@gmail.com               # Email address to send status updates to
#SBATCH --partition scavenge                            # Train on private partition
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=10gb                                 # Job memory request
#SBATCH --time=05:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

PREFIX="/home/mr2238/project/conda_envs/ad_traj"

module reset

module load miniconda
cd /home/mr2238/ad-trajectory/

# this is just brute force env install because getting conda to work with .yml is terrible
mamba create --prefix "$PREFIX" \
    python=3.10 \
    pip \
    numpy \
    pandas \
    matplotlib \
    tqdm
conda activate "$PREFIX"

# install additional packages
pip3 install torch torchvision torchaudio
pip install accelerate datasets transformers
pip install tabpfn
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
pip install wandb

conda deactivate

# to delete the environment, run:
# conda remove -n ad_traj --all