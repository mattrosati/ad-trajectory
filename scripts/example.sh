#!/bin/bash
#SBATCH --job-name=brainlm                         # Job name
#SBATCH --output log_brainlm_%J.log                # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<USER>@gmail.com               # Email address to send status updates to
#SBATCH --partition gpu                            # Train on private partition
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --constraint=a5000|rtx3090
#SBATCH --cpus-per-task=3
#SBATCH --mem=32gb                                 # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
conda activate /home/ahf38/.conda/envs/brain_lm_flash_attn_2
cd /home/sr2464/Desktop/BrainLM/


# Training from scratch
python train.py \
    --output_dir /home/sr2464/palmer_scratch/brainlm/training-runs/test_run \
    --model_name_or_path facebook/vit-mae-base \
    --train_dataset_path /home/sr2464/palmer_scratch/datasets/UKB_Large_rsfMRI_and_tffMRI_Arrow_WithRegression_v3_with_metadata/train_ukbiobank \
    --val_dataset_path /home/sr2464/palmer_scratch/datasets/UKB_Large_rsfMRI_and_tffMRI_Arrow_WithRegression_v3_with_metadata/val_ukbiobank \
    --coords_dataset_path /home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4_no_metadata/Brain_Region_Coordinates \
    --recording_col_name Voxelwise_RobustScaler_Normalized_Recording \
    --num_timepoints_per_voxel 200 \
    --timepoint_patching_size 20 \
    --decoder_hidden_size 768 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 20 \
    --logging_strategy steps \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 5 \
    --dataloader_num_workers 5 \
    --dataloader_pin_memory True \
    --max_eval_samples 800 \
    --mask_ratio 0.75 \
    --wandb_logging True \
    --wandb_path /home/sr2464/palmer_scratch/BrainLM_wandb \
    --loss_fn mse