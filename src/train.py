import argparse
import json
import wandb
from transformers import TrainingArguments
from models.gp_vae import GPVAE
from trainer_gpvae import GPVAETrainer
from dataset import GPVAEDataset

def main():
    # --- Parse CLI args ---
    parser = argparse.ArgumentParser(description="Train a GP-VAE model with Hugging Face Trainer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON file")
    parser.add_argument("--data", type=str, default="data/example_dataset.npy", help="Path to .npy dataset file")
    parser.add_argument("--project", type=str, default="gpvae-project", help="wandb project name")
    parser.add_argument("--run_name", type=str, default="gpvae-run", help="wandb run name")
    args = parser.parse_args()


    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # W&B init
    wandb.init(project="gpvae-project", config=config)

    # Dataset
    dataset = GPVAEDataset("data/example_dataset.npy")
    input_dim = dataset[0]["x"].shape[-1]

    # Model
    model = GPVAE(input_dim, config["hidden_dim"], config["latent_dim"])

    # HF TrainingArguments
    args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        logging_dir="./logs",
        logging_steps=config["logging_steps"],
        report_to=["wandb"],
    )

    # Trainer
    trainer = GPVAETrainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()