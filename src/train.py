import logging
import os
import sys
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from random import randint
import json
import numpy as np

import torch
import wandb
from datasets import load_from_disk, DatasetDict

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch.utils.data import DataLoader
from transformers import Trainer

from models.gpvae import GPVAEModel, GPVAEConfig
from trainer import GPVAEWandbCallback, GPVAETrainer

from data_constants import *

# os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")
require_version("datasets>=1.8.0", "To fix: conda env create -f environment.yml")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_path: str = field(
        default="./data",
        metadata={
            "help": "Path to saved dataset dict of tabPFN embeddings."
        }
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        self.data_files = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization .Don't set if you want to train a model from scratch."
            )
        },
    )
    model_output_dir: str = field(
        default="./models-checkpoints",
        metadata={
            "help": (
                "Where to save the models."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    latent_size: int = field(default=128, metadata={"help": "Encoder hidden size."})
    num_hidden_layers: int = field(default=2, metadata={"help": "Encoder num layers."})
    intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in encoder layers."}
    )
    decoder_hidden_size: int = field(
        default=128, metadata={"help": "Decoder hidden size."}
    )
    decoder_num_hidden_layers: int = field(
        default=10, metadata={"help": "Decoder num layers."}
    )
    decoder_intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in decoder layers."}
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for layer activations."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./results", metadata={"help": "Directory to store models and predictions."}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Don't remove unused columns."}
    )
    do_train: int = field(default=True, metadata={"help": "Whether to do training."})
    do_eval: int = field(default=True, metadata={"help": "Whether to do eval."})
    base_learning_rate: float = field(
        default=1e-3,
        metadata={
            "help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."
        },
    )
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={"help": "What learning rate scheduler to use."},
    )
    sigma: float = field(
        default=1.005,
        metadata={
            "help": "Sigma value for the GP prior."
        },
    )
    length_scale: float = field(
        default=7.0,
        metadata={
            "help": "Length scale value for the GP prior."
        },
    )
    beta: float = field(
        default=0.2,
        metadata={
            "help": "Factor to weigh the KL term (similar to beta-VAE)."
        },
    )
    weight_decay: float = field(
        default=0.05,
        metadata={
            "help": "Weight decay (L2 regularization coefficient) for optimizer."
        },
    )
    num_train_epochs: int = field(
        default=100, metadata={"help": "Number of epochs to train for."}
    )
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "Warmup ratio for learning rate scheduler."}
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for each device used during training."},
    )
    per_device_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for each device used during evaluation."},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={
            "help": "How often to log training metrics. If choose 'steps', specify logging_steps."
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={
            "help": "If logging_strategy is 'steps', log training metrics every X iterations."
        },
    )
    eval_strategy: str = field(
        default="steps", metadata={"help": "How often to log eval results."}
    )
    eval_steps: int = field(
        default=10,
        metadata={
            "help": "If evaluation_strategy is 'steps', calculate validation metrics every X iterations."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "How often to save results and models."}
    )
    save_steps: int = field(
        default=10,
        metadata={
            "help": "If save_strategy is 'steps', save model checkpoint every X iterations."
        },
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "At the end, load the best model."}
    )
    save_total_limit: int = field(
        default=5, metadata={"help": "Maximum number of models to save."}
    )
    seed: int = field(default=42, metadata={"help": "Random seed."})
    wandb_logging: bool = field(
        default=False,
        metadata={
            "help": "Whether to log metrics to weights & biases during training."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={
            "help": "Trainer will include model inputs in call to metrics calculation function. Depends on 'input_ids' being one of the input parameters to model, comes from tokenizer used? Currently incompatible with single-cell dataloader, leave as False."
        },
    )
    use_tanh_decoder: bool = field(
        default=False,
        metadata={
            "help": "If we want to use TanH as the nonlinearity for the output layer."
        },
    )
    num_labels: int = field(
        default=len(targets), metadata={"help": "Number of labels to predict."}
    )

def collate_fn(examples):
    # Collates everything ready for model.
    batch = {
        "inputs": [],
        "labels": [],
        "times": [],
    }

    for e in examples:
        row = []
        row_label = []
        row_time = []
        for field in e.keys():

            if "f_" in field:
                row.append(e[field])
            elif field in targets:
                row_label.append(e[field])
            elif field == "months":
                row_time.append(e[field])

        batch["inputs"].append(row)
        batch["labels"].append(row_label)
        batch["times"].append(row_time)

    for k, v in batch.items():
        batch[k] = (torch.tensor(v)).float()

    return batch

def main():
    # See all possible arguments by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        with open(os.path.abspath(sys.argv[1]), "r") as f:
            c = json.load(f)
        model_args, data_args, training_args = parser.parse_dict(c)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(model_args.model_output_dir)
        if last_checkpoint is None and len(os.listdir(model_args.model_output_dir)) > 0:
            raise ValueError(
                f"Output directory ({tmodel_args.model_output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # --- Initialize Weights & Biases Logging ---#
    date_time_stamp = training_args.output_dir.split("/")[-1]

    if training_args.wandb_logging:
        # NOTE: Change project name to your own project name in your weights & biases account
        name = data_args.dataset_path.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = name + "-" + timestamp
        wandb.init(
            project="tabpfn-gpvae", name=name, config = c
        )
    
    # --- Initialize Dataset ---#
    # Load arrow datasets
    ds = load_from_disk(data_args.dataset_path)
    training_args.input_dim = ds["train"].shape[-1] - training_args.num_labels - 3 # removing labels and index+PTID and month
    print(training_args.input_dim)

    # Load gene information dataset (containing gene names, expression mean and std dev)
    # coords_ds = load_from_disk(data_args.coords_dataset_path)

    
    # --- Initialize Model ---#
    # Load model
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = GPVAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = GPVAEConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = GPVAEConfig(
            # Model-related
            latent_size=model_args.latent_size,
            num_hidden_layers=model_args.num_hidden_layers,
            intermediate_size=model_args.intermediate_size,
            decoder_hidden_size=model_args.decoder_hidden_size,
            decoder_num_hidden_layers=model_args.decoder_num_hidden_layers,
            decoder_intermediate_size=model_args.decoder_intermediate_size,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            use_tanh_decoder=training_args.use_tanh_decoder,
            num_labels=training_args.num_labels,
            input_dim=training_args.input_dim,
            sigma = training_args.sigma,
            length_scale = training_args.length_scale,
            beta = training_args.beta,
        )

        logger.warning("You are instantiating a new config instance from scratch.")
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

    # create model
    if model_args.model_name_or_path:
        model = GPVAEModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = GPVAEModel(config)

    if training_args.wandb_logging:
        training_args.report_to = "wandb"

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = (
                ds["train"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )
        # Set the training transforms

    if training_args.do_eval:
        if "val" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["val"] = (
                ds["val"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = (
            training_args.base_learning_rate * total_train_batch_size / 256
        )

    # Initialize our trainer
    print(type(model))
    trainer = GPVAETrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["val"] if training_args.do_eval else None,
        data_collator=collate_fn,
    )


    batch = next(iter(trainer.get_train_dataloader()))

    print("\nBatch shapes:")
    for k, v in batch.items():
        print(f"{k}: {v.shape}")

    print('len(ds["train"]): ',len(ds["train"])) #800
    print('len(ds["val"]):', len(ds["val"])) #100
    print('len(ds["test"]):', len(ds["test"])) #100



    for b in iter(trainer.get_train_dataloader()):
        inputs = b["inputs"]
        times = b["times"]     # [B, T]
        labels = b["labels"]

        # Find indices where time is very close to 0
        # also find MCI patient
        m1 = times <= 1e-5
        m2 = labels[:, -1] == 2
        print(m1.shape, m2.shape)
        m = m1.squeeze(1).bool() & m2
        print(m.shape)

        # Extract time-zero embeddings per sample (only one time==0 allowed per sample)
        masked = inputs[m]
        if masked.shape[0] > 0:
            some_input_embedding = masked[0, :].detach().clone()
            break

    callback = GPVAEWandbCallback(
        x_cond=some_input_embedding,
        t_cond=0.0,
        t_query=torch.linspace(0, 60, 150).to(training_args.device),
        target_idx=4,
        log_every_n_steps=training_args.logging_steps,
    )

    trainer.add_callback(callback)


    # Training - uncomment once init of everything is working
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions = trainer.predict(test_dataset)
        trainer.log_metrics("test", predictions.metrics)
        trainer.save_metrics("test", predictions.metrics)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()