#!/usr/bin/env python

import faulthandler
import logging
import math
import os
import socket
import warnings

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf, open_dict

from unhippo.utils import (
    filter_device_available,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)
from unhippo.utils.callbacks import ConfigInCheckpoint

# Log traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# If data loading is really not a bottleneck for you, uncomment this to silence the
# warning about it
warnings.filterwarnings(
    "ignore",
    r"The '\w+_dataloader' does not have many workers",
    module="lightning",
)
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used",
    module="lightning",
)
logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
    filter_device_available
)


def if_eq(a, b, then, otherwise):
    """A conditional for OmegaConf interpolations."""
    if a == b:
        return then
    else:
        return otherwise


def resolve_eval(expr):
    """Resolve an arbitrary expression in OmegaConf interpolations."""
    # We trust our own configuration not to delete all our files, so just eval the
    # expression
    return eval(expr, {}, {"math": math})


OmegaConf.register_new_resolver("if_eq", if_eq)
OmegaConf.register_new_resolver("eval", resolve_eval)


log = get_logger()


def store_job_info(config: DictConfig):
    host = socket.gethostname()
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    process_id = os.getpid()

    with open_dict(config):
        config.host = host
        config.process_id = process_id
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id


def get_callbacks(config, logger):
    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/{logger.name}/{logger.version}",
            save_top_k=-1,
            # This ensures that a checkpoint is saved after every validation
            every_n_epochs=1,
        ),
        TQDMProgressBar(),
        LearningRateMonitor(),
        ConfigInCheckpoint(config),
    ]
    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    set_seed(config)

    # Log host and slurm job ID
    store_job_info(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)

    log.info("Loading data")
    datamodule = instantiate(config.data)

    log.info("Instantiating model")
    model = instantiate(config.model)

    log.info("Instantiating task")
    task = instantiate(config.task, model)

    logger = instantiate(
        config.wandb,
        _target_="lightning.pytorch.loggers.WandbLogger",
        resume=(config.wandb.mode == "online") and "allow",
        # Don't upload any checkpoints to save space
        log_model=False,
        # Use trainer's default_root_dir
        save_dir=None,
    )

    log_hyperparameters(logger, config, model)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config, logger)
    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=f"checkpoints/{logger.name}/{logger.version}",
    )

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
