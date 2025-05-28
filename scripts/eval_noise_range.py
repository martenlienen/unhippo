#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from unhippo.utils import print_config, set_seed

warnings.filterwarnings(
    "ignore",
    "The '\\w+_dataloader' does not have many workers",
    module="lightning",
)
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used",
    module="lightning",
)

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", default="test", help="Data split to evaluate")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint")
    parser.add_argument(
        "--stds",
        nargs="+",
        required=True,
        type=float,
        help="Eval noise stds the checkpoint will be evaluated on",
    )
    parser.add_argument("-o", "--out", help="Out directory")
    args = parser.parse_args()

    split = args.split
    ckpt_path = Path(args.checkpoint)
    noise_stds = np.asarray(args.stds)
    out_dir = Path(args.out)

    wandb_id = str(ckpt_path.parent.name)
    out_path = out_dir / f"{wandb_id}.json"

    if out_path.is_file():
        print(f"{out_path} exists, abort.")
        return

    ckpt = torch.load(ckpt_path, weights_only=False)
    log.info("Load config from checkpoint")
    config = OmegaConf.create(ckpt["config"])
    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)

    model = instantiate(config.model)
    task = instantiate(config.task, model)
    trainer = Trainer(**config.trainer, logger=False)

    noise_metrics = []
    for noise_std in noise_stds:
        set_seed(config)

        config.data.eval_noises = float(noise_std)
        datamodule = instantiate(config.data, _convert_="all")

        datamodule.prepare_data()
        if split == "test":
            datamodule.setup("test")
            dataloader = datamodule.test_dataloader()
        elif split == "val":
            datamodule.setup("validate")
            dataloader = datamodule.val_dataloader()
        else:
            log.error(f"Unknown data split {split}")
            sys.exit(1)

        metrics = trainer.test(model=task, ckpt_path=ckpt_path, dataloaders=dataloader)
        noise_metrics.append(metrics[0])

        # Write intermediate results on every iteration
        results = {
            "config": OmegaConf.to_container(config),
            "noise_stds": noise_stds.tolist(),
            "metrics": noise_metrics,
        }
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.open("w") as f:
            json.dump(results, f)
            # Force writing of data, so that it is saved to disk in case of an
            # mysterious exception which I have observed sporadically
            os.fsync(f)


if __name__ == "__main__":
    main()
