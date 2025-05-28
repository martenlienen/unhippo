import logging
import math
import os
import subprocess
from pathlib import Path

import einops as eo
import lightning.pytorch as pl
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .atomic import open_atomic
from .sampler import InfiniteRandomSampler

log = logging.getLogger(__name__)


class FSDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        train_noise: float | None = None,
        eval_noises: float | None | list[float | None] = None,
        batch_size: int = 16,
        eval_batch_size: int = 16,
        on_device: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_noise = train_noise
        self.eval_noises = eval_noises
        if not isinstance(self.eval_noises, list):
            self.eval_noises = [self.eval_noises]

        self.data_dir = Path(root)
        self.decoded_path = self.data_dir / "decoded.pt"

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.on_device = on_device

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

        self.seed = 14207596599548160619

        if self.on_device:
            if torch.cuda.is_available():
                if torch.cuda.device_count() == 1:
                    self.device = torch.device("cuda")
                else:
                    log.warning(
                        "On-device caching requested but training uses multiple "
                        "GPUs, so caching on one of them does not work."
                    )
                    self.device = None
            else:
                log.warning("On-device caching requested but no GPUs are available.")
                self.device = None
        else:
            self.device = None

    def prepare_data(self):
        repo_path = self.data_dir / "repo"
        if not repo_path.exists() or not any(repo_path.iterdir()):
            repo_path.parent.mkdir(exist_ok=True, parents=True)
            subprocess.call(
                [
                    "git",
                    "clone",
                    "--branch",
                    "v1.0.10",
                    "--depth",
                    "1",
                    "https://github.com/Jakobovski/free-spoken-digit-dataset",
                    str(repo_path),
                ]
            )

        if not self.decoded_path.is_file():
            filenames = []
            data = []
            for path in tqdm(
                list(repo_path.glob("**/*.wav")), desc="Decoding .wav files"
            ):
                audio, sample_rate = torchaudio.load(str(path), normalize=True)
                assert sample_rate == 8_000

                # If is shorter than 1 second, loop it. This also cuts off 20 longer
                # recordings at 1 second.
                audio = audio.repeat(1, math.ceil(8_000 / audio.shape[1]))[:, :8_000]

                filenames.append(str(path.relative_to(repo_path)))
                data.append(audio)
            data = torch.stack(data)

            with open_atomic(self.decoded_path, "wb") as f:
                torch.save((filenames, data), f)
            del data

        noises = [self.train_noise, *self.eval_noises]
        for noise in noises:
            processed_dir = self._processed_dir(noise)
            if not processed_dir.is_dir() or not any(processed_dir.iterdir()):
                self._preprocess_noise(noise)

    def _processed_dir(self, noise: float | None):
        noise_tag = "clean" if noise is None else f"std{noise}"
        return self.data_dir / "processed" / noise_tag

    def _preprocess_noise(self, noise: float | None):
        log.info(f"Preprocessing std {noise}")

        filenames, data = torch.load(self.decoded_path)

        # Add noise
        if noise is not None:
            gen = torch.Generator().manual_seed(14601034571221182198 + int(noise * 10**8))
            epsilon = torch.normal(torch.zeros_like(data), std=noise, generator=gen)
            data = data + epsilon

        # Transpose into channels-last
        data = eo.rearrange(data, "b c t -> b t c")

        y = []
        people = []
        for i, f in enumerate(filenames):
            name, _ = os.path.splitext(os.path.basename(f))
            digit, person, rec_num = name.split("_")
            y.append(int(digit))
            people.append(person)
        y = torch.asarray(y, dtype=torch.long)

        # Split data into train/val/test
        f_train, f_eval, X_train, X_eval, y_train, y_eval, _, people_eval = (
            train_test_split(
                filenames,
                data,
                y,
                people,
                train_size=0.6,
                stratify=people,
                random_state=3497962377,
            )
        )
        f_val, f_test, X_val, X_test, y_val, y_test = train_test_split(
            f_eval,
            X_eval,
            y_eval,
            train_size=0.5,
            stratify=people_eval,
            random_state=1149005927,
        )

        # Normalize data
        mean = X_train.mean(dim=(0, 1))
        std = X_train.std(dim=(0, 1))

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        # Write data to disk
        processed_dir = self._processed_dir(noise)
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open_atomic(processed_dir / "train.pt", "wb") as f:
            torch.save((f_train, X_train, y_train), f)
        with open_atomic(processed_dir / "val.pt", "wb") as f:
            torch.save((f_val, X_val, y_val), f)
        with open_atomic(processed_dir / "test.pt", "wb") as f:
            torch.save((f_test, X_test, y_test), f)
        with open_atomic(processed_dir / "stats.pt", "wb") as f:
            torch.save((mean, std), f)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = self._dataset("train", self.train_noise)
        if stage in ("fit", "validate"):
            self.val_datasets = [
                self._dataset("val", noise) for noise in self.eval_noises
            ]
        if stage == "test":
            self.test_datasets = [
                self._dataset("test", noise) for noise in self.eval_noises
            ]

    def _dataset(self, phase: str, noise: float | None):
        data_dir = self._processed_dir(noise)
        _, X, y = torch.load(data_dir / f"{phase}.pt")
        return TensorDataset(
            torch.asarray(X, device=self.device), torch.asarray(y, device=self.device)
        )

    def _dataloader(self, dataset, *, batch_size: int, sampler=None):
        return DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)

    def train_dataloader(self):
        gen = torch.Generator().manual_seed(self.seed)
        return self._dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=InfiniteRandomSampler(self.train_dataset, generator=gen),
        )

    def val_dataloader(self):
        return [
            self._dataloader(dataset, batch_size=self.eval_batch_size)
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            self._dataloader(dataset, batch_size=self.eval_batch_size)
            for dataset in self.test_datasets
        ]

    @property
    def class_labels(self):
        return [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
