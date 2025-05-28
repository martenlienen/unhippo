import logging
import tarfile
import urllib.request
from pathlib import Path

import einops as eo
import lightning.pytorch as pl
import torch
import torchaudio
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .atomic import open_atomic
from .sampler import InfiniteRandomSampler

log = logging.getLogger(__name__)


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        all_classes: bool = True,
        train_noise: float | None = None,
        eval_noises: float | None | list[float | None] = None,
        batch_size: int = 16,
        eval_batch_size: int = 16,
        on_device: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.all_classes = all_classes
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

        self.seed = 7405844763673681755

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
        archive_path = self.data_dir / "speech_commands.tar.gz"
        if not archive_path.exists():
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading Speech Commands archive")
            urllib.request.urlretrieve(
                "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                archive_path,
            )

        raw_path = self.data_dir / "raw"
        if not raw_path.exists() or not any(raw_path.iterdir()):
            with tarfile.open(archive_path, "r:gz") as f:
                for member in tqdm(f.getmembers(), desc="Extracting data"):
                    # Check that we don't overwrite anything outside of the raw
                    # directory (path traversal attack)
                    if (
                        not (raw_path / member.name)
                        .resolve()
                        .is_relative_to(raw_path.resolve())
                    ):
                        log.error(
                            f"{member.name} is trying to write outside of the extraction directory. Attack?"
                        )
                        raise RuntimeError()

                    f.extract(member, raw_path)

        if not self.decoded_path.is_file():
            filenames = []
            data = []
            for path in tqdm(list(raw_path.glob("**/*.wav")), desc="Decoding .wav files"):
                audio, sample_rate = torchaudio.load(str(path), normalize=True)
                assert sample_rate == 16_000
                if audio.shape[1] != 16_000:
                    # About 10% of samples are shorter or longer and we just discard
                    # them
                    continue
                filenames.append(str(path.relative_to(raw_path)))
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
        n_classes = (
            len(self.ALL_CLASSES) if self.all_classes else len(self.SUBSET_CLASSES)
        )
        noise_tag = "clean" if noise is None else f"std{noise}"
        return self.data_dir / "processed" / f"{n_classes}-{noise_tag}"

    def _preprocess_noise(self, noise: float | None):
        log.info(f"Preprocessing std {noise}")

        filenames, data = torch.load(self.decoded_path)

        if not self.all_classes:
            selector = torch.asarray(
                [f.split("/")[0] in self.SUBSET_CLASSES for f in filenames],
                dtype=torch.bool,
            )
            filenames = [f for predicate, f in zip(selector, filenames) if predicate]
            data = data[selector]

        # Add noise
        if noise is not None:
            gen = torch.Generator().manual_seed(14601034571221182198 + int(noise * 10**8))
            epsilon = torch.normal(torch.zeros_like(data), std=noise, generator=gen)
            data = data + epsilon

        # Transpose into channels-last
        data = eo.rearrange(data, "b c t -> b t c")

        class_list = self.ALL_CLASSES if self.all_classes else self.SUBSET_CLASSES
        y = torch.tensor(
            [class_list.index(f.split("/")[0]) for f in filenames], dtype=torch.long
        )
        X = data

        # Split data into train/val/test
        val_files = set(
            (self.data_dir / "raw" / "validation_list.txt").read_text().splitlines()
        )
        test_files = set(
            (self.data_dir / "raw" / "testing_list.txt").read_text().splitlines()
        )

        val_mask = torch.tensor([f in val_files for f in filenames], dtype=torch.bool)
        test_mask = torch.tensor([f in test_files for f in filenames], dtype=torch.bool)
        train_mask = ~val_mask & ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        f_train = [f for i, f in enumerate(filenames) if train_mask[i]]
        f_val = [f for i, f in enumerate(filenames) if val_mask[i]]
        f_test = [f for i, f in enumerate(filenames) if test_mask[i]]

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
        return self.ALL_CLASSES if self.all_classes else self.SUBSET_CLASSES

    SUBSET_CLASSES = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]
    ALL_CLASSES = [
        "bed",
        "cat",
        "down",
        "five",
        "forward",
        "go",
        "house",
        "left",
        "marvin",
        "no",
        "on",
        "right",
        "sheila",
        "tree",
        "up",
        "visual",
        "yes",
        "backward",
        "bird",
        "dog",
        "eight",
        "follow",
        "four",
        "happy",
        "learn",
        "nine",
        "off",
        "one",
        "seven",
        "six",
        "stop",
        "three",
        "two",
        "wow",
        "zero",
    ]
