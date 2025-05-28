import math

import torch
from torch.utils.data import Sampler


class InfiniteRandomSampler(Sampler):
    """Yields an endless stream of random samples from a data source."""

    def __init__(self, data_source, *, generator: torch.Generator | None = None):
        super().__init__()

        self.data_source = data_source
        self.generator = generator

    def __len__(self):
        return math.inf

    def __iter__(self):
        n = len(self.data_source)
        while True:
            yield from torch.randperm(n, generator=self.generator)
