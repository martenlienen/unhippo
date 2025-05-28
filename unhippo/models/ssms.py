from functools import partial
from typing import Literal

from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn

from unhippo.nn import KwargsSequential

from .layers import ExactUnLSSLLayer, LSSLLayer, UnLSSLLayer


class SSM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        layers: list[nn.Module],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(8, self.hidden_channels)
        # GroupNorm expects channels at index 1
        self.time_norm = nn.Sequential(
            Rearrange("time batch hidden_channels -> time hidden_channels batch"),
            self.norm,
            Rearrange("time hidden_channels batch -> time batch hidden_channels"),
        )

        self.encoder = KwargsSequential(
            nn.Linear(in_channels, self.hidden_channels),
            Rearrange("batch time hidden_channels -> time batch hidden_channels"),
        )
        self.decoder = KwargsSequential(
            Rearrange("time batch hidden_channels -> batch time hidden_channels"),
            nn.Linear(self.hidden_channels, out_channels),
        )

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: Float[Tensor, "batch time in_channels"]
    ) -> Float[Tensor, "batch time out_channels"]:
        x = self.encoder(x)
        for layer in self.layers:
            x = x + self.dropout(layer(self.time_norm(x)))
        return self.decoder(x)


class ExactUnLSSL(SSM):
    """UnLSSL variant with the exact, time-varying dynamics."""

    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n: int,
        max_steps: int,
        n_latent_channels: int = 1,
        obs_sigma2: float = 1.0,
        trans_sigma2: float = 1.0,
        symmetrize: bool = True,
        joseph: bool = True,
        discretization_method: Literal[
            "expm", "bilinear", "forward", "backward"
        ] = "expm",
        dropout: float = 0.0,
        **kwargs,
    ):
        Layer = partial(
            ExactUnLSSLLayer,
            n=n,
            n_channels=hidden_channels,
            n_latent_channels=n_latent_channels,
            max_steps=max_steps,
            obs_sigma2=obs_sigma2,
            trans_sigma2=trans_sigma2,
            symmetrize=symmetrize,
            joseph=joseph,
            discretization_method=discretization_method,
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            layers=[Layer() for _ in range(n_layers)],
            dropout=dropout,
        )


class UnLSSL(SSM):
    """UnLSSL with learnable, time-invariant dynamics."""

    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n: int,
        trainable: bool,
        n_latent_channels: int = 1,
        obs_sigma2: float = 1.0,
        trans_sigma2: float = 1.0,
        symmetrize: bool = True,
        joseph: bool = True,
        krylov: bool = True,
        min_t: float = 10,
        max_t: float = 1000,
        discretization_method: Literal[
            "expm", "bilinear", "forward", "backward"
        ] = "expm",
        dropout: float = 0.0,
        **kwargs,
    ):
        Layer = partial(
            UnLSSLLayer,
            n=n,
            n_channels=hidden_channels,
            n_latent_channels=n_latent_channels,
            trainable=trainable,
            obs_sigma2=obs_sigma2,
            trans_sigma2=trans_sigma2,
            symmetrize=symmetrize,
            joseph=joseph,
            krylov=krylov,
            min_t=min_t,
            max_t=max_t,
            discretization_method=discretization_method,
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            layers=[Layer() for _ in range(n_layers)],
            dropout=dropout,
        )


class LSSL(SSM):
    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n: int,
        trainable: bool,
        n_latent_channels: int = 1,
        krylov: bool = True,
        min_t: float = 10,
        max_t: float = 1000,
        discretization_method: Literal[
            "expm", "bilinear", "forward", "backward"
        ] = "expm",
        dropout: float = 0.0,
        **kwargs,
    ):
        Layer = partial(
            LSSLLayer,
            n=n,
            n_channels=hidden_channels,
            n_latent_channels=n_latent_channels,
            trainable=trainable,
            krylov=krylov,
            min_t=min_t,
            max_t=max_t,
            discretization_method=discretization_method,
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            layers=[Layer() for _ in range(n_layers)],
            dropout=dropout,
        )
