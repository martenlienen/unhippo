import math
from contextlib import contextmanager
from typing import Literal

import einops as eo
import numpy as np
import torch
from einops.layers.torch import Rearrange
from jaxtyping import Float
from numpy.polynomial import Legendre
from torch import Tensor, nn

from .krylov_conv import causal_convolution, krylov


@contextmanager
def default_dtype(dtype: torch.dtype):
    """Temporarily set the default dtype for torch operations."""
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def kalman_filter_parameters(
    n: int,
    steps: int,
    A: Float[Tensor, "k n n"],
    B: Float[Tensor, "n"],
    sigma2: float,
    Sigma: Float[Tensor, "n n"],
    *,
    symmetrize: bool = True,
    joseph: bool = True,
) -> tuple[Float[Tensor, "k n n"], Float[Tensor, "k n"]]:
    """Compute the parameters for a Kalman filter with scalar observations.

    Args:
        n: State dimension
        steps: Maximum number of steps
        A: Linear dynamics operator
        B: Observation operator
        sigma2: Observation variance
        Sigma: Dynamics noise covariance
        symmetrize: Enforce symmetry of P
        joseph: Use numerically more stable Joseph formulation for P

    Returns:
        P: Posterior covariances
        K: Kalman gain
    """

    Id = torch.eye(n, dtype=A.dtype, device=A.device)
    P_0 = Id
    P = [P_0]
    K = []
    for k in range(steps):
        A_km1 = A[k]
        P_minus_k = A_km1 @ P[-1] @ A_km1.T + Sigma
        if symmetrize:
            P_minus_k = 0.5 * (P_minus_k + P_minus_k.T)
        s_k = B @ P_minus_k @ B + sigma2
        K_k = P_minus_k @ B / s_k
        if joseph:
            L = Id - torch.outer(K_k, B)
            P_k = L @ P_minus_k @ L.T + sigma2 * torch.outer(K_k, K_k)
        else:
            P_k = P_minus_k - s_k * torch.outer(K_k, K_k)
        P.append(P_k)
        K.append(K_k)
    P = torch.stack(P[1:])
    K = torch.stack(K)

    return P, K


def hippo_matrix(n: int) -> Float[Tensor, "n n"]:
    """Compute the original HiPPO matrix."""
    seq = torch.sqrt(2 * torch.arange(n) + 1)
    return torch.tril(torch.einsum("i, j -> ij", seq, seq)) - torch.diag(torch.arange(n))


def unregularized_hippo_matrix(n: int) -> Float[Tensor, "n n"]:
    """Compute the HiPPO matrix for the data-free dynamics."""
    return hippo_matrix(n).T - torch.eye(n)


@default_dtype(torch.double)
def regularized_hippo_matrix(n: int) -> Float[Tensor, "n n"]:
    """
    Compute the HiPPO matrix for the data-free dynamics with derivative regularization.

    This matrix needs to computed in double precision.
    """
    i = torch.arange(n)
    A_H = hippo_matrix(n)
    B_H = torch.sqrt(2 * i + 1)
    Q = torch.sqrt(2 * i + 1) * i * (i + 1) / 2
    lhs = torch.cat((torch.eye(n), B_H[None], Q[None]), dim=0)
    rhs = torch.cat((A_H.T - torch.eye(n), 2 * Q[None], Q[None]), dim=0)

    A_U = torch.linalg.lstsq(lhs, rhs).solution
    return A_U


def discretize_unhippo_dynamics(
    A: Float[Tensor, "n n"],
    t0: Float[Tensor, "k"],
    t1: Float[Tensor, "k"],
    method: Literal["expm", "bilinear", "bilinear-lssl", "forward", "backward"],
) -> Float[Tensor, "k n n"]:
    """Discretize the uncertainty-aware HiPPO dynamics dc/dt = 1/t Ac."""

    if method == "expm":
        # Closed-form solution
        return torch.linalg.matrix_exp(torch.log(t1 / t0)[:, None, None] * A)

    Id = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    dt = t1 - t0
    if method == "bilinear":
        inv = torch.linalg.inv(Id + (dt / (2 * t1))[:, None, None] * A)
        return inv @ (Id - (dt / (2 * t0))[:, None, None] * A)
    elif method == "bilinear-lssl":
        # Bilinear discretization as in the LSSL paper, i.e. with t1 used in both places
        # instead of t1 and t0.
        inv = torch.linalg.inv(Id + (dt / (2 * t1))[:, None, None] * A)
        return inv @ (Id - (dt / (2 * t1))[:, None, None] * A)
    elif method == "forward":
        return Id + (dt / t0)[:, None, None] * A
    elif method == "backward":
        # Implicit Euler discretization
        return torch.linalg.inv(Id - (dt / t1)[:, None, None] * A)
    else:
        raise RuntimeError(f"Unknown discretization method '{method}'")


def discretize_hippo_dynamics(
    A: Float[Tensor, "n n"],
    B: Float[Tensor, "n"],
    t0: Float[Tensor, "k"],
    t1: Float[Tensor, "k"],
    method: Literal["bilinear", "bilinear-lssl", "forward", "backward"],
) -> tuple[Float[Tensor, "k n n"], Float[Tensor, "k n"]]:
    """Discretize the HiPPO dynamics dc/dt = -1/t Ac + 1/t B."""
    dt = t1 - t0
    Id = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    if method == "bilinear":
        # Bilinear discretization
        inv = torch.linalg.inv(Id + (dt / (2 * t1))[:, None, None] * A)
        A_bar = inv @ (Id - (dt / (2 * t0))[:, None, None] * A)
        B_bar = (inv @ ((dt / t1)[:, None] * B)[..., None])[..., 0]
        return A_bar, B_bar
    if method == "bilinear-lssl":
        # Bilinear discretization exactly as in the LSSL paper, i.e. with t1 used in
        # both places instead of t1 and t0.
        inv = torch.linalg.inv(Id + (dt / (2 * t1))[:, None, None] * A)
        A_bar = inv @ (Id - (dt / (2 * t1))[:, None, None] * A)
        B_bar = (inv @ ((dt / t1)[:, None] * B)[..., None])[..., 0]
        return A_bar, B_bar
    elif method == "forward":
        return Id - (dt / t0)[:, None, None] * A, (dt / t0)[:, None] * B
    elif method == "backward":
        # Implicit Euler discretization
        inv = torch.linalg.inv(Id + (dt / t1)[:, None, None] * A)
        return inv, (inv @ ((dt / t1)[:, None] * B)[..., None])[..., 0]
    else:
        raise RuntimeError(f"Unknown discretization method '{method}'")


def reconstruct_legendre(coef: Float[Tensor, "n"], *, domain=None) -> Legendre:
    """Construct a Legendre polynomial from HiPPO coefficients."""
    B = torch.sqrt(2 * torch.arange(coef.shape[-1]) + 1)
    return Legendre((coef * B).numpy(force=True), domain=domain)


def unroll_time_variant_dynamics(
    x: Float[Tensor, "time batch channel"],
    A_U: Float[Tensor, "time n n"],
    K: Float[Tensor, "time n"],
) -> Float[Tensor, "time batch channel n"]:
    m = [x.new_zeros(*x.shape[1:], K.shape[-1])]
    Kx = torch.einsum("tn, tbf -> tbfn", K, x)
    for k in range(len(x)):
        m.append(torch.einsum("ij, bfj -> bfi", A_U[k], m[-1]) + Kx[k])
    return torch.stack(m[1:])


def unroll_time_invariant_dynamics(
    x: Float[Tensor, "time batch channel"],
    A_U: Float[Tensor, "channel n n"],
    K: Float[Tensor, "channel n"],
) -> Float[Tensor, "time batch channel n"]:
    m = [x.new_zeros(*x.shape[1:], K.shape[-1])]
    Kx = torch.einsum("fn, tbf -> tbfn", K, x)
    for k in range(len(x)):
        m.append(torch.einsum("fij, bfj -> bfi", A_U, m[-1]) + Kx[k])
    return torch.stack(m[1:])


class ExactUnLSSLLayer(nn.Module):
    """Uncertainty-aware LSSL layer with time-varying dynamics."""

    def __init__(
        self,
        n: int,
        max_steps: int,
        n_channels: int,
        n_latent_channels: int,
        *,
        obs_sigma2: float = 1.0,
        trans_sigma2: float = 1.0,
        symmetrize: bool = True,
        joseph: bool = True,
        discretization_method: Literal[
            "expm", "bilinear", "forward", "backward"
        ] = "expm",
    ):
        super().__init__()

        self.n = n
        self.max_steps = max_steps
        self.n_channels = n_channels
        self.n_latent_channels = n_latent_channels
        self.obs_sigma2 = obs_sigma2
        self.trans_sigma2 = trans_sigma2
        self.symmetrize = symmetrize
        self.joseph = joseph
        self.discretization_method = discretization_method

        A_R = regularized_hippo_matrix(n)
        if torch.cuda.is_available():
            A_R = A_R.to(device="cuda")

        k = torch.arange(self.max_steps, device=A_R.device)
        # Don't divide by zero in the first step, which is a special case anyway
        k[0] = 1
        A = discretize_unhippo_dynamics(
            A_R, k, k + 1, method=self.discretization_method
        ).float()
        B = torch.sqrt(2 * torch.arange(n, device=A.device) + 1)

        Id = torch.eye(n, device=A.device)
        P, K = kalman_filter_parameters(
            self.n,
            self.max_steps,
            A,
            B,
            sigma2=self.obs_sigma2,
            Sigma=self.trans_sigma2 * Id,
            symmetrize=self.symmetrize,
            joseph=self.joseph,
        )
        self.register_buffer(
            "A_U", (Id - torch.einsum("ki, j -> kij", K, B)) @ A, persistent=False
        )
        self.register_buffer("K", K, persistent=False)

        self.C = nn.Parameter(
            torch.empty((self.n_channels, self.n_latent_channels, self.n))
        )
        # Transpose C so that fan-in and fan-out are correctly computed
        nn.init.kaiming_normal_(eo.rearrange(self.C, "c m n -> n m c"), a=math.sqrt(5))
        self.D = nn.Parameter(torch.empty((self.n_channels, self.n_latent_channels)))
        nn.init.normal_(self.D)

        self.position_wise = nn.Sequential(
            nn.GELU(),
            Rearrange("time batch channel latent -> time batch (channel latent)"),
            nn.Linear(self.n_channels * self.n_latent_channels, self.n_channels),
        )

    def forward(
        self, x: Float[Tensor, "time batch channel"]
    ) -> Float[Tensor, "time batch channel"]:
        t = len(x)
        c = unroll_time_variant_dynamics(x, self.A_U[:t], self.K[:t])
        lssl_out = torch.einsum("cmn, tbcn -> tbcm", self.C, c) + torch.einsum(
            "cm, tbc -> tbcm", self.D, x
        )
        return self.position_wise(lssl_out)


class LSSLLikeLayer(nn.Module):
    """An LSSL-like layer with configurable A and B matrices.

    Applies a time-invariant dynamics via convolution with Krylov kernel. Optionally
    trainable.
    """

    def __init__(
        self,
        n: int,
        n_channels: int,
        n_latent_channels: int,
        A: Float[Tensor, "c n n"],
        B: Float[Tensor, "c n"],
        *,
        trainable: bool,
        krylov: bool = True,
    ):
        """
        Args:
            n: Number of coefficients (N in LSSL)
            n_channels: Number of channels/features in the data (H in LSSL)
            n_latent_channels: Number of latent channels (M in LSSL)
        """

        super().__init__()

        self.n = n
        self.n_channels = n_channels
        self.n_latent_channels = n_latent_channels
        self.trainable = trainable
        self.krylov = krylov

        # Krylov matrix is unstable in single precision
        A = A.double()
        B = B.double()

        if self.trainable:
            self.register_parameter("A", nn.Parameter(A))
            self.register_parameter("B", nn.Parameter(B))

            self.A.ssm_matrix_lr = True
            self.B.ssm_matrix_lr = True
        else:
            self.register_buffer("A", A, persistent=False)
            self.register_buffer("B", B, persistent=False)

        self.register_buffer("krylov_matrix", None, persistent=False)

        self.C = nn.Parameter(
            torch.empty((self.n_channels, self.n_latent_channels, self.n))
        )
        # Transpose C so that fan-in and fan-out are correctly computed
        nn.init.kaiming_normal_(eo.rearrange(self.C, "c m n -> n m c"), a=math.sqrt(5))
        self.D = nn.Parameter(torch.empty((self.n_channels, self.n_latent_channels)))
        nn.init.normal_(self.D)

        self.position_wise = nn.Sequential(
            nn.GELU(),
            Rearrange("time batch channel latent -> time batch (channel latent)"),
            nn.Linear(self.n_channels * self.n_latent_channels, self.n_channels),
        )

    def forward(
        self, x: Float[Tensor, "time batch channel"]
    ) -> Float[Tensor, "time batch channel"]:
        if self.krylov:
            T = len(x)
            krylov_matrix = self.krylov_matrix
            if krylov_matrix is None or krylov_matrix.shape[-1] < T:
                krylov_matrix = krylov(T, self.A, self.B).float()
                if not self.trainable:
                    self.krylov_matrix = krylov_matrix

            kernel = torch.einsum("cmn, cnt -> cmt", self.C, krylov_matrix[..., :T])
            y = causal_convolution(kernel, eo.rearrange(x, "t b c -> b c 1 t"))
            return self.position_wise(
                eo.rearrange(y, "b c m t -> t b c m")
                + torch.einsum("cm, tbc -> tbcm", self.D, x)
            )
        else:
            c = unroll_time_invariant_dynamics(x, self.A, self.B)
            lssl_out = torch.einsum("cmn, tbcn -> tbcm", self.C, c) + torch.einsum(
                "cm, tbc -> tbcm", self.D, x
            )
            return self.position_wise(lssl_out)


class UnLSSLLayer(LSSLLikeLayer):
    """Uncertainty-aware LSSL layer with time-invariant dynamics.

    Uses the same regularized uncertainty-aware HiPPO matrix at every step.
    """

    def __init__(
        self,
        n: int,
        n_channels: int,
        n_latent_channels: int,
        *,
        trainable: bool,
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
    ):
        self.obs_sigma2 = obs_sigma2
        self.trans_sigma2 = trans_sigma2
        self.symmetrize = symmetrize
        self.joseph = joseph
        self.discretization_method = discretization_method

        A_R = regularized_hippo_matrix(n)
        if torch.cuda.is_available():
            A_R = A_R.to(device="cuda")

        k = torch.arange(int(max_t) + 1, device=A_R.device)
        k[0] = 1
        A = discretize_unhippo_dynamics(A_R, k, k + 1, method=self.discretization_method)
        B = torch.sqrt(2 * torch.arange(n, device=A.device, dtype=torch.double) + 1)

        Id = torch.eye(n, device=A.device)
        P, K = kalman_filter_parameters(
            n,
            max_t + 1,
            A,
            B,
            sigma2=self.obs_sigma2,
            Sigma=self.trans_sigma2 * Id,
            symmetrize=self.symmetrize,
            joseph=self.joseph,
        )
        steps = torch.logspace(
            np.log10(min_t),
            np.log10(max_t),
            n_channels,
            dtype=torch.long,
            device=A.device,
        )
        K_steps = K[steps].clone()
        A_U = (Id - torch.einsum("cn, N -> cnN", K_steps, B)) @ A[steps]

        super().__init__(
            n=n,
            n_channels=n_channels,
            n_latent_channels=n_latent_channels,
            trainable=trainable,
            krylov=krylov,
            A=A_U,
            B=K_steps,
        )


class LSSLLayer(LSSLLikeLayer):
    """LSSL layer."""

    def __init__(
        self,
        n: int,
        n_channels: int,
        n_latent_channels: int,
        *,
        trainable: bool,
        krylov: bool = True,
        min_t: float = 10,
        max_t: float = 1000,
        discretization_method: Literal[
            "bilinear", "bilinear-lssl", "forward", "backward"
        ] = "bilinear-lssl",
    ):
        self.discretization_method = discretization_method

        A = hippo_matrix(n)
        B = torch.sqrt(2 * torch.arange(n) + 1)
        t = torch.logspace(np.log10(min_t), np.log10(max_t), n_channels)
        A_bar, B_bar = discretize_hippo_dynamics(
            A, B, t, t + 1, method=discretization_method
        )
        super().__init__(
            n=n,
            n_channels=n_channels,
            n_latent_channels=n_latent_channels,
            trainable=trainable,
            krylov=krylov,
            A=A_bar,
            B=B_bar,
        )
