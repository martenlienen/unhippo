"""Module adapted from S4 [1].

[1] https://github.com/state-spaces/s4
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .utils import maybe_compile


@maybe_compile()
def krylov(
    L: int, A: Float[Tensor, "... n n"], b: Float[Tensor, "... n"]
) -> Float[Tensor, "... n L"]:
    """Compute the Krylov subspace (b, Ab, A^2b, ..., A^{L-1}b).

    Uses the squaring trick to save on matrix powers.

    Adapted from the S4 code [1].

    [1] https://github.com/state-spaces/s4
    """

    assert L >= 1

    x = b[..., None]
    A_ = A
    while True:
        x = torch.cat((x, A_ @ x[..., : (L - x.shape[-1])]), dim=-1)
        if x.shape[-1] == L:
            break
        A_ = A_ @ A_

    return x


def triangular_toeplitz_multiply(u, v):
    n = u.shape[-1]
    u_expand = F.pad(u, (0, n))
    v_expand = F.pad(v, (0, n))
    u_f = torch.fft.rfft(u_expand, n=2 * n, dim=-1)
    v_f = torch.fft.rfft(v_expand, n=2 * n, dim=-1)
    uv_f = u_f * v_f
    output = torch.fft.irfft(uv_f, n=2 * n, dim=-1)[..., :n]
    return output


class TriangularToeplitzMultFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        n = u.shape[-1]
        u_expand = F.pad(u, (0, n))
        v_expand = F.pad(v, (0, n))
        u_f = torch.fft.rfft(u_expand, n=2 * n, dim=-1)
        v_f = torch.fft.rfft(v_expand, n=2 * n, dim=-1)

        ctx.save_for_backward(u_f, v_f)

        uv_f = u_f * v_f
        output = torch.fft.irfft(uv_f, n=2 * n, dim=-1)[..., :n]
        return output

    @staticmethod
    def backward(ctx, grad):
        u_f, v_f = ctx.saved_tensors
        n = grad.shape[-1]
        g_expand = F.pad(grad.flip(-1), (0, n))
        g_f = torch.fft.rfft(g_expand, n=2 * n, dim=-1)
        gu_f = g_f * u_f
        gv_f = g_f * v_f
        d_u = torch.fft.irfft(gv_f, n=2 * n, dim=-1)[..., :n]
        d_v = torch.fft.irfft(gu_f, n=2 * n, dim=-1)[..., :n]
        d_u = d_u.flip(-1)
        d_v = d_v.flip(-1)
        return d_u, d_v


triangular_toeplitz_multiply_fast = TriangularToeplitzMultFast.apply


def causal_convolution(
    u: Float[Tensor, "... t"], v: Float[Tensor, "... t"], fast: bool = True
) -> Float[Tensor, "... t"]:
    if fast:
        return triangular_toeplitz_multiply_fast(u, v)
    else:
        return triangular_toeplitz_multiply(u, v)
