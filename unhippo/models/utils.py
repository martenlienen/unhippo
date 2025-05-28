import torch


def maybe_compile(*args, **kwargs):
    """Apply torch.compile if the available GPU supports it."""

    # Triton compiler works on CPU or CUDA7.0
    supports_torch_compile = (
        not torch.cuda.is_available()
    ) or torch.cuda.get_device_properties(0).major >= 7
    if supports_torch_compile:
        return torch.compile(*args, **kwargs)
    else:
        return lambda f: f
