import torch
import cupy as cp


def cp_to_torch(x):
    torch_device = torch.device("cuda")
    torch_dtype  = torch.get_default_dtype()
    return (
        torch.utils.dlpack.from_dlpack(x)
        .to(device=torch_device,
            dtype=torch_dtype)
    )

def torch_to_cp(x):
    return cp.from_dlpack(
        x.detach()
    )
