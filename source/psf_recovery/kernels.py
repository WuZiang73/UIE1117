import math
import torch
import torch.nn.functional as F


def gaussian_kernel(sigma, radius=8, dtype=torch.float32):
    if sigma <= 1e-8:
        k = torch.zeros(2 * radius + 1, 2 * radius + 1, dtype=dtype)
        k[radius, radius] = 1.0
        return k
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=dtype),
        torch.arange(-radius, radius + 1, dtype=dtype),
        indexing="ij",
    )
    k = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return k / k.sum()


def kernel_bank(sigmas=(0.0, 0.6, 1.2, 2.4), radius=8, aspect=(1.0, 1.0), dtype=torch.float32):
    ks = []
    for s in sigmas:
        if s <= 1e-8:
            ks.append(gaussian_kernel(0.0, radius, dtype))
        else:
            y, x = torch.meshgrid(
                torch.arange(-radius, radius + 1, dtype=dtype),
                torch.arange(-radius, radius + 1, dtype=dtype),
                indexing="ij",
            )
            sx = s * math.sqrt(aspect[0] / aspect[1])
            sy = s * math.sqrt(aspect[1] / aspect[0])
            k = torch.exp(-(x * x / (2.0 * sx * sx) + y * y / (2.0 * sy * sy)))
            ks.append(k / k.sum())
    return torch.stack(ks, 0)
