import torch
from typing import Tuple
import torch.nn.functional as F


def pixel_unshuffle(x: torch.Tensor, factor: int):
    y = x[:, :, 0:x.size()[2] // factor * factor, 0:x.size()[3] // factor * factor]

    B, iC, iH, iW = y.shape
    oC, oH, oW = iC*(factor*factor), iH//factor, iW//factor
    y = y.reshape(B, iC, oH, factor, oW, factor)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, factor, factor, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


def pixel_shuffle_cat(x1: torch.Tensor, x2: torch.Tensor, factor: int):
    x1 = F.pixel_shuffle(x1, factor)

    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2))

    x = torch.cat([x2, x1], dim=1)

    return x