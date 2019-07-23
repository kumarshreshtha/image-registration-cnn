import torch.nn.functional as F
import torch.nn as nn


class Transformer3d(nn.Module):
    def __init__(self):
        super(Transformer3d, self).__init__()

    def forward(self, source, grad_grid, affine_grid=None):
        x = F.grid_sample(source, grad_grid)
        if affine_grid is not None:
            x = F.grid_sample(x, affine_grid)
        return x
