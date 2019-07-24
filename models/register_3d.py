import torch.nn as nn
from layers.transformer_3d import Transformer3d
from .deform_decoder import DRB
from .affine_decoder import LRB
from .encoder import Encoder
from utils.image_sampling import affine_grid_3d, gradient_grid_3d
from utils.image_sampling import build_affine_grid


class Register3d(nn.Module):
    def __init__(self, size, device, linear=True):
        super(Register3d, self).__init__()
        self.encoder = Encoder()
        self.d_decoder = DRB()
        self.a_decoder = LRB()
        self.transformer = Transformer3d()
        self.size = size
        self.linear = linear
        self.device = device
        if self.linear:
            self.linear_grid = build_affine_grid(self.size, self.device)

    def forward(self, source, target):
        x = self.encoder(source, target)
        grad = self.d_decoder(x)
        d1 = gradient_grid_3d(grad)
        if not self.linear:
            t = self.transformer(x, d1)
            return t, grad
        else:
            theta = self.a_decoder(x)
            d2 = affine_grid_3d(theta, self.linear_grid, self.size)
            t = self.transformer(x, d1, d2)
            return t, grad, theta
