import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv3d(128, 32, 3, padding=3, dilation=3)
        self.conv5 = nn.Conv3d(128, 32, 3, padding=5, dilation=5)

    def forward(self, source, target):
        x0 = torch.cat(source, target, dim=1)
        x1 = F.leaky_relu(F.instance_norm(self.conv1(x0)))
        x2 = F.leaky_relu(F.instance_norm(self.conv1(x1)))
        x3 = F.leaky_relu(F.instance_norm(self.conv1(x2)))
        x4 = F.leaky_relu(F.instance_norm(self.conv1(x3)))
        x5 = F.leaky_relu(F.instance_norm(self.conv1(x4)))
        return torch.cat((x0, x1, x2, x3, x4, x5), dim=1)
