import torch.nn.functional as F
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel//reduction, 1)
        self.conv2 = nn.Conv3d(channel//reduction, channel, 1)

    def forward(self, x):
        return x*F.sigmoid(self.conv2(F.relu(
            self.conv1(F.adaptive_avg_pool3d(x, (None, None, 1, 1))))))
