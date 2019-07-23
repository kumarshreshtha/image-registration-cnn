import torch.nn.functional as F
import torch.nn as nn


class LRB(nn.Module):
    def __init__(self):
        super(LRB, self).__init__()
        self.conv1 = nn.Conv3d(290, 12, 1)

    def forward(self, inputs):
        x = self.conv1(F.adaptive_avg_pool3d(
            inputs, (None, None, None, 1, 1))).view(-1, 3, 4)
        return x
