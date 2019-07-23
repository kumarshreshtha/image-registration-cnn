import torch.nn.functional as F
import torch.nn as nn
from layers.se_layer import SqueezeExcitation


class DRB(nn.Module):
    def __init__(self):
        super(DRB, self).__init__()
        self.se = SqueezeExcitation(290)
        self.conv1 = nn.Conv3d(290, 128, 3, padding=1)
        self.conv2 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv3d(32, 3, 3, padding=1)

    def forward(self, inputs):
        x = self.se(inputs)
        x = F.leaky_relu(F.instance_norm(self.conv1(x)))
        x = F.leaky_relu(F.instance_norm(self.conv2(x)))
        x = F.leaky_relu(F.instance_norm(self.conv3(x)))
        x = F.leaky_relu(F.instance_norm(self.conv4(x)))
        x = F.leaky_relu(F.instance_norm(self.conv5(x)))
        x = 2*F.sigmoid(self.conv6(x))

        return x
