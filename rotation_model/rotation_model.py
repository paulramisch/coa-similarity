__all__ = ["Model"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Input image size 128, 128, 3
        self.conv1 = nn.Conv2d(3, 50, 5, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(50, 100, 3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,), stride=2)

        self.conv3 = nn.Conv2d(100, 150, 3, stride=2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(150, 200, 3, stride=2)
        self.dropout4 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(200, 200, 3, stride=2)
        self.dropout5 = nn.Dropout(p=0.3)

        # Linear layers
        self.linear6 = nn.Linear(200, 100)
        self.dropout6 = nn.Dropout(p=0.4)

        self.linear7 = nn.Linear(100, 200)
        self.dropout7 = nn.Dropout(p=0.4)

        self.linear8 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)

        # Linear layers
        x = self.linear6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.linear7(x)
        x = F.relu(x)
        x = self.dropout7(x)

        x = self.linear8(x)
        x = torch.flatten(x)

        return x

