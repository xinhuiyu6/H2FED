import torch
import torch.nn as nn
import torch.nn.functional as F


class EHR_FCN(nn.Module):
    def __init__(self, dim):
        super(EHR_FCN, self).__init__()

        self.fc1 = nn.Linear(26, 128)  # Adjust the input features to match the output of last conv layer
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, dim)

    def forward(self, x):

        bs = x.size(0)
        x = x.view(bs, 26)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = x
        x = self.fc3(x)

        return x1, x


