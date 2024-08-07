import torch
import torch.nn as nn
import torch.nn.functional as F


class HARBox_CNN(nn.Module):
    def __init__(self, shrinkage_ratio):
        super(HARBox_CNN, self).__init__()
        self.shrinkage_ratio = shrinkage_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=int(16 * self.shrinkage_ratio),
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(int(16 * self.shrinkage_ratio)).requires_grad_(False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(16 * self.shrinkage_ratio),
                out_channels=int(32 * self.shrinkage_ratio),
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)).requires_grad_(False),
        )

        self.linear = nn.Linear(int(32 * self.shrinkage_ratio) * 8 * 8, 5)

    def forward(self, x):

        self.batch_size = x.size(0)
        x = x.view(self.batch_size, 1, 30, 30)

        conv_output = torch.relu(self.conv1(x))
        conv_output = torch.relu(self.conv2(conv_output))

        conv_output = conv_output.view(self.batch_size, int(32 * self.shrinkage_ratio) * 8 * 8)
        output = self.linear(conv_output)

        return conv_output, output


class HARBox_1DCNN(nn.Module):
    def __init__(self):
        super(HARBox_1DCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 112, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(-1, 1, 900)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(bs, -1)
        x1 = x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x1, x


class HARBox_1DCNNtiny(nn.Module):
    def __init__(self):
        super(HARBox_1DCNNtiny, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 225, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(-1, 1, 900)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 225)
        x1 = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x1, x



