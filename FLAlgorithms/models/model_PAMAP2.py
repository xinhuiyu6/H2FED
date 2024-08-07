import torch.nn as nn
import torch.nn.functional as F


class CNN_tiny(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN_tiny, self).__init__()

        self.layer1 = self._make_layers(input_channel, 16, (6, 2), (3,2), (1,0))
        self.layer2 = self._make_layers(16, 32, (6, 2), (3,2), (1,0))
        self.layer3 = self._make_layers(32, 64, (6, 2), (3,2), (1,0))
        self.fc = nn.Linear(1024, num_classes)  # 1024

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x, out


class CNN(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.fc = nn.Linear(36864, num_classes)  # 36864

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x, out


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, (3,1), 1, (1,0)),
            nn.BatchNorm2d(output_channel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = x + identity
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.fc = nn.Linear(36864, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return x, out


class ActivityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8, num_layers=1):
        super(ActivityLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, hidden1 = self.lstm1(x)
        x, hidden2 = self.lstm2(x)
        out = x[:, -1, :]  # Take the last time step output
        out = self.fc(out)
        return x, out