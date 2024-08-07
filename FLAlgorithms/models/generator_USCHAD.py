import torch
import torch.nn as nn
cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, z_size, input_feat, fc_units=400):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.input_feat = input_feat
        inp_unit = z_size

        self.model = nn.Sequential(
            nn.Linear(inp_unit, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fc_units),

            nn.Linear(fc_units, 2*fc_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2*fc_units),

            nn.Linear(2*fc_units, input_feat),
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        self.linear2 = nn.Linear(hidden_dim, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear4 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f):
        x = f
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x


















