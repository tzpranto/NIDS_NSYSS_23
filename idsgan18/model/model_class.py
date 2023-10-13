import torch
from torch import nn


class Blackbox_IDS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Dropout(0.2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.Dropout(0.2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 8, output_dim),
        )

    def forward(self, x):
        x = self.layer(x)
        x = torch.nn.Sigmoid()(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 144)
        self.fc2 = nn.Linear(144, 138)
        self.fc3 = nn.Linear(138, 132)
        self.fc4 = nn.Linear(132, 128)
        self.fc5 = nn.Linear(128, output_dim)

        self.relu = nn.Tanh()

    def forward(self, x, mask):
        out1 = self.fc1(x)
        out1 = self.relu(out1)
        out2 = self.fc2(out1)
        out2 = self.relu(out2)
        out3 = self.fc3(out2)
        out3 = self.relu(out3)
        out4 = self.fc4(out3)
        out4 = self.relu(out4)
        out5 = self.fc5(out4)
        out5 = self.relu(out5)

        out5 = out5 * mask

        return torch.clamp(out5, -.5, .5)


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 8, output_dim),
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.sigmoid(out)
