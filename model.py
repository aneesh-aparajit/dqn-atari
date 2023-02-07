import torch
import torch.nn as nn
import torch.optim as optim
import ipdb

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNModel(nn.Module):
    def __init__(self, obs_shape: tuple, n_actions: int, window_size: int = 4, lr: float = 1e-4):
        assert len(obs_shape) == 3, "The observation shape between a gray scale or a rgb image."
        super(DQNModel, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.lr = lr

        self.net = nn.Sequential(
            Conv(in_channels=window_size, out_channels=16, kernel_size=(8, 8), stride=(3, 3), padding=1),
            nn.ReLU(),
            Conv(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=1)
        )

        self.fc = nn.Linear(in_features=1568, out_features=self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


if __name__ == '__main__':
    m = DQNModel(obs_shape=(84, 84, 3), n_actions=4)
    x = torch.rand((32, 4, 84, 84))
    y = m(x)

    ipdb.set_trace()
