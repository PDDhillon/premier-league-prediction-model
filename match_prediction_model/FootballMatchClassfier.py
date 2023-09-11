from torch import nn
import torch

class FootballMatchClassfier(nn.Module):
    def __init__(self, input_shape):
        super(FootballMatchClassfier, self).__init__()
        self.fc1 = nn.Linear(input_shape, 300)
        self.fc2 = nn.Linear(300,100)
        self.fc3 = nn.Linear(100,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x.to(torch.float)))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.to(torch.float64)
