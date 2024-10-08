from torch import nn
import torch

class FootballMatchPredictionModel(nn.Module):
    def __init__(self, input_shape):
        super(FootballMatchPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input=input_shape, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x.to(torch.float64)
