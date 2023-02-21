import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, num_sensors, rnn_layers=2):
        super(Predictor, self).__init__()
        self.rnn = nn.GRU(input_size=num_sensors, hidden_size=num_sensors, num_layers=rnn_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ts):
        output, _ = self.rnn(ts)
        return output


if __name__ == '__main__':
    pre = Predictor(num_sensors=8, rnn_layers=2)
    ts = torch.ones((213, 645, 8)).float()
    print(pre(ts).shape)
