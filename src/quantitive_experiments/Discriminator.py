import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, num_sensors, rnn_layers=2):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=num_sensors, hidden_size=num_sensors, num_layers=rnn_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ts):
        _, hn = self.rnn(ts)
        output = hn.mean(dim=0).mean(dim=1)  # 先对堆叠RNN取平均，在对sensor取平均
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    dis = Discriminator(num_sensors=8, rnn_layers=2)
    ts = torch.ones((213, 645, 8)).float()
    print(dis(ts).shape)
