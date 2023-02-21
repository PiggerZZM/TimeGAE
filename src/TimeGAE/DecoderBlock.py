from torch import nn

from src.TimeGAE.TimeGraphConv import TimeGraphConv


class DecoderBlock(nn.Module):
    def __init__(self, seq_len, num_sensors, emb_size=16, num_heads=2, feat_extract="transformer",
                 num_layers=1, bidirectional=False, dropout_ratio=0):
        super(DecoderBlock, self).__init__()
        self.graph_conv = TimeGraphConv(seq_len=seq_len, num_sensors=num_sensors, feat_extract=feat_extract,
                                        num_heads=num_heads, emb_size=emb_size, activation=nn.ReLU(),
                                        num_layers=num_layers, bidirectional=bidirectional,
                                        dropout_ratio=dropout_ratio,
                                        edge_dropout_ratio=0)

    def forward(self, graph, features):
        fake_tss, graph_edge_weight = self.graph_conv(graph, features, edge_weight=True)
        return fake_tss, graph_edge_weight
