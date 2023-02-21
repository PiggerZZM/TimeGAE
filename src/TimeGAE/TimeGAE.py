from torch import nn

from src.TimeGAE.DecoderBlock import DecoderBlock
from src.TimeGAE.EncoderBlock import EncoderBlock


class TimeGAE(nn.Module):
    def __init__(self, seq_len, emb_size, num_sensors, feat_extract, num_heads, num_layers, bidirectional, dropout_ratio):
        super(TimeGAE, self).__init__()
        self.encoder = EncoderBlock(seq_len=seq_len, num_sensors=num_sensors, emb_size=emb_size,
                                    num_heads=num_heads,
                                    feat_extract=feat_extract,
                                    num_layers=num_layers, bidirectional=bidirectional,
                                    dropout_ratio=dropout_ratio)
        self.decoder = DecoderBlock(seq_len=seq_len, num_sensors=num_sensors, emb_size=emb_size,
                                    num_heads=num_heads,
                                    feat_extract=feat_extract,
                                    num_layers=num_layers, bidirectional=bidirectional,
                                    dropout_ratio=dropout_ratio)

    def encode(self, graph, feature):
        encode, encoder_edge_weight = self.encoder(graph, feature)
        return encode, encoder_edge_weight

    def decode(self, graph, encode):
        decode, decoder_edge_weight = self.decoder(graph, encode)
        return decode, decoder_edge_weight

    def forward(self, graph, feature):
        # feature.shape == (num_sensors, batch_size, length)
        encode, encoder_edge_weight = self.encode(graph, feature)
        decode, decoder_edge_weight = self.decode(graph, encode)
        # decode.shape == (num_sensors, batch_size, length)
        return decode, encoder_edge_weight, decoder_edge_weight
