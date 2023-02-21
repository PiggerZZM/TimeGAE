import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="TimeGAE")
    parser.add_argument('--emb_size', type=int, default=16,
                        help="embedding size")
    parser.add_argument('--num_layers', type=int, default=1,
                        help="num_layers for time series feature extractor")
    parser.add_argument('--gnn_layers', type=int, default=1,
                        help="stacked gnn_layers")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--dataset', type=str, default='golf',
                        help="available datasets: [golf, energy, winnipeg]")
    parser.add_argument('--test_size', type=float, default=0.1, help="test size")
    parser.add_argument('--test_interval', type=int, default=1, help="test when running test_interval epochs")
    parser.add_argument('--patience', type=int, default=100, help="reduce LR patience")
    parser.add_argument('--notice', type=str, default="normal", help="notice text")
    parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
    parser.add_argument('--num_heads', type=int, default=2, help="num_heads")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--dropout_ratio', type=float, default=0, help="dropout_ratio")
    parser.add_argument('--edge_dropout_ratio', type=float, default=0, help="edge_dropout_ratio")
    parser.add_argument('--bidirectional', action="store_true", help="use bidirectional rnn")
    parser.add_argument('--cuda', type=int, default=0, help="No. of cuda device")
    parser.add_argument('--feat_extract', type=str, default="transformer", help="type of feature extractor")
    parser.add_argument('--max_norm', type=float, default=-1,
                        help="max_norm of gradient clip. Not to use gradient clipping when max_norm == -1")
    parser.add_argument('--kld_weight', type=float, default=0.0001, help="weight of KL divergence loss")
    parser.add_argument('--hidden_dim', type=int, default=16, help="hidden_dim of encode")
    parser.add_argument('--optim', type=str, default="Adam", help="available optimizer: [SGD, Adam]")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum of SGD")
    parser.add_argument('--scheduler', action="store_true", help="use scheduler reduce LR")

    return parser.parse_args()
