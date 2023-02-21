import argparse
import sys

sys.path.append("/public/home/bierongfang/zzm/TSGAN")
sys.path.append("/home/zzm/TSGAN")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.quantitive_experiments.PredictiveDataSet import PredictiveDataSet
from src.quantitive_experiments.Predictor import Predictor
from src.utils.setup_seed import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Predictive Experiments Args")
    parser.add_argument('--rnn_layers', type=int, default=2,
                        help="the layer num of stacked rnn")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="the learning rate")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--test_interval', type=int, default=1, help="test when running test_interval epochs")
    parser.add_argument('--notice', type=str, default="normal", help="notice text")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument(
        "--model_name",
        choices=['TimeGAE', 'TimeVAE', 'TimeGAN', "TTSGAN"],
        default='TimeGAE',
        type=str
    )
    parser.add_argument(
        "--dataset",
        choices=["golf", "energy", "winnipeg"],
        default="golf",
        type=str
    )

    return parser.parse_args()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    args = parse_args()

    setup_seed(args.seed)

    torch.cuda.set_device(args.cuda)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using: {}".format(device))
    print("device: {}".format(args.cuda))

    if args.dataset == "golf":
        ori_data = np.load("../../golf_dataset/olddata/old_data_preprocess_no_aug.npy").swapaxes(1, 2)
    elif args.dataset == "energy":
        ori_data = np.load("../../energyDataset/energy.npy").swapaxes(1, 2)
    else:
        ori_data = np.load("../../WinnipegDataset/winnipeg_1.npy").swapaxes(1, 2)

    if args.model_name == "TimeGAN":
        generated_data = np.load(f"../TimeGAN/timegan_gen_{args.dataset}.npy")
    elif args.model_name == "TimeGAE":
        generated_data = np.load(f"../TimeGAE/timegae_gen_{args.dataset}_transformer.npy")
    elif args.model_name == "TimeVAE":
        generated_data = np.load(f"../TimeVAE/outputs/timevae_gen_{args.dataset}.npy")
    elif args.model_name == "TimeGVAE":
        generated_data = np.load(f"../TimeGAE/timegvae_gen_{args.dataset}_mlp.npy")
    else:
        generated_data = np.load(f"../TTSGAN/ttsgan_gen_{args.dataset}.npy").swapaxes(1, 2)

    train_set, test_set = ori_data, generated_data

    pre_train_set = PredictiveDataSet(train_set)
    pre_test_set = PredictiveDataSet(test_set)

    num_sensors = pre_train_set[0].shape[1]

    # dataloader
    train_dataloader = DataLoader(dataset=pre_train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=pre_test_set, batch_size=args.batch_size, shuffle=True)

    # model
    model = Predictor(num_sensors=num_sensors, rnn_layers=args.rnn_layers)
    model = model.to(device)
    print(model)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epochs):
        model.train()
        train_batch_loss = []
        for batch, ts_train in enumerate(train_dataloader):

            ts_before = ts_train[:, :-1, :].to(device)
            ts_next = ts_train[:, 1:, :].to(device)

            outputs = model(ts_before)

            loss = mse_loss(outputs, ts_next).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "train: [Epoch %d/%d] [Batch %d/%d] [MSE loss: %f]"
                % (epoch + 1, args.epochs, batch + 1, len(train_dataloader), loss.item())
            )
            train_batch_loss.append(loss.item())

        train_loss_list.append(np.mean(train_batch_loss))

        # eval
        if (epoch + 1) % args.test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_batch_loss = []
                for batch, ts_test in enumerate(test_dataloader):

                    ts_before = ts_test[:, :-1, :].double().to(device)
                    ts_next = ts_test[:, 1:, :].double().to(device)

                    outputs = model(ts_before)

                    loss = mae_loss(outputs, ts_next).mean()

                    print(
                        "test: [Epoch %d/%d] [Batch %d/%d] [MAE loss: %f]"
                        % (epoch + 1, args.epochs, batch + 1, len(test_dataloader), loss)
                    )
                    test_batch_loss.append(loss.item())

                test_loss_list.append(np.mean(test_batch_loss))

    # plot learning curve
    plt.figure()
    min_index = np.argmin(test_loss_list)
    plt.plot(min_index, test_loss_list[min_index], 'ks')
    show_max = '(' + str(min_index) + ',' + str(test_loss_list[min_index]) + ')'
    plt.annotate(show_max, xytext=(min_index, test_loss_list[min_index]), xy=(min_index, test_loss_list[min_index]))
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.title("learning curve")
    plt.legend()
    plt.savefig(f"./pre_score_{args.model_name}_{args.dataset}_learning_curve.png")
    plt.show()
