import argparse
import sys

sys.path.append("/public/home/bierongfang/zzm/TSGAN")
sys.path.append("/home/zzm/TSGAN")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.quantitive_experiments.DiscriminativeDataSet import DiscriminativeDataSet
from src.quantitive_experiments.Discriminator import Discriminator
from src.utils.setup_seed import setup_seed
from src.utils.train_test_split import concat_and_random_split_train_test


def parse_args():
    parser = argparse.ArgumentParser(description="Discriminative Experiments Args")
    parser.add_argument('--rnn_layers', type=int, default=2,
                        help="the layer num of stacked rnn")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="the learning rate")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--test_size', type=float, default=0.1, help="test size")
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
        generated_data = np.load(f"../TimeGAE/timegvae_gen_{args.dataset}_transformer.npy")
    else:
        generated_data = np.load(f"../TTSGAN/ttsgan_gen_{args.dataset}.npy").swapaxes(1, 2)

    train_set, test_set = concat_and_random_split_train_test(
        ori_data=ori_data, gen_data=generated_data, test_size=args.test_size)

    dis_train_set = DiscriminativeDataSet(train_set)
    dis_test_set = DiscriminativeDataSet(test_set)

    num_sensors = dis_train_set[0][0].shape[1]

    # dataloader
    train_dataloader = DataLoader(dataset=dis_train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=dis_test_set, batch_size=args.batch_size, shuffle=True)

    # model
    model = Discriminator(num_sensors=num_sensors, rnn_layers=args.rnn_layers)
    model = model.to(device)
    print(model)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCELoss()
    train_loss_list = []
    accuracy_list = []

    for epoch in range(args.epochs):
        model.train()
        train_batch_loss = []
        for batch, (ts_train, ts_label) in enumerate(train_dataloader):
            ts_train = ts_train.to(device)
            ts_label = ts_label.to(device)
            outputs = model(ts_train)

            loss = bce_loss(outputs, ts_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "train: [Epoch %d/%d] [Batch %d/%d] [BCE loss: %f]"
                % (epoch + 1, args.epochs, batch + 1, len(train_dataloader), loss.item())
            )
            train_batch_loss.append(loss.item())

        train_loss_list.append(np.mean(train_batch_loss))

        # eval
        if (epoch + 1) % args.test_interval == 0:
            model.eval()
            with torch.no_grad():
                batch_accuracy = []
                for batch, (ts_test, ts_label) in enumerate(test_dataloader):
                    ts_test = ts_test.to(device)
                    ts_label = ts_label.to(device)
                    outputs = model(ts_test)

                    accuracy = (outputs.round() == ts_label).sum() / args.batch_size

                    print(
                        "test: [Epoch %d/%d] [Batch %d/%d] [accuracy: %f]"
                        % (epoch + 1, args.epochs, batch + 1, len(test_dataloader), accuracy)
                    )
                    batch_accuracy.append(accuracy.item())

                accuracy_list.append(np.mean(batch_accuracy))

    # plot learning curve
    plt.figure()
    max_index = np.argmax(accuracy_list)
    plt.plot(max_index, accuracy_list[max_index], 'ks')
    show_max = '(' + str(max_index) + ',' + str(accuracy_list[max_index]) + ')'
    plt.annotate(show_max, xytext=(max_index, accuracy_list[max_index]), xy=(max_index, accuracy_list[max_index]))
    plt.plot(train_loss_list, label="train loss")
    plt.plot(accuracy_list, label="accuracy")
    plt.title("learning curve")
    plt.legend()
    plt.savefig(f"./dis_score_{args.model_name}_{args.dataset}_learning_curve.png")
    plt.show()
