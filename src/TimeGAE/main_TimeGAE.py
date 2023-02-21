import sys

sys.path.append("/public/home/bierongfang/zzm/TSGAN")
sys.path.append("/home/zzm/TSGAN")

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset.EnergyDataset import EnergyDataset
from src.TimeGAE.TimeGAE import TimeGAE
from src.dataset.GolfDataset import GolfDataset
from src.dataset.WinnipegDataset import WinnipegDataset
from src.utils.parse_args import parse_args
from src.utils.setup_seed import setup_seed
from src.utils.send_email import send_email

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args()
    setup_seed(args.seed)
    torch.cuda.set_device(args.cuda)
    try:
        print(args)

        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using: {}".format(device))
        print("device: {}".format(args.cuda))

        # load dataset
        if args.dataset == "energy":
            dataset = EnergyDataset(test_size=args.test_size)
        elif args.dataset == "winnipeg":
            dataset = WinnipegDataset(test_size=args.test_size)
        else:
            dataset = GolfDataset(test_size=args.test_size)
        print(dataset)

        # split train test set
        train_set, test_set = dataset.get_train_test_set()
        train_data, train_label = train_set[0], train_set[1]
        test_data, test_label = test_set[0], test_set[1]

        # load graph structure
        graph = dataset.get_graph()
        graph = graph.to(device)
        print(graph)

        # dataloader
        train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers)

        num_samples, num_sensors, seq_len = dataset.get_shape()

        # model
        model = TimeGAE(seq_len=seq_len, emb_size=args.emb_size, num_sensors=num_sensors,
                        feat_extract=args.feat_extract, num_heads=args.num_heads, num_layers=args.num_layers,
                        bidirectional=args.bidirectional, dropout_ratio=args.dropout_ratio)
        model = model.to(device)
        print(model)

        # optimizer
        if args.optim == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # scheduler
        if args.scheduler:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, verbose=True)

        mse_loss = nn.MSELoss()

        train_loss_list = []
        test_loss_list = []

        # train
        for epoch in range(args.epochs):
            model.train()
            train_batch_loss = []
            for batch, time_series_origin in enumerate(train_dataloader):
                # 需要transpose才能送入GNN
                # 一个结点表示一个sensor，特征是一个batch的time series
                time_series_origin = time_series_origin.transpose(0, 1).to(device)
                # time_series_origin.shape == (num_sensors, batch_size, length)
                # time_series_generate.shape == (num_sensors, batch_size, length)
                time_series_generate, encoder_edge_weight, decoder_edge_weight = model(graph, time_series_origin)

                loss = mse_loss(time_series_origin, time_series_generate)

                optimizer.zero_grad()
                loss.backward()

                # for name, parms in model.named_parameters():
                #     if parms.requires_grad and parms.grad is not None:
                #         print('-->name:', name)
                #         print('-->grad_value_norm:', torch.norm(parms.grad))

                # 梯度裁剪
                if args.max_norm != -1:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

                optimizer.step()

                print(
                    "train: [Epoch %d/%d] [Batch %d/%d] [MSE loss: %f]"
                    % (epoch + 1, args.epochs, batch + 1, len(train_dataloader), loss.item())
                )
                train_batch_loss.append(loss.item())

            mean_train_batch_loss = np.mean(train_batch_loss)
            train_loss_list.append(mean_train_batch_loss)

            # eval
            if (epoch + 1) % args.test_interval == 0:
                model.eval()
                with torch.no_grad():
                    test_batch_loss = []
                    for batch, time_series_origin in enumerate(test_dataloader):
                        time_series_origin = time_series_origin.transpose(0, 1).to(device)
                        # time_series_origin.shape == (num_sensors, batch_size, length)
                        time_series_generate, encoder_edge_weight, decoder_edge_weight = model(
                            graph, time_series_origin)
                        # time_series_generate.shape == (num_sensors, batch_size, length)

                        loss = mse_loss(time_series_origin, time_series_generate)

                        print(
                            "test: [Epoch %d/%d] [Batch %d/%d] [MSE loss: %f]"
                            % (epoch + 1, args.epochs, batch + 1, len(test_dataloader), loss.item())
                        )
                        test_batch_loss.append(loss.item())

                    mean_test_batch_loss = np.mean(test_batch_loss)
                    test_loss_list.append(mean_test_batch_loss)

                    if args.scheduler:
                        scheduler.step(mean_test_batch_loss)

        # generate
        with torch.no_grad():
            print("origin dataset shape: {}".format(dataset.data.shape))
            generate = []
            dataloader = DataLoader(dataset=dataset.data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers)
            for batch, time_series_origin in enumerate(tqdm(dataloader)):
                time_series_origin = time_series_origin.transpose(0, 1).to(device)
                # time_series_origin.shape == (num_sensors, batch_size, length)
                time_series_generate, encoder_edge_weight, decoder_edge_weight = model(graph, time_series_origin)
                # time_series_generate.shape == (num_sensors, batch_size, length)
                time_series_generate = time_series_generate.transpose(0, 1).transpose(1, 2).cpu().numpy()
                # time_series_generate.shape == (batch_size, length, num_sensors)

                generate.append(time_series_generate)
            generate = np.concatenate(generate, axis=0)
            print("generate dataset shape: {}".format(generate.shape))
            # generate.shape == (num_samples, length, num_sensors)
            np.save(f"./timegae_gen_{args.dataset}_{args.feat_extract}.npy", generate)

        # plot learning curve
        plt.figure()
        plt.plot(train_loss_list, label="train loss")
        plt.plot(test_loss_list, label="test loss")
        plt.title("learning curve")
        plt.legend()
        plt.savefig("./figures/TimeGAE_learning_curve.png")
        plt.show()

        title = f"TimeGAE_{args.dataset} run success!"
        content = title
        send_email(title, content)

    except Exception as e:
        title = f"TimeGAE_{args.dataset} run fail!"
        content = e
        send_email(title, content)
