import numpy as np
import torch


def concat_and_random_split_train_test(ori_data, gen_data, test_size=0.1):
    data = np.concatenate([ori_data, gen_data])
    label = np.concatenate([np.ones(ori_data.shape[0]), np.zeros(gen_data.shape[0])])

    data = torch.from_numpy(data)
    label = torch.from_numpy(label)

    samples = np.arange(data.shape[0]).astype(int)
    samples = np.random.permutation(samples)
    test_size = int(len(samples) * test_size) + 1

    train_samples = samples[test_size:]
    test_samples = samples[:test_size]

    train_set = data[train_samples], label[train_samples]
    test_set = data[test_samples], label[test_samples]

    return train_set, test_set
