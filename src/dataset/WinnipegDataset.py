import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from torch.utils.data import Dataset, DataLoader


class WinnipegDataset(DGLDataset, Dataset):

    def __init__(self, test_size):
        self.test_size = test_size
        super(WinnipegDataset, self).__init__(name='WinnipegDataset')

    def process(self):
        self.data = np.load("../../WinnipegDataset/winnipeg_1.npy")
        self.data = torch.from_numpy(self.data)
        self.label = np.load("../../WinnipegDataset/winnipeg_labels_1.npy")
        self.label = torch.from_numpy(self.label)

        self.num_samples, self.num_sensors, self.seq_len = self.data.shape

        self.train_set, self.test_set = self.__train_test_split_graph()

        self._create_graph()

    def get_dataset(self):
        return self.data, self.label

    def get_train_test_set(self):
        return self.train_set, self.test_set

    def _create_graph(self):
        u, v = [], []
        for id1 in range(self.num_sensors):
            for id2 in range(self.num_sensors):
                u.append(id1)
                v.append(id2)

        # complete graph
        self.graph = dgl.graph((u, v))

    def get_graph(self):
        return self.graph

    def __getitem__(self, sample):
        return self.data[sample, :, :]

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        return "num samples: {}, num sensors: {}, seq_len: {}".format(
            self.num_samples, self.num_sensors, self.seq_len)

    def get_shape(self):
        return self.num_samples, self.num_sensors, self.seq_len

    def __train_test_split_graph(self):
        samples = np.arange(self.num_samples).astype(int)
        samples = np.random.permutation(samples)
        test_size = int(len(samples) * self.test_size) + 1

        train_samples = samples[test_size:]
        test_samples = samples[:test_size]

        train_set = self.data[train_samples], self.label[train_samples]
        test_set = self.data[test_samples], self.label[test_samples]

        return train_set, test_set


if __name__ == '__main__':
    dataset = WinnipegDataset(test_size=0.1)
    print(dataset)

    train_set, test_set = dataset.get_train_test_set()
    train_data, train_label = train_set[0], train_set[1]
    test_data, test_label = test_set[0], test_set[1]

    graph = dataset.get_graph()

    train_dataloader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=test_data, batch_size=2, shuffle=True, num_workers=0)

    for batch, time_series_origin in enumerate(train_dataloader):
        print(time_series_origin.shape)
        break
