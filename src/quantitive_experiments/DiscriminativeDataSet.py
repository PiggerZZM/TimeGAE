from torch.utils.data import Dataset


class DiscriminativeDataSet(Dataset):
    def __init__(self, dataset):
        super(DiscriminativeDataSet, self).__init__()
        self.data, self.label = dataset

    def __getitem__(self, item):
        return self.data[item, :, :], self.label[item]

    def __len__(self):
        return self.data.shape[0]

