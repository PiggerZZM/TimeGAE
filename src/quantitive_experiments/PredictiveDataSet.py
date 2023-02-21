from torch.utils.data import Dataset


class PredictiveDataSet(Dataset):
    def __init__(self, dataset):
        super(PredictiveDataSet, self).__init__()
        self.data = dataset

    def __getitem__(self, item):
        return self.data[item, :, :]

    def __len__(self):
        return self.data.shape[0]
