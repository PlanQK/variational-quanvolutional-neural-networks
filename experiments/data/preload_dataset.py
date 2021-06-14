import torch
from torch.utils.data import Dataset, DataLoader


class PreLoadIntoMemory(Dataset):

    def __init__(self, dataset, num_workers=4):

        self.x = None
        self.y = None

        for i, (x, y) in enumerate(DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)):
            if i == 0:
                self.x = torch.empty((dataset.__len__(),) + x[0].shape, dtype=x.dtype)
                self.y = torch.empty((dataset.__len__(),) + y[0].shape, dtype=y.dtype)

            self.x[i] = x[0]
            self.y[i] = y[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class PreloadFromFile(PreLoadIntoMemory):

    def __init__(self, x_path, y_path):
        self.x = torch.load(x_path)
        self.y = torch.load(y_path)


class PreloadXFromFileYManually(PreLoadIntoMemory):

    def __init__(self, x_path, y):
        self.x = torch.load(x_path)
        self.y = y


class SetManually(PreLoadIntoMemory):

    def __init__(self, x, y):
        self.x = x
        self.y = y
