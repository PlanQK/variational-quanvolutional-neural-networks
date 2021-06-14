import os

import torch
from torch.utils.data import Dataset


class LoadIndividualTensorsSorted(Dataset):

    def __init__(self, dir_path):
        self.data = sorted(os.listdir(dir_path), key=lambda tensor_name: int(''.join(filter(str.isdigit, tensor_name))))
        self.data = [os.path.join(dir_path, tensor_name) for tensor_name in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.load(self.data[index])
