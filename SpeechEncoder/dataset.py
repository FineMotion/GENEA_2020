import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List


class MotionDataset(Dataset):
    def __init__(self, data_files: List[str], device: torch.device):
        self.X = []
        self.Y = []
        self.device = device

        for data_file in data_files:
            data = np.load(data_file)
            x, y = data['X'], data['Y']
            self.X.append(x)
            self.Y.append(y)
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)
        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item: int):
        return torch.from_numpy(self.X[item]), torch.from_numpy(self.Y[item])

    def collate_fn(self, batch):
        x, y = list(zip(*batch))
        batch_size = len(x)
        input_tensor = torch.empty((batch_size, x[0].size()[0], x[0].size()[1]), dtype=torch.float)
        output_tensor = torch.empty((batch_size, y[0].size()[0]), dtype=torch.float)

        for i in range(batch_size):
            input_tensor[i] = x[i]
            output_tensor[i] = y[i]

        return input_tensor.to(self.device), output_tensor.to(self.device)