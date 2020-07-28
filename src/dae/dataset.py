import numpy as np
import torch
from torch.utils.data import Dataset

def get_normalization_values(data: np.ndarray):
    max_val = np.amax(np.absolute(data), axis=0)
    mean_pose = data.mean(axis=0)
    return max_val, mean_pose


def normalize_data(data, max_val, mean_pose, eps = 1e-8):
    data_centered = data - mean_pose[np.newaxis, :]
    data_normalized = np.divide(data_centered, max_val[np.newaxis, :] + eps)
    return data_normalized


class NoisedMotionDataset(Dataset):
    def __init__(self, data: np.ndarray, device: torch.device, sigma, variance=0.01, train=True):
        self.data = data
        self.device = device
        self.sigma = sigma
        self.variance = variance
        self.eps = 1e-15
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        x = self.data[item]
        if self.train:
            noise = np.random.normal(0.0, np.multiply(self.sigma, self.variance) + self.eps, len(x))
            return torch.from_numpy(x+noise), torch.from_numpy(x)
        else:
            return torch.from_numpy(x), torch.from_numpy(x)

    def collate_fn(self, batch):
        x, y = list(zip(*batch))
        batch_size = len(x)
        input_tensor = torch.empty((batch_size, x[0].size()[0]), dtype=torch.float)
        output_tensor = torch.empty((batch_size, y[0].size()[0]), dtype=torch.float)

        for i in range(batch_size):
            input_tensor[i] = x[i]
            output_tensor[i] = y[i]

        return input_tensor.to(self.device), output_tensor.to(self.device)