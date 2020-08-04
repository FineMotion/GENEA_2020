import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import join, split, splitext

from vae import VariationalAutoEncoder

import sys
sys.path.append('../tools')
from tools.normalization import create_motion_array
from datasets import MotionDataset

if __name__ == '__main__':
    device = torch.device('cuda')
    data_filenames = listdir('../data/Normalized')
    data_files = [join('../data/Normalized', data_filename) for data_filename in data_filenames]

    model = VariationalAutoEncoder()
    model.load_state_dict(torch.load('vae.pt'))
    model.to(device)

    for data_file in data_files:
        _, data_filename = split(data_file)
        data_name, _ = splitext(data_filename)
        print(data_name)

        data_array = create_motion_array([data_file])
        dataset = MotionDataset(data_array, device, sigma=None, add_noise=False)
        sampler = SequentialSampler(dataset)
        iterator = DataLoader(dataset, batch_size=256, sampler=sampler, collate_fn=dataset.collate_fn)

        result = []
        for features, _ in tqdm(iterator):
            mu, sigma = model.encode(features)
            z = model.reparametrize(mu, sigma)
            result.append(z.detach().cpu().numpy())
        result = np.concatenate(result, axis=0)
        print(result.shape)
        np.save(join('Encoded', data_name + '.npy'), result)