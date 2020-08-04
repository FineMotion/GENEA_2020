import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from os import listdir
from os.path import join

from vae import VariationalAutoEncoder

import sys
sys.path.append('../tools')

from normalization import get_normalization_values, create_motion_array
from datasets import MotionDataset

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src')
    arg_parser.add_argument('--dst')
    args = arg_parser.parse_args()

    device = torch.device('cuda')

    data = np.load(args.src)
    model = VariationalAutoEncoder()
    model.load_state_dict(torch.load('vae_train.pt'))
    model.to(device)

    dataset = MotionDataset(data, device, sigma=None, add_noise=False)
    sampler = SequentialSampler(dataset)
    iterator = DataLoader(dataset, batch_size=256, sampler=sampler, collate_fn=dataset.collate_fn)

    result = []
    for features, _ in tqdm(iterator):
        predict = model.decode(features)
        result.append(predict.detach().cpu().numpy())
    result = np.concatenate(result, axis=0)
    print(result.shape)

    data_filenames = listdir('../data/Ready')
    data_files = [join('../data/Ready', data_filename) for data_filename in data_filenames]
    train_array = create_motion_array(data_files[1:])
    max_val, mean_pose = get_normalization_values(train_array)

    eps = 1e-8
    reconstructed = np.multiply(result, max_val[np.newaxis, :] + eps)
    reconstructed = reconstructed + mean_pose[np.newaxis, :]
    print(reconstructed.shape)

    np.save(args.dst, reconstructed)

