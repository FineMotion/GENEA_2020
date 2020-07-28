from os import listdir
from os.path import join, split, splitext
from typing import List
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm

from src.dae import get_normalization_values, normalize_data, NoisedMotionDataset, DenoisingAutoEncoder
import numpy as np
from argparse import ArgumentParser


def create_motion_array(data_files: List[str]) -> np.ndarray:
    result = []
    for data_file in data_files:
        data = np.load(data_file)
        y = data['Y']
        result.append(y)
    return np.concatenate(result, axis=0)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src')
    arg_parser.add_argument('--dst')
    args = arg_parser.parse_args()

    device = torch.device('cuda')

    data = np.load(args.src)
    model = DenoisingAutoEncoder()
    model.load_state_dict(torch.load('DenoisingAutoEncoder.pt'))
    model.to(device)

    dataset = NoisedMotionDataset(data, device, sigma=None, train=False)
    sampler = SequentialSampler(dataset)
    iterator = DataLoader(dataset, batch_size=256, sampler=sampler, collate_fn=dataset.collate_fn)

    result = []
    for features, _ in tqdm(iterator):
        predict = model.decoder(features)
        result.append(predict.detach().cpu().numpy())
    result = np.concatenate(result, axis=0)
    print(result.shape)

    data_filenames = listdir('../data/Ready')
    data_files = [join('../data/Ready', data_filename) for data_filename in data_filenames]
    train_array = create_motion_array(data_files[1:])
    max_val, mean_pose = get_normalization_values(train_array)

    eps = 1e-15
    reconstructed = np.multiply(result, max_val[np.newaxis, :] + eps)
    reconstructed = reconstructed + mean_pose[np.newaxis, :]
    print(reconstructed.shape)

    np.save(args.dst, reconstructed)

