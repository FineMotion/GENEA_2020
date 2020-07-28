import torch
from os import listdir
from os.path import join
from typing import List

import numpy as np
from torch.utils.data import RandomSampler, DataLoader

from src.dae import get_normalization_values, normalize_data, NoisedMotionDataset, DenoisingAutoEncoder
from src.trainer import MotionTrainer


def create_motion_array(data_files: List[str]) -> np.ndarray:
    result = []
    for data_file in data_files:
        data = np.load(data_file)
        y = data['Y']
        result.append(y)
    return np.concatenate(result, axis=0)


if __name__ == "__main__":
    device = torch.device('cuda')
    data_filenames = listdir('../data/Ready')
    data_files = [join('../data/Ready', data_filename) for data_filename in data_filenames]
    train_array = create_motion_array(data_files[1:])
    test_array = create_motion_array(data_files[:1])

    max_val, mean_pose = get_normalization_values(train_array)
    train_normalized = normalize_data(train_array, max_val, mean_pose)
    test_normalized = normalize_data(test_array, max_val, mean_pose)

    sigma = np.std(train_normalized, axis=(0, 1))
    print(sigma)
    exit(1)
    train_dataset = NoisedMotionDataset(train_normalized, device, sigma)
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=256, sampler=train_sampler,
                                collate_fn=train_dataset.collate_fn)

    test_dataset = NoisedMotionDataset(train_normalized, device, sigma)
    test_sampler = RandomSampler(test_dataset)
    test_iterator = DataLoader(test_dataset, batch_size=256, sampler=test_sampler,
                               collate_fn=test_dataset.collate_fn)
    model = DenoisingAutoEncoder()
    model.to(device)

    trainer = MotionTrainer(train_iterator, test_iterator, model, 'DenoisingAutoEncoder.pt')
    for epoch in range(100):
        print('Epoch %d' % (epoch + 1))
        trainer.train_epoch()
        trainer.test_epoch()