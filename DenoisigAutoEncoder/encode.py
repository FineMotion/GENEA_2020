import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import join, split, splitext

from .dataset import NoisedMotionDataset
from .model import DenoisingAutoEncoder
from ..tools.normalization import get_normalization_values, normalize_data, create_motion_array


if __name__ == '__main__':
    device = torch.device('cuda')
    data_filenames = listdir('../data/Ready')
    data_files = [join('../data/Ready', data_filename) for data_filename in data_filenames]
    train_array = create_motion_array(data_files[1:])

    max_val, mean_pose = get_normalization_values(train_array)

    model = DenoisingAutoEncoder()
    model.load_state_dict(torch.load('DenoisingAutoEncoder.pt'))
    model.to(device)

    for data_file in data_files:
        _, data_filename = split(data_file)
        data_name, _ = splitext(data_filename)
        print(data_name)

        data_array = create_motion_array([data_file])
        data_normalized = normalize_data(data_array, max_val, mean_pose)
        dataset = NoisedMotionDataset(data_normalized, device, sigma=None, train=False)
        sampler = SequentialSampler(dataset)
        iterator = DataLoader(dataset, batch_size=256, sampler=sampler, collate_fn=dataset.collate_fn)

        result = []
        for features, _ in tqdm(iterator):
            predict = model.encoder(features)
            result.append(predict.detach().cpu().numpy())
        result = np.concatenate(result, axis=0)
        print(result.shape)
        np.save(join('Encoded', data_name + '.npy'), result)