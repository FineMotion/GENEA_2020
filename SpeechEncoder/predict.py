import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import join

from .dataset import MotionDataset
from .model import SpeechMotionModel

if __name__ == '__main__':
    device = torch.device('cuda')
    data_filenames = listdir('../data/Ready')
    data_files = [join('../data/Ready', data_filename) for data_filename in data_filenames]
    print(data_files)

    test_dataset = MotionDataset(data_files=data_files[:1], device=device)
    test_sampler = SequentialSampler(test_dataset)
    test_iterator = DataLoader(test_dataset, batch_size=256, sampler=test_sampler,
                               collate_fn=test_dataset.collate_fn)

    model = SpeechMotionModel()
    model.load_state_dict(torch.load('best.pt'))
    model.to(device)

    model.eval()

    result = []
    for features, _ in tqdm(test_iterator):
        predict = model(features)
        result.append(predict.detach().cpu().numpy())

    result = np.concatenate(result, axis=0)
    print(result.shape)
    np.save('predict.npy', result)