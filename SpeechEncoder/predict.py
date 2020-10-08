import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import join

from datasets import SpeechMotionDataset
from model import SpeechMotionModel
from scipy.signal import savgol_filter


def smoothing(motion):

    smoothed = [savgol_filter(motion[:,i], 9, 3) for i in range(motion.shape[1])]
    new_motion = np.array(smoothed).transpose()
    return new_motion


if __name__ == '__main__':
    device = torch.device('cuda')
    data_filenames = listdir('../data/Encoded_dae')
    data_files = [join('../data/Encoded_dae', data_filename) for data_filename in data_filenames]
    print(data_files)

    test_dataset = SpeechMotionDataset(data_files=data_files[:1], device=device)
    test_sampler = SequentialSampler(test_dataset)
    test_iterator = DataLoader(test_dataset, batch_size=256, sampler=test_sampler,
                               collate_fn=test_dataset.collate_fn)

    model = SpeechMotionModel()
    model.load_state_dict(torch.load('results/dae_tanh/speech_encoder.pt'))
    model.to(device)

    model.eval()

    result = []
    for features, _ in tqdm(test_iterator):
        predict = model(features)
        result.append(predict.detach().cpu().numpy())

    result = np.concatenate(result, axis=0)
    print(result.shape)
    result = smoothing(result)
    print(result.shape)
    np.save('results/dae_tanh/predict.npy', result)