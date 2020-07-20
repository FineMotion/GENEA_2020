from os import listdir
from os.path import join

import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from src.dataset import MotionDataset
from src.trainer import MotionTrainer
from src.model import SpeechMotionModel


if __name__ == '__main__':
    device = torch.device('cuda')
    data_filenames = listdir('data/Ready')
    data_files = [join('data/Ready', data_filename) for data_filename in data_filenames]
    dataset = MotionDataset(data_files=data_files, device=device)
    sampler = RandomSampler(dataset)
    iterator = DataLoader(dataset, batch_size=256, sampler=sampler, collate_fn=dataset.collate_fn)
    model = SpeechMotionModel()
    model.to(device)
    trainer = MotionTrainer(iterator, model)
    for _ in range(50):
        trainer.train()
