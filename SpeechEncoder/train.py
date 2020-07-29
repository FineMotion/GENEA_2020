import torch
from torch.utils.data import RandomSampler, DataLoader
from os import listdir
from os.path import join

from .dataset import MotionDataset
from .model import SpeechMotionModel
from ..tools.trainer import MotionTrainer

if __name__ == '__main__':
    device = torch.device('cuda')
    data_filenames = listdir('../data/Ready')
    data_files = [join('../data/Ready', data_filename) for data_filename in data_filenames]
    print(data_files)

    train_dataset = MotionDataset(data_files=data_files[1:], device=device)
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=256, sampler=train_sampler,
                                collate_fn=train_dataset.collate_fn)

    test_dataset = MotionDataset(data_files=data_files[:1], device=device)
    test_sampler = RandomSampler(test_dataset)
    test_iterator = DataLoader(test_dataset, batch_size=256, sampler=test_sampler,
                               collate_fn=test_dataset.collate_fn)

    model = SpeechMotionModel()
    model.to(device)
    trainer = MotionTrainer(train_iterator, test_iterator, model)
    for epoch in range(500):
        print('Epoch %d' % (epoch + 1))
        trainer.train_epoch()
        trainer.test_epoch()
