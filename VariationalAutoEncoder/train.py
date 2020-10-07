import torch
from torch.utils.data import RandomSampler, DataLoader
from os import listdir
from os.path import join
from vae import VariationalAutoEncoder, VAELoss

import sys
sys.path.append('../tools')
from trainer import MotionTrainer
from normalization import create_motion_array
from datasets import  MotionDataset

if __name__ == "__main__":
    device = torch.device('cuda')
    data_filenames = listdir('../data/Normalized')
    data_files = [join('../data/Normalized', data_filename) for data_filename in data_filenames]
    train_array = create_motion_array(data_files[1:])
    test_array = create_motion_array(data_files[:1])

    train_dataset = MotionDataset(train_array, device, sigma=None, add_noise=False)
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=256, sampler=train_sampler,
                                collate_fn=train_dataset.collate_fn)

    test_dataset = MotionDataset(test_array, device, sigma=None, add_noise=False)
    test_sampler = RandomSampler(test_dataset)
    test_iterator = DataLoader(test_dataset, batch_size=256, sampler=test_sampler,
                               collate_fn=test_dataset.collate_fn)
    model = VariationalAutoEncoder()
    model.to(device)

    trainer = MotionTrainer(train_iterator, test_iterator, model, 'vae.pt', criterion=VAELoss())
    trainer.train(100, 20)
    torch.save(trainer.model.state_dict(), 'vae_train.pt')
