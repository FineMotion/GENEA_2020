import torch
from torch.utils.data import RandomSampler, DataLoader
from os import listdir
from os.path import join
import sys
from datasets import SpeechMotionDataset
from model import SpeechMotionModel

sys.path.append('../tools')
from trainer import MotionTrainer

if __name__ == '__main__':
    device = torch.device('cuda')
    data_filenames = listdir('../data/Encoded_dae')
    data_files = [join('../data/Encoded_dae', data_filename) for data_filename in data_filenames]
    print(data_files)

    train_dataset = SpeechMotionDataset(data_files=data_files[1:], device=device)
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=256, sampler=train_sampler,
                                collate_fn=train_dataset.collate_fn)

    test_dataset = SpeechMotionDataset(data_files=data_files[:1], device=device)
    test_sampler = RandomSampler(test_dataset)
    test_iterator = DataLoader(test_dataset, batch_size=256, sampler=test_sampler,
                               collate_fn=test_dataset.collate_fn)

    model = SpeechMotionModel()
    model.to(device)
    trainer = MotionTrainer(train_iterator, test_iterator, model, 'results/dae_tanh/speech_encoder.pt')
    for epoch in range(100):
        print('Epoch %d' % (epoch + 1))
        trainer.train_epoch()
    torch.save(trainer.model.state_dict(), 'results/dae_tanh/speech_encoder.pt')
    # trainer.train(500, 50)
