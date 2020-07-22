import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math


class MotionTrainer:
    def __init__(self, train_iterator: DataLoader, test_iterator: DataLoader, model: nn.Module):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999))
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.criterion = nn.MSELoss()
        self.best_loss = math.inf

    def train_epoch(self):
        total_loss = 0
        self.model.train()
        for batch_idx, (features, labels) in enumerate(self.train_iterator):
            self.optimizer.zero_grad()
            predict = self.model(features)
            loss = self.criterion(predict, labels)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            print('\rBatch: %d of %d\tLoss: %4f' % (batch_idx + 1, len(self.train_iterator), total_loss / (batch_idx + 1)),
                  end='')
        print()

    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        for batch_idx, (features, labels) in enumerate(self.test_iterator):
            predict = self.model(features)
            loss = self.criterion(predict, labels)
            total_loss += loss.item()
            print('\rBatch: %d of %d\tLoss: %4f' % (batch_idx + 1, len(self.test_iterator), total_loss / (batch_idx + 1)),
                  end='')
        print()
        loss = total_loss / len(self.test_iterator)
        if loss < self.best_loss:
            print('New best loss on test: %4f' % loss)
            self.best_loss = loss
            torch.save(self.model.state_dict(), 'best.pt')

