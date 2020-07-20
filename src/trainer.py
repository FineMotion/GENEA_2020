import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MotionTrainer:
    def __init__(self, iterator: DataLoader, model: nn.Module):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())
        self.iterator = iterator
        self.loss = nn.MSELoss()

    def train(self):
        total_loss = 0
        self.model.train()
        for batch_idx, (features, labels) in enumerate(self.iterator):
            self.optimizer.zero_grad()
            predict = self.model(features)
            batch_loss = self.loss(predict, labels)
            batch_loss.backward()
            total_loss += batch_loss.item()
            self.optimizer.step()
            print('\rBatch: %d of %d\tLoss: %4f' % (batch_idx + 1, len(self.iterator), total_loss / (batch_idx + 1)),
                  end='')
        print()