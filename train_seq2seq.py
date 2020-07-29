from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from src.seq2seq_model import Encoder, Decoder
from src.dataset import Seq2SeqDataset


class Seq2SeqSystem(pl.LightningModule):

    def __init__(self, predicted_poses: int, previous_poses: int):
        super().__init__()
        self.encoder = Encoder(26, 150, 1)
        self.decoder = Decoder(45, 150, 300, max_gen=predicted_poses)
        self.predicted_poses = predicted_poses
        self.previous_poses = previous_poses
        self.loss = MSELoss()

    def forward(self, x, y, p):
        output, hidden = self.encoder(x)
        predicted_poses = self.decoder(output, hidden, p)
        return predicted_poses

    def training_step(self, batch, batch_nb):
        x, y, p = batch
        pred_poses = self.forward(x, y, p)
        loss = self.loss(pred_poses, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset = Seq2SeqDataset(Path("data/Ready").glob("*.npz"), self.previous_poses, self.predicted_poses)
        loader = DataLoader(dataset, batch_size=50, shuffle=False, collate_fn=dataset.collate_fn)
        return loader



system = Seq2SeqSystem(20, 10)
trainer = pl.Trainer(gpus=1)
trainer.fit(system)
