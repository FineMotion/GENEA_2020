from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from model import Encoder, Decoder
from dataset import Seq2SeqDataset


class Seq2SeqSystem(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train-folder", type=str)
        parser.add_argument("--test-folder", type=str)
        parser.add_argument("--predicted-poses", type=int, default=20)
        parser.add_argument("--previous_poses", type=int, default=10)
        return parser

    def __init__(
        self,
        train_folder: str = "data/dataset/train",
        test_folder: str = "data/dataset/test",
        predicted_poses: int = 20,
        previous_poses: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(26, 150, 1)
        self.decoder = Decoder(45, 150, 300, max_gen=predicted_poses)
        self.predicted_poses = predicted_poses
        self.previous_poses = previous_poses
        self.loss = MSELoss()
        self.train_folder = train_folder
        self.test_folder = test_folder

    def forward(self, x, p):
        output, hidden = self.encoder(x)
        predicted_poses = self.decoder(output, hidden, p)
        return predicted_poses

    def calculate_loss(self, p, y):
        mse_loss = self.loss(p, y)
        cont_loss = torch.norm(p[1:] - p[:-1]) / (p.size(0) - 1)
        loss = mse_loss + cont_loss * 0.01
        return loss

    def training_step(self, batch, batch_nb):
        x, y, p = batch
        pred_poses = self.forward(x, p)
        loss = self.calculate_loss(pred_poses, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y, p = batch
        pred_poses = self.forward(x, p)
        loss = self.calculate_loss(pred_poses, y)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        d = {"val_loss": 0}
        for out in outputs:
            d["val_loss"] += out["loss"]
        return d

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset = Seq2SeqDataset(
            Path(self.train_folder).glob("*.npz"), self.previous_poses, self.predicted_poses
        )
        loader = DataLoader(
            dataset, batch_size=50, shuffle=True, collate_fn=dataset.collate_fn
        )
        return loader

    def val_dataloader(self):
        dataset = Seq2SeqDataset(
            Path(self.test_folder).glob("*.npz"),
            self.previous_poses,
            self.predicted_poses,
        )
        loader = DataLoader(
            dataset, batch_size=50, shuffle=True, collate_fn=dataset.collate_fn
        )
        return loader
