from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from model import Encoder, Decoder
from dataset import Seq2SeqDataset, Vocab


class Seq2SeqSystem(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train-folder", type=str, default="data/dataset/train")
        parser.add_argument("--test-folder", type=str, default="data/dataset/test")
        parser.add_argument("--predicted-poses", type=int, default=20)
        parser.add_argument("--previous-poses", type=int, default=10)
        parser.add_argument("--alpha", type=float, default=0.01, help="Continuity loss multiplier.")
        parser.add_argument("--beta", type=float, default=1.0, help="Variance loss multiplier.")
        parser.add_argument("--stride", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=50)
        parser.add_argument("--with_context", action="store_true", default=False)
        parser.add_argument("--embedding", type=str, default=None)
        parser.add_argument("--text_folder", type=str, default=None)
        return parser

    def __init__(
        self,
        train_folder: str,
        test_folder: str,
        alpha: float = 0.01,
        beta: float = 1.0,
        predicted_poses: int = 20,
        previous_poses: int = 10,
        stride: int = None,
        batch_size: int = 50,
        with_context: bool = False,
        embedding: str = None,
        text_folder: str = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(26, 150, 1, with_context)
        self.decoder = Decoder(45, 150, 300, max_gen=predicted_poses)
        self.predicted_poses = predicted_poses
        self.previous_poses = previous_poses
        self.loss = MSELoss()
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.alpha = alpha
        self.beta = beta
        self.stride = predicted_poses if stride is None else stride
        self.batch_size = batch_size
        self.with_context = with_context
        if embedding is not None:
            self.vocab = Vocab(embedding)
            self.word_embedder = nn.Embedding(len(self.vocab.token_to_idx), len(self.vocab.weights[0]),
                                              _weight=torch.FloatTensor(self.vocab.weights))
            self.word_encoder = nn.GRU(len(self.vocab.weights[0]), 100, bidirectional=True)
        else:
            self.vocab = None
        self.text_folder = text_folder

    def forward(self, x, p, w):
        words = self.word_embedder(w)
        words = self.word_encoder(words)
        output, hidden = self.encoder(x)
        predicted_poses = self.decoder(output, hidden, p, words=words)
        return predicted_poses

    def custom_loss(self, output, target):
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)

        n_element = output.numel()
        # MSE
        l1_loss = torch.nn.functional.l1_loss(output, target)

        # continuous motion
        diff = [abs(output[:, n, :] - output[:, n - 1, :]) for n in range(1, output.shape[1])]
        cont_loss = torch.sum(torch.stack(diff)) / n_element
        cont_loss *= self.alpha

        # motion variance
        norm = torch.norm(output, 2, 1)
        var_loss = -torch.sum(norm) / n_element
        var_loss *= self.beta

        loss = l1_loss + cont_loss + var_loss

        return loss

    def calculate_loss(self, p, y):
        return self.custom_loss(p, y)

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
            Path(self.train_folder).glob("*.npz"),
            self.previous_poses,
            self.predicted_poses,
            self.stride,
            self.with_context,
            text_folder=self.text_folder,
            vocab=self.vocab
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )
        return loader

    def val_dataloader(self):
        dataset = Seq2SeqDataset(
            Path(self.test_folder).glob("*.npz"),
            self.previous_poses,
            self.predicted_poses,
            self.stride,
            self.with_context,
            text_folder=self.text_folder,
            vocab=self.vocab
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )
        return loader
