from collections import OrderedDict
from pathlib import Path
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from model import Encoder, Decoder, Discriminator
from dataset import Seq2SeqDataset


class Seq2SeqSystem(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train-folder", type=str, default="data/dataset/train")
        parser.add_argument("--test-folder", type=str, default="data/dataset/test")
        parser.add_argument("--predicted-poses", type=int, default=20)
        parser.add_argument("--previous-poses", type=int, default=10)
        return parser

    def __init__(
        self,
        train_folder: str,
        test_folder: str,
        predicted_poses: int = 20,
        previous_poses: int = 10,
        *args,
        **kwargs
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
    
    
class AdversarialSeq2SeqSystem(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train-folder", type=str,
                            default="data/dataset/train")
        parser.add_argument("--test-folder", type=str,
                            default="data/dataset/test")
        parser.add_argument("--predicted-poses", type=int, default=20)
        parser.add_argument("--previous-poses", type=int, default=10)

        parser.add_argument("--w_cont_loss", type=float, default=0.01)
        parser.add_argument("--w_adv_loss", type=float, default=0.01)
        return parser

    def __init__(
            self,
            train_folder: str = "data/dataset/train",
            test_folder: str = "data/dataset/test",
            predicted_poses: int = 20,
            previous_poses: int = 10,
            w_cont_loss: float = 0.01,
            w_adv_loss: float = 0.01,
            **kwargs
    ):
        super().__init__()
        self.hparams = Namespace(**{
            'prev_poses': previous_poses,
            'pred_poses': predicted_poses,
            'lr': 1e-3,
            'beta1': 0.5,
            'beta2': 0.9,
            'w_cont_loss': w_cont_loss,
            'w_adv_loss': w_adv_loss
        })
        self.encoder = Encoder(26, 150, 1)
        self.decoder = Decoder(45, 150, 300, max_gen=predicted_poses)
        self.discriminator = Discriminator(45, ch_hid=100)
        self.predicted_poses = predicted_poses
        self.previous_poses = previous_poses
        self.base_loss = MSELoss()
        self.train_folder = train_folder
        self.test_folder = test_folder

    def forward(self, x, p):
        output, hidden = self.encoder(x)
        predicted_poses = self.decoder(output, hidden, p)
        return predicted_poses

    def calculate_loss(self, p, y):
        base_loss = self.base_loss(p, y)
        cont_loss = torch.norm(p[1:] - p[:-1]) / (p.size(0) - 1)
        loss = base_loss + cont_loss * self.hparams.w_cont_loss
        return loss

    def training_step(self, batch, batch_nb, optimizer_idx):
        audio_features, real_poses, prev_poses = batch
        pred_poses = self.forward(audio_features, prev_poses)
        batch_size = pred_poses.size(1)

        # train generator
        if optimizer_idx == 0:
            base_loss = self.base_loss(pred_poses, real_poses)
            cont_loss = torch.norm(pred_poses[1:] - pred_poses[:-1]) / (pred_poses.size(0) - 1)

            is_real = torch.ones((batch_size, 1)).to(pred_poses.device)
            adv_loss = F.binary_cross_entropy_with_logits(self.discriminator(pred_poses), is_real)

            loss = (base_loss + self.hparams.w_cont_loss * cont_loss
                    + self.hparams.w_adv_loss * adv_loss)
            # aux metrics
            d_fake_score = F.sigmoid(self.discriminator(pred_poses)).mean()
            logs = {
                'loss': loss,
                # loss components
                'g_adv_loss': adv_loss,
                'base_loss': base_loss,
                'cont_loss': cont_loss,
                # metrics
                'd_fake_score': d_fake_score
            }
            logs = {f'{k}/train': v for k, v in logs.items()}

            return OrderedDict({
                'loss': loss,
                'progress_bar': {'total_loss': loss, 'd_fake_score': d_fake_score},
                'log': logs
            })

        # train discriminator
        if optimizer_idx == 1:
            d_real_scores = self.discriminator(real_poses)
            d_fake_scores = self.discriminator(pred_poses.detach())

            is_real = torch.ones((batch_size, 1)).to(pred_poses.device)
            real_loss = F.binary_cross_entropy_with_logits(d_real_scores, is_real)
            fake_loss = F.binary_cross_entropy_with_logits(d_fake_scores, 1 - is_real)

            d_loss = (real_loss + fake_loss) / 2
            # aux metrics
            d_fake_scores = F.sigmoid(d_fake_scores).mean()
            d_real_scores = F.sigmoid(d_real_scores).mean()

            logs = {
                'loss': d_loss,
                'd_real_loss': real_loss,
                'd_fake_loss': fake_loss,
                'd_fake_score': d_fake_scores,
                'd_real_score': d_real_scores
            }
            logs = {f'{k}/train': v for k, v in logs.items()}

            return OrderedDict({
                'loss': d_loss,
                'progress_bar': {'d_loss': d_loss, 'd_fake_score': d_fake_scores, 'd_real_score': d_real_scores},
                'log': logs
            })

    def validation_step(self, batch, batch_nb):
        audio_features, real_poses, prev_poses = batch
        pred_poses = self.forward(audio_features, prev_poses)
        batch_size = pred_poses.size(1)

        base_loss = self.base_loss(pred_poses, real_poses)
        cont_loss = torch.norm(pred_poses[1:] - pred_poses[:-1]) / (
                    pred_poses.size(0) - 1)

        is_real = torch.ones((batch_size, 1)).to(pred_poses.device)
        adv_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(pred_poses), is_real)

        loss = (base_loss + self.hparams.w_cont_loss * cont_loss
                + self.hparams.w_adv_loss * adv_loss)
        # aux metrics
        d_fake_score = F.sigmoid(self.discriminator(pred_poses)).mean()
        logs = {
            'loss': loss,
            # loss components
            'g_adv_loss': adv_loss,
            'base_loss': base_loss,
            'cont_loss': cont_loss,
            # metrics
            'd_fake_score': d_fake_score,
            'd_real_score': F.sigmoid(self.discriminator(real_poses)).mean()
        }
        pbar = logs.copy()
        logs = {f'{k}/valid': v for k, v in logs.items()}
        return OrderedDict({
            'loss': loss,
            'progress_bar': pbar,
            'log': logs
        })

    def validation_epoch_end(self, outputs):
        assert len(outputs) > 0
        d = {}
        for i in range(len(outputs)):
            for key, value in outputs[i]["log"].items():
                if key not in d:
                    d[key] = []
                d[key].append(value)
        for key in d:
            if len(d[key]) == 1:
                d[key] = d[key][0]
            else:
                d[key] = torch.stack(d[key], 0).mean()  # на самом деле нифига подобного если последний батч неполный, но мне лень
        return {"log": d}

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2

        opt_g = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

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
            dataset, batch_size=50, shuffle=False, collate_fn=dataset.collate_fn
        )
        return loader
