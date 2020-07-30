from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint


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

    def validation_step(self, batch, batch_nb):
        x, y, p = batch
        pred_poses = self.forward(x, y, p)
        loss = self.loss(pred_poses, y)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        d = {"val_loss": 0}
        for out in outputs:
            d['val_loss'] += out['loss']
        return d

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset = Seq2SeqDataset(Path("data/Ready").glob("*.npz"), self.previous_poses, self.predicted_poses)
        loader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=dataset.collate_fn)
        return loader

    def val_dataloader(self):
        dataset = Seq2SeqDataset(Path("data/Ready/dev").glob("*.npz"), self.previous_poses, self.predicted_poses)
        loader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=dataset.collate_fn)
        return loader

if __name__ == "__main__":
    system = Seq2SeqSystem(50, 20)
    checkpoint_callback = ModelCheckpoint(
        filepath="./seq2seq",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    trainer = pl.Trainer(gpus=1, default_root_dir="./seq2seq", checkpoint_callback=checkpoint_callback,
                         max_epochs=50)
    trainer.fit(system)
    trainer.save_checkpoint("./seq2seq_checkpoint")