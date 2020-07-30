import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from system import Seq2SeqSystem


if __name__ == "__main__":
    system = Seq2SeqSystem()
    checkpoint_callback = ModelCheckpoint(
        filepath="./seq2seq",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )
    trainer = pl.Trainer(
        gpus= 1 if torch.cuda.is_available() else 0,
        default_root_dir="./seq2seq",
        checkpoint_callback=checkpoint_callback,
        max_epochs=50,
    )
    trainer.fit(system)
    trainer.save_checkpoint("./seq2seq_checkpoint")
