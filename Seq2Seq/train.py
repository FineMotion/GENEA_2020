import torch
import pytorch_lightning as pl

from system import Seq2SeqSystem


if __name__ == "__main__":
    system = Seq2SeqSystem()
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=50,
    )
    trainer.fit(system)
    trainer.save_checkpoint("./seq2seq_checkpoint")
