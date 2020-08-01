import torch
import pytorch_lightning as pl

from system import Seq2SeqSystem
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--train', help="train folder", default="data/dataset/train")
parser.add_argument('--test', help='test folder', default="data/dataset/test")

if __name__ == "__main__":
    args = parser.parse_args()
    system = Seq2SeqSystem(train_folder=args.train,
                           test_folder=args.test)
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=50,
    )
    trainer.fit(system)
    trainer.save_checkpoint("./seq2seq_checkpoint")
