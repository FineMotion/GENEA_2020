from argparse import ArgumentParser

import pytorch_lightning as pl

from system import Seq2SeqSystem


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Seq2SeqSystem.add_model_specific_args(parser)
    args = parser.parse_args()
    system = Seq2SeqSystem(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(system)
    trainer.save_checkpoint("./seq2seq_checkpoint")
