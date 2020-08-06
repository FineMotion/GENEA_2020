from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger

from system import Seq2SeqSystem, AdversarialSeq2SeqSystem


def int_or_str(p):
    try:
        return int(p)
    except:
        return p


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--experiment_series', help="used as version in dir name", type=str, default=None)
    parser.add_argument('--experiment_id', help="used as version in dir name", type=int_or_str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Seq2SeqSystem.add_model_specific_args(parser)
    args = parser.parse_args()
    # system = Seq2SeqSystem(**vars(args))
    system = AdversarialSeq2SeqSystem(**vars(args))

    if args.experiment_series is not None:
        import os
        args.default_root_dir = args.default_root_dir or os.getcwd()
        logger = TensorBoardLogger(args.default_root_dir, name=args.experiment_series, version=args.experiment_id)
        args.logger = logger
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(system)
    trainer.save_checkpoint("./seq2seq_checkpoint")
