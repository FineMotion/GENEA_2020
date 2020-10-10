from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl

from system import Seq2SeqSystem


def main():
    parser = ArgumentParser()
    parser.add_argument('--serialize-dir', type=str, required=True)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Seq2SeqSystem.add_model_specific_args(parser)
    args = parser.parse_args()
    try:
        Path(args.serialize_dir).mkdir(parents=True)
    except FileExistsError:
        print(f"{args.serialize_dir} already exists, please choose another directory.")
        return
    system = Seq2SeqSystem(**vars(args))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.serialize_dir,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='',
        save_top_k=-1,
        save_last=True,
        period=2
    )
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    trainer.fit(system)


if __name__ == "__main__":
    main()
