import numpy as np
import logging
from argparse import ArgumentParser
from os import mkdir, listdir
from os.path import exists, splitext, join

from .audio_utils import calculate_mfcc


def process_folder(src_dir: str, dst_dir: str):
    if not exists(dst_dir):
        mkdir(dst_dir)

    for audio in listdir(src_dir):
        recording_name, _ = splitext(audio)
        mfccs = calculate_mfcc(join(src_dir, audio))
        logging.info(f"{recording_name}:{mfccs.shape}")
        np.save(join(dst_dir, recording_name + '.npy'), mfccs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src_dir', help='Path to recorded speech folder')
    arg_parser.add_argument('--dst_dir', help='Path where extracted audio features will be stored')

    args = arg_parser.parse_args()
    process_folder(args.src_dir, args.dst_dir)
