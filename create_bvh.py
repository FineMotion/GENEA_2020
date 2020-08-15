import logging
from argparse import ArgumentParser
from os import mkdir
from os.path import exists
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import sys
sys.path.append('./DataProcessing')
from DataProcessing.reconstruct_data import load_mean, denormalize
from DataProcessing.process_motions import create_bvh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def smoothing(motion):

    smoothed = [savgol_filter(motion[:,i], 9, 3) for i in range(motion.shape[1])]
    new_motion = np.array(smoothed).transpose()
    return new_motion


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pred",
        "--prediction",
        type=str,
        required=True,
        help="directory with .npy files with predictions of " "shape (N x 45)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="directory to save results",
    )
    parser.add_argument(
        "--mean",
        type=str,
        default="DataProcessing/mean_pose.npz",
        help="File with normalization values.",
    )
    parser.add_argument(
        "--pipe",
        type=str,
        default="pipe",
        help="pipe folder with pre/post processing."
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=False,
        help="Flag to apply smoothing."
        )
    args = parser.parse_args()
    if not exists(args.dest):
        mkdir(args.dest)
    for pred_file in Path(args.pred).glob('*.npy'):
        logging.info(str(pred_file))
        prediction = np.load(str(pred_file))
        if args.smooth:
            logger.info("Smoothing prediction")
            prediction = smoothing(prediction)

        logging.info("Reconstructing data by denormalizing it.")
        max_val, mean_pose = load_mean(args.mean)
        prediction = denormalize(prediction, max_val, mean_pose)

        logging.info("Creating .bvh. This requires pipe")

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            np.save(tmpdir / pred_file.name, prediction)
            create_bvh(tmpdir / pred_file.name, args.dest, args.pipe)
