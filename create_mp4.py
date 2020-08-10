import logging
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


from DataProcessing.reconstruct_data import load_mean, denormalize
from DataProcessing.process_motions import create_bvh
from DataProcessing.cut_bvh import main as cut_bvh
from DataProcessing.visualization_example import main as get_mp4
from DataProcessing.create_clip import create_clip

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
        help=".npy file with predictions of " "shape (N x 45)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="output.mp4",
        help="mp4 file name to save prediction to.",
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
    parser.add_argument(
        "--audio",
        type=str,
        help="path to original audio file"
    )
    args = parser.parse_args()
    prediction = np.load(args.pred)
    if args.smooth:
        logger.info("Smoothing prediction")
        prediction = smoothing(prediction)

    logging.info("Reconstructing data by denormalizing it.")
    max_val, mean_pose = load_mean(args.mean)
    prediction = denormalize(prediction, max_val, mean_pose)

    logging.info("Creating .bvh. This requires pipe")
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        np.save(tmpdir / "pred.npy", prediction)
        create_bvh(tmpdir / "pred.npy", tmpdir, args.pipe)
        assert (tmpdir / "pred.bvh").exists()

        logging.info("Cutting bvh.")
        cut_bvh(tmpdir / "pred.bvh", tmpdir / "cut_pred.bvh")

        logging.info("Sending to visualize")
        get_mp4(tmpdir / "cut_pred.bvh", tmpdir / "video.mp4")

        logging.info("Merging video and audio")
        create_clip(tmpdir / "video.mp4", Path(args.audio), Path(args.dest))