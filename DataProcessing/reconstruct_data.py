from argparse import ArgumentParser

import numpy as np


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--src", help=".npy file with prediction")
    arg_parser.add_argument(
        "--mean", default="./mean_pose.npz", help="File with normalization statistics."
    )
    arg_parser.add_argument(
        "--dst",
        default="./pred_unnormalized.npy",
        help="File to write denormalized result.",
    )
    args = arg_parser.parse_args()

    data = np.load(args.src)
    mean_pose = np.load(args.mean)
    max_val = mean_pose["max_val"]
    mean_pose = mean_pose["mean_pose"]
    eps = 1e-8
    reconstructed = np.multiply(data, max_val[np.newaxis, :] + eps)
    reconstructed = reconstructed + mean_pose[np.newaxis, :]
    print(reconstructed.shape)
    np.save(args.dst, reconstructed)
