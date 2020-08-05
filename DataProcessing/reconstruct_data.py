from argparse import ArgumentParser

import numpy as np


def load_mean(filename: str):
    mean_pose = np.load(filename)
    return mean_pose["max_val"], mean_pose["mean_pose"]


def denormalize(data, max_val, mean_pose):
    eps = 1e-8
    reconstructed = np.multiply(data, max_val[np.newaxis, :] + eps)
    reconstructed = reconstructed + mean_pose[np.newaxis, :]
    return reconstructed


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
    max_val, mean_pose = load_mean(args.mean)
    reconstructed = denormalize(data, max_val, mean_pose)
    print(reconstructed.shape)
    np.save(args.dst, reconstructed)
