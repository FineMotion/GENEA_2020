from os import listdir
from os.path import join
from argparse import ArgumentParser
import numpy as np
import sys

sys.path.append('../tools')
from normalization import get_normalization_values, create_motion_array

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src')
    arg_parser.add_argument('--dst')
    arg_parser.add_argument('--raw_dataset', default='../data/Ready')
    args = arg_parser.parse_args()

    data = np.load(args.src)

    data_filenames = listdir(args.raw_dataset)
    data_files = [join(args.raw_dataset, data_filename) for data_filename in data_filenames]
    train_array = create_motion_array(sorted(data_files)[1:])
    max_val, mean_pose = get_normalization_values(train_array)

    eps = 1e-8
    reconstructed = np.multiply(data, max_val[np.newaxis, :] + eps)
    reconstructed = reconstructed + mean_pose[np.newaxis, :]
    print(reconstructed.shape)

    np.save(args.dst, reconstructed)