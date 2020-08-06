import sys
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

sys.path.append('../tools')
from normalization import create_motion_array, get_normalization_values, normalize_data

parser = ArgumentParser()
parser.add_argument('--src', default='../data/Ready')
parser.add_argument('--dst', default='../data/Normalized')
parser.add_argument('--dst_mean', default='./mean_pose.npz')


if __name__ == '__main__':
    args = parser.parse_args()
    source_folder = Path(args.src)
    result_folder = Path(args.dst)
    result_folder.mkdir(parents=True, exist_ok=True)
    data_files = sorted(list(source_folder.glob("*npz")))
    train_files = data_files[1:]
    dev_files = data_files[:1]
    assert dev_files[0].name == "data_001.npz"

    train_array = create_motion_array(train_files)
    max_val, mean_pose = get_normalization_values(train_array)

    for data_file in data_files:
        name = data_file.name
        print(name)
        data = np.load(data_file)
        motions = data['Y']
        audio = data['X']
        motions_normalized = normalize_data(motions, max_val, mean_pose)
        np.savez(result_folder / name, X=audio, Y=motions_normalized)
    print(f"Saving mean into {args.dst_mean}")
    np.savez(args.dst_mean, max_val=max_val, mean_pose=mean_pose)