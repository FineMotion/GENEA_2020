import numpy as np
from pathlib import Path
from normalization import create_motion_array, get_normalization_values, normalize_data
from argparse import ArgumentParser

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--src', type=str, help='Path to the folder with aligned data')
    argparser.add_argument('--dst', type=str, help='Path to the folder where normalized data will be stored')
    argparser.add_argument('--values', type=str, default="./mean_pose.npz",
                           help='Path to the npz-file where normalizing values will be stored')
    args = argparser.parse_args()

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
    print("Saving mean into %s" % args.values)
    np.savez(args.values, max_val=max_val, mean_pose=mean_pose)