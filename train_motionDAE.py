from os import listdir
from os.path import join
from typing import List

import numpy as np
from src.dae import get_normalization_values, normalize_data


def create_motion_array(data_files: List[str]) -> np.ndarray:
    result = []
    for data_file in data_files:
        data = np.load(data_file)
        y = data['Y']
        result.append(y)
    return np.concatenate(result, axis=0)


if __name__ == "__main__":
    data_filenames = listdir('data/Ready')
    data_files = [join('data/Ready', data_filename) for data_filename in data_filenames]
    train_array = create_motion_array(data_files[1:])
    test_array = create_motion_array(data_files[:1])

    max_val, mean_pose = get_normalization_values(train_array)
    train_normalized = normalize_data(train_array, max_val, mean_pose)
    test_normalized = normalize_data(test_array, max_val, mean_pose)

