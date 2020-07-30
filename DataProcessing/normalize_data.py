import numpy as np
from os import listdir, mkdir
from os.path import join, split, exists
import sys

sys.path.append('../tools')
from normalization import create_motion_array, get_normalization_values, normalize_data

if __name__ == '__main__':
    data_filenames = listdir('../data/Ready')
    result_folder = '../data/Normalized'
    if not exists(result_folder):
        mkdir(result_folder)
    data_files = [join('../data/Ready', data_filename) for data_filename in data_filenames]

    train_array = create_motion_array(sorted(data_files)[1:])
    max_val, mean_pose = get_normalization_values(train_array)

    for data_file in data_files:
        _, data_filename = split(data_file)
        print(data_filename)
        data = np.load(data_file)
        motions = data['Y']
        audio = data['X']

        motions_normalized = normalize_data(motions, max_val, mean_pose)
        np.savez(join(result_folder, data_filename), X=audio, Y=motions_normalized)
