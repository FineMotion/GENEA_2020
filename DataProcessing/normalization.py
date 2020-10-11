from typing import List
import numpy as np


def create_motion_array(data_files: List[str]) -> np.ndarray:
    result = []
    for data_file in data_files:
        data = np.load(data_file)
        y = data['Y']
        result.append(y)
    return np.concatenate(result, axis=0)


def get_normalization_values(data: np.ndarray):
    max_val = np.amax(np.absolute(data), axis=0)
    mean_pose = data.mean(axis=0)
    return max_val, mean_pose


def normalize_data(data, max_val, mean_pose, eps = 1e-8):
    data_centered = data - mean_pose[np.newaxis, :]
    data_normalized = np.divide(data_centered, max_val[np.newaxis, :] + eps)
    return data_normalized
