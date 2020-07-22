import numpy as np


def get_normalization_values(data: np.ndarray):
    max_val = np.amax(np.absolute(data), axis=0)
    mean_pose = data.mean(axis=0)
    return max_val, mean_pose


def normalize_data(data, max_val, mean_pose, eps = 1e-8):
    data_centered = data - mean_pose[np.newaxis, :]
    data_normalized = np.divide(data_centered, max_val[np.newaxis, :] + eps)
    return data_normalized


# class MotionAutoEncoderDataset