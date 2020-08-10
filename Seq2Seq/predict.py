from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from scipy.signal import savgol_filter

from system import Seq2SeqSystem, AdversarialSeq2SeqSystem
from dataset import Seq2SeqDataset

def smoothing(motion):

    smoothed = [savgol_filter(motion[:,i], 9, 3) for i in range(motion.shape[1])]
    new_motion = np.array(smoothed).transpose()
    return new_motion

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    args = parser.parse_args()
    system = AdversarialSeq2SeqSystem.load_from_checkpoint(args.checkpoint, train_folder=None, test_folder="data/dataset/test")
    system = system.eval().cuda()
    dataset = Seq2SeqDataset(Path("data/dataset/test").glob("*001.npz"),
                             previous_poses=system.previous_poses,
                             predicted_poses=system.predicted_poses)
    prev_poses = system.predicted_poses
    pred_poses = system.previous_poses

    all_predictions = []
    dataset_iter = iter(dataset)

    x, y, p = next(dataset_iter)
    x = x.unsqueeze(1).cuda()
    p = p.unsqueeze(1).cuda()
    pose = system(x, p)
    all_predictions.append(pose.squeeze(1).detach().cpu().numpy())

    for sample in dataset_iter:
        x, _, p = sample
        x = x.unsqueeze(1).cuda()
        pose = system(x, pose[-pred_poses:])
        all_predictions.append(pose.squeeze(1).detach().cpu().numpy())

    al = np.concatenate(all_predictions, 0)
    al = smoothing(al)
    np.save(args.dest, al)