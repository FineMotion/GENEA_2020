from pathlib import Path

import numpy as np

from system import Seq2SeqSystem
from dataset import Seq2SeqDataset


system = Seq2SeqSystem.load_from_checkpoint("seq2seq_checkpoint", train_folder=None, test_folder="data/dataset/test").eval().cuda()
dataset = Seq2SeqDataset(Path("data/dataset/test").glob("*001.npz"), previous_poses=20, predicted_poses=50)

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
np.save("pred.npy", al)