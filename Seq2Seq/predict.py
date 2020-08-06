from pathlib import Path

import numpy as np

from system import Seq2SeqSystem, AdversarialSeq2SeqSystem
from dataset import Seq2SeqDataset

import os
test_folder = os.environ.get("TEST_FOLDER", "data/dataset/test")
try:
    system = Seq2SeqSystem.load_from_checkpoint("seq2seq_checkpoint", train_folder=None, test_folder=test_folder).eval().cuda()
except:
    system = AdversarialSeq2SeqSystem.load_from_checkpoint("seq2seq_checkpoint", train_folder=None, test_folder=test_folder).eval().cuda()
dataset = Seq2SeqDataset(Path(test_folder).glob("*001.npz"), previous_poses=20, predicted_poses=50)  # TODO: схерали здесь такие параметры вообще? и *001.npz тоже плохую службу сослужит

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