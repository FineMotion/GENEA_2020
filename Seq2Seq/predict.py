from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from system import Seq2SeqSystem
from dataset import Seq2SeqDataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--src", type=str, default=None)
    args = parser.parse_args()
    system = Seq2SeqSystem.load_from_checkpoint(args.checkpoint)
    system = system.eval().cuda()
    dataset = Seq2SeqDataset(Path("data/dataset/test").glob("*001.npz") if args.src is None else [Path(args.src)],
                             previous_poses=system.previous_poses,
                             predicted_poses=system.predicted_poses,
                             stride=system.predicted_poses,
                             with_context=system.with_context)
    prev_poses = system.predicted_poses
    pred_poses = system.previous_poses

    all_predictions = []
    dataset_iter = iter(dataset)

    with torch.no_grad():
        x, _, p = next(dataset_iter)
        x = x.unsqueeze(1).cuda()
        p = p.unsqueeze(1).cuda()
        pose = system(x, p)
        all_predictions.append(pose.squeeze(1).detach().cpu().numpy())

        for sample in tqdm(dataset_iter):
            x, _, p = sample
            x = x.unsqueeze(1).cuda()
            pose = system(x, pose[-pred_poses:])
            all_predictions.append(pose.squeeze(1).detach().cpu().numpy())

    al = np.concatenate(all_predictions, 0)
    print(al.shape)
    print(len(dataset))
    np.save(args.dest, al)