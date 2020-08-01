# import torch
# import pytorch_lightning as pl
#
# from system import Seq2SeqSystem
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--src', help="folder with .npz X&Y", default="D:/data/GENEA_2020_Data/Dataset/split/test/data_001.npz")
parser.add_argument('--ckpt', help='checkpoint to load', default='lightning_logs/version_4/checkpoints/epoch=46.ckpt')
parser.add_argument('--dst', help='results', default="pred.npy")
#
# if __name__ == "__main__":
#     args = parser.parse_args()
#     system = Seq2SeqSystem(test_folder=args.test)
#     trainer = pl.Trainer(
#         gpus=1 if torch.cuda.is_available() else 0,
#         max_epochs=50,
#     )
#     trainer.fit(system)
#     trainer.save_checkpoint("./seq2seq_checkpoint")

from system import Seq2SeqSystem
from dataset import Seq2SeqDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

system = Seq2SeqSystem()
args = parser.parse_args()
system = system.load_from_checkpoint(args.ckpt).eval().cuda()

from pathlib import Path
dataset = Seq2SeqDataset([args.src], 10, 20)

len(dataset.features)

all_predictions = []
real = []
# x, y, pose =
for sample in dataset:
    x, y, p = sample
    x = x.unsqueeze(1).cuda()
    y = y.unsqueeze(1).cuda()
    p = p.unsqueeze(1).cuda()
    pose = system(x, p)
    all_predictions.append(pose.squeeze(1).detach().cpu().numpy())
    real.append(y.squeeze(1).detach().cpu().numpy())
print(len(all_predictions), len(dataset))

import numpy as np
al = np.concatenate(all_predictions, 0)
real = np.concatenate(real, 0)
print(al.shape, real.shape)
np.save("../pred.npy", al)