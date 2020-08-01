#!/usr/bin/env bash

source activate genea_challenge

cd DataProcessing
python reconstruct_data.py --src pred.npy --dst reconstructed.npy --raw_dataset D:/data/GENEA_2020_Data/Dataset/AlignedWithContext
python process_motions.py --src reconstructed.npy --dst . --bvh
python cut_bvh.py --src reconstructed.bvh --dst cut.bvh
cd ..
python visualization_example.py DataProcessing/cut.bvh