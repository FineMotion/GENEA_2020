#!/usr/bin/env bash

DATA="D:/data/GENEA_2020_Data/Dataset/Normalized"
TRAIN="D:/data/GENEA_2020_Data/Dataset/split/train"
TEST="D:/data/GENEA_2020_Data/Dataset/split/test"

BASELOGDIR="Seq2Seq/lightning_logs"

source activate genea_challenge

EXP_SERIES="abl_epochs"  # TODO: rename it before EVERY run of this script
#EXP_ID="ep_$EPOCHS"
for EPOCHS in 1 5 10 20 50 100 200 400 700 1000; do
    EXP_ID="ep_$EPOCHS"
    echo $EXP_ID
    LOGDIR="$BASELOGDIR/$EXP_SERIES/$EXP_ID"
    python Seq2Seq/train.py --train-folder $TRAIN --test-folder $TEST --default_root_dir $BASELOGDIR \
            --experiment_series $EXP_SERIES --experiment_id $EXP_ID --max_epochs $EPOCHS
    TEST_FOLDER=$TEST python Seq2Seq/predict.py
    python create_mp4.py --pred ./pred.npy --dest "$LOGDIR/output.mp4" --pipe DataProcessing/pipe
done
#LOGDIR="$BASELOGDIR/$EXP_SERIES/$EXP_ID"
#python Seq2Seq/train.py --train-folder $TRAIN --test-folder $TEST --default_root_dir $BASELOGDIR \
#        --experiment_series $EXP_SERIES --experiment_id $EXP_ID --max_epochs 1
#TEST_FOLDER=$TEST python Seq2Seq/predict.py
#python create_mp4.py --pred Seq2Seq/pred.npy --dest "$LOGDIR/output.mp4" --pipe DataProcessing/pipe