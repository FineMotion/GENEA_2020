#@echo off
DATA="D:/data/GENEA_2020_Data/Dataset"
MOTIONS="$DATA/Motion"
FEATURES="$DATA/Features"
AUDIO="$DATA/Audio"
MFCC="$DATA/MFCC"
READY="$DATA/ReadyNew"

source activate genea_challenge

cd ./DataProcessing/

echo "Generating motion features..."
python process_motions.py --src $MOTIONS --dst $FEATURES
echo "Generating audio features..."
python process_audio.py --src_dir $AUDIO --dst_dir $MFCC
echo "Aligning data..."
python align_data.py --motion_dir $FEATURES --audio_dir $MFCC --dst_dir $READY --with_context
echo "Done. Data lies in $READY"

# normalize
python normalize_data.py