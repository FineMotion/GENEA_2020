@echo off
set MOTIONS="data\Motion"
set FEATURES="data\Features"
set AUDIO="data\Audio"
set MFCC="data\MFCC"

echo "Generating motion features..."
python process_motions.py --src_dir %MOTIONS% --dst_dir %FEATURES%
echo "Generating audio features..."
python precess_audio.py --src_dir %AUDIO% --dst_dir %MFCC%