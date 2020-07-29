@echo off
set MOTIONS="data\Motion"
set FEATURES="data\Features"
set AUDIO="data\Audio"
set MFCC="data\MFCC"
set READY="data\Ready"

echo "Generating motion features..."
python process_motions.py --src_dir %MOTIONS% --dst_dir %FEATURES%
echo "Generating audio features..."
python process_audio.py --src_dir %AUDIO% --dst_dir %MFCC%
echo "Aligning data..."
python align_data.py --motion_dir %FEATURES% --audio_dir %MFCC% --dst_dir %READY% --with_context
echo "Done. Data lies in %READY%"