# FineMotion
This repo provides a solution for GENEA Challange 2020.
## Table of content
- [Data processing](#Data-processing) 

## Data processing
The folder `DataProcessing` contains scripts for features extraction, data normalization and generation of output video.
The pipeline is based on one of [baselines](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder/blob/GENEA_2020/data_processing/).

Assume that challenge dataset for train located in folder `data` with following files:
 - `./data/Audio/Recoding_%3d.wav` - input raw audio files with recorded speech
 - `./data/Motion/Recoding_%3d.bvh` - motion data corresponding to audio files
 - `./data/Transcripts/Recording_%3d.json` - text transcripts for recorded speech
 
There are several scripts to prepare data:
1. `process_motions.py` - converts motion data into numpy arrays and stores them into `*.npy` files. 
    Arguments:
    - `--src` - path to the folder with motion data
    - `--dst` - path to the folder the processed arrays will be stored.
    - `--pipe` - (optional, default=`./pipe`) - the path where sklearn pipeline will be stored or read.
    - `--bvh` - (flag) if exists inverse transform: generate bvh-files from npy
    
    Example:
    ```
    python DataProcessing/process_motions.py --src data/Motion --dst data/Features
    ```
   
2. `process_audio.py` - extracts MFCC features from speech recordings, averages 5 successive frames to match FPS
  and stores obtained arrays into `*.npy` files./
  Arguments:
    - `--src` - path to the folder with audio files
    - `--dst` - path to the folder the extracted MFCCs features will be stored.
    
    Example:
    ```
    python DataProcessing/process_audio.py --src data/Audio --dst data/MFCC
    ```
        
3. `align_data.py` - aligns motion and audio features and stores them into numpy archives with `X` and `Y` keys for 
audio and motion features respectively. Optionally adds contexts to the each audio frame.
    Arguments:
    - `--motion_dir` - (optional) path to the folder with motion features arrays.
    If not set, saves contextualized audio features into numpy array.
    - `--audio_dir` - path to the folder with audio features arrays
    - `--dst_dir` - path to the folder the aligned data will be stored
    - `--with_context` - (flag) if set the audio features will be stored with the context window for each frame
    - `--context_length` - (optional. default=60) context window size
    
    Example:
    ```
    python DataProcessing/align_data.py --motion_dir data/Features --audio_dir data/MFCC --dst_dir data/Ready --with_context
    ```
4. `normalize_data.py` - normalizes motion features to [-1,1], maximum and mean values are calculated on train dataset 
(all recording except the first one).\
    Arguments:
    - `--src` - path to the folder with aligned data
    - `--dst` - path to the folder the normalized data will be stored.
    - `--values` - (optional, default=`./mean_pose.npz`) path to npz file where normalizing values will be stored
    
    Example:
    ```
    python DataProcessing/normalize_data.py --src data/Ready --dst data/Normalized
    ```

After running these 4 scripts listed above we get numpy archives appropriate for training models.

