# FineMotion
This repo provides a solution for GENEA Challange 2020.
## Table of contents
- [Data processing](#Data-processing) 
- [Training](#Training)
- [Predicting](#Predicting)
- [Processing test dataset](#Processing-test-dataset)
- [Extra](#Extra)
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

5. Splitting into train & valid. We were validating on Recording_001, the rest was used for training.

```
mkdir -p data/dataset/train data/dataset/test
cp data/Ready/* data/dataset/train
mv data/dataset/train/data_001.npz data/dataset/test
```


After running these 4 scripts listed above we get numpy archives appropriate for training models.

## Training
There are 2 types of models in this repository, both are based on a seq2seq architecture.
* ContextSeq2Seq - a seq2seq with a context encoder, which at each steps combines words and audio features in an encoder cell. This model is described in section 3.2 of our article.
```
python ContextSeq2Seq\train.py  --gpus 1 --predicted-poses 20 --previous-poses 10 --serialize-dir new  --max_epochs 100 --stide  --batch_size 50 --embedding .\embeddings\glove.6B.100d.txt --text_folder data\Transcripts 
```
* WordsSeq2Seq - seq2seq, which uses attention over encoded words & audio features. This model is described in section 3.2 of our work.

```
python WordsSeq2Seq/train.py --gpus 1 --predicted-poses 20 --previous-poses 10 --serialize-dir new --max_epochs 100 --stride 1 --batch_size 512 --with_context --embedding embeddings/glove.6B.100d.txt --text_folder data/Transcripts
```

Both models have common params:
- Parameters for pytorch lightning [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#)
- `--predicted-poses` - number of frames in sequence per instance
- `--previous-poses` - number of previous poses to initialize decoder state
- `--serialize-dir` - folder to save checkpoints
- `--stride` - margin between two successive instances
- `--embedding` - path to GLOVE embeddings file
- `--text_folder` - path to text transcripts
For WordsSeq2Seq model argument `--with_context` add contexts for audio encoder

## Predicting

The `predict.py` file should be used from an according model folder. This code example predicts motion features 
from valid file:
```
python WordsSeq2Seq/predict.py --src data/dataset/test/data_001.npz --checkpoint new/last.ckpt --dest predictions/data_001.npy -text_folder data/Transcripts
```

To create bvh or mp4 files from predicted features use `create_bvh.py` or `create_mp4.py`:
```
python create_mp4.py --pred predicions/pred.npy --dest vid.mp4 --smooth --audio data/Audio/Recording_001.wav --mean mean_pose.npz
python create_bvh.py --pred predictions --dest results --smooth --mean mean_pose.npz
```

`create_bvh.py` takes folder on input (`--pred`) and generates bvh-file to output folder (`--dest`) for each npy-file from input folder\
`create_mp4.py` takes only one npy-file (`--pred`) to generate mp4-file (`--dest`) using visualization server. 
Also takes `--audio` parameter - path to the input audio file to merge it with silent visualization video.

Common parameters:
- `--smooth` - apply Savitzky-Golay filter to predicted motion features
- `--mean` - file with mean values obtained from normalization
- `--pipe` - pipeline path obtained from processing motions

## Processing test dataset
Assume that test audio and transcripts are placed in folders `data/Test/Audio` and `data/Test/Transcripts`. To make predictions for the test dataset you need to follow these steps:
- Get MFCC features from audio:
```
python DataProcessing/process_audio.py --src data/Test/Audio --dst data/Test/MFCC
```
- Add contexts to Audio features:
```
python DataProcessing/align_data.py --audio_dir data/Test/MFCC --dst_dir data/Test/Ready --with_context
```
- Rename transcripts (`TestSeq%3d.json -> Recording_%3d.json`): 
```shell script
powershell:
cd data\Test\Transcripts
Dir | Rename-Item -NewName {$_.name -replace "TestSeq","Recording_"}
```
- Predict:
```shell script
python ContextSeq2Seq/predict.py --src data/Test/Ready/TestSeq001.npy --checkpoint text_encoder/last.ckpt --dest predictions/data_001.npy --text_folder data/Test/Transcripts
```
Visualization is the same as with train dataset.

## Extra
Checkpoints and some examples of generated motions can be found
 [here](https://drive.google.com/drive/folders/17a2guNld9YBLzy3477nn7UECBOpwmLAN?usp=sharing)
 
There are some other approaches we tried on this repo.
 - Folders `DenoisingAutoEncoder` and `SpeechEncoder` are our 
pytorch reimplementation of one of the 
[baselines](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder/tree/GENEA_2020). 
 - `VariationalAutoEncoder` - is our attempt to change autoencoder in baseline mentioned before by Variational Auto Encoder.
- Our experiments with adversarial learning are located in branches `seq2seq_asmekal` and `seq2seq_gan`.
 
The code listed above has not been tested, so it may not work.

