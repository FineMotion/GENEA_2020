from typing import Union

from moviepy.editor import VideoFileClip, AudioFileClip
from argparse import ArgumentParser
from pathlib import WindowsPath


def create_clip(video_path: Union[str, WindowsPath],
                audio_path: Union[str, WindowsPath],
                dest: Union[str, WindowsPath]):

    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))
        # .subclip(0, 60)
    result = video.set_audio(audio)
    result.write_videofile(str(dest))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video')
    parser.add_argument('--audio')
    parser.add_argument('--dest')
    args = parser.parse_args()

    create_clip(args.vidio, args.audio, args.dest)
