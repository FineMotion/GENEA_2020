from moviepy.editor import VideoFileClip, AudioFileClip
from argparse import ArgumentParser


def create_clip(video_path: str, audio_path: str, dest: str):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).subclip(0, 60)
    result = video.set_audio(audio)
    result.write_videofile(dest)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video')
    parser.add_argument('--audio')
    parser.add_argument('--dest')
    args = parser.parse_args()

    create_clip(args.vidio, args.audio, args.dest)