import json
from typing import NamedTuple, List


class Token(NamedTuple):
    start: float
    end: float
    word: str

    def __str__(self):
        return '(%0.3f:%0.3f) %s' % (self.start, self.end, self.word)


class TokenContainer:

    def __init__(self, window_size=60, fps=20.):
        self.tokens = []  # type: List[Token]
        self.window_size = window_size
        self.frame_time = 1. / fps

    def __getitem__(self, item):

        end_time = (item + self.window_size // 2) * self.frame_time
        start_time = max(item - self.window_size // 2, 0) * self.frame_time

        result = filter(lambda x: x.start >= start_time and x.end <= end_time, self.tokens)
        return list(result)


def parse_transcripts(json_path: str) -> TokenContainer:
    with open(json_path, 'r') as input_file:
        data = json.load(input_file)
        container = TokenContainer()
        for frame in data:
            frame_words = frame['alternatives'][0]['words']
            for frame_word in frame_words:
                token = Token(
                    start=float(frame_word['start_time'][:-1]),
                    end=float(frame_word['end_time'][:-1]),
                    word=frame_word['word'])
                container.tokens.append(token)
    return container


if __name__ == '__main__':
    container = parse_transcripts(r'../data/Transcripts/Recording_001.json')
    for token in container[100]:
        print(token)