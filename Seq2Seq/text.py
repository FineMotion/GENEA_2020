import json
from typing import NamedTuple, List

import torch
from tqdm import tqdm


class Token(NamedTuple):
    start: float
    end: float
    word: str

    def __str__(self):
        return '(%0.3f:%0.3f) %s' % (self.start, self.end, self.word)


class Vocab:

    def __init__(self, embedding_file: str = "embeddings/glove.6B.100d.txt", embegging_dim=100):
        self.weights = [
            [0.0]*embegging_dim,  # paddings
            [1.0]*embegging_dim   # unknown
        ]
        self.token_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        self.idx_to_token = {
            0: '<PAD>',
            1: '<UNK>'
        }

        for line in tqdm(open(embedding_file, encoding='utf-8')):
            arr = line.strip().split()
            word = arr[0]
            vec = list(map(float, arr[1:]))
            self.token_to_idx[word] = len(self.token_to_idx)
            self.idx_to_token[len(self.idx_to_token)] = word
            self.weights.append(vec)

    def words_to_idx(self, words: List[str]) -> List[int]:
        # lower is important
        # there is not unk, so we return 0 as 'the'
        words = [word[:-1] if word.endswith('.') else word for word in words]
        return [self.token_to_idx.get(word.lower(), 1) for word in words]


class TokenContainer:

    def __init__(self, vocab: Vocab, window_size=60, fps=20.):
        self.tokens = []  # type: List[Token]
        self.window_size = window_size
        self.frame_time = 1. / fps
        self.vocab = vocab

    def _get(self, item):
        end_time = (item + self.window_size // 2) * self.frame_time
        start_time = max(item - self.window_size // 2, 0) * self.frame_time

        tokens = filter(lambda x: x.start >= start_time and x.end <= end_time, self.tokens)
        return self.vocab.words_to_idx([token.word for token in tokens])

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get(item)
        if isinstance(item, slice):
            result = []
            for i in range(item.start, item.stop):
                result.append(self._get(i))
            return result


def parse_transcripts(json_path: str, vocab: Vocab) -> TokenContainer:
    with open(json_path, 'r') as input_file:
        data = json.load(input_file)
        container = TokenContainer(vocab)
        for frame in data:
            frame_words = frame['alternatives'][0]['words']
            for frame_word in frame_words:
                token = Token(
                    start=float(frame_word['start_time'][:-1]),
                    end=float(frame_word['end_time'][:-1]),
                    word=frame_word['word'])
                container.tokens.append(token)
    return container


def pad_matrix(matrices: List[torch.Tensor]):
    max_paths = max([m.size(0) for m in matrices])
    max_len = max([m.size(1) for m in matrices])

    out_tensor = torch.zeros((len(matrices), max_paths, max_len), dtype=torch.long)
    for i, matrix in enumerate(matrices):
        paths = matrix.size(0)
        length = matrix.size(1)

        out_tensor[i, :paths, :length] = matrix

    return out_tensor