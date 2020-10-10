import torch
import torch.nn as nn


class LinearWithBatchNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearWithBatchNorm, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(61)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.context_highway = nn.Sequential(
            LinearWithBatchNorm(input_dim, hidden_dim),
            LinearWithBatchNorm(hidden_dim, hidden_dim),
            LinearWithBatchNorm(hidden_dim, hidden_dim)
        )
        self.context_gru = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, bidirectional=True, num_layers=num_layers, batch_first=False
        )
        self.hidden_dim = hidden_dim

        self.words_gru = nn.GRU(100, 100, 1, batch_first=True, bidirectional=True)
        self.merge_layer = nn.Linear(hidden_dim + 200, hidden_dim)

    def forward(self, x, w):
        seq_len, batch, context, input_dim = x.shape
        x = x.reshape(seq_len * batch, context, input_dim)  # batch*seq_len, context, input_dim
        x = self.context_highway(x)  # batch*seq_len, context, hidden_dim
        x = self.context_gru(x)[0][:, -1, :]  # batch*seq_len, hidden_dim
        x = self.batch_norm(x)
        x = torch.relu(x)

        _, _, words_count, embedding = w.shape
        w = w.reshape(seq_len * batch, words_count, embedding)
        w = self.words_gru(w)[0][:, -1, :]  # batch*seq_len, words_hidden
        w = torch.relu(w)

        x = x.reshape(seq_len, batch, self.hidden_dim)  # seq_len, batch, hidden_dim
        w = w.reshape(seq_len, batch, 200)  # seq_len, batch, words_hidden

        x = torch.cat((x, w), dim=2)
        x = self.merge_layer(x)  # seq_len, batch, hidden_dim

        output, hidden = self.rnn(x)
        return self.dropout(output), self.dropout(hidden)


class Attention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)

    def forward(self, encoder_states, decoder_hidden):
        # encoder_states - enc_len, batch, enc_dim
        encoder_states = self.enc_to_dec(encoder_states)
        # encoder_states - enc_len, batch, dec_dim
        # decoder_hidden - 1, batch, dec_dim
        encoder_states = encoder_states.transpose(0, 1)  # batch, enc_len, dec_dim
        decoder_hidden = decoder_hidden.permute(1, 2, 0)  # batch, dec_dim, 1
        weights = torch.matmul(encoder_states, decoder_hidden).squeeze(
            2
        )  # batch, enc_len
        scores = torch.softmax(weights, dim=1).unsqueeze(2)
        # encoder_states - batch, enc_len, dec_dim
        # scores - batch, enc_len, 1
        encoder_states = encoder_states.transpose(1, 2)
        # encoder_states - batch, dec_dim, enc_len
        weighted_states = torch.matmul(encoder_states, scores)  # batch, dec_dim, 1
        weighted_states = weighted_states.permute(2, 0, 1)
        # 1, batch, dec_dim
        return weighted_states


class Decoder(nn.Module):
    def __init__(
        self, output_dim: int, hidden_dim: int, enc_dim: int, max_gen: int = 20
    ):
        super().__init__()
        self.enc_dec_linear = nn.Linear(enc_dim, hidden_dim)
        self.rnn = nn.GRU(output_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.max_gen = max_gen
        self.output_dim = output_dim
        self.attention = Attention(enc_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_hidden,
        previous_poses=None,
        real_poses_len: int = None,
    ):
        # encoder_states - seq_len, batch, enc_dim
        # encoder_hidden - num_layers * num_directions, batch, enc_hidden
        # previous_poses - [len1, batch_size, output_dim]
        # real_poses_len
        seq_len, batch_size, enc_dim = encoder_states.shape
        dec_hidden = encoder_hidden
        dec_hidden = dec_hidden.view(-1, 2, batch_size, enc_dim // 2)
        dec_hidden = dec_hidden[-1:]  # 1, 2, batch_size, enc_dim // 2
        dec_hidden = dec_hidden.transpose(1, 2)
        # 1, batch_size, 2, enc_dim // 2
        dec_hidden = dec_hidden.reshape(1, batch_size, -1)
        # dec_hidden - 1, batch, enc_dim
        dec_hidden = self.enc_dec_linear(dec_hidden)
        # dec_hidden - 1, batch, dec_hidden

        output, dec_hidden = self.rnn(previous_poses, dec_hidden)
        # dec_hidden - num_layers, batch, hidden
        start_pose = previous_poses[-1:]  # 1, batch, output_dim

        if real_poses_len is not None:
            max_gen_len = real_poses_len
        else:
            max_gen_len = self.max_gen

        # start_pose - 1, batch, output_dim
        # dec_hidden - 1, batch, hidden_dim
        poses = []
        for i in range(max_gen_len):
            output, dec_hidden = self.rnn(start_pose, dec_hidden)
            attention = self.attention(encoder_states, dec_hidden)
            concat = torch.cat((dec_hidden, attention), dim=2)
            # concat - 1, batch, hidden_dim * 2
            start_pose = self.linear(concat)
            # start_pose - 1, batch, output_dim
            poses.append(start_pose)
        # poses [len, batch, output_dim]
        poses = torch.cat(poses, dim=0)
        # poses = torch.stack(poses, dim=0).squeeze(0)
        return poses
