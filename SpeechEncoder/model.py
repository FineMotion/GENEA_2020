import torch
import torch.nn as nn

HIDDEN_DIM = 150
INPUT_DIM = 26
OUTPUT_DIM = 45
CONTEXT_SIZE = 61
DROPOUT_RATE = 0.1


class LinearWithBatchNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearWithBatchNorm, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(CONTEXT_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class SpeechMotionModel(nn.Module):
    def __init__(self):
        super(SpeechMotionModel, self).__init__()
        self.ff_1 = LinearWithBatchNorm(INPUT_DIM, HIDDEN_DIM)
        self.ff_2 = LinearWithBatchNorm(HIDDEN_DIM, HIDDEN_DIM)
        self.ff_3 = LinearWithBatchNorm(HIDDEN_DIM, HIDDEN_DIM)

        self.gru = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, 1, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.out = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_1(x)  # batch_size, context_dim, input_dim
        x = self.ff_2(x)  # batch_size, context_dim, hidden_dim
        x = self.ff_3(x)  # batch_size, context_dim, hidden_dim

        x = self.gru(x, None)[0][:, -1, :]    # batch_size, hidden_dim
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
