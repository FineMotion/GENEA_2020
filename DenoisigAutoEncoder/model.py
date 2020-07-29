import torch
import torch.nn as nn

HIDDEN_DIM = 312
INPUT_DIM = 45
OUTPUT_DIM = 40


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.hidden = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.hidden = nn.Linear(OUTPUT_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


class DenoisingAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
