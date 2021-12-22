import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import random
random.seed(10)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, linear_out_dim, hidden_dim, lstm_layers,
                 bidirectional, dropout_val, tag_size):
        super(BiLSTM, self).__init__()
        """ Hyper Parameters """
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.embedding_dim = embedding_dim
        self.linear_out_dim = linear_out_dim
        self.tag_size = tag_size
        self.num_directions = 2

        """ Initializing Network """
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * self.num_directions, linear_out_dim)
        self.dropout = nn.Dropout(dropout_val)
        self.elu = nn.ELU(alpha=0.01)
        self.classifier = nn.Linear(linear_out_dim, self.tag_size)

    def init_hidden(self, batch_size):
        h, c = (torch.zeros(self.lstm_layers * self.num_directions,
                            batch_size, self.hidden_dim),
                torch.zeros(self.lstm_layers * self.num_directions,
                            batch_size, self.hidden_dim))
        return h, c

    def forward(self, sen, sen_len):  # sen_len

        batch_size = sen.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(sen).float()
        packed_embedded = pack_padded_sequence(embedded, sen_len, batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM(packed_embedded, (h_0, c_0))
        output_unpacked, _ = pad_packed_sequence(output, batch_first=True)
        dropout = self.dropout(output_unpacked)
        lin = self.fc(dropout)
        pred = self.elu(lin)
        pred = self.classifier(pred)
        return pred
