import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.1):
        super(QuantumDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, input_dim)
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, seq_len, hidden_dim)
        logits = self.fc_out(output)  # (batch_size, seq_len, vocab_size)
        return logits, hidden
