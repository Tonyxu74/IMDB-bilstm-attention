import torch
from torch.autograd import Variable
from torch import nn


class LSTMAttention(torch.nn.Module):
    def __init__(self):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = 128
        self.batch_size = 8
        self.embedding_dim = 300
        self.use_gpu = torch.cuda.is_available()

        self.num_layers = 1
        self.dropout = 0.8
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, batch_first=True, num_layers=self.num_layers,
                              dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim, 2)
        self.hidden = self.init_hidden()
        self.mean = 'mean'
        self.attn_fc = torch.nn.Linear(self.embedding_dim, 1)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda()
            c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda()
        else:
            h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)
            c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)
        return (h0, c0)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    # end method attention

    def forward(self, X):
        hidden = self.init_hidden(X.size()[0])  #
        rnn_out, hidden = self.bilstm(X, hidden)
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n)
        logits = self.hidden2label(attn_out)
        return logits