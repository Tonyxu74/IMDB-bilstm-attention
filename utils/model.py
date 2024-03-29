import torch
from torch import nn
from opts import args


class LSTMAttention(torch.nn.Module):
    def __init__(self):
        super(LSTMAttention, self).__init__()
        self.class_out = args.classes
        self.hidden_dim = args.hidden_dims
        self.batch_size = args.batch_size
        self.embedding_dim = args.embedding_dims
        self.pad_length = args.review_length
        self.use_gpu = torch.cuda.is_available()

        self.num_layers = args.num_layers
        self.dropout = 0.8
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, batch_first=True, num_layers=self.num_layers,
                              dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim, self.class_out)
        self.hidden = self.init_hidden()
        self.mean = 'mean'
        self.attn_fc = torch.nn.Linear(self.hidden_dim, 1)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = torch.zeros(
                2 * self.num_layers, batch_size, self.hidden_dim // 2
            ).cuda()
            c0 = torch.zeros(
                2 * self.num_layers, batch_size, self.hidden_dim // 2
            ).cuda()
        else:
            h0 = torch.zeros(
                2 * self.num_layers, batch_size, self.hidden_dim // 2
            )
            c0 = torch.zeros(
                2 * self.num_layers, batch_size, self.hidden_dim // 2
            )
        return (h0, c0)

    def attention(self, rnn_out, state):
        att = self.attn_fc(rnn_out)
        att = torch.softmax(att, dim=1)
        r_att = torch.sum(att * rnn_out, dim=1)
        return r_att

    def forward(self, X):
        hidden = self.init_hidden(X.size()[0])
        rnn_out, hidden = self.bilstm(X, hidden)
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n)
        logits = self.hidden2label(attn_out)
        return logits

    def predict(self, review, text_field):
        preprocessed_review = text_field.preprocess(review)

        if len(preprocessed_review) > self.pad_length:
            preprocessed_review = preprocessed_review[:self.pad_length]
        else:
            preprocessed_review = preprocessed_review + \
                ['<pad>'] * (self.pad_length - len(preprocessed_review))

        embedding = [
            text_field.vocab.vectors[text_field.vocab.stoi[word]] for word in preprocessed_review
        ]
        embedding = torch.stack(embedding).unsqueeze(0).cuda()
        prediction = self.forward(embedding)
        prob = torch.softmax(prediction, dim=1)
        pred_class = torch.argmax(prob, dim=1)[0].cpu().detach().numpy()

        prob = prob.cpu().detach().numpy()

        return prob, pred_class
