import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, phi_h=torch.tanh, phi_y=torch.sigmoid):
        super().__init__()
        self.phi_h, self.phi_y = phi_h, phi_y

        self.fc_xh = nn.Linear(in_dim, h_dim, bias=False)
        self.fc_hh = nn.Linear(h_dim, h_dim, bias=True)
        self.fc_hy = nn.Linear(h_dim, out_dim, bias=True)

    def forward(self, xt, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(xt.shape[0], self.fc_hh.in_features)

        ht = self.phi_h(self.fc_xh(xt) + self.fc_hh(h_prev))

        yt = self.fc_hy(ht)

        if self.phi_y is not None:
            yt = self.phi_y(yt)

        return yt, ht


class SentimentRNN(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, out_dim):
        super().__init__()

        # nn.Embedding converts from token index to dense tensor
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)

        # Our own Vanilla RNN layer, without phi_y so it outputs a class score
        self.rnn = RNNLayer(in_dim=embedding_dim, h_dim=h_dim, out_dim=out_dim, phi_y=None)

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # X shape: (S, B) Note batch dim is not first!

        embedded = self.embedding(X)  # embedded shape: (S, B, E)

        # Loop over (batch of) tokens in the sentence(s)
        ht = None
        for xt in embedded:  # xt is (B, E)
            yt, ht = self.rnn(xt, ht)  # yt is (B, D_out)

        # Class scores to log-probability
        yt_log_proba = self.log_softmax(yt)

        return yt_log_proba
