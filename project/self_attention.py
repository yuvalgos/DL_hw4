import numpy as np
import torch
import torch.nn as nn
import math


class SentimentSelfAttention(nn.Module):
    def __init__(self, embeddings: np.ndarray, num_classes=3, num_heads=1,
                 dropout=.0, freeze_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float())

        self.embedding_layer.weight.requires_grad = False if freeze_embedding else True

        self.PositionalEncoding = PositionalEncoding(self.embedding_dim)

        self.SelfAttention1 = nn.MultiheadAttetntion(self.embedding_dim, num_heads, dropout)

        #self.SelfAttention2 = nn.MultiheadAttetntion(self.embedding_dim, num_heads, dropout)

        self.dense_linear = torch.nn.Linear(self.embedding_dim, num_classes)

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # X shape is (S, B). contains tokens
        X_embedded = self.embedding_layer(X)

        X_embedded_pe = self.PositionalEncoding(X_embedded)
        # self Attention returns
        _, h_n = self.SelfAttention1(X_embedded_pe, X_embedded_pe, X_embedded_pe)

        # we only need the last hidden state at the last time step
        class_scores = self.dense_linear(h_n[-1, :, :])

        log_prob = self.log_softmax(class_scores)
        sentiment_class = torch.argmax(class_scores, dim=1)

        # log prob shape is(B, C)
        return sentiment_class, log_prob


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 5000):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x
