import numpy as np
import torch
import torch.nn as nn
import math


class LinearWithBN(nn.Module):
    """
     A Linear layer + batch normalization right after it that can work
     on sequnce models with an input of (S, B, E). It is necessary because
     BN layers expects batch dimention first.
    """
    def __init__(self, in_f, out_f):
        super().__init__()

        self.lin = torch.nn.Linear(in_f, out_f)
        self.bn = torch.nn.BatchNorm1d(out_f, track_running_stats=False)

    def forward(self, X):
        linout = self.lin(X)
        linout = linout.permute(1, 2, 0)
        out = self.bn(linout)
        out = out.permute(2, 0, 1)
        return out


class SentimentSelfAttention(nn.Module):
    def __init__(self, embeddings: np.ndarray, d_model=256, num_classes=3, num_heads=1,
                 dropout=.0, kqv_dropout=.0, dense_dropout=.0, two_layers=False, freeze_embedding=True):
        super().__init__()

        self.embedding_dim = embeddings.shape[1]
        self.d_model = d_model
        self.num_heads = num_heads
        self.two_layers = two_layers

        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float())
        self.embedding_layer.weight.requires_grad = False if freeze_embedding else True

        self.PositionalEncoding = PositionalEncoding(self.embedding_dim)
        self.PositionalEncoding.requires_grad_(False)

        self.embeddings_dropout = torch.nn.Dropout(p=dropout)

        self.q_feedforward = torch.nn.Sequential(LinearWithBN(self.embedding_dim, d_model),
                                                 torch.nn.Dropout(p=kqv_dropout),
                                                 torch.nn.ReLU())
        self.k_feedforward = torch.nn.Sequential(LinearWithBN(self.embedding_dim, d_model),
                                                 torch.nn.Dropout(p=kqv_dropout),
                                                 torch.nn.ReLU())
        self.v_feedforward = torch.nn.Sequential(LinearWithBN(self.embedding_dim, d_model),
                                                 torch.nn.Dropout(p=kqv_dropout),
                                                 torch.nn.ReLU())

        self.SelfAttention1 = torch.nn.MultiheadAttention(d_model, num_heads, dropout)

        if two_layers:
            self.q2_feedforward = torch.nn.Sequential(LinearWithBN(d_model, d_model),
                                                      torch.nn.Dropout(p=kqv_dropout),
                                                      torch.nn.ReLU())
            self.k2_feedforward = torch.nn.Sequential(LinearWithBN(d_model, d_model),
                                                      torch.nn.Dropout(p=kqv_dropout),
                                                      torch.nn.ReLU())
            self.v2_feedforward = torch.nn.Sequential(LinearWithBN(d_model, d_model),
                                                      torch.nn.Dropout(p=kqv_dropout),
                                                      torch.nn.ReLU())

            self.SelfAttention2 = torch.nn.MultiheadAttention(d_model, num_heads, dropout)

        self.attention_out_feedforward = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                             torch.nn.BatchNorm1d(d_model, track_running_stats=False),
                                                             torch.nn.Dropout(p=dropout),
                                                             torch.nn.ReLU())

        self.dense_linear = torch.nn.Sequential(torch.nn.Linear(d_model, num_classes),
                                                torch.nn.Dropout(p=dense_dropout),)

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X, get_attention_weights=False):
        attention_weights = []

        # X shape is (S, B). contains tokens
        X_embedded = self.embedding_layer(X)
        X_embedded = self.embeddings_dropout(X_embedded)

        X_embedded_pe = self.PositionalEncoding(X_embedded)

        q = self.q_feedforward(X_embedded_pe)
        k = self.k_feedforward(X_embedded_pe)
        v = self.v_feedforward(X_embedded_pe)
        # q, k, v dimensions are (S, B, d_model)

        attention_out, weights1 = self.SelfAttention1(q, k, v)  # (S, B, d_model)
        attention_weights.append(weights1)

        if self.two_layers:
            # new keys and values are calculated using residual connection
            q2 = attention_out + self.q2_feedforward(attention_out)
            k2 = attention_out + self.k2_feedforward(attention_out)
            v2 = attention_out + self.v2_feedforward(attention_out)

            # layer 2 is connected in a residual connection:
            attention_out2, weights2 = self.SelfAttention2(q2, k2, v2)
            attention_out = attention_out + attention_out2
            attention_weights.append(weights2)

        attention_out = attention_out.mean(dim=0)
        out_ff = self.attention_out_feedforward(attention_out)

        class_scores = self.dense_linear(out_ff)

        log_prob = self.log_softmax(class_scores)
        sentiment_class = torch.argmax(class_scores, dim=1)

        if get_attention_weights:
            return sentiment_class, log_prob, attention_weights
        else:
            return sentiment_class, log_prob


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 500):
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
