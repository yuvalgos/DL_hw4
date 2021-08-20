import numpy as np
import torch
import torch.nn as nn
import math


class SentimentSelfAttention(nn.Module):
    def __init__(self, embeddings: np.ndarray, d_model=256, num_classes=3, num_heads=1,
                 dropout=.0, freeze_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float())
        self.embedding_layer.weight.requires_grad = False if freeze_embedding else True

        self.PositionalEncoding = PositionalEncoding(self.embedding_dim)
        self.PositionalEncoding.requires_grad_(False)

        self.q_feedforward = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, d_model),
                                                 torch.nn.ReLU())
        self.k_feedforward = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, d_model),
                                                 torch.nn.ReLU())
        self.v_feedforward = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, d_model),
                                                 torch.nn.ReLU())

        self.SelfAttention1 = nn.MultiheadAttention(d_model, num_heads, dropout)

        #self.SelfAttention2 = nn.MultiheadAttetntion(self.embedding_dim, num_heads, dropout)

        self.attention_out_feedforward = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                             torch.nn.ReLU())

        self.dense_linear = torch.nn.Linear(d_model, num_classes)

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # X shape is (S, B). contains tokens
        X_embedded = self.embedding_layer(X)

        X_embedded_pe = self.PositionalEncoding(X_embedded)

        q = self.q_feedforward(X_embedded_pe)
        k = self.k_feedforward(X_embedded_pe)
        v = self.v_feedforward(X_embedded_pe)
        # q, k, v dimensions are (S, B, d_model)

        attention_out, _ = self.SelfAttention1(q, k, v)  # (S, B, d_model)

        attention_out_avg = attention_out.mean(dim=0)

        out_ff = self.attention_out_feedforward(attention_out_avg)

        class_scores = self.dense_linear(out_ff)
        #print("seq_class_scores shape:", seq_class_scores.shape)

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
