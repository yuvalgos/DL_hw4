import numpy as np
import torch
import torch.nn as nn
import math


class SentimentSelfAttention(nn.Module):
    def __init__(self, embeddings: np.ndarray, d_model=256, num_classes=3, num_heads=1,
                 dropout=.0, two_layers=False, freeze_embedding=True):
        super().__init__()

        self.embedding_dim = embeddings.shape[1]
        self.d_model = d_model
        self.num_heads = num_heads
        self.two_layers = two_layers

        self.last_attention_weights = None

        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float())
        self.embedding_layer.weight.requires_grad = False if freeze_embedding else True

        self.PositionalEncoding = PositionalEncoding(self.embedding_dim)
        self.PositionalEncoding.requires_grad_(False)

        self.q_feedforward = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, d_model),
                                                 torch.nn.Dropout(p=dropout),
                                                 torch.nn.ReLU())
        self.k_feedforward = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, d_model),
                                                 torch.nn.Dropout(p=dropout),
                                                 torch.nn.ReLU())
        self.v_feedforward = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, d_model),
                                                 torch.nn.Dropout(p=dropout),
                                                 torch.nn.ReLU())

        self.SelfAttention1 = torch.nn.MultiheadAttention(d_model, num_heads, dropout)

        if two_layers:
            self.SelfAttention2 = torch.nn.MultiheadAttention(d_model, num_heads, dropout)

            self.q2_feedforward = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                      torch.nn.Dropout(p=dropout),
                                                      torch.nn.ReLU())
            self.k2_feedforward = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                      torch.nn.Dropout(p=dropout),
                                                      torch.nn.ReLU())
            self.v2_feedforward = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                      torch.nn.Dropout(p=dropout),
                                                      torch.nn.ReLU())

        self.attention_out_feedforward = torch.nn.Sequential(torch.nn.Linear(d_model, d_model),
                                                             torch.nn.Dropout(p=dropout),
                                                             torch.nn.ReLU())

        self.dense_linear = torch.nn.Linear(d_model, num_classes)

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # torch.autograd.set_detect_anomaly(True)
        attention_weights = []

        # X shape is (S, B). contains tokens
        X_embedded = self.embedding_layer(X)

        X_embedded_pe = self.PositionalEncoding(X_embedded)

        q = self.q_feedforward(X_embedded_pe)
        k = self.k_feedforward(X_embedded_pe)
        v = self.v_feedforward(X_embedded_pe)
        # q, k, v dimensions are (S, B, d_model)

        attention_out, weights1 = self.SelfAttention1(q, k, v)  # (S, B, d_model)
        attention_weights.append(weights1)

        if self.two_layers:
            q2 = self.q2_feedforward(attention_out)
            k2 = self.k2_feedforward(attention_out)
            v2 = self.v2_feedforward(attention_out)

            # layer 2 is connected in a residual connection:
            attention_out2, weights2 = self.SelfAttention2(q2, k2, v2)
            attention_out = attention_out + attention_out2
            attention_weights.append(weights2)

        attention_out_avg = attention_out.mean(dim=0)

        out_ff = self.attention_out_feedforward(attention_out_avg)

        class_scores = self.dense_linear(out_ff)
        # print("seq_class_scores shape:", seq_class_scores.shape)

        log_prob = self.log_softmax(class_scores)
        sentiment_class = torch.argmax(class_scores, dim=1)

        self.last_attention_weights = attention_weights

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
