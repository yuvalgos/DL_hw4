import numpy as np
import torch
import torch.nn as nn


class SentimentGRU(nn.Module):
    def __init__(self, embeddings: np.ndarray, num_classes=3, hidden_size=128, num_layers=2,
                 dropout=.0, freeze_embedding=True):
        super().__init__()

        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float())

        self.embedding_layer.weight.requires_grad = False if freeze_embedding else True

        self.gru = torch.nn.GRU(input_size=self.embedding_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout)

        # GRU output shape after taking last layer at last time step is (B, hidden_size),
        # we need a dense layer to get class scores from it
        self.dense_linear = torch.nn.Sequential(torch.nn.Linear(hidden_size, num_classes),)
                                                # torch.nn.Dropout(p=dropout),
                                                # torch.nn.ReLU())

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # X shape is (S, B). contains tokens
        X_embedded = self.embedding_layer(X)

        # GRU input should be (S, B, E)
        _, h_n= self.gru(X_embedded)

        # we only need the last hidden state at the last time step
        class_scores = self.dense_linear(h_n[-1, :, :])

        log_prob = self.log_softmax(class_scores)
        sentiment_class = torch.argmax(class_scores, dim=1)

        # log prob shape is(B, C)
        return sentiment_class, log_prob
