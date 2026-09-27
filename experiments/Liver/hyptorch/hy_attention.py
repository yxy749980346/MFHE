"""Bilinear self- and cross-attention for hyperbolic features."""
import torch
from torch import nn
from torch.nn import functional as F


class HyperbolicAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.metric = nn.Parameter(torch.eye(feature_dim))
        self.self_norm = nn.LayerNorm(feature_dim)
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.cross_weight = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.cross_weight.weight)

    def attend(self, query, key):
        # Multiplication by sqrt(d) cancels the attention kernel scaling, yielding softmax(Q M K^T) K.
        return F.scaled_dot_product_attention(
            ((query @ self.metric) * query.shape[-1] ** 0.5).unsqueeze(1),
            key.unsqueeze(1), key.unsqueeze(1)
        ).squeeze(1)

    def self_attention(self, features):
        return F.relu(self.self_norm(self.attend(features, features)))

    def cross_attention(self, query, key):
        update = F.relu(self.cross_norm(self.attend(query, key)))
        return query + self.cross_weight(update)

    def forward(self, ref_features, src_features):
        ref_self = self.self_attention(ref_features)
        src_self = self.self_attention(src_features)
        # Both directions use the same pre-cross-attention features.
        return self.cross_attention(ref_self, src_self), self.cross_attention(src_self, ref_self)
