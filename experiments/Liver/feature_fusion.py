"""Fuse backbone features by bidirectional cross-attention within each point cloud."""
import torch
from torch import nn
from torch.nn import functional as F


class FeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def fuse_cloud(self, first, second):
        # Each backbone attends to the other using softmax(Q K^T / sqrt(d)) V.
        first_to_second = F.scaled_dot_product_attention(
            self.query(first)[None, None], self.key(second)[None, None], self.value(second)[None, None]
        )[0, 0]
        second_to_first = F.scaled_dot_product_attention(
            self.query(second)[None, None], self.key(first)[None, None], self.value(first)[None, None]
        )[0, 0]
        return self.weights[0] * first_to_second + self.weights[1] * second_to_first

    def forward(self, first, second, ref_length):
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError('Backbone features must have matching (N, C) shapes.')
        if not 0 <= ref_length <= len(first):
            raise ValueError('Invalid reference cloud length.')
        return torch.cat([
            self.fuse_cloud(first[:ref_length], second[:ref_length]),
            self.fuse_cloud(first[ref_length:], second[ref_length:]),
        ], dim=0)
