
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointDualMatching(nn.Module):
    def __init__(self, dim):
        r"""Initialize the shared feature projection for dual-softmax matching."""
        super(PointDualMatching, self).__init__()
        self.proj1 = nn.Linear(dim, dim, True)
        self.inf = np.inf

    def forward(self, ref_node_corr_knn_feats, src_node_corr_knn_feats, ref_node_corr_knn_scores, src_node_corr_knn_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks):
        r"""Compute dual-softmax scores weighted by per-point confidence.
        
        Args:
            ref_node_corr_knn_feats: reference patch features (P, K_ref, D).
            src_node_corr_knn_feats: source patch features (P, K_src, D).
            ref_node_corr_knn_scores: reference point confidences (P, K_ref).
            src_node_corr_knn_scores: source point confidences (P, K_src).
            ref_node_corr_knn_masks: reference validity masks (P, K_ref).
            src_node_corr_knn_masks: source validity masks (P, K_src).
        
        Returns:
            Matching scores (P, K_ref, K_src)."""
        m_ref_feats, m_src_feats = self.proj1(ref_node_corr_knn_feats), self.proj1(src_node_corr_knn_feats)

        scores = torch.einsum('bnd,bmd->bnm', m_ref_feats, m_src_feats)  # (P, K_ref, K_src)
        scores = scores / m_ref_feats.shape[-1] ** 0.5

        batch_size, num_row, num_col = scores.shape
        padded_row_masks = torch.zeros(size=(batch_size, num_row, num_col)).cuda()
        padded_row_masks.masked_fill_(~src_node_corr_knn_masks[:, None, :], float('-inf'))

        padded_col_masks = torch.zeros(size=(batch_size, num_row, num_col)).cuda()
        padded_col_masks.masked_fill_(~ref_node_corr_knn_masks[:, :, None], float('-inf'))
        matching_scores = F.softmax(scores + padded_row_masks, -1) * F.softmax(scores + padded_col_masks, 1)
        matching_scores = matching_scores * ref_node_corr_knn_scores[:, :, None] * src_node_corr_knn_scores[:, None, :]
        return matching_scores

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
