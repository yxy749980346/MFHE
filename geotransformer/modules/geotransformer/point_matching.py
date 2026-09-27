import torch
import torch.nn as nn


class PointMatching(nn.Module):
    def __init__(
        self,
        k: int,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        remove_duplicate: bool = False,
    ):
        r"""Top-k point correspondences from patch matching scores.

        Args:
            k (int): top-k selection for matching.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
        """
        super(PointMatching, self).__init__()
        self.k = k
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.remove_duplicate = remove_duplicate

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute a boolean point correspondence matrix for each patch pair."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        k_ref = min(self.k, score_mat.shape[2])
        k_src = min(self.k, score_mat.shape[1])
        ref_values, ref_indices = score_mat.topk(k=k_ref, dim=2)
        src_values, src_indices = score_mat.topk(k=k_src, dim=1)
        ref_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)
        src_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)
        ref_corr_mat.scatter_(2, ref_indices, ref_values > self.confidence_threshold)
        src_corr_mat.scatter_(1, src_indices, src_values > self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]

        corr_mat = torch.logical_and(corr_mat, mask_mat)

        return corr_mat

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        ref_knn_indices,
        src_knn_indices,
        score_mat,
        global_scores,
    ):
        r"""Extract point correspondences without estimating a pose.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            ref_knn_indices (LongTensor): (B, K)
            src_knn_indices (LongTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            ref_corr_points (Tensor): (C, 3)
            src_corr_points (Tensor): (C, 3)
            ref_corr_indices (LongTensor): (C,)
            src_corr_indices (LongTensor): (C,)
            corr_scores (Tensor): (C,)
        """
        score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        ref_corr_indices = ref_knn_indices[batch_indices, ref_indices]
        src_corr_indices = src_knn_indices[batch_indices, src_indices]
        ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        src_corr_points = src_knn_points[batch_indices, src_indices]
        corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        return ref_corr_points, src_corr_points, ref_corr_indices, src_corr_indices, corr_scores
