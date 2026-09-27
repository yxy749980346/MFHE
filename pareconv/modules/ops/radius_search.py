import torch
from pareconv.extensions.pointops.functions import pointops

def radius_search(q_points, s_points, q_lengths, s_lengths, num_neighbors):
    r"""Compute k nearest neighbors for two stacked query/support point clouds.

    This function is implemented on GPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): lengths of the two clouds in q_points
        s_lengths (Tensor): lengths of the two clouds in s_points
        num_neighbors (int): number of neighbors

    Returns:
        neighbors (Tensor): indices into s_points of each query's k nearest neighbors (N, k).
    """
    q_pcd1 = q_points[:q_lengths[0]].unsqueeze(0)
    q_pcd2 = q_points[q_lengths[0]:].unsqueeze(0)

    s_pcd1 = s_points[:s_lengths[0]].unsqueeze(0)
    s_pcd2 = s_points[s_lengths[0]:].unsqueeze(0)


    ind_local1 = pointops.knnquery_heap(num_neighbors, s_pcd1, q_pcd1)
    ind_local2 = pointops.knnquery_heap(num_neighbors, s_pcd2, q_pcd2)

    ind_local2 = ind_local2 + s_lengths[0]
    index = torch.cat([ind_local1, ind_local2], 1)

    return index.squeeze(0)


