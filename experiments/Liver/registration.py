"""Rigid pose estimation from predicted correspondences using Open3D RANSAC."""
import numpy as np
import torch
from torch import nn


def estimate_ransac(src_points, ref_points, distance_threshold, num_points, num_iterations, confidence=None):
    src_points = np.asarray(src_points, dtype=np.float64)
    ref_points = np.asarray(ref_points, dtype=np.float64)
    if src_points.shape != ref_points.shape or src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError('Correspondences must be paired arrays of shape (N, 3).')
    if distance_threshold <= 0 or num_points < 3 or num_iterations < 1:
        raise ValueError('Invalid RANSAC distance, sample count, or iteration count.')
    if not np.isfinite(src_points).all() or not np.isfinite(ref_points).all():
        raise ValueError('Correspondences must contain only finite coordinates.')
    identity = np.eye(4, dtype=np.float64)
    if len(src_points) < num_points:
        return identity, False
    # Collinear points cannot determine a unique 3D rigid rotation.
    for points in (src_points, ref_points):
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            return identity, False

    import open3d as o3d

    source, reference = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_points)
    reference.points = o3d.utility.Vector3dVector(ref_points)
    indices = np.arange(len(src_points), dtype=np.int32)
    correspondences = o3d.utility.Vector2iVector(np.column_stack([indices, indices]))
    criteria_args = {'max_iteration': int(num_iterations)}
    if confidence is not None:
        criteria_args['confidence'] = confidence
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, reference, correspondences,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=num_points,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(**criteria_args),
    )
    transform = np.array(result.transformation, copy=True)
    success = result.fitness > 0 and np.isfinite(transform).all()
    return (transform if success else identity), bool(success)


def require_model_transform(output_dict):
    """Return a finite 4x4 transform from a successful model registration."""
    if not bool(output_dict.get('registration_success', False)):
        raise RuntimeError('Registration failed: the model did not estimate a valid rigid transform.')
    transform = output_dict['estimated_transform']
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise RuntimeError('Registration failed: expected a finite 4x4 transform.')
    return transform


def estimate_transform_from_output(output_dict, distance_threshold, num_points, num_iterations,
                                   confidence=None, topk_corr=None):
    """Estimate a rigid transform from score-ranked predicted correspondences."""
    src = output_dict['src_corr_points'].detach().cpu().numpy()
    ref = output_dict['ref_corr_points'].detach().cpu().numpy()
    scores = output_dict['corr_scores'].detach().cpu().numpy()
    order = np.argsort(-scores)
    if topk_corr is not None and topk_corr > 0:
        order = order[:int(topk_corr)]
    src, ref = src[order], ref[order]
    if len(src) < num_points:
        return require_model_transform(output_dict)
    transform, success = estimate_ransac(
        src, ref, distance_threshold, num_points, num_iterations, confidence=confidence,
    )
    if not success:
        raise RuntimeError('Registration failed: RANSAC could not estimate a valid rigid transform.')
    return torch.as_tensor(transform).to(output_dict['estimated_transform'])


class RANSACRegistration(nn.Module):
    def __init__(self, distance_threshold, num_points, num_iterations):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.num_points = num_points
        self.num_iterations = num_iterations

    @torch.no_grad()
    def forward(self, src_points, ref_points):
        transform, success = estimate_ransac(
            src_points.detach().cpu().numpy(), ref_points.detach().cpu().numpy(),
            self.distance_threshold, self.num_points, self.num_iterations,
        )
        return (torch.as_tensor(transform, dtype=src_points.dtype, device=src_points.device),
                torch.tensor(success, device=src_points.device))
