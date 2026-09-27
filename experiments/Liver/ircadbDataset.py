from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def split_samples(data_root, split_seed='seed'):
    """Reserve patients 03/05 for testing; split remaining samples into 80/20 train/val sets."""
    if split_seed == 'seed':
        raise ValueError('Replace the seed placeholder with an integer split_seed before loading the dataset.')
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {root}')

    development, test = [], []
    patients = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: (int(p.name), p.name),
    )
    for patient in patients:
        samples = sorted(
            (p for p in patient.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: (int(p.name), p.name),
        )
        if int(patient.name) in (3, 5):
            test.extend(samples)
        else:
            development.extend(samples)

    random.Random(split_seed).shuffle(development)
    num_train = int(len(development) * 0.8)
    return {
        'train': development[:num_train],
        'val': development[num_train:],
        'test': test,
    }


class IRCADbDataset(Dataset):
    """Load surface.stl as source and partialSurface.stl as target, in meters.

    Each pair shares a directory with its source-to-target transform.npy.
    STL vertices are used as points; no surface resampling is performed.
    """

    def __init__(self, mode, data_root, max_points=10000,
                 data_augmentation=False, split_seed='seed', augmentation_noise=0.001):
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown dataset split: {mode}')
        if max_points is not None and max_points <= 0:
            raise ValueError('max_points must be positive or None.')
        self.mode = mode
        self.root = Path(data_root)
        self.max_points = max_points
        self.data_augmentation = data_augmentation
        if not np.isfinite(augmentation_noise) or augmentation_noise < 0:
            raise ValueError("augmentation_noise must be a finite nonnegative amplitude in meters.")
        self.augment_noise = augmentation_noise
        self.rng = np.random
        self.samples = split_samples(self.root, split_seed)[mode]
        self.length = len(self.samples)
        if not self.samples:
            raise ValueError(f'No samples found for split {mode!r} under {self.root}')

    def __getstate__(self):
        state = self.__dict__.copy()
        # The process-local NumPy RNG is restored when the worker deserializes this dataset.
        if state['rng'] is np.random:
            state['rng'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.rng is None:
            self.rng = np.random

    def _load_stl_points(self, path):
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path))
        mesh.remove_duplicated_vertices()
        points = np.asarray(mesh.vertices, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f'Empty or invalid STL mesh: {path}')
        if not np.isfinite(points).all():
            raise ValueError(f'STL mesh contains non-finite coordinates: {path}')
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = self.rng.permutation(points.shape[0])[:self.max_points]
            points = points[indices]
        return points.copy()

    def __getitem__(self, index):
        sample = self.samples[index]
        source_path = sample / 'surface.stl'
        target_path = sample / 'partialSurface.stl'
        transform_path = sample / 'transform.npy'
        for path in (source_path, target_path, transform_path):
            if not path.is_file():
                raise FileNotFoundError(f'Missing sample file: {path}')

        transform = np.load(transform_path, allow_pickle=False)
        if transform.shape != (4, 4) or not np.issubdtype(transform.dtype, np.number) or np.iscomplexobj(transform):
            raise ValueError(f'Expected a real numeric 4x4 transform: {transform_path}')
        if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError(f'Invalid homogeneous transform: {transform_path}')

        src_points = self._load_stl_points(source_path)
        ref_points = torch.from_numpy(self._load_stl_points(target_path))
        data_dict = self._prepare_pair(src_points, ref_points, transform)
        return data_dict

    def _prepare_pair(self, src_points, ref_points, transform):
        """Preprocess and augment a pair whose coordinates and translation are in meters."""
        src_points = np.array(src_points, dtype=np.float32, copy=True)
        if src_points.ndim != 2 or src_points.shape[1] != 3 or src_points.shape[0] == 0:
            raise ValueError('Source point cloud must be a nonempty (N, 3) array.')
        if not np.isfinite(src_points).all():
            raise ValueError('Source point cloud contains non-finite coordinates.')
        if ref_points.ndim != 2 or ref_points.shape[1] != 3 or ref_points.shape[0] == 0:
            raise ValueError('Target point cloud must be a nonempty (N, 3) tensor.')
        if not torch.isfinite(ref_points).all():
            raise ValueError('Target point cloud contains non-finite coordinates.')
        pose_R = np.array(transform[:3, :3], dtype=np.float32, copy=True)
        pose_t = np.array(transform[:3, 3:4], dtype=np.float32, copy=True)
        ref_points = self.remove_duplicated_points(ref_points)
        ref_points, _ = self.remove_statistical_outlier(ref_points, nb_neighbors=20, std_ratio=2.0)
        ref_points = ref_points.numpy()
        ref_points = ref_points[~np.all(ref_points == 0., axis=-1)]
        if ref_points.shape[0] == 0:
            raise ValueError('Target point cloud has no valid points after preprocessing.')

        if self.max_points is not None:
            if src_points.shape[0] > self.max_points:
                indices = self.rng.permutation(src_points.shape[0])[:self.max_points]
                src_points = src_points[indices]
            if ref_points.shape[0] > self.max_points:
                indices = self.rng.permutation(ref_points.shape[0])[:self.max_points]
                ref_points = ref_points[indices]
            else:
                ref_points = np.pad(
                    ref_points,
                    ((0, self.max_points - ref_points.shape[0]), (0, 0)),
                    'wrap',
                )

        if self.data_augmentation:
            # Apply uniform target-coordinate noise using the configured amplitude in meters.
            # The stored source-to-target rigid transform is kept unchanged.
            ref_points += self.rng.uniform(-self.augment_noise, self.augment_noise, ref_points.shape)

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = pose_R
        transform[:3, 3:] = pose_t
        data_dict = {
            'src_points': src_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'transform': transform,
        }
        return data_dict

    def remove_duplicated_points(self, points):
        return torch.unique(points, dim=0)

    def remove_statistical_outlier(self, points, nb_neighbors=20, std_ratio=2.0):
        points_np = points.cpu().numpy()
        if points_np.shape[0] <= nb_neighbors:
            return points, torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

        from sklearn.neighbors import KDTree

        tree = KDTree(points_np)
        distances, _ = tree.query(points_np, k=nb_neighbors + 1)
        avg_distances = distances[:, 1:].mean(axis=1)
        threshold = avg_distances.mean() + std_ratio * avg_distances.std()
        keep_mask = torch.from_numpy(avg_distances <= threshold).to(points.device)
        return points[keep_mask], keep_mask

    def random_idx(self):
        return self.rng.randint(0, self.length)

    def __len__(self):
        return self.length
