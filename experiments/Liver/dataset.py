from poseDataset import PoseDataset
from ircadbDataset import IRCADbDataset
from pareGeo.utils.data import registration_collate_fn_stack_mode, build_dataloader_stack_mode


def build_pair_dataset(cfg, mode):
    """Build the selected dataset with split-specific preprocessing."""
    max_points = cfg.train.point_limit if mode == 'train' else cfg.test.point_limit
    kwargs = dict(
        data_root=cfg.data.dataset_root,
        max_points=max_points,
        data_augmentation=mode == 'train' and cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
    )
    dataset_name = getattr(cfg.data, 'dataset', 'P2I-LReg').lower()
    if dataset_name == 'p2i-lreg':
        # The provided P2I-LReg test split is also used for validation.
        return PoseDataset(mode='train' if mode == 'train' else 'test', real_syn='syn', **kwargs)
    if dataset_name == '3dircadb':
        return IRCADbDataset(mode=mode, split_seed=getattr(cfg.data, 'split_seed', cfg.seed), **kwargs)
    raise ValueError(f'Unknown dataset: {dataset_name}')


def train_valid_data_loader(cfg, distributed):
    """Build training and validation loaders with multiscale point neighborhoods."""
    train_dataset = build_pair_dataset(cfg, 'train')

    # Use configured neighbor limits.
    neighbor_limits = cfg.backbone.num_neighbors

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        neighbor_limits,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=True
    )

    valid_dataset = build_pair_dataset(cfg, 'val')

    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        neighbor_limits,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark=None):
    """Build the test loader with the configured neighborhood limits."""
    test_dataset = build_pair_dataset(cfg, 'test')

    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, cfg.backbone.num_neighbors
