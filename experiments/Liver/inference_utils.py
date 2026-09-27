from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]


def select_device(device):
    device = torch.device(device)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        device = torch.device('cuda', index)
    return device


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module."):]: value for key, value in state_dict.items()}


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state_dict = strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'Checkpoint architecture does not match this model. '
            'Use a checkpoint trained with the same model configuration.'
        ) from error
    print(f"Loaded checkpoint: {checkpoint_path}")


def find_latest_checkpoint(exp_name):
    snapshot_root = ROOT_DIR / "output" / exp_name / "snapshots"
    if not snapshot_root.is_dir():
        return None
    # Run directories use YYYYMMDD-HHMMSS names; epochs need numeric ordering.
    run_dirs = sorted((path for path in snapshot_root.iterdir() if path.is_dir()), reverse=True)
    for run_dir in run_dirs:
        snapshot = run_dir / "snapshot.pth.tar"
        if snapshot.is_file():
            return snapshot
        epoch_candidates = []
        for path in run_dir.glob("epoch-*.pth.tar"):
            epoch = path.name[len("epoch-"):-len(".pth.tar")]
            if path.is_file() and epoch.isdecimal():
                epoch_candidates.append((int(epoch), path))
        if epoch_candidates:
            return max(epoch_candidates)[1]
    return None


def add_hecpg_neighbors(batch, model_cfg):
    from pareGeo.utils.data import precompute_neibors

    neighbor_data = precompute_neibors(
        batch["points"],
        batch["lengths"],
        model_cfg.backbone.num_stages,
        model_cfg.backbone.num_neighbors,
    )
    batch.update(neighbor_data)
    return batch
