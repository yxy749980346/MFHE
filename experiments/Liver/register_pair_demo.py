import argparse
import sys
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[1]

for path in [str(CURRENT_DIR), str(ROOT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from config import make_cfg
from model import create_model
from pareGeo.utils.data import registration_collate_fn_stack_mode
from inference_utils import load_checkpoint, find_latest_checkpoint, add_hecpg_neighbors, select_device
from registration import require_model_transform, estimate_transform_from_output


def move_batch_to_device(batch, device):
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            batch[key] = [item.to(device) for item in value]
    return batch


def make_parser():
    parser = argparse.ArgumentParser(
        description="Register two point clouds with a trained model and save the estimated rigid transform."
    )
    parser.add_argument("--src", required=True, help="Source point cloud path.")
    parser.add_argument("--ref", required=True, help="Reference/target point cloud path.")
    parser.add_argument("--checkpoint", default="your-path", help="Checkpoint path. Replace your-path with the actual checkpoint file path.")
    parser.add_argument(
        "--output-transform",
        default="transform.txt",
        help="Output transform path. Supports .txt/.csv/.npy and OpenCV .yml/.yaml.",
    )
    parser.add_argument(
        "--input-unit",
        choices=["m", "mm"],
        default="m",
        help="Unit of input point clouds. Model inference is always done in meters.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--transform-source",
        choices=["model", "ransac"],
        default="ransac",
        help="Use the model's estimated transform or estimate it from predicted correspondences with Open3D RANSAC.",
    )
    parser.add_argument(
        "--topk-corr",
        type=int,
        default=0,
        help="Use only top-K correspondences in ransac mode. <=0 means use all.",
    )
    parser.add_argument(
        "--voxel-downsample",
        type=float,
        default=0.0,
        help="Optional Open3D voxel downsample size in the same unit as the point clouds.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Randomly keep at most this many points per cloud after downsampling. Defaults to config point limit.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for point sampling. Defaults to config seed.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize source, reference and transformed source with Open3D.",
    )
    parser.add_argument(
        "--centerize",
        action="store_true",
        help="Center source and reference point clouds before registration, then restore the transform to the original coordinates.",
    )
    parser.add_argument(
        "--save-aligned-src",
        default="",
        help="Optional path to save the transformed source point cloud.",
    )
    parser.add_argument(
        "--mesh-sample-points",
        type=int,
        default=30000,
        help="If the input is a mesh file, sample this many points from the mesh surface.",
    )
    return parser


def maybe_import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is required for point cloud IO/visualization in this script.") from exc
    return o3d


def load_points_from_text(file_path):
    suffix = file_path.suffix.lower()
    delimiter = "," if suffix == ".csv" else None
    points = np.loadtxt(file_path, delimiter=delimiter, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        if points.size % 3 != 0:
            raise ValueError(f"Text file {file_path} does not contain a valid Nx3 array.")
        points = points.reshape(-1, 3)
    if points.shape[1] < 3:
        raise ValueError(f"Text file {file_path} must have at least 3 columns.")
    return points[:, :3]


def load_points(file_path, mesh_sample_points):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        points = np.load(file_path)
    elif suffix == ".npz":
        npz_data = np.load(file_path)
        if not npz_data.files:
            raise ValueError(f"No arrays found in {file_path}.")
        points = npz_data[npz_data.files[0]]
    elif suffix in {".txt", ".csv", ".xyz", ".pts"}:
        points = load_points_from_text(file_path)
    else:
        o3d = maybe_import_open3d()
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points, dtype=np.float32)
        else:
            mesh = o3d.io.read_triangle_mesh(str(file_path))
            if mesh.is_empty():
                raise ValueError(f"Unsupported or empty point cloud file: {file_path}")
            mesh.compute_vertex_normals()
            sampled = mesh.sample_points_uniformly(number_of_points=int(mesh_sample_points))
            points = np.asarray(sampled.points, dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud {file_path} must be an Nx3 or Nx>=3 array.")
    points = points[:, :3]
    if np.isnan(points).any() or np.isinf(points).any():
        raise ValueError(f"Point cloud {file_path} contains NaN or Inf values.")
    if points.shape[0] < 3:
        raise ValueError(f"Point cloud {file_path} must contain at least 3 points.")
    return points


def get_unit_scale(input_unit):
    if input_unit == "m":
        return 1.0
    if input_unit == "mm":
        return 1000.0
    raise ValueError(f"Unsupported input unit: {input_unit}")


def convert_points_to_model_unit(points, input_unit):
    scale = get_unit_scale(input_unit)
    return (points / scale).astype(np.float32)


def restore_transform_to_input_unit(transform, input_unit):
    restored_transform = np.array(transform, copy=True)
    scale = get_unit_scale(input_unit)
    restored_transform[:3, 3] *= scale
    return restored_transform


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0:
        return points
    o3d = maybe_import_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    down_points = np.asarray(down.points, dtype=np.float32)
    if down_points.shape[0] < 3:
        raise ValueError("Too few points remain after voxel downsampling.")
    return down_points


def preprocess_src_points(points, max_points, rng):
    if max_points is None or max_points <= 0:
        return points
    if points.shape[0] > max_points:
        indices = rng.permutation(points.shape[0])[:max_points]
        points = points[indices]
    return points


def preprocess_ref_points(points, max_points, rng):
    if max_points is None or max_points <= 0:
        return points
    if points.shape[0] > max_points:
        indices = rng.permutation(points.shape[0])[:max_points]
        points = points[indices]
    return points.astype(np.float32)


def centerize_points(src_points, ref_points):
    src_center = src_points.mean(axis=0).astype(np.float32)
    ref_center = ref_points.mean(axis=0).astype(np.float32)
    src_points_centered = (src_points - src_center).astype(np.float32)
    ref_points_centered = (ref_points - ref_center).astype(np.float32)
    center_meta = {
        "src_center": src_center,
        "ref_center": ref_center,
    }
    return src_points_centered, ref_points_centered, center_meta


def restore_transform_from_centerized(transform, center_meta):
    restored_transform = np.array(transform, copy=True)
    rotation = restored_transform[:3, :3]
    translation = restored_transform[:3, 3]
    src_center = center_meta["src_center"]
    ref_center = center_meta["ref_center"]
    restored_transform[:3, 3] = translation + ref_center - rotation @ src_center
    return restored_transform


def build_hecpg_input(src_points, ref_points, model_cfg):
    transform = np.eye(4, dtype=np.float32)
    sample_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": np.ones((ref_points.shape[0], 1), dtype=np.float32),
        "src_feats": np.ones((src_points.shape[0], 1), dtype=np.float32),
        "transform": transform,
    }

    return registration_collate_fn_stack_mode(
        [sample_dict],
        num_stages=model_cfg.backbone.num_stages,
        voxel_size=model_cfg.backbone.init_voxel_size,
        num_neighbors=model_cfg.backbone.num_neighbors,
        subsample_ratio=model_cfg.backbone.subsample_ratio,
        precompute_data=True,
    )


def estimate_transform_with_open3d(output_dict, src_points, ref_points, topk_corr=None):
    return estimate_transform_from_output(
        output_dict, distance_threshold=0.005, num_points=4,
        num_iterations=100000, confidence=0.999, topk_corr=topk_corr,
    )


def resolve_transform(output_dict, src_points, ref_points, transform_source, topk_corr=None):
    if transform_source == "model":
        return require_model_transform(output_dict)
    if transform_source == "ransac":
        return estimate_transform_with_open3d(output_dict, src_points, ref_points, topk_corr=topk_corr)
    raise ValueError(f"Unsupported transform source: {transform_source}")


def transform_points(points, transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation


def save_transform(transform, output_path):
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in {".npy", ".txt", ".csv", ".yml", ".yaml"}:
        raise ValueError(f"Unsupported transform format: {output_path.suffix}. Use .npy, .txt, .csv, .yml, or .yaml.")
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Transform must be a finite 4x4 matrix.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".npy":
        with output_path.open("wb") as f:
            np.save(f, transform, allow_pickle=False)
        return
    if suffix in {".txt", ".csv"}:
        np.savetxt(output_path, transform, delimiter="," if suffix == ".csv" else " ", fmt="%.18e")
        return

    flat_values = ", ".join(np.format_float_scientific(value, precision=16, unique=False, trim="k") for value in transform.reshape(-1))
    content = (
        "%YAML:1.0\n"
        "---\n"
        "M: !!opencv-matrix\n"
        f"   rows: {transform.shape[0]}\n"
        f"   cols: {transform.shape[1]}\n"
        "   dt: d\n"
        f"   data: [ {flat_values} ]\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def save_point_cloud(points, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".npy":
        np.save(output_path, points.astype(np.float32))
        return
    if suffix in {".txt", ".csv", ".xyz", ".pts"}:
        delimiter = "," if suffix == ".csv" else " "
        np.savetxt(output_path, points, fmt="%.8f", delimiter=delimiter)
        return

    o3d = maybe_import_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if not o3d.io.write_point_cloud(str(output_path), pcd):
        raise ValueError(f"Failed to save point cloud to {output_path}")


def visualize_registration(ref_points, aligned_src_points):
    o3d = maybe_import_open3d()

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points.astype(np.float64))
    ref_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_src_points.astype(np.float64))
    aligned_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries(
        [ref_pcd, aligned_pcd],
        window_name="MFHE Registration",
    )


def main():
    args = make_parser().parse_args()
    device = select_device(args.device)
    model_cfg = make_cfg()
    seed = model_cfg.seed if args.seed is None else args.seed
    if seed == 'seed':
        raise ValueError('Set an integer seed in config.py or pass --seed before running the demo.')
    max_points = model_cfg.test.point_limit
    if args.max_points is not None:
        max_points = args.max_points
    rng = np.random.default_rng(seed)

    src_points_raw = load_points(args.src, args.mesh_sample_points)
    ref_points_raw = load_points(args.ref, args.mesh_sample_points)

    src_points = convert_points_to_model_unit(src_points_raw, args.input_unit)
    ref_points = convert_points_to_model_unit(ref_points_raw, args.input_unit)

    voxel_size = args.voxel_downsample / get_unit_scale(args.input_unit)
    src_points = voxel_downsample(src_points, voxel_size)
    ref_points = voxel_downsample(ref_points, voxel_size)
    src_points = preprocess_src_points(src_points, max_points, rng)
    ref_points = preprocess_ref_points(ref_points, max_points, rng)
    center_meta = None
    if args.centerize:
        src_points, ref_points, center_meta = centerize_points(src_points, ref_points)

    print(f"Source points: {src_points.shape[0]}")
    print(f"Reference points: {ref_points.shape[0]}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(model_cfg.exp_name)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found. Please pass --checkpoint explicitly.")

    model = create_model(model_cfg).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    hecpg_inputs = build_hecpg_input(src_points, ref_points, model_cfg)
    hecpg_inputs = move_batch_to_device(hecpg_inputs, device)
    hecpg_inputs = add_hecpg_neighbors(hecpg_inputs, model_cfg)

    with torch.no_grad():
        output_dict = model(hecpg_inputs)
        transform = resolve_transform(
            output_dict,
            src_points,
            ref_points,
            args.transform_source,
            topk_corr=args.topk_corr,
        )

    transform = transform.detach().cpu().numpy()
    if args.centerize:
        transform = restore_transform_from_centerized(transform, center_meta)
    output_transform = restore_transform_to_input_unit(transform, args.input_unit)
    aligned_src_points_output = transform_points(src_points_raw, output_transform)

    save_transform(output_transform, args.output_transform)
    print(f"Saved transform to: {args.output_transform}")
    print(f"Input unit: {args.input_unit}")
    print(f"Config point limit: {max_points}")
    print(f"Centerize before registration: {args.centerize}")
    print("Estimated transform:")
    print(output_transform)

    if args.save_aligned_src:
        save_point_cloud(aligned_src_points_output, args.save_aligned_src)
        print(f"Saved transformed source point cloud to: {args.save_aligned_src}")

    if args.visualize:
        visualize_registration(ref_points_raw, aligned_src_points_output)


if __name__ == "__main__":
    main()
