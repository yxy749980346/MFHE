import argparse
import csv
import sys
from pathlib import Path
import types

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[1]
SELF_P2IR_ROOT = Path('your-path')

for path in [str(CURRENT_DIR), str(ROOT_DIR), str(SELF_P2IR_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from config import make_cfg
from model import create_model
from pareGeo.utils.data import registration_collate_fn_stack_mode
from inference_utils import load_checkpoint, find_latest_checkpoint, add_hecpg_neighbors, select_device
from registration import require_model_transform, estimate_transform_from_output
import importlib.util


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f'Cannot load module from {file_path}')
    spec.loader.exec_module(module)
    return module


selfp2ir_configs_models = load_module_from_file(
    'selfp2ir_configs_models', SELF_P2IR_ROOT / 'configs' / 'models.py'
)
selfp2ir_dataset_module = load_module_from_file(
    'selfp2ir_dataset_module', SELF_P2IR_ROOT / 'datasets' / 'dataset.py'
)
selfp2ir_mesh_render_module = load_module_from_file(
    'selfp2ir_mesh_render_module', SELF_P2IR_ROOT / 'models' / 'mesh_render.py'
)

architectures = selfp2ir_configs_models.architectures
SelfP2IRPoseDataset = selfp2ir_dataset_module.PoseDataset
MeshRender = selfp2ir_mesh_render_module.MeshRender


def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join(map(str, seq))


yaml.add_constructor('!join', join)


def load_render_config(path, device):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = edict(cfg)
    cfg.kpfcn_config.architecture = architectures[cfg.dataset]
    cfg.device = torch.device(device)
    return cfg


def move_batch_to_device(batch, device):
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
        elif isinstance(value, list) and len(value) > 0 and torch.is_tensor(value[0]):
            batch[key] = [item.to(device) for item in value]
    return batch


def binary_dice_coeff(pred_mask, gt_mask, smooth=1.0, p=2):
    """Compute the smoothed Dice coefficient for each mask pair."""
    pred_mask = pred_mask.float().contiguous().view(pred_mask.shape[0], -1)
    gt_mask = gt_mask.float().contiguous().view(gt_mask.shape[0], -1)
    intersection = torch.sum(pred_mask * gt_mask, dim=1)
    denom = torch.sum(pred_mask.pow(p) + gt_mask.pow(p), dim=1) + smooth
    dice = (2.0 * intersection + smooth) / denom
    return dice


def prepare_image_for_save(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_overlay(image, pred_mask, gt_mask, save_path):
    image = prepare_image_for_save(image)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]

    pred_vis = (pred_mask > 0.5).astype(np.uint8) * 255
    gt_vis = (gt_mask > 0.5).astype(np.uint8) * 255

    overlay = image.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = pred_vis
    red = np.zeros_like(overlay)
    red[..., 2] = gt_vis
    overlay = cv2.addWeighted(overlay, 1.0, green, 0.35, 0)
    overlay = cv2.addWeighted(overlay, 1.0, red, 0.25, 0)
    imageio.imwrite(save_path, overlay)


def patch_renderer_setup(renderer):
    from pytorch3d.renderer import (
        PerspectiveCameras,
        AmbientLights,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        AlphaCompositor,
    )

    def setup_renderer(self, camera):
        if camera is None:
            raise ValueError('camera must not be None')

        w2c = camera['w2c']
        c2w = torch.linalg.inv(w2c)
        R, T = c2w[:3, :3], c2w[:3, 3:]
        R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)
        new_c2w = torch.cat([R, T], 1)
        w2c = torch.linalg.inv(
            torch.cat((new_c2w, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)), 0)
        )
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
        R = R[None]
        T = T[None]

        H, W = int(camera['H']), int(camera['W'])
        intrinsics = camera['intrinsics']
        image_size = ((H, W),)
        fcl_screen = ((intrinsics[0][0], intrinsics[1][1]),)
        prp_screen = ((intrinsics[0][2], intrinsics[1][2]),)

        cameras = PerspectiveCameras(
            focal_length=fcl_screen,
            principal_point=prp_screen,
            in_ndc=False,
            image_size=image_size,
            R=R,
            T=T,
            device=self.device,
        )
        raster_settings = PointsRasterizationSettings(
            image_size=image_size[0],
            radius=0.02,
            points_per_pixel=1,
            bin_size=0,
        )
        lights = AmbientLights(device=self.device)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer_mod = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        return {
            'cameras': cameras,
            'raster_settings': raster_settings,
            'lights': lights,
            'rasterizer': rasterizer,
            'renderer': renderer_mod,
        }

    renderer.setup_renderer = types.MethodType(setup_renderer, renderer)
    return renderer


def build_dataset(cfg, split):
    mode = 'train' if split == 'train' else 'test'
    return SelfP2IRPoseDataset(mode, cfg, data_augmentation=False)


def build_hecpg_input(sample, model_cfg):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = sample['pose_R']
    transform[:3, 3:] = sample['pose_t']

    sample_dict = {
        'ref_points': sample['intra_liver_pcd_tgt'],
        'src_points': sample['preope_pcd_src'],
        'ref_feats': sample['tgt_feats'],
        'src_feats': sample['src_feats'],
        'transform': transform,
    }

    return registration_collate_fn_stack_mode(
        [sample_dict],
        num_stages=model_cfg.backbone.num_stages,
        voxel_size=model_cfg.backbone.init_voxel_size,
        num_neighbors=model_cfg.backbone.num_neighbors,
        subsample_ratio=model_cfg.backbone.subsample_ratio,
        precompute_data=True,
    )


def build_render_tensors(sample, device):
    render_inputs = {
        'batched_src_pcd': torch.from_numpy(sample['preope_pcd_src']).float().unsqueeze(0).to(device),
        'rgbs': torch.from_numpy(sample['rgbs']).float().unsqueeze(0).to(device),
        'img_size': [torch.from_numpy(sample['img_size']).to(device)],
        'cam_k': torch.from_numpy(sample['cam_K']).float().unsqueeze(0).to(device),
        'ocv2blender': torch.from_numpy(sample['ocv2blender']).float().unsqueeze(0).to(device),
        'scale': torch.from_numpy(np.atleast_1d(sample['scale']).astype(np.float32)).float().to(device),
        'bbx_center': torch.from_numpy(sample['bbx_center']).float().unsqueeze(0).to(device),
        'liver_label': torch.from_numpy(sample['liver_labels']).float().unsqueeze(0).to(device),
        'ori_imgs': torch.from_numpy(sample['liver_imgs']).unsqueeze(0).to(device),
        'which_patient': int(np.asarray(sample['which_patient']).item()),
    }
    return render_inputs


def extract_pose_from_transform(estimated_transform):
    if estimated_transform.ndim == 3:
        estimated_transform = estimated_transform[0]
    pred_R = estimated_transform[:3, :3].unsqueeze(0)
    pred_t = estimated_transform[:3, 3].reshape(1, 3, 1)
    return pred_R, pred_t


def estimate_transform_with_open3d(output_dict, sample, model_cfg, topk_corr=None):
    return estimate_transform_from_output(
        output_dict, distance_threshold=0.008, num_points=3,
        num_iterations=100000, confidence=0.999, topk_corr=topk_corr,
    )


def resolve_transform(output_dict, sample, model_cfg, transform_source, topk_corr=None):
    if transform_source == 'model':
        return require_model_transform(output_dict)
    if transform_source == 'ransac':
        return estimate_transform_with_open3d(output_dict, sample, model_cfg, topk_corr=topk_corr)
    raise ValueError(f'Unsupported transform source: {transform_source}')


def project_with_renderer(renderer, render_inputs, pred_R, pred_t, apply_scale_correction=True):
    src = render_inputs['batched_src_pcd']
    rigid_src = torch.bmm(src, pred_R.permute(0, 2, 1)) + pred_t.permute(0, 2, 1)
    rigid_src = torch.bmm(render_inputs['ocv2blender'], rigid_src.permute(0, 2, 1)).permute(0, 2, 1)

    if apply_scale_correction:
        center = render_inputs['bbx_center'].unsqueeze(1)
        scale = 1.0 / render_inputs['scale'].unsqueeze(1)
        scale = scale.unsqueeze(-1).repeat(1, rigid_src.shape[1], 1)
        rigid_src = rigid_src - center
        rigid_src = rigid_src * scale
        rigid_src = rigid_src + center

    pose = {'rot': pred_R, 'trans': pred_t}
    pred_mask, pred_depth = renderer(
        rigid_src,
        pose,
        render_inputs['rgbs'],
        render_inputs['img_size'],
        render_inputs['cam_k'],
        render_inputs,
    )
    return pred_mask, pred_depth


def make_parser():
    parser = argparse.ArgumentParser(description='Rigid projection Dice evaluation on Self-P2IR real data.')
    parser.add_argument('--config', default='your-path', help='Self-P2IR config YAML.')
    parser.add_argument('--data-root', default='your-path', help='Dataset root.')
    parser.add_argument('--checkpoint', default='', help='Model checkpoint path.')
    parser.add_argument('--output-dir', default='output', help='Directory for results.')
    parser.add_argument('--device', default='cuda:0', help='Torch device.')
    parser.add_argument('--patient', type=int, default=None, help='Optional patient ID filter.')
    parser.add_argument('--save-vis', action='store_true', help='Save overlay images.')
    parser.add_argument('--split', choices=['test', 'train'], default='test', help='Dataset split to evaluate.')
    parser.add_argument('--transform-source', choices=['model', 'ransac'], default='model', help="Use the model's estimated transform or estimate it from predicted correspondences with Open3D RANSAC.")
    parser.add_argument('--topk-corr', type=int, default=0, help='Use only top-K highest-confidence correspondences in the ransac branch. Set <=0 to use all correspondences.')
    parser.add_argument('--disable-scale-correction', action='store_true', help='Disable Self-P2IR scale/bbox correction before rendering.')
    return parser


def main():
    args = make_parser().parse_args()
    device = select_device(args.device)

    render_cfg = load_render_config(args.config, device)
    render_cfg.data_root = args.data_root
    render_cfg.dataset = 'real'
    render_cfg.mode = 'test'
    render_cfg.batch_size = 1
    render_cfg.num_workers = 0
    render_cfg.local_rank = 0
    render_cfg.gpus = 1

    model_cfg = make_cfg()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(model_cfg.exp_name)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError('Checkpoint not found. Please pass --checkpoint explicitly.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading dataset...')
    dataset = build_dataset(render_cfg, args.split)

    print('Building model...')
    model = create_model(model_cfg).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    print('Building renderer...')
    renderer = MeshRender(render_cfg).to(device)
    patch_renderer_setup(renderer)
    renderer.eval()

    rows = []
    all_dice = []

    with torch.no_grad():
        for sample_idx in tqdm(range(len(dataset)), desc='Evaluating'):
            sample = dataset[sample_idx]
            which_patient = int(np.asarray(sample['which_patient']).item())
            if args.patient is not None and which_patient != args.patient:
                continue

            hecpg_inputs = build_hecpg_input(sample, model_cfg)
            hecpg_inputs = move_batch_to_device(hecpg_inputs, device)
            hecpg_inputs = add_hecpg_neighbors(hecpg_inputs, model_cfg)
            output_dict = model(hecpg_inputs)

            transform = resolve_transform(output_dict, sample, model_cfg, args.transform_source, topk_corr=args.topk_corr)
            pred_R, pred_t = extract_pose_from_transform(transform)
            render_inputs = build_render_tensors(sample, device)
            pred_mask, _ = project_with_renderer(
                renderer,
                render_inputs,
                pred_R,
                pred_t,
                apply_scale_correction=not args.disable_scale_correction,
            )

            gt_mask = (render_inputs['liver_label'] > 0).float()
            pred_mask_soft = pred_mask.float()
            dice = binary_dice_coeff(pred_mask_soft, gt_mask)
            dice_val = float(dice.mean().item())
            all_dice.append(dice_val)

            label_path = Path(dataset.list_label[sample_idx])
            frame_name = label_path.parent.name
            save_name = f'patient_{which_patient:02d}_{frame_name}.png'

            rows.append({
                'patient': which_patient,
                'frame': frame_name,
                'dice': dice_val,
            })

            if args.save_vis:
                image = render_inputs['ori_imgs'][0].detach().cpu().numpy()
                gt_np = gt_mask[0].detach().cpu().numpy()
                pred_np = pred_mask_soft[0].detach().cpu().numpy()
                save_overlay(image, pred_np, gt_np, output_dir / save_name)

    csv_path = output_dir / 'dice_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient', 'frame', 'dice'])
        writer.writeheader()
        writer.writerows(rows)

    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'Frames evaluated: {len(all_dice)}\n')
        f.write(f'Mean Dice: {mean_dice:.6f}\n')
        f.write('Render radius: 0.02\n')
        f.write(f'Transform source: {args.transform_source}\n')
        f.write(f'Top-K correspondences: {args.topk_corr}\n')
        f.write(f'RANSAC distance threshold: {0.008 if args.transform_source == "ransac" else model_cfg.ransac.distance_threshold}\n')
        f.write(f'Scale correction: {not args.disable_scale_correction}\n')

    print(f'Saved results to: {csv_path}')
    print(f'Frames evaluated: {len(all_dice)}')
    print(f'Mean Dice: {mean_dice:.6f}')


if __name__ == '__main__':
    main()
