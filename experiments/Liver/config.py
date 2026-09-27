import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

_C.seed = 'seed'

_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

_C.data = edict()
_C.data.dataset = 'P2I-LReg'
_C.data.dataset_root = 'your-path'
_C.data.split_seed = _C.seed
_C.data.metadata_root = osp.join(_C.root_dir, 'data', 'LiverPointCloud', 'metadata')

_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 12
_C.train.point_limit = 10000
_C.train.use_augmentation = True
_C.train.augmentation_crop = True
_C.train.augmentation_noise = 0.001
_C.train.augmentation_rotation = 1.0
_C.train.point_keep_ratio = 0.7
_C.train.matching_radius = 0.03

_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None

_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.02
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.01
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.01
_C.eval.feat_rre_threshold = 20.0

_C.ransac = edict()
_C.ransac.distance_threshold = 0.01
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000 

_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 80
_C.optim.grad_acc_steps = 1

_C.backbone = edict()
_C.backbone.init_voxel_size = 0.003
_C.backbone.num_stages = 4
_C.backbone.subsample_ratio = 2
_C.backbone.num_neighbors = [35] * _C.backbone.num_stages

_C.backbone1 = edict()
_C.backbone1.base_radius = 2.5
_C.backbone1.base_sigma = 2.0
_C.backbone1.init_radius = _C.backbone1.base_radius * _C.backbone.init_voxel_size
_C.backbone1.init_sigma = _C.backbone1.base_sigma * _C.backbone.init_voxel_size
_C.backbone1.kernel_size = 16
_C.backbone1.group_norm = 12
_C.backbone1.input_dim = 1
_C.backbone1.init_dim = 48
_C.backbone1.output_dim = 255

_C.backbone2 = edict()
_C.backbone2.kernel_size = 4
_C.backbone2.share_nonlinearity = False
_C.backbone2.conv_way = 'edge_conv'
_C.backbone2.use_xyz = True
_C.backbone2.init_dim = 96
_C.backbone2.output_dim = 256

_C.model = edict()
_C.model.ground_truth_matching_radius = 0.03
_C.model.num_points_in_patch = 20
_C.model.num_sinkhorn_iterations = 100

_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

_C.hyperbolic = edict()
_C.hyperbolic.curvature = 0.1
_C.hyperbolic.input_scale = 1.0

_C.geotransformer = edict()
_C.geotransformer.input_dim = 768
_C.geotransformer.hidden_dim = 432  
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.embedding_type = 'relative_mlp'
_C.geotransformer.sigma_d = 0.05
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.02
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.005
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5
_C.fine_matching.use_encoder_re_feats = True
_C.fine_matching.num_hypotheses = 2000

_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.01
_C.fine_loss.alpha = 1.0
_C.fine_loss.gamma = 2.0
_C.fine_loss.negative_radius = 0.06
_C.fine_loss.positive_margin = 0.1
_C.fine_loss.negative_margin = 1.4

_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
