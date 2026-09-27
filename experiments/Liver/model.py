import torch
import torch.nn as nn
import torch.nn.functional as F
from pareconv.modules.ops import point_to_node_partition, index_select
from pareconv.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
)

from feature_fusion import FeatureFusion
from geotransformer.modules.geotransformer.point_matching import PointMatching
from registration import RANSACRegistration


from backbone1 import KPConvFPN
from backbone2 import PAREConvFPN
from hyptorch.nn import ToPoincare
from hyptorch.hy_attention import HyperbolicAttention

torch.backends.cuda.enable_mem_efficient_sdp(False)

class PareNetHe(nn.Module):
    """MFHE registration with dual-backbone fusion and hyperbolic feature embedding."""

    def __init__(self, cfg):
        super(PareNetHe, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch 
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone1 = KPConvFPN(
            cfg.backbone1.input_dim,
            cfg.backbone1.output_dim,
            cfg.backbone1.init_dim,
            cfg.backbone1.kernel_size,
            cfg.backbone1.init_radius,
            cfg.backbone1.init_sigma,
            cfg.backbone1.group_norm
        )
        
        self.backbone2 = PAREConvFPN(
            cfg.backbone2.init_dim,
            cfg.backbone2.output_dim,
            cfg.backbone2.kernel_size,
            cfg.backbone2.share_nonlinearity,
            cfg.backbone2.conv_way,
            cfg.backbone2.use_xyz,
            cfg.fine_matching.use_encoder_re_feats
        )

        self.coarse_fusion = FeatureFusion(cfg.geotransformer.input_dim)
        self.fine_fusion = FeatureFusion(cfg.backbone1.output_dim)
        self.topoincare = ToPoincare(c=cfg.hyperbolic.curvature, ball_dim=cfg.geotransformer.input_dim,
                                    riemannian=False, clip_r=None)
        self.hyperbolic_input_scale = cfg.hyperbolic.input_scale
        self.hyperbolic_attention = HyperbolicAttention(cfg.geotransformer.input_dim)

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
            embedding_type=getattr(cfg.geotransformer, 'embedding_type', 'geometric'),
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = PointMatching(
            cfg.fine_matching.topk,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
        )
        self.ransac = RANSACRegistration(
            cfg.ransac.distance_threshold, cfg.ransac.num_points, cfg.ransac.num_iterations
        )
        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        # Read points and features at each resolution.
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]  
        src_points_c = points_c[ref_length_c:]  
        ref_points_f = points_f[:ref_length_f]  
        src_points_f = points_f[ref_length_f:] 
        ref_points = points[:ref_length]      
        src_points = points[ref_length:]       

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points


        # Generate ground-truth node correspondences.
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition( 
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(  
            src_points_f, src_points_c, self.num_points_in_patch
        )
        output_dict['ref_node_knn_indices'] = ref_node_knn_indices
        output_dict['src_node_knn_indices'] = src_node_knn_indices

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)  
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0) 
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)  
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)  

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(  
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # Extract features with KPConv and PAREConv backbones.
        feats_list = self.backbone1(feats, data_dict)
        re_feats_f_b2, feats_f_b2, re_feats_c_b2, feats_c_b2, m_scores = self.backbone2(feats, data_dict)

        feats_c_b1 = feats_list[-1] 
        feats_f_b1 = feats_list[0]  

        # Fuse each point cloud independently at coarse and fine resolutions.
        feats_c = self.coarse_fusion(feats_c_b1, feats_c_b2, ref_length_c)
        feats_f = self.fine_fusion(feats_f_b1, feats_f_b2, ref_length_f)

        # Embed coarse features in the Poincare ball before bilinear attention.
        ref_feats_c_hy = self.topoincare(feats_c[:ref_length_c] * self.hyperbolic_input_scale)
        src_feats_c_hy = self.topoincare(feats_c[ref_length_c:] * self.hyperbolic_input_scale)
        ref_feats_c, src_feats_c = self.hyperbolic_attention(
            ref_feats_c_hy.unsqueeze(0), src_feats_c_hy.unsqueeze(0)
        )
        ref_feats_c = ref_feats_c.squeeze(0)
        src_feats_c = src_feats_c.squeeze(0)

        ref_feats_c, src_feats_c = self.transformer(  
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )


        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1) 
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)  

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # Split fine-resolution features into reference and source clouds.
        ref_feats_f = feats_f[:ref_length_f]  
        src_feats_f = feats_f[ref_length_f:]  
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        m_ref_scores = m_scores[:ref_length_f]
        m_src_scores = m_scores[ref_length_f:]
        re_ref_feats_f_b2 = re_feats_f_b2[:ref_length_f]
        re_src_feats_f_b2 = re_feats_f_b2[ref_length_f:]

        output_dict['m_ref_scores'] = m_ref_scores
        output_dict['m_src_scores'] = m_src_scores
        output_dict['re_ref_feats_f'] = re_ref_feats_f_b2
        output_dict['re_src_feats_f'] = re_src_feats_f_b2


        # Select coarse correspondences by feature similarity.
        with torch.no_grad():

            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # Sample ground-truth node correspondences during training.
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(  
                    gt_node_corr_indices, gt_node_corr_overlaps
                )


        # Gather points and features for the selected node pairs.
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices] 
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices] 
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices] 
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices] 
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices] 

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)  
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0) 
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0) 
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)

        m_ref_padded_scores = torch.cat([m_ref_scores, torch.zeros_like(m_ref_scores[:1])], dim=0)
        m_src_padded_scores = torch.cat([m_src_scores, torch.zeros_like(m_src_scores[:1])], dim=0)
        ref_node_corr_knn_scores = index_select(m_ref_padded_scores, ref_node_corr_knn_indices, dim=0)  # (P, K)
        src_node_corr_knn_scores = index_select(m_src_padded_scores, src_node_corr_knn_indices, dim=0)  # (P, K)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        re_ref_padded_feats_f = torch.cat([re_ref_feats_f_b2, torch.zeros_like(re_ref_feats_f_b2[:1])], dim=0)
        re_src_padded_feats_f = torch.cat([re_src_feats_f_b2, torch.zeros_like(re_src_feats_f_b2[:1])], dim=0)
        re_ref_node_corr_knn_feats = index_select(re_ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C_re, 3)
        re_src_node_corr_knn_feats = index_select(re_src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C_re, 3)

        output_dict['re_ref_node_corr_knn_feats'] = re_ref_node_corr_knn_feats
        output_dict['re_src_node_corr_knn_feats'] = re_src_node_corr_knn_feats

        # Compute fine matching scores using scaled similarity and log optimal transport.
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(
            matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks
        )

        output_dict['matching_scores'] = matching_scores  # (P, K+1, K+1), log scores with dustbin
        output_dict['ref_node_corr_knn_scores'] = ref_node_corr_knn_scores
        output_dict['src_node_corr_knn_scores'] = src_node_corr_knn_scores

        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]
            ref_corr_points, src_corr_points, ref_corr_indices, src_corr_indices, corr_scores = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                ref_node_corr_knn_indices,
                src_node_corr_knn_indices,
                matching_scores,
                node_corr_scores,
            )

            estimated_transform, registration_success = self.ransac(src_corr_points, ref_corr_points)

        output_dict['registration_success'] = registration_success
        output_dict['ref_corr_points'] = ref_corr_points
        output_dict['src_corr_points'] = src_corr_points
        output_dict['corr_scores'] = corr_scores
        output_dict['estimated_transform'] = estimated_transform
        output_dict['transform'] = transform

        return output_dict


def create_model(config):
    model = PareNetHe(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
