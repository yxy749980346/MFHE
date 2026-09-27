import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))


class PoseDataset(data.Dataset):
    """Load synthetic P2I-LReg pairs with source-to-target transforms in meters.

    Source points use the preoperative world frame; target points use the camera frame.
    Serialized target coordinates and camera translations are converted from mm to m.
    The loader reads train_syn for training and test_syn for all other modes.
    """

    def __init__(self, mode, data_root, real_syn='syn', max_points=10000, data_augmentation=False, augmentation_noise=0.001):

        self.patients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        self.mode = mode
        self.root = data_root
        self.real_syn = real_syn  # Stored as metadata; this loader always reads synthetic pairs.

        self._init_preprocessing(max_points, data_augmentation, augmentation_noise)

        self.list_liverPcd = []
        self.list_patient = []
        self.list_rank = []
        self.list_posemeta = {}
        self.pre_model = {}

        item_count = 0
        for patient in self.patients:
            if self.mode == 'train':
                input_file = open('{0}/{1}/train_syn.txt'.format(self.root, '%02d' % patient))
            else:
                input_file = open('{0}/{1}/test_syn.txt'.format(self.root, '%02d' % patient))

            while 1:
                item_count = item_count + 1
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]

                self.list_liverPcd.append('{0}/{1}/syn/liverPcds_pth/{2}.pth'.format(self.root, '%02d' % patient, input_line))
                self.list_patient.append(patient)
                self.list_rank.append(input_line[-5:])

            # Read the synthetic camera-to-world poses.
            pose_file = open('{0}/{1}/syn/camPose.yml'.format(self.root, '%02d' % patient), 'r')
            self.list_posemeta[patient] = yaml.safe_load(pose_file)

            # Preoperative source points are stored in meters.
            self.preope_mesh_path = '{0}/{1}/model/reconstructed_mesh_world_m.pth'.format(self.root, '%02d' % patient)
            preope_pcd = self._load_point_cloud(self.preope_mesh_path)
            self.pre_model[patient] = np.array(preope_pcd)

            print("Patient {0} buffer loaded".format(patient))

        self.length = len(self.list_liverPcd)
        print("{0} data is {1}".format(self.mode, self.length))

    def _init_preprocessing(self, max_points, data_augmentation, augmentation_noise=0.001):
        if max_points is not None and max_points <= 0:
            raise ValueError('max_points must be positive or None.')
        self.data_augmentation = data_augmentation
        if not np.isfinite(augmentation_noise) or augmentation_noise < 0:
            raise ValueError("augmentation_noise must be a finite nonnegative amplitude in meters.")
        self.augment_noise = augmentation_noise
        self.max_points = max_points
        self.overlap_radius = 0.0065
        self.rng = np.random

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

    def random_idx(self):
        n = self.length
        idx = self.rng.randint(0, n)
        return idx

    def _load_point_cloud(self, file_name):
        """Load a serialized point tensor and apply the optional point limit."""
        points = torch.load(file_name)
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[indices]
        return points

    def __getitem__(self, index):
        which_patient = self.list_patient[index]
        gt_pose = self.list_posemeta[which_patient]
        rank = self.list_rank[index]

        # Read the camera-to-world pose and convert its translation to meters.
        pose_R = np.array(gt_pose[rank][0]["cam_R_c2w"]).reshape(3, 3).astype(np.float32)
        pose_t = np.array(gt_pose[rank][0]["cam_t_c2w"]).reshape(3, 1).astype(np.float32) / 1000.0  # mm -> m

        # Invert the pose to map source world coordinates into the target camera frame.
        from lib.benchmark_utils import to_tsfm
        tsfm = np.linalg.inv(to_tsfm(pose_R, pose_t))
        intra_liver_pcd = self._load_point_cloud(self.list_liverPcd[index])
        return self._prepare_pair(self.pre_model[which_patient], intra_liver_pcd, tsfm, ref_unit_divisor=1000.0)

    def _prepare_pair(self, src_points, ref_points, transform, ref_unit_divisor=1.0):
        """Prepare a pair whose source coordinates and transform translation are in meters.

        Divide target coordinates by ref_unit_divisor to convert them to meters.
        """
        preope_pcd_src = np.array(src_points, dtype=np.float32, copy=True)
        if preope_pcd_src.ndim != 2 or preope_pcd_src.shape[1] != 3 or preope_pcd_src.shape[0] == 0:
            raise ValueError('Source point cloud must be a nonempty (N, 3) array.')
        if not np.isfinite(preope_pcd_src).all():
            raise ValueError('Source point cloud contains non-finite coordinates.')
        if ref_points.ndim != 2 or ref_points.shape[1] != 3 or ref_points.shape[0] == 0:
            raise ValueError('Target point cloud must be a nonempty (N, 3) tensor.')
        if not torch.isfinite(ref_points).all():
            raise ValueError('Target point cloud contains non-finite coordinates.')
        intra_liver_pcd = ref_points
        pose_R = np.array(transform[:3, :3], dtype=np.float32, copy=True)
        pose_t = np.array(transform[:3, 3:4], dtype=np.float32, copy=True)
        intra_liver_pcd = self.remove_duplicated_points(intra_liver_pcd)
        intra_liver_pcd, _ = self.remove_statistical_outlier(intra_liver_pcd, nb_neighbors=20, std_ratio=2.0)

        # Convert target coordinates to meters and remove zero-coordinate placeholders.
        intra_liver_pcd_tgt = intra_liver_pcd.numpy() / ref_unit_divisor
        intra_liver_pcd_tgt = intra_liver_pcd_tgt[~np.all(intra_liver_pcd_tgt == 0., axis=-1)]
        if intra_liver_pcd_tgt.shape[0] == 0:
            raise ValueError('Target point cloud has no valid points after preprocessing.')

        # Limit source points; sample or repeat target points to the requested size.
        if self.max_points is not None and preope_pcd_src.shape[0] > self.max_points:
            idx = self.rng.permutation(preope_pcd_src.shape[0])[:self.max_points]
            preope_pcd_src = preope_pcd_src[idx]

        if self.max_points is None:
            pass
        elif intra_liver_pcd_tgt.shape[0] > self.max_points:
            idx = self.rng.permutation(intra_liver_pcd_tgt.shape[0])[:self.max_points]
            intra_liver_pcd_tgt = intra_liver_pcd_tgt[idx]
        else:
            intra_liver_pcd_tgt = np.pad(
                intra_liver_pcd_tgt,
                ((0, self.max_points - intra_liver_pcd_tgt.shape[0]), (0, 0)),
                'wrap'
            )

        if self.data_augmentation:
            # Apply uniform target-coordinate noise using the configured amplitude in meters.
            # The stored source-to-target rigid transform is kept unchanged.
            intra_liver_pcd_tgt += self.rng.uniform(-self.augment_noise, self.augment_noise, intra_liver_pcd_tgt.shape)

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = pose_R
        transform[:3, 3:] = pose_t

        # src_points are preoperative; ref_points are the synthetic intraoperative target.
        data_dict = {
            'src_points': preope_pcd_src.astype(np.float32),
            'ref_points': intra_liver_pcd_tgt.astype(np.float32),
            'src_feats': np.ones((preope_pcd_src.shape[0], 1), dtype=np.float32),
            'ref_feats': np.ones((intra_liver_pcd_tgt.shape[0], 1), dtype=np.float32),
            'transform': transform,
        }

        return data_dict

    def remove_duplicated_points(self, points):
        """Return unique point coordinates."""
        unique_points = torch.unique(points, dim=0)
        return unique_points

    def remove_statistical_outlier(self, points, nb_neighbors=20, std_ratio=2.0):
        """Filter by mean neighbor distance and return retained points and their mask."""
        pts_np = points.cpu().numpy()

        if pts_np.shape[0] <= nb_neighbors:
            return points, torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

        from sklearn.neighbors import KDTree
        tree = KDTree(pts_np)
        dists, _ = tree.query(pts_np, k=nb_neighbors + 1)

        avg_knn_dists = dists[:, 1:].mean(axis=1)
        mean = avg_knn_dists.mean()
        std = avg_knn_dists.std()

        keep_mask = avg_knn_dists <= (mean + std_ratio * std)
        keep_mask_tensor = torch.from_numpy(keep_mask).to(points.device)

        filtered = points[keep_mask_tensor]
        return filtered, keep_mask_tensor

    def __len__(self):
        return self.length


if __name__ == '__main__':
    # Inspect one prepared synthetic pair.
    dataset = PoseDataset(
        mode='train',
        data_root='your-path',
        real_syn='syn',
        max_points=30000,
        data_augmentation=True
    )

    print(f"数据集大小: {len(dataset)}")

    data = dataset[0]
    print(f"src_points shape: {data['src_points'].shape}")
    print(f"ref_points shape: {data['ref_points'].shape}")
    print(f"src_feats shape: {data['src_feats'].shape}")
    print(f"ref_feats shape: {data['ref_feats'].shape}")
    print(f"transform shape: {data['transform'].shape}")
    print(f"transform:\n{data['transform']}")
