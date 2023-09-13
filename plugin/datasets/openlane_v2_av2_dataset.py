import os
import tempfile
import warnings
import random
import copy

import numpy as np
import torch
import mmcv
import cv2

from math import factorial
from pyquaternion import Quaternion
from shapely.geometry import LineString, box, Polygon
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

def fix_pts_interpolate(lane, n_points):
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane


@DATASETS.register_module()
class OpenLaneV2_Av2_Dataset(Custom3DDataset):
    CAMS = ('ring_front_center', 'ring_front_left', 'ring_front_right',
            'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right')
    LANE_CLASSES = ('centerline')
    TE_CLASSES = ('traffic_light', 'road_sign')
    ATTR_CLASSES = ('unknown', 'red', 'green', 'yellow',
                    'go_straight', 'turn_left', 'turn_right',
                    'no_left_turn', 'no_right_turn', 'u_turn', 'no_u_turn',
                    'slight_left', 'slight_right')
    MAP_CHANGE_LOGS = [
        '75e8adad-50a6-3245-8726-5e612db3d165',
        '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
        'af170aac-8465-3d7b-82c5-64147e94af7d',
        '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 queue_length=1,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 filter_empty_te=False,
                 split='train',
                 filter_map_change=False,
                 test_mode=False):
        self.filter_map_change = filter_map_change
        self.split = split
        super().__init__(data_root, ann_file, pipeline, classes, modality, box_type_3d, filter_empty_gt, test_mode)
        self.queue_length = queue_length
        self.filter_empty_te = filter_empty_te

        if not os.path.exists(self.data_root):
            print('change data_root to s3://odl-flat/Argoverse2/Sensor_Dataset/sensor/')
            self.data_root = 's3://odl-flat/Argoverse2/Sensor_Dataset/sensor/'

    def load_annotations(self, ann_file):
        """Load annotation from a json file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: Annotation info from the json file.
        """
        data_infos = mmcv.load(ann_file, file_format='pkl')
        if isinstance(data_infos, dict):
            # self.raw_data_keys = list(data_infos.keys())
            if self.filter_map_change and self.split == 'train':
                data_infos = [info for info in data_infos.values() if info['meta_data']['source_id'] not in self.MAP_CHANGE_LOGS]
            else:
                data_infos = list(data_infos.values())
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # pts_filename = os.path.join(self.data_root, info['lidar_path'])
        input_dict = dict(
            # pts_filename=pts_filename,
            sample_idx=info['timestamp'],  # use timestamp as sample_idx
            scene_token=info['segment_id']
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_name, cam_info in info['sensor'].items():
                # tmp fix for raw av2 dataset
                if self.data_root.startswith('s3://odl-flat/'):
                    image_path = cam_info['image_path'].replace(
                        f"{info['segment_id']}/image", f"{info['meta_data']['source_id']}/sensors/cameras", 1)
                else:
                    image_path = cam_info['image_path']

                image_paths.append(os.path.join(self.data_root, image_path))
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['extrinsic']['rotation'])
                lidar2cam_t = cam_info['extrinsic']['translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = np.array(cam_info['intrinsic']['K'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_lane_labels_3d']) == 0:
                return None
            if self.filter_empty_te and len(annos['labels']) == 0:
                return None

        can_bus = np.zeros(18)
        rotation = Quaternion._from_matrix(np.array(info['pose']['rotation']))
        can_bus[:3] = info['pose']['translation']
        can_bus[3:7] = rotation
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus
        # input_dict['ego2global_rotation'] = np.array(info['pose']['rotation'])
        # input_dict['ego2global_translation'] = np.array(info['pose']['translation'])
        
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information
        """
        info = self.data_infos[index]
        ann_info = info['annotation']

        gt_lanes = [np.array(lane['points'], dtype=np.float32) for lane in ann_info['lane_centerline']]
        gt_lane_labels_3d = np.zeros(len(gt_lanes), dtype=np.int64)
        lane_adj = np.array(ann_info['topology_lclc'], dtype=np.float32)

        # only use traffic light attribute
        te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
        te_labels = np.array([sign['attribute'] for sign in ann_info['traffic_element']], dtype=np.int64)
        if len(te_bboxes) == 0:
            te_bboxes = np.zeros((0, 4), dtype=np.float32)
            te_labels = np.zeros((0, ), dtype=np.int64)

        lane_lcte_adj = np.array(ann_info['topology_lcte'], dtype=np.float32)

        assert len(gt_lanes) == lane_adj.shape[0]
        assert len(gt_lanes) == lane_adj.shape[1]
        assert len(gt_lanes) == lane_lcte_adj.shape[0]
        assert len(te_bboxes) == lane_lcte_adj.shape[1]

        annos = dict(
            gt_lanes_3d = gt_lanes,
            gt_lane_labels_3d = gt_lane_labels_3d,
            gt_lane_adj = lane_adj,
            bboxes = te_bboxes,
            labels = te_labels,
            gt_lane_lcte_adj = lane_lcte_adj
        )
        return annos

    def prepare_train_data(self, index):
        data_queue = []
        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)


        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        sample_idx = input_dict['sample_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or len(example['gt_lane_labels_3d']._data) == 0):
            return None
        if self.filter_empty_te and \
                (example is None or len(example['gt_labels']._data) == 0):
            return None

        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['sample_idx'] < sample_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                    (example is None or len(example['gt_lane_labels_3d']._data) == 0):
                    return None
                sample_idx = input_dict['sample_idx']
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def format_openlanev2_gt(self):
        gt_dict = {}
        for idx in range(len(self.data_infos)):
            info = copy.deepcopy(self.data_infos[idx])
            key = (self.split, info['segment_id'], str(info['timestamp']))
            for lane in info['annotation']['lane_centerline']:
                if len(lane['points']) == 201:
                    lane['points'] = lane['points'][::20]  # downsample points: 201 --> 11

            gt_dict[key] = info
        return gt_dict

    def format_results(self, results, jsonfile_prefix=None):
        bezier_A = None
        pred_dict = {}
        pred_dict['method'] = 'TopoNet'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'litianyu@pjlab.org.cn'
        pred_dict['institution / company'] = 'Shanghai AI Laboratory'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        for idx, result in enumerate(results):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))

            pred_info = dict(
                lane_centerline=[],
                traffic_element=[],
                topology_lclc=None,
                topology_lcte=None
            )

            if result['bbox_results'] is not None:
                te_results = result['bbox_results']
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id = 20000 + pred_idx,
                        category = 1 if class_idx < 4 else 2,
                        attribute = class_idx,
                        points = te.reshape(2, 2).astype(np.float32),
                        confidence = score
                    )
                    pred_info['traffic_element'].append(te_info)

            if result['lane_results'] is not None:
                lane_results = result['lane_results']
                scores = lane_results[1]
                valid_indices = np.argsort(-scores)
                lanes = lane_results[0][valid_indices]
                lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)
                if lanes.shape[1] in [3, 5]: ## hardcode for bezier_interpolate

                    if bezier_A is None:
                        def comb(n, k):
                            return factorial(n) // (factorial(k) * factorial(n - k))
                        n_points = 101
                        n_control = lanes.shape[1]
                        A = np.zeros((n_points, n_control))
                        t = np.arange(n_points) / (n_points - 1)
                        for i in range(n_points):
                            for j in range(n_control):
                                A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
                        bezier_A = torch.tensor(A, dtype=torch.float32, device='cpu')

                    lanes = torch.tensor(lanes, dtype=torch.float32, device='cpu')
                    lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
                    lanes = lanes.numpy()

                scores = scores[valid_indices]
                for pred_idx, (lane, score) in enumerate(zip(lanes, scores)):
                    points = fix_pts_interpolate(lane, 11)
                    lc_info = dict(
                        id = 10000 + pred_idx,
                        points = points.astype(np.float32),
                        confidence = score.item()
                    )
                    pred_info['lane_centerline'].append(lc_info)

            if result['lclc_results'] is not None:
                pred_info['topology_lclc'] = result['lclc_results'].astype(np.float32)[valid_indices][:, valid_indices]
            else:
                pred_info['topology_lclc'] = np.zeros((len(pred_info['lane_centerline']), len(pred_info['lane_centerline'])), dtype=np.float32)

            if result['lcte_results'] is not None:
                pred_info['topology_lcte'] = result['lcte_results'].astype(np.float32)[valid_indices][:, te_valid_indices]
            else:
                pred_info['topology_lcte'] = np.zeros((len(pred_info['lane_centerline']), len(pred_info['traffic_element'])), dtype=np.float32)

            pred_dict['results'][key] = dict(predictions=pred_info)

        return pred_dict

    # def evaluate(self, results, logger=None, show=False, out_dir=None, **kwargs):
    #     """Evaluation in OpenlaneV2 av2 dataset.

    #     Args:
    #         results (list): Testing results of the dataset.
    #         metric (str): Metric to be performed.
    #         iou_thr (float): IoU threshold for evaluation.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         show (bool): Whether to visualize the results.
    #         out_dir (str): Path of directory to save the results.
    #         pipeline (list[dict]): Processing pipeline.

    #     Returns:
    #         dict: Evaluation results for evaluation metric.
    #     """
    #     if show:
    #         assert out_dir, 'Expect out_dir when show is set.'
    #         logger.info(f'Visualizing results at {out_dir}...')
    #         self.show(results, out_dir)
    #         logger.info(f'Visualize done.')

    #     logger.info(f'Starting format results...')
    #     gt_dict = self.format_openlanev2_gt()
    #     pred_dict = self.format_results(results)

    #     logger.info(f'Starting openlanev2 evaluate...')
    #     metric_results = openlanev2_evaluate(gt_dict, pred_dict)
    #     metric_results = format_metric(metric_results)
    #     return metric_results

    # def show(self, results, out_dir, score_thr=0.3, show_num=20, **kwargs):
    #     """Show the results.

    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         out_dir (str): Path of directory to save the results.
    #         score_thr (float): The threshold of score.
    #         show_num (int): The number of images to be shown.
    #     """
    #     n_show = 0
    #     for idx, result in enumerate(results):
    #         if idx % 6 != 0:
    #             continue
    #         if n_show >= show_num:
    #             break
    #         n_show += 1

    #         # calculate metric
    #         pred_result = self.format_results([result])
    #         key = list(pred_result['results'].keys())[0]

    #         info = copy.deepcopy(self.data_infos[idx])
    #         for lane in info['annotation']['lane_centerline']:
    #             if len(lane['points']) == 201:
    #                 lane['points'] = lane['points'][::20]  # downsample points: 201 --> 11

    #         gt_result = {key: info}
    #         metric_results = openlanev2_evaluate(gt_result, pred_result, verbose=False)
    #         metric_str = []
    #         for k, v in metric_results['OpenLane-V2 Score'].items():
    #             if k == 'score':
    #                 continue
    #             metric_str.append(f'{k}: {(lambda x: "%.2f" % x)(v)}')
    #         metric_str = ' / '.join(metric_str)
    #         # metric_results = format_metric(metric_results, verbose=False)

    #         result = list(pred_result['results'].values())[0]['predictions']
    #         input_dict = self.get_data_info(idx)
    #         img_paths = input_dict['img_filename']
    #         lidar2imgs = input_dict['lidar2img']
    #         images = [mmcv.imread(img_path) for img_path in img_paths]

    #         gt_lanes = []
    #         for lane in info['annotation']['lane_centerline']:
    #             gt_lanes.append(lane['points'])
    #         gt_lclc = info['annotation']['topology_lclc']

    #         gt_tes = []
    #         for te in info['annotation']['traffic_element']:
    #             gt_tes.append((te['points'].flatten(), te['attribute']))


    #         pred_lanes = []
    #         lane_scores = []
    #         for lane in result['lane_centerline']:
    #             pred_lanes.append(lane['points'])
    #             lane_scores.append(lane['confidence'])
    #         pred_lanes = np.array(pred_lanes)
    #         lane_scores = np.array(lane_scores)
    #         mask = lane_scores > score_thr
    #         pred_lanes = pred_lanes[mask]
    #         pred_lclc = result['topology_lclc'][mask][:, mask] >= 0.5

    #         pred_tes = []
    #         if result['traffic_element'] is not None:
    #             for pred_te in result['traffic_element']:
    #                 if pred_te['confidence'] < score_thr:
    #                     continue
    #                 class_idx = pred_te['attribute']
    #                 te = pred_te['points'].flatten()
    #                 pred_tes.append((te, class_idx))

    #         images = show_results(images, lidar2imgs, gt_lanes, pred_lanes, gt_tes, pred_tes)
    #         for cam_idx, image in enumerate(images[:1]):
    #             output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/{self.CAMS[cam_idx]}.jpg')
    #             mmcv.imwrite(image, output_path)

    #         surround_img = self._render_surround_img(images)
    #         output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/surround.jpg')
    #         mmcv.imwrite(surround_img, output_path)

    #         bev_img = show_bev_results(gt_lanes, pred_lanes, map_size=[-52, 55, -27, 27], scale=20)
    #         bev_img = cv2.putText(bev_img, metric_str, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #         output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/bev.jpg')
    #         mmcv.imwrite(bev_img, output_path)

    #         conn_img_gt = show_bev_results(gt_lanes, pred_lanes, gt_lclc, pred_lclc, only='gt', map_size=[-52, 55, -27, 27], scale=20)
    #         conn_img_pred = show_bev_results(gt_lanes, pred_lanes, gt_lclc, pred_lclc, only='pred', map_size=[-52, 55, -27, 27], scale=20)
    #         divider = np.ones((conn_img_gt.shape[0], 7, 3), dtype=np.uint8) * 128
    #         conn_img = np.concatenate([conn_img_gt, divider, conn_img_pred], axis=1)
    #         conn_img = cv2.putText(conn_img, metric_str, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    #         output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/conn.jpg')
    #         mmcv.imwrite(conn_img, output_path)

    # def _render_surround_img(self, images):
    #     all_image = []
    #     img_height = images[1].shape[0]

    #     for idx in [1, 0, 2, 5, 3, 4, 6]:
    #         if idx  == 0:
    #             all_image.append(images[idx][356:1906, :])
    #             all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))
    #         elif idx == 6 or idx == 2:
    #             all_image.append(images[idx])
    #         else:
    #             all_image.append(images[idx])
    #             all_image.append(np.full((img_height, 20, 3), (255, 255, 255), dtype=np.uint8))

    #     surround_img_upper = None
    #     surround_img_upper = np.concatenate(all_image[:5], 1)

    #     surround_img_down = None
    #     surround_img_down = np.concatenate(all_image[5:], 1)
    #     scale = surround_img_upper.shape[1] / surround_img_down.shape[1]
    #     surround_img_down = cv2.resize(surround_img_down, None, fx=scale, fy=scale)

    #     divider = np.full((25, surround_img_down.shape[1], 3), (255, 255, 255), dtype=np.uint8)

    #     surround_img = np.concatenate((surround_img_upper, divider, surround_img_down), 0)
    #     surround_img = cv2.resize(surround_img, None, fx=0.5, fy=0.5)

    #     return surround_img
