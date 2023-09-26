from .argo_dataset import AV2Dataset
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
from time import time
import mmcv
from IPython import embed

import os
import tempfile
import warnings
import random
import copy
from tqdm import tqdm
import numpy as np
import torch
import mmcv
import cv2
import math
from scipy.spatial.distance import cdist, euclidean
import matplotlib as mpl

from pyquaternion import Quaternion
from shapely.geometry import LineString, box, Polygon, MultiPolygon, MultiLineString
from shapely import ops

from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from .openlane_v2_av2_dataset import OpenLaneV2_Av2_Dataset
from .openlanev2_lanesegment.evaluation import evaluate as openlanev2_evaluate
from .openlanev2.utils import format_metric
from .olv2_utils import show_bev_results_lanesegment, fix_pts_interpolate


@DATASETS.register_module()
class AV2_UniMapping_Dataset(OpenLaneV2_Av2_Dataset):
   
    LANE_CLASSES = ('lane_segment', 'ped_crossing', 'road_boundary')

    def __init__(self,
                 *args,
                 scene_map_file,
                 map_size=[-50, -25, 50, 25],
                 points_num=11,
                 virtual_lane='lane_segment',
                 boundary_type='line',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.map_infos = mmcv.load(scene_map_file, file_format='pkl')
        self.map_size = map_size
        self.points_num = points_num
        self.LANE_CLASSES = self.CLASSES

        assert virtual_lane in ['lane_segment', 'centerline', 'none']
        self.virtual_lane = virtual_lane

        assert boundary_type in ['line', 'polygon']
        self.boundary_type = boundary_type

        if not os.path.exists(self.data_root + self.split):
            print('change data_root to s3://odl-flat/Argoverse2/Sensor_Dataset/sensor/')
            self.data_root = 's3://odl-flat/Argoverse2/Sensor_Dataset/sensor/'
        
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def crop_scene_map(self, index):

        def _matching(centerline, divider):
            min_distance = 1e3
            for i, subline in enumerate(divider):
                distance = centerline.distance(subline)
                if distance < min_distance:
                    min_distance = distance
                    closest_line_index = i
                    closet_line = subline
            return closet_line

        LANE_TYPE_MAP = {
            'NONE': 0,
            'SOLID': 1,
            'DASH': 2
        }

        info = self.data_infos[index]
        map_info = self.map_infos[(self.split, info['segment_id'])]
        translation = info['pose']['translation']
        rotation = info['pose']['rotation']

        scene_ls = copy.deepcopy(map_info['annotation']['lane_segment'])

        ls_crop_ids = []
        ls_infos = []
        for idx, lane in enumerate(scene_ls):
            lc_city = lane['centerline']
            lc_ego = (lc_city - translation) @ np.linalg.pinv(rotation).T
            lc_ego_crop = LineString(lc_ego).intersection(box(*self.map_size))

            l_laneline = lane['left_laneline']
            l_laneline_ego = (l_laneline - translation) @ np.linalg.pinv(rotation).T
            l_laneline_ego_crop = LineString(l_laneline_ego).intersection(box(*self.map_size))

            r_laneline = lane['right_laneline']
            right_laneline_ego = (r_laneline - translation) @ np.linalg.pinv(rotation).T
            r_laneline_ego_crop = LineString(right_laneline_ego).intersection(box(*self.map_size))

            if (lc_ego_crop.geom_type == 'MultiLineString' and 
                l_laneline_ego_crop.geom_type == 'MultiLineString' and 
                r_laneline_ego_crop.geom_type == 'MultiLineString'):
                num_pieces = len(lc_ego_crop.geoms)
                for idx_piece, piece in enumerate(lc_ego_crop.geoms):
                    if piece.is_empty:
                        continue
                    closet_left_line = _matching(piece, l_laneline_ego_crop.geoms)
                    closet_right_line = _matching(piece, r_laneline_ego_crop.geoms)
                    if idx_piece == 0:
                        # keep first piece's predeccessor relation
                        new_id = lane['id']
                    elif idx_piece == num_pieces - 1:
                        # keep last piece's sucessor relation
                        new_id = f'{lane["id"]}_crop_last'
                        scene_ls[idx]['id'] = new_id
                    else:
                        # rare case
                        continue
                        new_id = f'{lane["id"]}_crop_{idx_piece}'

                    ls_crop_ids.append(new_id)
                    ls_info = dict()
                    ls_info['id'] = new_id
                    ls_info['centerline'] = np.array(piece.coords)
                    ls_info['left_laneline'] = np.array(closet_left_line.coords)
                    ls_info['right_laneline'] = np.array(closet_right_line.coords)
                    ls_info['left_laneline_type'] = LANE_TYPE_MAP[lane['left_laneline_type']]
                    ls_info['right_laneline_type'] = LANE_TYPE_MAP[lane['right_laneline_type']]
                    ls_info['is_intersection_or_connector'] = lane['is_intersection_or_connector']
                    ls_infos.append(ls_info)

            elif (lc_ego_crop.geom_type == 'LineString' and not lc_ego_crop.is_empty and 
                  l_laneline_ego_crop.geom_type == 'LineString' and not l_laneline_ego_crop.is_empty and 
                  r_laneline_ego_crop.geom_type == 'LineString' and not r_laneline_ego_crop.is_empty):
                ls_crop_ids.append(lane['id'])
                ls_info = dict()
                ls_info['id'] = lane['id']
                ls_info['centerline'] = np.array(lc_ego_crop.coords)
                ls_info['left_laneline'] = np.array(l_laneline_ego_crop.coords)
                ls_info['right_laneline'] = np.array(r_laneline_ego_crop.coords)
                ls_info['left_laneline_type'] = LANE_TYPE_MAP[lane['left_laneline_type']]
                ls_info['right_laneline_type'] = LANE_TYPE_MAP[lane['right_laneline_type']]
                ls_info['is_intersection_or_connector'] = lane['is_intersection_or_connector']
                ls_infos.append(ls_info)

        topology_lsls = np.zeros((len(ls_crop_ids), len(ls_crop_ids)), dtype=bool)
        for lane in scene_ls:
            if lane['id'] not in ls_crop_ids:
                continue
            for id in lane['sucessor_id']:
                if id in ls_crop_ids:
                    topology_lsls[ls_crop_ids.index(lane['id']), ls_crop_ids.index(id)] = True

        # Virtual Lane (Dual None).
        new_ls_infos = []
        deleted_ls_ids = []
        for idx, ls in enumerate(ls_infos):
            if ls['left_laneline_type'] != 0 or ls['right_laneline_type'] != 0:
                new_ls_infos.append(ls)
                continue
            ls = copy.deepcopy(ls)
            if self.virtual_lane == 'lane_segment':
                new_ls_infos.append(ls)
            elif self.virtual_lane == 'centerline':
                ls['left_laneline'] = ls['centerline'].copy()
                ls['right_laneline'] = ls['centerline'].copy()
                new_ls_infos.append(ls)
            elif self.virtual_lane == 'none':
                deleted_ls_ids.append(idx)
                continue
        topology_lsls = np.delete(topology_lsls, deleted_ls_ids, axis=0)
        topology_lsls = np.delete(topology_lsls, deleted_ls_ids, axis=1)

        areas = []
        if 'ped_crossing' in self.LANE_CLASSES:
            ped_areas = [area for area in map_info['annotation']['area'] if area['category'] == 1]
            ped_infos = []
            for ped in ped_areas:
                pts_city = ped['points']
                pts_ego = (pts_city - translation) @ np.linalg.pinv(rotation).T
                assert pts_ego.shape[0] == 5
                l_laneline = LineString(pts_ego[[0, 1]])
                r_laneline = LineString(pts_ego[[3, 2]])

                l_laneline_ego_crop = l_laneline.intersection(box(*self.map_size))
                r_laneline_ego_crop = r_laneline.intersection(box(*self.map_size))
                if (l_laneline_ego_crop.geom_type == 'LineString' and not l_laneline_ego_crop.is_empty and 
                    r_laneline_ego_crop.geom_type == 'LineString' and not r_laneline_ego_crop.is_empty):
                    pts_ego_crop = np.concatenate([l_laneline_ego_crop.coords, r_laneline_ego_crop.coords[::-1], l_laneline_ego_crop.coords[0:1]], axis=0)
                    ped_info = dict(
                        id = ped['id'],
                        points = pts_ego_crop,
                        category = 1
                    )
                    ped_infos.append(ped_info)

            areas.extend(ped_infos)

        if 'road_boundary' in self.LANE_CLASSES:
            bound_areas = [area for area in map_info['annotation']['area'] if area['category'] == 2]
            polygons = []
            for bound in bound_areas:
                pts_city = bound['points']
                if np.isnan(pts_city).any():
                    mask = np.isnan(pts_city).any(-1)
                    pts_city = pts_city[~mask]
                pts_ego = (pts_city - translation) @ np.linalg.pinv(rotation).T
                if not np.array_equal(pts_ego[0], pts_ego[-1]):
                    continue

                polygon = Polygon(pts_ego)

                if not polygon.is_valid:
                    continue
                polygon = polygon.intersection(box(*self.map_size))
                if polygon.is_empty or not polygon.is_valid:
                    continue

                if polygon.geom_type == 'Polygon':
                    polygon = MultiPolygon([polygon])
                polygons.append(polygon)

            union_polygon = ops.unary_union(polygons)
            if union_polygon.geom_type != 'MultiPolygon':
                union_polygon = MultiPolygon([union_polygon])

            exteriors = []
            interiors = []
            for poly in union_polygon.geoms:
                exteriors.append(poly.exterior)
                for inter in poly.interiors:
                    interiors.append(inter)
            polys = []
            for exter in exteriors:
                if exter.is_ccw:
                    exter.coords = np.array(exter.coords)[::-1]
                polys.append(exter)
            for inter in interiors:
                if not inter.is_ccw:
                    inter.coords = np.array(inter.coords)[::-1]
                polys.append(inter)

            if self.boundary_type == 'line':
                bound_lines = []
                local_patch = np.array(self.map_size) + np.array([0.2, 0.2, -0.2, -0.2])
                local_patch = box(*local_patch)
                for poly in polys:
                    lines = poly.intersection(local_patch)
                    if isinstance(lines, MultiLineString):
                        lines = ops.linemerge(lines)
                        if lines.geom_type == 'MultiLineString':
                            bound_lines.extend([np.array(line.coords) for line in lines.geoms])
                        else:
                            assert lines.geom_type == 'LineString'
                            bound_lines.append(np.array(lines.coords))
            elif self.boundary_type == 'polygon':
                bound_lines = [np.array(poly.coords) for poly in polys]
            
            bound_infos = []
            for idx, line in enumerate(bound_lines):
                bound_info = dict(
                    id = 10000 + idx,
                    points = line,
                    category = 2
                )
                bound_infos.append(bound_info)
            areas.extend(bound_infos)

        ann_info = dict(
            lane_segment=new_ls_infos,
            area=areas,
            traffic_element=[],
            topology_lsls=topology_lsls,
            topology_lste=np.zeros((len(ls_infos), 0), dtype=bool)
        )
        return ann_info

    def ped2lane_segment(self, points):
        assert points.shape[0] == 5
        dir_vector = points[1] - points[0]
        dir = np.rad2deg(np.arctan2(dir_vector[1], dir_vector[0]))

        if dir < -45 or dir > 135:
            left_boundary = points[[2, 3]]
            right_boundary = points[[1, 0]]
        else:
            left_boundary = points[[0, 1]]
            right_boundary = points[[3, 2]]
        
        centerline = LineString((left_boundary + right_boundary) / 2)
        left_boundary = LineString(left_boundary)
        right_boundary = LineString(right_boundary)

        return centerline, left_boundary, right_boundary

    def change_boundary_dir(self, points):
        dir_vector = points[-1] - points[0]
        dir = np.rad2deg(np.arctan2(dir_vector[1], dir_vector[0]))
        if dir < -45 or dir > 135:
            points = points[::-1]
        return points

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information
        """
        info = self.data_infos[index]
        if 'annotation_cache' not in info:
            ann_info = self.crop_scene_map(index)
            self.data_infos[index]['annotation_cache'] = ann_info
        else:
            ann_info = info['annotation_cache']

        gt_lanes = []
        gt_lane_labels_3d = []
        gt_lane_left_type = []
        gt_lane_right_type = []

        for idx, lane in enumerate(ann_info['lane_segment']):
            centerline = lane['centerline']
            LineString_lane = LineString(centerline)
            left_boundary = lane['left_laneline']
            LineString_left_boundary = LineString(left_boundary)
            right_boundary = lane['right_laneline']
            LineString_right_boundary = LineString(right_boundary)
            # remove virtual lanes
            if lane['left_laneline_type'] and lane['right_laneline_type']:
                gt_lanes.append([LineString_lane, LineString_left_boundary, LineString_right_boundary])
                gt_lane_labels_3d.append(1) # StreamMapNet only
                gt_lane_left_type.append(lane['left_laneline_type'])
                gt_lane_right_type.append(lane['right_laneline_type'])
            else:
                continue

        for area in ann_info['area']:
            if area['category'] == 1 and 'ped_crossing' in self.LANE_CLASSES:
                centerline, left_boundary, right_boundary = self.ped2lane_segment(area['points'])
                gt_lanes.append([centerline, left_boundary, right_boundary])
                gt_lane_labels_3d.append(0) # StreamMapNet only

            elif area['category'] == 2 and 'road_boundary' in self.LANE_CLASSES:
                bound = area['points']
                if self.boundary_type == 'line':
                    bound = LineString(self.change_boundary_dir(bound))
                gt_lanes.append([bound, bound, bound])
                gt_lane_labels_3d.append(2)

            gt_lane_left_type.append(0)
            gt_lane_right_type.append(0)

        topology_lsls = ann_info['topology_lsls']
        annos = dict(
            gt_lanes_3d = gt_lanes,
            gt_lane_labels_3d = gt_lane_labels_3d,
            gt_lane_adj = topology_lsls,
            gt_lane_left_type = gt_lane_left_type,
            gt_lane_right_type = gt_lane_right_type,
        )

        return annos

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
            scene_token=info['segment_id'],
            scene_name=info['segment_id'],
            token=info['segment_id']
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
                    ego2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    ego2cam=lidar2cam_rts,
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
        input_dict['ego2global_rotation'] = np.array(info['pose']['rotation'])
        input_dict['ego2global_translation'] = np.array(info['pose']['translation'])
        
        return input_dict

    def format_openlanev2_gt(self):
        gt_dict = {}
        for idx in tqdm(range(len(self.data_infos))):
            info = copy.deepcopy(self.data_infos[idx])
            if 'annotation_cache' not in info:
                ann_info = self.crop_scene_map(idx)
                self.data_infos[idx]['annotation_cache'] = ann_info
            else:
                ann_info = info['annotation_cache']
            key = (self.split, info['segment_id'], str(info['timestamp']))

            info['annotation'] = dict(
                lane_segment = [],
                area = [],
                traffic_element = [],
                topology_lsls = None,
                topology_lste = None,
            )

            for ls_info in ann_info['lane_segment']:
                ls_info = copy.deepcopy(ls_info)
                if ls_info['left_laneline_type'] and ls_info['right_laneline_type']:
                    ls_info['centerline'] = fix_pts_interpolate(ls_info['centerline'], 10)[:,:2]
                    ls_info['left_laneline'] = fix_pts_interpolate(ls_info['left_laneline'], 10)[:,:2]
                    ls_info['right_laneline'] = fix_pts_interpolate(ls_info['right_laneline'], 10)[:,:2]
                    info['annotation']['lane_segment'].append(ls_info)
                else:
                    continue

            info['annotation']['topology_lsls'] = ann_info['topology_lsls']
            info['annotation']['topology_lste'] = np.zeros((len(info['annotation']['lane_segment']), 0), dtype=bool)

            for area in ann_info['area']:
                area = copy.deepcopy(area)
                if area['category'] == 1:
                    points = area['points']
                    left_boundary = fix_pts_interpolate(points[[0, 1]], 10)[:,:2]
                    right_boundary = fix_pts_interpolate(points[[2, 3]], 10)[:,:2]
                    area['points'] = np.concatenate([left_boundary, right_boundary], axis=0)
                elif area['category'] == 2:
                    area['points'] = fix_pts_interpolate(area['points'], 20)[:,:2]

                info['annotation']['area'].append(area)

            gt_dict[key] = info

        return gt_dict

    def format_results(self, results, jsonfile_prefix=None):
        # TODO: reformulate from StreamMapNet's output to OpenLaneV2's output
        pred_dict = {}
        pred_dict['method'] = 'dummy'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'dummy'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        # breakpoint()
        # result keys: vectors(100,20,2), scores(100, ), labels(100, ), prop_mask(100,), token(str)
        for idx, result in enumerate(tqdm(results)):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))

            pred_info = dict(
                lane_segment = [],
                area = [],
                traffic_element = [],
                topology_lclc = None,
                topology_lcte = None
            )

            result['bbox_results'] = None
            result['lclc_results'] = None
            result['lcte_results'] = None

            if result['vectors'] is not None:
                lane_results = result['vectors']
                # scores = lane_results[1]
                scores = result['scores']
                valid_indices = np.argsort(-scores)
                lanes = lane_results[valid_indices]
                labels = result['labels'][valid_indices]
                scores = scores[valid_indices]
                # lanes = lanes.reshape(-1, lanes.shape[-1] // 2, 2)

                has_lane_type = False
                # if len(lane_results) in [7, 8]:
                #     left_type_scores = lane_results[3][valid_indices]
                #     left_type_labels = lane_results[4][valid_indices]
                #     right_type_scores = lane_results[5][valid_indices]
                #     right_type_labels = lane_results[6][valid_indices]
                #     has_lane_type = True

                pred_area_index = []
                for pred_idx, (lane, score, label) in enumerate(zip(lanes, scores, labels)):
                    if label == 1:
                        points = lane.astype(np.float32)
                        pred_lane_segment = {}
                        pred_lane_segment['id'] = 20000 + pred_idx
                        # pred_lane_segment['centerline']     = fix_pts_interpolate(points[:self.points_num], 10)
                        # pred_lane_segment['left_laneline']  = fix_pts_interpolate(points[self.points_num:self.points_num * 2], 10)
                        pred_lane_segment['left_laneline'] = fix_pts_interpolate(points[:self.points_num], 10)
                        # pred_lane_segment['right_laneline'] = fix_pts_interpolate(points[self.points_num * 2:], 10)
                        pred_lane_segment['right_laneline'] = fix_pts_interpolate(points[self.points_num:self.points_num*2], 10)
                        # pred_lane_segment['left_laneline_type'] = left_type_labels[pred_idx] if has_lane_type else -1
                        # pred_lane_segment['right_laneline_type'] = right_type_labels[pred_idx] if has_lane_type else -1
                        pred_lane_segment['centerline'] = (pred_lane_segment['left_laneline'] + pred_lane_segment['right_laneline'])/2
                        pred_lane_segment['confidence'] = score.item()
                        pred_info['lane_segment'].append(pred_lane_segment)

                    elif label == 0:
                        points = lane.astype(np.float32)
                        pred_ped = {}
                        pred_ped['id'] = 20000 + pred_idx
                        pred_points = np.concatenate((points[:self.points_num],
                                                      points[self.points_num:self.points_num * 2][::-1]), axis=0)
                        pred_ped['points'] = fix_pts_interpolate(pred_points, 20)
                        pred_ped['category'] = 1 # hard code, adapt to StreamMapNet
                        pred_ped['confidence'] = score.item()
                        pred_info['area'].append(pred_ped)
                        pred_area_index.append(pred_idx)

                    elif label == 2:
                        points = lane.astype(np.float32)
                        pred_bound = {}
                        pred_bound['id'] = 20000 + pred_idx
                        pred_points = np.zeros((self.points_num * 2, 2), dtype=np.float32)
                        pred_points[0::2] = points[:self.points_num]
                        pred_points[1::2] = points[self.points_num:]
                        pred_bound['points'] = fix_pts_interpolate(pred_points, 20)
                        pred_bound['category'] = label
                        pred_bound['confidence'] = score.item()
                        pred_info['area'].append(pred_bound)
                        pred_area_index.append(pred_idx)

            if result['bbox_results'] is not None:
                te_results = result['bbox_results']
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id = 10000 + pred_idx,
                        category = 1 if class_idx < 4 else 2,
                        attribute = class_idx,
                        points = te.reshape(2, 2).astype(np.float32),
                        confidence = score
                    )
                    pred_info['traffic_element'].append(te_info)

            if result['lclc_results'] is not None:
                topology_lsls_ped = result['lclc_results'].astype(np.float32)[valid_indices][:, valid_indices]
                topology_lsls_ped = np.delete(topology_lsls_ped, pred_area_index, axis=0)
                topology_lsls = np.delete(topology_lsls_ped, pred_area_index, axis=1)
                pred_info['topology_lsls'] = topology_lsls
            else:
                pred_info['topology_lsls'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['lane_segment'])), dtype=np.float32)

            if result['lcte_results'] is not None:
                topology_lste = result['lcte_results'].astype(np.float32)[valid_indices]
                topology_lste = np.delete(topology_lste, pred_area_index, axis=0)
                pred_info['topology_lste'] = topology_lste
            else:
                pred_info['topology_lste'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['traffic_element'])), dtype=np.float32)

            pred_dict['results'][key] = dict(predictions=pred_info)

        return pred_dict

    def evaluate(self, results, logger=None, show=False, out_dir=None, **kwargs):
        """Evaluation in OpenlaneV2 av2 dataset. TODO Adapt to OLV2 evaluation metric.

        Args:
            results (list): Testing results of the dataset.
            metric (str): Metric to be performed.
            iou_thr (float): IoU threshold for evaluation.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            show (bool): Whether to visualize the results.
            out_dir (str): Path of directory to save the results.
            pipeline (list[dict]): Processing pipeline.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        if show:
            assert out_dir, 'Expect out_dir when show is set.'
            logger.info(f'Visualizing results at {out_dir}...')
            self.show(results, out_dir)
            logger.info(f'Visualize done.')
        print(f'Starting storing results...')
        mmcv.dump(results, './results.pkl')
        print(f'Starting format results...')
        gt_dict = self.format_openlanev2_gt()
        pred_dict = self.format_results(results)

        print(f'Starting openlanev2 evaluate...')
        metric_results = openlanev2_evaluate(gt_dict, pred_dict)
        metric_results = format_metric(metric_results)
        return metric_results

    def show(self, results, out_dir, score_thr=0.3, show_num=20, **kwargs):
        """Show the results.

        Args:
            results (list[dict]): Testing results of the dataset.
            out_dir (str): Path of directory to save the results.
            score_thr (float): The threshold of score.
            show_num (int): The number of images to be shown.
        """
        for idx, result in enumerate(results):

            if idx % 6 != 0:
                continue
            if idx // 6 > show_num:
                break

            info = self.data_infos[idx]

            gt_lanes_centerline = []
            gt_lanes_left = []
            gt_lanes_right = []

            if 'annotation_cache' not in info:
                ann_info = self.crop_scene_map(idx)
                self.data_infos[idx]['annotation_cache'] = ann_info
            else:
                ann_info = info['annotation_cache']
            for idx, lane_segment in enumerate(ann_info['lane_segment']):
                left_line = np.array(lane_segment['left_laneline'], dtype=np.float32)
                right_line = np.array(lane_segment['right_laneline'], dtype=np.float32)
                centerline = (left_line + right_line) / 2

                gt_lanes_centerline.append(centerline)
                gt_lanes_left.append(left_line)
                gt_lanes_right.append(right_line)

            gt_areas = []
            for idx, area in enumerate(ann_info['area']):
                gt_areas.append(area['points'])

            lane_results = result['lane_results']
            lanes = lane_results[0]
            scores = lane_results[1]
            labels = lane_results[2]
            lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)
            if len(lane_results) in [4, 8]:
                lane_masks = lane_results[-1]
                have_mask = True
            else:
                lane_masks = [None] * len(lanes)
                have_mask = False


            pred_lanes_centerline = []
            pred_lanes_left = []
            pred_lanes_right = []
            pred_lanes_mask = []
            lane_scores = []

            pred_areas = []
            pred_area_mask = []
            area_scores = []

            for pred_idx, (lane, score, label, mask) in enumerate(zip(lanes, scores, labels, lane_masks)):
                if label == 0:
                    points = lane.astype(np.float32)
                    pred_lanes_centerline.append(fix_pts_interpolate(points[:self.points_num], 20))
                    pred_lanes_left.append(fix_pts_interpolate(points[self.points_num:self.points_num * 2], 20))
                    pred_lanes_right.append(fix_pts_interpolate(points[self.points_num * 2:], 20))
                    lane_scores.append(score.item())
                    pred_lanes_mask.append(mask)

                elif label == 1:
                    points = lane.astype(np.float32)
                    pred_ped = np.concatenate((points[self.points_num:self.points_num * 2],
                                               points[self.points_num * 2:][::-1],
                                               points[self.points_num][None]), axis=0)
                    pred_areas.append(pred_ped)
                    area_scores.append(score.item())
                    pred_area_mask.append(mask)

                elif label == 2:
                    points = lane.astype(np.float32)
                    pred_bound = np.zeros((self.points_num * 2, 3), dtype=np.float32)
                    pred_bound[0::2] = points[self.points_num:self.points_num * 2]
                    pred_bound[1::2] = points[self.points_num * 2:]
                    pred_areas.append(pred_bound)
                    area_scores.append(score.item())
                    pred_area_mask.append(mask)


            lane_scores = np.array(lane_scores)
            mask_lane = lane_scores > score_thr
            pred_lanes_centerline = np.array(pred_lanes_centerline)[mask_lane]
            pred_lanes_left = np.array(pred_lanes_left)[mask_lane]
            pred_lanes_right = np.array(pred_lanes_right)[mask_lane]
            pred_lanes_mask = np.array(pred_lanes_mask)[mask_lane]

            area_scores = np.array(area_scores)
            mask_ped = area_scores > score_thr
            pred_areas = np.array(pred_areas)[mask_ped]
            pred_area_mask = np.array(pred_area_mask)[mask_ped]

            vis_map_size = np.array(self.map_size)[[0, 2, 1, 3]] + np.array([-2, 2, -2, 2])
            img_gt = show_bev_results_lanesegment(gt_lanes_centerline, pred_lanes_centerline,
                                                  gt_lanes_left, pred_lanes_left,
                                                  gt_lanes_right, pred_lanes_right,
                                                  gt_areas, pred_areas,
                                                  only='gt',
                                                  map_size=vis_map_size,
                                                  scale=20)
            img_pred = show_bev_results_lanesegment(gt_lanes_centerline, pred_lanes_centerline,
                                                    gt_lanes_left, pred_lanes_left,
                                                    gt_lanes_right, pred_lanes_right,
                                                    gt_areas, pred_areas,
                                                    only='pred',
                                                    map_size=vis_map_size,
                                                    scale=20)
            divider = np.ones((img_gt.shape[0], 7, 3), dtype=np.uint8) * 128
            bev_img = np.concatenate([img_gt, divider, img_pred], axis=1)

            output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/bev.jpg')
            mmcv.imwrite(bev_img, output_path)

            if not have_mask:
                continue

            cmap = mpl.colormaps['inferno']
            vis_map_size = np.array(self.map_size)[[0, 2, 1, 3]]
            for inst_idx, (lane, mask) in enumerate(zip(pred_ped_crossing, pred_ped_mask)):
                inst_img = show_bev_results_lanesegment([], [],
                                                        [], [],
                                                        [], [],
                                                        [], [lane],
                                                        only='pred',
                                                        map_size=vis_map_size,
                                                        scale=10)
                mask = (cmap(mask)[..., [2, 1, 0]] * 255).astype(np.uint8)
                mask = np.transpose(mask, (1, 0, 2))
                mask = np.flip(mask, 1)
                mask = np.flip(mask, 0)
                mask = cv2.resize(mask, (inst_img.shape[1], inst_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                inst_img = cv2.addWeighted(inst_img, 1, mask, 0.8, 0)
                output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/ped/{inst_idx}.jpg')
                mmcv.imwrite(inst_img, output_path)

            for inst_idx, (lane, left, right, mask) in enumerate(zip(pred_lanes_centerline, pred_lanes_left, pred_lanes_right, pred_lanes_mask)):
                inst_img = show_bev_results_lanesegment([], [lane],
                                                        [], [left],
                                                        [], [right],
                                                        [], [],
                                                        only='pred',
                                                        map_size=vis_map_size,
                                                        scale=10)
                mask = (cmap(mask)[..., [2, 1, 0]] * 255).astype(np.uint8)
                mask = np.transpose(mask, (1, 0, 2))
                mask = np.flip(mask, 1)
                mask = np.flip(mask, 0)
                mask = cv2.resize(mask, (inst_img.shape[1], inst_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                inst_img = cv2.addWeighted(inst_img, 1, mask, 0.8, 0)
                output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/inst/{inst_idx}.jpg')
                mmcv.imwrite(inst_img, output_path)

            if 'seg_results' in result:

                bev_img = (result['seg_results'].argmax(0) * 255/2).astype(np.uint8)
                output_path = os.path.join(out_dir, f'{info["segment_id"]}/{info["timestamp"]}/bev_seg.jpg')
                mmcv.imwrite(bev_img, output_path)