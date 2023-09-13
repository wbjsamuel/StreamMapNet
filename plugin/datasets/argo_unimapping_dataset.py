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

import numpy as np
import torch
import mmcv
import cv2
import math
from scipy.spatial.distance import cdist, euclidean

from pyquaternion import Quaternion
from shapely.geometry import LineString, box, Polygon
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from .openlane_v2_av2_dataset import OpenLaneV2_Av2_Dataset



def fix_pts_interpolate(lane, n_points):
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane

@DATASETS.register_module()
class AV2_UniMapping_Dataset(OpenLaneV2_Av2_Dataset):
   
    LANE_CLASSES = ('lane_segment', 'ped_crossing', 'road_boundary')

    def __init__(self, *args, points_num=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.points_num = points_num
        self.LANE_CLASSES = self.CLASSES

    def polygon_2_lanesegment(self, points):
        polygon = Polygon(points)
        trapezoid = polygon.minimum_rotated_rectangle
        edges_2d = np.array(trapezoid.exterior.coords)

        edge_lengths = [((x2 - x1)**2 + (y2 - y1)**2)**0.5 for (x1, y1), (x2, y2) in zip(edges_2d[:-1], edges_2d[1:])]
        longest_side_indices = sorted(range(len(edge_lengths)), key=lambda i: edge_lengths[i], reverse=True)[:2]
        edges = np.zeros((len(edges_2d), 3))
        for i, edge_point in enumerate(edges_2d):
            distances = [np.linalg.norm(np.array(edge_point) - np.array(p[:2])) for p in points]
            nearest_point_index = np.argmin(distances)
            nearest_point_z = points[nearest_point_index][2]
            edges[i] = np.array([edge_point[0], edge_point[1], nearest_point_z])

        parallel_sides_vector_1 = edges[longest_side_indices[0]] - edges[longest_side_indices[0] + 1]
        parallel_sides_vector_2 = edges[longest_side_indices[1]] - edges[longest_side_indices[1] + 1]
        # long_side_index = edge_lengths.index(max(edge_lengths))
        # long_side_vertices = [edges[long_side_index], edges[(long_side_index + 1) % len(edges)]]
        if not -45 <= math.degrees(math.atan2(parallel_sides_vector_1[1], parallel_sides_vector_1[0])) <= 135:
            boundary_1 = np.array([edges[longest_side_indices[0]], edges[(longest_side_indices[0] + 1) % len(edges)]])
        else:
            boundary_1 = np.array([edges[(longest_side_indices[0] + 1) % len(edges)], edges[longest_side_indices[0]]])
        if not -45 <= math.degrees(math.atan2(parallel_sides_vector_2[1], parallel_sides_vector_2[0])) <= 135:
            boundary_2 = np.array([edges[longest_side_indices[1]], edges[(longest_side_indices[1] + 1) % len(edges)]])
        else:
            boundary_2 = np.array([edges[(longest_side_indices[1] + 1) % len(edges)], edges[longest_side_indices[1]]])
        centerline = (boundary_1 + boundary_2) / 2
        LineString_lane = LineString(centerline)
        # a[0] * b[1] - a[1] * b[0]
        if (boundary_1[1] - centerline[0])[0] * (centerline[1] - centerline[0])[1] - (boundary_1[1] - centerline[0])[1] * (centerline[1] - centerline[0])[0] > 0:
            #Anticlockwise rotation->right
            right_boundary = boundary_1
            LineString_right_boundary = LineString(right_boundary)
            left_boundary = boundary_2
            LineString_left_boundary = LineString(left_boundary)
        else:
            right_boundary = boundary_2
            LineString_right_boundary = LineString(right_boundary)
            left_boundary = boundary_1
            LineString_left_boundary = LineString(left_boundary)
        return LineString_lane, LineString_right_boundary, LineString_left_boundary

    def find_top_left(self, points):
        top_left = points[0]
        top_left_index = 0
        for index, point in enumerate(points[1:]):
            if point[0] > top_left[0]:
                top_left = point
                top_left_index = index + 1
            elif point[0] == top_left[0] and point[1] < top_left[1]:
                top_left = point
                top_left_index = index + 1

        ped_points = np.roll(points, top_left_index)
        ped = LineString(ped_points)
        return ped

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
        input_dict['ego2global_rotation'] = np.array(info['pose']['rotation'])
        input_dict['ego2global_translation'] = np.array(info['pose']['translation'])
        
        return input_dict
    
    def get_ann_info(self, index):
        gt_lane_crop_ids = []
        gt_lane_crop = []
        gt_lane_labels_3d = []
        gt_lane_left_type = []
        gt_lane_right_type = []
        vectors = []
        info = self.data_infos[index]
        # 'ped_crossing': 0, 'divider': 1, 'boundary': 2 
        for idx, lane in enumerate(info['annotation']['lane_segment']):
            centerline = lane['centerline']
            LineString_lane = LineString(centerline)
            left_boundary = lane['left_laneline']
            LineString_left_boundary = LineString(left_boundary)
            right_boundary = lane['right_laneline']
            LineString_right_boundary = LineString(right_boundary)
            gt_lane_crop_ids.append(lane['id'])
            gt_lane_crop.append([LineString_lane, LineString_left_boundary, LineString_right_boundary])
            gt_lane_labels_3d.append(0)
            gt_lane_left_type.append(lane['left_laneline_type'])
            gt_lane_right_type.append(lane['right_laneline_type'])
            vectors.append({'1': [LineString_left_boundary, LineString_right_boundary[::-1]]})
            
        for idx, lane in enumerate(info['annotation']['area']):
            if lane['category'] == 1 and 'ped_crossing' in self.LANE_CLASSES:
                LineString_lane, LineString_right_boundary, LineString_left_boundary = self.polygon_2_lanesegment(lane['points'])
                gt_lane_crop_ids.append(lane['id'])
                gt_lane_crop.append([LineString_lane, LineString_left_boundary, LineString_right_boundary])
                gt_lane_labels_3d.append(1)
                gt_lane_left_type.append(0)
                gt_lane_right_type.append(0)
                vectors.append({'0': [LineString_left_boundary, LineString_right_boundary[::-1]]})
                
            elif lane['category'] == 2 and 'road_boundary' in self.LANE_CLASSES:
                if (lane['points'][0] == lane['points'][-1]).all():
                    LineString_lane, LineString_right_boundary, LineString_left_boundary = self.polygon_2_lanesegment(lane['points'])
                    gt_lane_crop.append([LineString_lane, LineString_left_boundary, LineString_right_boundary])
                    vectors.append({'2': [LineString_left_boundary, LineString_right_boundary[::-1]]})
                else:
                    line = self.find_top_left(lane['points'])
                    gt_lane_crop.append([line, line, line])
                    vectors.append({'2': [line]})
                gt_lane_crop_ids.append(lane['id'])
                gt_lane_labels_3d.append(2)
                gt_lane_left_type.append(0)
                gt_lane_right_type.append(0)

        topology_lsls = info['annotation']['topology_lsls']
        gt_te = np.array([element['points'].flatten() for element in info['annotation']['traffic_element']], dtype=np.float32).reshape(-1, 4)
        gt_te_labels = np.array([element['attribute'] for element in info['annotation']['traffic_element']], dtype=np.int64)
        topology_lste = info['annotation']['topology_lste']
        gt_area = [element['points'] for element in info['annotation']['area']]
        gt_arae_labels = np.array([element['category'] for element in info['annotation']['area']], dtype=np.int64)

        annos = dict(
            gt_lanes_3d = gt_lane_crop,
            gt_lane_labels_3d = gt_lane_labels_3d,
            gt_lane_adj = topology_lsls,
            gt_lane_left_type = gt_lane_left_type,
            gt_lane_right_type = gt_lane_right_type,
            bboxes = gt_te,
            labels = gt_te_labels,
            gt_lane_lcte_adj = topology_lste,
            gt_area = gt_area,
            gt_arae_labels = gt_arae_labels,
            vectors = vectors
        )

        return annos       

    