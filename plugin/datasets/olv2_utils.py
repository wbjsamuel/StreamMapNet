import numpy as np
import mmcv
import cv2
from plugin.datasets.openlanev2.visualization.utils import COLOR_DICT
import random

from pyquaternion import Quaternion
from shapely.geometry import LineString, box, Polygon, MultiPolygon, MultiLineString
from shapely import ops

GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
AERA_COLOR = (0, 255, 255)

GT_ELE_COLOR = {'0': (0, 0, 255), '1': (255, 0, 0), '2': (255, 255, 0)}
PRED_ELE_COLOR = {'0': (0, 0, 255), '1': (255, 0, 0), '2': (255, 255, 0)}

def fix_pts_interpolate(lane, n_points):
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane

def show_bev_results_lanesegment(gt_lane, pred_lane,
                                 gt_lanes_left, pred_lanes_left,
                                 gt_lanes_right, pred_lanes_right,
                                 gt_areas, pred_areas,
                                 only=None, map_size=[-55, 55, -30, 30], scale=10):
    image = np.zeros((int(scale * (map_size[1] - map_size[0])), int(scale * (map_size[3] - map_size[2])), 3), dtype=np.uint8)
    if only is None or only == 'gt':
        for lane_left, lane_right in zip(gt_lanes_left, gt_lanes_right):
            new_image = np.zeros_like(image)
            lane_coords = np.concatenate([lane_left, lane_right[::-1]], axis=0)[:, :2]
            draw_coor = (scale * (-lane_coords + np.array([map_size[1], map_size[3]]))).astype(int)
            new_image = cv2.polylines(new_image, [draw_coor[:, [1, 0]]], True, GT_COLOR, max(round(scale * 0.2), 1))
            image = cv2.addWeighted(image, 1, new_image, 0.5, 0)

        for lane in gt_lane:
            draw_coor = (scale * (-lane[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            image = cv2.polylines(image, [draw_coor[:, [1, 0]]], False, GT_COLOR, max(round(scale * 0.2), 1))
            image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(2, round(scale * 0.5)), (0, 255, 0), -1)
            image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(2, round(scale * 0.5)), (255, 0, 0), -1)

        for ped in gt_areas:
            draw_coor = (scale * (-ped[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            image = cv2.polylines(image, [draw_coor[:, [1, 0]]], False, AERA_COLOR, max(round(scale * 0.2), 1))

    if only is None or only == 'pred':

        for lane_left, lane_right in zip(pred_lanes_left, pred_lanes_right):
            new_image = np.zeros_like(image)
            lane_coords = np.concatenate([lane_left, lane_right[::-1]], axis=0)[:, :2]
            draw_coor = (scale * (-lane_coords + np.array([map_size[1], map_size[3]]))).astype(int)
            new_image = cv2.polylines(new_image, [draw_coor[:, [1, 0]]], True, PRED_COLOR, max(round(scale * 0.1), 1))
            image = cv2.addWeighted(image, 1, new_image, 0.5, 0)

        for lane in pred_lane:
            draw_coor = (scale * (-lane[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            image = cv2.polylines(image, [draw_coor[:, [1, 0]]], False, PRED_COLOR, max(round(scale * 0.2), 1))
            image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(2, round(scale * 0.5)), (0, 255, 0), -1)
            image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(2, round(scale * 0.5)), (255, 0, 0), -1)

        for ped in pred_areas:
            draw_coor = (scale * (-ped[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            image = cv2.polylines(image, [draw_coor[:, [1, 0]]], False, AERA_COLOR, max(round(scale * 0.2), 1))
    return image
