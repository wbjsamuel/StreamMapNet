import numpy as np
from numpy import random
import torch
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from math import factorial
from shapely.geometry import LineString, Polygon
import cv2

@PIPELINES.register_module()
class LaneSegmentParameterize3D(object):
    def __init__(self, method, method_para, only="none"):
        method_list = ["downsample"]
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para
        assert only in ["none", "centerline", "laneline"]
        self.only = only

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # breakpoint()
        lanes = results['ann_info']["gt_lanes_3d"]
        para_lanes = getattr(self, self.method)(lanes, **self.method_para)
        results['gt_lanes_3d'] = para_lanes
        results['gt_lane_labels_3d'] = results['ann_info']['gt_lane_labels_3d']
        return results

    def downsample(self, input_data, n_points=11):
        lane_list = []
        for lane in input_data:
            ls = lane[1]
            distances = np.linspace(0, ls.length, n_points)
            left_line = np.array(
                [ls.interpolate(distance).coords[0] for distance in distances]
            )

            ls = lane[2]
            distances = np.linspace(0, ls.length, n_points)
            right_line = np.array(
                [ls.interpolate(distance).coords[0] for distance in distances]
            )

            centerline = (left_line + right_line) / 2.0
            if self.only == "centerline":
                line = centerline.flatten()
            elif self.only == "laneline":
                line = np.concatenate([left_line.flatten(), right_line.flatten()])
            else:
                line = np.concatenate(
                    [centerline.flatten(), left_line.flatten(), right_line.flatten()]
                )
            lane_list.append(line)

        return np.array(lane_list, dtype=np.float32)