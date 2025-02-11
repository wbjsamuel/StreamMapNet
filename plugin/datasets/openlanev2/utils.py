# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# utils.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

TRAFFIC_ELEMENT_ATTRIBUTE = {
    'unknown':          0,
    'red':              1,
    'green':            2,
    'yellow':           3,
    'go_straight':      4,
    'turn_left':        5,
    'turn_right':       6,
    'no_left_turn':     7,
    'no_right_turn':    8,
    'u_turn':           9,
    'no_u_turn':        10,
    'slight_left':      11,
    'slight_right':     12,
}

LANELINE_CATEGORY = {
    'none':             0,
    'solid':            1,
    'dash':             2,
}

AREA_CATEGOTY = {
    'pedestrian_crossing':  1,
    'road_boundary':        2,
}


def format_metric(metric, verbose=True):
    results = {}
    for key, val in metric.items():
        if verbose:
            print(f'{key} - {val["score"]:.6f}')
        results[key] = val['score']
        for k, v in val.items():
            if 'score' not in k:
                if verbose:
                    print(f'    {k} - {v:.6f}')
                results[k] = v
    return results

def format_lc_metric(metric, verbose=True):
    results = {}
    for key, val in metric.items():
        # if verbose:
        #     print(f'{key} - {val["score"]:.6f}')
        if isinstance(val, dict):
            for k, v in val.items():
                if 'score' not in k:
                    if verbose:
                        print(f'    {k} - {v:.6f}')
                    results[k] = v
        else:
            print(f'    {key} - {val:.6f}')
            results[key] = val
    return results
