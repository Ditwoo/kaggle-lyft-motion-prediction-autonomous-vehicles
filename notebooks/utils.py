from collections import namedtuple
import numpy as np
import cv2
from l5kit.geometry import transform_points


Traj = namedtuple('Traj', ['xx0', 'yy0', 'xx1', 'yy1'])


def traj_from_item(item):
    history = transform_points(
        item['history_positions'][item['history_availabilities'].astype(bool)],
        item['world_from_agent']
    )
    future = transform_points(
        item['target_positions'][item['target_availabilities'].astype(bool)],
        item['world_from_agent']
    )
    return Traj(xx0=history[:, 0], yy0=history[:, 1], xx1=future[:, 0], yy1=future[:, 1])


def calc_traj_size(traj):
    k = 1000
    xx = np.concatenate([traj.xx0, traj.xx1])
    yy = np.concatenate([traj.yy0, traj.yy1])
    rect = cv2.minAreaRect(np.array([np.array([k * x, k * y], np.int) for (x, y) in zip(xx, yy)]))
    h, w = (rect[1][0] / k), (rect[1][1] / k)
    size = np.sqrt(h ** 2 + w ** 2)
    return size, h, w


def traj_geometry_from_item(item):
    """
    we picked the following thresholds:
    size_th = 6
    ratio_th = 4.5
    """
    traj = traj_from_item(item)
    size, h, w = calc_traj_size(traj)
    return size, h, w