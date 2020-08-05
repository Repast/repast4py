import numpy as np
from numba import jit


_xoffset_1d = np.array([-1, 0, 1], dtype=np.int32)
_xoffset_2d = np.tile(_xoffset_1d, 3)
_yoffset_2d = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int32)


@jit(nopython=True)
def find_1d_nghs_sticky(pt, min_max):
    """
    :param pt: 3d numpy array of length 3 and int64
    :param min_max: numpy array with format [min_x, max_x]
    :return: a 1d numpy array with the neighboring points according to sticky border
    semantics
    """
    x_pts = pt[0] + _xoffset_1d
    x_idxs = (x_pts >= min_max[0]) & (x_pts <= min_max[1])
    in_range_pts = x_pts[x_idxs]
    return in_range_pts


@jit(nopython=True)
def find_2d_nghs_sticky(pt, min_max, row_major=False):
    """
    :param pt: 3d numpy array of length 3 and int64
    :param min_max: numpy array with format [min_x, max_x, min_y, max_y]
    :return: a 2d numpy array with the neighboring points according to sticky border
    semantics, starting with the x dimension, then the y.
    """
    x_pts = pt[0] + _xoffset_2d
    y_pts = pt[1] + _yoffset_2d

    x_idxs = (x_pts >= min_max[0]) & (x_pts <= min_max[1])
    # remove out of range xs
    x_pts = x_pts[x_idxs]
    y_pts = y_pts[x_idxs]

    # remove out of range ys
    y_idxs = (y_pts >= min_max[2]) & (y_pts <= min_max[3])
    x_pts = x_pts[y_idxs]
    y_pts = y_pts[y_idxs]

    return np.stack((y_pts, x_pts), axis=0) if row_major else np.stack((x_pts, y_pts), axis=0)

    
@jit(nopython=True)
def find_1d_nghs_periodic(pt, min_max):
    """
    :param pt: 3d numpy array of length 3 and int64
    :param min_max: numpy array with format [min_x, max_x]
    :return: a 1d numpy array with the neighboring points according to sticky border
    semantics
    """
    x_extent = min_max[1] - min_max[0] + 1
    x_pts = pt[0] + _xoffset_1d
    ncs = np.mod((x_pts - min_max[0]), x_extent)
    x_pts = np.where(ncs < 0, min_max[1] + ncs, min_max[0] + ncs)
    return x_pts


@jit(nopython=True)
def find_2d_nghs_periodic(pt, min_max, row_major=False):
    """
    :param pt: 3d numpy array of length 3 and int64
    :param min_max: numpy array with format [min_x, max_x]
    :return: a 1d numpy array with the neighboring points according to sticky border
    semantics
    """
    x_extent = min_max[1] - min_max[0] + 1
    x_pts = pt[0] + _xoffset_2d
    ncs = np.mod((x_pts - min_max[0]), x_extent)
    x_pts = np.where(ncs < 0, min_max[1] + ncs, min_max[0] + ncs)

    y_extent = min_max[3] - min_max[2] + 1
    y_pts = pt[1] + _yoffset_2d
    ncs = np.mod((y_pts - min_max[2]), y_extent)
    y_pts = np.where(ncs < 0, min_max[3] + ncs, min_max[2] + ncs)
    
    return np.stack((y_pts, x_pts), axis=0) if row_major else np.stack((x_pts, y_pts), axis=0)
