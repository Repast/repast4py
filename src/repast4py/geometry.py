# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

import sys
from collections import namedtuple
import numpy as np
from numba import jit
# from numba.core.types import NamedTuple


if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    BoundingBox = namedtuple('BoundingBox', ['xmin', 'xextent', 'ymin', 'yextent', 'zmin', 'zextent'],
                             defaults=[0, 0])
else:
    BoundingBox = namedtuple('BoundingBox', ['xmin', 'xextent', 'ymin', 'yextent', 'zmin', 'zextent'])


def get_num_dims(box: BoundingBox) -> int:
    """Gets the BoundingBox's number of dimensions.

    A dimension with an extent of 0 is considered not to exist.

    Args:
        box: the bounding box
    Returns:
        int: the number of dimensions.

    Examples:
        1D BoundingBox

        >>> bb = BoundingBox(xmin=0, xextent=10, ymin=0, yextent=0, zmin=0, zextent=0)
        >>> get_num_dims(bb)
        1

        2D BoundingBox

        >>> bb = BoundingBox(xmin=0, xextent=10, ymin=0, yextent=20, zmin=0, zextent=0)
        >>> get_num_dims(bb)
        2

        3D BoundingBox

        >>> bb = BoundingBox(xmin=0, xextent=10, ymin=0, yextent=10, zmin=0, zextent=12)
        >>> get_num_dims(bb)
        3
    """
    if box.yextent == 0 and box.zextent == 0:
        return 1
    elif box.zextent == 0:
        return 2
    return 3


_xoffset_1d = np.array([-1, 0, 1], dtype=np.int32)
_xoffset_2d = np.tile(_xoffset_1d, 3)
_yoffset_2d = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int32)
_xoffset_3d = np.tile(_xoffset_1d, 9)
_yoffset_3d = np.tile(_yoffset_2d, 3)
_zoffset_3d = np.array([-1] * 9 + [0] * 9 + [1] * 9)


@jit(nopython=True)
def find_1d_nghs_sticky(pt: np.array, min_max: np.array):
    """Finds the neighboring 1D points of the specified point within
    the min_max inclusive range, using "sticky" semantics. See :class:`repast4py.space.BorderType` and
    :class:`repast4py.space.GridStickyBorders` for more on sticky border semantics.

    Args:
        pt: A numpy scalar array with at least one element
        min_max: A numpy array of format [min_x, max_x]
    Returns:
        A 1d numpy array with the neighboring points according to sticky border
        semantics, including the source point.

    Examples:
        Find the neighboring points of pt(4) within the bounds of 2 and 10.

        >>> from repast4py.space import DiscretePoint
        >>> import numpy as np
        >>> pt = DiscretePoint(4, 0, 0)
        >>> min_max = np.array([2, 10])
        >>> nghs = find_1d_nghs_sticky(pt.coordinates, min_max)
        >>> nghs
        array([3, 4, 5])
    """
    x_pts = pt[0] + _xoffset_1d
    x_idxs = (x_pts >= min_max[0]) & (x_pts <= min_max[1])
    in_range_pts = x_pts[x_idxs]
    return in_range_pts


@jit(nopython=True)
def find_2d_nghs_sticky(pt: np.array, min_max: np.array, pytorch_order: bool = False):
    """Finds the neighboring 2D points of the specifed point within
    the min_max inclusive range, using "sticky" semantics.
    See :class:`repast4py.space.BorderType` and
    :class:`repast4py.space.GridStickyBorders` for more on sticky border semantics.

    Args:
        pt: A numpy scalar array with at least two elements
        min_max: A numpy array of format [min_x, max_x, min_y, max_y]
        pytorch_order: determines the order in which the neighbors are returned. If True,
            the y coordinates are returned first, otherwise, the x coordinates are returned first.
    Returns:
        A 2d numpy array with the neighboring points according to sticky border
        semantics, in the order determined by the pytorch_order argument,
        including the source point
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

    return np.stack((y_pts, x_pts), axis=0) if pytorch_order else np.stack((x_pts, y_pts), axis=0)


@jit(nopython=True)
def find_3d_nghs_sticky(pt: np.array, min_max: np.array, pytorch_order=False):
    """Finds the neighboring 3D point of the specifed point within
    the min_max inclusive range, using "sticky" semantics. See :class:`repast4py.space.BorderType` and
    :class:`repast4py.space.GridStickyBorders` for more on sticky border semantics.

    Args:
        pt: A numpy scalar array with at least three elements
        min_max: A numpy array of format [min_x, max_x, min_y, max_y, min_z, max_z]
        pytorch_order: determines the order in which the neighbors are returned. If True,
            the y coordinates are returned first, otherwise, the x coordinates are returned first.
    Returns:
        A 3d numpy array with the neighboring points according to sticky border
        semantics, in the order determined by the pytorch_order argument, including the source point.
    """
    x_pts = pt[0] + _xoffset_3d
    y_pts = pt[1] + _yoffset_3d
    z_pts = pt[2] + _zoffset_3d

    x_idxs = (x_pts >= min_max[0]) & (x_pts <= min_max[1])
    # remove out of range xs
    x_pts = x_pts[x_idxs]
    y_pts = y_pts[x_idxs]
    z_pts = z_pts[x_idxs]

    y_idxs = (y_pts >= min_max[2]) & (y_pts <= min_max[3])
    # remove out of range ys
    x_pts = x_pts[y_idxs]
    y_pts = y_pts[y_idxs]
    z_pts = z_pts[y_idxs]

    z_idxs = (z_pts >= min_max[4]) & (z_pts <= min_max[5])
    # remove out of range zs
    x_pts = x_pts[z_idxs]
    y_pts = y_pts[z_idxs]
    z_pts = z_pts[z_idxs]

    return np.stack((z_pts, y_pts, x_pts), axis=0) if pytorch_order else np.stack((x_pts, y_pts, z_pts), axis=0)


@jit(nopython=True)
def find_1d_nghs_periodic(pt: np.array, min_max: np.array):
    """Finds the neighboring 1D points of the specified point within
    the min_max inclusive range, using "periodic" semantics. See :class:`repast4py.space.BorderType` and
    :class:`repast4py.space.GridPeriodicBorders` for more on periodic border semantics.

    Args:
        pt: A numpy scalar array with at least one element
        min_max: A numpy array of format [min_x, max_x]

    Returns:
        A 1d numpy array with the neighboring points according to periodic border
        semantics, including the source point.

    Examples:
        Find the neighboring points of pt(4) within the bounds of 4 and 10.

        >>> from repast4py.space import DiscretePoint
        >>> import numpy as np
        >>> pt = DiscretePoint(4, 0, 0)
        >>> min_max = np.array([4, 10])
        >>> nghs = find_1d_nghs_periodic(pt.coordinates, min_max)
        >>> nghs
        array([10, 4, 5])
    """
    x_extent = min_max[1] - min_max[0] + 1
    x_pts = pt[0] + _xoffset_1d
    ncs = np.mod((x_pts - min_max[0]), x_extent)
    x_pts = np.where(ncs < 0, min_max[1] + ncs, min_max[0] + ncs)
    return x_pts


@jit(nopython=True)
def find_2d_nghs_periodic(pt: np.array, min_max: np.array, pytorch_order=False):
    """Finds the neighboring 2D points of the specified point within
    the min_max inclusive range, using "periodic" semantics. See :class:`repast4py.space.BorderType` and
    :class:`repast4py.space.GridPeriodicBorders` for more on periodic border semantics.

    Args:
        pt: A numpy scalar array with at least two elements
        min_max: A numpy array of format [min_x, max_x, min_y, max_y]
        pytorch_order: determines the order in which the neighbors are returned. If True,
            the y coordinates are returned first, otherwise, the x coordinates are returned first.

    Returns:
        A 2d numpy array with the neighboring points according to periodic border
        semantics, in the order determined by the pytorch_order argument, including the source point.
    """
    x_extent = min_max[1] - min_max[0] + 1
    x_pts = pt[0] + _xoffset_2d
    ncs = np.mod((x_pts - min_max[0]), x_extent)
    x_pts = np.where(ncs < 0, min_max[1] + ncs, min_max[0] + ncs)

    y_extent = min_max[3] - min_max[2] + 1
    y_pts = pt[1] + _yoffset_2d
    ncs = np.mod((y_pts - min_max[2]), y_extent)
    y_pts = np.where(ncs < 0, min_max[3] + ncs, min_max[2] + ncs)

    return np.stack((y_pts, x_pts), axis=0) if pytorch_order else np.stack((x_pts, y_pts), axis=0)


@jit(nopython=True)
def find_3d_nghs_periodic(pt: np.array, min_max: np.array, pytorch_order=False):
    """Finds the neighboring 3D points of the specified point within
    the min_max inclusive range, using "periodic" semantics. See :class:`repast4py.space.BorderType` and
    :class:`repast4py.space.GridPeriodicBorders` for more on periodic border semantics.

    Args:
        pt: A numpy scalar array with at least three elements
        min_max: A numpy array of format [min_x, max_x, min_y, max_y, min_z, max_z]
        pytorch_order: determines the order in which the neighbors are returned. If True,
            the y coordinates are returned first, otherwise, the x coordinates are returned first.
    Returns:
        A 3d numpy array with the neighboring points according to periodic border
        semantics, in the order determined by the pytorch_order argument, including the source point.
    """
    x_extent = min_max[1] - min_max[0] + 1
    x_pts = pt[0] + _xoffset_3d
    ncs = np.mod((x_pts - min_max[0]), x_extent)
    x_pts = np.where(ncs < 0, min_max[1] + ncs, min_max[0] + ncs)

    y_extent = min_max[3] - min_max[2] + 1
    y_pts = pt[1] + _yoffset_3d
    ncs = np.mod((y_pts - min_max[2]), y_extent)
    y_pts = np.where(ncs < 0, min_max[3] + ncs, min_max[2] + ncs)

    z_extent = min_max[5] - min_max[4] + 1
    z_pts = pt[2] + _zoffset_3d
    ncs = np.mod((z_pts - min_max[4]), z_extent)
    z_pts = np.where(ncs < 0, min_max[5] + ncs, min_max[4] + ncs)

    return np.stack((z_pts, y_pts, x_pts), axis=0) if pytorch_order else np.stack((x_pts, y_pts, z_pts), axis=0)
