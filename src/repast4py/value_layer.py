import torch
import numpy as np

from . import geometry

from .space import BorderType, GridStickyBorders, GridPeriodicBorders, CartesianTopology, BoundingBox
from .space import DiscretePoint as dpt


class Impl1D:

    def __init__(self, bounds, borders, init_value, dtype):
        self.tpt = dpt(0, 0, 0)
        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)

        if init_value == 'random':
            self.grid = torch.rand((bounds.xextent,), dtype=dtype)
        else:
            self.grid = torch.full((bounds.xextent,), init_value, dtype=dtype)
        self.ngh_translation = 0 - bounds.xmin
        self.translation = np.array([self.ngh_translation, 0, 0])
        self.ngh_finder = geometry.find_1d_nghs_sticky if borders == BorderType.Sticky else geometry.find_1d_nghs_periodic
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1])
        
    
    def get(self, pt):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.translation
        return self.grid[npt[0]]

    def set(self, pt, val):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.translation
        self.grid[npt[0]] = val

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max)
        grid_idxs = nghs + self.ngh_translation
        return (self.grid[grid_idxs], nghs)


class Impl2D:

    def __init__(self, bounds, borders, init_value, dtype):
        self.tpt = dpt(0, 0, 0)
        self.idx = np.array([[0], [0]])
        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)

        if init_value == 'random':
            self.grid = torch.rand(
                (bounds.yextent, bounds.xextent), dtype=dtype)
        else:
            self.grid = torch.full((bounds.yextent, bounds.xextent), init_value, dtype=dtype)

        self.pt_translation = np.array([0 - bounds.xmin, 0 - bounds.ymin, 0])
        self.ngh_translation = np.array([[0 - bounds.ymin], [0 - bounds.xmin]])
        self.ngh_finder = geometry.find_2d_nghs_sticky if borders == BorderType.Sticky else geometry.find_2d_nghs_periodic
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1, bounds.ymin, bounds.ymin + bounds.yextent - 1])
            
    
    def _update_idx(self, pt):
        self.idx[0, 0] = pt[1]
        self.idx[1, 0] = pt[0]

        return self.idx


    def get(self, pt):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        return self.grid[self._update_idx(npt)]

    def set(self, pt, val):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        self.grid[self._update_idx(npt)] = val

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max, row_major=True)
        grid_idxs = nghs + self.ngh_translation
        # swap to remove the row major
        nghs[[0, 1]] = nghs[[1, 0]]
        return (self.grid[grid_idxs], nghs)


class Impl3D:

    def __init__(self, bounds, borders, init_value, dtype):
        self.tpt = dpt(0, 0, 0)
        self.idx = np.array([[0], [0], [0]])
        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)

        if init_value == 'random':
            self.grid = torch.rand(
                (bounds.yextent, bounds.xextent, bounds.zextent), dtype=dtype)
        else:
            self.grid = torch.full(
                (bounds.yextent, bounds.xextent, bounds.zextent), init_value, dtype=dtype)

        # row major
        self.pt_translation = np.array([0 - bounds.xmin, 0 - bounds.ymin, 0 - bounds.zmin])
        self.ngh_translation = np.array([[self.pt_translation[1]], [self.pt_translation[0]], [self.pt_translation[2]]])
        self.ngh_finder = None #geometry.find_3d_nghs_sticky if borders == BorderType.Sticky else geometry.find_3d_nghs_periodic,
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1,
             bounds.ymin, bounds.ymin + bounds.xextent - 1,
             bounds.zmin, bounds.zmin + bounds.zextent - 1])

    def _update_idx(self, pt):
        self.idx[0, 0] = pt[1]
        self.idx[1, 0] = pt[0]
        self.idx[2, 0] = pt[2]
        
        return self.idx

    def get(self, pt):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        return self.grid[self._update_idx(npt)]

    def set(self, pt, val):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        self.grid[self._update_idx(npt)] = val

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max, row_major=True) 
        grid_idxs = nghs + self.ngh_translation
        # swap to remove the row major
        nghs[[0, 1, 2]] = nghs[[1, 0, 2]]
        return (self.grid[grid_idxs], nghs)


class ValueLayer:

    def __init__(self, bounds, borders, init_value, dtype=torch.float64):

        if bounds.yextent == 0:
            self.impl = Impl1D(bounds, borders, init_value, dtype)
        elif bounds.zextent == 0:
            self.impl = Impl2D(bounds, borders, init_value, dtype)
        else:
            self.impl = Impl3D(bounds, borders, init_value, dtype)
        self.bbounds = bounds

    @property
    def grid(self):
        return self.impl.grid

    @property
    def bounds(self):
        return self.bbounds
            
    def get(self, pt):
        return self.impl.get(pt)
        
    def set(self, pt, val):
        self.impl.set(pt, val)

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        return self.impl.get_nghs(pt, extent)
        

class SharedValueLayer(ValueLayer):

    def __init__(self, comm, bounds, borders, buffer_size, init_value, dtype=torch.float64):
        periodic = borders == BorderType.Periodic
        ct = CartesianTopology(comm, bounds, periodic)
        self.cart_comm = ct.comm
        # tuple of length dims
        self.coords = ct.coordinates
        self.buffer_size = buffer_size
        # bounds in as BoundingBox
        local_bounds = ct.local_bounds
        bxmin = local_bounds.xmin - buffer_size
        bxexent = local_bounds.xextent + buffer_size
        bymin = byextent = 0
        if local_bounds.yextent > 0:
            bymin = local_bounds.ymin - buffer_size
            byextent = local_bounds.yextent + buffer_size
        bzmin = bzextent = 0
        if local_bounds.zextent > 0:
            bzmin = local_bounds.zmin - buffer_size
            bzextent = local_bounds.zextent + buffer_size

        self.local_bounds = local_bounds
        
        buffered_bounds = BoundingBox(bxmin, bxexent, bymin, byextent, bzmin, bzextent)
        super().__init__(buffered_bounds, borders, init_value, dtype)
        self.buffer_nghs = [x for x in ct.compute_buffer_nghs(buffer_size)]

