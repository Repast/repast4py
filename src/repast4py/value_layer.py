import torch
import numpy as np

from .space import BorderType, GridStickyBorders, GridPeriodicBorders, BoundingBox
from .space import DiscretePoint as dpt


def index1D(pt):
    return pt[0]


def index2D(pt):
    # row major so swap
    pt[0], pt[1] = pt[1], pt[0]
    return pt[ : -1]


def index3D(pt):
    # row major so swap
    pt[0], pt[1] = pt[1], pt[0]
    return pt


class ValueLayer:

    def __init__(self, bounds, borders, init_value, dtype=torch.float64):
        self.tpt = dpt(0, 0, 0)
        if bounds.yextent == 0:
            # 1D
            self.grid = torch.full((bounds.xextent,), init_value, dtype=dtype)
            self.translation = np.array([0 - bounds.xmin])
            self.index = index1D
        elif bounds.zextent == 0:
            # 2D
            # row major
            self.grid = torch.full(init_value, (bounds.yextent, bounds.xextent), dtype=dtype)
            self.translation = np.array([0 - bounds.xmin,  0 - bounds.ymin])
            self.index = index2D
        else:
            # 3D
            # row major
            self.grid = torch.full(init_value, (bounds.yextent, bounds.xextent, bounds.zextent), dtype=dtype)
            self.translation = np.array([0 - bounds.xmin,  0 - bounds.ymin, 0 - bounds.zmin])
            self.index = index3D

        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)
        else:
            raise ValueError('Invalid border type {}'.format(borders))
        
        
    def get(self, pt):
        self.borders._transform(pt, self.tpt)
        npt = self.tpt.coordinates + self.translation
        return self.grid[self.index(npt)]
    
    def set(self, pt, val):
        self.borders._transform(pt, self.tpt)
        npt = self.tpt.coordinates + self.translation
        self.grid[self.index(npt)] = val
        

    def get_ngh(self):
        # get ngh via a slice and return as tensor
        pass


