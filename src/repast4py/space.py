from mpi4py import MPI

from ._space import Grid, DiscretePoint
from ._space import SharedGrid as _SharedGrid

from enum import Enum
from collections import namedtuple
import sys

class BorderType:
    Sticky = 0

class OccupancyType:
    Multiple = 0

if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    BoundingBox = namedtuple('BoundingBox', ['xmin', 'xextent', 'ymin', 'yextent', 'zmin', 'zextent'],
        defaults=[0, 0])
else:
    BoundingBox = namedtuple('BoundingBox', ['xmin', 'xextent', 'ymin', 'yextent', 'zmin', 'zextent'])



class SharedGrid(_SharedGrid):

    def __init__(self, name, bounds, borders, occupancy, buffersize, comm):
        super().__init__(name, bounds, borders, occupancy, buffersize, comm)

    def synchronize_buffer(self, create_agent):
        pass

    
