import unittest
import sys
import os

from mpi4py import MPI

try:
    from repast4py.space import DiscretePoint as dpt
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py.space import DiscretePoint as dpt

from repast4py.space import BorderType, BoundingBox, CartesianTopology


# local bounds are used to determine if an agent
# moves out of bounds. So, of local bounds are
# correct then that part of the moving code is
# correct
class LocalBoundsTests(unittest.TestCase):

    def test_local_bounds_1x9_sticky(self):
        exp = {}
        for i in range(0, 9):
            exp[i] = BoundingBox(xmin=i * 10, xextent=10, ymin=0, yextent=0, zmin=0, zextent=0)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(xmin=0, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        topo = CartesianTopology(comm, bounds, False)
        self.assertEqual(exp[rank], topo.local_bounds, msg=f'r: {rank}')

    def test_local_bounds_1x9_periodic(self):
        exp = {}
        for i in range(0, 9):
            exp[i] = BoundingBox(xmin=i * 10, xextent=10, ymin=0, yextent=0, zmin=0, zextent=0)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(xmin=0, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        topo = CartesianTopology(comm, bounds, True)
        self.assertEqual(exp[rank], topo.local_bounds, msg=f'r: {rank}')

    def test_local_bounds_3x3_sticky(self):
        exp = {}
        i = 0
        for x in range(0, 90, 30):
            for y in range(0, 120, 40):
                exp[i] = BoundingBox(xmin=x, xextent=30, ymin=y, yextent=40, zmin=0, zextent=0)
                i += 1

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        topo = CartesianTopology(comm, bounds, False)
        self.assertEqual(exp[rank], topo.local_bounds, msg=f'r: {rank}')

    def test_local_bounds_3x3_periodic(self):
        exp = {}
        i = 0
        for x in range(0, 90, 30):
            for y in range(0, 120, 40):
                exp[i] = BoundingBox(xmin=x, xextent=30, ymin=y, yextent=40, zmin=0, zextent=0)
                i += 1

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        topo = CartesianTopology(comm, bounds, True)
        self.assertEqual(exp[rank], topo.local_bounds, msg=f'r: {rank}')

    def test_local_bounds_4x2_sticky(self):
        exp = {}
        i = 0
        for x in range(0, 40, 10):
            for y in range(0, 120, 60):
                exp[i] = BoundingBox(xmin=x, xextent=10, ymin=y, yextent=60, zmin=0, zextent=0)
                i += 1

        new_group = MPI.COMM_WORLD.Get_group().Excl([8])
        comm = MPI.COMM_WORLD.Create(new_group)
        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            bounds = BoundingBox(xmin=0, xextent=40, ymin=0, yextent=120, zmin=0, zextent=0)
            topo = CartesianTopology(comm, bounds, False)
            self.assertEqual(exp[rank], topo.local_bounds, msg=f'r: {rank}')
