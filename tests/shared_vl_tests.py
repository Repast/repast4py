import sys
import os
import torch
import unittest

from mpi4py import MPI

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, BoundingBox, CartesianTopology
from repast4py.value_layer import SharedValueLayer

# TODO: TEST:
# value_layer.buffered bounds -- 1D, 2D, 3D for sticky and periodic
# value_layer.synch for 1D, 2D, and 3D -- use 2D synch as template


# run with mpirun -n 9
class SharedValueLayerTests(unittest.TestCase):
    def test_bbounds_1x9_sticky(self):
        exp = {
            0: BoundingBox(xmin=0, xextent=12, ymin=0, yextent=0, zmin=0, zextent=0),
            1: BoundingBox(xmin=8, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            2: BoundingBox(xmin=18, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            3: BoundingBox(xmin=28, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            4: BoundingBox(xmin=38, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            5: BoundingBox(xmin=48, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            6: BoundingBox(xmin=58, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            7: BoundingBox(xmin=68, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            8: BoundingBox(xmin=78, xextent=12, ymin=0, yextent=0, zmin=0, zextent=0)
        }
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, rank)
        self.assertEqual(exp[rank], vl.buffered_bounds, msg=f'r: {rank}')

    def test_bbounds_1x9_periodic(self):
        exp = {
            0: BoundingBox(xmin=0, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            1: BoundingBox(xmin=8, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            2: BoundingBox(xmin=18, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            3: BoundingBox(xmin=28, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            4: BoundingBox(xmin=38, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            5: BoundingBox(xmin=48, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            6: BoundingBox(xmin=58, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            7: BoundingBox(xmin=68, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0),
            8: BoundingBox(xmin=78, xextent=14, ymin=0, yextent=0, zmin=0, zextent=0)
        }
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Periodic, 2, rank)
        self.assertEqual(exp[rank], vl.buffered_bounds, msg=f'r: {rank}')

    def test_bbounds_3x3_sticky(self):
        exp = {
            0: BoundingBox(xmin=0, xextent=32, ymin=0, yextent=42, zmin=0, zextent=0),
            1: BoundingBox(xmin=0, xextent=32, ymin=38, yextent=44, zmin=0, zextent=0),
            2: BoundingBox(xmin=0, xextent=32, ymin=78, yextent=42, zmin=0, zextent=0),
            3: BoundingBox(xmin=28, xextent=34, ymin=0, yextent=42, zmin=0, zextent=0),
            4: BoundingBox(xmin=28, xextent=34, ymin=38, yextent=44, zmin=0, zextent=0),
            5: BoundingBox(xmin=28, xextent=34, ymin=78, yextent=42, zmin=0, zextent=0),
            6: BoundingBox(xmin=58, xextent=32, ymin=0, yextent=42, zmin=0, zextent=0),
            7: BoundingBox(xmin=58, xextent=32, ymin=38, yextent=44, zmin=0, zextent=0),
            8: BoundingBox(xmin=58, xextent=32, ymin=78, yextent=42, zmin=0, zextent=0)
        }

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, rank)
        self.assertEqual(exp[rank], vl.buffered_bounds, msg=f'r: {rank}')

    def test_bbounds_3x3_periodic(self):
        exp = {
            0: BoundingBox(xmin=0, xextent=34, ymin=0, yextent=44, zmin=0, zextent=0),
            1: BoundingBox(xmin=0, xextent=34, ymin=38, yextent=44, zmin=0, zextent=0),
            2: BoundingBox(xmin=0, xextent=34, ymin=78, yextent=44, zmin=0, zextent=0),
            3: BoundingBox(xmin=28, xextent=34, ymin=0, yextent=44, zmin=0, zextent=0),
            4: BoundingBox(xmin=28, xextent=34, ymin=38, yextent=44, zmin=0, zextent=0),
            5: BoundingBox(xmin=28, xextent=34, ymin=78, yextent=44, zmin=0, zextent=0),
            6: BoundingBox(xmin=58, xextent=34, ymin=0, yextent=44, zmin=0, zextent=0),
            7: BoundingBox(xmin=58, xextent=34, ymin=38, yextent=44, zmin=0, zextent=0),
            8: BoundingBox(xmin=58, xextent=34, ymin=78, yextent=44, zmin=0, zextent=0)
        }

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Periodic, 2, rank)
        self.assertEqual(exp[rank], vl.buffered_bounds, msg=f'r: {rank}')

    def test_synch_1x9_sticky(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, rank)
        self.assertEqual(2, vl.buffer_size)

        grid = vl.grid[:]
        self.assertEqual(grid.shape[0], vl.buffered_bounds.xextent)

        # offsets for slice
        lb = vl.local_bounds
        x1 = lb.xmin + vl.impl.pt_translation[0]
        x2 = lb.xmin + lb.xextent + vl.impl.pt_translation[0]
        # get local part of grid
        grid = vl.grid[x1: x2]
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        vl._synch_ghosts()

        grid = vl.grid[x1: x2]
        # Test: values = rank within local bounds
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        # TEST: buffer border areas == neighboring rank
        exp = {
            0: [(1, (10, 12))],
            8: [(7, (78, 80))]
        }
        for i in range(1, 8):
            r1 = i - 1
            c1r1 = i * 10 - 2
            c2r1 = c1r1 + 2

            r2 = i + 1
            c1r2 = r2 * 10
            c2r2 = c1r2 + 2
            exp[i] = [
                (r1, (c1r1, c2r1)),
                (r2, (c1r2, c2r2))
            ]

        for exp_val, xs in exp[rank]:
            # Prior version of value_layer implemented slicing (but not comppletely),
            # so that's why this looks kind of clunky as do the slice translations
            # here, rather than in the vl code as previous.
            grid = vl.grid[xs[0] + vl.impl.pt_translation[0]: xs[1] + vl.impl.pt_translation[0]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val)), msg=f'{rank}: {exp_val} {xs} {grid}')

        # TEST: Update local area, synch, borders should have changed
        vl.grid[x1: x2] = rank + 101.1
        vl._synch_ghosts()

        for exp_val, xs in exp[rank]:
            # Prior version of value_layer implemented slicing (but not comppletely),
            # so that's why this looks kind of clunky as do the slice translations
            # here, rather than in the vl code as previous.
            grid = vl.grid[xs[0] + vl.impl.pt_translation[0]: xs[1] + vl.impl.pt_translation[0]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val + 101.1)), msg=f'{rank}: {exp_val} {xs} {grid}')

    def test_synch_1x9_periodic(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Periodic, 2, rank)
        self.assertEqual(2, vl.buffer_size)

        # offsets for slice
        lb = vl.local_bounds
        x1 = lb.xmin + vl.impl.pt_translation[0]
        x2 = lb.xmin + lb.xextent + vl.impl.pt_translation[0]
        # get local part of grid
        grid = vl.grid[x1: x2]
        self.assertFalse(torch.any(torch.ne(grid, rank)))
        self.assertEqual(vl.grid[:].shape[0], vl.buffered_bounds.xextent)

        vl._synch_ghosts()

        # Test: values = rank within local bounds
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        # TEST: buffer border areas == neighboring rank
        exp = {
            0: [
                (1, (12, 14)),
                (8, (0, 2))
            ],
            8: [
                (7, (0, 2)),
                (0, (12, 14))
            ]
        }
        for i in range(1, 8):
            r1 = i - 1
            r2 = i + 1
            exp[i] = [
                (r1, (0, 2)),
                (r2, (12, 24))
            ]

        for exp_val, xs in exp[rank]:
            grid = vl.grid[xs[0]: xs[1]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val)), msg=f'{rank}: {exp_val} {xs} {grid}')

        # TEST: Update local area, synch, borders should have changed
        vl.grid[x1: x2] = rank + 101.1
        vl._synch_ghosts()

        for exp_val, xs in exp[rank]:
            grid = vl.grid[xs[0]: xs[1]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val + 101.1)), msg=f'{rank}: {exp_val} {xs} {grid}')

    def test_synch_1x9_periodic_with_poffset(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=4, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Periodic, 2, rank)
        self.assertEqual(2, vl.buffer_size)

        # offsets for slice
        lb = vl.local_bounds
        x1 = lb.xmin + vl.impl.pt_translation[0]
        x2 = lb.xmin + lb.xextent + vl.impl.pt_translation[0]
        # get local part of grid
        grid = vl.grid[x1: x2]
        self.assertFalse(torch.any(torch.ne(grid, rank)))
        self.assertEqual(vl.grid[:].shape[0], vl.buffered_bounds.xextent)

        vl._synch_ghosts()

        # Test: values = rank within local bounds
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        # TEST: buffer border areas == neighboring rank
        exp = {
            0: [
                (1, (12, 14)),
                (8, (0, 2))
            ],
            8: [
                (7, (0, 2)),
                (0, (12, 14))
            ]
        }
        for i in range(1, 8):
            r1 = i - 1
            r2 = i + 1
            exp[i] = [
                (r1, (0, 2)),
                (r2, (12, 24))
            ]

        for exp_val, xs in exp[rank]:
            grid = vl.grid[xs[0]: xs[1]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val)), msg=f'{rank}: {exp_val} {xs} {grid}')

        # TEST: Update local area, synch, borders should have changed
        vl.grid[x1: x2] = rank + 101.1
        vl._synch_ghosts()

        for exp_val, xs in exp[rank]:
            grid = vl.grid[xs[0]: xs[1]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val + 101.1)), msg=f'{rank}: {exp_val} {xs} {grid}')

    def test_synch_1x9_periodic_with_noffset(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=-4, xextent=90, ymin=0, yextent=0, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Periodic, 2, rank)
        self.assertEqual(2, vl.buffer_size)

        # offsets for slice
        lb = vl.local_bounds
        x1 = lb.xmin + vl.impl.pt_translation[0]
        x2 = lb.xmin + lb.xextent + vl.impl.pt_translation[0]
        # get local part of grid
        grid = vl.grid[x1: x2]
        self.assertFalse(torch.any(torch.ne(grid, rank)))
        self.assertEqual(vl.grid[:].shape[0], vl.buffered_bounds.xextent)

        vl._synch_ghosts()

        # Test: values = rank within local bounds
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        # TEST: buffer border areas == neighboring rank
        exp = {
            0: [
                (1, (12, 14)),
                (8, (0, 2))
            ],
            8: [
                (7, (0, 2)),
                (0, (12, 14))
            ]
        }
        for i in range(1, 8):
            r1 = i - 1
            r2 = i + 1
            exp[i] = [
                (r1, (0, 2)),
                (r2, (12, 24))
            ]

        for exp_val, xs in exp[rank]:
            grid = vl.grid[xs[0]: xs[1]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val)), msg=f'{rank}: {exp_val} {xs} {grid}')

        # TEST: Update local area, synch, borders should have changed
        vl.grid[x1: x2] = rank + 101.1
        vl._synch_ghosts()

        for exp_val, xs in exp[rank]:
            grid = vl.grid[xs[0]: xs[1]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val + 101.1)), msg=f'{rank}: {exp_val} {xs} {grid}')

    def test_synch_3x3_sticky(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, rank)
        self.assertEqual(2, vl.buffer_size)

        grid = vl.grid[:]
        self.assertEqual(grid.shape[0], vl.buffered_bounds.yextent)
        self.assertEqual(grid.shape[1], vl.buffered_bounds.xextent)

        # offsets for slice
        lb = vl.local_bounds
        y1 = lb.ymin + vl.impl.pt_translation[1]
        y2 = lb.ymin + lb.yextent + vl.impl.pt_translation[1]
        x1 = lb.xmin + vl.impl.pt_translation[0]
        x2 = lb.xmin + lb.xextent + vl.impl.pt_translation[0]
        grid = vl.grid[y1: y2, x1: x2]

        self.assertFalse(torch.any(torch.ne(grid, rank)))

        vl._synch_ghosts()
        # == rank within local bounds
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        # expected border values
        exp = {
            0: [
                (3, ((30, 32), (0, 40))),
                (1, ((0, 30), (40, 42))),
                (4, ((30, 32), (40, 42)))
            ],

            1: [
                (0, ((0, 30), (38, 40))),
                (2, ((0, 30), (80, 82))),
                (4, ((30, 32), (40, 80))),
                (5, ((30, 32), (80, 82))),
                (3, ((30, 32), (38, 40))),
            ],

            2: [
                (1, ((0, 30), (78, 80))),
                (5, ((30, 32), (80, 120))),
                (4, ((30, 32), (78, 80)))
            ],

            3: [
                (0, ((28, 30), (0, 40))),
                (6, ((60, 62), (0, 40))),
                (4, ((30, 60), (40, 42))),
                (1, ((28, 30), (40, 42))),
                (7, ((60, 62), (40, 42)))
            ],

            4: [
                (0, ((28, 30), (38, 40))),
                (1, ((28, 30), (40, 80))),
                (2, ((28, 30), (80, 82))),
                (3, ((30, 60), (38, 40))),
                (5, ((30, 60), (80, 82))),
                (6, ((60, 62), (38, 40))),
                (7, ((60, 62), (40, 80))),
                (8, ((60, 62), (80, 82)))
            ],

            5: [
                (1, ((28, 30), (78, 80))),
                (2, ((28, 30), (80, 120))),
                (4, ((30, 60), (78, 80))),
                (7, ((60, 62), (78, 80))),
                (8, ((60, 62), (80, 120)))
            ],

            6: [
                (3, ((58, 60), (0, 40))),
                (4, ((58, 60), (40, 42))),
                (7, ((60, 90), (40, 42)))
            ],

            7: [
                (3, ((58, 60), (38, 40))),
                (6, ((60, 70), (38, 40))),
                (4, ((58, 60), (40, 80))),
                (5, ((58, 60), (80, 82))),
                (8, ((60, 90), (80, 82)))
            ],

            8: [
                (4, ((58, 60), (78, 80))),
                (7, ((60, 90), (78, 80))),
                (5, ((58, 60), (80, 120)))
            ]
        }

        for exp_val, slices in exp[rank]:
            xs, ys = slices
            # Prior version of value_layer implemented slicing (but not comppletely),
            # so that's why this looks kind of clunky as do the slice translations
            # here, rather than in the vl code as previous.
            grid = vl.grid[ys[0] + vl.impl.pt_translation[1]: ys[1] + vl.impl.pt_translation[1],
                           xs[0] + vl.impl.pt_translation[0]: xs[1] + vl.impl.pt_translation[0]]
            self.assertFalse(torch.any(torch.ne(grid, exp_val)))

        # update unbuffered section
        vl.grid[y1: y2, x1: x2] = rank + 101.1

        vl._synch_ghosts()

        for exp_val, slices in exp[rank]:
            xs, ys = slices
            grid = vl.grid[ys[0] + vl.impl.pt_translation[1]: ys[1] + vl.impl.pt_translation[1],
                           xs[0] + vl.impl.pt_translation[0]: xs[1] + vl.impl.pt_translation[0]]
            # if rank == 0:
            #     print(grid.shape)
            #     print(grid)
            self.assertFalse(torch.any(torch.ne(grid, exp_val + 101.1)))
