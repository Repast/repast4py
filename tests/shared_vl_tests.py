import sys
import os
import torch
import unittest

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, BoundingBox, CartesianTopology
from repast4py.value_layer import SharedValueLayer

from mpi4py import MPI


# run with mpirun -n 9
class SharedValueLayerTests(unittest.TestCase):

    def test_buffers_1x2(self):
        # make 2 rank comm
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            bounds = BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, "random")
            self.assertEqual(2, vl.buffer_size)
            if rank == 0:
                exp_bounds = BoundingBox(0, 10, 0, 40, 0, 0)
                self.assertEqual(exp_bounds, vl.local_bounds)
                # self.assertEqual(1, len(vl.buffer_nghs))
                # self.assertEqual((1, (8, 10, 0, 40, 0, 0)), vl.buffer_nghs[0])
            else:
                exp_bounds = BoundingBox(10, 10, 0, 40, 0, 0)
                self.assertEqual(exp_bounds, vl.local_bounds)
                # self.assertEqual(1, len(vl.buffer_nghs))
                # self.assertEqual((0, (10, 12, 0, 40, 0, 0)), vl.buffer_nghs[0])

    def do_buffer_test(self, expected, buffer_nghs, msg):
        for bd in buffer_nghs:
            exp = expected.pop(bd[0])
            self.assertEqual(exp, bd[1], msg)
        self.assertEqual(0, len(expected), msg)

    def test_buffers_3x3_sticky(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, "random")
        self.assertEqual(2, vl.buffer_size)

        all_expected = {
            0: {
                1: (0, 30, 38, 40, 0, 0),
                3: (28, 30, 0, 40, 0, 0),
                4: (28, 30, 38, 40, 0, 0)
            },

            1: {
                0: (0, 30, 40, 42, 0, 0),
                2: (0, 30, 78, 80, 0, 0),
                3: (28, 30, 40, 42, 0, 0),
                4: (28, 30, 40, 80, 0, 0),
                5: (28, 30, 78, 80, 0, 0)
            },

            2:  {
                1: (0, 30, 80, 82, 0, 0),
                4: (28, 30, 80, 82, 0, 0),
                5: (28, 30, 80, 120, 0, 0)
            },

            3:  {
                0: (30, 32, 0, 40, 0, 0),
                1: (30, 32, 38, 40, 0, 0),
                4: (30, 60, 38, 40, 0, 0),
                6: (58, 60, 0, 40, 0, 0),
                7: (58, 60, 38, 40, 0, 0)
            },

            4: {
                0: (30, 32, 40, 42, 0, 0),
                1: (30, 32, 40, 80, 0, 0),
                2: (30, 32, 78, 80, 0, 0),
                3: (30, 60, 40, 42, 0, 0),
                5: (30, 60, 78, 80, 0, 0),
                6: (58, 60, 40, 42, 0, 0),
                7: (58, 60, 40, 80, 0, 0),
                8: (58, 60, 78, 80, 0, 0)
            },

            5: {
                1: (30, 32, 80, 82, 0, 0),
                2: (30, 32, 80, 120, 0, 0),
                4: (30, 60, 80, 82, 0, 0),
                7: (58, 60, 80, 82, 0, 0),
                8: (58, 60, 80, 120, 0, 0)
            },

            6: {
                3: (60, 62, 0, 40, 0, 0),
                4: (60, 62, 38, 40, 0, 0),
                7: (60, 90, 38, 40, 0, 0)
            },

            7: {
                3: (60, 62, 40, 42, 0, 0),
                4: (60, 62, 40, 80, 0, 0),
                5: (60, 62, 78, 80, 0, 0),
                6: (60, 90, 40, 42, 0, 0),
                8: (60, 90, 78, 80, 0, 0)
            },

            8: {
                4: (60, 62, 80, 82, 0, 0),
                5: (60, 62, 80, 120, 0, 0),
                7: (60, 90, 80, 82, 0, 0)
            }
        }

        # self.do_buffer_test(all_expected[rank], vl.buffer_nghs, str(rank))

    def test_buffers_3x3_periodic(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Periodic, 2, "random")
        self.assertEqual(2, vl.buffer_size)

        all_expected = {
            0:  {
                8: (0, 2, 0, 2, 0, 0),
                6: (0, 2, 0, 40, 0, 0),
                7: (0, 2, 38, 40, 0, 0),
                2: (0, 30, 0, 2, 0, 0),
                1: (0, 30, 38, 40, 0, 0),
                5: (28, 30, 0, 2, 0, 0),
                3: (28, 30, 0, 40, 0, 0),
                4: (28, 30, 38, 40, 0, 0)
            },

            1:  {
                6: (0, 2, 40, 42, 0, 0),
                7: (0, 2, 40, 80, 0, 0),
                8: (0, 2, 78, 80, 0, 0),
                0: (0, 30, 40, 42, 0, 0),
                2: (0, 30, 78, 80, 0, 0),
                3: (28, 30, 40, 42, 0, 0),
                4: (28, 30, 40, 80, 0, 0),
                5: (28, 30, 78, 80, 0, 0)
            },

            2:  {
                7: (0, 2, 80, 82, 0, 0),
                8: (0, 2, 80, 120, 0, 0),
                6: (0, 2, 118, 120, 0, 0),
                1: (0, 30, 80, 82, 0, 0),
                0: (0, 30, 118, 120, 0, 0),
                4: (28, 30, 80, 82, 0, 0),
                5: (28, 30, 80, 120, 0, 0),
                3: (28, 30, 118, 120, 0, 0)
            },

            3:  {
                2: (30, 32, 0, 2, 0, 0),
                0: (30, 32, 0, 40, 0, 0),
                1: (30, 32, 38, 40, 0, 0),
                5: (30, 60, 0, 2, 0, 0),
                4: (30, 60, 38, 40, 0, 0),
                8: (58, 60, 0, 2, 0, 0),
                6: (58, 60, 0, 40, 0, 0),
                7: (58, 60, 38, 40, 0, 0)
            },

            4: {
                0: (30, 32, 40, 42, 0, 0),
                1: (30, 32, 40, 80, 0, 0),
                2: (30, 32, 78, 80, 0, 0),
                3: (30, 60, 40, 42, 0, 0),
                5: (30, 60, 78, 80, 0, 0),
                6: (58, 60, 40, 42, 0, 0),
                7: (58, 60, 40, 80, 0, 0),
                8: (58, 60, 78, 80, 0, 0)
            },

            5: {
                1: (30, 32, 80, 82, 0, 0),
                2: (30, 32, 80, 120, 0, 0),
                0: (30, 32, 118, 120, 0, 0),
                4: (30, 60, 80, 82, 0, 0),
                3: (30, 60, 118, 120, 0, 0),
                7: (58, 60, 80, 82, 0, 0),
                8: (58, 60, 80, 120, 0, 0),
                6: (58, 60, 118, 120, 0, 0)
            },

            6: {
                5: (60, 62, 0, 2, 0, 0),
                3: (60, 62, 0, 40, 0, 0),
                4: (60, 62, 38, 40, 0, 0),
                8: (60, 90, 0, 2, 0, 0),
                7: (60, 90, 38, 40, 0, 0),
                2: (88, 90, 0, 2, 0, 0),
                0: (88, 90, 0, 40, 0, 0),
                1: (88, 90, 38, 40, 0, 0)
            },

            7: {
                3: (60, 62, 40, 42, 0, 0),
                4: (60, 62, 40, 80, 0, 0),
                5: (60, 62, 78, 80, 0, 0),
                6: (60, 90, 40, 42, 0, 0),
                8: (60, 90, 78, 80, 0, 0),
                0: (88, 90, 40, 42, 0, 0),
                1: (88, 90, 40, 80, 0, 0),
                2: (88, 90, 78, 80, 0, 0)
            },

            8: {
                4: (60, 62, 80, 82, 0, 0),
                5: (60, 62, 80, 120, 0, 0),
                3: (60, 62, 118, 120, 0, 0),
                7: (60, 90, 80, 82, 0, 0),
                6: (60, 90, 118, 120, 0, 0),
                1: (88, 90, 80, 82, 0, 0),
                2: (88, 90, 80, 120, 0, 0),
                0: (88, 90, 118, 120, 0, 0)
            }
        }

        # self.do_buffer_test(all_expected[rank], vl.buffer_nghs, str(rank))

    def test_buffers_3x3x3_sticky(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if comm.size != 18:
            if rank == 0:
                print("3D buffer tests not run -- run with -n 18")
            return
        bounds = BoundingBox(xmin=0, xextent=90, ymin=0,
                             yextent=120, zmin=0, zextent=60)
        vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, "random")
        self.assertEqual(2, vl.buffer_size)
        all_expected = {
            0: {
                1: (0, 30, 0, 40, 28, 30),
                2: (0, 30, 38, 40, 0, 30),
                3: (0, 30, 38, 40, 28, 30),
                6: (28, 30, 0, 40, 0, 30),
                7: (28, 30, 0, 40, 28, 30),
                8: (28, 30, 38, 40, 0, 30),
                9: (28, 30, 38, 40, 28, 30)
            },

            8: {
                0: (30, 32, 40, 42, 0, 30),
                1: (30, 32, 40, 42, 28, 30),
                2: (30, 32, 40, 80, 0, 30),
                3: (30, 32, 40, 80, 28, 30),
                4: (30, 32, 78, 80, 0, 30),
                5: (30, 32, 78, 80, 28, 30),
                6: (30, 60, 40, 42, 0, 30),
                7: (30, 60, 40, 42, 28, 30),
                9: (30, 60, 40, 80, 28, 30),
                10: (30, 60, 78, 80, 0, 30),
                11: (30, 60, 78, 80, 28, 30),
                12: (58, 60, 40, 42, 0, 30),
                13: (58, 60, 40, 42, 28, 30),
                14: (58, 60, 40, 80, 0, 30),
                15: (58, 60, 40, 80, 28, 30),
                16: (58, 60, 78, 80, 0, 30),
                17: (58, 60, 78, 80, 28, 30)
            },

            15: {
                6: (60, 62, 40, 42, 30, 32),
                7: (60, 62, 40, 42, 30, 60),
                8: (60, 62, 40, 80, 30, 32),
                9: (60, 62, 40, 80, 30, 60),
                10: (60, 62, 78, 80, 30, 32),
                11: (60, 62, 78, 80, 30, 60),
                12: (60, 90, 40, 42, 30, 32),
                13: (60, 90, 40, 42, 30, 60),
                14: (60, 90, 40, 80, 30, 32),
                16: (60, 90, 78, 80, 30, 32),
                17: (60, 90, 78, 80, 30, 60)
            }
        }
        # if rank in all_expected:
            # self.do_buffer_test(all_expected[rank], vl.buffer_nghs, str(rank))
        
    def test_buffers_3x3x3_periodic(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if comm.size != 18:
            if rank == 0:
                print("3D buffer tests not run -- run with -n 18")
            return
        bounds = BoundingBox(xmin=0, xextent=90, ymin=0,
                             yextent=120, zmin=0, zextent=60)
        vl = SharedValueLayer(comm, bounds, BorderType.Periodic, 2, "random")
        self.assertEqual(2, vl.buffer_size)
        all_expected = {
            0: {
                1: (0, 30, 0, 40, 28, 30),
                2: (0, 30, 38, 40, 0, 30),
                3: (0, 30, 38, 40, 28, 30),
                4: (0, 30, 0, 2, 0, 30),
                5: (0, 30, 0, 2, 28, 30),
                6: (28, 30, 0, 40, 0, 30),
                7: (28, 30, 0, 40, 28, 30),
                8: (28, 30, 38, 40, 0, 30),
                9: (28, 30, 38, 40, 28, 30),
                10: (28, 30, 0, 2, 0, 30),
                11: (28, 30, 0, 2, 28, 30),
                12: (0, 2, 0, 40, 0, 30),
                13: (0, 2, 0, 40, 28, 30),
                14: (0, 2, 38, 40, 0, 30),
                15: (0, 2, 38, 40, 28, 30),
                16: (0, 2, 0, 2, 0, 30),
                17: (0, 2, 0, 2, 28, 30)
            },

            15: {
                0: (88, 90, 40, 42, 30, 32),
                1: (88, 90, 40, 42, 30, 60),
                2: (88, 90, 40, 80, 30, 32),
                3: (88, 90, 40, 80, 30, 60),
                4: (88, 90, 78, 80, 30, 32),
                5: (88, 90, 78, 80, 30, 60),
                6: (60, 62, 40, 42, 30, 32),
                7: (60, 62, 40, 42, 30, 60),
                8: (60, 62, 40, 80, 30, 32),
                9: (60, 62, 40, 80, 30, 60),
                10: (60, 62, 78, 80, 30, 32),
                11: (60, 62, 78, 80, 30, 60),
                12: (60, 90, 40, 42, 30, 32),
                13: (60, 90, 40, 42, 30, 60),
                14: (60, 90, 40, 80, 30, 32),
                16: (60, 90, 78, 80, 30, 32),
                17: (60, 90, 78, 80, 30, 60)
            }
        }

        # if rank in all_expected:
        #    self.do_buffer_test(all_expected[rank], vl.buffer_nghs, str(rank))

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


    def test_sync_3x3_sticky(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        bounds = BoundingBox(
            xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        vl = SharedValueLayer(comm, bounds, BorderType.Sticky, 2, rank)
        self.assertEqual(2, vl.buffer_size)

        grid = vl[:,:]
        self.assertEqual(grid.shape[0], vl.buffered_bounds.yextent)
        self.assertEqual(grid.shape[1], vl.buffered_bounds.xextent)
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        vl._synch_ghosts(None, None)

        lb = vl.local_bounds
        grid = vl[lb.xmin: lb.xmin + lb.xextent, lb.ymin: lb.ymin + lb.yextent]
        # == rank within local bounds
        self.assertFalse(torch.any(torch.ne(grid, rank)))

        # expected border values
        exp = {
            0: [
                  (3, (slice(30, 32), slice(0, 40))),
                  (1, (slice(0, 30), slice(40, 42))),
                  (4, (slice(30, 32), slice(40, 42)))
            ],

            1: [
                (0, (slice(0, 30), slice(38, 40))),
                (2, (slice(0, 30), slice(80, 82))),
                (4, (slice(30, 32), slice(40, 80))),
                (5, (slice(30, 32), slice(80, 82))),
                (3, (slice(30, 32), slice(38, 40))),
            ],

            2: [
                (1, (slice(0, 30), slice(78, 80))),
                (5, (slice(30, 32), slice(80, 120))),
                (4, (slice(30, 32), slice(78, 80)))
            ],

            3: [
                (0, (slice(28, 30), slice(0, 40))),
                (6, (slice(60, 62), slice(0, 40))),
                (4, (slice(30, 60), slice(40, 42))),
                (1, (slice(28, 30), slice(40, 42))),
                (7, (slice(60, 62), slice(40, 42)))
            ],
            
            4: [
                (0, (slice(28, 30), slice(38, 40))),
                (1, (slice(28, 30), slice(40, 80))),
                (2, (slice(28, 30), slice(80, 82))),
                (3, (slice(30, 60), slice(38, 40))),
                (5, (slice(30, 60), slice(80, 82))),
                (6, (slice(60, 62), slice(38, 40))),
                (7, (slice(60, 62), slice(40, 80))),
                (8, (slice(60, 62), slice(80, 82)))
            ],

            5: [
                (1, (slice(28, 30), slice(78, 80))),
                (2, (slice(28, 30), slice(80, 120))),
                (4, (slice(30, 60), slice(78, 80))),
                (7, (slice(60, 62), slice(78, 80))),
                (8, (slice(60, 62), slice(80, 120)))
            ],

            6: [
                (3, (slice(58, 60), slice(0, 40))),
                (4, (slice(58, 60), slice(40, 42))),
                (7, (slice(60, 90), slice(40, 42)))
            ],

            7: [
                (3, (slice(58, 60), slice(38, 40))),
                (6, (slice(60, 70), slice(38, 40))),
                (4, (slice(58, 60), slice(40, 80))),
                (5, (slice(58, 60), slice(80, 82))),
                (8, (slice(60, 90), slice(80, 82)))
            ],

            8: [
                (4, (slice(58, 60), slice(78, 80))),
                (7, (slice(60, 90), slice(78, 80))),
                (5, (slice(58, 60), slice(80, 120)))
            ]
        }

        
        for exp_val, slices in exp[rank]:
            grid = vl[slices]
            self.assertFalse(torch.any(torch.ne(grid, exp_val)))

        vl[lb.xmin: lb.xmin + lb.xextent, lb.ymin: lb.ymin + lb.yextent] = rank + 101.1
        grid = vl[lb.xmin: lb.xmin + lb.xextent, lb.ymin: lb.ymin + lb.yextent]
        self.assertFalse(torch.any(torch.ne(grid, rank + 101.1)))
    
        vl._synch_ghosts(None, None)
        
        for exp_val, slices in exp[rank]:
            grid = vl[slices]
            self.assertFalse(torch.any(torch.ne(grid, exp_val + 101.1)))
