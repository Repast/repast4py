import sys
import os
from mpi4py import MPI
import numpy as np
import random
import math
import traceback
import unittest

try:
    from repast4py import core, space, geometry, random
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py import core, space, geometry, random

from repast4py import context as ctx
from repast4py.space import ContinuousPoint as CPt
from repast4py.space import DiscretePoint as DPt
from repast4py.space import BorderType, OccupancyType, CartesianTopology

# run with -n 9

# void create_test_comm(MPI_Comm* test_comm) {
#     MPI_Group world_group;
#     MPI_Comm_group(MPI_COMM_WORLD, &world_group);
#     const int exclude[1] = {8};
#     MPI_Group test_group;
#     MPI_Group_excl(world_group, 1, exclude, &test_group);
#     MPI_Comm_create_group(MPI_COMM_WORLD, test_group, 0, test_comm);
# }


def mp(x, y):
    return space.DiscretePoint(x, y)


def cp(x, y):
    return space.ContinuousPoint(x, y)


def make_move(grid, agent, x, y, target, expected):
    grid.move(agent, space.DiscretePoint(x, y))
    expected[agent.uid] = (target, np.array([x, y, 0]))


class SharedCSTests(unittest.TestCase):

    long_message = True

    def test_num_agents(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)
            b1 = core.Agent(1, 1, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)

            cspace.add(a1)
            cspace.add(a2)
            cspace.add(b1)

            pt = CPt(1.1, 1.4, 0)
            cspace.move(a1, pt)
            cspace.move(a2, pt)
            cspace.move(b1, pt)

            self.assertEqual(3, cspace.get_num_agents(pt))
            self.assertEqual(2, cspace.get_num_agents(pt, 0))
            self.assertEqual(1, cspace.get_num_agents(pt, agent_type=1))

    def test_ops(self):
        # make 2 rank comm
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)
            self.assertEqual('shared_space', cspace.name)

            cspace.add(a1)
            cspace.add(a2)

            if (rank == 0):
                pt = space.ContinuousPoint(5.5, 15.7)
                cspace.move(a1, pt)
                self.assertEqual(a1, cspace.get_agent(pt))
                self.assertEqual(pt, cspace.get_location(a1))

                agents = cspace.get_agents(pt)
                expected = [a1]
                count = 0
                for i, agent in enumerate(agents):
                    self.assertEqual(expected[i], agent)
                    count += 1
                self.assertEqual(1, count)

                cspace.remove(a1)
                self.assertIsNone(cspace.get_agent(pt))

            if (rank == 1):
                pt = space.ContinuousPoint(12, 38.2)
                cspace.move(a2, pt)
                self.assertEqual(a2, cspace.get_agent(pt))
                self.assertEqual(pt, cspace.get_location(a2))

                a3 = core.Agent(3, 0, rank)
                cspace.add(a3)
                cspace.move(a3, pt)

                agents = cspace.get_agents(pt)
                expected = [a2, a3]
                count = 0
                for i, agent in enumerate(agents):
                    self.assertEqual(expected[i], agent)
                    count += 1
                self.assertEqual(2, count)

    def test_oob(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_cspace", bounds=box, borders=BorderType.Sticky,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)

            cspace.add(a1)
            cspace.add(a2)

            if (rank == 0):
                pt = space.ContinuousPoint(5.5, 20)
                cspace.move(a1, pt)
                # move out of bounds
                pt = space.ContinuousPoint(12.3, 22.75)
                cspace.move(a1, pt)

                expected = {(1, 0, 0): (1, np.array([12.3, 22.75, 0]))}
                for ob in cspace._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

            if (rank == 1):
                a3 = core.Agent(1, 0, rank)
                cspace.add(a3)

                pt = space.ContinuousPoint(12, 39)
                cspace.move(a2, pt)
                cspace.move(a3, pt)

                cspace.move(a2, space.ContinuousPoint(0, 1))
                cspace.move(a3, space.ContinuousPoint(8, 200))

                expected = {(2, 0, 1): (0, np.array([0.0, 1.0, 0.0])),
                            (1, 0, 1): (0, np.array([8.0, 39.99999999, 0.0]))}
                for ob in cspace._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]), msg='{}, {}'.format(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

    def test_oob_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_cspace", bounds=box, borders=BorderType.Periodic,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)

            cspace.add(a1)
            cspace.add(a2)

            if (rank == 0):
                pt = space.ContinuousPoint(5, 20)
                cspace.move(a1, pt)
                # move out of bounds
                pt = space.ContinuousPoint(-3.5, 22.2)
                cspace.move(a1, pt)

                expected = {(1, 0, 0): (1, np.array([16.5, 22.2, 0]))}
                for ob in cspace._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

            if (rank == 1):
                a3 = core.Agent(1, 0, rank)
                cspace.add(a3)

                pt = space.ContinuousPoint(12, 39)
                cspace.move(a2, pt)
                cspace.move(a3, pt)

                cspace.move(a2, space.ContinuousPoint(25, -3.1))
                cspace.move(a3, space.ContinuousPoint(21, 42))

                expected = {(2, 0, 1): (0, np.array([5, 36.9, 0])),
                            (1, 0, 1): (0, np.array([1, 2, 0]))}
                for ob in cspace._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

    def test_buffer_data_2x1_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            buffer_size = 2
            box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_cspace", bounds=box, borders=BorderType.Periodic,
                                        occupancy=OccupancyType.Multiple, buffer_size=buffer_size, comm=comm,
                                        tree_threshold=100)

            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            bottom = lb.ymin + lb.yextent
            right = lb.xmin + lb.xextent

            # e, w, ne, nw, se, sw. self is n/s neighbor.
            slices = [
                (right - buffer_size, right, lb.ymin, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (right - buffer_size, right, bottom - buffer_size, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, bottom - buffer_size, bottom, 0, 0)
            ]

            nghs = {
                0: (1, 1, 1, 1, 1, 1),
                1: (0, 0, 0, 0, 0, 0)
            }

            exp_vals = [[] for i in range(6)]
            for i, v in enumerate(nghs[rank]):
                exp_vals[v].append(slices[i])

            for ngh, box in cspace._get_buffer_data():
                self.assertTrue(box in exp_vals[ngh], msg=f'{rank}, {ngh}, {box}, {exp_vals[ngh]}')
                exp_vals[ngh].remove(box)

            for ngh in nghs[rank]:
                self.assertEqual(0, len(exp_vals[ngh]))

    def test_buffer_data_2x2_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Excl([4, 5, 6, 7, 8])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            buffer_size = 2
            box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_cspace", bounds=box, borders=BorderType.Periodic,
                                        occupancy=OccupancyType.Multiple, buffer_size=buffer_size, comm=comm,
                                        tree_threshold=100)

            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            bottom = lb.ymin + lb.yextent
            right = lb.xmin + lb.xextent

            # if rank == 3:
            #     for bd in cspace._get_buffer_data():
            #         print(bd, flush=True)
            # exp in n, s, e, w, ne, nw, se, sw order
            slices = [
                (lb.xmin, lb.xmin + lb.xextent, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + lb.xextent, bottom - buffer_size, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (right - buffer_size, right, bottom - buffer_size, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, bottom - buffer_size, bottom, 0, 0)
            ]
            nghs = {
                0: (1, 1, 2, 2, 3, 3, 3, 3),
                1: (0, 0, 3, 3, 2, 2, 2, 2),
                2: (3, 3, 0, 0, 1, 1, 1, 1),
                3: (2, 2, 1, 1, 0, 0, 0, 0)
            }

            exp_vals = [[] for i in range(4)]
            for i, v in enumerate(nghs[rank]):
                exp_vals[v].append(slices[i])

            for ngh, box in cspace._get_buffer_data():
                self.assertTrue(box in exp_vals[ngh], msg=f'{rank}, {ngh}, {box}, {exp_vals[ngh]}')
                exp_vals[ngh].remove(box)

            for ngh in nghs[rank]:
                self.assertEqual(0, len(exp_vals[ngh]))

    def test_buffer_data_4x2_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Excl([8])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            buffer_size = 2
            box = geometry.BoundingBox(xmin=0, xextent=80, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_cspace", bounds=box, borders=BorderType.Periodic,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)
            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            bottom = lb.ymin + lb.yextent
            right = lb.xmin + lb.xextent

            # if rank == 0:
            #     for bd in cspace._get_buffer_data():
            #         print(bd, flush=True)
            # exp in n, s, e, w, ne, nw, se, sw order
            slices = [
                (lb.xmin, lb.xmin + lb.xextent, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + lb.xextent, bottom - buffer_size, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (right - buffer_size, right, bottom - buffer_size, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, bottom - buffer_size, bottom, 0, 0)
            ]
            nghs = {
                0: [1, 1, 2, 6, 3, 7, 3, 7],
                1: [0, 0, 3, 7, 2, 6, 2, 6],
                2: [3, 3, 4, 0, 5, 1, 5, 1],
                3: [2, 2, 5, 1, 4, 0, 4, 0],
                4: [5, 5, 6, 2, 7, 3, 7, 3],
                5: [4, 4, 7, 3, 6, 2, 6, 2],
                6: [7, 7, 0, 4, 1, 5, 1, 5],
                7: [6, 6, 1, 5, 0, 4, 0, 4]
            }

            exp_vals = [[] for i in range(8)]
            for i, v in enumerate(nghs[rank]):
                exp_vals[v].append(slices[i])

            # TODO remove the empty lists

            for ngh, box in cspace._get_buffer_data():
                self.assertTrue(box in exp_vals[ngh], msg=f'{rank}, {ngh}, {box}, {exp_vals[ngh]}')
                exp_vals[ngh].remove(box)

            for ngh in nghs[rank]:
                self.assertEqual(0, len(exp_vals[ngh]))


class SharedGridTests(unittest.TestCase):

    long_message = True

    def test_num_agents(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)
            b1 = core.Agent(1, 1, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
            self.assertEqual('shared_grid', grid.name)

            grid.add(a1)
            grid.add(a2)
            grid.add(b1)

            pt = DPt(1, 1, 0)
            grid.move(a1, pt)
            grid.move(a2, pt)
            grid.move(b1, pt)

            self.assertEqual(3, grid.get_num_agents(pt))
            self.assertEqual(2, grid.get_num_agents(pt, 0))
            self.assertEqual(1, grid.get_num_agents(pt, agent_type=1))

    def test_ops(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
            self.assertEqual('shared_grid', grid.name)

            grid.add(a1)
            grid.add(a2)

            if (rank == 0):
                pt = space.DiscretePoint(5, 20)
                grid.move(a1, pt)
                self.assertEqual(a1, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(a1))

                agents = grid.get_agents(pt)
                expected = [a1]
                count = 0
                for i, agent in enumerate(agents):
                    self.assertEqual(expected[i], agent)
                    count += 1
                self.assertEqual(1, count)

                grid.remove(a1)
                self.assertIsNone(grid.get_agent(pt))

            if (rank == 1):
                pt = space.DiscretePoint(12, 39)
                grid.move(a2, pt)
                self.assertEqual(a2, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(a2))

                a3 = core.Agent(1, 0, rank)
                grid.add(a3)
                grid.move(a3, pt)

                agents = grid.get_agents(pt)
                expected = [a2, a3]
                count = 0
                for i, agent in enumerate(agents):
                    self.assertEqual(expected[i], agent)
                    count += 1
                self.assertEqual(2, count)

    def test_oob(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

            grid.add(a1)
            grid.add(a2)

            if (rank == 0):
                pt = space.DiscretePoint(5, 20)
                grid.move(a1, pt)
                # move out of bounds
                pt = space.DiscretePoint(12, 22)
                grid.move(a1, pt)

                expected = {(1, 0, 0): (1, np.array([12, 22, 0]))}
                for ob in grid._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

            if (rank == 1):
                a3 = core.Agent(1, 0, rank)
                grid.add(a3)

                pt = space.DiscretePoint(12, 39)
                grid.move(a2, pt)
                grid.move(a3, pt)

                grid.move(a2, space.DiscretePoint(0, 1))
                grid.move(a3, space.DiscretePoint(8, 200))

                expected = {(2, 0, 1): (0, np.array([0, 1, 0])),
                            (1, 0, 1): (0, np.array([8, 39, 0]))}
                for ob in grid._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

    def test_oob_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

            grid.add(a1)
            grid.add(a2)

            if (rank == 0):
                pt = space.DiscretePoint(5, 20)
                grid.move(a1, pt)
                # move out of bounds
                pt = space.DiscretePoint(-3, 22)
                grid.move(a1, pt)

                expected = {(1, 0, 0): (1, np.array([17, 22, 0]))}
                for ob in grid._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

            if (rank == 1):
                a3 = core.Agent(1, 0, rank)
                grid.add(a3)

                pt = space.DiscretePoint(12, 39)
                grid.move(a2, pt)
                grid.move(a3, pt)

                grid.move(a2, space.DiscretePoint(25, -3))
                grid.move(a3, space.DiscretePoint(21, 42))

                expected = {(2, 0, 1): (0, np.array([5, 37, 0])),
                            (1, 0, 1): (0, np.array([1, 2, 0]))}
                for ob in grid._get_oob():
                    exp = expected.pop(ob[0])
                    self.assertEqual(exp[0], ob[1])
                    self.assertTrue(np.array_equal(exp[1], ob[2]))
                self.assertEqual(0, len(expected))

    def test_oob_3x3(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        agents = [core.Agent(1, 0, rank), core.Agent(2, 0, rank), core.Agent(3, 0, rank),
                  core.Agent(4, 0, rank), core.Agent(5, 0, rank), core.Agent(6, 0, rank),
                  core.Agent(7, 0, rank), core.Agent(8, 0, rank)]

        box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

        # print('{}: bounds: {}'.format(rank, grid.get_local_bounds()))
        for a in agents:
            grid.add(a)

        expected = {}
        if rank == 0:
            make_move(grid, agents[0], 32, 20, 3, expected)
            make_move(grid, agents[1], 35, 45, 4, expected)
            make_move(grid, agents[2], 15, 45, 1, expected)

        elif rank == 1:
            make_move(grid, agents[0], 15, 15, 0, expected)
            make_move(grid, agents[1], 35, 10, 3, expected)
            make_move(grid, agents[2], 45, 45, 4, expected)
            make_move(grid, agents[3], 35, 82, 5, expected)
            make_move(grid, agents[4], 15, 85, 2, expected)

        elif rank == 2:
            make_move(grid, agents[0], 15, 45, 1, expected)
            make_move(grid, agents[1], 35, 45, 4, expected)
            make_move(grid, agents[2], 35, 85, 5, expected)

        elif rank == 3:
            make_move(grid, agents[0], 15, 15, 0, expected)
            make_move(grid, agents[1], 15, 45, 1, expected)
            make_move(grid, agents[2], 45, 45, 4, expected)
            make_move(grid, agents[3], 65, 45, 7, expected)
            make_move(grid, agents[4], 65, 15, 6, expected)

        elif rank == 4:
            make_move(grid, agents[0], 15, 15, 0, expected)
            make_move(grid, agents[1], 15, 45, 1, expected)
            make_move(grid, agents[2], 15, 85, 2, expected)
            make_move(grid, agents[3], 35, 10, 3, expected)
            # make_move(grid, agents[4], 45, 45, 4, expected)
            make_move(grid, agents[5], 35, 85, 5, expected)
            make_move(grid, agents[7], 65, 15, 6, expected)
            make_move(grid, agents[6], 65, 45, 7, expected)
            make_move(grid, agents[4], 65, 85, 8, expected)

        elif rank == 5:
            make_move(grid, agents[1], 15, 45, 1, expected)
            make_move(grid, agents[4], 15, 85, 2, expected)
            make_move(grid, agents[2], 45, 45, 4, expected)
            make_move(grid, agents[3], 65, 45, 7, expected)
            make_move(grid, agents[7], 65, 85, 8, expected)

        elif rank == 6:
            make_move(grid, agents[3], 35, 10, 3, expected)
            make_move(grid, agents[4], 45, 45, 4, expected)
            make_move(grid, agents[6], 65, 45, 7, expected)

        elif rank == 7:
            make_move(grid, agents[3], 35, 10, 3, expected)
            make_move(grid, agents[4], 45, 45, 4, expected)
            make_move(grid, agents[5], 35, 85, 5, expected)
            make_move(grid, agents[7], 65, 15, 6, expected)
            make_move(grid, agents[1], 65, 85, 8, expected)

        elif rank == 8:
            make_move(grid, agents[4], 45, 45, 4, expected)
            make_move(grid, agents[5], 35, 85, 5, expected)
            make_move(grid, agents[6], 65, 45, 7, expected)

        for ob in grid._get_oob():
            exp = expected.pop(ob[0])
            self.assertEqual(exp[0], ob[1], rank)
            self.assertTrue(np.array_equal(exp[1], ob[2]), rank)
        self.assertEqual(0, len(expected), rank)

    def test_buffer_data_2x1_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            buffer_size = 2
            box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                    occupancy=OccupancyType.Multiple, buffer_size=buffer_size, comm=comm)

            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            bottom = lb.ymin + lb.yextent
            right = lb.xmin + lb.xextent

            # e, w, ne, nw, se, sw. self is n/s neighbor.
            slices = [
                (right - buffer_size, right, lb.ymin, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (right - buffer_size, right, bottom - buffer_size, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, bottom - buffer_size, bottom, 0, 0)
            ]

            nghs = {
                0: (1, 1, 1, 1, 1, 1),
                1: (0, 0, 0, 0, 0, 0)
            }

            exp_vals = [[] for i in range(6)]
            for i, v in enumerate(nghs[rank]):
                exp_vals[v].append(slices[i])

            for ngh, box in grid._get_buffer_data():
                self.assertTrue(box in exp_vals[ngh], msg=f'{rank}, {ngh}, {box}, {exp_vals[ngh]}')
                exp_vals[ngh].remove(box)

            for ngh in nghs[rank]:
                self.assertEqual(0, len(exp_vals[ngh]))

    def test_buffer_data_2x2_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Excl([4, 5, 6, 7, 8])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            buffer_size = 2
            box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                    occupancy=OccupancyType.Multiple, buffer_size=buffer_size, comm=comm)
            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            bottom = lb.ymin + lb.yextent
            right = lb.xmin + lb.xextent

            # exp in n, s, e, w, ne, nw, se, sw order
            slices = [
                (lb.xmin, lb.xmin + lb.xextent, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + lb.xextent, bottom - buffer_size, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (right - buffer_size, right, bottom - buffer_size, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, bottom - buffer_size, bottom, 0, 0)
            ]
            nghs = {
                0: (1, 1, 2, 2, 3, 3, 3, 3),
                1: (0, 0, 3, 3, 2, 2, 2, 2),
                2: (3, 3, 0, 0, 1, 1, 1, 1),
                3: (2, 2, 1, 1, 0, 0, 0, 0)
            }

            exp_vals = [[] for i in range(4)]
            for i, v in enumerate(nghs[rank]):
                exp_vals[v].append(slices[i])

            for ngh, box in grid._get_buffer_data():
                self.assertTrue(box in exp_vals[ngh], msg=f'{rank}, {ngh}, {box}, {exp_vals[ngh]}')
                exp_vals[ngh].remove(box)

            for ngh in nghs[rank]:
                self.assertEqual(0, len(exp_vals[ngh]))

    def test_buffer_data_4x2_periodic(self):
        new_group = MPI.COMM_WORLD.Get_group().Excl([8])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            buffer_size = 2
            box = geometry.BoundingBox(xmin=0, xextent=80, ymin=0, yextent=120, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                    occupancy=OccupancyType.Multiple, buffer_size=buffer_size, comm=comm)
            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            bottom = lb.ymin + lb.yextent
            right = lb.xmin + lb.xextent

            if rank == 0:
                for bd in grid._get_buffer_data():
                    print(bd, flush=True)
            # exp in n, s, e, w, ne, nw, se, sw order
            slices = [
                (lb.xmin, lb.xmin + lb.xextent, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + lb.xextent, bottom - buffer_size, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, bottom, 0, 0),
                (right - buffer_size, right, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, lb.ymin, lb.ymin + buffer_size, 0, 0),
                (right - buffer_size, right, bottom - buffer_size, bottom, 0, 0),
                (lb.xmin, lb.xmin + buffer_size, bottom - buffer_size, bottom, 0, 0)
            ]
            nghs = {
                0: [1, 1, 2, 6, 3, 7, 3, 7],
                1: [0, 0, 3, 7, 2, 6, 2, 6],
                2: [3, 3, 4, 0, 5, 1, 5, 1],
                3: [2, 2, 5, 1, 4, 0, 4, 0],
                4: [5, 5, 6, 2, 7, 3, 7, 3],
                5: [4, 4, 7, 3, 6, 2, 6, 2],
                6: [7, 7, 0, 4, 1, 5, 1, 5],
                7: [6, 6, 1, 5, 0, 4, 0, 4]
            }

            exp_vals = [[] for i in range(8)]
            for i, v in enumerate(nghs[rank]):
                exp_vals[v].append(slices[i])

            # TODO remove the empty lists

            for ngh, box in grid._get_buffer_data():
                self.assertTrue(box in exp_vals[ngh], msg=f'{rank}, {ngh}, {box}, {exp_vals[ngh]}')
                exp_vals[ngh].remove(box)

            for ngh in nghs[rank]:
                self.assertEqual(0, len(exp_vals[ngh]))

    def test_buffer_data_3x3_periodic(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

        if rank == 0:
            expected = {
                8: (0, 2, 0, 2, 0, 0),
                6: (0, 2, 0, 40, 0, 0),
                7: (0, 2, 38, 40, 0, 0),
                2: (0, 30, 0, 2, 0, 0),
                1: (0, 30, 38, 40, 0, 0),
                5: (28, 30, 0, 2, 0, 0),
                3: (28, 30, 0, 40, 0, 0),
                4: (28, 30, 38, 40, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 1:
            expected = {
                6: (0, 2, 40, 42, 0, 0),
                7: (0, 2, 40, 80, 0, 0),
                8: (0, 2, 78, 80, 0, 0),
                0: (0, 30, 40, 42, 0, 0),
                2: (0, 30, 78, 80, 0, 0),
                3: (28, 30, 40, 42, 0, 0),
                4: (28, 30, 40, 80, 0, 0),
                5: (28, 30, 78, 80, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 2:
            expected = {
                7: (0, 2, 80, 82, 0, 0),
                8: (0, 2, 80, 120, 0, 0),
                6: (0, 2, 118, 120, 0, 0),
                1: (0, 30, 80, 82, 0, 0),
                0: (0, 30, 118, 120, 0, 0),
                4: (28, 30, 80, 82, 0, 0),
                5: (28, 30, 80, 120, 0, 0),
                3: (28, 30, 118, 120, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 3:
            expected = {
                2: (30, 32, 0, 2, 0, 0),
                0: (30, 32, 0, 40, 0, 0),
                1: (30, 32, 38, 40, 0, 0),
                5: (30, 60, 0, 2, 0, 0),
                4: (30, 60, 38, 40, 0, 0),
                8: (58, 60, 0, 2, 0, 0),
                6: (58, 60, 0, 40, 0, 0),
                7: (58, 60, 38, 40, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 4:
            expected = {
                0: (30, 32, 40, 42, 0, 0),
                1: (30, 32, 40, 80, 0, 0),
                2: (30, 32, 78, 80, 0, 0),
                3: (30, 60, 40, 42, 0, 0),
                5: (30, 60, 78, 80, 0, 0),
                6: (58, 60, 40, 42, 0, 0),
                7: (58, 60, 40, 80, 0, 0),
                8: (58, 60, 78, 80, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 5:
            expected = {
                1: (30, 32, 80, 82, 0, 0),
                2: (30, 32, 80, 120, 0, 0),
                0: (30, 32, 118, 120, 0, 0),
                4: (30, 60, 80, 82, 0, 0),
                3: (30, 60, 118, 120, 0, 0),
                7: (58, 60, 80, 82, 0, 0),
                8: (58, 60, 80, 120, 0, 0),
                6: (58, 60, 118, 120, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 6:
            expected = {
                5: (60, 62, 0, 2, 0, 0),
                3: (60, 62, 0, 40, 0, 0),
                4: (60, 62, 38, 40, 0, 0),
                8: (60, 90, 0, 2, 0, 0),
                7: (60, 90, 38, 40, 0, 0),
                2: (88, 90, 0, 2, 0, 0),
                0: (88, 90, 0, 40, 0, 0),
                1: (88, 90, 38, 40, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 7:
            expected = {
                3: (60, 62, 40, 42, 0, 0),
                4: (60, 62, 40, 80, 0, 0),
                5: (60, 62, 78, 80, 0, 0),
                6: (60, 90, 40, 42, 0, 0),
                8: (60, 90, 78, 80, 0, 0),
                0: (88, 90, 40, 42, 0, 0),
                1: (88, 90, 40, 80, 0, 0),
                2: (88, 90, 78, 80, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 8:
            expected = {
                4: (60, 62, 80, 82, 0, 0),
                5: (60, 62, 80, 120, 0, 0),
                3: (60, 62, 118, 120, 0, 0),
                7: (60, 90, 80, 82, 0, 0),
                6: (60, 90, 118, 120, 0, 0),
                1: (88, 90, 80, 82, 0, 0),
                2: (88, 90, 80, 120, 0, 0),
                0: (88, 90, 118, 120, 0, 0)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

    def test_buffer_data_3d(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if comm.size != 18:
            if rank == 0:
                print("3D buffer tests not run -- run with -n 18")
            return

        box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=60)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

        # print('{}: bounds: {}'.format(rank, grid.get_local_bounds()))
        if rank == 0:
            expected = {
                1: (0, 30, 0, 40, 28, 30),
                2: (0, 30, 38, 40, 0, 30),
                3: (0, 30, 38, 40, 28, 30),
                6: (28, 30, 0, 40, 0, 30),
                7: (28, 30, 0, 40, 28, 30),
                8: (28, 30, 38, 40, 0, 30),
                9: (28, 30, 38, 40, 28, 30)
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 8:
            expected = {
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
            }
            for bd in grid._get_buffer_data():
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

        elif rank == 15:
            expected = {
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
            for bd in grid._get_buffer_data():
                # print('{}: {},'.format(bd[0], bd[1]))
                exp = expected.pop(bd[0])
                self.assertEqual(exp, bd[1])
            self.assertEqual(0, len(expected))

    def test_buffer_data_3d_periodic(self):
        pass
        # TODO: Get 3D Working
        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()

        # if comm.size != 18:
        #     if rank == 0:
        #         print("3D buffer tests not run -- run with -n 18")
        #     return

        # box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=60)
        # grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
        #                         occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

        # if rank == 0:
        #     expected = {
        #         1: (0, 30, 0, 40, 28, 30),
        #         2: (0, 30, 38, 40, 0, 30),
        #         3: (0, 30, 38, 40, 28, 30),
        #         4: (0, 30, 0, 2, 0, 30),
        #         5: (0, 30, 0, 2, 28, 30),
        #         6: (28, 30, 0, 40, 0, 30),
        #         7: (28, 30, 0, 40, 28, 30),
        #         8: (28, 30, 38, 40, 0, 30),
        #         9: (28, 30, 38, 40, 28, 30),
        #         10: (28, 30, 0, 2, 0, 30),
        #         11: (28, 30, 0, 2, 28, 30),
        #         12: (0, 2, 0, 40, 0, 30),
        #         13: (0, 2, 0, 40, 28, 30),
        #         14: (0, 2, 38, 40, 0, 30),
        #         15: (0, 2, 38, 40, 28, 30),
        #         16: (0, 2, 0, 2, 0, 30),
        #         17: (0, 2, 0, 2, 28, 30)
        #     }
        #     for bd in grid._get_buffer_data():
        #         exp = expected.pop(bd[0])
        #         self.assertEqual(exp, bd[1])
        #     self.assertEqual(0, len(expected))

        # elif rank == 15:
        #     expected = {
        #         0: (88, 90, 40, 42, 30, 32),
        #         1: (88, 90, 40, 42, 30, 60),
        #         2: (88, 90, 40, 80, 30, 32),
        #         3: (88, 90, 40, 80, 30, 60),
        #         4: (88, 90, 78, 80, 30, 32),
        #         5: (88, 90, 78, 80, 30, 60),
        #         6: (60, 62, 40, 42, 30, 32),
        #         7: (60, 62, 40, 42, 30, 60),
        #         8: (60, 62, 40, 80, 30, 32),
        #         9: (60, 62, 40, 80, 30, 60),
        #         10: (60, 62, 78, 80, 30, 32),
        #         11: (60, 62, 78, 80, 30, 60),
        #         12: (60, 90, 40, 42, 30, 32),
        #         13: (60, 90, 40, 42, 30, 60),
        #         14: (60, 90, 40, 80, 30, 32),
        #         16: (60, 90, 78, 80, 30, 32),
        #         17: (60, 90, 78, 80, 30, 60)
        #     }
        #     for bd in grid._get_buffer_data():
        #         exp = expected.pop(bd[0])
        #         self.assertEqual(exp, bd[1])
        #     self.assertEqual(0, len(expected))


class EAgent(core.Agent):

    def __init__(self, id, agent_type, rank, energy):
        super().__init__(id=id, type=agent_type, rank=rank)
        self.energy = energy
        self.restored = False

    def save(self):
        return (self.uid, self.energy)

    def update(self, data):
        # update
        self.energy = data


def create_agent(agent_data):
    # agent_data: [aid_tuple, energy]
    uid = agent_data[0]
    return EAgent(uid[0], uid[1], uid[2], agent_data[1])


class SharedContextTests1(unittest.TestCase):

    long_message = True

    # tests same named proj in context throws exception
    def test_duplicate_projection(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

            context = ctx.SharedContext(comm)
            context.add_projection(grid)
            with self.assertRaises(ValueError) as _:
                context.add_projection(grid)

    # tests adding / removing from context
    def test_add_remove1(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 0, rank)
            a2 = core.Agent(2, 0, rank)
            agents = [a1, a2]

            context = ctx.SharedContext(comm)
            context.add(a1)
            context.add(a2)

            count = 0
            for i, ag in enumerate(context.agents()):
                self.assertEqual(agents[i], ag)
                self.assertEqual(agents[i].id, ag.id)
                count += 1

            self.assertEqual(2, count)

            context.remove(a1)
            count = 0
            for ag in context.agents():
                self.assertEqual(a2, ag)
                count += 1
            self.assertEqual(1, count)

    # tests context.count
    def test_counts(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            a1 = core.Agent(1, 1, rank)
            a2 = core.Agent(2, 0, rank)
            a3 = core.Agent(2, 1, rank)

            context = ctx.SharedContext(comm)
            context.add(a1)
            context.add(a3)
            context.add(a2)

            counts = context.size()
            self.assertEqual(3, counts[-1])

            counts = context.size([0, 1])
            self.assertEqual(2, len(counts))
            self.assertEqual(2, counts[1])
            self.assertEqual(1, counts[0])

            counts = context.size([1])
            self.assertEqual(1, len(counts))
            self.assertEqual(2, counts[1])

            context.remove(a1)
            counts = context.size([0, 1])
            self.assertEqual(2, len(counts))
            self.assertEqual(1, counts[1])
            self.assertEqual(1, counts[0])

    # tests context.contains_type
    def test_has_type(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            context = ctx.SharedContext(comm)
            self.assertFalse(context.contains_type(0))
            a = core.Agent(0, 0, 0)
            context.add(a)
            self.assertTrue(context.contains_type(0))
            context.remove(a)
            self.assertFalse(context.contains_type(0))

    # tests context.agents()
    def test_get_agents(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            context = ctx.SharedContext(comm)
            for i in range(10):
                t = 0 if i % 2 == 0 else 1
                a = core.Agent(i, t, 0)
                context.add(a)

            exp = []
            for i in range(10):
                t = 0 if i % 2 == 0 else 1
                exp.append((i, t, 0))

            actual = []
            for a in context.agents():
                actual.append(a.uid)
            self.assertEqual(exp, actual)

            exp.clear()
            actual.clear()
            for i in range(10):
                if i % 2 == 0:
                    exp.append((i, 0, 0))

            for a in context.agents(agent_type=0):
                actual.append(a.uid)
            self.assertEqual(exp, actual)

            exp.clear()
            actual.clear()
            for i in range(10):
                if i % 2 != 0:
                    exp.append((i, 1, 0))

            for a in context.agents(agent_type=1):
                actual.append(a.uid)
            self.assertEqual(exp, actual)

            exp = [(0, 0, 0), (2, 0, 0), (4, 0, 0), (6, 0, 0)]
            actual.clear()
            for a in context.agents(agent_type=0, count=4):
                actual.append(a.uid)
            self.assertEqual(4, len(actual))
            self.assertEqual(exp, actual)

            exp = [(0, 0, 0), (1, 1, 0), (2, 0, 0)]
            actual.clear()
            for a in context.agents(count=3):
                actual.append(a.uid)
            self.assertEqual(3, len(actual))
            self.assertEqual(exp, actual)

            random.init(42)
            exp = [(5, 1, 0), (6, 0, 0), (0, 0, 0), (7, 1, 0), (3, 1, 0), (2, 0, 0),
                   (4, 0, 0), (9, 1, 0), (1, 1, 0), (8, 0, 0)]
            actual.clear()
            for a in context.agents(shuffle=True):
                actual.append(a.uid)
            self.assertEqual(exp, actual)

            random.init(42)
            exp = [(5, 1, 0), (6, 0, 0)]
            actual.clear()
            for a in context.agents(shuffle=True, count=2):
                actual.append(a.uid)
            self.assertEqual(exp, actual)

    # tests adding / removing from context, adds / removes from projection
    def test_add_remove2(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

            context = ctx.SharedContext(comm)
            context.add_projection(grid)

            # test that adding to context
            # adds to projection
            if rank == 0:
                a1 = core.Agent(1, 0, rank)
                context.add(a1)
                pt = space.DiscretePoint(0, 5)
                grid.move(a1, pt)
                self.assertEqual(a1, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(a1))

                context.remove(a1)
                self.assertIsNone(grid.get_agent(pt))

            else:
                a2 = core.Agent(2, 0, rank)
                context.add(a2)
                pt = space.DiscretePoint(12, 30)
                grid.move(a2, pt)
                self.assertEqual(a2, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(a2))
                count = 0
                for agent in context.agents():
                    count += 1
                    self.assertEqual(a2, agent)
                self.assertEqual(1, count)

    # tests context synchronization in 2D 2 rank
    # adds agents moves them oob in a grid and
    # tests if moved to correct location
    def test_synch(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

            context = ctx.SharedContext(comm)
            context.add_projection(grid)

            # test that adding to context
            # adds to projection
            if rank == 0:
                a1 = EAgent(1, 0, rank, 12)
                context.add(a1)
                # oob
                pt = space.DiscretePoint(12, 5)
                grid.move(a1, pt)

            else:
                a2 = EAgent(2, 0, rank, 3)
                a3 = EAgent(3, 0, rank, 2)
                context.add(a2)
                context.add(a3)
                # oob
                pt = space.DiscretePoint(5, 30)
                grid.move(a2, pt)
                grid.move(a3, space.DiscretePoint(3, 20))

            context.synchronize(create_agent)

            if rank == 0:
                # should now have a2 and a3
                self.assertEqual(2, len(context._agent_manager._local_agents))
                # a2
                agent = context._agent_manager._local_agents[(2, 0, 1)]
                self.assertEqual((2, 0, 1), agent.uid)
                self.assertEqual(3, agent.energy)
                pt = space.DiscretePoint(5, 30)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

                # a3
                agent = context._agent_manager._local_agents[(3, 0, 1)]
                self.assertEqual((3, 0, 1), agent.uid)
                self.assertEqual(2, agent.energy)
                pt = space.DiscretePoint(3, 20)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

            else:
                # should now have a1
                self.assertEqual(1, len(context._agent_manager._local_agents))
                agent = context._agent_manager._local_agents[(1, 0, 0)]
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(12, agent.energy)
                pt = space.DiscretePoint(12, 5)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

            if rank == 0:
                pt = space.DiscretePoint(5, 30)
                agent = grid.get_agent(pt)
                grid.move(agent, space.DiscretePoint(12, 38))
                agent.energy = -10

            context.synchronize(create_agent)

            if rank == 0:
                self.assertEqual(1, len(context._agent_manager._local_agents))
                # a3
                agent = context._agent_manager._local_agents[(3, 0, 1)]
                self.assertEqual((3, 0, 1), agent.uid)
                self.assertEqual(2, agent.energy)
                pt = space.DiscretePoint(3, 20)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

            if rank == 1:
                # a2 now back in 1
                self.assertEqual(2, len(context._agent_manager._local_agents))
                agent = context._agent_manager._local_agents[(2, 0, 1)]
                self.assertEqual((2, 0, 1), agent.uid)
                self.assertEqual(-10, agent.energy)
                pt = space.DiscretePoint(12, 38)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

                agent = context._agent_manager._local_agents[(1, 0, 0)]
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(12, agent.energy)
                pt = space.DiscretePoint(12, 5)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

    # Tests that buffer is filled on synchronization in 2D, 2 rank world
    def test_buffer(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)

            context = ctx.SharedContext(comm)
            context.add_projection(grid)

            if rank == 0:
                a01 = EAgent(1, 0, rank, 1)
                a02 = EAgent(2, 0, rank, 2)
                a03 = EAgent(3, 0, rank, 3)
                context.add(a01)
                context.add(a02)
                context.add(a03)

                pt = space.DiscretePoint(8, 20)
                grid.move(a01, pt)
                pt = space.DiscretePoint(9, 15)
                grid.move(a02, pt)
                pt = space.DiscretePoint(9, 15)
                grid.move(a03, pt)

            if rank == 1:
                a11 = EAgent(1, 0, rank, 11)
                a12 = EAgent(2, 0, rank, 12)
                a13 = EAgent(3, 0, rank, 13)

                context.add(a11)
                context.add(a12)
                context.add(a13)

                pt = space.DiscretePoint(10, 20)
                grid.move(a11, pt)
                pt = space.DiscretePoint(11, 15)
                grid.move(a12, pt)
                pt = space.DiscretePoint(15, 15)
                grid.move(a13, pt)

            grid._pre_synch_ghosts(context._agent_manager)
            self.assertEqual(0, len(context._agent_manager._ghost_agents))
            context.synchronize(create_agent, False)
            grid._synch_ghosts(context._agent_manager, create_agent)

            if rank == 0:
                pt = space.DiscretePoint(10, 20)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((1, 0, 1), agent.uid)
                self.assertEqual(11, agent.energy)

                pt = space.DiscretePoint(11, 15)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((2, 0, 1), agent.uid)
                self.assertEqual(12, agent.energy)

                self.assertEqual(2, len(context._agent_manager._ghost_agents))
                self.assertEqual(3, len(context._agent_manager._local_agents))

                # moves these agents out of the buffer zone
                pt = space.DiscretePoint(9, 15)
                for a in grid.get_agents(pt):
                    grid.move(a, space.DiscretePoint(5, 5))

            if rank == 1:
                pt = space.DiscretePoint(8, 20)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(1, agent.energy)

                pt = space.DiscretePoint(9, 15)
                expected = {(2, 0, 0): 2, (3, 0, 0): 3}
                for a in grid.get_agents(pt):
                    energy = expected.pop(a.uid)
                    self.assertEqual(energy, a.energy)
                self.assertEqual(0, len(expected))

                self.assertEqual(3, len(context._agent_manager._ghost_agents))
                self.assertEqual(3, len(context._agent_manager._local_agents))

            grid._pre_synch_ghosts(context._agent_manager)
            grid._synch_ghosts(context._agent_manager, create_agent)

            if rank == 1:
                pt = space.DiscretePoint(8, 20)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(1, agent.energy)

                # nothing at 9, 15 now
                pt = space.DiscretePoint(9, 15)
                self.assertIsNone(grid.get_agent(pt))

    def test_buffer_3x3(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = {}
        for a in [EAgent(1, 0, rank, 1), EAgent(2, 0, rank, 2), EAgent(3, 0, rank, 3),
                  EAgent(4, 0, rank, 4), EAgent(5, 0, rank, 5), EAgent(6, 0, rank, 6),
                  EAgent(7, 0, rank, 7), EAgent(8, 0, rank, 8)]:

            agents[a.uid] = a
            context.add(a)

        box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=1, comm=comm)
        context.add_projection(grid)

        data = {"0S": ((1, 0, 0), mp(15, 39)),
                "0SE": ((2, 0, 0), mp(29, 39)),
                "0E": ((3, 0, 0), mp(29, 20)),

                "1N": ((1, 0, 1), mp(15, 40)),
                "1S": ((2, 0, 1), mp(15, 79)),
                "1NE": ((3, 0, 1), mp(29, 40)),
                "1E": ((4, 0, 1), mp(29, 45)),
                "1SE": ((5, 0, 1), mp(29, 79)),

                '2N': ((1, 0, 2), mp(15, 80)),
                '2NE': ((2, 0, 2), mp(29, 80)),
                '2E': ((3, 0, 2), mp(29, 85)),

                '3W': ((1, 0, 3), mp(30, 15)),
                '3SW': ((2, 0, 3), mp(30, 39)),
                '3S': ((3, 0, 3), mp(35, 39)),
                '3SE': ((4, 0, 3), mp(59, 39)),
                '3E': ((5, 0, 3), mp(59, 15)),

                '4NW': ((1, 0, 4), mp(30, 40)),
                '4W': ((2, 0, 4), mp(30, 45)),
                '4SW': ((3, 0, 4), mp(30, 79)),
                '4S': ((4, 0, 4), mp(35, 79)),
                '4SE': ((5, 0, 4), mp(59, 79)),
                '4E': ((6, 0, 4), mp(59, 45)),
                '4NE': ((7, 0, 4), mp(59, 40)),
                '4N': ((8, 0, 4), mp(35, 40)),

                '5NW': ((1, 0, 5), mp(30, 80)),
                '5W': ((3, 0, 5), mp(30, 85)),
                '5E': ((4, 0, 5), mp(59, 85)),
                '5NE': ((5, 0, 5), mp(59, 80)),
                '5N': ((6, 0, 5), mp(35, 80)),

                '6W': ((1, 0, 6), mp(60, 15)),
                '6SW': ((2, 0, 6), mp(60, 39)),
                '6S': ((3, 0, 6), mp(65, 39)),

                '7N': ((1, 0, 7), mp(65, 40)),
                '7NW': ((3, 0, 7), mp(60, 40)),
                '7W': ((4, 0, 7), mp(60, 45)),
                '7SW': ((5, 0, 7), mp(60, 79)),
                '7S': ((6, 0, 7), mp(65, 79)),

                '8N': ((1, 0, 8), mp(65, 80)),
                '8NW': ((3, 0, 8), mp(60, 80)),
                '8W': ((4, 0, 8), mp(60, 85))}

        for k, v in data.items():
            if k.startswith(str(rank)):
                grid.move(agents[v[0]], v[1])

        grid._pre_synch_ghosts(context._agent_manager)
        grid._synch_ghosts(context._agent_manager, create_agent)

        exp = [
            ['1N', '3W', '4NW'],
            ['0S', '2N', '5NW', '4W', '3SW'],
            ['1S', '5W', '4SW'],
            ['0E', '1NE', '4N', '7NW', '6W'],
            ['0SE', '1E', '2NE', '5N', '8NW', '7W', '6SW', '3S'],
            ['1SE', '2E', '4S', '7SW', '8W'],
            ['3E', '4NE', '7N'],
            ['6S', '3SE', '4E', '5NE', '8N'],
            ['7S', '4SE', '5E']
        ]

        for e in exp[rank]:
            uid, pt = data[e]
            agent = grid.get_agent(pt)
            self.assertIsNotNone(agent, (e, rank))
            self.assertEqual(uid, agent.uid, (e, rank))


# Same tests as above but using continuous space
class SharedContextTests2(unittest.TestCase):

    long_message = True

    def test_add_remove(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_cspace", bounds=box, borders=BorderType.Sticky,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm, tree_threshold=100)

            context = ctx.SharedContext(comm)
            context.add_projection(cspace)

            # test that adding to context
            # adds to projection
            if rank == 0:
                a1 = core.Agent(1, 0, rank)
                context.add(a1)
                pt = space.ContinuousPoint(0, 5)
                cspace.move(a1, pt)
                self.assertEqual(a1, cspace.get_agent(pt))
                self.assertEqual(pt, cspace.get_location(a1))

                context.remove(a1)
                self.assertIsNone(cspace.get_agent(pt))

            else:
                a2 = core.Agent(2, 0, rank)
                context.add(a2)
                pt = space.ContinuousPoint(12, 30)
                cspace.move(a2, pt)
                self.assertEqual(a2, cspace.get_agent(pt))
                self.assertEqual(pt, cspace.get_location(a2))
                count = 0
                for agent in context.agents():
                    count += 1
                    self.assertEqual(a2, agent)
                self.assertEqual(1, count)

    def test_synch(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedCSpace("shared_cspace", bounds=box, borders=BorderType.Sticky,
                                      occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                      tree_threshold=100)

            context = ctx.SharedContext(comm)
            context.add_projection(grid)

            # test that adding to context
            # adds to projection
            if rank == 0:
                a1 = EAgent(1, 0, rank, 12)
                context.add(a1)
                # oob
                pt = space.ContinuousPoint(12, 5)
                grid.move(a1, pt)

            else:
                a2 = EAgent(2, 0, rank, 3)
                a3 = EAgent(3, 0, rank, 2)
                context.add(a2)
                context.add(a3)
                # oob
                pt = space.ContinuousPoint(5, 30)
                grid.move(a2, pt)
                grid.move(a3, space.ContinuousPoint(3, 20))

            context.synchronize(create_agent, sync_ghosts=False)

            if rank == 0:
                # should now have a2 and a3
                self.assertEqual(2, len(context._agent_manager._local_agents))
                # a2
                agent = context._agent_manager._local_agents[(2, 0, 1)]
                self.assertEqual((2, 0, 1), agent.uid)
                self.assertEqual(3, agent.energy)
                pt = space.ContinuousPoint(5, 30)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

                # a3
                agent = context._agent_manager._local_agents[(3, 0, 1)]
                self.assertEqual((3, 0, 1), agent.uid)
                self.assertEqual(2, agent.energy)
                pt = space.ContinuousPoint(3, 20)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

            else:
                # should now have a1
                self.assertEqual(1, len(context._agent_manager._local_agents))
                agent = context._agent_manager._local_agents[(1, 0, 0)]
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(12, agent.energy)
                pt = space.ContinuousPoint(12, 5)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

            if rank == 0:
                pt = space.ContinuousPoint(5, 30)
                agent = grid.get_agent(pt)
                grid.move(agent, space.ContinuousPoint(12, 38))
                agent.energy = -10

            context.synchronize(create_agent, sync_ghosts=False)

            if rank == 0:
                self.assertEqual(1, len(context._agent_manager._local_agents))
                # a3
                agent = context._agent_manager._local_agents[(3, 0, 1)]
                self.assertEqual((3, 0, 1), agent.uid)
                self.assertEqual(2, agent.energy)
                pt = space.ContinuousPoint(3, 20)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

            if rank == 1:
                # a2 now back in 1
                self.assertEqual(2, len(context._agent_manager._local_agents))
                agent = context._agent_manager._local_agents[(2, 0, 1)]
                self.assertEqual((2, 0, 1), agent.uid)
                self.assertEqual(-10, agent.energy)
                pt = space.ContinuousPoint(12, 38)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

                agent = context._agent_manager._local_agents[(1, 0, 0)]
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(12, agent.energy)
                pt = space.ContinuousPoint(12, 5)
                self.assertEqual(agent, grid.get_agent(pt))
                self.assertEqual(pt, grid.get_location(agent))

    def test_buffer(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()

            box = geometry.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
            grid = space.SharedCSpace("shared_grid", bounds=box, borders=BorderType.Sticky,
                                      occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                      tree_threshold=100)

            context = ctx.SharedContext(comm)
            context.add_projection(grid)

            if rank == 0:
                a01 = EAgent(1, 0, rank, 1)
                a02 = EAgent(2, 0, rank, 2)
                a03 = EAgent(3, 0, rank, 3)
                context.add(a01)
                context.add(a02)
                context.add(a03)

                pt = space.ContinuousPoint(8, 20)
                grid.move(a01, pt)
                pt = space.ContinuousPoint(9, 15)
                grid.move(a02, pt)
                pt = space.ContinuousPoint(9, 15)
                grid.move(a03, pt)

            if rank == 1:
                a11 = EAgent(1, 0, rank, 11)
                a12 = EAgent(2, 0, rank, 12)
                a13 = EAgent(3, 0, rank, 13)

                context.add(a11)
                context.add(a12)
                context.add(a13)

                pt = space.ContinuousPoint(10, 20)
                grid.move(a11, pt)
                pt = space.ContinuousPoint(11, 15)
                grid.move(a12, pt)
                pt = space.ContinuousPoint(15, 15)
                grid.move(a13, pt)

            context.synchronize(create_agent)

            if rank == 0:
                pt = space.ContinuousPoint(10, 20)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((1, 0, 1), agent.uid)
                self.assertEqual(11, agent.energy)

                pt = space.ContinuousPoint(11, 15)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((2, 0, 1), agent.uid)
                self.assertEqual(12, agent.energy)

                self.assertEqual(3, len(context._agent_manager._local_agents))

                # moves these agents out of the buffer zone
                pt = space.ContinuousPoint(9, 15)
                for a in grid.get_agents(pt):
                    grid.move(a, space.ContinuousPoint(5, 5))

            if rank == 1:
                pt = space.ContinuousPoint(8, 20)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(1, agent.energy)

                pt = space.ContinuousPoint(9, 15)
                expected = {(2, 0, 0): 2, (3, 0, 0): 3}
                for a in grid.get_agents(pt):
                    energy = expected.pop(a.uid)
                    self.assertEqual(energy, a.energy)
                self.assertEqual(0, len(expected))

                self.assertEqual(3, len(context._agent_manager._local_agents))

            grid._pre_synch_ghosts(context._agent_manager)
            grid._synch_ghosts(context._agent_manager, create_agent)

            if rank == 1:
                pt = space.ContinuousPoint(8, 20)
                agent = grid.get_agent(pt)
                self.assertIsNotNone(agent)
                self.assertEqual((1, 0, 0), agent.uid)
                self.assertEqual(1, agent.energy)

                # nothing at 9, 15 now
                pt = space.ContinuousPoint(9, 15)
                self.assertIsNone(grid.get_agent(pt))

    def test_buffer_3x3(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = {}
        for a in [EAgent(1, 0, rank, 1), EAgent(2, 0, rank, 2), EAgent(3, 0, rank, 3),
                  EAgent(4, 0, rank, 4), EAgent(5, 0, rank, 5), EAgent(6, 0, rank, 6),
                  EAgent(7, 0, rank, 7), EAgent(8, 0, rank, 8)]:

            agents[a.uid] = a
            context.add(a)

        box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        grid = space.SharedCSpace("shared_grid", bounds=box, borders=BorderType.Sticky,
                                  occupancy=OccupancyType.Multiple, buffer_size=1, comm=comm,
                                  tree_threshold=100)
        context.add_projection(grid)

        data = {"0S": ((1, 0, 0), cp(15, 39)),
                "0SE": ((2, 0, 0), cp(29, 39)),
                "0E": ((3, 0, 0), cp(29, 20)),

                "1N": ((1, 0, 1), cp(15, 40)),
                "1S": ((2, 0, 1), cp(15, 79)),
                "1NE": ((3, 0, 1), cp(29, 40)),
                "1E": ((4, 0, 1), cp(29, 45)),
                "1SE": ((5, 0, 1), cp(29, 79)),

                '2N': ((1, 0, 2), cp(15, 80)),
                '2NE': ((2, 0, 2), cp(29, 80)),
                '2E': ((3, 0, 2), cp(29, 85)),

                '3W': ((1, 0, 3), cp(30, 15)),
                '3SW': ((2, 0, 3), cp(30, 39)),
                '3S': ((3, 0, 3), cp(35, 39)),
                '3SE': ((4, 0, 3), cp(59, 39)),
                '3E': ((5, 0, 3), cp(59, 15)),

                '4NW': ((1, 0, 4), cp(30, 40)),
                '4W': ((2, 0, 4), cp(30, 45)),
                '4SW': ((3, 0, 4), cp(30, 79)),
                '4S': ((4, 0, 4), cp(35, 79)),
                '4SE': ((5, 0, 4), cp(59, 79)),
                '4E': ((6, 0, 4), cp(59, 45)),
                '4NE': ((7, 0, 4), cp(59, 40)),
                '4N': ((8, 0, 4), cp(35, 40)),

                '5NW': ((1, 0, 5), cp(30, 80)),
                '5W': ((3, 0, 5), cp(30, 85)),
                '5E': ((4, 0, 5), cp(59, 85)),
                '5NE': ((5, 0, 5), cp(59, 80)),
                '5N': ((6, 0, 5), cp(35, 80)),

                '6W': ((1, 0, 6), cp(60, 15)),
                '6SW': ((2, 0, 6), cp(60, 39)),
                '6S': ((3, 0, 6), cp(65, 39)),

                '7N': ((1, 0, 7), cp(65, 40)),
                '7NW': ((3, 0, 7), cp(60, 40)),
                '7W': ((4, 0, 7), cp(60, 45)),
                '7SW': ((5, 0, 7), cp(60, 79)),
                '7S': ((6, 0, 7), cp(65, 79)),

                '8N': ((1, 0, 8), cp(65, 80)),
                '8NW': ((3, 0, 8), cp(60, 80)),
                '8W': ((4, 0, 8), cp(60, 85))}

        for k, v in data.items():
            if k.startswith(str(rank)):
                grid.move(agents[v[0]], v[1])

        grid._pre_synch_ghosts(context._agent_manager)
        grid._synch_ghosts(context._agent_manager, create_agent)

        exp = [
            ['1N', '3W', '4NW'],
            ['0S', '2N', '5NW', '4W', '3SW'],
            ['1S', '5W', '4SW'],
            ['0E', '1NE', '4N', '7NW', '6W'],
            ['0SE', '1E', '2NE', '5N', '8NW', '7W', '6SW', '3S'],
            ['1SE', '2E', '4S', '7SW', '8W'],
            ['3E', '4NE', '7N'],
            ['6S', '3SE', '4E', '5NE', '8N'],
            ['7S', '4SE', '5E']
        ]

        for e in exp[rank]:
            uid, pt = data[e]
            agent = grid.get_agent(pt)
            self.assertIsNotNone(agent, (e, rank))
            self.assertEqual(uid, agent.uid, (e, rank))


def get_random_pts(box):
    x = random.default_rng.uniform(box.xmin, box.xmin + box.xextent)
    y = random.default_rng.uniform(box.ymin, box.ymin + box.yextent)
    return (CPt(x, y), DPt(math.floor(x), math.floor(y)))


class TempAgent(core.Agent):

    def __init__(self, uid):
        super().__init__(id=uid[0], type=uid[1], rank=uid[2])


# 9 ranks grid and cspace buffer and movement sync
class SharedContextTests3(unittest.TestCase):

    long_message = True

    # Idea here is move agents into buffered areas
    # and record their locations and ids. That
    # data is sent to rank where the agents are
    # also sent via synch, and we use that
    # as expected info.
    def test_multi_proj(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = []
        for i in range(100):
            a = EAgent(i, 0, rank, 1)
            agents.append(a)
            context.add(a)

        box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                    tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
        context.add_projection(cspace)
        context.add_projection(grid)

        g_moved = [[] for i in range(grid._cart_comm.size)]
        c_moved = [[] for i in range(grid._cart_comm.size)]

        gm = []
        cm = []

        if rank == 0:
            b = grid.get_local_bounds()
            box = geometry.BoundingBox(b.xmin + b.xextent, 2, b.ymin, b.yextent, 0, 0)
            for i, a in enumerate(agents[0:30]):
                cp, gp = get_random_pts(box)
                gp = grid.move(a, gp)
                cp = cspace.move(a, cp)
                gm.append(gp)
                cm.append(cp)
                g_moved[3].append((gp.coordinates, a.uid))
                c_moved[3].append((cp.coordinates, a.uid))

        if rank == 4:
            b = grid.get_local_bounds()
            box = geometry.BoundingBox(b.xmin - 4, 2, b.ymin, b.yextent, 0, 0)
            for i, a in enumerate(agents[0:4]):
                cp, gp = get_random_pts(box)
                gp = grid.move(a, gp)
                cp = cspace.move(a, cp)
                gm.append(gp)
                cm.append(cp)
                g_moved[1].append((gp.coordinates, a.uid))
                c_moved[1].append((cp.coordinates, a.uid))

            box = geometry.BoundingBox(b.xmin, b.xextent, b.ymin + b.yextent, 2, 0, 0)
            for i, a in enumerate(agents[4:15]):
                cp, gp = get_random_pts(box)
                gp = grid.move(a, gp)
                cp = cspace.move(a, cp)
                gm.append(gp)
                self.assertIsNotNone(grid.get_location(a))
                cm.append(cp)
                g_moved[5].append((gp.coordinates, a.uid))
                c_moved[5].append((cp.coordinates, a.uid))

        if rank == 7:
            b = grid.get_local_bounds()
            pts = [CPt(60.2, 44.3, 0), CPt(61.5, 55.6, 0), CPt(60.1, 40.2, 0), CPt(89.1, 41.5)]
            for i, a in enumerate(agents[0:4]):
                pt = pts[i]
                cspace.move(a, pt)
                grid.move(a, DPt(math.floor(pt.x), math.floor(pt.y), 0))

        context.synchronize(create_agent)

        recv_g_moved = grid._cart_comm.alltoall(g_moved)
        recv_c_moved = cspace._cart_comm.alltoall(c_moved)

        for lst in recv_g_moved:
            dp = DPt(0, 0, 0)
            for pt, uid in lst:
                dp._reset_from_array(pt)
                a = TempAgent(uid)
                gp = grid.get_location(a)
                self.assertEqual(dp, gp, msg='rank: {}, agent: {}, pt: {}'.format(rank, a, pt))

        for lst in recv_c_moved:
            dp = CPt(0, 0, 0)
            for pt, uid in lst:
                dp._reset_from_array(pt)
                a = TempAgent(uid)
                gp = cspace.get_location(a)
                self.assertEqual(dp, gp)

        if rank == 4:
            # agents buffered from 7
            pts = [CPt(60.2, 44.3, 0), CPt(61.5, 55.6, 0), CPt(60.1, 40.2, 0)]
            for i, pt in enumerate(pts):
                dp = DPt(math.floor(pt.x), math.floor(pt.y), 0)
                a = cspace.get_agent(pt)
                self.assertIsNotNone(a)
                self.assertEqual((i, 0, 7), a.uid)
                a = grid.get_agent(dp)
                self.assertIsNotNone(a)
                self.assertEqual((i, 0, 7), a.uid)

        if rank == 3 or rank == 6 or rank == 7:
            # buffered from 7
            pts = [CPt(60.1, 40.2, 0)]
            for pt in pts:
                dp = DPt(math.floor(pt.x), math.floor(pt.y), 0)
                a = cspace.get_agent(pt)
                self.assertIsNotNone(a)
                self.assertEqual((2, 0, 7), a.uid)
                a = grid.get_agent(dp)
                self.assertIsNotNone(a)
                self.assertEqual((2, 0, 7), a.uid)

        if rank == 7:
            # buffered from 7
            pts = [CPt(89.1, 41.5)]
            for pt in pts:
                dp = DPt(math.floor(pt.x), math.floor(pt.y), 0)
                a = cspace.get_agent(pt)
                self.assertIsNotNone(a, msg="{}".format(rank))
                self.assertEqual((3, 0, 7), a.uid)
                a = grid.get_agent(dp)
                self.assertIsNotNone(a)
                self.assertEqual((3, 0, 7), a.uid)

    # def test_duplicate_requested(self):
    #     new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1, 2, 3])
    #     comm = MPI.COMM_WORLD.Create_group(new_group)

    #     if comm != MPI.COMM_NULL:
    #         rank = comm.Get_rank()
    #         context = ctx.SharedContext(comm)
    #         agents = [EAgent(x, 0, rank, x) for x in range(10)]
    #         for a in agents:
    #             context.add(a)
    #             self.assertEqual(rank, a.local_rank)

    #         requests = []
    #         if rank == 0:
    #             requests.append(((1, 0, 1), 1))
    #             requests.append(((1, 0, 2), 2))
    #             requests.append(((2, 0, 1), 1))

    #         ghosts = context.request_agents(requests, create_agent)
    #         self.assertEqual(3, len(ghosts))

    #         requests = []
    #         if rank == 0:
    #             requests.append(((1, 0, 1), 1))
    #             requests.append(((1, 0, 2), 2))
    #             requests.append(((2, 0, 1), 1))

    #         ghosts = context.request_agents(requests, create_agent)
    #         self.assertEqual(0, len(ghosts))

    def test_requested(self):
        # 4 rank comm
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1, 2, 3])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            context = ctx.SharedContext(comm)
            agents = [EAgent(x, 0, rank, x) for x in range(10)]
            for a in agents:
                context.add(a)
                self.assertEqual(rank, a.local_rank)

            requests = []
            if rank == 0:
                requests.append(((1, 0, 1), 1))
                requests.append(((1, 0, 2), 2))
                requests.append(((2, 0, 1), 1))
            elif rank == 3:
                requests.append(((1, 0, 0), 0))
                requests.append(((4, 0, 2), 2))
                requests.append(((2, 0, 1), 1))

            context.request_agents(requests, create_agent)

            manager = context._agent_manager

            if rank == 0:
                ga1 = manager._ghost_agents[(1, 0, 1)]
                self.assertEqual(3, len(manager._ghost_agents))
                self.assertEqual((1, 0, 1), ga1.agent.uid)
                self.assertEqual(0, ga1.ref_count)
                self.assertEqual(1, ga1.agent.local_rank)

                ga2 = manager._ghost_agents[(1, 0, 2)]
                self.assertEqual((1, 0, 2), ga2.agent.uid)
                self.assertEqual(0, ga2.ref_count)
                self.assertEqual(2, ga2.agent.local_rank)

                ga3 = manager._ghost_agents[(2, 0, 1)]
                self.assertEqual((2, 0, 1), ga3.agent.uid)
                self.assertEqual(0, ga3.ref_count)
                self.assertEqual(1, ga3.agent.local_rank)

                self.assertEqual(1, len(manager._ghosted_agents))
                ghosted1 = manager._ghosted_agents.get((1, 0, 0))
                self.assertIsNotNone(ghosted1)
                self.assertEqual((1, 0, 0), ghosted1.agent.uid)
                self.assertEqual(1, len(ghosted1.ghost_ranks))
                self.assertTrue(3 in ghosted1.ghost_ranks)

            elif rank == 2:
                self.assertEqual(0, len(manager._ghost_agents))

                self.assertEqual(2, len(manager._ghosted_agents))
                ghosted1 = manager._ghosted_agents.get((1, 0, 2))
                self.assertIsNotNone(ghosted1)
                self.assertEqual((1, 0, 2), ghosted1.agent.uid)
                self.assertEqual(1, len(ghosted1.ghost_ranks))
                self.assertTrue(0 in ghosted1.ghost_ranks)

                ghosted2 = manager._ghosted_agents.get((4, 0, 2))
                self.assertIsNotNone(ghosted2)
                self.assertEqual((4, 0, 2), ghosted2.agent.uid)
                self.assertEqual(1, len(ghosted2.ghost_ranks))
                self.assertTrue(3 in ghosted2.ghost_ranks)

            elif rank == 3:
                self.assertEqual(3, len(manager._ghost_agents))

                ga1 = manager._ghost_agents[(1, 0, 0)]
                self.assertEqual((1, 0, 0), ga1.agent.uid)
                self.assertEqual(0, ga1.ref_count)
                self.assertEqual(0, ga1.agent.local_rank)

                ga2 = manager._ghost_agents[(4, 0, 2)]
                self.assertEqual((4, 0, 2), ga2.agent.uid)
                self.assertEqual(0, ga2.ref_count)
                self.assertEqual(2, ga2.agent.local_rank)

                ga3 = manager._ghost_agents[(2, 0, 1)]
                self.assertEqual((2, 0, 1), ga3.agent.uid)
                self.assertEqual(0, ga3.ref_count)
                self.assertEqual(1, ga3.agent.local_rank)

                self.assertEqual(0, len(manager._ghosted_agents))

            elif rank == 1:
                self.assertEqual(0, len(manager._ghost_agents))

                self.assertEqual(2, len(manager._ghosted_agents))
                ghosted1 = manager._ghosted_agents.get((1, 0, 1))
                self.assertIsNotNone(ghosted1)
                self.assertEqual((1, 0, 1), ghosted1.agent.uid)
                self.assertEqual(1, len(ghosted1.ghost_ranks))
                self.assertTrue(0 in ghosted1.ghost_ranks)

                ghosted2 = manager._ghosted_agents.get((2, 0, 1))
                self.assertIsNotNone(ghosted2)
                self.assertEqual((2, 0, 1), ghosted2.agent.uid)
                self.assertEqual(2, len(ghosted2.ghost_ranks))
                self.assertTrue(0 in ghosted2.ghost_ranks)
                self.assertTrue(3 in ghosted2.ghost_ranks)

    def test_requested_synch(self):
        """ * Request Agents
            * Update Agent state on local rank
            * Synch
            * Test that state update propogates to ghosts
        """
        # 4 rank comm
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1, 2, 3])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            context = ctx.SharedContext(comm)
            manager = context._agent_manager
            agents = [EAgent(x, 0, rank, x) for x in range(10)]
            for a in agents:
                context.add(a)
                self.assertEqual(rank, a.local_rank)

            expected = {(1, 0, 1): 23, (1, 0, 2): 12, (2, 0, 1): 534,
                        (1, 0, 0): 143, (4, 0, 2): 34}

            requests = []
            if rank == 0:
                # requestion (1, 0, 1) from 1
                requests.append(((1, 0, 1), 1))
                requests.append(((1, 0, 2), 2))
                requests.append(((2, 0, 1), 1))
            elif rank == 3:
                requests.append(((1, 0, 0), 0))
                requests.append(((4, 0, 2), 2))

            context.request_agents(requests, create_agent)

            if rank == 0:
                manager.get_local((1, 0, 0)).energy = expected[(1, 0, 0)]
            elif rank == 1:
                manager.get_local((1, 0, 1)).energy = expected[(1, 0, 1)]
                manager.get_local((2, 0, 1)).energy = expected[(2, 0, 1)]
            elif rank == 2:
                manager.get_local((4, 0, 2)).energy = expected[(4, 0, 2)]
                manager.get_local((1, 0, 2)).energy = expected[(1, 0, 2)]

            context.synchronize(create_agent)

            if rank == 0:
                uids = [(1, 0, 1), (1, 0, 2), (2, 0, 1)]
                for uid in uids:
                    self.assertEqual(expected[uid], manager.get_ghost(uid, incr=0).energy)
            elif rank == 3:
                uids = [(1, 0, 0), (4, 0, 2)]
                for uid in uids:
                    self.assertEqual(expected[uid], manager.get_ghost(uid, incr=0).energy)

    def test_requested_with_proj(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1, 2, 3])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            context = ctx.SharedContext(comm)
            agents = {}
            for a in [EAgent(x, 0, rank, x) for x in range(10)]:
                context.add(a)
                self.assertEqual(rank, a.local_rank)
                agents[a.uid] = a

            requests = []
            if rank == 0:
                requests.append(((1, 0, 1), 1))
                requests.append(((1, 0, 2), 2))
                requests.append(((2, 0, 1), 1))
            elif rank == 3:
                requests.append(((1, 0, 0), 0))
                requests.append(((4, 0, 2), 2))

            context.request_agents(requests, create_agent)

            manager = context._agent_manager

            box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
            context.add_projection(cspace)
            context.add_projection(grid)

            if rank == 0:
                ga1 = manager._ghost_agents[(1, 0, 1)]
                self.assertEqual(2, ga1.ref_count)

                ga2 = manager._ghost_agents[(1, 0, 2)]
                self.assertEqual(2, ga2.ref_count)

                ga3 = manager._ghost_agents[(2, 0, 1)]
                self.assertEqual(2, ga3.ref_count)

            elif rank == 3:
                self.assertEqual(2, len(manager._ghost_agents))

                ga1 = manager._ghost_agents[(1, 0, 0)]
                self.assertEqual(2, ga1.ref_count)

                ga2 = manager._ghost_agents[(4, 0, 2)]
                self.assertEqual(2, ga2.ref_count)

            # move
            # ghosted on 0: (1, 0, 1), (1, 0, 2), (2, 0, 1),
            # ghosted on 3: (1, 0, 0),(4, 0, 2)

            # local bounds
            # 0: BoundingBox(xmin=0, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
            # 1: BoundingBox(xmin=0, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)
            # 2: BoundingBox(xmin=45, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
            # 3: BoundingBox(xmin=45, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)

            if rank == 0:
                # to 1
                grid.move(agents[(1, 0, 0)], DPt(20, 65))

            elif rank == 1:
                # to 0
                grid.move(agents[(1, 0, 1)], DPt(20, 20))
                # to 2
                grid.move(agents[(2, 0, 1)], DPt(50, 20))
            elif rank == 2:
                # to 3
                grid.move(agents[(1, 0, 2)], DPt(50, 65))
                # to 0
                grid.move(agents[(4, 0, 2)], DPt(20, 30))

            try:
                context.synchronize(create_agent)
            except Exception as e:
                e = traceback.format_exc()
                print(e, flush=True)

            if rank == 0:
                # (4, 0, 2) now on 0, and ghosted to 3
                # (1, 0, 1) now on 0, and was ghosted to 0, now just local
                self.assertIsNotNone(manager.get_local((4, 0, 2)))
                self.assertIsNotNone(manager.get_local((1, 0, 1)))
                self.assertIsNone(manager.get_ghost((1, 0, 1), incr=0))
                self.assertEqual(0, manager.get_local((1, 0, 1)).local_rank)
                self.assertEqual(1, len(manager._ghosted_agents))
                self.assertTrue(3 in manager._ghosted_agents[(4, 0, 2)].ghost_ranks)
            elif rank == 1:
                # (1, 0, 0) moved to 1, and ghosted to 3
                self.assertEqual(1, len(manager._ghosted_agents))
                self.assertTrue(3 in manager._ghosted_agents[(1, 0, 0)].ghost_ranks)
                self.assertIsNotNone(manager.get_local((1, 0, 0)))
            elif rank == 2:
                # 2, 0, 1 moved to 2 ghosted on 0
                self.assertEqual(1, len(manager._ghosted_agents))
                self.assertTrue(0 in manager._ghosted_agents[(2, 0, 1)].ghost_ranks)
                self.assertIsNotNone(manager.get_local((2, 0, 1)))
            elif rank == 3:
                self.assertEqual(1, len(manager._ghosted_agents))
                self.assertTrue(0 in manager._ghosted_agents[(1, 0, 2)].ghost_ranks)
                self.assertIsNotNone(manager.get_local((1, 0, 2)))

            if rank == 1:
                context.agent((1, 0, 0)).energy = 1253

            try:
                context.synchronize(create_agent)
            except Exception as e:
                e = traceback.format_exc()
                print(e, flush=True)

            if rank == 3:
                # 1,0,0 was moved to 1 from 0, and ghosted on 3
                # test that state update on 1 propogates to 3
                self.assertEqual(1253, manager.get_ghost((1, 0, 0), incr=0).energy)

    def test_requested_with_removed(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1, 2, 3])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            context = ctx.SharedContext(comm)
            agents = {}
            for a in [EAgent(x, 0, rank, x) for x in range(10)]:
                context.add(a)
                self.assertEqual(rank, a.local_rank)
                agents[a.uid] = a

            requests = []
            if rank == 0:
                requests.append(((1, 0, 1), 1))
                requests.append(((1, 0, 2), 2))
                requests.append(((2, 0, 1), 1))
                requests.append(((4, 0, 2), 2))
            elif rank == 3:
                requests.append(((1, 0, 0), 0))
                requests.append(((4, 0, 2), 2))

            context.request_agents(requests, create_agent)

            manager = context._agent_manager

            box = geometry.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
            context.add_projection(cspace)
            context.add_projection(grid)

            if rank == 2:
                # remove (4,0,2) from two, and so as ghosts on 0 and 3
                context.remove(agents[(4, 0, 2)])
                self.assertIsNone(context.agent((4, 0, 2)))

            try:
                context.synchronize(create_agent)
            except Exception as e:
                e = traceback.format_exc()
                print(e, flush=True)

            if rank == 0 or rank == 3:
                self.assertIsNone(manager.get_ghost((4, 0, 2), incr=0))


class PeriodicSyncTests(unittest.TestCase):

    long_message = True

    def test_4x2_synch(self):
        new_group = MPI.COMM_WORLD.Get_group().Excl([8])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            context = ctx.SharedContext(comm)

            agents = [EAgent(x, 0, rank, x) for x in range(10)]
            for a in agents:
                context.add(a)

            box = geometry.BoundingBox(xmin=0, xextent=80, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Periodic,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
            context.add_projection(cspace)
            context.add_projection(grid)

            # put agents in center, so buffer areas are empty
            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            cpt = CPt(lb.xmin + lb.xextent / 2, lb.ymin + lb.yextent / 2)
            dpt = DPt(math.floor(cpt.x), math.floor(cpt.y))
            for a in agents:
                grid.move(a, dpt)
                cspace.move(a, cpt)

            context.synchronize(create_agent)
            self.assertTrue(10, len(context._agent_manager._local_agents))
            for a in agents:
                self.assertEqual(cpt, cspace.get_location(a))
                self.assertEqual(dpt, grid.get_location(a))

            # Test:
            # Move agents to boundaries, sync, and make sure ghosts are where we expect
            mid_x = lb.xmin + lb.xextent / 2
            mid_y = lb.ymin + lb.yextent / 2
            max_x = lb.xmin + lb.xextent - 1
            max_y = lb.ymin + lb.yextent - 1
            n_pt = DPt(int(mid_x), lb.ymin)
            s_pt = DPt(int(mid_x), max_y)
            e_pt = DPt(max_x, int(mid_y))
            w_pt = DPt(lb.xmin, int(mid_y))
            ne_pt = DPt(max_x, lb.ymin)
            nw_pt = DPt(lb.xmin, lb.ymin)
            se_pt = DPt(max_x, max_y)
            sw_pt = DPt(lb.xmin, max_y)

            pts = [n_pt, s_pt, e_pt, w_pt, ne_pt, nw_pt, se_pt, sw_pt]

            nghs = {
                0: [1, 1, 2, 6, 3, 7, 3, 7],
                1: [0, 0, 3, 7, 2, 6, 2, 6],
                2: [3, 3, 4, 0, 5, 1, 5, 1],
                3: [2, 2, 5, 1, 4, 0, 4, 0],
                4: [5, 5, 6, 2, 7, 3, 7, 3],
                5: [4, 4, 7, 3, 6, 2, 6, 2],
                6: [7, 7, 0, 4, 1, 5, 1, 5],
                7: [6, 6, 1, 5, 0, 4, 0, 4]
            }

            send_data = [[] for i in range(comm.size)]

            # move agents to buffer zone points
            # nghs contains list of ranks corresponding to those points
            # i.e., 0 sends agents at north point to 1, and so on
            # send the expected agents and location to ngh rank
            # test on ngh rank using that expected data
            for i in range(8):
                a = agents[i]
                pt = pts[i]
                grid.move(a, pt)
                cspace.move(a, CPt(pt.x, pt.y))
                ngh = nghs[rank][i]
                send_data[ngh].append((a.uid, pt.coordinates))

            recv_data = comm.alltoall(send_data)

            context.synchronize(create_agent)

            for lst in recv_data:
                for uid, coords in lst:
                    dpt = DPt(coords[0], coords[1])
                    cpt = CPt(coords[0], coords[1])
                    agent = context.ghost_agent(uid)
                    self.assertIsNotNone(agent)
                    self.assertEqual(dpt, grid.get_location(agent))
                    self.assertEqual(cpt, cspace.get_location(agent))

    def test_2x1_synch(self):
        new_group = MPI.COMM_WORLD.Get_group().Incl([0, 1])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            context = ctx.SharedContext(comm)

            agents = [EAgent(x, 0, rank, x) for x in range(10)]
            for a in agents:
                context.add(a)

            box = geometry.BoundingBox(xmin=0, xextent=80, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Periodic,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
            context.add_projection(cspace)
            context.add_projection(grid)

            # put agents in center, so buffer areas are empty
            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            cpt = CPt(lb.xmin + lb.xextent / 2, lb.ymin + lb.yextent / 2)
            dpt = DPt(math.floor(cpt.x), math.floor(cpt.y))
            for a in agents:
                grid.move(a, dpt)
                cspace.move(a, cpt)

            context.synchronize(create_agent)
            self.assertTrue(10, len(context._agent_manager._local_agents))
            for a in agents:
                self.assertEqual(cpt, cspace.get_location(a))
                self.assertEqual(dpt, grid.get_location(a))

            # Test:
            # Move agents to boundaries, sync, and make sure ghosts are where we expect
            mid_y = lb.ymin + lb.yextent / 2
            max_x = lb.xmin + lb.xextent - 1
            max_y = lb.ymin + lb.yextent - 1
            e_pt = DPt(max_x, int(mid_y))
            w_pt = DPt(lb.xmin, int(mid_y))
            ne_pt = DPt(max_x, lb.ymin)
            nw_pt = DPt(lb.xmin, lb.ymin)
            se_pt = DPt(max_x, max_y)
            sw_pt = DPt(lb.xmin, max_y)

            # omit n/s as those wrap to self
            pts = [e_pt, w_pt, ne_pt, nw_pt, se_pt, sw_pt]
            nghs = {
                0: [1, 1, 1, 1, 1, 1],
                1: [0, 0, 0, 0, 0, 0]
            }

            send_data = [[] for i in range(comm.size)]

            # move agents to buffer zone points
            # nghs contains list of ranks corresponding to those points
            # i.e., 0 sends agents at north point to 1, and so on
            # send the expected agents and location to ngh rank
            # test on ngh rank using that expected data
            for i in range(6):
                a = agents[i]
                pt = pts[i]
                grid.move(a, pt)
                cspace.move(a, CPt(pt.x, pt.y))
                ngh = nghs[rank][i]
                send_data[ngh].append((a.uid, pt.coordinates))

            recv_data = comm.alltoall(send_data)

            context.synchronize(create_agent)

            for lst in recv_data:
                for uid, coords in lst:
                    dpt = DPt(coords[0], coords[1])
                    cpt = CPt(coords[0], coords[1])
                    agent = context.ghost_agent(uid)
                    self.assertIsNotNone(agent)
                    self.assertEqual(dpt, grid.get_location(agent))
                    self.assertEqual(cpt, cspace.get_location(agent))

    def test_2x2_synch(self):
        new_group = MPI.COMM_WORLD.Get_group().Excl([4, 5, 6, 7, 8])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            context = ctx.SharedContext(comm)

            agents = [EAgent(x, 0, rank, x) for x in range(10)]
            for a in agents:
                context.add(a)

            box = geometry.BoundingBox(xmin=0, xextent=80, ymin=0, yextent=120, zmin=0, zextent=0)
            cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Periodic,
                                        occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm,
                                        tree_threshold=100)
            grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Periodic,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
            context.add_projection(cspace)
            context.add_projection(grid)

            # put agents in center, so buffer areas are empty
            topo = CartesianTopology(comm, box, True)
            lb = topo.local_bounds
            cpt = CPt(lb.xmin + lb.xextent / 2, lb.ymin + lb.yextent / 2)
            dpt = DPt(math.floor(cpt.x), math.floor(cpt.y))
            for a in agents:
                grid.move(a, dpt)
                cspace.move(a, cpt)

            context.synchronize(create_agent)
            self.assertTrue(10, len(context._agent_manager._local_agents))
            for a in agents:
                self.assertEqual(cpt, cspace.get_location(a))
                self.assertEqual(dpt, grid.get_location(a))

            # Test:
            # Move agents to boundaries, sync, and make sure ghosts are where we expect
            mid_x = lb.xmin + lb.xextent / 2
            mid_y = lb.ymin + lb.yextent / 2
            max_x = lb.xmin + lb.xextent - 1
            max_y = lb.ymin + lb.yextent - 1
            n_pt = DPt(int(mid_x), lb.ymin)
            s_pt = DPt(int(mid_x), max_y)
            e_pt = DPt(max_x, int(mid_y))
            w_pt = DPt(lb.xmin, int(mid_y))
            ne_pt = DPt(max_x, lb.ymin)
            nw_pt = DPt(lb.xmin, lb.ymin)
            se_pt = DPt(max_x, max_y)
            sw_pt = DPt(lb.xmin, max_y)

            pts = [n_pt, s_pt, e_pt, w_pt, ne_pt, nw_pt, se_pt, sw_pt]

            nghs = {
                0: [1, 1, 2, 2, 3, 3, 3, 3],
                1: [0, 0, 3, 3, 2, 2, 2, 2],
                2: [3, 3, 0, 0, 1, 1, 1, 1],
                3: [2, 2, 1, 1, 0, 0, 0, 0],
            }

            send_data = [[] for i in range(comm.size)]

            # move agents to buffer zone points
            # nghs contains list of ranks corresponding to those points
            # i.e., 0 sends agents at north point to 1, and so on
            # send the expected agents and location to ngh rank
            # test on ngh rank using that expected data
            for i in range(8):
                a = agents[i]
                pt = pts[i]
                grid.move(a, pt)
                cspace.move(a, CPt(pt.x, pt.y))
                ngh = nghs[rank][i]
                send_data[ngh].append((a.uid, pt.coordinates))

            recv_data = comm.alltoall(send_data)

            context.synchronize(create_agent)

            for lst in recv_data:
                for uid, coords in lst:
                    dpt = DPt(coords[0], coords[1])
                    cpt = CPt(coords[0], coords[1])
                    agent = context.ghost_agent(uid)
                    self.assertIsNotNone(agent)
                    self.assertEqual(dpt, grid.get_location(agent))
                    self.assertEqual(cpt, cspace.get_location(agent))
