import sys
import os
from mpi4py import MPI

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

import unittest
from repast4py import core, space
from repast4py.space import BorderType, OccupancyType


class SharedGridTests(unittest.TestCase):

    def test_ops(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        a1 = core.Agent(1, 0, rank)
        a2 = core.Agent(2, 0, rank)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky, 
            occupancy=OccupancyType.Multiple, buffersize=2, comm=MPI.COMM_WORLD)

        grid.add(a1)
        grid.add(a2)

        if (rank == 0):
            pt = space.DiscretePoint(5, 20)
            grid.move(a1, pt)
            self.assertEqual(a1, grid.get_agent(pt))
            self.assertEqual(pt, grid.get_location(a1))

            agents = grid.get_agents(pt)
            expected = [a1]
            count  = 0
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
            count  = 0
            for i, agent in enumerate(agents):
                self.assertEqual(expected[i], agent)
                count += 1
            self.assertEqual(2, count)

    def test_move(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        a1 = core.Agent(1, 0, rank)
        a2 = core.Agent(2, 0, rank)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky, 
            occupancy=OccupancyType.Multiple, buffersize=2, comm=MPI.COMM_WORLD)

        grid.add(a1)
        grid.add(a2)

        if (rank == 0):
            pt = space.DiscretePoint(5, 20)
            grid.move(a1, pt)
            # move out of bounds
            pt = space.DiscretePoint(12, 22)
            grid.move(a1, pt)

            expected = {(1, 0, 0) : (1, space.DiscretePoint(12, 22))}
            for ob in grid._get_oob():
                exp = expected.pop(ob[0])
                self.assertEqual(exp[0], ob[1])
                self.assertEqual(exp[1], ob[2])
            self.assertEqual(0, len(expected))


        if (rank == 1):
            a3 = core.Agent(1, 0, rank)
            grid.add(a3)

            pt = space.DiscretePoint(12, 39)
            grid.move(a2, pt)
            grid.move(a3, pt)

            grid.move(a2, space.DiscretePoint(0, 1))
            grid.move(a3, space.DiscretePoint(8, 200))
            
            expected = {(2, 0, 1) : (0, space.DiscretePoint(0, 1, 0)),
                (1, 0, 1) : (0, space.DiscretePoint(8, 39, 0))}
            for ob in grid._get_oob():
                exp = expected.pop(ob[0])
                self.assertEqual(exp[0], ob[1])
                self.assertEqual(exp[1], ob[2])
            self.assertEqual(0, len(expected))

            


        
    
