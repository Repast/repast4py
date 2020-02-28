import sys
import os
from mpi4py import MPI
import numpy as np

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

            expected = {(1, 0, 0) : (1, np.array([12, 22, 0]))}
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
            
            expected = {(2, 0, 1) : (0, np.array([0, 1, 0])),
                (1, 0, 1) : (0, np.array([8, 39, 0]))}
            for ob in grid._get_oob():
                exp = expected.pop(ob[0])
                self.assertEqual(exp[0], ob[1])
                self.assertTrue(np.array_equal(exp[1], ob[2]))
            self.assertEqual(0, len(expected))

    def test_buffer_data(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky, 
            occupancy=OccupancyType.Multiple, buffersize=2, comm=MPI.COMM_WORLD)

        if (rank == 0):
            a01 = core.Agent(1, 0, rank)
            a02 = core.Agent(2, 0, rank)
            a03 = core.Agent(3, 0, rank)
            grid.add(a01)
            grid.add(a02)
            grid.add(a03)

            pt = space.DiscretePoint(8, 20)
            grid.move(a01, pt)
            pt = space.DiscretePoint(9, 15)
            grid.move(a02, pt)
            pt = space.DiscretePoint(1, 15)
            grid.move(a03, pt)

            expected = (1, (8, 10, 0, 40, 0, 0))
            count = 0
            for bd in grid._get_buffer_data():
                self.assertEqual(expected, bd)
                count += 1
            self.assertEqual(1, count)


        if (rank == 1):
            a11 = core.Agent(1, 0, rank)
            a12 = core.Agent(2, 0, rank)
            a13 = core.Agent(3, 0, rank)

            grid.add(a11)
            grid.add(a12)
            grid.add(a13)

            pt = space.DiscretePoint(10, 20)
            grid.move(a11, pt)
            pt = space.DiscretePoint(11, 15)
            grid.move(a12, pt)
            pt = space.DiscretePoint(15, 15)
            grid.move(a13, pt)

            expected = (0, (10, 12, 0, 40, 0, 0))
            count = 0
            for bd in grid._get_buffer_data():
                self.assertEqual(expected, bd)
                count += 1
            self.assertEqual(1, count)
                

class EAgent(core.Agent):

    def __init__(self, id, agent_type, rank, energy):
        super().__init__(id=id, type=agent_type, rank=rank)
        self.energy = energy
        self.restored = False

    def save(self):
        return (self.uid, self.energy)

    def restore(self, data):
        self.restored = True
        self.energy = data[1]


def create_agent(agent_data):
    # agent_data: [aid_tuple, energy]
    uid = agent_data[0]
    return EAgent(uid[0], uid[1], uid[2], agent_data[1])


class SharedContextTests(unittest.TestCase):

    def test_add_remove(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky, 
            occupancy=OccupancyType.Multiple, buffersize=2, comm=MPI.COMM_WORLD)

        context = core.SharedContext(comm)
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


    def test_synch(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky, 
            occupancy=OccupancyType.Multiple, buffersize=2, comm=MPI.COMM_WORLD)

        context = core.SharedContext(comm)
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
            self.assertEqual(2, len(context._local_agents))
            # a2
            agent = context._local_agents[(2, 0, 1)]
            self.assertEqual((2, 0, 1), agent.uid)
            self.assertEqual(3, agent.energy)
            pt = space.DiscretePoint(5, 30)
            self.assertEqual(agent, grid.get_agent(pt))
            self.assertEqual(pt, grid.get_location(agent))

            # a3
            agent = context._local_agents[(3, 0, 1)]
            self.assertEqual((3, 0, 1), agent.uid)
            self.assertEqual(2, agent.energy)
            pt = space.DiscretePoint(3, 20)
            self.assertEqual(agent, grid.get_agent(pt))
            self.assertEqual(pt, grid.get_location(agent))

        else:
            # should now have a1
            self.assertEqual(1, len(context._local_agents))
            agent = context._local_agents[(1, 0, 0)]
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
            self.assertEqual(1, len(context._local_agents))
            # a3
            agent = context._local_agents[(3, 0, 1)]
            self.assertEqual((3, 0, 1), agent.uid)
            self.assertEqual(2, agent.energy)
            pt = space.DiscretePoint(3, 20)
            self.assertEqual(agent, grid.get_agent(pt))
            self.assertEqual(pt, grid.get_location(agent))

        if rank == 1:
            # a2 now back in 1
            self.assertEqual(2, len(context._local_agents))
            agent = context._local_agents[(2, 0, 1)]
            self.assertEqual((2, 0, 1), agent.uid)
            self.assertEqual(-10, agent.energy)
            pt = space.DiscretePoint(12, 38)
            self.assertEqual(agent, grid.get_agent(pt))
            self.assertEqual(pt, grid.get_location(agent))

            agent = context._local_agents[(1, 0, 0)]
            self.assertEqual((1, 0, 0), agent.uid)
            self.assertEqual(12, agent.energy)
            pt = space.DiscretePoint(12, 5)
            self.assertEqual(agent, grid.get_agent(pt))
            self.assertEqual(pt, grid.get_location(agent))



    def test_buffer(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=40, zmin=0, zextent=0)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky, 
            occupancy=OccupancyType.Multiple, buffersize=2, comm=MPI.COMM_WORLD)

        context = core.SharedContext(comm)
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

        context.synchronize(create_agent)
        grid.synchronize_buffer(create_agent)

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

            self.assertEqual(3, len(context._local_agents))

        if rank == 1:
            pt = space.DiscretePoint(8, 20)
            agent = grid.get_agent(pt)
            self.assertIsNotNone(agent)
            self.assertEqual((1, 0, 0), agent.uid)
            self.assertEqual(1, agent.energy)

            pt = space.DiscretePoint(9, 15)
            expected = {(2,0,0) : 2, (3, 0, 0) : 3}
            for a in grid.get_agents(pt):
                energy = expected.pop(a.uid) 
                self.assertEqual(energy, a.energy)
            self.assertEqual(0, len(expected))

            self.assertEqual(3, len(context._local_agents))

