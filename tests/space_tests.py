import sys
import os

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

import unittest
from repast4py import core, space


class GridTests(unittest.TestCase):

    def test_move(self):
        a1 = core.Agent(1, 0)

        grid = space.Grid("grid")
        grid.add_agent(a1)

        # move after initial add
        grid.move(a1, space.DiscretePoint(2, 4))
        pt = grid.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        # move to same location
        grid.move(a1, space.DiscretePoint(2, 4))
        pt = grid.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        # # move to new position after initial add
        grid.move(a1, space.DiscretePoint(10, 12))
        pt = grid.get_location(a1)
        self.assertEqual(10, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        # # move to same location
        grid.move(a1, space.DiscretePoint(10, 12))
        pt = grid.get_location(a1)
        self.assertEqual(10, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)


    def test_get(self):
        grid = space.Grid("grid")
        agent = grid.get_agent(space.DiscretePoint(0, 0))
        self.assertIsNone(agent)

        a1 = core.Agent(1, 0)
        grid.add_agent(a1)
        pt = space.DiscretePoint(2, 4)
        grid.move(a1, pt)
        agent = grid.get_agent(pt)
        self.assertEqual(agent, a1)

        agents = grid.get_agents(pt)
        expected = [a1]
        count  = 0
        for i, agent in enumerate(agents):
            self.assertEqual(expected[i], agent)
            count += 1
        self.assertEqual(1, count)

        a2 = core.Agent(2, 0)
        grid.add_agent(a2)
        grid.move(a2, pt)

        # gets the first added agent
        agent = grid.get_agent(pt)
        self.assertEqual(agent, a1)

        # get all should now return both
        expected = [a1, a2]
        count = 0
        for i, agent in enumerate(agents):
            self.assertEqual(expected[i], agent)
            count += 1
        self.assertEqual(2, count)

        # add a1 to same spot and make sure still
        # just the two
        grid.move(a1, pt)
        count = 0
        for i, agent in enumerate(agents):
            self.assertEqual(expected[i], agent)
            count += 1
        self.assertEqual(2, count)

        pt2 = space.DiscretePoint(14, 1)
        grid.move(a2, pt2)
        agents = grid.get_agents(pt2)
        count = 0
        for agent in agents:
            self.assertEqual(a2, agent)
            count += 1
        self.assertEqual(1, count)
        agent = grid.get_agent(pt2)
        self.assertEqual(agent, a2)

        agents = grid.get_agents(pt)
        count = 0
        for agent in agents:
            self.assertEqual(a1, agent)
            count += 1
        self.assertEqual(1, count)

        agent = grid.get_agent(pt)
        self.assertEqual(agent, a1)





        





