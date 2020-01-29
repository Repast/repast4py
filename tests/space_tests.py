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

        # move to new position after initial add
        grid.move(a1, space.DiscretePoint(10, 12))
        pt = grid.get_location(a1)
        self.assertEqual(10, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        # move to same location
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
        





