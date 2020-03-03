import sys
import os
import numpy as np

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

import unittest
from repast4py import core, space
from repast4py.space import BorderType, OccupancyType


class ContinuousPointTests(unittest.TestCase):

    def test_ops(self):
        # test z default to 0
        pt = space.ContinuousPoint(1, 2)
        self.assertEqual(1.0, pt.x)
        self.assertEqual(2.0, pt.y)
        self.assertEqual(0.0, pt.z)

        pt = space.ContinuousPoint(2.34, 34.1, 11.1)
        self.assertEqual(2.34, pt.x)
        self.assertEqual(34.1, pt.y)
        self.assertEqual(11.1, pt.z)
        self.assertTrue(np.array_equal(np.array([2.34, 34.1, 11.1]), pt.coordinates))

        pt._reset1D(12.1)
        self.assertEqual(12.1, pt.x)

        pt._reset2D(10.1, -1.2)
        self.assertEqual(10.1, pt.x)
        self.assertEqual(-1.2, pt.y)

        pt._reset3D(0.1, -11.2, 11111.3) 
        self.assertEqual(0.1, pt.x)
        self.assertEqual(-11.2, pt.y)
        self.assertEqual(11111.3, pt.z)

        pt._reset((1.233, -134.1, 10.1))
        self.assertEqual(1.233, pt.x)
        self.assertEqual(-134.1, pt.y)
        self.assertEqual(10.1, pt.z)



class GridTests(unittest.TestCase):

    def test_move(self):
        a1 = core.Agent(1, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        grid = space.Grid("grid", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple)

        grid.add(a1)

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
        
        grid.move(a1, space.DiscretePoint(-1, 12))
        pt = grid.get_location(a1)
        self.assertEqual(0, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(21, 12))
        pt = grid.get_location(a1)
        self.assertEqual(19, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(5, -23))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(0, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(5, 25))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(24, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(5, 10, -2))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(10, pt.y)
        self.assertEqual(-1, pt.z)

        grid.move(a1, space.DiscretePoint(5, 10, 4))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(10, pt.y)
        self.assertEqual(3, pt.z)

    
    def test_periodic_move(self):
        a1 = core.Agent(1, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        grid = space.Grid("grid", bounds=box, borders=BorderType.Periodic, occupancy=OccupancyType.Multiple)

        grid.add(a1)

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
        
        grid.move(a1, space.DiscretePoint(-1, 12))
        pt = grid.get_location(a1)
        # wraps around
        self.assertEqual(19, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(21, 12))
        pt = grid.get_location(a1)
        self.assertEqual(1, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(5, -23))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(2, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(5, 25))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(0, pt.y)
        self.assertEqual(0, pt.z)

        grid.move(a1, space.DiscretePoint(5, 10, -2))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(10, pt.y)
        self.assertEqual(3, pt.z)

        grid.move(a1, space.DiscretePoint(5, 10, 4))
        pt = grid.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(10, pt.y)
        self.assertEqual(-1, pt.z)


    def test_remove(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=0, zextent=0)
        grid = space.Grid("grid", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple)
        agent = grid.get_agent(space.DiscretePoint(0, 0))
        self.assertIsNone(agent)

        a1 = core.Agent(1, 0)
        grid.add(a1)
        pt = space.DiscretePoint(2, 4)
        grid.move(a1, pt)
        agent = grid.get_agent(pt)
        self.assertEqual(agent, a1)
        pt1 = grid.get_location(a1)
        self.assertEqual(pt.x, pt1.x)
        self.assertEqual(pt.y, pt1.y)
        self.assertEqual(pt.z, pt1.z)

        ret = grid.remove(agent)
        self.assertTrue(ret)
        ret = grid.get_location(agent)
        self.assertIsNone(ret)


    def test_get(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        grid = space.Grid("grid", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple)
        
        agent = grid.get_agent(space.DiscretePoint(0, 0))
        self.assertIsNone(agent)

        a1 = core.Agent(1, 0)
        grid.add(a1)
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
        grid.add(a2)
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

class CSpaceTests(unittest.TestCase):

    def test_move(self):
        a1 = core.Agent(1, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple)

        cspace.add(a1)

        # move after initial add
        cspace.move(a1, space.ContinuousPoint(2.1, 4.4))
        pt = cspace.get_location(a1)
        self.assertEqual(2.1, pt.x)
        self.assertEqual(4.4, pt.y)
        self.assertEqual(0, pt.z)

        # move to same location
        cspace.move(a1, space.ContinuousPoint(2.1, 4.4))
        pt = cspace.get_location(a1)
        self.assertEqual(2.1, pt.x)
        self.assertEqual(4.4, pt.y)
        self.assertEqual(0, pt.z)

        # # move to new position after initial add
        cspace.move(a1, space.ContinuousPoint(10.2, 12.1))
        pt = cspace.get_location(a1)
        self.assertEqual(10.2, pt.x)
        self.assertEqual(12.1, pt.y)
        self.assertEqual(0, pt.z)
        self.assertIsNone(cspace.get_agent(space.ContinuousPoint(2.1, 4.4)))

        # # move to same location
        cspace.move(a1, space.ContinuousPoint(10.2, 12.1))
        pt = cspace.get_location(a1)
        self.assertEqual(10.2, pt.x)
        self.assertEqual(12.1, pt.y)
        self.assertEqual(0, pt.z)
        
        # test sticky borders
        cspace.move(a1, space.ContinuousPoint(-1.1, 12))
        pt = cspace.get_location(a1)
        self.assertEqual(0, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(21.2, 12))
        pt = cspace.get_location(a1)
        self.assertEqual(19, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(5, -23))
        pt = cspace.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(0, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(5, 25.1))
        pt = cspace.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(24, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(5, 10, -1.2))
        pt = cspace.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(10, pt.y)
        self.assertEqual(-1, pt.z)

        cspace.move(a1, space.ContinuousPoint(5.1, 10.34, 4.134))
        pt = cspace.get_location(a1)
        self.assertEqual(5.1, pt.x)
        self.assertEqual(10.34, pt.y)
        self.assertEqual(3.0, pt.z)

    
    def test_periodic_move(self):
        a1 = core.Agent(1, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Periodic, occupancy=OccupancyType.Multiple)

        cspace.add(a1)

        # move after initial add
        cspace.move(a1, space.ContinuousPoint(2, 4))
        pt = cspace.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        # move to same location
        cspace.move(a1, space.ContinuousPoint(2, 4))
        pt = cspace.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        # # move to new position after initial add
        cspace.move(a1, space.ContinuousPoint(10, 12))
        pt = cspace.get_location(a1)
        self.assertEqual(10, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        # # move to same location
        cspace.move(a1, space.ContinuousPoint(10, 12))
        pt = cspace.get_location(a1)
        self.assertEqual(10, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)
        
        cspace.move(a1, space.ContinuousPoint(-1, 12))
        pt = cspace.get_location(a1)
        # wraps around
        self.assertEqual(19, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(21, 12))
        pt = cspace.get_location(a1)
        self.assertEqual(1, pt.x)
        self.assertEqual(12, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(5, -23))
        pt = cspace.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(2, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(5, 25))
        pt = cspace.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(0, pt.y)
        self.assertEqual(0, pt.z)

        cspace.move(a1, space.ContinuousPoint(5, 10, -2))
        pt = cspace.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(10, pt.y)
        self.assertEqual(3, pt.z)

        cspace.move(a1, space.ContinuousPoint(5, 10, 4))
        pt = cspace.get_location(a1)
        self.assertEqual(5, pt.x)
        self.assertEqual(10, pt.y)
        self.assertEqual(-1, pt.z)


    def test_remove(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=0, zextent=0)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple)
        agent = cspace.get_agent(space.ContinuousPoint(0, 0))
        self.assertIsNone(agent)

        a1 = core.Agent(1, 0)
        cspace.add(a1)
        pt = space.ContinuousPoint(2, 4)
        cspace.move(a1, pt)
        agent = cspace.get_agent(pt)
        self.assertEqual(agent, a1)
        pt1 = cspace.get_location(a1)
        self.assertEqual(pt.x, pt1.x)
        self.assertEqual(pt.y, pt1.y)
        self.assertEqual(pt.z, pt1.z)

        ret = cspace.remove(agent)
        self.assertTrue(ret)
        ret = cspace.get_location(agent)
        self.assertIsNone(ret)


    def test_get(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple)
        
        agent = cspace.get_agent(space.ContinuousPoint(0, 0))
        self.assertIsNone(agent)

        a1 = core.Agent(1, 0)
        cspace.add(a1)
        pt = space.ContinuousPoint(2, 4)
        cspace.move(a1, pt)
        agent = cspace.get_agent(pt)
        self.assertEqual(agent, a1)

        agents = cspace.get_agents(pt)
        expected = [a1]
        count  = 0
        for i, agent in enumerate(agents):
            self.assertEqual(expected[i], agent)
            count += 1
        self.assertEqual(1, count)

        a2 = core.Agent(2, 0)
        cspace.add(a2)
        cspace.move(a2, pt)

        # gets the first added agent
        agent = cspace.get_agent(pt)
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
        cspace.move(a1, pt)
        count = 0
        for i, agent in enumerate(agents):
            self.assertEqual(expected[i], agent)
            count += 1
        self.assertEqual(2, count)

        pt2 = space.ContinuousPoint(14, 1)
        cspace.move(a2, pt2)
        agents = cspace.get_agents(pt2)
        count = 0
        for agent in agents:
            self.assertEqual(a2, agent)
            count += 1
        self.assertEqual(1, count)
        agent = cspace.get_agent(pt2)
        self.assertEqual(agent, a2)

        agents = cspace.get_agents(pt)
        count = 0
        for agent in agents:
            self.assertEqual(a1, agent)
            count += 1
        self.assertEqual(1, count)

        agent = cspace.get_agent(pt)
        self.assertEqual(agent, a1)



if __name__ == "__main__":
    unittest.main()





