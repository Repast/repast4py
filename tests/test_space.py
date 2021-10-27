import sys
import os
import numpy as np
import random
import unittest

try:
    from repast4py import core, space
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py import core, space

from repast4py.space import BorderType, OccupancyType, GridStickyBorders, GridPeriodicBorders
from repast4py.space import DiscretePoint as DPt


class PointTests(unittest.TestCase):

    def test_cp_ops(self):
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

        arr = np.array([23.31, 43.3, 423.34])
        pt._reset_from_array(arr)
        self.assertEqual(23.31, pt.x)
        self.assertEqual(43.3, pt.y)
        self.assertEqual(423.34, pt.z)

    def test_dp_ops(self):
        # test z default to 0
        pt = space.DiscretePoint(1, 2)
        self.assertEqual(1, pt.x)
        self.assertEqual(2, pt.y)
        self.assertEqual(0, pt.z)

        pt = space.DiscretePoint(2, 34, 11)
        self.assertEqual(2, pt.x)
        self.assertEqual(34, pt.y)
        self.assertEqual(11, pt.z)
        self.assertTrue(np.array_equal(np.array([2, 34, 11]), pt.coordinates))

        pt._reset1D(12)
        self.assertEqual(12, pt.x)

        pt._reset2D(10, -1)
        self.assertEqual(10, pt.x)
        self.assertEqual(-1, pt.y)

        pt._reset3D(0, -11, 11111)
        self.assertEqual(0, pt.x)
        self.assertEqual(-11, pt.y)
        self.assertEqual(11111, pt.z)

        pt._reset((1, -134, 10))
        self.assertEqual(1, pt.x)
        self.assertEqual(-134, pt.y)
        self.assertEqual(10, pt.z)

        arr = np.array([23, 43, 423])
        pt._reset_from_array(arr)
        self.assertEqual(23, pt.x)
        self.assertEqual(43, pt.y)
        self.assertEqual(423, pt.z)


class BorderTests(unittest.TestCase):

    def test_gsb_transform(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0,
                                yextent=25, zmin=-1, zextent=5)
        borders = GridStickyBorders(box)
        tpt = DPt(0, 0)
        pt = DPt(2, 15, 3)
        borders._transform(pt, tpt)
        self.assertEqual(2, tpt.x)
        self.assertEqual(15, tpt.y)
        self.assertEqual(3, tpt.z)

        pt._reset3D(-1, -2, -3)
        borders._transform(pt, tpt)
        self.assertEqual(0, tpt.x)
        self.assertEqual(0, tpt.y)
        self.assertEqual(-1, tpt.z)

        pt._reset3D(25, 40, 33)
        borders._transform(pt, tpt)
        self.assertEqual(19, tpt.x)
        self.assertEqual(24, tpt.y)
        self.assertEqual(3, tpt.z)

    def test_pb_transform(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0,
                                yextent=25, zmin=-1, zextent=5)
        borders = GridPeriodicBorders(box)
        tpt = DPt(0, 0)
        pt = DPt(2, 15, 3)
        borders._transform(pt, tpt)
        self.assertEqual(2, tpt.x)
        self.assertEqual(15, tpt.y)
        self.assertEqual(3, tpt.z)

        pt._reset3D(-1, -2, -3)
        borders._transform(pt, tpt)
        self.assertEqual(19, tpt.x)
        self.assertEqual(23, tpt.y)
        self.assertEqual(2, tpt.z)

        pt._reset3D(25, 40, 34)
        borders._transform(pt, tpt)
        self.assertEqual(5, tpt.x)
        self.assertEqual(15, tpt.y)
        self.assertEqual(-1, tpt.z)


class GridTests(unittest.TestCase):

    def test_move(self):
        a1 = core.Agent(1, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        grid = space.Grid("grid", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple)
        self.assertEqual('grid', grid.name)

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
        count = 0
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

    def test_single_occ(self):
        a1 = core.Agent(1, 0)
        a2 = core.Agent(2, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        grid = space.Grid("grid", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Single)
        self.assertEqual('grid', grid.name)

        grid.add(a1)

        # move after initial add
        grid.move(a1, space.DiscretePoint(2, 4))
        pt = grid.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        grid.add(a2)
        pt = grid.move(a2, space.DiscretePoint(2, 4))
        self.assertIsNone(pt)
        agents = []
        for a in grid.get_agents(space.DiscretePoint(2, 4)):
            agents.append(a)
        self.assertEqual(1, len(agents))
        self.assertEqual(a1, agents[0])

        pt = grid.move(a2, space.DiscretePoint(3, 4))
        self.assertIsNotNone(pt)
        self.assertEqual(3, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        pt = grid.move(a1, space.DiscretePoint(3, 4))
        self.assertIsNone(pt)
        pt = grid.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        pt = grid.move(a2, space.DiscretePoint(4, 4))
        a = grid.get_agent(space.DiscretePoint(3, 4))
        self.assertIsNone(a)

        grid.remove(a2)
        pt = grid.get_location(a2)
        self.assertIsNone(pt)
        a = grid.get_agent(space.DiscretePoint(4, 4))
        self.assertIsNone(a)

        pt = grid.move(a1, space.DiscretePoint(4, 4))
        self.assertIsNotNone(pt)
        self.assertEqual(4, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        # Test tha periodic + single returns valid object
        grid = space.Grid("grid", bounds=box, borders=BorderType.Periodic, occupancy=OccupancyType.Single)
        self.assertEqual('grid', grid.name)

        grid.add(a1)

        # move after initial add
        grid.move(a1, space.DiscretePoint(2, 4))
        pt = grid.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        grid.add(a2)
        pt = grid.move(a2, space.DiscretePoint(2, 4))
        self.assertIsNone(pt)
        agents = []
        for a in grid.get_agents(space.DiscretePoint(2, 4)):
            agents.append(a)
        self.assertEqual(1, len(agents))
        self.assertEqual(a1, agents[0])


class CSpaceTests(unittest.TestCase):

    def test_move(self):
        a1 = core.Agent(1, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, tree_threshold=100)
        self.assertEqual('cspace', cspace.name)

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
        self.assertEqual(19.99999999, pt.x)
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
        self.assertEqual(24.99999999, pt.y)
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
        self.assertEqual(3.99999999, pt.z)

    def test_periodic_move(self):
        a1 = core.Agent(1, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Periodic, occupancy=OccupancyType.Multiple, tree_threshold=100)

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

        cspace.move(a1, space.ContinuousPoint(20.5, -1.5, 6.5))
        pt = cspace.get_location(a1)
        self.assertEqual(0.5, pt.x)
        self.assertEqual(23.5, pt.y)
        self.assertEqual(1.5, pt.z)

    def test_remove(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=0, zextent=0)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, tree_threshold=100)
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

    def test_single_occ(self):
        a1 = core.Agent(1, 0)
        a2 = core.Agent(2, 0)

        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Single, tree_threshold=100)

        cspace.add(a1)

        # move after initial add
        cspace.move(a1, space.ContinuousPoint(2, 4))
        pt = cspace.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        cspace.add(a2)
        pt = cspace.move(a2, space.ContinuousPoint(2, 4))
        self.assertIsNone(pt)
        agents = []
        for a in cspace.get_agents(space.ContinuousPoint(2, 4)):
            agents.append(a)
        self.assertEqual(1, len(agents))
        self.assertEqual(a1, agents[0])

        pt = cspace.move(a2, space.ContinuousPoint(3, 4))
        self.assertIsNotNone(pt)
        self.assertEqual(3, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        pt = cspace.move(a1, space.ContinuousPoint(3, 4))
        self.assertIsNone(pt)
        pt = cspace.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        pt = cspace.move(a2, space.ContinuousPoint(4, 4))
        a = cspace.get_agent(space.ContinuousPoint(3, 4))
        self.assertIsNone(a)

        cspace.remove(a2)
        pt = cspace.get_location(a2)
        self.assertIsNone(pt)
        a = cspace.get_agent(space.ContinuousPoint(4, 4))
        self.assertIsNone(a)

        pt = cspace.move(a1, space.ContinuousPoint(4, 4))
        self.assertIsNotNone(pt)
        self.assertEqual(4, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Periodic, occupancy=OccupancyType.Single, tree_threshold=100)

        cspace.add(a1)

        # move after initial add
        cspace.move(a1, space.ContinuousPoint(2, 4))
        pt = cspace.get_location(a1)
        self.assertEqual(2, pt.x)
        self.assertEqual(4, pt.y)
        self.assertEqual(0, pt.z)

        cspace.add(a2)
        pt = cspace.move(a2, space.ContinuousPoint(2, 4))
        self.assertIsNone(pt)
        agents = []
        for a in cspace.get_agents(space.ContinuousPoint(2, 4)):
            agents.append(a)
        self.assertEqual(1, len(agents))
        self.assertEqual(a1, agents[0])

    def test_get(self):
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=-1, zextent=5)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, tree_threshold=100)

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
        count = 0
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

    def get_random_pt(self, box):
        x = random.uniform(box.xmin, box.xextent)
        y = random.uniform(box.ymin, box.yextent)
        z = 0
        if box.zextent > 0:
            z = random.uniform(box.zmin, box.zextent)

        return space.ContinuousPoint(x, y, z)

    def get_random_bounds(self, box):
        x1 = random.randint(box.xmin, box.xextent - 1)
        y1 = random.randint(box.ymin, box.yextent - 1)
        x2 = random.randint(x1 + 1, box.xextent)
        y2 = random.randint(y1 + 1, box.yextent)
        z1 = z2 = 0

        if box.zextent > 0:
            z1 = random.randint(box.zmin, box.zextent - 1)
            z2 = random.randint(z1 + 1, box.zextent)

        return space.BoundingBox(min(x1, x2), abs(x1 - x2), min(y1, y2), abs(y1 - y2),
                                 min(z1, z2), abs(z1 - z2))

    def within(self, pt, bounds):
        r = pt.x >= bounds.xmin and pt.x < bounds.xmin + bounds.xextent and pt.y >= bounds.ymin and pt.y < bounds.ymin + bounds.yextent

        if bounds.zextent > 0:
            return r and pt.z >= bounds.zmin and pt.z < bounds.zmin + bounds.zextent
        return r

    def within_test(self, box, cspace, pt_map):
        bounds = self.get_random_bounds(box)
        actual = set([a.uid for a in cspace.get_agents_within(bounds)])

        exp_within = []
        for aid, v in pt_map.items():
            if self.within(v[0], bounds):
                exp_within.append(aid)
        expected = set(exp_within)

        if len(expected) != len(actual):
            for a in (expected - actual):
                print(cspace.get_location(pt_map[a][1]))

        self.assertEqual(len(expected), len(actual))
        self.assertEqual(0, len(expected - actual))

    def test_within2D(self):
        random.seed(42)
        pt_map = {}
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=10, zextent=0)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, tree_threshold=10)

        agents = []
        for i in range(10000):
            a = core.Agent(i, 0)
            agents.append(a)
            cspace.add(a)
        for i in range(100):
            for a in agents:
                pt = self.get_random_pt(box)
                pt1 = cspace.move(a, pt)
                pt_map[a.uid] = (pt1, a)

            self.within_test(box, cspace, pt_map)
            pt_map.clear()

    def test_within3D(self):
        random.seed(42)
        pt_map = {}
        box = space.BoundingBox(xmin=0, xextent=20, ymin=0, yextent=25, zmin=0, zextent=40)
        cspace = space.ContinuousSpace("cspace", bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, tree_threshold=10)

        agents = []
        for i in range(10000):
            a = core.Agent(i, 0)
            agents.append(a)
            cspace.add(a)
        for i in range(100):
            for a in agents:
                pt = self.get_random_pt(box)
                pt1 = cspace.move(a, pt)
                pt_map[a.uid] = (pt1, a)

            self.within_test(box, cspace, pt_map)
            pt_map.clear()


if __name__ == "__main__":
    unittest.main()
