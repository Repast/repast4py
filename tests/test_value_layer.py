import unittest
import sys
import os
import numpy as np
import torch


try:
    from repast4py import geometry
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py import geometry

from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType
from repast4py.geometry import BoundingBox
from repast4py.value_layer import ValueLayer


class GeometryTests(unittest.TestCase):

    def test_sticky_ngh(self):
        # 1d
        bounds = np.array([0, 9])
        pt = dpt(4, 0, 0)
        nghs = geometry.find_1d_nghs_sticky(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([3, 4, 5])))
        pt = dpt(0, 0, 0)
        nghs = geometry.find_1d_nghs_sticky(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([0, 1])))
        pt = dpt(9, 0, 0)
        nghs = geometry.find_1d_nghs_sticky(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([8, 9])))

        bounds = np.array([5, 9])
        pt = dpt(4, 0, 0)
        nghs = geometry.find_1d_nghs_sticky(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([5])), msg=nghs)

        bounds = np.array([-1, 9])
        pt = dpt(0, 0, 0)
        nghs = geometry.find_1d_nghs_sticky(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([-1, 0, 1])), msg=nghs)

        # 2d
        bounds = np.array([0, 9, -2, 12])
        pt = dpt(4, 1, 0)
        nghs = geometry.find_2d_nghs_sticky(pt.coordinates, bounds)
        xs = np.tile(np.array([3, 4, 5]), 3)
        ys = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys])), msg=nghs)

        pt = dpt(0, 0, 0)
        nghs = geometry.find_2d_nghs_sticky(pt.coordinates, bounds)
        xs = np.array([0, 1, 0, 1, 0, 1])
        ys = np.array([-1, -1, 0, 0, 1, 1])
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys])), msg=nghs)

        pt = dpt(0, 12, 0)
        nghs = geometry.find_2d_nghs_sticky(pt.coordinates, bounds)
        xs = np.array([0, 1, 0, 1])
        ys = np.array([11, 11, 12, 12])
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys])), msg=nghs)

        # 3d
        bounds = np.array([-1, 9, -2, 12, -4, 22])
        pt = dpt(4, 10, 18)
        nghs = geometry.find_3d_nghs_sticky(pt.coordinates, bounds)
        xs = np.array([3, 4, 5] * 9)
        ys = np.array([9, 9, 9, 10, 10, 10, 11, 11, 11] * 3)
        zs = np.array([17] * 9 + [18] * 9 + [19] * 9)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)

        pt = dpt(-1, -2, -4)
        nghs = geometry.find_3d_nghs_sticky(pt.coordinates, bounds)
        xs = np.array([-1, 0] * 4)
        ys = np.array([-2, -2, -1, -1] * 2)
        zs = np.array([-4] * 4 + [-3] * 4)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)

        pt = dpt(-1, -2, -3)
        nghs = geometry.find_3d_nghs_sticky(pt.coordinates, bounds)
        xs = np.array([-1, 0] * 6)
        ys = np.array([-2, -2, -1, -1] * 3)
        zs = np.array([-4] * 4 + [-3] * 4 + [-2] * 4)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)

        pt = dpt(9, 12, 22)
        nghs = geometry.find_3d_nghs_sticky(pt.coordinates, bounds)
        xs = np.array([8, 9] * 4)
        ys = np.array([11, 11, 12, 12] * 2)
        zs = np.array([21] * 4 + [22] * 4)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)

    def test_periodic_nghs(self):
        # 1d
        bounds = np.array([0, 9])
        pt = dpt(4, 0, 0)
        nghs = geometry.find_1d_nghs_periodic(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([3, 4, 5])))
        pt = dpt(0, 0, 0)
        nghs = geometry.find_1d_nghs_periodic(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([9, 0, 1])))
        pt = dpt(9, 0, 0)
        nghs = geometry.find_1d_nghs_periodic(pt.coordinates, bounds)
        self.assertTrue(np.array_equal(nghs, np.array([8, 9, 0])))

        # 2d
        bounds = np.array([0, 9, -2, 12])
        pt = dpt(4, 1, 0)
        nghs = geometry.find_2d_nghs_periodic(pt.coordinates, bounds)
        xs = np.tile(np.array([3, 4, 5]), 3)
        ys = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys])), msg=nghs)

        pt = dpt(0, 0, 0)
        nghs = geometry.find_2d_nghs_periodic(pt.coordinates, bounds)
        xs = np.array([9, 0, 1, 9, 0, 1, 9, 0, 1])
        ys = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys])), msg=nghs)

        pt = dpt(0, 12, 0)
        nghs = geometry.find_2d_nghs_periodic(pt.coordinates, bounds)
        xs = np.array([9, 0, 1, 9, 0, 1, 9, 0, 1])
        ys = np.array([11, 11, 11, 12, 12, 12, -2, -2, -2])
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys])), msg=nghs)

        pt = dpt(9, -2, 0)
        nghs = geometry.find_2d_nghs_periodic(pt.coordinates, bounds)
        xs = np.array([8, 9, 0, 8, 9, 0, 8, 9, 0])
        ys = np.array([12, 12, 12, -2, -2, -2, -1, -1, -1])
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys])), msg=nghs)

        # 3d
        bounds = np.array([-1, 9, -2, 12, -4, 22])
        pt = dpt(4, 10, 18)
        nghs = geometry.find_3d_nghs_periodic(pt.coordinates, bounds)
        xs = np.array([3, 4, 5] * 9)
        ys = np.array([9, 9, 9, 10, 10, 10, 11, 11, 11] * 3)
        zs = np.array([17] * 9 + [18] * 9 + [19] * 9)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)

        pt = dpt(-1, -2, -4)
        nghs = geometry.find_3d_nghs_periodic(pt.coordinates, bounds)
        xs = np.array([9, -1, 0] * 9)
        ys = np.array([12, 12, 12, -2, -2, -2, -1, -1, -1] * 3)
        zs = np.array([22] * 9 + [-4] * 9 + [-3] * 9)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)

        pt = dpt(-1, -2, -3)
        nghs = geometry.find_3d_nghs_periodic(pt.coordinates, bounds)
        xs = np.array([9, -1, 0] * 9)
        ys = np.array([12, 12, 12, -2, -2, -2, -1, -1, -1] * 3)
        zs = np.array([-4] * 9 + [-3] * 9 + [-2] * 9)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)

        pt = dpt(9, 12, 22)
        nghs = geometry.find_3d_nghs_periodic(pt.coordinates, bounds)
        xs = np.array([8, 9, -1] * 9)
        ys = np.array([11, 11, 11, 12, 12, 12, -2, -2, -2] * 3)
        zs = np.array([21] * 9 + [22] * 9 + [-4] * 9)
        self.assertTrue(np.array_equal(nghs, np.array([xs, ys, zs])), msg=nghs)


class ValueLayerTests(unittest.TestCase):

    def test_accesors(self):
        bounds = BoundingBox(xmin=0, xextent=20, ymin=0,
                             yextent=0, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Sticky, 12.1)
        self.assertFalse(torch.any(torch.ne(vl.grid, 12.1)))

        pt = dpt(0, 0, 0)
        self.assertEqual(12.1, vl.get(pt))
        vl.set(pt, 24.1)
        self.assertEqual(24.1, vl.get(pt))
        # all 24.1s to 1, and rest to 0, so sum should be 1 if only a single 24.1
        s = torch.where(vl.grid == 24.1, torch.tensor(1), torch.tensor(0)).sum()
        self.assertEqual(1, s)

        bounds = BoundingBox(xmin=15, xextent=20, ymin=0,
                             yextent=30, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Sticky, 4.2)
        sz = vl.grid.size()
        self.assertEqual(bounds.yextent, sz[0])
        self.assertEqual(bounds.xextent, sz[1])
        self.assertFalse(torch.any(torch.ne(vl.grid, 4.2)))

        pt._reset2D(18, 24)
        vl.set(pt, 0.2)
        self.assertEqual(0.2, vl.get(pt))
        s = torch.where(vl.grid == 0.2, torch.tensor(1), torch.tensor(0)).sum()
        self.assertEqual(1, s)

        bounds = BoundingBox(xmin=15, xextent=20, ymin=0,
                             yextent=30, zmin=3, zextent=11)
        vl = ValueLayer(bounds, BorderType.Sticky, 4.2)
        sz = vl.grid.size()
        self.assertEqual(bounds.zextent, sz[0])
        # row major
        self.assertEqual(bounds.yextent, sz[1])
        self.assertEqual(bounds.xextent, sz[2])
        self.assertFalse(torch.any(torch.ne(vl.grid, 4.2)))

        pt._reset3D(18, 24, 5)
        vl.set(pt, 0.2)
        self.assertEqual(0.2, vl.get(pt))
        s = torch.where(vl.grid == 0.2, torch.tensor(1), torch.tensor(0)).sum()
        self.assertEqual(1, s)

    def test_borders(self):
        # 3D is enough to test all, given that borders have been tested separately
        bounds = BoundingBox(xmin=15, xextent=20, ymin=0,
                             yextent=30, zmin=-1, zextent=11)
        vl = ValueLayer(bounds, BorderType.Periodic, 4.2)
        sz = vl.grid.size()
        self.assertEqual(bounds.zextent, sz[0])
        # row major
        self.assertEqual(bounds.yextent, sz[1])
        self.assertEqual(bounds.xextent, sz[2])
        self.assertFalse(torch.any(torch.ne(vl.grid, 4.2)))

        pt = dpt(14, -2, -3)
        vl.set(pt, 0.2)
        self.assertEqual(0.2, vl.get(pt))

        pt._reset3D(34, 28, 8)
        self.assertEqual(0.2, vl.get(pt))

        pt = dpt(36, 32, 14)
        vl.set(pt, 0.1)
        self.assertEqual(0.1, vl.get(pt))

        pt._reset3D(16, 2, 3)
        self.assertEqual(0.1, vl.get(pt))

    def test_ngh_sticky(self):
        bounds = BoundingBox(xmin=12, xextent=20, ymin=0, yextent=0, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Sticky, 'random')
        pt = dpt(14, 0, 0)
        exp_pts = np.array([13, 14, 15])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[i], 0, 0)))

        pt = dpt(12, 0, 0)
        exp_pts = np.array([12, 13])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts))
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[i], 0, 0)))

        pt = dpt(31, 0, 0)
        exp_pts = np.array([30, 31])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts))
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[i], 0, 0)))

        bounds = BoundingBox(xmin=12, xextent=20, ymin=-3,
                             yextent=13, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Sticky, 'random')
        pt = dpt(14, -1, 0)
        exp_pts = np.array([[13, 14, 15, 13, 14, 15, 13, 14, 15],
                           [-2, -2, -2, -1, -1, -1, 0, 0, 0]])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[0][i], pts[1][i], 0)))

        # 2D
        bounds = BoundingBox(xmin=12, xextent=20, ymin=-3,
                             yextent=13, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Sticky, 'random')
        pt = dpt(31, 9, 0)
        exp_pts = np.array([[30, 31, 30, 31],
                            [8, 8, 9, 9]])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[0][i], pts[1][i], 0)))

        # 3D
        bounds = BoundingBox(xmin=12, xextent=20, ymin=-3,
                             yextent=13, zmin=1, zextent=22)
        vl = ValueLayer(bounds, BorderType.Sticky, 'random')
        pt = dpt(30, 8, 12)
        exp_pts = np.array([[29, 30, 31] * 9,
                           [7, 7, 7, 8, 8, 8, 9, 9, 9] * 3,
                           [11] * 9 + [12] * 9 + [13] * 9])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[0][i], pts[1][i], pts[2][i])))

    def test_ngh_periodic(self):
        bounds = BoundingBox(xmin=12, xextent=20, ymin=0,
                             yextent=0, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Periodic, 'random')
        pt = dpt(14, 0, 0)
        exp_pts = np.array([13, 14, 15])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[i], 0, 0)))

        pt = dpt(12, 0, 0)
        exp_pts = np.array([31, 12, 13])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts))
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[i], 0, 0)))

        pt = dpt(31, 0, 0)
        exp_pts = np.array([30, 31, 12])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts))
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[i], 0, 0)))

        # 2D
        bounds = BoundingBox(xmin=12, xextent=20, ymin=-3,
                             yextent=13, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Periodic, 'random')
        pt = dpt(14, -1, 0)
        exp_pts = np.array([[13, 14, 15, 13, 14, 15, 13, 14, 15],
                            [-2, -2, -2, -1, -1, -1, 0, 0, 0]])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[0][i], pts[1][i], 0)))

        bounds = BoundingBox(xmin=12, xextent=20, ymin=-3,
                             yextent=13, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Periodic, 'random')
        pt = dpt(31, 9, 0)
        exp_pts = np.array([[30, 31, 12, 30, 31, 12, 30, 31, 12],
                            [8, 8, 8, 9, 9, 9, -3, -3, -3]])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[0][i], pts[1][i], 0)))

        # 3D
        bounds = BoundingBox(xmin=12, xextent=20, ymin=-3,
                             yextent=13, zmin=1, zextent=22)
        vl = ValueLayer(bounds, BorderType.Periodic, 'random')
        pt = dpt(12, 9, 1)
        exp_pts = np.array([[31, 12, 13] * 9,
                           [8, 8, 8, 9, 9, 9, -3, -3, -3] * 3,
                           [22] * 9 + [1] * 9 + [2] * 9])
        vals, pts = vl.get_nghs(pt)
        self.assertTrue(np.array_equal(pts, exp_pts), pts)
        for i, val in enumerate(vals):
            self.assertEqual(val, vl.get(dpt(pts[0][i], pts[1][i], pts[2][i])))
