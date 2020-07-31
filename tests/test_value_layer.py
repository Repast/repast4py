import unittest
import sys
import os
import numpy as np
import random

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, BoundingBox
from repast4py.value_layer import ValueLayer

class ValueLayerTests(unittest.TestCase):

    def test_get(self):
        bounds = BoundingBox(xmin=0, xextent=20, ymin=0,
                                   yextent=0, zmin=0, zextent=0)
        vl = ValueLayer(bounds, BorderType.Sticky, 12.1)
        pt = dpt(0, 0, 0)
        self.assertEqual(12.1, vl.get(pt))

        vl.set(pt, 24.1)
        self.assertEqual(24.1, vl.get(pt))

        # TODO test 2D

        

