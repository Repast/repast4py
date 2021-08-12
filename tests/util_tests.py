
import sys
import unittest
import os
import json

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

from repast4py.util import find_free_filename
from repast4py.parameters import init_params
from repast4py import parameters
from repast4py import random


class UtilTests(unittest.TestCase):

    def test_freefile(self):
        p = find_free_filename('./test_data/a_file.csv')
        self.assertEqual('test_data/a_file.csv', str(p))

        # file exists
        p = find_free_filename('./test_data/test_file.csv')
        self.assertEqual('test_data/test_file_1.csv', str(p))

        # file exists, no ext
        p = find_free_filename('./test_data/test_file')
        self.assertEqual('test_data/test_file_1', str(p))

    def test_parse_params(self):
        params = init_params('./test_data/test_params.yaml', '')
        self.assertEqual(3, len(params))
        self.assertEqual(4, params['x'])
        self.assertEqual('ffc', params['y'])
        self.assertEqual(6, params['random.seed'])
        self.assertEqual(6, random.seed)

        j_params = {'x': 201, 'random.seed': 42, 'z': 'Z'}
        init_params('./test_data/test_params.yaml', json.dumps(j_params))
        params = parameters.params
        self.assertEqual(4, len(params))
        self.assertEqual(201, params['x'])
        self.assertEqual('Z', params['z'])
        self.assertEqual('ffc', params['y'])
        self.assertEqual(42, params['random.seed'])
        self.assertEqual(42, random.seed)
