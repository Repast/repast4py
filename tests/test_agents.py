import sys
import unittest
import os

try:
    from repast4py import core
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py import core


class EAgent(core.Agent):

    def __init__(self, id, agent_type, rank, energy):
        super().__init__(id=id, type=agent_type, rank=rank)
        self.energy = energy


class AgentTests(unittest.TestCase):

    def test_agent_id(self):
        a1 = core.Agent(id=1, type=2, rank=3)
        self.assertEqual(1, a1.id)
        self.assertEqual(2, a1.type)
        self.assertEqual(3, a1.uid_rank)

        a2 = core.Agent(4, 5)
        self.assertEqual(4, a2.id)
        self.assertEqual(5, a2.type)
        self.assertEqual(0, a2.uid_rank)
        self.assertEqual((4, 5, 0), a2.uid)

    def test_subclassing(self):
        a1 = EAgent(1, 2, 3, 0)
        self.assertEqual(1, a1.id)
        self.assertEqual(2, a1.type)
        self.assertEqual(3, a1.uid_rank)

        a2 = EAgent(4, 5, 0, 1)
        self.assertEqual(4, a2.id)
        self.assertEqual(5, a2.type)
        self.assertEqual(0, a2.uid_rank)
        self.assertEqual((4, 5, 0), a2.uid)
        self.assertEqual(1, a2.energy)

    def test_local_rank(self):
        a1 = EAgent(1, 2, 3, 0)
        self.assertEqual(a1.local_rank, 3)
        a1.local_rank = 10
        self.assertEqual(10, a1.local_rank)

    def test_type_pos(self):
        self.assertRaises(ValueError, lambda: core.Agent(id=1, type=-1, rank=3))
