import sys
import unittest
import os

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
        self.assertEqual(3, a1.rank)

        a2 = core.Agent(4, 5)
        self.assertEqual(4, a2.id)
        self.assertEqual(5, a2.type)
        self.assertEqual(0, a2.rank)
        self.assertEqual((4, 5, 0), a2.tag)

    def test_subclassing(self):
        a1 = EAgent(1, 2, 3, 0)
        self.assertEqual(1, a1.id)
        self.assertEqual(2, a1.type)
        self.assertEqual(3, a1.rank)

        a2 = EAgent(4, 5, 0, 1)
        self.assertEqual(4, a2.id)
        self.assertEqual(5, a2.type)
        self.assertEqual(0, a2.rank)
        self.assertEqual((4, 5, 0), a2.tag)
        self.assertEqual(1, a2.energy)



            


