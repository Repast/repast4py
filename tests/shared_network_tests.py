import sys
import os
from mpi4py import MPI
import unittest

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

from repast4py.network import UndirectedSharedNetwork, DirectedSharedNetwork
from repast4py import core

# run with -n 9


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


class SharedNetworkTests(unittest.TestCase):

    long_message = True

    def test_add_remove_d(self):
        # make 1 rank comm for basic add remove tests
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            g = DirectedSharedNetwork('network', comm)
            self.assertEqual('network', g.name)
            self.assertTrue(g.is_directed)
            self.assertEqual(0, g.node_count)
            self.assertEqual(0, g.edge_count)

            agents = [EAgent(x, 0, comm.Get_rank(), x) for x in range(10)]
            g.add(agents[0])
            g.add_nodes(agents[1:4])
            self.assertEqual(4, g.node_count)
            nodes = [x for x in g.graph.nodes]
            self.assertEqual(nodes, [x for x in agents[0:4]])

            g.add_edge(agents[0], agents[1])
            g.add_edge(agents[0], agents[3])
            g.add_edge(agents[5], agents[6], weight=12)
            # 2 nodes added via edge
            self.assertEqual(6, g.node_count)
            self.assertEqual(3, g.edge_count)
            edges = [x for x in g.graph.edges(agents[5])]
            self.assertEqual(edges, [(agents[5], agents[6])])
            edges = [x for x in g.graph.edges(agents[0])]
            self.assertEqual(edges, [(agents[0], agents[1]), (agents[0], agents[3])])
            edges = [x for x in g.graph.edges(agents[6])]
            self.assertEqual(edges, [])
            edge = g.graph.edges[agents[5], agents[6]]
            self.assertEqual(12, edge['weight'])

            self.assertTrue(g.contains_edge(agents[0], agents[1]))
            self.assertFalse(g.contains_edge(agents[1], agents[0]))
            self.assertTrue(not g.contains_edge(agents[7], agents[6]))

            g.remove(agents[0])
            self.assertEqual(5, g.node_count)
            self.assertEqual(1, g.edge_count)

    def test_add_remove_u(self):
        # make 1 rank comm for basic add remove tests
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create_group(new_group)

        if comm != MPI.COMM_NULL:
            g = UndirectedSharedNetwork('network', comm)
            self.assertEqual('network', g.name)
            self.assertFalse(g.is_directed)
            self.assertEqual(0, g.node_count)
            self.assertEqual(0, g.edge_count)

            agents = [EAgent(x, 0, comm.Get_rank(), x) for x in range(10)]
            g.add(agents[0])
            g.add_nodes(agents[1:4])
            self.assertEqual(4, g.node_count)
            nodes = [x for x in g.graph.nodes]
            self.assertEqual(nodes, [x for x in agents[0:4]])

            g.add_edge(agents[0], agents[1])
            g.add_edge(agents[0], agents[3])
            g.add_edge(agents[5], agents[6], weight=12)
            # 2 nodes added via edge
            self.assertEqual(6, g.node_count)
            self.assertEqual(3, g.edge_count)
            edges = [x for x in g.graph.edges(agents[5])]
            self.assertEqual(edges, [(agents[5], agents[6])])
            edges = [x for x in g.graph.edges(agents[0])]
            self.assertEqual(edges, [(agents[0], agents[1]), (agents[0], agents[3])])
            edges = [x for x in g.graph.edges(agents[6])]
            self.assertEqual(edges, [(agents[6], agents[5])])
            edge = g.graph.edges[agents[5], agents[6]]
            self.assertEqual(12, edge['weight'])

            self.assertTrue(g.contains_edge(agents[0], agents[1]))
            self.assertTrue(g.contains_edge(agents[1], agents[0]))
            self.assertTrue(not g.contains_edge(agents[7], agents[6]))

            g.remove(agents[0])
            self.assertEqual(5, g.node_count)
            self.assertEqual(1, g.edge_count)

