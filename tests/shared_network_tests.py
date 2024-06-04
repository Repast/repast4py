import sys
import os
from mpi4py import MPI
import unittest
from collections import OrderedDict
import networkx as nx
import re

try:
    from repast4py.network import UndirectedSharedNetwork, DirectedSharedNetwork, read_network, write_network
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py.network import UndirectedSharedNetwork, DirectedSharedNetwork, read_network, write_network

from repast4py import core, space, random
from repast4py import context as ctx

from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType


class EAgent(core.Agent):

    def __init__(self, id, agent_type, rank, energy):
        super().__init__(id=id, type=agent_type, rank=rank)
        self.energy = energy
        self.restored = False

    def save(self):
        return (self.uid, self.energy)

    def update(self, data):
        self.restored = True
        self.energy = data


def restore_agent(agent_data):
    # agent_data: [aid_tuple, energy]
    uid = agent_data[0]
    return EAgent(uid[0], uid[1], uid[2], agent_data[1])


class SharedDirectedNetworkTests(unittest.TestCase):

    long_message = True

    def test_add_remove(self):
        # make 1 rank comm for basic add remove tests
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create(new_group)

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

            g.add_edge(agents[4], agents[5])
            self.assertEqual(2, g.num_edges(agents[5]))
            self.assertEqual(1, g.num_edges(agents[4]))

            exp = {(agents[5], agents[6]), (agents[4], agents[5])}
            for edge in g._edges(agents[5]):
                exp.remove(edge)
            self.assertEqual(0, len(exp))

    def test_sync_1(self):
        # Tests add, update and remove edge
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        context = ctx.SharedContext(comm)
        g = DirectedSharedNetwork('network', comm)
        context.add_projection(g)
        self.assertEqual(0, g.node_count)

        agents = [EAgent(x, 0, rank, x) for x in range(10)]
        for a in agents:
            context.add(a)
            self.assertEqual(rank, a.local_rank)

        self.assertEqual(10, g.node_count)

        requests = []
        if rank == 0:
            requests.append(((1, 0, 1), 1))
            requests.append(((1, 0, 2), 2))
            requests.append(((2, 0, 1), 1))
        elif rank == 3:
            requests.append(((1, 0, 0), 0))
            requests.append(((4, 0, 2), 2))
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0 or rank == 3:
            self.assertEqual(13, g.node_count)

        # Edges: 0: (0, 0, 0) -> (1, 0, 1)
        #        0: (2, 0, 1) -> (0, 0, 0)
        #        1: (0, 0, 1) -> (1, 0, 1)
        #        3: (0, 0, 3) -> (2, 0, 1)
        #        3: (1, 0, 0) -> (1, 0, 3)
        if rank == 0:
            other = context.ghost_agent((1, 0, 1))
            g.add_edge(agents[0], other, weight=2)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 2})])

            other = context.ghost_agent((2, 0, 1))
            g.add_edge(other, agents[0], rate=2)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.in_edges(agents[0], data=True)]
            self.assertEqual(edges[0], (other, agents[0], {'rate': 2}))
        elif rank == 1:
            g.add_edge(agents[0], agents[1], weight=3)
            self.assertEqual(1, g.edge_count)

        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.add_edge(agents[0], other, weight=10)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 10})])

            other = context.ghost_agent((1, 0, 0))
            g.add_edge(other, agents[1], rate=2.1)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.in_edges(agents[1], data=True)]
            self.assertEqual(edges[0], (other, agents[1], OrderedDict({'rate': 2.1})))

        context.synchronize(restore_agent)

        # TEST vertices and edges created on ghost ranks
        if rank == 0:
            self.assertEqual(3, g.edge_count)
            agent = context.agent((0, 0, 0))
            # get out and in, in that order
            edges = [x for x in g._edges(agent, data=True)]
            self.assertEqual(2, len(edges))
            exp = [(agent, context.ghost_agent((1, 0, 1)), {'weight': 2}),
                   (context.ghost_agent((2, 0, 1)), agent, {'rate': 2})]
            for e in edges:
                self.assertTrue(e in exp)

        elif rank == 1:
            self.assertEqual(4, g.edge_count)
            agent = context.agent((1, 0, 1))
            edges = [x for x in g._edges(agent, data=True)]
            self.assertEqual(2, len(edges))
            # looks like in undirected graph when request via edges(k)
            # the edge is returned as (k, other) regardless of how the
            # edge is originally inserted
            exp = [(agents[0], agent, {'weight': 3}),
                   (context.ghost_agent((0, 0, 0)), agent, {'weight': 2})]
            self.assertEqual(edges, exp)

            agent = context.agent((2, 0, 1))
            edges = [x for x in g._edges(agent, data=True)]
            self.assertEqual(2, len(edges))
            exp = [(agent, context.ghost_agent((0, 0, 0)), {'rate': 2}),
                   (context.ghost_agent((0, 0, 3)), agent, {'weight': 10})]
            self.assertEqual(edges, exp)

        elif rank == 3:
            self.assertEqual(2, g.edge_count)
            agent = agents[0]
            edges = [x for x in g.graph.edges(agent, data=True)]
            self.assertEqual(1, len(edges))
            self.assertEqual(edges[0], (agent, context.ghost_agent((2, 0, 1)), {'weight': 10}))

            agent = agents[1]
            edges = [x for x in g.graph.in_edges(agent, data=True)]
            self.assertEqual(1, len(edges))
            self.assertEqual(edges[0], (context.ghost_agent((1, 0, 0)), agent, {'rate': 2.1}))

        # TEST: update 2,0,1 agent and see if updates on 0 and 3
        if rank == 1:
            agents[2].energy = 134324

        # TEST: update edge with 2,0,1 and see if updates 1 and 0
        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.update_edge(agents[0], other, weight=14.2)

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertEqual(134324, context.ghost_agent((2, 0, 1)).energy)
        elif rank == 1:
            self.assertEqual(14.2, g.graph.edges[context.ghost_agent((0, 0, 3)), agents[2]]['weight'])
        elif rank == 3:
            self.assertEqual(134324, context.ghost_agent((2, 0, 1)).energy)

        # TEST: remove edge 0: (2, 0, 1) -> (0, 0, 0)
        #                   3: (0, 0, 3) -> (2, 0, 1)
        if rank == 0:
            other = context.ghost_agent((2, 0, 1))
            g.remove_edge(other, agents[0])
            self.assertEqual(2, g.edge_count)
        elif rank == 1:
            self.assertIsNotNone(context.ghost_agent((0, 0, 3)))
        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.remove_edge(agents[0], other)

        context.synchronize(restore_agent)

        # TEST: 3 removed (0, 0, 3) -> (2, 0, 1), so
        # that edge should be removed, and (0, 0, 3) no
        # longer ghosted to 1
        if rank == 1:
            self.assertEqual(2, g.edge_count)
            self.assertTrue((agents[0], agents[1]) in g.graph.edges())
            self.assertTrue((context.ghost_agent((0, 0, 0)), agents[1]) in g.graph.edges())
            self.assertIsNone(context.ghost_agent((0, 0, 3)))

    def test_sync_2(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        context = ctx.SharedContext(comm)
        g = DirectedSharedNetwork('network', comm)
        context.add_projection(g)
        self.assertEqual(0, g.node_count)

        agents = [EAgent(x, 0, rank, x) for x in range(10)]
        for a in agents:
            context.add(a)
            self.assertEqual(rank, a.local_rank)

        self.assertEqual(10, g.node_count)

        requests = []
        if rank == 0:
            requests.append(((1, 0, 1), 1))
            requests.append(((1, 0, 2), 2))
            requests.append(((2, 0, 1), 1))
        elif rank == 3:
            requests.append(((1, 0, 0), 0))
            requests.append(((4, 0, 2), 2))
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0 or rank == 3:
            self.assertEqual(13, g.node_count)

        # Create edges with requested vertices
        # Edges: 0: (0, 0, 0) -> (1, 0, 1)
        #        0: (2, 0, 1) -> (0, 0, 0)
        #        1: (0, 0, 1) -> (1, 0, 1)
        #        3: (0, 0, 3) -> (2, 0, 1)
        #        3: (1, 0, 0) -> (1, 0, 3)
        if rank == 0:
            other = context.ghost_agent((1, 0, 1))
            g.add_edge(agents[0], other, weight=2)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 2})])

            other = context.ghost_agent((2, 0, 1))
            g.add_edge(other, agents[0], rate=2)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.in_edges(agents[0], data=True)]
            self.assertEqual(edges[0], (other, agents[0], {'rate': 2}))
        elif rank == 1:
            g.add_edge(agents[0], agents[1], weight=3)
            self.assertEqual(1, g.edge_count)

        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.add_edge(agents[0], other, weight=10)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 10})])

            other = context.ghost_agent((1, 0, 0))
            g.add_edge(other, agents[1], rate=2.1)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.in_edges(agents[1], data=True)]
            self.assertEqual(edges[0], (other, agents[1], {'rate': 2.1}))

        context.synchronize(restore_agent)

        # TEST: remove (1, 0, 1) from 1
        # edges should be removed across processes
        if rank == 1:
            context.remove(agents[1])
            self.assertEqual(2, g.edge_count)
            self.assertTrue((agents[2], context.ghost_agent((0, 0, 0))) in g.graph.edges())
            self.assertTrue((context.ghost_agent((0, 0, 3)), agents[2]) in g.graph.edges())
            self.assertTrue((1, 0, 1) not in context._agent_manager._ghosted_agents)

        context.synchronize(restore_agent)

        # (0, 0, 0) -> (1, 0, 1) removed
        if rank == 0:
            # 1, 0, 1 no longer ghosted to 0
            self.assertTrue((1, 0, 1) not in context._agent_manager._ghost_agents)
            self.assertEqual(2, len(g.graph.edges()))
            self.assertTrue((context.ghost_agent((2, 0, 1)), agents[0]) in g.graph.edges())
            self.assertTrue((agents[1], context.ghost_agent((1, 0, 3))) in g.graph.edges())

        # Add edge on 1 then move one of those
        if rank == 1:
            g.add_edge(agents[2], agents[4], weight=42)
            # print(g.graph.nodes())
            # print(g.graph.edges())

        # print(rank, g.graph.edges(), flush=True)

        # TEST: Move (2, 0, 1) to 0
        # * local to 0
        # * ghosted from 0 to 3 and 1
        moved = []
        if rank == 1:
            # move 2,0,1 to o
            moved.append(((2, 0, 1), 0))

        context.move_agents(moved, restore_agent)

        # 2,0,1 is on 0 with local_rank of 0
        if rank == 0:
            self.assertIsNotNone(context.agent((2, 0, 1)))
            self.assertEqual(0, context.agent((2, 0, 1)).local_rank)
            self.assertTrue(g.graph.has_edge(context.agent((2, 0, 1)), context.agent((0, 0, 0))))
            self.assertTrue((2, 0, 1) in context._agent_manager._ghosted_agents)
            ghosted_to_ranks = context._agent_manager._ghosted_agents[(2, 0, 1)].ghost_ranks
            self.assertTrue(3 in ghosted_to_ranks)
            self.assertTrue(g.graph.has_edge(context.ghost_agent((0, 0, 3)), context.agent((2, 0, 1))))

            self.assertTrue(1 in ghosted_to_ranks)
            self.assertTrue(g.graph.has_edge(context.agent((2, 0, 1)), context.ghost_agent((4, 0, 1))))
            self.assertEqual(42, g.graph.edges[context.agent((2, 0, 1)),
                                               context.ghost_agent((4, 0, 1))]['weight'])

            # original 10 + moved 2,0,1 + requested 1,0,2 + 0,0,3 and 0,0,0 and 1,0,3 through
            # synchronized edges
            self.assertEqual(15, g.node_count)

        elif rank == 1:
            # print(g.graph.edges())
            # 2,0,1 moved
            self.assertIsNone(context.agent((2, 0, 1)))
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))
            # these were ghost agents through edges with 2,0,1
            # but should now be removed
            self.assertIsNone(context.ghost_agent((0, 0, 3)))
            self.assertIsNone(context.ghost_agent((0, 0, 0)))
            self.assertFalse(g.graph.has_edge(context.ghost_agent((4, 0, 1)), context.agent((0, 0, 3))))
            self.assertFalse(g.graph.has_edge(context.ghost_agent((4, 0, 1)), context.agent((0, 0, 0))))
            # ghosted from 0 to 1 because of 2,0,1 -> 4,0,1 edge
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))
            self.assertTrue(g.graph.has_edge(context.ghost_agent((2, 0, 1)), context.agent((4, 0, 1))))
        elif rank == 3:
            self.assertIsNone(context.agent((2, 0, 1)))
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))

        # Test: update edge(0,0,3 - 2,0,1) with new data
        #       ghost edge on 0 reflects change
        if rank == 3:
            g.update_edge(context.agent((0, 0, 3)), context.ghost_agent((2, 0, 1)), weight=12)
            self.assertEqual(12, g.graph.edges[context.agent((0, 0, 3)),
                                               context.ghost_agent((2, 0, 1))]['weight'])

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertEqual(12, g.graph.edges[context.ghost_agent((0, 0, 3)),
                                               context.agent((2, 0, 1))]['weight'])

    def test_with_oob(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = []
        for i in range(20):
            a = EAgent(i, 0, rank, 1)
            agents.append(a)
            context.add(a)

        box = space.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
        net = DirectedSharedNetwork('network', comm)
        context.add_projection(cspace)
        context.add_projection(grid)
        context.add_projection(net)

        random.init(42)
        bounds = grid.get_local_bounds()
        xs = random.default_rng.integers(low=bounds.xmin, high=bounds.xmin + bounds.xextent, size=20)
        ys = random.default_rng.integers(low=bounds.ymin, high=bounds.ymin + bounds.yextent, size=20)
        for i, agent in enumerate(agents):
            grid.move(agent, dpt(xs[i], ys[i]))
            cspace.move(agent, cpt(xs[i], ys[i]))

        # TEST:
        # 1. request agents from neighboring ranks
        # 2. make edges between local agent and ghosts
        # 3. move agents oob, such that
        #    a. former ghost is now local
        #    b. ghost is now on different rank
        # 4. do tests

        requests = []
        if rank == 0:
            requests.append(((1, 0, 1), 1))
            requests.append(((1, 0, 2), 2))
            requests.append(((2, 0, 1), 1))
        elif rank == 3:
            requests.append(((1, 0, 0), 0))
            requests.append(((4, 0, 2), 2))
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0:
            net.add_edge(agents[1], context.ghost_agent((2, 0, 1)), color='red')
            net.add_edge(agents[10], context.ghost_agent((1, 0, 1)))
        elif rank == 1:
            net.add_edge(agents[2], agents[1])

        elif rank == 3:
            net.add_edge(agents[1], context.ghost_agent((1, 0, 0)))
            net.add_edge(agents[5], context.ghost_agent((2, 0, 1)))

        context.synchronize(restore_agent)

        # TESTS edges
        if rank == 0:
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[10], context.ghost_agent((1, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((1, 0, 3)), agents[1]))

        elif rank == 1:
            self.assertEqual(4, net.edge_count)
            self.assertTrue(net.graph.has_edge(context.ghost_agent((10, 0, 0)), agents[1], ))
            self.assertTrue(net.graph.has_edge(agents[2], agents[1]))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((1, 0, 0)), agents[2]))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((5, 0, 3)), agents[2]))
            self.assertEqual('red', net.graph.edges[context.ghost_agent((1, 0, 0)), agents[2]]['color'])
        elif rank == 2:
            self.assertEqual(0, net.edge_count)
        elif rank == 3:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((1, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[5], context.ghost_agent((2, 0, 1))))

        # Bounds:
        # print('{}: bounds: {}'.format(rank, grid.get_local_bounds()), flush=True)
        # 0: bounds: BoundingBox(xmin=0, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 1: bounds: BoundingBox(xmin=0, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)
        # 2: bounds: BoundingBox(xmin=45, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 3: bounds: BoundingBox(xmin=45, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)

        # Move (2, 0, 1) to 2's bounds
        if rank == 1:
            grid.move(agents[2], dpt(46, 35))
            cspace.move(agents[2], cpt(46.2, 35.1))

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[10], context.ghost_agent((1, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((1, 0, 3)), agents[1]))
        elif rank == 1:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(context.ghost_agent((10, 0, 0)), agents[1]))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((2, 0, 1)), agents[1]))
        elif rank == 2:
            agent_201 = context.agent((2, 0, 1))
            self.assertIsNotNone(agent_201)
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(context.ghost_agent((1, 0, 0)), agent_201))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((5, 0, 3)), agent_201))
            self.assertTrue(net.graph.has_edge(agent_201, context.ghost_agent((1, 0, 1))))
            self.assertEqual('red', net.graph.edges[context.ghost_agent((1, 0, 0)), agent_201]['color'])
        elif rank == 3:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((1, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[5], context.ghost_agent((2, 0, 1))))

        if rank == 0:
            agents[1].energy = 101
        elif rank == 2:
            # 201 to 3
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(46, 80))
            cspace.move(agent_201, cpt(46.2, 80.1))

        context.synchronize(restore_agent)

        # print(f'{rank}: {net.graph.edges()}', flush=True)

        if rank == 0:
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[10], context.ghost_agent((1, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((1, 0, 3)), agents[1]))
        elif rank == 1:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(context.ghost_agent((10, 0, 0)), agents[1]))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((2, 0, 1)), agents[1]))
        elif rank == 2:
            self.assertEqual(0, net.edge_count)
        elif rank == 3:
            self.assertEqual(4, net.edge_count)
            agent_201 = context.agent((2, 0, 1))
            agent_100 = context.ghost_agent((1, 0, 0))
            self.assertIsNotNone(agent_201)
            self.assertTrue(net.graph.has_edge(agents[1], agent_100))
            self.assertTrue(net.graph.has_edge(agents[5], agent_201))
            self.assertTrue(net.graph.has_edge(agent_201, context.ghost_agent((1, 0, 1))))
            self.assertTrue(net.graph.has_edge(agent_100, agent_201))

            self.assertEqual(101, context.ghost_agent((1, 0, 0)).energy)

    def test_in_buffer(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = []
        for i in range(20):
            a = EAgent(i, 0, rank, 1)
            agents.append(a)
            context.add(a)

        box = space.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
        net = DirectedSharedNetwork('network', comm)
        context.add_projection(net)
        context.add_projection(cspace)
        context.add_projection(grid)

        random.init(42)
        bounds = grid.get_local_bounds()
        xs = random.default_rng.integers(low=bounds.xmin, high=bounds.xmin + bounds.xextent, size=20)
        ys = random.default_rng.integers(low=bounds.ymin, high=bounds.ymin + bounds.yextent, size=20)
        for i, agent in enumerate(agents):
            grid.move(agent, dpt(xs[i], ys[i]))
            cspace.move(agent, cpt(xs[i], ys[i]))

        # Bounds:
        # print('{}: bounds: {}'.format(rank, grid.get_local_bounds()))
        # 0: bounds: BoundingBox(xmin=0, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 1: bounds: BoundingBox(xmin=0, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)
        # 2: bounds: BoundingBox(xmin=45, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 3: bounds: BoundingBox(xmin=45, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)

        # TEST:
        # Request agent that's in buffer, then moves off of buffer.
        # Is still properly ghosted?

        if rank == 1:
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 60))
            cspace.move(agent_201, cpt(10.2, 60.1))

        context.synchronize(restore_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)

        requests = []
        if rank == 0:
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)

        if rank == 1:
            # move off of buffer
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 66))
            cspace.move(agent_201, cpt(10.2, 66.1))

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))


class SharedUndirectedNetworkTests(unittest.TestCase):

    long_message = True

    def test_add_remove(self):
        # make 1 rank comm for basic add remove tests
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create(new_group)

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

            g.add_edge(agents[4], agents[5])
            self.assertEqual(2, g.num_edges(agents[5]))
            self.assertEqual(1, g.num_edges(agents[4]))

            # Note (5, 4) because getting edges by node from
            # undirected network, returns the asked for node first
            exp = {(agents[5], agents[6]), (agents[5], agents[4])}
            for edge in g._edges(agents[5]):
                exp.remove(edge)
            self.assertEqual(0, len(exp))

    def test_sync_1(self):
        # Tests add, update and remove edge
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        context = ctx.SharedContext(comm)
        g = UndirectedSharedNetwork('network', comm)
        context.add_projection(g)
        self.assertEqual(0, g.node_count)

        agents = [EAgent(x, 0, rank, x) for x in range(10)]
        for a in agents:
            context.add(a)
            self.assertEqual(rank, a.local_rank)

        self.assertEqual(10, g.node_count)

        requests = []
        if rank == 0:
            requests.append(((1, 0, 1), 1))
            requests.append(((1, 0, 2), 2))
            requests.append(((2, 0, 1), 1))
        elif rank == 3:
            requests.append(((1, 0, 0), 0))
            requests.append(((4, 0, 2), 2))
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0 or rank == 3:
            self.assertEqual(13, g.node_count)

        # Edges: 0: (0, 0, 0) -> (1, 0, 1)
        #        0: (2, 0, 1) -> (0, 0, 0)
        #        1: (0, 0, 1) -> (1, 0, 1)
        #        3: (0, 0, 3) -> (2, 0, 1)
        #        3: (1, 0, 0) -> (1, 0, 3)
        if rank == 0:
            other = context.ghost_agent((1, 0, 1))
            g.add_edge(agents[0], other, weight=2)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 2})])

            other = context.ghost_agent((2, 0, 1))
            g.add_edge(other, agents[0], rate=2)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges[1], (agents[0], other, {'rate': 2}))
        elif rank == 1:
            g.add_edge(agents[0], agents[1], weight=3)
            self.assertEqual(1, g.edge_count)

        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.add_edge(agents[0], other, weight=10)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 10})])

            other = context.ghost_agent((1, 0, 0))
            g.add_edge(other, agents[1], rate=2.1)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.edges(agents[1], data=True)]
            self.assertEqual(edges[0], (agents[1], other, {'rate': 2.1}))

        context.synchronize(restore_agent)

        # TEST vertices and edges created on ghost ranks
        if rank == 0:
            self.assertEqual(3, g.edge_count)
            agent = context.agent((0, 0, 0))
            edges = [x for x in g.graph.edges(agent, data=True)]
            self.assertEqual(2, len(edges))
            exp = [(agent, context.ghost_agent((2, 0, 1)), {'rate': 2}),
                   (agent, context.ghost_agent((1, 0, 1)), {'weight': 2})]
            for e in edges:
                self.assertTrue(e in exp)

        elif rank == 1:
            self.assertEqual(4, g.edge_count)
            agent = context.agent((1, 0, 1))
            edges = [x for x in g.graph.edges(agent, data=True)]
            self.assertEqual(2, len(edges))
            # looks like in undirected graph when request via edges(k)
            # the edge is returned as (k, other) regardless of how the
            # edge is originally inserted
            exp = [(agent, agents[0], {'weight': 3}), (agent, context.ghost_agent((0, 0, 0)), {'weight': 2})]
            self.assertEqual(edges, exp)

            agent = context.agent((2, 0, 1))
            edges = [x for x in g.graph.edges(agent, data=True)]
            self.assertEqual(2, len(edges))
            exp = [(agent, context.ghost_agent((0, 0, 0)), {'rate': 2}),
                   (agent, context.ghost_agent((0, 0, 3)), {'weight': 10})]
            self.assertEqual(edges, exp)

        elif rank == 3:
            self.assertEqual(2, g.edge_count)
            agent = agents[0]
            edges = [x for x in g.graph.edges(agent, data=True)]
            self.assertEqual(1, len(edges))
            self.assertEqual(edges[0], (agent, context.ghost_agent((2, 0, 1)), {'weight': 10}))

            agent = agents[1]
            edges = [x for x in g.graph.edges(agent, data=True)]
            self.assertEqual(1, len(edges))
            self.assertEqual(edges[0], (agent, context.ghost_agent((1, 0, 0)), {'rate': 2.1}))

        # TEST: update 2,0,1 agent and see if updates on 0 and 3
        if rank == 1:
            agents[2].energy = 134324

        # TEST: update edge with 2,0,1 and see if updates 1 and 0
        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.update_edge(agents[0], other, weight=14.2)

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertEqual(134324, context.ghost_agent((2, 0, 1)).energy)
        elif rank == 1:
            self.assertEqual(14.2, g.graph.edges[agents[2], context.ghost_agent((0, 0, 3))]['weight'])
        elif rank == 3:
            self.assertEqual(134324, context.ghost_agent((2, 0, 1)).energy)

        # TEST: remove edge 0: (2, 0, 1) -> (0, 0, 0)
        #                   3: (0, 0, 3) -> (2, 0, 1)
        if rank == 0:
            other = context.ghost_agent((2, 0, 1))
            g.remove_edge(agents[0], other)
            self.assertEqual(2, g.edge_count)
        elif rank == 1:
            self.assertIsNotNone(context.ghost_agent((0, 0, 3)))
        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.remove_edge(other, agents[0])

        context.synchronize(restore_agent)

        # TEST: 3 removed (0, 0, 3) -> (2, 0, 1), so
        # that edge should be removed, and (0, 0, 3) no
        # longer ghosted to 1
        if rank == 1:
            self.assertEqual(2, g.edge_count)
            self.assertTrue((agents[0], agents[1]) in g.graph.edges())
            self.assertTrue((agents[1], context.ghost_agent((0, 0, 0))) in g.graph.edges())
            self.assertIsNone(context.ghost_agent((0, 0, 3)))

    def test_sync_2(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        context = ctx.SharedContext(comm)
        g = UndirectedSharedNetwork('network', comm)
        context.add_projection(g)
        self.assertEqual(0, g.node_count)

        agents = [EAgent(x, 0, rank, x) for x in range(10)]
        for a in agents:
            context.add(a)
            self.assertEqual(rank, a.local_rank)

        self.assertEqual(10, g.node_count)

        requests = []
        if rank == 0:
            requests.append(((1, 0, 1), 1))
            requests.append(((1, 0, 2), 2))
            requests.append(((2, 0, 1), 1))
        elif rank == 3:
            requests.append(((1, 0, 0), 0))
            requests.append(((4, 0, 2), 2))
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0 or rank == 3:
            self.assertEqual(13, g.node_count)

        # Create edges with requested vertices
        # Edges: 0: (0, 0, 0) -> (1, 0, 1)
        #        0: (2, 0, 1) -> (0, 0, 0)
        #        1: (0, 0, 1) -> (1, 0, 1)
        #        3: (0, 0, 3) -> (2, 0, 1)
        #        3: (1, 0, 0) -> (1, 0, 3)
        if rank == 0:
            other = context.ghost_agent((1, 0, 1))
            g.add_edge(agents[0], other, weight=2)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 2})])

            other = context.ghost_agent((2, 0, 1))
            g.add_edge(other, agents[0], rate=2)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges[1], (agents[0], other, {'rate': 2}))
        elif rank == 1:
            g.add_edge(agents[0], agents[1], weight=3)
            self.assertEqual(1, g.edge_count)

        elif rank == 3:
            other = context.ghost_agent((2, 0, 1))
            g.add_edge(agents[0], other, weight=10)
            self.assertEqual(1, g.edge_count)
            edges = [x for x in g.graph.edges(agents[0], data=True)]
            self.assertEqual(edges, [(agents[0], other, {'weight': 10})])

            other = context.ghost_agent((1, 0, 0))
            g.add_edge(other, agents[1], rate=2.1)
            self.assertEqual(2, g.edge_count)
            edges = [x for x in g.graph.edges(agents[1], data=True)]
            self.assertEqual(edges[0], (agents[1], other, {'rate': 2.1}))

        context.synchronize(restore_agent)

        # TEST: remove (1, 0, 1) from 1
        # edges should be removed across processes
        if rank == 1:
            context.remove(agents[1])
            self.assertEqual(2, g.edge_count)
            self.assertTrue((agents[2], context.ghost_agent((0, 0, 0))) in g.graph.edges())
            self.assertTrue((agents[2], context.ghost_agent((0, 0, 3))) in g.graph.edges())
            self.assertTrue((1, 0, 1) not in context._agent_manager._ghosted_agents)

        context.synchronize(restore_agent)

        # (0, 0, 0) -> (1, 0, 1) removed
        if rank == 0:
            # 1, 0, 1 no longer ghosted to 0
            self.assertTrue((1, 0, 1) not in context._agent_manager._ghost_agents)
            self.assertEqual(2, len(g.graph.edges()))
            self.assertTrue((agents[0], context.ghost_agent((2, 0, 1))) in g.graph.edges())
            self.assertTrue((agents[1], context.ghost_agent((1, 0, 3))) in g.graph.edges())

        # Add edge on 1 then move one of those
        if rank == 1:
            g.add_edge(agents[2], agents[4], weight=42)
            # print(g.graph.nodes())
            # print(g.graph.edges())

        # print(rank, g.graph.edges(), flush=True)

        # TEST: Move (2, 0, 1) to 0
        # * local to 0
        # * ghosted from 0 to 3 and 1
        moved = []
        if rank == 1:
            # move 2,0,1 to o
            moved.append(((2, 0, 1), 0))

        context.move_agents(moved, restore_agent)

        # TODO: 2,0,1 is on 0 with local_rank of 0
        if rank == 0:

            self.assertIsNotNone(context.agent((2, 0, 1)))
            self.assertEqual(0, context.agent((2, 0, 1)).local_rank)
            self.assertTrue(g.graph.has_edge(context.agent((2, 0, 1)), context.agent((0, 0, 0))))
            self.assertTrue((2, 0, 1) in context._agent_manager._ghosted_agents)
            ghosted_to_ranks = context._agent_manager._ghosted_agents[(2, 0, 1)].ghost_ranks
            self.assertTrue(3 in ghosted_to_ranks)
            self.assertTrue(g.graph.has_edge(context.agent((2, 0, 1)), context.ghost_agent((0, 0, 3))))

            self.assertTrue(1 in ghosted_to_ranks)
            self.assertTrue(g.graph.has_edge(context.agent((2, 0, 1)), context.ghost_agent((4, 0, 1))))
            self.assertEqual(42, g.graph.edges[context.agent((2, 0, 1)),
                                               context.ghost_agent((4, 0, 1))]['weight'])

            # original 10 + moved 2,0,1 + requested 1,0,2 + 0,0,3 and 0,0,0 and 1,0,3 through
            # synchronized edges
            self.assertEqual(15, g.node_count)

        elif rank == 1:
            # print(g.graph.edges())
            # 2,0,1 moved
            self.assertIsNone(context.agent((2, 0, 1)))
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))
            # these were ghost agents through edges with 2,0,1
            # but should now be removed
            self.assertIsNone(context.ghost_agent((0, 0, 3)))
            self.assertIsNone(context.ghost_agent((0, 0, 0)))
            self.assertFalse(g.graph.has_edge(context.ghost_agent((4, 0, 1)), context.agent((0, 0, 3))))
            self.assertFalse(g.graph.has_edge(context.ghost_agent((4, 0, 1)), context.agent((0, 0, 0))))
            # ghosted from 0 to 1 because of 2,0,1 -> 4,0,1 edge
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))
            self.assertTrue(g.graph.has_edge(context.ghost_agent((2, 0, 1)), context.agent((4, 0, 1))))
        elif rank == 3:
            self.assertIsNone(context.agent((2, 0, 1)))
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))

        # Test: update edge(0,0,3 - 2,0,1) with new data
        #       ghost edge on 0 reflects change
        if rank == 3:
            g.update_edge(context.agent((0, 0, 3)), context.ghost_agent((2, 0, 1)), weight=12)
            self.assertEqual(12, g.graph.edges[context.agent((0, 0, 3)),
                                               context.ghost_agent((2, 0, 1))]['weight'])

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertEqual(12, g.graph.edges[context.ghost_agent((0, 0, 3)),
                                               context.agent((2, 0, 1))]['weight'])

    def test_with_oob(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = []
        for i in range(20):
            a = EAgent(i, 0, rank, 1)
            agents.append(a)
            context.add(a)

        box = space.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
        net = UndirectedSharedNetwork('network', comm)
        context.add_projection(cspace)
        context.add_projection(grid)
        context.add_projection(net)

        random.init(42)
        bounds = grid.get_local_bounds()
        xs = random.default_rng.integers(low=bounds.xmin, high=bounds.xmin + bounds.xextent, size=20)
        ys = random.default_rng.integers(low=bounds.ymin, high=bounds.ymin + bounds.yextent, size=20)
        for i, agent in enumerate(agents):
            grid.move(agent, dpt(xs[i], ys[i]))
            cspace.move(agent, cpt(xs[i], ys[i]))

        # TEST:
        # 1. request agents from neighboring ranks
        # 2. make edges between local agent and ghosts
        # 3. move agents oob, such that
        #    a. former ghost is now local
        #    b. ghost is now on different rank
        # 4. do tests

        requests = []
        if rank == 0:
            requests.append(((1, 0, 1), 1))
            requests.append(((1, 0, 2), 2))
            requests.append(((2, 0, 1), 1))
        elif rank == 3:
            requests.append(((1, 0, 0), 0))
            requests.append(((4, 0, 2), 2))
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0:
            net.add_edge(agents[1], context.ghost_agent((2, 0, 1)), color='red')
            net.add_edge(agents[10], context.ghost_agent((1, 0, 1)))
        elif rank == 1:
            net.add_edge(agents[2], agents[1])

        elif rank == 3:
            net.add_edge(agents[1], context.ghost_agent((1, 0, 0)))
            net.add_edge(agents[5], context.ghost_agent((2, 0, 1)))

        context.synchronize(restore_agent)

        # TESTS edges
        if rank == 0:
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[10], context.ghost_agent((1, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((1, 0, 3))))

        elif rank == 1:
            self.assertEqual(4, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((10, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[1], agents[2]))
            self.assertTrue(net.graph.has_edge(agents[2], context.ghost_agent((1, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[2], context.ghost_agent((5, 0, 3))))
            self.assertEqual('red', net.graph.edges[agents[2], context.ghost_agent((1, 0, 0))]['color'])
        elif rank == 2:
            self.assertEqual(0, net.edge_count)
        elif rank == 3:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((1, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[5], context.ghost_agent((2, 0, 1))))

        # Bounds:
        # print('{}: bounds: {}'.format(rank, grid.get_local_bounds()), flush=True)
        # 0: bounds: BoundingBox(xmin=0, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 1: bounds: BoundingBox(xmin=0, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)
        # 2: bounds: BoundingBox(xmin=45, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 3: bounds: BoundingBox(xmin=45, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)

        # Move (2, 0, 1) to 2's bounds
        if rank == 1:
            grid.move(agents[2], dpt(46, 35))
            cspace.move(agents[2], cpt(46.2, 35.1))

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[10], context.ghost_agent((1, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((1, 0, 3))))
        elif rank == 1:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((10, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
        elif rank == 2:
            agent_201 = context.agent((2, 0, 1))
            self.assertIsNotNone(agent_201)
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(agent_201, context.ghost_agent((1, 0, 0))))
            self.assertTrue(net.graph.has_edge(agent_201, context.ghost_agent((5, 0, 3))))
            self.assertTrue(net.graph.has_edge(agent_201, context.ghost_agent((1, 0, 1))))
            self.assertEqual('red', net.graph.edges[agent_201, context.ghost_agent((1, 0, 0))]['color'])
        elif rank == 3:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((1, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[5], context.ghost_agent((2, 0, 1))))

        if rank == 0:
            agents[1].energy = 101
        elif rank == 2:
            # 201 to 3
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(46, 80))
            cspace.move(agent_201, cpt(46.2, 80.1))

        context.synchronize(restore_agent)

        # print(f'{rank}: {net.graph.edges()}', flush=True)

        if rank == 0:
            self.assertEqual(3, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[10], context.ghost_agent((1, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((1, 0, 3))))
        elif rank == 1:
            self.assertEqual(2, net.edge_count)
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((10, 0, 0))))
            self.assertTrue(net.graph.has_edge(agents[1], context.ghost_agent((2, 0, 1))))
        elif rank == 2:
            self.assertEqual(0, net.edge_count)
        elif rank == 3:
            self.assertEqual(4, net.edge_count)
            agent_201 = context.agent((2, 0, 1))
            agent_100 = context.ghost_agent((1, 0, 0))
            self.assertIsNotNone(agent_201)
            self.assertTrue(net.graph.has_edge(agents[1], agent_100))
            self.assertTrue(net.graph.has_edge(agents[5], agent_201))
            self.assertTrue(net.graph.has_edge(context.ghost_agent((1, 0, 1)), agent_201))
            self.assertTrue(net.graph.has_edge(agent_201, agent_100))

            self.assertEqual(101, context.ghost_agent((1, 0, 0)).energy)

    def test_in_buffer(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = []
        for i in range(20):
            a = EAgent(i, 0, rank, 1)
            agents.append(a)
            context.add(a)

        box = space.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
        net = UndirectedSharedNetwork('network', comm)
        context.add_projection(net)
        context.add_projection(cspace)
        context.add_projection(grid)

        random.init(42)
        bounds = grid.get_local_bounds()
        xs = random.default_rng.integers(low=bounds.xmin, high=bounds.xmin + bounds.xextent, size=20)
        ys = random.default_rng.integers(low=bounds.ymin, high=bounds.ymin + bounds.yextent, size=20)
        for i, agent in enumerate(agents):
            grid.move(agent, dpt(xs[i], ys[i]))
            cspace.move(agent, cpt(xs[i], ys[i]))

        # Bounds:
        # print('{}: bounds: {}'.format(rank, grid.get_local_bounds()))
        # 0: bounds: BoundingBox(xmin=0, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 1: bounds: BoundingBox(xmin=0, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)
        # 2: bounds: BoundingBox(xmin=45, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 3: bounds: BoundingBox(xmin=45, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)

        # TEST:
        # Request agent that's in buffer, then moves off of buffer.
        # Is still properly ghosted?

        if rank == 1:
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 60))
            cspace.move(agent_201, cpt(10.2, 60.1))

        context.synchronize(restore_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)

        requests = []
        if rank == 0:
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, restore_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)

        if rank == 1:
            # move off of buffer
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 66))
            cspace.move(agent_201, cpt(10.2, 66.1))

        context.synchronize(restore_agent)

        if rank == 0:
            self.assertIsNotNone(context.ghost_agent((2, 0, 1)))

    def test_edge_in_buffer(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        context = ctx.SharedContext(comm)

        agents = []
        for i in range(20):
            a = EAgent(i, 0, rank, 1)
            agents.append(a)
            context.add(a)

        box = space.BoundingBox(xmin=0, xextent=90, ymin=0, yextent=120, zmin=0, zextent=0)
        cspace = space.SharedCSpace("shared_space", bounds=box, borders=BorderType.Sticky,
                                    occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm)
        net = UndirectedSharedNetwork('network', comm)
        context.add_projection(net)
        context.add_projection(cspace)
        context.add_projection(grid)

        random.init(42)
        bounds = grid.get_local_bounds()
        xs = random.default_rng.integers(low=bounds.xmin, high=bounds.xmin + bounds.xextent, size=20)
        ys = random.default_rng.integers(low=bounds.ymin, high=bounds.ymin + bounds.yextent, size=20)
        for i, agent in enumerate(agents):
            grid.move(agent, dpt(xs[i], ys[i]))
            cspace.move(agent, cpt(xs[i], ys[i]))

        # Bounds:
        # print('{}: bounds: {}'.format(rank, grid.get_local_bounds()))
        # 0: bounds: BoundingBox(xmin=0, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 1: bounds: BoundingBox(xmin=0, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)
        # 2: bounds: BoundingBox(xmin=45, xextent=45, ymin=0, yextent=60, zmin=0, zextent=0)
        # 3: bounds: BoundingBox(xmin=45, xextent=45, ymin=60, yextent=60, zmin=0, zextent=0)

        # TEST:
        # Request agent that's in buffer, then moves off of buffer.
        # Is still properly ghosted?

        if rank == 1:
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 60))
            cspace.move(agent_201, cpt(10.2, 60.1))

        context.synchronize(restore_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)
            net.add_edge(agent_201, agents[0])

        if rank == 1:
            # move off of buffer
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 66))
            cspace.move(agent_201, cpt(10.2, 66.1))

        context.synchronize(restore_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)
            self.assertTrue(net.contains_edge(agent_201, agents[0]))

        # TEST:
        # 1. Update agent that was on buffer and made edge on 1
        # 2. Test that change is reflected on 1
        if rank == 1:
            self.assertTrue(net.contains_edge(agent_201, context.ghost_agent((0, 0, 0))))
            agent_201.energy = 20

        context.synchronize(restore_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertEqual(agent_201.energy, 20)
            self.assertIsNotNone(agent_201)
            self.assertTrue(net.contains_edge(agent_201, agents[0]))


def construct_agent(nid, agent_type, rank, **kwargs):
    energy = kwargs['energy'] if 'energy' in kwargs else -1
    return EAgent(nid, agent_type, rank, energy)


class InitNetworkTests(unittest.TestCase):

    long_message = True

    def testNghs(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        context = ctx.SharedContext(comm)

        fpath = './test_data/simple_net.txt'
        read_network(fpath, context, construct_agent, restore_agent)
        g = context.get_projection("network")
        self.assertTrue(g.is_directed)

        if rank == 0:
            ss = [n.uid for n in g.graph.successors(context.agent((1, 0, 0)))]
            self.assertEqual(2, len(ss))
            self.assertTrue((3, 0, 2) in ss)
            self.assertTrue((2, 0, 1) in ss)

    def testInitWithAttributes(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        context = ctx.SharedContext(comm)

        fpath = './test_data/net_with_attribs.txt'
        read_network(fpath, context, construct_agent, restore_agent)
        g = context.get_projection("friend_network")
        self.assertFalse(g.is_directed)
        if rank == 0:
            self.assertEqual(1, context.size()[-1])
            a1 = context.agent((3, 0, 0))
            self.assertEqual(30, a1.energy)
            self.assertEqual(1, len(context._agent_manager._ghost_agents))
            self.assertIsNotNone(context.ghost_agent((1, 0, 1)))
            g1 = context.ghost_agent((1, 0, 1))
            self.assertEqual(23, g1.energy)
            edges = [x for x in g.graph.edges(a1, data=True)]
            self.assertEqual(edges, [(a1, g1, {'weight': 0.75})])
            # self.assertTrue((a1, g1) in g.graph.edges())
            # self.assertTrue((context.ghost_agent((0, 0, 0)), agents[1]) in g.graph.edges())

        elif rank == 1:
            self.assertEqual(2, context.size()[-1])
            a1 = context.agent((1, 0, 1))
            self.assertIsNotNone(a1)
            a2 = context.agent((2, 0, 1))
            self.assertIsNotNone(a2)
            self.assertEqual(3, len(context._agent_manager._ghost_agents))
            g3 = context.ghost_agent((3, 0, 0))
            self.assertIsNotNone(g3)
            g5 = context.ghost_agent((5, 0, 3))
            self.assertIsNotNone(g5)
            self.assertEqual(32, g5.energy)
            g4 = context.ghost_agent((4, 0, 2))
            self.assertIsNotNone(g4)
            self.assertEqual(5, g.edge_count)
            self.assertTrue(g.contains_edge(a1, a2))
            self.assertTrue(g.contains_edge(a1, g3))
            self.assertTrue(g.contains_edge(a1, g5))
            self.assertTrue(g.contains_edge(a2, g5))
            self.assertTrue(g.contains_edge(g4, a2))

        elif rank == 2:
            self.assertEqual(2, context.size()[-1])
            a4 = context.agent((4, 0, 2))
            self.assertIsNotNone(a4)
            a6 = context.agent((6, 0, 2))
            self.assertIsNotNone(a6)
            self.assertEqual(2, len(context._agent_manager._ghost_agents))
            g2 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(g2)
            g5 = context.ghost_agent((5, 0, 3))
            self.assertIsNotNone(g5)
            self.assertEqual(2, g.edge_count)
            self.assertTrue(g.contains_edge(a6, g5))
            self.assertTrue(g.contains_edge(a4, g2))

        elif rank == 3:
            self.assertEqual(1, context.size()[-1])
            a5 = context.agent((5, 0, 3))
            self.assertIsNotNone(a5)
            self.assertEqual(3, len(context._agent_manager._ghost_agents))
            g6 = context.ghost_agent((6, 0, 2))
            self.assertIsNotNone(g6)
            g1 = context.ghost_agent((1, 0, 1))
            self.assertIsNotNone(g1)
            g2 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(g2)
            self.assertEqual(3, g.edge_count)
            self.assertTrue(g.contains_edge(a5, g6))
            self.assertTrue(g.contains_edge(a5, g1))
            self.assertTrue(g.contains_edge(g2, a5))

    def testInitWithDiNoAttrib(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        context = ctx.SharedContext(comm)

        fpath = './test_data/net_with_no_attribs.txt'
        read_network(fpath, context, construct_agent, restore_agent)
        g = context.get_projection("friend_network")
        self.assertTrue(g.is_directed)
        if rank == 0:
            self.assertEqual(1, context.size()[-1])
            a1 = context.agent((3, 0, 0))
            self.assertEqual(-1, a1.energy)
            self.assertEqual(1, len(context._agent_manager._ghost_agents))
            self.assertIsNotNone(context.ghost_agent((1, 0, 1)))
            g1 = context.ghost_agent((1, 0, 1))
            self.assertEqual(-1, g1.energy)
            edges = [x for x in g.graph.edges(a1, data=True)]
            self.assertEqual(edges, [(a1, g1, {})])
            # self.assertTrue((a1, g1) in g.graph.edges())
            # self.assertTrue((context.ghost_agent((0, 0, 0)), agents[1]) in g.graph.edges())

        elif rank == 1:
            self.assertEqual(2, context.size()[-1])
            a1 = context.agent((1, 0, 1))
            self.assertIsNotNone(a1)
            a2 = context.agent((2, 0, 1))
            self.assertIsNotNone(a2)
            self.assertEqual(3, len(context._agent_manager._ghost_agents))
            g3 = context.ghost_agent((3, 0, 0))
            self.assertIsNotNone(g3)
            g5 = context.ghost_agent((5, 0, 3))
            self.assertIsNotNone(g5)
            self.assertEqual(-1, g5.energy)
            g4 = context.ghost_agent((4, 0, 2))
            self.assertIsNotNone(g4)
            self.assertEqual(5, g.edge_count)
            self.assertTrue(g.contains_edge(a1, a2))
            self.assertTrue(g.contains_edge(g3, a1))
            self.assertTrue(g.contains_edge(a1, g5))
            self.assertTrue(g.contains_edge(a2, g5))
            self.assertTrue(g.contains_edge(g4, a2))

        elif rank == 2:
            self.assertEqual(2, context.size()[-1])
            a4 = context.agent((4, 0, 2))
            self.assertIsNotNone(a4)
            a6 = context.agent((6, 0, 2))
            self.assertIsNotNone(a6)
            self.assertEqual(2, len(context._agent_manager._ghost_agents))
            g2 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(g2)
            g5 = context.ghost_agent((5, 0, 3))
            self.assertIsNotNone(g5)
            self.assertEqual(2, g.edge_count)
            self.assertTrue(g.contains_edge(a6, g5))
            self.assertTrue(g.contains_edge(a4, g2))

        elif rank == 3:
            self.assertEqual(1, context.size()[-1])
            a5 = context.agent((5, 0, 3))
            self.assertIsNotNone(a5)
            self.assertEqual(3, len(context._agent_manager._ghost_agents))
            g6 = context.ghost_agent((6, 0, 2))
            self.assertIsNotNone(g6)
            g1 = context.ghost_agent((1, 0, 1))
            self.assertIsNotNone(g1)
            g2 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(g2)
            self.assertEqual(3, g.edge_count)
            self.assertTrue(g.contains_edge(g6, a5))
            self.assertTrue(g.contains_edge(g1, a5))
            self.assertTrue(g.contains_edge(g2, a5))

    def test_generation1(self):
        # make 1 rank comm for basic add remove tests
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create(new_group)

        if comm != MPI.COMM_NULL:
            g = nx.generators.watts_strogatz_graph(30, 2, 0.25)
            fname = './test_data/gen_net_test.txt'
            write_network(g, "test", fname, 3)

            with open(fname, 'r') as f_in:
                line = f_in.readline().strip()
                vals = line.split(' ')
                self.assertEqual('test', vals[0])
                self.assertEqual('0', vals[1])

                nid_count = 0
                r_counts = [0, 0, 0]
                line = f_in.readline().strip()
                while line != 'EDGES':
                    nid, n_type, rank = [int(x) for x in line.split(' ')]
                    self.assertTrue(nid in g)
                    self.assertEqual(0, n_type)
                    r_counts[rank] += 1
                    nid_count += 1
                    line = f_in.readline().strip()

                self.assertEqual(30, nid_count)
                self.assertEqual(g.number_of_nodes(), nid_count)
                self.assertEqual(r_counts, [10, 10, 10])

                edge_count = 0
                for line in f_in:
                    line = line.strip()
                    u, v = [int(x) for x in line.split(' ')]
                    edge_count += 1
                    self.assertTrue(g.has_edge(u, v))

                self.assertEqual(g.number_of_edges(), edge_count)

    def test_generation2(self):
        # make 1 rank comm for basic add remove tests
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create(new_group)

        if comm != MPI.COMM_NULL:
            g = nx.generators.dual_barabasi_albert_graph(60, 2, 1, 0.25)
            attr = {}
            for i in range(30):
                attr[i] = {'agent_type': 0, 'a': 3}
            for i in range(30, 60):
                attr[i] = {'agent_type': 1, 'a': 4}
            nx.set_node_attributes(g, attr)

            fname = './test_data/gen_net_test.txt'
            write_network(g, "test", fname, 3, partition_method='random')

            p = re.compile('\{[^}]+\}|\S+')

            with open(fname, 'r') as f_in:
                line = f_in.readline().strip()
                vals = line.split(' ')
                self.assertEqual('test', vals[0])
                self.assertEqual('0', vals[1])

                nid_count = 0
                r_counts = [0, 0, 0]
                line = f_in.readline().strip()
                while line != 'EDGES':
                    nid, n_type, rank, attr = p.findall(line.strip())
                    nid = int(nid)
                    n_type = int(n_type)
                    rank = int(rank)
                    self.assertTrue(nid in g)
                    if nid < 30:
                        self.assertEqual(0, n_type)
                        self.assertEqual('{"a": 3}', attr)
                    else:
                        self.assertEqual(1, n_type)
                        self.assertEqual('{"a": 4}', attr)
                    r_counts[rank] += 1
                    nid_count += 1
                    line = f_in.readline().strip()

                self.assertEqual(60, nid_count)
                self.assertEqual(g.number_of_nodes(), nid_count)
                self.assertEqual(r_counts, [20, 20, 20])

                edge_count = 0
                for line in f_in:
                    line = line.strip()
                    u, v = [int(x) for x in line.split(' ')]
                    edge_count += 1
                    self.assertTrue(g.has_edge(u, v))

                self.assertEqual(g.number_of_edges(), edge_count)

    def test_generation3(self):
        # make 1 rank comm for basic add remove tests
        new_group = MPI.COMM_WORLD.Get_group().Incl([0])
        comm = MPI.COMM_WORLD.Create(new_group)

        if comm != MPI.COMM_NULL:
            g = nx.complete_graph(60)
            fname = './test_data/gen_net_test.txt'
            try:
                import nxmetis
                options = nxmetis.types.MetisOptions(seed=1)
                write_network(g, "metis_test", fname, 3, partition_method='metis', options=options)

                p = re.compile('\{[^}]+\}|\S+')
                ranks = {}

                with open(fname, 'r') as f_in:
                    line = f_in.readline().strip()
                    vals = line.split(' ')
                    self.assertEqual('metis_test', vals[0])
                    self.assertEqual('0', vals[1])

                    line = f_in.readline().strip()
                    while line != 'EDGES':
                        nid, n_type, rank = p.findall(line.strip())
                        nid = int(nid)
                        n_type = int(n_type)
                        rank = int(rank)
                        self.assertTrue(nid in g)
                        self.assertEqual(0, n_type)
                        ranks[nid] = rank
                        line = f_in.readline().strip()

                self.assertEqual(60, len(ranks))
                _, partitions = nxmetis.partition(g, 3, options=options)
                for i, partition in enumerate(partitions):
                    for nid in partition:
                        self.assertEqual(i, ranks[nid])
            except ModuleNotFoundError:
                print("Ignoring nxmetis test")

