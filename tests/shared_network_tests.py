import sys
import os
from mpi4py import MPI
import unittest

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

from repast4py.network import UndirectedSharedNetwork, DirectedSharedNetwork
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

    def load(self, data):
        self.restored = True
        self.energy = data


def create_agent(agent_data):
    # agent_data: [aid_tuple, energy]
    uid = agent_data[0]
    return EAgent(uid[0], uid[1], uid[2], agent_data[1])


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

        context.request_agents(requests, create_agent)

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

        context.synchronize(create_agent)

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

        context.synchronize(create_agent)

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

        context.synchronize(create_agent)

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

        context.request_agents(requests, create_agent)

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

        context.synchronize(create_agent)

        # TEST: remove (1, 0, 1) from 1
        # edges should be removed across processes
        if rank == 1:
            context.remove(agents[1])
            self.assertEqual(2, g.edge_count)
            self.assertTrue((agents[2], context.ghost_agent((0, 0, 0))) in g.graph.edges())
            self.assertTrue((agents[2], context.ghost_agent((0, 0, 3))) in g.graph.edges())
            self.assertTrue((1, 0, 1) not in context._agent_manager._ghosted_agents)

        context.synchronize(create_agent)

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

        context.move_agents(moved, create_agent)

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

        context.synchronize(create_agent)

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
                                    occupancy=OccupancyType.Multiple, buffersize=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffersize=2, comm=comm)
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

        context.request_agents(requests, create_agent)

        if rank == 0:
            net.add_edge(agents[1], context.ghost_agent((2, 0, 1)), color='red')
            net.add_edge(agents[10], context.ghost_agent((1, 0, 1)))
        elif rank == 1:
            net.add_edge(agents[2], agents[1])

        elif rank == 3:
            net.add_edge(agents[1], context.ghost_agent((1, 0, 0)))
            net.add_edge(agents[5], context.ghost_agent((2, 0, 1)))

        context.synchronize(create_agent)

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

        context.synchronize(create_agent)

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

        context.synchronize(create_agent)

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
                                    occupancy=OccupancyType.Multiple, buffersize=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffersize=2, comm=comm)
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

        context.synchronize(create_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)

        requests = []
        if rank == 0:
            requests.append(((2, 0, 1), 1))

        context.request_agents(requests, create_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)

        if rank == 1:
            # move off of buffer
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 66))
            cspace.move(agent_201, cpt(10.2, 66.1))

        context.synchronize(create_agent)

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
                                    occupancy=OccupancyType.Multiple, buffersize=2, comm=comm, tree_threshold=100)
        grid = space.SharedGrid("shared_grid", bounds=box, borders=BorderType.Sticky,
                                occupancy=OccupancyType.Multiple, buffersize=2, comm=comm)
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

        context.synchronize(create_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertIsNotNone(agent_201)
            net.add_edge(agent_201, agents[0])

        if rank == 1:
            # move off of buffer
            agent_201 = context.agent((2, 0, 1))
            grid.move(agent_201, dpt(10, 66))
            cspace.move(agent_201, cpt(10.2, 66.1))

        context.synchronize(create_agent)

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

        context.synchronize(create_agent)

        if rank == 0:
            agent_201 = context.ghost_agent((2, 0, 1))
            self.assertEqual(agent_201.energy, 20)
            self.assertIsNotNone(agent_201)
            self.assertTrue(net.contains_edge(agent_201, agents[0]))
