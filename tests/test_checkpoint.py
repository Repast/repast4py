import unittest
import dill as pickle
from mpi4py import MPI
from typing import Tuple
import networkx as nx
import json
import os

try:
    from repast4py import random, checkpoint, schedule, core
except ModuleNotFoundError:
    import sys
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py import random, checkpoint, schedule, core

from repast4py.schedule import PriorityType
from repast4py.context import SharedContext
from repast4py.network import DirectedSharedNetwork
import repast4py.network


class EAgent(core.Agent):

    def __init__(self, id, agent_type, rank, val=None):
        super().__init__(id=id, type=agent_type, rank=rank)
        if val is None:
            self.val = [random.default_rng.random()]
        else:
            self.val = val

    def update(self):
        self.val.append(random.default_rng.random())

    def save(self) -> Tuple:
        return (self.uid, list(self.val))


class OAgent(core.Agent):

    def __init__(self, id, agent_type, rank,):
        super().__init__(id=id, type=agent_type, rank=rank)
        self.val = 0

    def update(self):
        self.val += 1

    def save(self) -> Tuple:
        return (self.uid, list(self.val))


def restore_agent(agent_data: Tuple):
    uid = agent_data[0]
    val = agent_data[1]
    agent = EAgent(uid[0], uid[1], uid[2], val)
    return agent


class Model:

    def __init__(self, comm: MPI.Intracomm, ckp: checkpoint.Checkpoint = None):
        self.context = SharedContext(comm)
        if ckp is None:
            rank = comm.Get_rank()
            for i in range(3):
                self.context.add(EAgent(i, 0, rank))
            self.runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
            self.runner.schedule_repeating_event(1.0, 1.0, self.step,
                                                 metadata={'name': 'model.step'})
        else:
            def restore_schedule(data):
                return self.step

            self.runner = ckp.restore_schedule(MPI.COMM_WORLD, restore_schedule)
            for agent in ckp.restore_agents(restore_agent):
                self.context.add(agent)

    def step(self):
        for agent in self.context.agents():
            agent.update()


class NetworkModel:

    def __init__(self, comm: MPI.Intracomm, ckp: checkpoint.Checkpoint = None, network_path=None):
        self.context = SharedContext(comm)
        if ckp is None:
            self.context = SharedContext(comm)
            self.network = DirectedSharedNetwork('test', comm)
            self.context.add_projection(self.network)

            self.rank = comm.Get_rank()
            g: nx.Graph = nx.connected_watts_strogatz_graph(20, 2, 0.25)
            node_map = {}
            for node in g.nodes:
                agent = EAgent(node, 0, self.rank)
                self.context.add(agent)
                node_map[node] = agent

            for edge in g.edges:
                u, v = edge
                self.network.add_edge(node_map[u], node_map[v], weight=u)

            self.runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
            self.runner.schedule_repeating_event(1.0, 1.0, self.step,
                                                 metadata={'name': 'model.step'})
        else:
            def restore_schedule(data):
                return self.step

            self.runner = ckp.restore_schedule(MPI.COMM_WORLD, restore_schedule)
            agent_map = {agent.uid: agent for agent in ckp.restore_agents(restore_agent)}

            def create_agent(node_id, agent_type, rank, **agent_attributes):
                return agent_map[tuple(agent_attributes['uid'])]
            repast4py.network.read_network(network_path, self.context, create_agent, None)
            self.network = self.context.get_projection('test')

    def step(self):
        u, v = random.default_rng.integers(0, 20, size=2)
        head = self.context.agent((u, 0, self.rank))
        tail = self.context.agent((v, 0, self.rank))
        if self.network.contains_edge(head, tail):
            self.network.remove_edge(head, tail)
        else:
            self.network.add_edge(head, tail)


class CheckpointTests(unittest.TestCase):

    def pickle(self, fname, checkpoint):
        with open(fname, 'wb') as fout:
            pickle.dump(checkpoint, fout)

    def unpickle(self, fname):
        with open(fname, 'rb') as fin:
            obj = pickle.load(fin)
        return obj

    def test_random(self):
        random.init(42)
        # generate 10 values
        random.default_rng.random((10,))
        ckp = checkpoint.Checkpoint()
        ckp.save_random()

        exp_vals = random.default_rng.random((10,))
        random.init(31)
        ckp.restore_random()
        self.assertEqual(42, random.seed)
        vals = random.default_rng.random((10,))
        self.assertEqual(list(exp_vals), list(vals))

        # test pickling
        fname = './test_data/checkpoint.pkl'
        self.pickle(fname, ckp)
        ckp = self.unpickle(fname)
        ckp.restore_random()
        self.assertEqual(42, random.seed)
        vals = random.default_rng.random((10,))
        self.assertEqual(list(exp_vals), list(vals))

    def test_one_time_schedule(self):
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        agent = EAgent(1, 1, 0)
        runner.schedule_event(1.0, agent.update)
        runner.schedule.execute()
        self.assertEqual(2, len(agent.val))
        runner.schedule_event(2.0, agent.update, metadata={'name': 'agent.update'})

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        runner.schedule.execute()
        expected = list(agent.val)

        def restorer(data):
            self.assertEqual(data['name'], 'agent.update')
            return agent.update

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer)

        self.assertEqual(1, len(runner.schedule.queue))

        # mimic restoring agent from checkpoint by resetting its internal
        # state to that at the time of checkpoint
        agent.val = agent.val[:2]

        # current tick is time of checkpoint
        self.assertEqual(1.0, runner.tick())
        # execute the restored schedule
        runner.schedule.execute()
        self.assertEqual(2.0, runner.tick())
        self.assertEqual(expected, agent.val)

    def test_repeating_schedule(self):
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        agent = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, agent.update, metadata={'name': 'agent.update'})
        runner.schedule.execute()
        runner.schedule.execute()

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        runner.schedule.execute()
        runner.schedule.execute()
        expected = list(agent.val)

        def restorer(data):
            self.assertEqual(data['name'], 'agent.update')
            return agent.update

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer)

        self.assertEqual(1, len(runner.schedule.queue))

        # mimic restoring agent from checkpoint by resetting its internal
        # state to that at the time of checkpoint
        agent.val = agent.val[:3]

        # current tick is time of checkpoint
        self.assertEqual(2.0, runner.tick())
        # execute the restored schedule
        runner.schedule.execute()
        runner.schedule.execute()
        self.assertEqual(4.0, runner.tick())
        self.assertEqual(expected, agent.val)

    def test_same_tick_schedule(self):
        """Tests multiple at same tick -- checkpoint preserves event adding
        order prior to random sort for execution.
        """
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        a1 = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a1.update, metadata={'name': 'a1'})
        a2 = EAgent(2, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a2.update, metadata={'name': 'a2'})
        a3 = EAgent(3, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a3.update, metadata={'name': 'a3'})

        runner.schedule.execute()
        runner.schedule.execute()

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        n = 200
        for _ in range(n):
            runner.schedule.execute()

        expected = [list(a1.val), list(a2.val), list(a3.val)]

        def restorer(data):
            if data['name'] == 'a1':
                return a1.update
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'a3':
                return a3.update
            else:
                self.fail()

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer)

        a1.val = a1.val[:-n]
        a2.val = a2.val[:-n]
        a3.val = a3.val[:-n]

        self.assertEqual(2.0, runner.tick())
        # execute the restored schedule
        for _ in range(n):
            runner.schedule.execute()
        self.assertEqual(n + 2.0, runner.tick())
        self.assertEqual(expected[0], a1.val)
        self.assertEqual(expected[1], a2.val)
        self.assertEqual(expected[2], a3.val)

    def test_priority_first(self):
        """Tests priority first flag preserved."""
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        a1 = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a1.update, priority_type=PriorityType.FIRST,
                                        metadata={'name': 'a1'})
        a2 = EAgent(2, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a2.update, metadata={'name': 'a2'})
        a3 = EAgent(3, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a3.update, metadata={'name': 'a3'})

        runner.schedule.execute()
        runner.schedule.execute()

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        n = 200
        for _ in range(n):
            runner.schedule.execute()
        expected = [list(a1.val), list(a2.val), list(a3.val)]

        def restorer(data):
            if data['name'] == 'a1':
                return a1.update
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'a3':
                return a3.update
            else:
                self.fail()

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer)

        a1.val = a1.val[:-n]
        a2.val = a2.val[:-n]
        a3.val = a3.val[:-n]

        self.assertEqual(2.0, runner.tick())
        # execute the restored schedule
        for _ in range(n):
            runner.schedule.execute()

        self.assertEqual(n + 2.0, runner.tick())
        self.assertEqual(expected[0], a1.val)
        self.assertEqual(expected[1], a2.val)
        self.assertEqual(expected[2], a3.val)

    def test_priority_last(self):
        """Tests priority last flag preserved."""
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        a1 = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a1.update, priority_type=PriorityType.LAST,
                                        metadata={'name': 'a1'})
        a2 = EAgent(2, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a2.update, metadata={'name': 'a2'})
        a3 = EAgent(3, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a3.update, metadata={'name': 'a3'})

        runner.schedule.execute()
        runner.schedule.execute()

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        n = 200
        for _ in range(n):
            runner.schedule.execute()
        expected = [list(a1.val), list(a2.val), list(a3.val)]

        def restorer(data):
            if data['name'] == 'a1':
                return a1.update
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'a3':
                return a3.update
            else:
                self.fail()

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer)

        a1.val = a1.val[:-n]
        a2.val = a2.val[:-n]
        a3.val = a3.val[:-n]

        self.assertEqual(2.0, runner.tick())
        # execute the restored schedule
        for _ in range(n):
            runner.schedule.execute()
        self.assertEqual(n + 2.0, runner.tick())
        self.assertEqual(expected[0], a1.val)
        self.assertEqual(expected[1], a2.val)
        self.assertEqual(expected[2], a3.val)

    def test_priority_valu(self):
        """Tests by priority"""
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        a1 = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a1.update, priority_type=PriorityType.BY_PRIORITY, priority=1,
                                        metadata={'name': 'a1'})
        a2 = EAgent(2, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a2.update, metadata={'name': 'a2'})
        a3 = EAgent(3, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a3.update, priority_type=PriorityType.BY_PRIORITY, priority=2,
                                        metadata={'name': 'a3'})

        runner.schedule.execute()
        runner.schedule.execute()

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        n = 200
        for _ in range(n):
            runner.schedule.execute()
        expected = [list(a1.val), list(a2.val), list(a3.val)]

        def restorer(data):
            if data['name'] == 'a1':
                return a1.update
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'a3':
                return a3.update
            else:
                self.fail()

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer)

        a1.val = a1.val[:-n]
        a2.val = a2.val[:-n]
        a3.val = a3.val[:-n]

        self.assertEqual(2.0, runner.tick())
        # execute the restored schedule
        for _ in range(n):
            runner.schedule.execute()
        self.assertEqual(n + 2.0, runner.tick())
        self.assertEqual(expected[0], a1.val)
        self.assertEqual(expected[1], a2.val)
        self.assertEqual(expected[2], a3.val)

    def test_stop_at(self):
        """Tests stop at."""
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        a1 = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a1.update, priority_type=PriorityType.BY_PRIORITY, priority=1,
                                        metadata={'name': 'a1'})
        a2 = EAgent(2, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a2.update, metadata={'name': 'a2'})
        a3 = EAgent(3, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a3.update, priority_type=PriorityType.BY_PRIORITY, priority=2,
                                        metadata={'name': 'a3'})
        runner.schedule_stop(12.1)

        ckp = checkpoint.Checkpoint()

        def save():
            ckp.save_random()
            ckp.save_schedule()

        runner.schedule_event(3.1, save)
        runner.execute()

        expected = [list(a1.val), list(a2.val), list(a3.val)]

        def restorer(data):
            if data['name'] == 'a1':
                return a1.update
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'a3':
                return a3.update
            else:
                self.fail()

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer, restore_stop_at=True)
        self.assertEqual(3.1, runner.tick())

        # restore agent state to state a checkpoint
        # 1 draw from constructor + 3 from scheduled events
        a1.val = a1.val[:4]
        a2.val = a2.val[:4]
        a3.val = a3.val[:4]

        runner.execute()

        self.assertEqual(12.1, runner.tick())
        self.assertEqual(expected[0], a1.val)
        self.assertEqual(expected[1], a2.val)
        self.assertEqual(expected[2], a3.val)

    def test_end_evts(self):
        """Test end actions"""
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        a1 = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a1.update, priority_type=PriorityType.BY_PRIORITY, priority=1,
                                        metadata={'name': 'a1'})
        runner.schedule_stop(12.1)

        ckp = checkpoint.Checkpoint()

        def save():
            ckp.save_random()
            ckp.save_schedule()

        runner.schedule_event(3.1, save)

        def end_evt(evt_vals):
            evt_vals.append(random.default_rng.random())

        n = 100
        vals = []
        evt = schedule.create_arg_evt(end_evt, vals)
        for _ in range(n):
            runner.schedule_end_event(evt, metadata={'name': 'end_evt'})
        runner.execute()

        self.assertEqual(100, len(vals))
        expected = list(vals)

        vals = []
        evt = schedule.create_arg_evt(end_evt, vals)

        def restorer(data):
            if data['name'] == 'a1':
                return a1.update
            elif data['name'] == 'end_evt':
                return evt

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer, restore_stop_at=True)
        self.assertEqual(3.1, runner.tick())

        runner.execute()
        self.assertEqual(expected, vals)

    def test_pickled_schedule(self):
        """Tests pickled obj works"""
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        fname = './test_data/checkpoint.pkl'

        a1 = EAgent(1, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a1.update, priority_type=PriorityType.BY_PRIORITY, priority=1,
                                        metadata={'name': 'a1'})
        a2 = EAgent(2, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a2.update, metadata={'name': 'a2'})
        a3 = EAgent(3, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a3.update, priority_type=PriorityType.BY_PRIORITY, priority=2,
                                        metadata={'name': 'a3'})
        runner.schedule_stop(12.1)

        def save():
            ckp = checkpoint.Checkpoint()
            ckp.save_random()
            ckp.save_schedule()
            self.pickle(fname, ckp)

        runner.schedule_event(3.1, save)
        runner.execute()

        expected = [list(a1.val), list(a2.val), list(a3.val)]

        def restorer(data):
            if data['name'] == 'a1':
                return a1.update
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'a3':
                return a3.update
            else:
                self.fail()

        ckp1: checkpoint.Checkpoint = self.unpickle(fname)
        ckp1.restore_random()
        runner = ckp1.restore_schedule(MPI.COMM_WORLD, restorer, restore_stop_at=True)

        self.assertEqual(3.1, runner.tick())

        # restore agent state to state a checkpoint
        # 1 draw from constructor + 3 from scheduled events
        a1.val = a1.val[:4]
        a2.val = a2.val[:4]
        a3.val = a3.val[:4]

        a4 = OAgent(4, 1, 0)
        self.assertEqual(0, a4.val)
        runner.schedule_event(3.9, a4.update)
        runner.schedule.execute()
        self.assertEqual(1, a4.val)

        runner.execute()
        self.assertEqual(12.1, runner.tick())
        self.assertEqual(expected[0], a1.val)
        self.assertEqual(expected[1], a2.val)
        self.assertEqual(expected[2], a3.val)

    def test_voided(self):
        "Test voided events are not serialized"
        random.init(42)
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)

        a1 = EAgent(1, 1, 0)
        evt = runner.schedule_repeating_event(1.0, 1.0, a1.update, priority_type=PriorityType.BY_PRIORITY, priority=1,
                                              metadata={'name': 'a1'})
        a2 = EAgent(2, 1, 0)
        runner.schedule_repeating_event(1.0, 1.0, a2.update, metadata={'name': 'a2'})
        runner.schedule_stop(12.1)

        def end_evt():
            pass
        e_evt = runner.schedule_end_event(evt=end_evt, metadata={'name': 'end'})

        runner.schedule.execute()
        runner.schedule.execute()
        evt.void()
        e_evt.void()

        a1_expected = list(a1.val)

        # checkpoint a voided evt
        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()

        def restorer(data):
            if data['name'] == 'a1':
                self.fail()
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'end':
                self.fail()

        ckp.restore_random()
        runner = ckp.restore_schedule(MPI.COMM_WORLD, restorer, restore_stop_at=True)
        self.assertEqual(2.0, runner.tick())

        runner.execute()
        self.assertEqual(3, len(a1.val))
        self.assertEqual(a1_expected, a1.val)

    def test_agents(self):
        random.init(42)

        model = Model(MPI.COMM_WORLD)

        model.runner.schedule.execute()
        model.runner.schedule.execute()

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()
        ckp.save_agents(model.context.agents(shuffle=False))

        model.runner.schedule.execute()
        model.runner.schedule.execute()
        expected = {agent.uid: agent.val for agent in model.context.agents()}

        ckp.restore_random()
        model = Model(MPI.COMM_WORLD, ckp)

        for agent in model.context.agents(shuffle=False):
            self.assertEqual(agent.val, expected[agent.uid][:-2])

        # run forwards
        model.runner.schedule.execute()
        model.runner.schedule.execute()

        for agent in model.context.agents():
            self.assertEqual(agent.val, expected[agent.uid])

    def test_network(self):
        random.init(42)
        nm = NetworkModel(MPI.COMM_WORLD)

        nm.runner.schedule.execute()
        nm.runner.schedule.execute()

        if not os.path.exists('./test_out'):
            os.mkdir('./test_out')
        fpath = './test_out/network.txt'
        with open(fpath, 'w') as fout:
            graph = nm.network.graph
            is_d = 1 if nx.is_directed(graph) else 0
            fout.write(f'{nm.network.name} {is_d}\n')
            node_map = {}
            for i, nd in enumerate(graph.nodes(data=True)):
                repast4py.network._write_node(fout, (i, {'uid': nd[0].uid}), nd[0].uid_rank)
                node_map[nd[0]] = i

            fout.write('EDGES\n')
            for u, v, attribute in graph.edges(data=True):
                line = f'{node_map[u]} {node_map[v]}'
                if (len(attribute) > 0):
                    line = f'{line} {json.dumps(attribute)}'

                fout.write(line)
                fout.write('\n')

        expected = [(u.uid, v.uid, attribute) for u, v, attribute in graph.edges(data=True)]

        ckp = checkpoint.Checkpoint()
        ckp.save_random()
        ckp.save_schedule()
        ckp.save_agents(nm.context.agents(shuffle=False))

        nm.runner.schedule.execute()
        nm.runner.schedule.execute()

        ckp.restore_random()
        nm = NetworkModel(MPI.COMM_WORLD, ckp, fpath)
        actual = [(u.uid, v.uid, attribute) for u, v, attribute in nm.network.graph.edges(data=True)]

        self.assertEqual(len(actual), len(expected))
        for item in expected:
            self.assertTrue(item in actual)
