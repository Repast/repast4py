import unittest
import pickle
from mpi4py import MPI

from repast4py import random, checkpoint, schedule, core
from repast4py.schedule import PriorityType


class EAgent(core.Agent):

    def __init__(self, id, agent_type, rank):
        super().__init__(id=id, type=agent_type, rank=rank)
        self.val = [random.default_rng.random()]

    def update(self):
        self.val.append(random.default_rng.random())


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
        checkpoint.save_random(ckp)

        exp_vals = random.default_rng.random((10,))
        random.init(31)
        checkpoint.restore_random(ckp)
        self.assertEqual(42, random.seed)
        vals = random.default_rng.random((10,))
        self.assertEqual(list(exp_vals), list(vals))

        # test pickling
        fname = './test_data/checkpoint.pkl'
        self.pickle(fname, ckp)
        ckp = self.unpickle(fname)
        checkpoint.restore_random(ckp)
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
        checkpoint.save_random(ckp)
        checkpoint.save_schedule(ckp)

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        runner.schedule.execute()
        expected = list(agent.val)

        def restorer(data):
            self.assertEqual(data['name'], 'agent.update')
            return agent.update

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)

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
        checkpoint.save_random(ckp)
        checkpoint.save_schedule(ckp)

        # run forward to add another val to agent.val
        # running from checkpoint should match this agent.val
        runner.schedule.execute()
        runner.schedule.execute()
        expected = list(agent.val)

        def restorer(data):
            self.assertEqual(data['name'], 'agent.update')
            return agent.update

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)

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
        checkpoint.save_random(ckp)
        checkpoint.save_schedule(ckp)

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

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)

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
        checkpoint.save_random(ckp)
        checkpoint.save_schedule(ckp)

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

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)

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
        checkpoint.save_random(ckp)
        checkpoint.save_schedule(ckp)

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

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)

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
        checkpoint.save_random(ckp)
        checkpoint.save_schedule(ckp)

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

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)

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
            checkpoint.save_random(ckp)
            checkpoint.save_schedule(ckp)

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

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)
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
            checkpoint.save_random(ckp)
            checkpoint.save_schedule(ckp)

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

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)
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
            checkpoint.save_random(ckp)
            checkpoint.save_schedule(ckp)
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

        ckp1 = self.unpickle(fname)
        checkpoint.restore_random(ckp1)
        runner = checkpoint.restore_schedule(ckp1, restorer, MPI.COMM_WORLD)

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
        checkpoint.save_random(ckp)
        checkpoint.save_schedule(ckp)

        def restorer(data):
            if data['name'] == 'a1':
                self.fail()
            elif data['name'] == 'a2':
                return a2.update
            elif data['name'] == 'end':
                self.fail()

        checkpoint.restore_random(ckp)
        runner = checkpoint.restore_schedule(ckp, restorer, MPI.COMM_WORLD)
        self.assertEqual(2.0, runner.tick())

        runner.execute()
        self.assertEqual(3, len(a1.val))
        self.assertEqual(a1_expected, a1.val)
