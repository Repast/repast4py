import unittest
import pickle
from mpi4py import MPI

from repast4py import random, checkpoint, schedule, core


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
        runner.schedule.execute()
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

        a1.val = a1.val[:-2]
        a2.val = a2.val[:-2]
        a3.val = a3.val[:-2]

        self.assertEqual(2.0, runner.tick())
        # execute the restored schedule
        runner.schedule.execute()
        runner.schedule.execute()
        self.assertEqual(4.0, runner.tick())
        self.assertEqual(expected[0], a1.val)
        self.assertEqual(expected[1], a2.val)
        self.assertEqual(expected[2], a3.val)

# multiple at same tick with random priority (tests order_idx setting)
# priority first
# priority last
# priority value
# stop_at
# end_at
# multiple at same tick with additional evts scheduled after checkpoint