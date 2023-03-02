import unittest
import sys
import os
from mpi4py import MPI

try:
    from repast4py import schedule
    from repast4py.schedule import PriorityType
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py import schedule
    from repast4py.schedule import PriorityType


class Agent:

    def __init__(self, schedule):
        self.sched = schedule
        self.at = -1.0
        self.evts = []

    def run(self):
        self.at = self.sched.tick

    def tag(self, val):
        self.evts.append(val)


class Agent2:
    def __init__(self):
        self.at = 0
        self.result = 0.0

    def run(self):
        self.at = schedule.runner().tick()

    def op(self, a, b, op='+'):
        if op == '+':
            self.result = a + b
        elif op == '*':
            self. result = a * b


class Agent3:

    def __init__(self, schedule):
        self.sched = schedule
        self.ticks = []

    def run(self):
        self.ticks.append(self.sched.tick)
        if len(self.ticks) < 3:
            self.sched.schedule_event(self.sched.tick, self.run)


def gen_reset_stop(evt, runner, new_stop):
    def f():
        evt.void()
        runner.schedule_stop(new_stop)

    return f


# Run from parent dir: python -m unittest tests.schedule_tests
class ScheduleTests(unittest.TestCase):

    def test_schedule_0(self):
        sched = schedule.Schedule()
        a1 = Agent(sched)

        sched.schedule_event(0, a1.run)
        sched.execute()
        self.assertEqual(0, a1.at)
        sched.execute()
        self.assertEqual(0, a1.at)

        sched = schedule.Schedule()
        a1 = Agent(sched)

        sched.schedule_repeating_event(0, 1, a1.run)
        sched.execute()
        self.assertEqual(0, a1.at)
        sched.execute()
        self.assertEqual(1, a1.at)
        sched.execute()
        self.assertEqual(2, a1.at)

    def test_schedule_at(self):
        sched = schedule.Schedule()
        a1 = Agent(sched)
        a2 = Agent(sched)
        a3 = Agent(sched)

        sched.schedule_event(1.3, a1.run)
        sched.schedule_event(1.3, a2.run)
        sched.schedule_event(2.5, a3.run)

        sched.execute()
        self.assertEqual(1.3, a1.at)
        self.assertEqual(1.3, a2.at)
        self.assertEqual(-1, a3.at)

        sched.execute()
        self.assertEqual(1.3, a1.at)
        self.assertEqual(1.3, a2.at)
        self.assertEqual(2.5, a3.at)

        sched.execute()
        self.assertEqual(1.3, a1.at)
        self.assertEqual(1.3, a2.at)
        self.assertEqual(2.5, a3.at)

    def test_schedule_repeating(self):
        sched = schedule.Schedule()
        a1 = Agent(sched)
        a2 = Agent(sched)
        a3 = Agent(sched)

        sched.schedule_repeating_event(1, 1, a1.run)
        sched.schedule_repeating_event(1, 1, a2.run)
        sched.schedule_repeating_event(1.5, 0.25, a3.run)

        sched.execute()
        self.assertEqual(1, a1.at)
        self.assertEqual(1, a2.at)
        self.assertEqual(-1, a3.at)

        sched.execute()
        self.assertEqual(1, a1.at)
        self.assertEqual(1, a2.at)
        self.assertEqual(1.5, a3.at)

        sched.execute()
        self.assertEqual(1, a1.at)
        self.assertEqual(1, a2.at)
        self.assertEqual(1.75, a3.at)

        sched.execute()
        self.assertEqual(2, a1.at)
        self.assertEqual(2, a2.at)
        self.assertEqual(2, a3.at)

    def test_default_schedule(self):
        self.assertRaises(RuntimeError, lambda: schedule.runner())

        a1 = Agent2()

        from mpi4py import MPI
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        runner.schedule_event(1.1, a1.run)
        runner.schedule_stop(2.0)
        runner.execute()
        self.assertEqual(1.1, a1.at)

        # test runner() function
        a1.at = 0
        schedule.init_schedule_runner(MPI.COMM_WORLD)
        runner = schedule.runner()
        runner.schedule_event(1.1, a1.run)
        runner.schedule_stop(2.0)
        runner.execute()
        self.assertEqual(1.1, a1.at)

    def test_schedule_same_tick(self):
        sched = schedule.Schedule()
        a1 = Agent3(sched)
        sched.schedule_event(1.3, a1.run)
        sched.execute()
        self.assertEqual(3, len(a1.ticks))
        self.assertEqual([1.3, 1.3, 1.3], a1.ticks)

    def test_void_evt(self):
        a1 = Agent2()

        from mpi4py import MPI
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        runner.schedule_event(1.1, a1.run)
        runner.schedule_stop(2.0)
        runner.execute()
        self.assertEqual(1.1, a1.at)

        a1.at = 0
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        sevt = runner.schedule_event(1.1, a1.run)
        sevt.void()
        runner.schedule_stop(2.0)
        runner.execute()
        self.assertEqual(0, a1.at)

        a1.at = 0
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        sevt = runner.schedule_repeating_event(1.1, 1, a1.run)
        runner.schedule_event(8, sevt.void)
        runner.schedule_stop(10.0)
        runner.execute()
        # if the void correctly stops rescheduling, then
        # there should be nothing the queue
        self.assertEqual(0, len(runner.schedule.queue))
        self.assertEqual(7.1, a1.at)

        a2 = Agent2()
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        runner.schedule_stop(2.0)
        eevt = runner.schedule_end_event(a2.run)
        runner.execute()
        self.assertEqual(2.0, a2.at)

        a2.at = 0
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        runner.schedule_stop(2.0)
        eevt = runner.schedule_end_event(a2.run)
        eevt.void()
        runner.execute()
        self.assertEqual(0.0, a2.at)

        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        evt = runner.schedule_stop(2.0)
        f = gen_reset_stop(evt, runner, 10)
        runner.schedule_event(1, f)
        runner.execute()
        self.assertEqual(runner.tick(), 10)

    def test_evt_args(self):
        a2 = Agent2()
        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        runner.schedule_stop(2.0)
        evt = schedule.create_arg_evt(a2.op, 1, 2)
        # evt2 = runner.create_arg_evt(a2.op, 3, 10, op='*')
        runner.schedule_event(1, evt)
        runner.execute()
        self.assertEqual(3, a2.result)

        runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
        runner.schedule_stop(2.0)
        evt = schedule.create_arg_evt(a2.op, 3, 10, op='*')
        runner.schedule_event(1, evt)
        a2.result = 0
        runner.execute()
        self.assertEqual(30, a2.result)

    def test_priority(self):
        for _ in range(100):
            # Run repeatedly to make sure not random scheduling and only works by chance
            runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
            a = Agent(runner.schedule)
            runner.schedule_stop(2.0)
            evt = schedule.create_arg_evt(a.tag, 'a')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=1.0)
            evt = schedule.create_arg_evt(a.tag, 'b')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=2.0)
            evt = schedule.create_arg_evt(a.tag, 'c')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=3.0)
            runner.execute()
            self.assertEqual(['a', 'b', 'c'], a.evts)

    def test_priority_repeating(self):
        for _ in range(100):
            # Run repeatedly to make sure not random scheduling and only works by chance
            runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
            a = Agent(runner.schedule)
            runner.schedule_stop(2.0)
            evt = schedule.create_arg_evt(a.tag, 'a')
            runner.schedule_repeating_event(1.0, 0.4, evt, priority_type=PriorityType.BY_PRIORITY, priority=1.0)
            evt = schedule.create_arg_evt(a.tag, 'b')
            runner.schedule_repeating_event(1.0, 0.4, evt, priority_type=PriorityType.BY_PRIORITY, priority=2.0)
            evt = schedule.create_arg_evt(a.tag, 'c')
            runner.schedule_repeating_event(1.0, 0.4, evt, priority_type=PriorityType.BY_PRIORITY, priority=3.0)
            runner.execute()
            self.assertEqual(['a', 'b', 'c'] * 3, a.evts)

    def test_first_last(self):
        for _ in range(100):
            # Run repeatedly to make sure not random scheduling and only works by chance
            runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
            a = Agent(runner.schedule)
            runner.schedule_stop(2.0)
            evt = schedule.create_arg_evt(a.tag, 'a')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=1.0)
            evt = schedule.create_arg_evt(a.tag, 'b')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=2.0)
            evt = schedule.create_arg_evt(a.tag, 'c')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=3.0)

            for i in range(10):
                evt = schedule.create_arg_evt(a.tag, i)
                runner.schedule_event(1.0, evt, priority_type=PriorityType.FIRST)

            for i in range(20, 30):
                evt = schedule.create_arg_evt(a.tag, i)
                runner.schedule_event(1.0, evt, priority_type=PriorityType.LAST)

            runner.execute()
            self.assertEqual([x for x in range(10)] + ['a', 'b', 'c'] + [x for x in range(20, 30)], a.evts)

    def test_random_priority(self):
        counts = {}
        for _ in range(500):
            runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
            a = Agent(runner.schedule)
            runner.schedule_stop(2.0)
            evt = schedule.create_arg_evt(a.tag, 'a')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.RANDOM)
            evt = schedule.create_arg_evt(a.tag, 'b')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.RANDOM)
            evt = schedule.create_arg_evt(a.tag, 'c')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.RANDOM)
            runner.execute()
            k = tuple(a.evts)
            if k in counts:
                counts[k] += 1
            else:
                counts[k] = 1
        self.assertTrue(len(counts) > 3)

    def test_mixed(self):
        d_idxs = set()
        e_idxs = set()
        for _ in range(100):
            # Run repeatedly to make sure not random scheduling and only works by chance
            runner = schedule.init_schedule_runner(MPI.COMM_WORLD)
            a = Agent(runner.schedule)
            runner.schedule_stop(2.0)
            evt = schedule.create_arg_evt(a.tag, 'a')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=1.0)
            evt = schedule.create_arg_evt(a.tag, 'b')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=2.0)
            evt = schedule.create_arg_evt(a.tag, 'c')
            runner.schedule_event(1.0, evt, priority_type=PriorityType.BY_PRIORITY, priority=3.0)

            evt = schedule.create_arg_evt(a.tag, 'd')
            runner.schedule_event(1.0, evt)
            evt = schedule.create_arg_evt(a.tag, 'e')
            runner.schedule_event(1.0, evt)

            for i in range(10):
                evt = schedule.create_arg_evt(a.tag, i)
                runner.schedule_event(1.0, evt, priority_type=PriorityType.FIRST)

            for i in range(20, 30):
                evt = schedule.create_arg_evt(a.tag, i)
                runner.schedule_event(1.0, evt, priority_type=PriorityType.LAST)

            runner.execute()
            self.assertEqual([x for x in range(10)], a.evts[:10])
            self.assertEqual([x for x in range(20, 30)], a.evts[-10:])
            self.assertEqual(25, len(a.evts))
            mixed = a.evts[10:15]
            for v in ['a', 'b', 'c', 'd', 'e']:
                self.assertTrue(v in mixed)

            self.assertTrue(mixed.index('a') < mixed.index('b'))
            self.assertTrue(mixed.index('b') < mixed.index('c'))
            d_idxs.add(mixed.index('d'))
            e_idxs.add(mixed.index('e'))

            # + ['a', 'b', 'c'] + , a.evts)
        self.assertTrue(len(d_idxs) > 2)
        self.assertTrue(len(e_idxs) > 2)


if __name__ == "__main__":
    unittest.main()
