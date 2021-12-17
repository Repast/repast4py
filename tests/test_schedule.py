import unittest
import sys
import os
from mpi4py import MPI

try:
    from repast4py import schedule
except ModuleNotFoundError:
    sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))
    from repast4py import schedule


class Agent:

    def __init__(self, schedule):
        self.sched = schedule
        self.at = 0

    def run(self):
        self.at = self.sched.tick


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
        self.assertEqual(0, a3.at)

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
        self.assertEqual(0, a3.at)

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



if __name__ == "__main__":
    unittest.main()
