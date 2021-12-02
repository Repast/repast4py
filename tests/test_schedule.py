import unittest
import sys
import os

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

    def run(self):
        self.at = schedule.runner().tick()


class Agent3:

    def __init__(self, schedule):
        self.sched = schedule
        self.ticks = []

    def run(self):
        self.ticks.append(self.sched.tick)
        if len(self.ticks) < 3:
            self.sched.schedule_event(self.sched.tick, self.run)


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


if __name__ == "__main__":
    unittest.main()
