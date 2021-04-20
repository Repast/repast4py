import heapq
import itertools
from mpi4py import MPI


class ScheduledEvent:

    def __init__(self, at, evt):
        self.evt = evt
        self.at = at

    def __call__(self):
        self.evt()


class RepeatingEvent(ScheduledEvent):

    def __init__(self, at, interval, evt):
        super().__init__(at, evt)
        self.interval = interval

    def reschedule(self, queue, sequence_count):
        self.at += self.interval
        heapq.heappush(queue, (self.at, sequence_count, self))


class OneTimeEvent(ScheduledEvent):

    def __init__(self, at, evt):
        super().__init__(at, evt)

    def reschedule(self, queue, sequence_count):
        pass


class Schedule:

    def __init__(self):
        self.queue = []
        self.tick = 0
        self.counter = itertools.count()

    def push_event(self, at, evt):
        count = next(self.counter)
        heapq.heappush(self.queue, (at, count, evt))

    def schedule_event(self, at, evt):
        scheduled_evt = OneTimeEvent(at, evt)
        self.push_event(at, scheduled_evt)

    def schedule_repeating_event(self, at, interval, evt):
        scheduled_evt = RepeatingEvent(at, interval, evt)
        self.push_event(at, scheduled_evt)

    def next_tick(self):
        if len(self.queue) == 0:
            return -1

        return self.queue[0][0]

    def execute(self):
        if len(self.queue) > 0:
            self.tick, sequence_count, evt = self.queue[0]
            go = True
            while go:
                heapq.heappop(self.queue)
                evt()
                evt.reschedule(self.queue, sequence_count)
                if len(self.queue) == 0:
                    go = False
                else:
                    next_tick, sequence_count, evt = self.queue[0]
                    go = next_tick == self.tick


class SharedScheduleRunner:

    def __init__(self, comm):
        self.comm = comm
        self.schedule = Schedule()
        self.next_tick = -1
        self.end_evts = []
        self.go = True

    def schedule_event(self, at, evt):
        self.schedule.schedule_event(at, evt)
        self.next_tick = self.schedule.next_tick()

    def schedule_repeating_event(self, at, interval, evt):
        self.schedule.schedule_repeating_event(at, interval, evt)
        self.next_tick = self.schedule.next_tick()

    def schedule_end_event(self, evt):
        self.end_evts.append(evt)

    def schedule_stop(self, at):
        self.schedule.schedule_event(at, self.stop)

    def stop(self):
        self.go = False

    def tick(self):
        return self.schedule.tick

    def execute(self):
        while self.go:
            global_next_tick = self.comm.allreduce(self.next_tick, op=MPI.MIN)
            if self.next_tick == global_next_tick:
                self.schedule.execute()
            self.next_tick = self.schedule.next_tick()

        for evt in self.end_evts:
            evt()
