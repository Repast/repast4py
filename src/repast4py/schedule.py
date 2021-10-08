# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

import heapq
import itertools
from typing import Callable, List

from mpi4py import MPI


class ScheduledEvent:
    """Base class for all scheduled events.

    Args:
        at: the time of the event.
        evt: the callable to execute when this event is executed.
    """

    def __init__(self, at: float, evt: Callable):
        self.evt = evt
        self.at = at

    def __call__(self):
        self.evt()

    def reschedule(self, queue, sequence_count):
        pass


class RepeatingEvent(ScheduledEvent):
    """Scheduled event that repeats at some specified interval.

    Args:
        at: the time of the event.
        interval: the interval at which to repeat.
        evt: the callable to execute when this event is executed.

    """
    def __init__(self, at: float, interval: float, evt: Callable):
        super().__init__(at, evt)
        self.interval = interval

    def reschedule(self, queue: List, sequence_count: int):
        """Reschedules this event to occur after it's iterval has passed.

        Args:
            queue: the priority queue list to schedule this event on
            sequence_count: the original sequeuce count for this event.
                The sequence count is used to order events scheduled for the same tick.


        """
        self.at += self.interval
        heapq.heappush(queue, (self.at, sequence_count, self))


class OneTimeEvent(ScheduledEvent):
    """Scheduled event that executes only once.

    Args:
        at: the time of the event.
        evt: the callable to execute when this event is executed.

    """

    def __init__(self, at, evt):
        super().__init__(at, evt)

    def reschedule(self, queue, sequence_count):
        """Null-op as this OneTimeEvent only occurs once.
        """
        pass


class Schedule:
    """Encapsulates a dynamic schedule of events and a method
    for iterating  through the schedule and exectuting those events.

    Events are added to the scheduled for execution at a particular "tick".
    The first valid tick is 0.
    Events will be executed in "tick" order, earliest before latest. Events
    scheduled for the same tick will be executed in the order in which they
    were added. If during the execution of a tick, an event is scheduled
    before the executing tick (i.e., scheduled to occur in the past) then
    that event is ignored.
    """

    def __init__(self):
        self.queue = []
        self.tick = 0
        self.counter = itertools.count()

    def _push_event(self, at: float, evt: ScheduledEvent):
        """Pushes the specified event onto the queue for execution at the specified tick.

        Args:
            at: the time of the event.
            evt: the event to schedule.

        """
        if at >= self.tick:
            count = next(self.counter)
            heapq.heappush(self.queue, (at, count, evt))

    def schedule_event(self, at: float, evt: Callable):
        """Schedules the specified event to execute at the specified tick.

        Args:
            at: the time of the event.
            evt: the Callable to execute when the event occurs.

        """
        scheduled_evt = OneTimeEvent(at, evt)
        self._push_event(at, scheduled_evt)

    def schedule_repeating_event(self, at: float, interval: float, evt: Callable):
        """Schedules the specified event to execute at the specified tick,
        and repeat at the specified interval.

        Args:
            at: the time of the event.
            interval: the interval at which to repeat event execution.
            evt: the Callable to execute when the event occurs.

        """
        scheduled_evt = RepeatingEvent(at, interval, evt)
        self._push_event(at, scheduled_evt)

    def next_tick(self) -> float:
        """Gets the tick of the next scheduled event.

        Returns:
            float: the tick at which the next scheduled event will occur or -1 if nothing is scheduled.

        """
        if len(self.queue) == 0:
            return -1

        return self.queue[0][0]

    def execute(self):
        """Executes this schedule by repeatedly poping the next scheduled events off the queue and
        executing them.
        """
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
    """Encapsulates a dynamic schedule of executable events shared and
    synchronized across processes.

    Events are added to the scheduled for execution at a particular "tick".
    The first valid tick is 0.
    Events will be executed in "tick" order, earliest before latest. Events
    scheduled for the same tick will be executed in the order in which they
    were added. If during the execution of a tick, an event is scheduled
    before the executing tick (i.e., scheduled to occur in the past) then
    that event is ignored. The scheduled is synchronized across process ranks
    by determining the global cross-process minimum next scheduled even time, and executing events
    for that time. In this way, no schedule runs ahead of any other.

    Args:
        comm: the communicator over which this scheduled is shared.
    """

    def __init__(self, comm: MPI.Intracomm):
        self.comm = comm
        self.schedule = Schedule()
        self.next_tick = -1
        self.end_evts = []
        self.go = True

    def schedule_event(self, at: float, evt: Callable):
        """Schedules the specified event to execute at the specified tick.

        Args:
            at: the time of the event.
            evt: the Callable to execute when the event occurs.

        """
        self.schedule.schedule_event(at, evt)
        self.next_tick = self.schedule.next_tick()

    def schedule_repeating_event(self, at: float, interval: float, evt: Callable):
        """Schedules the specified event to execute at the specified tick,
        and repeat at the specified interval.

        Args:
            at: the time of the event.
            interval: the interval at which to repeat event execution.
            evt: the Callable to execute when the event occurs.

        """
        self.schedule.schedule_repeating_event(at, interval, evt)
        self.next_tick = self.schedule.next_tick()

    def schedule_end_event(self, evt: Callable):
        """Schedules the specified callable for execution when the schedule terminates and the
        simulation ends.

        Args:
            evt: the Callable to execute when simulation ends.
        """
        self.end_evts.append(evt)

    def schedule_stop(self, at: float):
        """Schedules the execution of this schedule to stop at the specified tick.

        Args:
            at: the tick at which the schedule will stop.

        """
        self.schedule.schedule_event(at, self.stop)

    def stop(self):
        """Stops scheduled execution. All events scheduled for the current tick will execute and 
        then schedule execution will stop.
        """
        self.go = False

    def tick(self) -> float:
        """Gets the current tick.

        Returns:
            float: the currently executing tick.
        """
        return self.schedule.tick

    def execute(self):
        """Executes this SharedSchedule by repeatedly poping the next scheduled events off the queue and
        executing them.
        """
        while self.go:
            global_next_tick = self.comm.allreduce(self.next_tick, op=MPI.MIN)
            if self.next_tick == global_next_tick:
                self.schedule.execute()
            self.next_tick = self.schedule.next_tick()

        for evt in self.end_evts:
            evt()


__runner = None


def init_schedule_runner(comm: MPI.Intracomm) -> SharedScheduleRunner:
    """Initializes the default schedule runner, a dynamic schedule of executable events shared and
    synchronized across processes.

    Events are added to the scheduled for execution at a particular "tick".
    The first valid tick is 0.
    Events will be executed in "tick" order, earliest before latest. Events
    scheduled for the same tick will be executed in the order in which they
    were added. If during the execution of a tick, an event is scheduled
    before the executing tick (i.e., scheduled to occur in the past) then
    that event is ignored. The scheduled is synchronized across process ranks
    by determining the global cross-process minimum next scheduled even time, and executing events
    for that time. In this way, no schedule runs ahead of any other.

    Args:
        comm: the communicator over which this scheduled is shared.
    Returns:
        SharedScheduleRunner: The default SharedScheduledRunner instance that can be used to 
        schedule events.
    """
    global __runner
    __runner = SharedScheduleRunner(comm)
    return __runner


def runner() -> SharedScheduleRunner:
    """Gets the default schedule runner, a dynamic schedule of executable events shared and
    synchronized across processes.

    Events are added to the scheduled for execution at a particular "tick".
    The first valid tick is 0.
    Events will be executed in "tick" order, earliest before latest. Events
    scheduled for the same tick will be executed in the order in which they
    were added. If during the execution of a tick, an event is scheduled
    before the executing tick (i.e., scheduled to occur in the past) then
    that event is ignored. The scheduled is synchronized across process ranks
    by determining the global cross-process minimum next scheduled even time, and executing events
    for that time. In this way, no schedule runs ahead of any other.

    Returns:
        SharedScheduleRunner: The default SharedScheduledRunner instance that can be used to 
        schedule events.
    """

    if not __runner:
        raise RuntimeError('Schedule runner must be initialized with schedule.init_schedule_runner before being used')
    return __runner
