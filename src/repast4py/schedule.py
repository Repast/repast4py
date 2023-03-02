# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""
This module includes classes and functions for scheduling events in a Repast4py
simulation. Users will typically only use the :class:`repast4py.schedule.SharedScheduleRunner`.
"""

import heapq
import itertools
from typing import Callable, List
import types
from enum import IntEnum

from mpi4py import MPI

from . import random


def _noop_evt():
    pass


def _noop_reschedule(self, queue):
    pass


class PriorityType(IntEnum):
    """Enums used to specify the priority type for scheduled events."""
    FIRST = 0
    LAST = 1
    RANDOM = 2
    BY_PRIORITY = 3


class ScheduledEvent:
    """A callable base class for all scheduled events. Calling
    instances of this class will execute the Callable evt.

    Args:
        at: the time of the event.
        evt: the callable to execute when this event is executed.
        priority_type: a :class:`repast4py.schedule.PriorityType` specifying
            the type of priority used to order events that occur at the same tick.
        priority: when priority_type is PriorityType.BY_PRIORITY, the priority
            value is used to order all the events assigned a BY_PRIORITY priority type.
            Otherwise, this value is ignored. Lower values have a higher priority and will
            execute before those with a higher value.
    """
    def __init__(self, at: float, evt: Callable, priority_type: PriorityType, priority: float):
        self.evt = evt
        self.at = at
        self.order_idx = 0
        self.priority_type = priority_type
        self.priority = priority

    def __call__(self):
        self.evt()

    def reschedule(self, queue):
        """
        Implemented by subclasses.
        """
        pass

    def void(self):
        """Voids this ScheduledEvent so that it will not execute"""
        pass


class RepeatingEvent(ScheduledEvent):
    """Scheduled event that repeats at some specified interval.

    Args:
        at: the time of the event.
        interval: the interval at which to repeat.
        evt: the callable to execute when this event is executed.
        priority_type: a :class:`repast4py.schedule.PriorityType` specifying
            the type of priority used to order events that occur at the same tick.
        priority: when priority_type is PriorityType.BY_PRIORITY, the priority
            value is used to order all the events assigned a BY_PRIORITY priority type.
            Otherwise, this value is ignored. Lower values have a higher priority and will
            execute before those with a higher value.
    """
    def __init__(self, at: float, interval: float, evt: Callable, priority_type: PriorityType = PriorityType.RANDOM,
                 priority: float = float('nan')):
        super().__init__(at, evt, priority_type, priority)
        self.interval = interval

    def reschedule(self, queue: List):
        """Reschedules this event to occur after its interval has passed.

        Args:
            queue: the priority queue list to schedule this event on
            sequence_count: the original sequeuce count for this event.
                The sequence count is used to order events scheduled for the same tick.


        """
        self.at += self.interval
        heapq.heappush(queue, (self.at, self.order_idx, self))

    def void(self):
        """Voids this ScheduledEvent so that it will not execute"""
        self.evt = _noop_evt
        # change the reschedule method on the self instance, so
        # it will not reschedule
        self.reschedule = types.MethodType(_noop_reschedule, self)


class OneTimeEvent(ScheduledEvent):
    """Scheduled event that executes only once.

    Args:
        at: the time of the event.
        evt: the callable to execute when this event is executed.
        priority_type: a :class:`repast4py.schedule.PriorityType` specifying
            the type of priority used to order events that occur at the same tick.
        priority: when priority_type is PriorityType.BY_PRIORITY, the priority
            value is used to order all the events assigned a BY_PRIORITY priority type.
            Otherwise, this value is ignored. Lower values have a higher priority and will
            execute before those with a higher value.
    """

    def __init__(self, at: float, evt: Callable, priority_type: PriorityType = PriorityType.RANDOM,
                 priority: float = float('nan')):
        super().__init__(at, evt, priority_type, priority)

    def reschedule(self, queue):
        """Null-op as this OneTimeEvent only occurs once.
        """
        pass

    def void(self):
        """Voids this ScheduledEvent so that it will not execute"""
        self.evt = _noop_evt


class ScheduleGroup:
    """A ScheduleGroup is used internally by repast4py to order events for execution.
    """

    def __init__(self):
        self.first = []
        self.last = []
        self.random = []
        self.prioritized = []
        self.evts_to_execute = []
        self.evts_added_during_execution = []
        self.executing = False
        self.interrupted = False
        self.size_ = 0

    def reset(self):
        self.first.clear()
        self.last.clear()
        self.random.clear()
        self.prioritized.clear()
        self.evts_added_during_execution.clear()
        self.size_ = 0
        self.interrupted = False

    @property
    def size(self) -> int:
        return self.size_

    def add_evt(self, evt: ScheduledEvent):
        if self.executing:
            self.evts_added_during_execution.append(evt)
            self.interrupted = True
        else:
            self._add_evt(evt)

    def _add_evt(self, evt: ScheduledEvent):
        p_type = evt.priority_type
        if p_type == PriorityType.RANDOM:
            self.random.append(evt)
        elif p_type == PriorityType.FIRST:
            self.first.append(evt)
        elif p_type == PriorityType.LAST:
            self.last.append(evt)
        else:
            self.prioritized.append(evt)
        self.size_ += 1

    def sort(self):
        # sort according to the order of insertion
        # so we have a predictable starting point for randomization
        self.random.sort(key=lambda evt: evt.order_idx)
        self.prioritized.sort(key=lambda evt: evt.order_idx)
        # first and last order_idx determines execution order
        self.first.sort(key=lambda evt: evt.order_idx)
        self.last.sort(key=lambda evt: evt.order_idx)

        if self.prioritized:
            rng = random.default_rng
            # add random actions randomly to prioritized actions
            self.prioritized.sort(key=lambda evt: evt.priority)
            for evt in self.random:
                idx = rng.integers(0, len(self.prioritized), endpoint=True)
                self.prioritized.insert(idx, evt)
        else:
            # shuffle random and add to actions
            random.default_rng.shuffle(self.random)
            self.prioritized.extend(self.random)

        self.random.clear()

    def _add_during_ex_evts(self):
        for evt in self.evts_added_during_execution:
            self._add_evt(evt)
        self.sort()
        self.evts_added_during_execution.clear()

    def execute_evts(self, evts: List, queue: List) -> bool:
        interrupted = False

        next_idx = 0
        for evt in evts:
            self.size_ -= 1
            evt()
            evt.reschedule(queue)
            next_idx += 1
            if self.interrupted:
                interrupted = True
                break

        del evts[:next_idx]

        if interrupted:
            self._add_during_ex_evts()
            self.interrupted = False

        return interrupted

    def execute(self, queue: List):
        self.executing = True
        interrupted = self.execute_evts(self.first, queue)
        if interrupted:
            return
        interrupted = self.execute_evts(self.prioritized, queue)
        if interrupted:
            return
        interrupted = self.execute_evts(self.last, queue)
        if interrupted:
            return
        self.executing = False


class Schedule:
    """Encapsulates a dynamic schedule of events and a method
    for iterating through that schedule and exectuting those events.

    Events are added to the schedule for execution at a particular *tick*, and
    with a particular priority. The first valid tick is 0.
    Events will be executed in tick order, earliest before latest. When
    multiple events are scheduled for the same tick, the events' priorities
    will be used to determine the order of execution within that tick.
    If an event is scheduled before the current executing tick (i.e., scheduled to occur in the past)
    then that event is ignored.
    """

    def __init__(self):
        self.queue = []
        # 0 is first valid tick
        self.tick = -0.000000000000000000000000000000000000001
        self.counter = itertools.count()
        self.executing_group = ScheduleGroup()

    def _push_event(self, at: float, evt: ScheduledEvent):
        """Pushes the specified event onto the queue for execution at the specified tick.

        Args:
            at: the time of the event.
            evt: the event to schedule.

        """
        if at >= self.tick:
            count = next(self.counter)
            evt.order_idx = count
            if at == self.tick:
                self.executing_group.add_evt(evt)
            else:
                heapq.heappush(self.queue, (at, count, evt))

    def schedule_event(self, at: float, evt: Callable, priority_type: PriorityType = PriorityType.RANDOM,
                       priority: float = float('nan')) -> ScheduledEvent:
        """Schedules the specified event to execute at the specified tick with
        the specified priority. By default, events are scheduled with a random priority type.

        An event's priority_type and priority determines when it will execute
        with respect to other events scheduled for the same tick. The priority types are:

        * PriorityType.FIRST - events will execute before those with other PriorityTypes.
            All events with a FIRST priority type will execute in the order in which they are scheduled
            with respect to other FIRST priority type events.
        * PriorityType.RANDOM - events will execute in a random order, after the FIRST
            priority type events, and before the LAST priority type events. If there are BY_PRIORITY
            events scheduled for the same tick as RANDOM events, the RANDOM events will be shuffled at
            random into the ordered BY_PRIORITY events.
        * PriorityType.BY_PRIORITY - events will execute in the order specified by the
            priority parameter (lower values are higher priority), and after any FIRST priority events
            and before any LAST priority events. If there are RANDOM priority events scheduled
            for the same tick as BY_PRIORITY events, those will be shuffled at random into the
            ordered BY_PRIORITY events.
        * PriorityType.LAST - events will execute after those with other priority types.
            All events with a LAST priority type will execute in the order in which they are scheduled
            with respect to other LAST priority type events.

        Args:
            at: the time of the event.
            evt: the Callable to execute when the event occurs.
            priority_type: a :class:`repast4py.schedule.PriorityType` specifying
                the type of priority used to order events that occur at the same tick.
            priority: when priority_type is PriorityType.BY_PRIORITY, the priority
                value is used to order all the events assigned a BY_PRIORITY priority type.
                Otherwise, this value is ignored. Lower values have a higher priority and will
                execute before those with a higher value.

        Returns:
            The ScheduledEvent instance that was scheduled for execution.
        """
        scheduled_evt = OneTimeEvent(at, evt, priority_type, priority)
        self._push_event(at, scheduled_evt)
        return scheduled_evt

    def schedule_repeating_event(self, at: float, interval: float, evt: Callable,
                                 priority_type: PriorityType = PriorityType.RANDOM,
                                 priority: float = float('nan')) -> ScheduledEvent:
        """Schedules the specified event to execute at the specified tick with the specified
        priority, and to repeat at the specified interval. By default, events are scheduled with
        a random priority type.

        An event's priority_type and priority determines when it will execute
        with respect to other events scheduled for the same tick. The priority types are:

        * PriorityType.FIRST - events will execute before those with other PriorityTypes.
            All events with a FIRST priority type will execute in the order in which they are scheduled
            with respect to other FIRST priority type events.
        * PriorityType.RANDOM - events will execute in a random order, after the FIRST
            priority type events, and before the LAST priority type events. If there are BY_PRIORITY
            events scheduled for the same tick as RANDOM events, the RANDOM events will be shuffled at
            random into the ordered BY_PRIORITY events.
        * PriorityType.BY_PRIORITY - events will execute in the order specified by the
            priority parameter (lower values are higher priority), and after any FIRST priority events
            and before any LAST priority events. If there are RANDOM priority events scheduled
            for the same tick as BY_PRIORITY events, those will be shuffled at random into the
            ordered BY_PRIORITY events.
        * PriorityType.LAST - events will execute after those with other priority types.
            All events with a LAST priority type will execute in the order in which they are scheduled
            with respect to other LAST priority type events.

        Args:
            at: the time of the event.
            interval: the interval at which to repeat event execution.
            evt: the Callable to execute when the event occurs.
            priority_type: a :class:`repast4py.schedule.PriorityType` specifying
                the type of priority used to order events that occur at the same tick.
            priority: when priority_type is PriorityType.BY_PRIORITY, the priority
                value is used to order all the events assigned a BY_PRIORITY priority type.
                Otherwise, this value is ignored. Lower values have a higher priority and will
                execute before those with a higher value.

        Returns:
            The ScheduledEvent instance that was scheduled for execution.
        """
        scheduled_evt = RepeatingEvent(at, interval, evt, priority_type, priority)
        self._push_event(at, scheduled_evt)
        return scheduled_evt

    def next_tick(self) -> float:
        """Gets the tick of the next scheduled event.

        Returns:
            float: the tick at which the next scheduled event will occur or -1 if nothing is scheduled.
        """
        if len(self.queue) == 0:
            return -1

        return self.queue[0][0]

    def execute(self):
        """Executes this schedule by repeatedly popping the next scheduled events off the queue and
        executing them.
        """
        if len(self.queue) > 0:
            # fill the executing group
            self.executing_group.reset()
            self.tick, _, evt = self.queue[0]
            go = True
            while go:
                heapq.heappop(self.queue)
                self.executing_group.add_evt(evt)
                if len(self.queue) == 0:
                    go = False
                else:
                    next_tick, _, evt = self.queue[0]
                    go = next_tick == self.tick

            # sort the group into ordinary, random etc.
            self.executing_group.sort()
            while self.executing_group.size > 0:
                # execute the actions in the group in the sorted order
                self.executing_group.execute(self.queue)


def create_arg_evt(evt: Callable, *args, **kwargs) -> Callable:
    """Creates a new Callable that will call the specified Callable, passing
    it the specified arguments when it is executed as part of a schedule.
    The returned Callable can be scheduled using the SharedScheduleRunner's schedule_* methods.

    Args:
        evt: the Callable to execute
        args: the positional arguments to pass to the evt Callable
        kwargs: the keyword arguments to pass to the evt Callable.

    Returns:
        A new Callable that is compatible with SharedScheduleRunner's schedule_* methods arguments.
    """
    def f():
        evt(*args, **kwargs)
    return f


class SharedScheduleRunner:
    """Encapsulates a dynamic schedule of executable events shared and
    synchronized across processes.

    Events are added to the schedule for execution at a particular *tick*, and
    with a particular priority. The first valid tick is 0.
    Events will be executed in tick order, earliest before latest. When
    multiple events are scheduled for the same tick, the events' priorities
    will be used to determine the order of execution within that tick.
    If an event is scheduled before the current executing tick (i.e., scheduled to occur in the past)
    then that event is ignored. The scheduled is synchronized across process ranks
    by determining the global cross-process minimum next scheduled event time, and executing
    only the events schedule for that time. In this way, no schedule runs ahead of any other.

    Args:
        comm: the communicator over which this schedule is shared.
    """

    def __init__(self, comm: MPI.Intracomm):
        self.comm = comm
        self.schedule = Schedule()
        self.next_tick = -1
        self.end_evts = []
        self.go = True

    def schedule_event(self, at: float, evt: Callable,
                       priority_type: PriorityType = PriorityType.RANDOM,
                       priority: float = float('nan')) -> ScheduledEvent:
        """Schedules the specified event to execute at the specified tick with
        the specified priority. By default, events are scheduled with a random priority type.

        An event's priority_type and priority determines when it will execute
        with respect to other events scheduled for the same tick. The priority types are:

        * PriorityType.FIRST - events will execute before those with other PriorityTypes.
            All events with a FIRST priority type will execute in the order in which they are scheduled
            with respect to other FIRST priority type events.
        * PriorityType.RANDOM - events will execute in a random order, after the FIRST
            priority type events, and before the LAST priority type events. If there are BY_PRIORITY
            events scheduled for the same tick as RANDOM events, the RANDOM events will be shuffled at
            random into the ordered BY_PRIORITY events.
        * PriorityType.BY_PRIORITY - events will execute in the order specified by the
            priority parameter (lower values are higher priority), and after any FIRST priority events
            and before any LAST priority events. If there are RANDOM priority events scheduled
            for the same tick as BY_PRIORITY events, those will be shuffled at random into the
            ordered BY_PRIORITY events.
        * PriorityType.LAST - events will execute after those with other priority types.
            All events with a LAST priority type will execute in the order in which they are scheduled
            with respect to other LAST priority type events.

        Args:
            at: the time of the event.
            evt: the Callable to execute when the event occurs.
            priority_type: a :class:`repast4py.schedule.PriorityType` specifying
                the type of priority used to order events that occur at the same tick.
            priority: when priority_type is PriorityType.BY_PRIORITY, the priority
                value is used to order all the events assigned a BY_PRIORITY priority type.
                Otherwise, this value is ignored. Lower values have a higher priority and will
                execute before those with a higher value.

        Returns:
            The ScheduledEvent instance that was scheduled for execution.
        """

        sch_evt = self.schedule.schedule_event(at, evt, priority_type, priority)
        self.next_tick = self.schedule.next_tick()
        return sch_evt

    def schedule_repeating_event(self, at: float, interval: float, evt: Callable,
                                 priority_type: PriorityType = PriorityType.RANDOM,
                                 priority: float = float('nan')) -> ScheduledEvent:
        """Schedules the specified event to execute at the specified tick with the specified
        priority, and to repeat at the specified interval. By default, events are scheduled with
        a random priority type.

        An event's priority_type and priority determines when it will execute
        with respect to other events scheduled for the same tick. The priority types are:

        * PriorityType.FIRST - events will execute before those with other PriorityTypes.
            All events with a FIRST priority type will execute in the order in which they are scheduled
            with respect to other FIRST priority type events.
        * PriorityType.RANDOM - events will execute in a random order, after the FIRST
            priority type events, and before the LAST priority type events. If there are BY_PRIORITY
            events scheduled for the same tick as RANDOM events, the RANDOM events will be shuffled at
            random into the ordered BY_PRIORITY events.
        * PriorityType.BY_PRIORITY - events will execute in the order specified by the
            priority parameter (lower values are higher priority), and after any FIRST priority events
            and before any LAST priority events. If there are RANDOM priority events scheduled
            for the same tick as BY_PRIORITY events, those will be shuffled at random into the
            ordered BY_PRIORITY events.
        * PriorityType.LAST - events will execute after those with other priority types.
            All events with a LAST priority type will execute in the order in which they are scheduled
            with respect to other LAST priority type events.

        Args:
            at: the time of the event.
            interval: the interval at which to repeat event execution.
            evt: the Callable to execute when the event occurs.
            priority_type: a :class:`repast4py.schedule.PriorityType` specifying
                the type of priority used to order events that occur at the same tick.
            priority: when priority_type is PriorityType.BY_PRIORITY, the priority
                value is used to order all the events assigned a BY_PRIORITY priority type.
                Otherwise, this value is ignored. Lower values have a higher priority and will
                execute before those with a higher value.

        Returns:
            The ScheduledEvent instance that was scheduled for execution.
        """
        sch_evt = self.schedule.schedule_repeating_event(at, interval, evt, priority_type, priority)
        self.next_tick = self.schedule.next_tick()
        return sch_evt

    def schedule_end_event(self, evt: Callable) -> ScheduledEvent:
        """Schedules the specified event (a Callable) for execution when the schedule terminates and the
        simulation ends.

        Args:
            evt: the Callable to execute when simulation ends.
        Returns:
            The ScheduledEvent instance that was scheduled to execute at the end.
        """
        sch_evt = OneTimeEvent(float('inf'), evt, )
        self.end_evts.append(sch_evt)
        return sch_evt

    def schedule_stop(self, at: float) -> ScheduledEvent:
        """Schedules the execution of this schedule to stop at the specified tick.

        Args:
            at: the tick at which the schedule will stop.
        Returns:
            The ScheduledEvent instance that executes the stop event.
        """
        sch_evt = self.schedule.schedule_event(at, self.stop)
        return sch_evt

    def stop(self):
        """Stops schedule execution. All events scheduled for the current tick will execute and
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
        """Executes this SharedSchedule by repeatedly popping the next scheduled events off the queue and
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

    Events are added to the scheduled for execution at a particular *tick*.
    The first valid tick is 0.
    Events will be executed in tick order, earliest before latest. Events
    scheduled for the same tick will be executed in the order in which they
    were added. If during the execution of a tick, an event is scheduled
    before the executing tick (i.e., scheduled to occur in the past) then
    that event is ignored. The scheduled is synchronized across process ranks
    by determining the global cross-process minimum next scheduled event time, and executing
    only the events schedule for that time. In this way, no schedule runs ahead of any other.

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

    Returns:
        SharedScheduleRunner: The default SharedScheduledRunner instance that can be used to
        schedule events.
    """

    if not __runner:
        raise RuntimeError('Schedule runner must be initialized with schedule.init_schedule_runner before being used')
    return __runner
