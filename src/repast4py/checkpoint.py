# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""
This module includes classes and functions for checkpointing a repast4py simulation.
"""

from typing import Union, Dict, Callable, Iterable
from dataclasses import dataclass, field
import warnings
from mpi4py import MPI
import itertools


from . import random
from . import schedule
# from . import parameters
from .core import Agent


IGNORE_EVT = 0
DEFAULT_TAG = '__default'


@dataclass
class EvtData:
    at: float = 0.0
    order_idx: int = 0
    priority_type: schedule.PriorityType = schedule.PriorityType.RANDOM
    priority: float = 0.0
    interval: float = 0.0
    metadata: Dict = field(default_factory=dict)


class Checkpoint:

    def __init__(self):
        """Creates a Checkpoint instance that can be used to save the simulation."""
        self.checkpoint_at: float = 0.0
        self.random_state = []
        self.schedule_state = {}
        self.agent_state = []
        self.other_state = {}

    def save_random(self):
        """Saves the current random state of the :attr:`random.default_rng`
        generator."""
        state = random.default_rng.bit_generator.state
        self.random_state = [random.seed, state]

    def restore_random(self):
        """Restores the checkpointed random state, setting
        :attr:`random.seed` and :attr:`random.default_rng` to the
        saved state.
        """
        random.seed = self.random_state[0]
        random.default_rng.bit_generator.state = self.random_state[1]

    # Checkpoint parameters, or assume user provides coherent ones?
    # def save_parameters(self, saver: Callable = lambda x: x):
    #     """Save parameters.params to the checkpoint object.

    #     Args:
    #         saver: an optional Callable to which each parameter value will be passed
    #             allowing any non-pickeable values to represented as a pickleable
    #             value. By default each parameters value will be saved as is.

    #     """
    #     params = parameters.params
    #     self.param_state = {k: saver(v) for k, v in params}

    # def restore_parameters(self, restorer: Callable = lambda x: x):
    #     params = parameters.params

    def _to_evt_data(self, evt: Union[schedule.OneTimeEvent, schedule.RepeatingEvent]):
        evt_data = EvtData()
        meta = {}
        for k, v in evt.metadata.items():
            if isinstance(v, Callable):
                meta[k] = v(evt.evt)
            else:
                meta[k] = v
        evt_data.metadata = meta
        evt_data.at = evt.at
        evt_data.order_idx = evt.order_idx
        evt_data.priority_type = evt.priority_type
        evt_data.priority = evt.priority
        if meta['__type'] == schedule.EvtType.REPEATING:
            evt_data.interval = evt.interval

        return evt_data

    def save_agents(self, agents: Iterable[Agent]):
        for agent in agents:
            self.agent_state.append(agent.save())

    def restore_agents(self, restore_agent: Callable):
        for agent_state in self.agent_state:
            agent = restore_agent(agent_state)
            yield agent

    def save_schedule(self):
        runner = schedule.runner()
        ss = self.schedule_state
        ss['last_count'] = runner.schedule.last_count
        ss['tick'] = runner.schedule.tick
        evts = [self._to_evt_data(evt) for evt in runner.end_evts if evt.metadata['__type'] != schedule.EvtType.VOID]
        # item: (at, count, evt) tuple
        evts += [self._to_evt_data(item[2]) for item in runner.schedule.queue if item[2].metadata['__type'] != schedule.EvtType.VOID]
        ss['evts'] = evts

    def _schedule_evt(self, runner: schedule.SharedScheduleRunner, evt_type: schedule.EvtType, evt_data: EvtData,
                      evt: Callable):
        # next(schedule.counter) in _push_event should now return
        # the serialized order_idx
        runner.schedule.counter = iter((evt_data.order_idx,))
        scheduled_evt = None
        if evt_type == schedule.EvtType.ONE_TIME:
            scheduled_evt = runner.schedule_event(evt_data.at, evt, priority_type=evt_data.priority_type,
                                                  priority=evt_data.priority, metadata=evt_data.metadata)
        elif evt_type == schedule.EvtType.REPEATING:
            scheduled_evt = runner.schedule_repeating_event(evt_data.at, evt_data.interval, evt,
                                                            priority_type=evt_data.priority_type,
                                                            priority=evt_data.priority,
                                                            metadata=evt_data.metadata)
        # elif evt_type == schedule.EvtType.STOP:
        #     scheduled_evt = runner.schedule_stop(evt_data.at)
        elif evt_type == schedule.EvtType.END:
            scheduled_evt = runner.schedule_end_event(evt, metadata=evt_data.metadata)

        return scheduled_evt

    def restore_schedule(self, comm: MPI.Intracomm, evt_creator: Callable,
                         evt_processor: Callable = lambda x, y: None,
                         tag=DEFAULT_TAG, initialize: bool = True):
        """
        Initializes the schedule runner.
        Args:
            checkpoint: the Checkpoint containing the schedule data
            evt_creator: callable (function, method, etc.) that takes the metata
                dictionary associated with an event as an argument and returns
                the Callable to schedule.
            comm: the communicator in which this is running
            initialize: if true, then the schedule runner is initialized with
                schedule.init_schedule_runner, otherwise the schedule runner
                is not initialized.
        """
        schedule_state = self.schedule_state
        if initialize:
            runner = schedule.init_schedule_runner(comm)
            runner.schedule.tick = schedule_state['tick']
        else:
            runner = schedule.runner()

        for evt_data in schedule_state['evts']:
            metadata = evt_data.metadata
            evt_tag = metadata.get('tag', DEFAULT_TAG)
            if evt_tag == tag:
                evt_type = metadata['__type']
                evt = None
                if evt_type != schedule.EvtType.STOP:
                    evt = evt_creator(metadata)
                    if evt is None:
                        warnings.warn(f"No callable evt returned for {metadata}")
                if evt != IGNORE_EVT:
                    scheduled_evt = self._schedule_evt(runner, evt_type, evt_data, evt)
                    evt_processor(metadata, scheduled_evt)

        last_count = schedule_state['last_count']
        runner.schedule.counter = itertools.count(last_count + 1)
        runner.schedule.last_count = last_count

        return runner

    def save(self, key, value):
        self.other_state[key] = value

    def restore(self, key, restorer: Callable, *args):
        return restorer(self.other_state[key], *args)
