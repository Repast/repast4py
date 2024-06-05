# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""
This module includes classes and functions for checkpointing a repast4py simulation.
simulation.
"""

from typing import Union, List, Dict, Callable, Iterable
from dataclasses import dataclass, field
from os import PathLike
import pickle
import warnings
from mpi4py import MPI

from . import random
from . import schedule
from .core import Agent


IGNORE_EVT = 0


@dataclass
class Checkpoint:

    checkpoint_at: float = 0.0
    random_state: List = field(default_factory=list)
    schedule_state: Dict = field(default_factory=dict)
    agent_state:  List = field(default_factory=list)


@dataclass
class EvtData:
    at: float = 0.0
    order_idx: int = 0
    priority_type: schedule.PriorityType = schedule.PriorityType.RANDOM
    priority: float = 0.0
    interval: float = 0.0
    metadata: Dict = field(default_factory=dict)


def save_random(checkpoint: Checkpoint):
    state = random.default_rng.bit_generator.state
    checkpoint.random_state = [random.seed, state]


def restore_random(checkpoint: Checkpoint):
    random.seed = checkpoint.random_state[0]
    random.default_rng.bit_generator.state = checkpoint.random_state[1]


def to_evt_data(evt: Union[schedule.OneTimeEvent, schedule.RepeatingEvent]):
    evt_data = EvtData()
    meta = evt.metadata
    evt_data.metadata = meta
    evt_data.at = evt.at
    evt_data.order_idx = evt.order_idx
    evt_data.priority_type = evt.priority_type
    evt_data.priority = evt.priority
    if meta['__type'] == schedule.EvtType.REPEATING:
        evt_data.interval = evt.interval

    return evt_data


def save_agents(checkpoint: Checkpoint, agents: Iterable[Agent]):
    for agent in agents:
        checkpoint.agent_state.append(agent.save())


def restore_agents(checkpoint: Checkpoint, restore_agent: Callable):
    for agent_state in checkpoint.agent_state:
        agent = restore_agent(agent_state)
        yield agent


def save_schedule(checkpoint: Checkpoint):
    runner = schedule.runner()
    ss = checkpoint.schedule_state
    ss['counter'] = runner.schedule.counter
    ss['tick'] = runner.schedule.tick
    evts = [to_evt_data(evt) for evt in runner.end_evts if evt.metadata['__type'] != schedule.EvtType.VOID]
    # item: (at, count, evt) tuple
    evts += [to_evt_data(item[2]) for item in runner.schedule.queue if item[2].metadata['__type'] != schedule.EvtType.VOID]
    ss['evts'] = evts


def _schedule_evt(runner: schedule.SharedScheduleRunner, evt_type: schedule.EvtType, evt_data: EvtData,
                  evt: Callable):
    # next(schedule.counter) in _push_event should now return
    # the serialized order_idx
    runner.schedule.counter = iter((evt_data.order_idx,))
    if evt_type == schedule.EvtType.ONE_TIME:
        runner.schedule_event(evt_data.at, evt, priority_type=evt_data.priority_type,
                              priority=evt_data.priority, metadata=evt_data.metadata)
    elif evt_type == schedule.EvtType.REPEATING:
        runner.schedule_repeating_event(evt_data.at, evt_data.interval, evt, priority_type=evt_data.priority_type,
                                        priority=evt_data.priority, metadata=evt_data.metadata)
    elif evt_type == schedule.EvtType.STOP:
        runner.schedule_stop(evt_data.at)
    elif evt_type == schedule.EvtType.END:
        runner.schedule_end_event(evt, metadata=evt_data.metadata)


def restore_schedule(checkpoint: Checkpoint, evt_creator: Callable, comm: MPI.Intracomm):
    """
    Initializes the schedule runner.
    Args:
        evt_creator: callable (function, method, etc.) that takes the metata
            dictionary associated with an event as an argument and returns
            the an event to schedule.
    """
    schedule_state = checkpoint.schedule_state
    runner = schedule.init_schedule_runner(comm)
    runner.schedule.tick = schedule_state['tick']

    for evt_data in schedule_state['evts']:
        evt_type = evt_data.metadata['__type']
        evt = None
        if evt_type != schedule.EvtType.STOP:
            evt = evt_creator(evt_data.metadata)
            if evt is None:
                warnings.warn(f"No callable evt returned for {evt_data.metadata}")
        if evt != IGNORE_EVT:
            _schedule_evt(runner, evt_type, evt_data, evt)

    runner.schedule.counter = schedule_state['counter']

    return runner


def save(fname: Union[str, PathLike], comm: MPI.Intracomm):
    checkpoint = Checkpoint()
    save_random(checkpoint)

    with open(fname, 'wb') as fout:
        pickle.dump(checkpoint, fout)
