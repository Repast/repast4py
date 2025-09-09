# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""
This module includes classes and functions for checkpointing a repast4py simulation.
"""

from typing import Union, Dict, Callable, Iterable, Tuple, Any
from dataclasses import dataclass, field
import warnings
import itertools

from mpi4py import MPI



from . import random
from . import schedule
# from . import parameters
from .core import Agent
from .context import SharedContext
from .network import SharedNetwork, DirectedSharedNetwork, UndirectedSharedNetwork


IGNORE_EVT: int = 0
"""Return value indicating that a saved evt should be ignored
when restoring the schedule."""

DEFAULT_TAG: str = '__default'
"""The default tag in saved event metadata dictionaries"""


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
        """Creates a Checkpoint instance that can be used to save the simulation.

        The expectation is that the user can save a Checkpoint instance using the dill
        module to pickle it. Consequently, all the state to save passed to a Checkpoint instance
        must be pickleable using dill. If that is not the case, then translate the state
        into something pickleable (e.g., a string or some bespoke representation).
        """
        self.checkpoint_at: float = 0.0
        self.random_state = []
        self.schedule_state = {}
        self.agent_state = []
        self.other_state = {}
        self.networks = {}

    def save_random(self):
        """Saves the current random state of the :py:mod:`repast4py.random.default_rng`
        generator and the py:mod:`repast4py.random.seed`."""
        state = random.default_rng.bit_generator.state
        self.random_state = [random.seed, state]

    def restore_random(self):
        """Restores the checkpointed random state, setting
        :py:mod:`repast4py.random.seed` and :py:mod:`repast4py.random.default_rng` to the
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
        """Saves the specified agents' state to this Checkpoint.

        This iterates over each agent and stores the result of each agents'
        `save()` method. The tuple returned by `save()` must be pickleable
        using the `dill` module.

        Args:
            agents: the Iterable of Agents to save.
        """
        for agent in agents:
            self.agent_state.append(agent.save())

    def restore_agents(self, restore_agent: Callable):
        """Restores saved agents one by one, returning a generator
        over the restored agents.

        Each saved agent state (see :attr:`save_agents`) is passed
        to the `restore_agent` Callable and the result of that call
        is returned, until there are no more agents to restore.

        Args:
            restore_agent: a Callable that takes agent state and returns
                an Agent.

        Examples:
            Restore an agent implemented in a `MyAgent` class with two attributes.

            >>> def restore_my_agent(agent_state):
                    a = agent_state[0]
                    b = agent_state[1]
                    return MyAgent(a, b)
            >>> for agent in ckp.restore_agents(restore_my_agent):
                    # do something with agent
        """
        for agent_state in self.agent_state:
            agent = restore_agent(agent_state)
            yield agent

    def save_schedule(self):
        """Saves the currently scheduled events to this Checkpoint.

        The state of the :py:mod:`repast4py.schedule.runner` is saved together
        with the currently scheduled events (Python Callables), and their metadata.
        The intention is that user provides enough metadata to reconstruct the scheduled
        Callable. For example, if the Callable is a Python class that updates
        some attribute on some agents, then the metadata would record the agent
        ids of those agents, and the name of the class. That data can then be
        used to re-create the class when the scheduled is restored.
        """
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
        elif evt_type == schedule.EvtType.STOP:
            scheduled_evt = runner.schedule_stop(evt_data.at)
        elif evt_type == schedule.EvtType.END:
            scheduled_evt = runner.schedule_end_event(evt, metadata=evt_data.metadata)

        return scheduled_evt

    def restore_schedule(self, comm: MPI.Intracomm, evt_creator: Callable,
                         evt_processor: Callable = lambda x, y: None,
                         tag=DEFAULT_TAG, initialize: bool = True, restore_stop_at: bool = False):
        """
        Restores the state of the schedule, and by default initializes
        the schedule runner.

        The saved metadata dictionaries for each currently scheduled
        event (see :py:mod:`~repast4py.schedule.SharedScheduleRunner.schedule_event` and
        :py:mod:`~repast4py.schedule.SharedScheduleRunner.schedule_repeating_event`)
        are passed to the `evt_creator` argument which is expected to return a Callable
        that can then be scheduled appropriately. If the `evt_creator` returns :py:mod:`~repast4py.checkpoint.IGNORE_EVT`,
        then the event will not be scheduled as part of restoring the schedule. If an `evt_processor`
        is specified, then each metadata dictonary and the :py:mod:`~repast4py.schedule.ScheduledEvent`
        created when restoring and scheduling saved events is passed to that. This can be useful,
        if, for example, a model needs to cache :py:mod:`~repast4py.schedule.ScheduledEvent` in order
        to `void` them before they are executed. Here an `evt_processor` can add these
        :py:mod:`~repast4py.schedule.ScheduledEvent` to the model's cache.

        Args:
            evt_creator: a Callable (function, method, etc.) that takes the metadata
                dictionary associated with an event as an argument and returns
                the Callable to schedule.
            evt_processor: a Callable that that takes the metadata
                dictionary associated with an event and the :py:mod:`~repast4py.schedule.ScheduledEvent`
                created by the `evt_creator` as arguments.
            comm: the communicator in which this is running
            tag: only saved scheduled events whose metadata dictionary 'tag'
                entry matches this tag will be restored.
            initialize: if true, then the schedule runner is initialized with
                schedule.init_schedule_runner, otherwise the schedule runner
                is not initialized. When restoring, this needs to be True in
                one call to restore_schedule.
            restore_stop_at: if true, then any stop actions will be restored, otherwise
                it is the user's responsibility to schedule a stop after restoring the
                schedule.

        Returns:
            The :py:mod:`repast4py.schedule.runner`.
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
                elif restore_stop_at:
                    if schedule_state['tick'] == evt_data.at:
                        warnings.warn(f"Scheduling stop at first tick {evt_data.at} in restored schedule")
                    evt = runner.stop
                
                if evt != IGNORE_EVT:
                    scheduled_evt = self._schedule_evt(runner, evt_type, evt_data, evt)
                    evt_processor(metadata, scheduled_evt)

        last_count = schedule_state['last_count']
        runner.schedule.counter = itertools.count(last_count + 1)
        runner.schedule.last_count = last_count

        return runner

    def save(self, key, value):
        """Saves arbitrary key value pairs into this Checkpoint.

        Args:
            key: an identifying key for the saved value.
            value: the value to save.
        """
        self.other_state[key] = value

    def restore(self, key, restorer: Callable, *args):
        """Restores a value identified by the specified key using the
        specified Callable.

        Args:
            key: the identifying key of the saved value to restore.
            restorer: a Callable that takes the saved value associated
                with the key, and returns the restored value.
            args: additional optional arguments that are passed to the restorer
                in addition to the saved value.

        Returns:
            The restored value associated with the key.

        Examples:
            Saving and restoring some arbitrary state from a `Model` object
            named `model`.

            >>> ckp = Checkpoint()
            >>> model_props = [model.a, model.b, model.c]
            >>> ckp.save('mprops', model_props)
            >>> def restore_props(prop_data, model):
                    model.a = prop_data[0]
                    model.b = prop_data[1]
                    model.c = prop_data[2]
                    return model
            >>> ckp.restore('mprops', restore_props, model)
        """
        return restorer(self.other_state[key], *args)
    
    def save_network(self, network: SharedNetwork, comm: MPI.Intracomm):
        data = []
        rank = comm.Get_rank()
        if network.is_directed:
            for nd in filter(lambda n: n.local_rank == rank, network.graph.nodes):
                edge_data = [nd.uid]
                edge_data.append([(o.uid, o.local_rank, network.graph[nd][o]) for o in network.graph.successors(nd)])
                edge_data.append([(o.uid, o.local_rank, network.graph[o][nd]) for o in network.graph.predecessors(nd)])
                data.append(edge_data)
        else:
            for nd in filter(lambda n: n.local_rank == rank, network.graph.nodes):
                edge_data = [nd.uid]
                edge_data.append([(n.uid, n.local_rank, network.graph[nd][n]) for n in network.graph.neighbors(nd)])
                data.append(edge_data)

        net_data = {"name": network.name, "directed": network.is_directed,
                    "data": data}
        self.networks["network.name"] = net_data

    def restore_networks(self, context: SharedContext, comm: MPI.Intracomm, create_agent: Callable):
        agent_map = {agent.uid: agent for agent in context.agents()}
        requests = {}
        restorers = []
        for _, network in self.networks.items():
            restorer = NetworkRestorer(context, network, comm)
            restorer.restore(agent_map, requests)
            restorers.append(restorer)

        requested_agents = context.request_agents([request for request in requests.values()],
                                                  create_agent)
        for agent in requested_agents:
            requests[agent.uid] = agent

        for restorer in restorers:
            restorer.create_ghost_edges(agent_map, requests)


class NetworkRestorer:

    def __init__(self, context, network_save, comm):
        directed = network_save["directed"]
        name = network_save["name"]
        self.network = DirectedSharedNetwork(name, comm) if directed else UndirectedSharedNetwork(name, comm)
        context.add_projection(self.network)
        self.network_save = network_save
        self.ghost_edges = []

    def _make_suc_edge(self, node_uid, other_uid, e_attribs):
        return (0, node_uid, other_uid, e_attribs)
    
    def _make_pre_edge(self, node_uid, other_uid, e_attribs):
        return (1, other_uid, node_uid, e_attribs)
    
    def _add_suc_edge(self, node, other, e_attribs):
        self.network.add_edge(node, other, **e_attribs)
    
    def _add_pre_edge(self, node, other, e_attribs):
        self.network.add_edge(other, node, **e_attribs)

    def _add_edges(self, node, others, edge_tuple_maker, edge_adder):
        for other_uid, other_local_rank, e_attributes in others:
            other_node = self.agent_map.get(other_uid)
            if other_node is None:
                self.requests[other_uid] = (other_uid, other_local_rank)
                self.ghost_edges.append(edge_tuple_maker(node.uid, other_uid, e_attributes))
            else:
                edge_adder(node, other_node, e_attributes)

    def create_ghost_edges(self, agent_map: Dict, requests: Dict):
        for edge_t, u, v, e_attribs in self.ghost_edges:
            if edge_t == 0:
                self.network.add_edge(agent_map[u], requests[v], **e_attribs)
            else:
                self.network.add_edge(requests[v], agent_map[u], **e_attribs)

    def restore(self, agent_map, requests: Dict):
        self.agent_map = agent_map
        self.requests = requests
        edges = self.network_save["data"]
        if self.network.is_directed:
            for node_uid, successors, predecessors in edges:
                node = agent_map[node_uid]
                self._add_edges(node, successors, self._make_suc_edge, self._add_suc_edge)
                self._add_edges(node, predecessors, self._make_pre_edge, self._add_pre_edge)
        else:
            for node_uid, others in edges:
                node = agent_map[node_uid]
                self._add_edges(node, others, self._make_suc_edge, self._add_suc_edge)
