import collections
import sys
import numpy as np
from ._core import Agent
from .random import default_rng as rng
from typing import Callable, List, Tuple
from dataclasses import dataclass

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@dataclass
class GhostAgent:
    agent: Agent
    # the number of projections that refer to this ghost
    ref_count: int


@dataclass
class GhostedAgent:
    agent: Agent
    # the rank on which the agent is a ghost
    ghost_ranks: List[int]


class AgentManager:
    """Manages local and non-local (ghost) agents as they move
    between processes
    """

    def __init__(self, rank, world_size):
        self._local_agents = collections.OrderedDict()
        # keys are agent uid, vals GhostedAgent and GhostAgent
        self._ghosted_agents = {}
        self._ghost_agents = {}
        self.rank = rank

    def delete_local(self, agent_id, ghosted_deleted: List) -> Agent:
        """Deletes the specified agent from the collection of local agents, and
        adds data any ghosts to be deleted.

        Args:
            agent_id (agent_id tuple): the id of the agent to remove
            ghosted_deleted: appended with a GhostedAgent if the agent to be removed is
            ghosted on another rank.

        Returns:
            The deleted agent or None if the agent does not exist
        """
        gh = self._ghosted_agents.pop(agent_id, None)
        if gh:
            ghosted_deleted.append(gh)

        return self._local_agents.pop(agent_id, None)

    def remove_local(self, agent_id) -> Agent:
        """Removes the specified agent from the collection of local agents.

        Args:
            agent_id (agent_id tuple): the id of the agent to remove
            ghosted: appended with a GhostedAgent if the agent to be removed is
            ghosted on another rank.

        Returns:
            The removed agent or None if the agent does not exist
        """
        return self._local_agents.pop(agent_id, None)

    def add_local(self, agent: Agent):
        """Adds the specified agent as a local agent

        Args:
            agent: the agent to add
        """
        agent.local_rank = self.rank
        self._local_agents[agent.uid] = agent
        if agent.uid in self._ghost_agents:
            del self._ghost_agents[agent.uid]

    def get_local(self, agent_id) -> Agent:
        """Gets the specified agent from the local collection.

        Args:
            agent_id (agent_id tuple): the id of the agent to get.
        Returns:
            The agent with the specified id or None if no such agent is
            in the local collection.
        """
        return self._local_agents.get(agent_id)

    def tag_as_ghosted(self, ghost_rank: int, agent_id) -> Agent:
        """Gets the specified agent from the local collection and marks it
        as ghosted on the specified rank.

        Args:
            ghost_rank: the rank the agent is being sent to as a ghost
            agent_id: the id of the agent to get
        Returns:
            The specified agent.
        """
        if agent_id in self._ghosted_agents:
            gh = self._ghosted_agents[agent_id]
            gh.ghost_ranks.add(ghost_rank)
            agent = gh.agent
        else:
            agent = self._local_agents[agent_id]
            self._ghosted_agents[agent_id] = GhostedAgent(agent, set([ghost_rank]))
        return agent

    def set_as_ghosted(self, ghost_ranks: List[int], agent_id):
        """Sets the specified agent as ghosted from this rank to
        the specified ranks.

        Args:
            ghost_ranks: the ranks where the agent is a ghost on
            agent_id: the id of the agent that is ghosted from this rank
        """
        gh_ranks = set(ghost_ranks)
        # if previously ghosted to this rank, but now on this rank, then
        # remove this rank from the ghosted ranks
        gh_ranks.discard(self.rank)
        if len(gh_ranks) > 0:
            self._ghosted_agents[agent_id] = GhostedAgent(self._local_agents[agent_id], gh_ranks)

    def remove_ghost(self, agent: Agent):
        ghost_agent = self._ghost_agents[agent.uid]
        ghost_agent.ref_count -= 1
        if ghost_agent.ref_count == 0:
            del self._ghost_agents[agent.uid]

    def delete_ghost(self, agent_id: Tuple):
        """Deletes the specified ghost agent. This is used when
        to clear the ghost when the non ghost has been removed from 
        the simulation
        """
        del self._ghost_agents[agent_id]

    def add_ghost(self, ghosted_rank: int, agent: Agent, incr: int=1):
        """Adds the specified agent to the ghost collection from
        the specified rank.

        Args:
            ghosted_rank: the rank the agent was received from
            agent: the agent to add
            incr: the amount to increment the reference count
        """
        uid = agent.uid
        agent.local_rank = ghosted_rank
        ghost_agent = GhostAgent(agent, incr)
        self._ghost_agents[uid] = ghost_agent

    def add_ghosts_to_projection(self, projection):
        """Adds all the ghost agents to the specified projection.

        Args:
            projection: The projection to add the ghosts to
        """
        for gh in self._ghost_agents.values():
            gh.ref_count += 1
            projection.add(gh.agent)

    def get_ghost(self, agent_uid: Tuple, incr: int=1) -> Agent:
        """Gets the agent with the specified id from the collection
        of ghost agents, or None if the agent does not exist in the
        ghost agent collection. If the agent exists, its projection
        reference count is incremented by the specified amount

        Args:
            agent_uid (agent uid tuple): the uid of the agent to get
            incr: the amount to increment the reference count
        Returns:
            The specified agent or None if the agent is not in the ghost
            agent collection. If the agent exists, its projection
            reference count is incremented.
        """
        val = self._ghost_agents.get(agent_uid, None)
        if val is not None:
            val.ref_count += incr
            return val.agent
        return None


@runtime_checkable
class SharedProjection(Protocol):

    def synchronize_ghosts(self, agent_manager: AgentManager, create_agent: Callable):
        """Synchronizes the ghosted part of this projection

        Args:
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        pass

    def clear_ghosts(self, agent_manager: AgentManager):
        """Clears the ghosted part of this projection"""
        pass


@runtime_checkable
class BoundedProjection(Protocol):

    def _get_oob(self):
        """Gets an Iterator over the out of bounds data for this BoundedProjection.
        out-of-bounds data is tuple ((aid.id, aid.type, aid.rank), ngh.rank, pt)
        """
        pass

    def _clear_oob(self):
        """Clears the out-of-bounds data
        """
        pass

    def _synch_move(self, agent, location: np.array):
        """Moves an agent to location in its new projection when
        that agent moves out of bounds from a previous projection.

        Args:
            agent: the agent to move
            location: the location for the agent to move to
        """
        pass


class SharedContext:
    """Encapsulates a population of agents on a single process rank.

    A SharedContext may have one or more projections associated with it to
    impose a relational structure on the agents in the context. It also
    provides functionality for synchroizes agents across processes, moving
    agents from one processe to another and managing any ghosting strategy.
    """
    def __init__(self, comm):
        """Initializes this SharedContext with the specified communicator.

        Args:
            comm (mpi4py communicator): the communicator uses to communicate
            among SharedContexts in the distributed model
        """
        self._agent_manager = AgentManager(comm.Get_rank(), comm.Get_size())
        self._agents_by_type = {}
        self.projections = {}
        self.removed_ghosteds = []
        self.projection_id = 0
        self.rank = comm.Get_rank()
        self.comm = comm
        self.ghosted_projs = []
        self.bounded_projs = {}

    def add(self, agent: Agent):
        """Adds the specified agent to this SharedContext.

        The agent will also be added to any projections currently in this SharedContext

        Args:
            agent: the agent to add
        """
        self._agent_manager.add_local(agent)
        self._agents_by_type.setdefault(agent.uid[1], collections.OrderedDict())[agent.uid] = agent
        for proj in self.projections.values():
            proj.add(agent)

    def _add_ghost(self, rank: int, agent: Agent):
        self._agent_manager.add_ghost(rank, agent, len(self.projections))
        for proj in self.projections.values():
            proj.add(agent)

    def add_projection(self, projection):
        """Adds the specified projection to this SharedContext.

        If the projection has `clear_buffer` and `synchronize_buffer` attributes,
        then clear_buffer will be called prior to synchornization, and synchronize buffer
        durring synchronization in the SharedProjection.synchronize method.

        Any agents currently in this context will be added to the projection.

        Args:
            projection: the projection add
        """
        for prj in self.projections.values():
            if projection.name == prj.name:
                raise ValueError('Context already contains the named projection "{}"'.format(prj.name))

        self.projections[self.projection_id] = projection

        for a in self._agent_manager._local_agents.values():
            projection.add(a)

        self._agent_manager.add_ghosts_to_projection(projection)

        if isinstance(projection, SharedProjection):
            self.ghosted_projs.append(projection)

        if isinstance(projection, BoundedProjection):
            self.bounded_projs[self.projection_id] = projection

        self.projection_id += 1

    def remove(self, agent: Agent):
        """Removes the specified agent from this SharedContext

        This agent is also removed from any projections associated with this SharedContext.
        If the agent is shared as a ghost on any other ranks it will be removed from those
        ranks during the next synchronization.
        """
        self._agent_manager.delete_local(agent.uid, self.removed_ghosteds)
        del self._agents_by_type[agent.uid[1]][agent.uid]
        for proj in self.projections.values():
            proj.remove(agent)

    def _fill_send_data(self, ghosted_moved: List):
        send_data = [[] for i in range(self.comm.size)]
        removed_agents = {}
        # gather agents to send from the out of bounds (oob) sequence
        for pid, proj in self.bounded_projs.items():
            for agent_id, ngh_rank, pt in proj._get_oob():
                agent = self._agent_manager.remove_local(agent_id)
                if agent is None:
                    # already removed
                    data = ((agent_id),)
                    proj.remove(removed_agents[agent_id])
                else:
                    removed_agents[agent_id] = agent
                    del self._agents_by_type[agent_id[1]][agent_id]
                    data = agent.save()
                    proj.remove(agent)
                    if agent_id in self._agent_manager._ghosted_agents:
                        ghosted_moved.append((ngh_rank, agent_id,
                                             self._agent_manager._ghosted_agents.pop(agent_id).ghost_ranks))

                # send to ngh rank a list of lists - inner list
                # is [(agent_id, agent_state), (projection id, and projection data)]
                send_data[ngh_rank].append([data, (pid, pt)])

            proj._clear_oob()

        return send_data

    def _process_recv_data(self, recv_data, create_agent):
        for data_list in recv_data:
            for data in data_list:
                # agent_data: tuple - agent id tuple, and agent state
                agent_data = data[0]
                # (projection_id, new location as np.array)
                proj_data = data[1]

                agent = self._agent_manager.get_local(agent_data[0])
                if not agent:
                    # print("New agent", agent_data)
                    agent = create_agent(agent_data)
                    # print("Adding", agent.id)
                    self.add(agent)

                    # add to the projection
                self.bounded_projs[proj_data[0]]._synch_move(agent, proj_data[1])

    def _update_removed_ghosts(self):
        # send ghosted removed updates
        send_data = [[] for i in range(self.comm.size)]
        for gh in self.removed_ghosteds:
            for rank in gh.ghost_ranks:
                send_data[rank].append(gh.agent.uid)
        recv_data = self.comm.alltoall(send_data)
        for updates in recv_data:
            for removed_id in updates:
                ghost = self._agent_manager.get_ghost(removed_id)
                for proj in self.projections.values():
                    proj.remove(ghost)
                self._agent_manager.delete_ghost(removed_id)

        self.removed_ghosteds.clear()

    def _update_ghosts(self, ghosted_moved: List):
        """
        Args:
            ghosted_moved: list of tuples describing agents that were ghosted from this rank
            but have moved off of it. Format is (destination_rank, agent_id, [ghost ranks])
        """
        self._update_removed_ghosts()

        # send ghosted moved updates
        send_data = [[] for i in range(self.comm.size)]
        for gh in ghosted_moved:
            send_data[gh[0]].append(gh[1:])
        recv_data = self.comm.alltoall(send_data)
        for updates in recv_data:
            for agent_id, gh_ranks in updates:
                self._agent_manager.set_as_ghosted(gh_ranks, agent_id)

        # send agent state updates to ghosts
        send_data = [[] for i in range(self.comm.size)]
        for _, gh_agent in self._agent_manager._ghosted_agents.items():
            for g_rank in gh_agent.ghost_ranks:
                send_data[g_rank].append(gh_agent.agent.save())
        recv_data = self.comm.alltoall(send_data)
        for updates in recv_data:
            for update in updates:
                ghost = self._agent_manager._ghost_agents[update[0]]
                ghost.agent.load(update[1])

    def synchronize(self, create_agent, sync_ghosts: bool=True):
        """Synchronizes the model state across processes by moving agents, filling
        projection buffers and so forth.

        Args:
            create_agent: a callable that takes agent data and creates and returns an agent instance from
            data. The data is a tuple consisting of the agent id tuple, and the serialized agent state
            sync_ghosts: if True, the ghosts in any SharedProjections and value layers associated
            with this SharedContext are also synchronized. Defaults to True.
        """

        if sync_ghosts:
            for proj in self.ghosted_projs:
                proj.clear_ghosts(self._agent_manager)

        ghosted_moved = []
        # These 3 synchronize bounded projections -- agents that
        # have moved out of a local bounded projection to that
        # controlled by another
        send_data = self._fill_send_data(ghosted_moved)
        recv_data = self.comm.alltoall(send_data)
        self._process_recv_data(recv_data, create_agent)

        if sync_ghosts:
            self._update_ghosts(ghosted_moved)
            for proj in self.ghosted_projs:
                proj.synchronize_ghosts(self._agent_manager, create_agent)

    def agents(self, agent_type: int=None, shuffle: bool=False):
        """Gets the agents in SharedContext, optionally of the specified type or shuffled.

        Args:
            agent_type: the type id of the agent, defaults to None.
            shuffle: whether or not the iteration order is shuffled. if true,
            the order is shuffled. If false, the iteration order is the order of
            insertion.

        Returns:
            iterable: An iterable over all the agents in the context.

            If the agent_type is not None then an iterable over agents of that type will be
            returned.
        """
        if shuffle:
            if agent_type is None:
                lst = list(self._agent_manager._local_agents.values())
                rng.shuffle(lst)
                return lst
            else:
                lst = list(self._agents_by_type[agent_type].values())
                rng.shuffle(lst)
                return lst
        else:
            if agent_type is None:
                return self._agent_manager._local_agents.values().__iter__()
            else:
                return self._agents_by_type[agent_type].values().__iter__()

    def agent(self, agent_id) -> Agent:
        """Gets the specified agent from the collection of local
        agents in this context.

        Args:
            agent_id: the uid tuple of the agent to return

        Returns:
            The agent with the specified id or None if no such agent is found.
        """
        return self._agent_manager.get_local(agent_id)

    def size(self, agent_type_ids: List[int]=None) -> dict:
        """Gets the number of agents in this SharedContext, optionally by type.

        Args:
            agent_type_ids: a list of the agent type ids identifying the agent types to count.
            If this is None then the total size is returned with an id of 0.

        Returns:
            A dictionary containing the counts (the dict values) by type (the dict keys).
        """
        counts = {}
        if agent_type_ids:
            for i in agent_type_ids:
                counts[i] = len(self._agents_by_type[i])
        else:
            counts[0] = len(self._agent_manager._local_agents)

        return counts

    def _send_requests(self, requested_agents: List):
        requests = [None for i in range(self.comm.size)]
        existing_ghosts = []
        for request in requested_agents:
            # 1 is the rank to request from
            # 0 is the id of the agent we are requesting
            agent = self._agent_manager.get_ghost(request[0], 0)
            if agent is None:
                lst = requests[request[1]]
                if lst is None:
                    requests[request[1]] = [request[0]]
                else:
                    requests[request[1]].append(request[0])
            else:
                existing_ghosts.append(agent)
        recv = self.comm.alltoall(requests)
        return (recv, existing_ghosts)

    def request_agents(self, requested_agents: List, create_agent: Callable) -> List[Agent]:
        """Requests agents from other ranks to be copied to this rank as ghosts.
        This is a collective operation and all ranks must call it, regardless
        of whether agents are being requested on that rank.

        Args:
            requested_agents: A list of tuples specifying requested agents and the rank
            to request from. Each tuple must contain the agents id tuple and the rank, for
            example ((id, type, rank), requested_rank).
            create_agent: a Callable that can take the result of an agent save() and
            return an agent.
        Returns:
            The list of requested agents.
        """
        sent_agents = [[] for i in range(self.comm.size)]
        requested, ghosts = self._send_requests(requested_agents)
        for rank, requests in enumerate(requested):
            if requests:
                for requested_id in requests:
                    agent = self._agent_manager.tag_as_ghosted(rank, requested_id)
                    sent_agents[rank].append(agent.save())

        recv_agents = self.comm.alltoall(sent_agents)
        # print(f'{self.comm.rank} {recv_agents}')
        for rank, agents in enumerate(recv_agents):
            for agent_data in agents:
                agent = create_agent(agent_data)
                ghosts.append(agent)
                self._add_ghost(rank, agent)

        return ghosts
