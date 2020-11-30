import collections
import numpy as np
from ._core import Agent
from .random import default_rng as rng
from typing import Callable, List

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class SharedProjection(Protocol):

    def synchronize_ghosts(self, create_agent: Callable):
        """Synchronizes the ghosted part of this projection

        Args:
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        pass

    def clear_ghosts(self):
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
        that agent moves out of bounds from a previous projection
        
        Args:
            agent: the agent to move
            location: the location for the agent to move to
        """
        pass


class AgentManager:
    """Manages local and non-local (ghost) agents as they move
    between processes
    """

    def __init__(self):
        self._local_agents = collections.OrderedDict()
        self._ghost_agents = {}
        self._ghosted_agents = {}

    def remove_local(self, agent_id) -> Agent:
        """Removes the specified agent from the collection of local agents.

        Args:
            agent_id (agent_id tuple): the id of the agent to remove

        Returns:
            The removed agent or None if the agent does not exist
        """
        # TODO check if agent is ghosted and update accordingly
        return self._local_agents.pop(agent_id, None)

    def add_local(self, agent: Agent):
        """Adds the specified agent as a local agent

        Args:
            agent: the agent to add
        """
        self._local_agents[agent.uid] = agent

    def get_local(self, agent_id) -> Agent:
        """Gets the specified agent from the local collection.

        Args:
            agent_id (agent_id tuple): the id of the agent to get.
        Returns:
            The agent with the specified id or None if no such agent is
            in the local collection.
        """
        return self._local_agents.get(agent_id)


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
        self._agent_manager = AgentManager()
        self._agents_by_type = {}
        self.projections = {}
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

        if isinstance(projection, SharedProjection):
            self.ghosted_projs.append(projection)

        if isinstance(projection, BoundedProjection):
            self.bounded_projs[self.projection_id] = projection
        
        self.projection_id += 1

    def remove(self, agent: Agent):
        """Removes the specified agent from this SharedContext

        This agent is also removed from any projections associated with this SharedContext.
        """
        self._agent_manager.remove_local(agent.uid)
        del self._agents_by_type[agent.uid[1]][agent.uid]
        for proj in self.projections.values():
            proj.remove(agent)

    def _fill_send_data(self):
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

                # send to ngh rank a list of lists - inner list 
                # is [(agent_id, agent_state), (projection id, and projection data)]
                send_data[ngh_rank].append([data, (pid, pt)])

            proj._clear_oob()

        removed_agents.clear()

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

    def synchronize(self, create_agent, sync_buffer: bool=True):
        """Synchronizes the model state across processes by moving agents, filling 
        projection buffers and so forth.

        Args:
            create_agent: a callable that takes agent data and creates and returns an agent instance from
            data. The data is a tuple consisting of the agent id tuple, and the serialized agent state
            synch_buffer: if True, the buffered areas in any buffered projections and value layers associated
            with this SharedContext are also synchronized. Defaults to True.
        """

        if sync_buffer:
            for proj in self.ghosted_projs:
                proj.clear_ghosts()

        send_data = self._fill_send_data()
        # print('{}: send data - {}'.format(self.rank, send_data))
        recv_data = self.comm.alltoall(send_data)
        # print('{}: recv data - {}'.format(self.rank, recv_data))
        self._process_recv_data(recv_data, create_agent)

        if sync_buffer:
            for proj in self.ghosted_projs:
                proj.synchronize_ghosts(create_agent)

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
                l = list(self._agent_manager._local_agents.values())
                rng.shuffle(l)
                return l
            else:
                l = list(self._agents_by_type[agent_type].values())
                rng.shuffle(l)
                return l
        else:
            if agent_type is None:
                return self._agent_manager._local_agents.values().__iter__()
            else:
                return self._agents_by_type[agent_type].values().__iter__()

    def size(self, agent_type_ids:List[int]=None) -> dict:
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
