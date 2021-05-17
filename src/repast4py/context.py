import itertools
import collections

from repast4py.value_layer import SharedValueLayer
from ._core import Agent
from .random import default_rng as rng
from typing import Callable, List

from .core import AgentManager, SharedProjection, BoundedProjection
from .util import is_empty


class SharedContext:
    """Encapsulates a population of agents on a single process rank.

    A SharedContext may have one or more projections associated with it to
    impose a relational structure on the agents in the context. It also
    provides functionality for synchronizing agents across processes, moving
    agents from one processe to another and managing any ghosting strategy.
    """
    def __init__(self, comm):
        """Initializes this SharedContext with the specified communicator.

        Args:
            comm (mpi4py communicator): the communicator used to communicate
                among SharedContexts in the distributed model
        """
        self._agent_manager = AgentManager(comm.Get_rank(), comm.Get_size())
        self._agents_by_type = {}
        self.projections = {}
        self.removed_ghosteds = []
        self.projection_id = 0
        self.rank = comm.Get_rank()
        self.comm = comm
        self.bounded_projs = {}
        self.non_bounded_projs = []
        self.value_layers = []

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
        self._agent_manager.add_ghost(rank, agent, incr=len(self.projections))
        for proj in self.projections.values():
            proj.add(agent)

    def add_value_layer(self, value_layer: SharedValueLayer):
        """Adds the specified value_layer to the this context.

        Args:
            value_layer: the value layer to add.
        """
        self.value_layers.append(value_layer)

    def add_projection(self, projection: SharedProjection):
        """Adds the specified projection to this SharedContext.

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

        if isinstance(projection, BoundedProjection):
            self.bounded_projs[self.projection_id] = projection
        else:
            self.non_bounded_projs.append(projection)

        self.projection_id += 1

    def remove(self, agent: Agent):
        """Removes the specified agent from this SharedContext

        This agent is also removed from any projections associated with this SharedContext.
        If the agent is shared as a ghost on any other ranks it will be removed from those
        ranks during the next synchronization.

        Args:
            agent: the agent to remove.
        """
        self._agent_manager.delete_local(agent.uid, self.removed_ghosteds)
        del self._agents_by_type[agent.uid[1]][agent.uid]
        for proj in self.projections.values():
            proj.remove(agent)

    def _gather_oob_data(self, oob_agents: List):
        """Gets the agent data from any agents that have moved out-of-bounds
        (oob), and adds to the specified list the agent data and the rank where the agent 
        will move.

        Args:
            oob_agents: an empty list into which a tuple (agent_data, ngh_rank) for
                each oob agent will be added.
        Return:
            A list of lists where the position in the first list is the rank
            to send the data in the second list to.
        """
        send_data = [[] for i in range(self.comm.size)]
        # gather agents to send from the out of bounds (oob) sequence
        for pid, proj in self.bounded_projs.items():
            for agent_id, ngh_rank, pt in proj._get_oob():
                agent = self._agent_manager.remove_local(agent_id)
                if agent is None:
                    # already removed
                    data = [(agent_id,)]
                else:
                    oob_agents.append((agent, ngh_rank))
                    del self._agents_by_type[agent_id[1]][agent_id]
                    data = [agent.save()]
                    if agent_id in self._agent_manager._ghosted_agents:
                        data.append(self._agent_manager.delete_ghosted(agent_id))

                    proj._agent_moving_rank(agent, ngh_rank, data, self._agent_manager)
                    for nb_proj in self.non_bounded_projs:
                        nb_proj._agent_moving_rank(agent, ngh_rank, data, self._agent_manager)

                # send to ngh rank a list of lists - inner list
                # is [(agent_id, agent_state), (projection id, and projection data)]
                send_data[ngh_rank].append([data, (pid, pt)])

            proj._clear_oob()

        return send_data

    def _process_recv_oob_data(self, recv_data, ghosts_to_remove: List, create_agent: Callable):
        for data_list in recv_data:
            for data in data_list:
                # agent_data_list: [(agent id tuple, agent state), optional ghosting data]
                agent_data_list = data[0]
                # (projection_id, new location as np.array)
                proj_data = data[1]

                uid = agent_data_list[0][0]
                agent = self._agent_manager.get_local(uid)
                if agent is None:
                    # print("New agent", agent_data)
                    agent = create_agent(agent_data_list[0])
                    # received what used to be ghost here
                    if agent.uid in self._agent_manager._ghost_agents:
                        ghosts_to_remove.append(agent.uid)
                    # print("Adding", agent.id)
                    self.add(agent)
                    if len(agent_data_list) == 2:
                        ghosted_data = agent_data_list[1]
                        # remove any ghosted data for this rank
                        # so this rank doesn't try to send to self
                        ghosted_data.pop(self.rank, None)
                        self._agent_manager.set_as_ghosted(ghosted_data, agent.uid)

                    # add to the projection
                self.bounded_projs[proj_data[0]]._move_oob_agent(agent, proj_data[1])

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

    def _update_ghosts(self):
        self._update_removed_ghosts()
        # send agent state updates to ghosts
        send_data = [[] for _ in range(self.comm.size)]
        for _, gh_agent in self._agent_manager._ghosted_agents.items():
            for g_rank in gh_agent.ghost_ranks:
                send_data[g_rank].append(gh_agent.agent.save())
        recv_data = self.comm.alltoall(send_data)
        for updates in recv_data:
            for update in updates:
                ghost = self._agent_manager._ghost_agents[update[0]]
                ghost.agent.load(update[1])

    def _pre_synch_ghosts(self):
        """Calls _pre_synch_ghosts on all "ghostable" projections
        and value layers.
        """
        for proj in self.projections.values():
            proj._pre_synch_ghosts(self._agent_manager)

    def _synch_ghosts(self, create_agent: Callable):
        """Calls _synch_ghosts on all "ghostable" projections
        and value layers.

        Args:
            create_agent: a Callable that when given serialized agent data
            return an agent.
        """
        self._update_ghosts()
        for proj in self.projections.values():
            proj._synch_ghosts(self._agent_manager, create_agent)

        for vl in self.value_layers:
            vl._synch_ghosts()

    def synchronize(self, create_agent: Callable, sync_ghosts: bool=True):
        """Synchronizes the model state across processes by moving agents, filling
        projection buffers with ghosts, and so forth.

        Args:
            create_agent: a callable that takes agent data and creates and returns an agent instance from
                data. The data is a tuple consisting of the agent id tuple, and the serialized agent state
            sync_ghosts: if True, the ghosts in any SharedProjections and value layers associated
                with this SharedContext are also synchronized. Defaults to True.
        """

        if sync_ghosts:
            self._pre_synch_ghosts()

        oob_moved_agents = []
        # These 3 synchronize bounded projections -- agents that
        # have moved out of a local bounded projection to that
        # controlled by another
        send_data = self._gather_oob_data(oob_moved_agents)
        recv_data = self.comm.alltoall(send_data)
        ghosts_to_remove = []
        self._process_recv_oob_data(recv_data, ghosts_to_remove, create_agent)
        any_moved = self._agents_moved(oob_moved_agents, self.projections.values())

        if any_moved:
            for gh_uid in ghosts_to_remove:
                self._agent_manager.delete_ghost(gh_uid)

        if sync_ghosts:
            self._synch_ghosts(create_agent)

        if any_moved:
            for proj in self.projections.values():
                proj._post_agents_moved_rank(self._agent_manager, create_agent)

            # synch ghosts for non-oob projections -- they don't need a double synch
            # to maintain coherences
            for proj in self.non_bounded_projs:
                proj._pre_synch_ghosts(self._agent_manager)

                self._update_ghosts()
                for pid, proj in self.projections.items():
                    if pid not in self.bounded_projs:
                        proj._synch_ghosts(self._agent_manager, create_agent)

    def agents(self, agent_type: int=None, shuffle: bool=False):
        """Gets the agents in this SharedContext, optionally of the specified type or shuffled.

        Args:
            agent_type (int): the type id of the agent, defaults to None.
            shuffle (bool): whether or not the iteration order is shuffled. If true,
                the order is shuffled. If false, the iteration order is the order of
                insertion.

        Returns:
            iterable: An iterable over all the agents in the context. If the agent_type is not None then
            an iterable over agents of that type will be returned.

        Examples:
            >>> PERSON_AGENT_TYPE = 1
            >>> for agent in ctx.agents(PERSON_AGENT_TYPE, shuffle=True)
                   ...
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

        Examples:
            >>> ctx = SharedContext(comm)
            >>> # .. Agents Added
            >>> agent1 = ctx.agent((1, 0, 1))
        """
        return self._agent_manager.get_local(agent_id)

    def ghost_agent(self, agent_id) -> Agent:
        """Gets the specified agent from the collection of ghost
        agents in this context.

        Args:
            agent_id: the uid tuple of the agent to return

        Returns:
            The ghost agent with the specified id or None if no such agent is found.
        """

        return self._agent_manager.get_ghost(agent_id, incr=0)

    def size(self, agent_type_ids: List[int]=None) -> dict:
        """Gets the number of agents in this SharedContext, optionally by type.

        Args:
            agent_type_ids: a list of the agent type ids identifying the agent types to count.
                If this is None then the total size is returned with an id of -1.

        Returns:
            A dictionary containing the counts (the dict values) by type (the dict keys).
        """
        counts = {}
        if agent_type_ids:
            for i in agent_type_ids:
                counts[i] = len(self._agents_by_type[i])
        else:
            counts[-1] = len(self._agent_manager._local_agents)

        return counts

    def _send_requests(self, requested_agents: List):
        """Sends a request to specified ranks for specified agents.

        Args:
            requested_agents: a list of tuples where each tuple is
                (id of requested agent, rank to request from)
        """
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
                self._agent_manager.add_req_ghost(agent.uid)
                for proj in self.projections.values():
                    if agent not in proj:
                        proj.add(agent)

        recv = self.comm.alltoall(requests)
        return (recv, existing_ghosts)

    def request_agents(self, requested_agents: List, create_agent: Callable) -> List[Agent]:
        """Requests agents from other ranks to be copied to this rank as ghosts.

        This is a collective operation and all ranks must call it, regardless
        of whether agents are being requested by that rank. The requested agents
        will be automatically added as ghosts to this rank.

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
                    # self._agent_manager.add_req_ghosted(requested_id)
                    sent_agents[rank].append(agent.save())

        recv_agents = self.comm.alltoall(sent_agents)
        for rank, agents in enumerate(recv_agents):
            for agent_data in agents:
                agent = create_agent(agent_data)
                ghosts.append(agent)
                self._add_ghost(rank, agent)
                self._agent_manager.add_req_ghost(agent.uid)

        return ghosts

    def _agents_moved(self, moved_agents: List, projs) -> bool:
        moved_ids = [(x.uid, dest) for x, dest in moved_agents]
        all_moved_agents = self.comm.allgather(moved_ids)
        any_moved = not is_empty(all_moved_agents)
        # replace agents moved by this rank with empty list
        # as we can ignore these given that they should be processed
        # in a previous call to proj._agent_moving_rank
        all_moved_agents[self.rank] = []
        if any_moved:
            for proj in projs:
                proj._agents_moved_rank(itertools.chain(*all_moved_agents), self._agent_manager)
        return any_moved

    def move_agents(self, agents_to_move: List, create_agent: Callable):
        """Moves agents from this rank to another rank where it becomes a
        local agent.

        The list of agents to move must be agents currently local to
        this rank. This performs a synchronize after moving the agents
        in order synchronize the new location of the agents.
        This is a collective operation and all ranks must call it, regardless
        of whether agents are being moved to or from that rank.

        Args:
            agents_to_move: A list of tuples specifying agents to move and the rank
                to request from. Each tuple must contain the agents id tuple and the rank, for
                example ((id, type, rank), rank_to_move_to).
            create_agent: a Callable that can take the result of an agent save() and
                return an agent.
        """
        sent_agents = [[] for i in range(self.comm.size)]
        moved_agents = []
        for uid, rank in agents_to_move:
            agent = self._agent_manager.remove_local(uid)
            del self._agents_by_type[uid[1]][uid]
            moved_agents.append((agent, rank))
            data = [agent.save()]
            sent_agents[rank].append(data)

            if uid in self._agent_manager._ghosted_agents:
                data.append(self._agent_manager.delete_ghosted(uid))

            for proj in self.projections.values():
                proj._agent_moving_rank(agent, rank, data, self._agent_manager)

        recv_data = self.comm.alltoall(sent_agents)
        ghosts_to_remove = []
        for agents in recv_data:
            for agent_data in agents:
                agent = create_agent(agent_data[0])
                if agent.uid in self._agent_manager._ghost_agents:
                    ghosts_to_remove.append(agent.uid)
                self.add(agent)
                # update with data where to ghost to
                if len(agent_data) == 2:
                    ghosted_data = agent_data[1]
                    # remove any ghosted data for this rank
                    # so this rank doesn't try to send to self
                    ghosted_data.pop(self.rank, None)
                    self._agent_manager.set_as_ghosted(ghosted_data, agent.uid)

        self._agents_moved(moved_agents, self.projections.values())

        # received agents that used to be ghosts so
        # remove them as ghosts as they are now local
        for gh_uid in ghosts_to_remove:
            self._agent_manager.delete_ghost(gh_uid)

        self._pre_synch_ghosts()
        send_data = self._gather_oob_data([])
        recv_data = self.comm.alltoall(send_data)
        ghosts_to_remove = []
        self._process_recv_oob_data(recv_data, ghosts_to_remove, create_agent)
        self._synch_ghosts(create_agent)

        for proj in self.projections.values():
            proj._post_agents_moved_rank(self._agent_manager, create_agent)
