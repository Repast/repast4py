# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

"""This module implements core functionality used by both contexts
and projections.
"""
import collections
import numpy as np
from ._core import Agent
from typing import Callable, List, Tuple, Dict
from dataclasses import dataclass

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@dataclass
class GhostAgent:
    """A non-local agent copied from another rank.

    GhostAgent is used by the AgentManager to track and manage ghosts agents on
    a rank.

    **This is class is internal to the repast4py implementation and is NOT for users.**

    Attributes:
        agent (Agent): the ghost agent
        ref_count (int): a reference count tracking the number of projections
            on this rank that refer to the agent

    """

    agent: Agent
    # the number of projections that refer to this ghost
    ref_count: int


@dataclass
class GhostedAgent:
    """An agent local to this rank that has been copied or "ghosted" to
    another rank.

    GhostedAgent is used by the AgentManager to track and manage agents that
    have been ghosted from this rank to other ranks.

    **This is class is internal to the repast4py implementation and is NOT for users.**

    Attributes:
        agent (Agent): the ghosted agent
        ghost_ranks (Dict): maps the ghosted to rank to the number references the agent has on that rank.

    """

    agent: Agent
    # maps ghost rank to number of references
    # on that rank
    ghost_ranks: Dict


class AgentManager:
    """Manages local and non-local (ghost) agents as they move
    between processes.

    **This is class is internal to the repast4py implementation and is NOT for users.**

    Args:
        rank: the local process rank
        world_size: the total number of ranks in the model

    """

    def __init__(self, rank: int, world_size: int):
        self._local_agents = collections.OrderedDict()
        # key: agent uid, val: GhostedAgent
        self._ghosted_agents = {}
        # key: agent uid, val: GhostAgent
        self._ghost_agents = {}
        self.rank = rank
        # track requested agents by id
        self._req_ghosts = set()

    def is_requested(self, agent_id: Tuple) -> bool:
        """Gets whether or not the specified agent is requested
        as a ghost on this rank.

        Args:
            agent_id: the id of the agent to check

        Returns:
            True if the agent is requested as a ghost on this rank, otherwise
            False.
        """
        return agent_id in self._req_ghosts

    def add_req_ghost(self, agent_id: Tuple):
        """Adds the specified agent to the set of requested agents
        that are ghosts on this rank.

        Args:
            agent_id: the id of the ghost requested agent
        """
        self._req_ghosts.add(agent_id)

    def delete_local(self, agent_id: Tuple, ghosted_deleted: List) -> Agent:
        """Deletes the specified agent from the collection of local agents, and
        adds data any ghosts to be deleted.

        Args:
            agent_id: the id of the agent to remove
            ghosted_deleted: appended with a GhostedAgent if the agent to be removed is
                ghosted on another rank.

        Returns:
            The deleted agent or None if the agent does not exist
        """
        gh = self._ghosted_agents.pop(agent_id, None)
        if gh:
            ghosted_deleted.append(gh)

        return self._local_agents.pop(agent_id, None)

    def remove_local(self, agent_id: Tuple) -> Agent:
        """Removes the specified agent from the collection of local agents.

        Args:
            agent_id: the id of the agent to remove

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

    def get_local(self, agent_id: Tuple) -> Agent:
        """Gets the specified agent from the collection of local agents.

        Args:
            agent_id: the unique id of the agent to get.
        Returns:
            The agent with the specified id or None if no such agent is
            in the local collection.
        """
        return self._local_agents.get(agent_id)

    def is_ghosted_to(self, ghost_rank: int, agent_id: Tuple) -> bool:
        """Gets whether or not the specified agent is ghosted to
        the specified rank.

        Args:
            ghost_rank: the rank the agent is being sent to as a ghost
            agent_id: the id of the agent to get
        Returns:
            True if the agent is ghosted to the rank, otherwise False.
        """
        return agent_id in self._ghosted_agents and ghost_rank in self._ghosted_agents[agent_id].ghost_ranks

    def tag_as_ghosted(self, ghost_rank: int, agent_id: Tuple) -> Agent:
        """Gets the specified agent from the local collection and marks it
        as ghosted on the specified rank.

        Args:
            ghost_rank: the rank the agent is being sent to as a ghost
            agent_id: the id of the agent to tag
        Returns:
            The specified agent.
        """
        if agent_id in self._ghosted_agents:
            gh = self._ghosted_agents[agent_id]
            ranks = gh.ghost_ranks
            if ghost_rank in ranks:
                ranks[ghost_rank] += 1
            else:
                ranks[ghost_rank] = 1
            agent = gh.agent
        else:
            agent = self._local_agents[agent_id]
            self._ghosted_agents[agent_id] = GhostedAgent(agent, {ghost_rank: 1})
        return agent

    def set_as_ghosted(self, ghost_ranks: Dict, agent_id: Tuple):
        """Sets the specified agent as ghosted from this rank to
        the specified ranks.

        Args:
            ghost_ranks: the ranks where the agent is a ghost on
            agent_id: the id of the agent that is ghosted from this rank
        """
        if len(ghost_ranks) > 0:
            self._ghosted_agents[agent_id] = GhostedAgent(self._local_agents[agent_id], ghost_ranks)

    def delete_ghosted(self, agent_id: Tuple) -> Dict[int, int]:
        """Removes the specified agent from the collection of agents ghosted from this
        rank and returns the ranks it is ghosted to.

        This is used when the ghosted agent moves off of a rank on to
        another rank.

        Args:
            agent_id: the unique id tuple of the agent to remove from the ghosted collection
        Returns:
            A dictionary where the key is the rank the agent is ghosted to, and
            the value is the projection reference count for the agent on that rank.
        """
        # self._req_ghosted.remove(agent_id)
        return self._ghosted_agents.pop(agent_id).ghost_ranks

    def untag_as_ghosted(self, ghost_rank: int, agent_id: Tuple):
        """Decrements the ghosted reference count for the specified agent
        on the specifed rank.

        If the reference count goes to 0, then the agent will no longer be
        ghosted to that rank.

        Args:
            ghost_rank: the rank the agent is being sent to as a ghost
            agent_id: the id of the agent to tag
        """
        gh = self._ghosted_agents[agent_id]
        ranks = gh.ghost_ranks
        ranks[ghost_rank] -= 1
        if ranks[ghost_rank] == 0:
            del ranks[ghost_rank]

    def remove_ghost(self, agent: Agent):
        """Decrements the ghost agents reference count and removes it from this rank, if its
        reference count is 0.

        Args:
            agent: the agent to remove

        """
        # don't remove requested agents
        if agent.uid not in self._req_ghosts:
            ghost_agent = self._ghost_agents[agent.uid]
            ghost_agent.ref_count -= 1
            if ghost_agent.ref_count == 0:
                del self._ghost_agents[agent.uid]

    def delete_ghost(self, agent_id: Tuple):
        """Deletes the specified ghost agent. This is used
        to clear the ghost when the non ghost has been removed from
        the simulation

        Args:
            agent_id: the unique id tuple of the ghost.
        """
        del self._ghost_agents[agent_id]

    def add_ghost(self, ghosted_rank: int, agent: Agent, incr: int = 1):
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

    def get_ghost(self, agent_uid: Tuple, incr: int = 1) -> Agent:
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
    """Protocol class that defines the API for projections that
    are shared across multiple processes
    """

    def _pre_synch_ghosts(self, agent_manager: AgentManager):
        """Called prior to synchronizing ghosts and before any cross-rank movement
        synchronization.

        Args:
            agent_manager: this rank's AgentManager
        """
        pass

    def _synch_ghosts(self, agent_manager: AgentManager, create_agent: Callable):
        """Synchronizes the ghosted part of this projection

        Args:
            agent_manager: this rank's AgentManager
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        pass

    def _agent_moving_rank(self, moving_agent: Agent, dest_rank: int, moving_agent_data: List,
                           agent_manager: AgentManager):
        """Notifies this projection that the specified agent is moving from the current rank
        to the destination rank.

        This is called whenever an agent is moving between ranks, but before the agent has been
        received by the destination rank. This allows this projection to prepare the relevant
        synchronization.

        Args:
            moving_agent: the agent that is moving.
            dest_rank: the destination rank
            moving_agent_data: A list where the first element is the serialized agent data. The
            list may also contain a second element that is a dictionary where the keys are the
            ranks the agent is currently ghosted to and the value is the number of projections
            using that ghost. Projections can add to this dictionary if they ghost an agent
            to another rank as part of their projection synchronization (e.g. ghosting edge).
            agent_manager: this rank's AgentManager

        """
        pass

    def _agents_moved_rank(self, moved_agents: List, agent_manager: AgentManager):
        """Notifies this projection that the specified agents have moved rank.

        This is called after the agents have moved ranks and includes all the agents
        that have moved on all ranks EXCEPT for those moved off of the current rank.
        All agents should have been moved and added to their destination ranks contexts
        and AgentManagers when this is called.

        Args:
            moved_agents: a list of tuples (agent.uid, destination rank) where each tuple
            is a moved agent. The list includes all the agents that have moved on all ranks
            EXCEPT for those moved off of the current rank.
            agent_manager: this rank's AgentManager
        """
        pass

    def _post_agents_moved_rank(self, agent_manager: AgentManager, create_agent: Callable):
        """Notifies this projection that all the agent movement synchronization has occured,
        allowing this projection to perform any necessary actions.

        Args:
            agent_manager: this rank's AgentManager
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        pass


@runtime_checkable
class BoundedProjection(Protocol):
    """Protocol class for projections that are bounded such that an agent
    can move beyond the bounds of one instance and into another on another rank.

    """

    def _get_oob(self):
        """Gets an Iterator over the out of bounds data for this BoundedProjection.
        out-of-bounds data is tuple ((aid.id, aid.type, aid.rank), ngh.rank, pt)
        """
        pass

    def _clear_oob(self):
        """Clears this projection's out-of-bounds data
        """
        pass

    def _move_oob_agent(self, agent: Agent, location: np.array):
        """Moves an agent to the location in its new projection when
        that agent moves out of bounds from a previous projection.

        Args:
            agent: the agent to move
            location: the location for the agent to move to
        """
        pass
