# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

"""
The network module contains classes and functions for simulating networks (i.e., graphs) where
agents are the nodes in the network. The network code is built-on the `networkx <https://networkx.org>`_
python package.
"""

from mpi4py import MPI
from dataclasses import dataclass
import numpy as np
import networkx as nx
from itertools import chain
import re
import math
import json

from typing import List, Iterable, Callable, Tuple, Dict

from ._core import Agent
from .core import AgentManager
from . import random


@dataclass
class GhostedEdge:
    local_agent: Agent
    ghost_agent: Agent
    edge_attr: Dict


class SharedNetwork:
    """A network that can be shared across multiple process ranks through
    ghost nodes and edges. This is the base class for the Directed and Undirected
    SharedNetwork classes. This class should **NOT** be instantiated directly by users.

    This wraps a `networkx <https://networkx.org>`_ Graph object and delegates all the network
    related operations to it. That Graph is exposed as the `graph`
    attribute. See the `networkx <https://networkx.org>`_ Graph documentation for more information
    on the network functionality that it provides. **The network structure
    should _NOT_ be changed using the networkx functions and methods
    (adding and removing nodes, for example)**. Use this class'
    methods for manipulating the network structure.

    Attributes:
        graph (networkx Graph): a `networkx <https://networkx.org>`_ graph object that
            provides the network functionality.

    Args:
        name: the name of the SharedNetwork
        comm: the communicator over which this SharedNetwork is distributed
        graph (networkx Graph): the `networkx <https://networkx.org>`_ graph object that provides the network
            functionality.
    """

    def __init__(self, name: str, comm: MPI.Comm, graph: nx.Graph):
        self.name = name
        self.comm = comm
        self.graph = graph
        self.rank = comm.Get_rank()

        # list of edge keys
        self.edges_to_update = set()
        # key is edge tuple, val is GhostedEdge
        self.ghosted_edges = {}
        # key is edge tuple, val is GhostedEdge
        self.new_edges = {}
        # key is edge tuple, val is idx of ghost agent in tuple
        self.edges_to_remove = {}
        # list of edges that need to be recreated when a
        # agent moves off of this rank. Edge is (u.uid, v.uid)
        self._agent_moved_edges = []
        # list of agent ids that should not be ghosted to this
        # rank after an agent move. The identification of
        # these agents and communicating to the ghosting rank
        # occur in different stages so we track this info as field
        self._no_longer_ghosts = [[] for _ in range(self.comm.size)]
        # list of ghosts agents that need to have their
        # ref count incremented. This can occur when an agent
        # in a buffer zone is added via an edge, but has not
        # yet has the ghost ref count incremented
        self.ghosts_to_ref = []

    def __iter__(self) -> Iterable[Agent]:
        """Gets an Iterable over all the nodes (agents) in
        the network.

        Returns:
            An Iterable over all the nodes (agents) in the network.
        """
        return self.graph.__iter__()

    def __contains__(self, node):
        return self.graph.has_node(node)

    @property
    def node_count(self) -> int:
        """Gets the number of nodes in this SharedNetwork."""

        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Gets the number of edges in this SharedNework."""

        return self.graph.number_of_edges()

    def add(self, agent: Agent):
        """Adds the specified agent as a node in the network.
        """

        self.graph.add_node(agent)

    def add_nodes(self, agents: List[Agent]):
        """Adds each agent in the specified list of agents as
        nodes in the network.

        Args:
            agents: the list of agents to add
        """
        self.graph.add_nodes_from(agents)

    def remove(self, agent: Agent):
        """Removes the specified agent from this SharedNetwork.

        Args:
            agent: the agent to remove

        Raises:
            NetworkXError: if the agent is not a node in this SharedNetwork.
        """
        self.graph.remove_node(agent)

    def update_edge(self, u_agent: Agent, v_agent: Agent, **kwattr):
        """Updates the edge between u_agent and v_agent with the specified
        attributes.

        Args:
            u_agent: the u node of the edge
            v_agent: the v node of the edge
            kwattr: keyword arguments containting the edge attribute updates

        Examples:
            Update the weight attribute for the edge between agent1 and agent2.

            >>> g.update_edge(agent1, agent2, weight=10.1)
        """
        edge_key = self._get_edge_key((u_agent, v_agent))
        for k, v in kwattr.items():
            self.graph.edges[edge_key][k] = v

        if (u_agent.local_rank == self.rank and v_agent.local_rank != self.rank) or \
           (u_agent.local_rank != self.rank and v_agent.local_rank == self.rank):
            self.edges_to_update.add(edge_key)

    def remove_edge(self, u_agent: Agent, v_agent: Agent):
        """Removes the edge between u_agent and v_agent.

        Args:
            u_agent: the u node of the edge.
            v_agent: the v node of the edge.
        Raises:
            NetworkXError: if there is no edge between u_agent and v_agent.
        """
        self.graph.remove_edge(u_agent, v_agent)

        if u_agent.local_rank != self.rank and v_agent.local_rank == self.rank or \
           u_agent.local_rank == self.rank and v_agent.local_rank != self.rank:
            edge_key = self._get_edge_key((u_agent, v_agent))
            # edge key order might be different than remove edge args
            # so need to check in edge key which is the ghost
            self.edges_to_remove[edge_key] = 0 if edge_key[1].local_rank == self.rank else 1
            del self.ghosted_edges[edge_key]
            # if removed before update sync or removed before new edge sync
            # then remove it here
            self.edges_to_update.discard(edge_key)
            self.new_edges.pop(edge_key, None)

        self._remove_edge_key(u_agent, v_agent)

    def clear_edges(self):
        """Removes all the edges.
        """
        self.graph.clear_edges()
        for edge, ghosted_edge in self.ghosted_edges.items():
            edge_key = self._get_edge_key(edge)
            self.edges_to_remove[edge_key] = 1 if edge_key[0].local_rank == self.rank else 0
            self.edges_to_update.discard(edge_key)

        self.new_edges.clear()
        self.ghosted_edges.clear()

    def contains_edge(self, u_agent: Agent, v_agent: Agent) -> bool:
        """Gets whether or not an edge exists between the u_agent and v_agent.

        Args:
            u_agent: the u node of the edge.
            v_agent: the v node of the edge.

        Returns:
            True if an edge exists, otherwise False.
        """
        return self.graph.has_edge(u_agent, v_agent)

    def _sync_removed(self, agent_manager: AgentManager):
        """Synchronizes the removed edges across processes

        Args:
            agent_manager: AgentManager used to manage agent synchronization
        """
        sync_edges = [[] for i in range(self.comm.size)]
        for edge, ghost_idx in self.edges_to_remove.items():
            ghost = edge[ghost_idx]
            sync_edges[ghost.local_rank].append((edge[0].uid, edge[1].uid))
            agent_manager.untag_as_ghosted(ghost.local_rank, edge[not (ghost_idx)].uid)

        self.edges_to_remove.clear()
        to_remove = self.comm.alltoall(sync_edges)
        for edges in to_remove:
            for u_id, v_id in edges:
                u_agent = agent_manager._ghost_agents.get(u_id)
                if u_agent is None:
                    # u_agent is local
                    u_agent = agent_manager._local_agents[u_id]
                    v_agent = agent_manager._ghost_agents[v_id].agent
                    self.graph.remove_edge(u_agent, v_agent)
                    # if ghost not participating in network, then remove
                    # Note that remove_ghost will check for inclusion in requested
                    if not self._has_edge(v_agent):
                        agent_manager.remove_ghost(v_agent)
                        self.graph.remove_node(v_agent)
                else:
                    v_agent = agent_manager._local_agents[v_id]
                    # u_agent is GhostAgent so get agent
                    u_agent = u_agent.agent
                    self.graph.remove_edge(u_agent, v_agent)
                    # if ghost not participating in network, then remove
                    # Note that remove_ghost will check for inclusion in requested
                    if not self._has_edge(u_agent):
                        agent_manager.remove_ghost(u_agent)
                        self.graph.remove_node(u_agent)

                self._remove_edge_key(u_agent, v_agent)

    def _sync_vertices(self, agent_manager: AgentManager, create_agent: Callable) -> List:
        """Synchronizes vertices across processes.

        When this rank creates an edge between a vertex local to this rank
        and a ghost, this will send the local agent to the ghost rank to synchronize
        the network. The edge data will be added to the List returned from this
        method, so that edge can be created on the ghost rank as well.

        Args:
            agent_manager: the agent manager for this model
            create_agent: a callable used to create any necessary agents

        Returns:
            A nested list of the edge data tuples, (u.uid, v.uid, edge_attributes), for
            edges created betwee agents local to this rank and ghosts on this rank. The
            list is formatted in "alltoall" format such that index of each nested element
            is the rank where to send those elements.
        """
        sync_agents = [[] for i in range(self.comm.size)]
        sync_edges = [[] for i in range(self.comm.size)]
        for nodes, ghosted_edge in self.new_edges.items():
            local_agent = ghosted_edge.local_agent
            ghost_agent = ghosted_edge.ghost_agent
            other_rank = ghost_agent.local_rank
            self.ghosted_edges[nodes] = ghosted_edge
            # not already ghosted so send agent itself
            if not agent_manager.is_ghosted_to(other_rank, local_agent.uid):
                sync_agents[other_rank].append(local_agent.save())
                agent_manager.tag_as_ghosted(other_rank, local_agent.uid)

            sync_edges[other_rank].append((nodes[0].uid, nodes[1].uid, ghosted_edge.edge_attr))
        self.new_edges.clear()

        for edge_key in self.edges_to_update:
            ghosted_edge = self.ghosted_edges[edge_key]
            other_rank = ghosted_edge.ghost_agent.local_rank
            sync_edges[other_rank].append((edge_key[0].uid, edge_key[1].uid, ghosted_edge.edge_attr))
        self.edges_to_update.clear()

        recv_agents = self.comm.alltoall(sync_agents)
        for rank, agents in enumerate(recv_agents):
            for agent_data in agents:
                agent = create_agent(agent_data)
                # 1 because added to network
                agent_manager.add_ghost(rank, agent, 1)
                self.add(agent)

        return sync_edges

    def _sync_edges(self, sync_edges: List, agent_manager: AgentManager):
        """Syncronizes edges across ranks between local and non-local agents

        When this rank creates an edge between a vertex local to this rank
        and a ghost, this will ghost that edge to the ghost rank. The sync_edges
        list is expected to be the return value of _sync_vertices.

        Args:
            sync_edges: a nested list of the edge data tuples, (u.uid, v.uid, edge_attributes), for
                edges created betwee agents local to this rank and ghosts on this rank. The
                list is formatted in "alltoall" format such that index of each nested element
                is the rank where to send those elements.
            agent_manager: the AgentManager
        """
        recv_edges = self.comm.alltoall(sync_edges)
        for edges in recv_edges:
            for u_id, v_id, edge_attr in edges:
                u_agent = agent_manager._ghost_agents.get(u_id)
                if u_agent is None:
                    # u_agent is local
                    u_agent = agent_manager._local_agents[u_id]
                    v_agent = agent_manager._ghost_agents[v_id].agent
                else:
                    # u_agent is GhostAgent so get agent from it
                    u_agent = u_agent.agent
                    v_agent = agent_manager._local_agents[v_id]

                self.graph.add_edge(u_agent, v_agent, **edge_attr)
                self._add_edge_key(u_agent, v_agent)

    def _synch_ghosts(self, agent_manager: AgentManager, create_agent: Callable):
        """Synchronizes the edges and nodes of this projection across ranks.

        Args:
            agent_manager: the AgentManager
            create_agent: a callable that can create an agent instance from
                an agent id and data.
        """
        self._sync_removed(agent_manager)
        sync_edges = self._sync_vertices(agent_manager, create_agent)
        self._sync_edges(sync_edges, agent_manager)

    def _pre_synch_ghosts(self, agent_manager: AgentManager):
        """Called prior to synchronizing ghosts and before any cross-rank movement
        synchronization.

        This synchronizes any necessary proj reference count updates to
        ranks that ghost to this rank.

        Args:
            agent_manager: this rank's AgentManager
        """
        send_msg = [[] for i in range(self.comm.size)]
        for agent in self.ghosts_to_ref:
            if self.graph.has_node(agent):
                ghost = agent_manager.get_ghost(agent.uid)
                if ghost is None:
                    agent_manager.add_ghost(agent.local_rank, agent)

                # ghosting rank need to increment ref count to this rank
                send_msg[agent.local_rank].append(agent.uid)

        self.ghosts_to_ref.clear()

        recv_msg = self.comm.alltoall(send_msg)
        for rank, agents in enumerate(recv_msg):
            for agent in agents:
                agent_manager.tag_as_ghosted(rank, agent)

    def _post_agents_moved_rank(self, agent_manager: AgentManager, create_agent: Callable):
        """Notifies this projection that all the agent movement synchronization has occured,
        allowing this projection to perform any necessary actions.

        Args:
            agent_manager: the AgentManager
            create_agent: a callable that can create an agent instance from
                an agent id and data.
        """
        for u_id, v_id, edge_attr in self._agent_moved_edges:
            u = agent_manager.get_local(u_id)
            if u is None:
                # incr 0 because should be already as ghost so
                # ref_count includes this network
                u = agent_manager.get_ghost(u_id, incr=0)

            v = agent_manager.get_local(v_id)
            if v is None:
                # incr 0 because should be already as ghost so
                # ref_count includes this network
                v = agent_manager.get_ghost(v_id, incr=0)
            self.add_edge(u, v, **edge_attr)

        self._agent_moved_edges.clear()
        self._synch_ghosts(agent_manager, create_agent)

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
            agent_manager: the AgentManager

        """
        # 1. Check if moving agent is in edge with ghost, and ghost
        # is not requested, meaning that the edge is an artifact of
        # network sync and not explicitly created on this rank. In that
        # case, remove the edge, and the node (if the node is not in any other
        # edges)
        #
        # 2. Check if moving agent is in edge with local agent. This edge
        # needs to be preserved. So, delete edge, and add this rank to the
        # dictionary in the moved agent_data list (the data used by the dest rank
        # to set up ghosting of the moved agent).
        nodes_to_remove = []
        edges_to_remove = []
        for u, v, attr in self._edges(moving_agent, data=True):
            other = v if u.uid == moving_agent.uid else u
            if other.local_rank != self.rank and not agent_manager.is_requested(other.uid):
                # other is ghosted and not requested so the edge is artifact of edge created
                # on other node between u and v, so remove it.
                edges_to_remove.append((u, v))
                if self.num_edges(other) == 1:
                    # other is only on this rank to to be part of the removed edge
                    # and now the edge is gone so remove as node and ghost
                    nodes_to_remove.append(other)
                    agent_manager.remove_ghost(other)
                    self._no_longer_ghosts[other.local_rank].append(other.uid)

            else:
                # moving agent participates in edges with local agents
                # this should tell destination rank to ghost the moved
                # agent back to this rank, so the edge remains live
                if len(moving_agent_data) == 1:
                    moving_agent_data.append({self.rank: 1})
                else:
                    if self.rank in moving_agent_data[1]:
                        moving_agent_data[1][self.rank] += 1
                    else:
                        moving_agent_data[1][self.rank] = 1

                # agent will be ghosted to this rank, so need to add it as a
                # ghost
                agent_manager.add_ghost(dest_rank, moving_agent, incr=0)
                self._agent_moved_edges.append((u.uid, v.uid, attr))

        for edge in edges_to_remove:
            self.graph.remove_edge(*edge)
            self._remove_edge_key(*edge)

        for node in nodes_to_remove:
            self.graph.remove_node(node)

        self.graph.remove_node(moving_agent)

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
            agent_manager: AgentManager
        """
        stop_ghosting_ids = self.comm.alltoall(self._no_longer_ghosts)
        for ghost_rank, ids in enumerate(stop_ghosting_ids):
            for id in ids:
                agent_manager.untag_as_ghosted(ghost_rank, id)
        self._no_longer_ghosts = [[] for _ in range(self.comm.size)]

        for uid, dest in moved_agents:
            if dest == self.rank:
                # Check if moved agent was formerly a ghost on this rank
                # and if so, then copy the edges the ghost participated in
                # to new edges created with the now local agent, and remove
                # the old ghost
                ghost = agent_manager.get_ghost(uid)
                if ghost is not None:
                    agent = agent_manager.get_local(uid)
                    edge_list = [e for e in self._edges(ghost, data=True)]
                    for u, v, attr in edge_list:
                        self.graph.remove_edge(u, v)
                        if u.uid == agent.uid:
                            self.graph.add_edge(agent, v, **attr)
                        else:
                            self.graph.add_edge(u, agent, **attr)
                    self.graph.remove_node(ghost)
            else:
                # check if moved agent is a ghost on this rank,
                # if so, then check if that agent participated in
                # a ghosted edge. If so, then update the local rank
                # of that ghost agent so that ghosted edge will be
                # ghosted to new destination, and add that edge to
                # the new edges, so it will be ghosted to new rank
                # during next sync.
                agent = agent_manager.get_ghost(uid, 0)
                if agent is not None:
                    for edge in self._edges(agent):
                        edge_key = self._get_edge_key(edge)
                        ge = self.ghosted_edges.get(edge_key, None)
                        if ge is not None:
                            ge.ghost_agent.local_rank = dest
                            self.new_edges[edge_key] = ge


class UndirectedSharedNetwork(SharedNetwork):
    """Encapsulates an undirected network shared over multiple processes.

    This wraps a `networkx <https://networkx.org>`_ Graph object and delegates all the network
    related operations to it. That Graph is exposed as the `graph`
    attribute. See the `networkx <https://networkx.org>`_ Graph documentation for more information
    on the network functionality that it provides. **The network structure
    should NOT be changed using the networkx functions and methods
    (adding and removing nodes, for example).** Use this class'
    methods for manipulating the network structure.

    Attributes:
        graph (networkx.Graph): a network graph object responsible for
            network operations.
        comm (MPI.Comm): the communicator over which the network is shared.
        name (str): the name of this network.

    Args:
        name: the name of the SharedNetwork
        comm: the communicator over which this SharedNetwork is distributed
        directed: specifies whether this SharedNetwork is directed or not
    """

    def __init__(self, name: str, comm: MPI.Comm):
        super().__init__(name, comm, nx.Graph())
        self.canonical_edge_keys = {}

    @property
    def is_directed(self) -> bool:
        """Gets whether or not this network is directed. Returns False

        Returns:
            False
        """
        return False

    def _has_edge(self, agent: Agent) -> bool:
        """Gets whether or not the specified agent participates in
        an edge in this network.

        Args:
            agent: the agent to check
        Returns:
            True if the agent is part of an edge, otherwise false
        """
        return len(self.graph.edges(agent)) > 0

    def _get_edge_key(self, edge: Tuple) -> Tuple:
        """Gets the canonical edge key used to manage dictionaries of edges.

        Args:
            edge: the edge whose key we are getting
        Returns:
            The canonical edge key
        """
        return self.canonical_edge_keys[edge]

    def _add_edge_key(self, u_agent: Agent, v_agent: Agent):
        """Creates a canonical edge key for the pair of nodes.

        Args:
            u_agent: the u node agent
            v_agent: the v node agent
        """
        edge_key = (u_agent, v_agent)
        self.canonical_edge_keys[edge_key] = edge_key
        self.canonical_edge_keys[(v_agent, u_agent)] = edge_key
        return edge_key

    def _remove_edge_key(self, u_agent: Agent, v_agent: Agent):
        """Removes the canonical edge key for the specified edge.

        Args:
            u_agent: the u node agent
            v_agent: the v node agent
        """
        edge_key = (u_agent, v_agent)
        self.canonical_edge_keys.pop(edge_key)
        self.canonical_edge_keys.pop((v_agent, u_agent))

    def add_edge(self, u_agent: Agent, v_agent: Agent, **kwattr):
        """Adds an edge between u_agent and v_agent.

        If the u and v agents are not existing nodes in the network, they
        will be added. Edge attributes can be added using keyword
        arguments.

        Args:
            u_agent: The u agent
            v_agent: The v agent
            kwattr: optional keyword arguments for assigning edge data.

        Examples:
            Add an edge with a weight attribute

            >>> g.add_edge(agent1, agent2, weight=3.1)
        """

        if u_agent.local_rank == self.rank and v_agent.local_rank != self.rank:
            if not self.graph.has_node(v_agent):
                self.ghosts_to_ref.append(v_agent)
            self.graph.add_edge(u_agent, v_agent, **kwattr)
            edge_key = self._add_edge_key(u_agent, v_agent)
            ge = GhostedEdge(u_agent, v_agent, self.graph.edges[edge_key])
            self.new_edges[edge_key] = ge
            self.edges_to_remove.pop(edge_key, None)
        elif u_agent.local_rank != self.rank and v_agent.local_rank == self.rank:
            # assume v_agent is local
            if not self.graph.has_node(u_agent):
                self.ghosts_to_ref.append(u_agent)
            self.graph.add_edge(u_agent, v_agent, **kwattr)
            edge_key = self._add_edge_key(u_agent, v_agent)
            ge = GhostedEdge(v_agent, u_agent, self.graph.edges[edge_key])
            self.new_edges[edge_key] = ge
            self.edges_to_remove.pop(edge_key, None)
        else:
            self._add_edge_key(u_agent, v_agent)
            self.graph.add_edge(u_agent, v_agent, **kwattr)

    def _edges(self, agent: Agent, data: bool = False):
        """Gets an iterator over the incoming and outgoing edges for the specifed agent.

        Args:
            agent: agent whose edges will be returned
            data: if true, the edge data dictionary will be returned, otherwise
            not

        Returns:
            An iterator over the incoming and outgoing edges for the specifed agent
        """
        return self.graph.edges(agent, data=data)

    def num_edges(self, agent: Agent) -> int:
        """Gets the number of edges that contain the specified agent.

        Args:
            agent: agent whose edge will be counted

        Returns:
            The number of edges that contain the specified agent
        """
        return len(self.graph.edges(agent))


class DirectedSharedNetwork(SharedNetwork):
    """Encapsulates a directed network shared over multiple process ranks.

    This wraps a `networkx <https://networkx.org>`_ DiGraph object and delegates all the network
    related operations to it. That Graph is exposed as the `graph`
    attribute. See the `networkx <https://networkx.org>`_ Graph documentation for more information
    on the network functionality that it provides. **The network structure
    must NOT be changed using the networkx functions and methods
    (adding and removing nodes, for example)**. Use this class'
    methods for manipulating the network structure.

    Attributes:
        graph (networkx.DiGraph): a `networkx <https://networkx.org>`_ graph object wrapped by this class.
        comm (MPI.Comm): the communicator over which the network is shared.
        names (str): the name of this network.

    Args:
        name: the name of the SharedNetwork
        comm: the communicator over which this DirectedSharedNetwork is distributed
    """

    def __init__(self, name: str, comm: MPI.Comm):
        super().__init__(name, comm, nx.DiGraph())

    @property
    def is_directed(self) -> bool:
        """Gets whether or not this network is directed.

        Returns:
            True
        """
        return True

    def _has_edge(self, agent: Agent) -> bool:
        """Gets whether or not the specified agent participates in
        an edge in this network.

        Args:
            agent: the agent to check
        Returns:
            True if the agent is part of an edge, otherwise false
        """
        return len(self.graph.in_edges(agent)) > 0 or len(self.graph.out_edges(agent)) > 0

    def _get_edge_key(self, edge: Tuple):
        """Gets the canonical edge key used to manage dictionaries of edges.

        Returns the passed in edge on a SharedDirectedNetwork.

        Args:
            edge: the edge whose key we are getting
        Returns:
            The canonical edge key
        """
        return edge

    def _remove_edge_key(self, u_agent: Agent, v_agent: Agent):
        """Removes the canonical edge key for the specified edge.

        Null op on SharedDirectedNetwork.

        Args:
            u_agent: the u node agent
            v_agent: the v node agent
        """
        pass

    def _add_edge_key(self, u_agent: Agent, v_agent: Agent):
        """Creates a canonical edge key for the pair of nodes.

        This is a NOOP on SharedDirectedNetwork

        Args:
            u_agent: the u node agent
            v_agent: the v node agent
        """
        pass

    def add_edge(self, u_agent: Agent, v_agent: Agent, **kwattr):
        """Adds an edge beteen u_agent and v_agent.

        If the u and v agents are not existing nodes in the network, they
        will be added. Edge attributes can be added using keyword
        arguments.

        Args:
            u_agent: The u agent
            v_agent: The v agent
            kwattr: optional keyword arguments for assigning edge data.

        Examples:
            Add an edge with a weight attribute

            >>> g.add_edge(agent1, agent2, weight=3.1)
        """
        if u_agent.local_rank == self.rank and v_agent.local_rank != self.rank:
            if not self.graph.has_node(v_agent):
                self.ghosts_to_ref.append(v_agent)
            self.graph.add_edge(u_agent, v_agent, **kwattr)
            edge_key = (u_agent, v_agent)
            ge = GhostedEdge(u_agent, v_agent, self.graph.edges[edge_key])
            self.new_edges[edge_key] = ge
            self.edges_to_remove.pop(edge_key, None)
        elif u_agent.local_rank != self.rank and v_agent.local_rank == self.rank:
            # assume v_agent is local
            if not self.graph.has_node(u_agent):
                self.ghosts_to_ref.append(u_agent)
            self.graph.add_edge(u_agent, v_agent, **kwattr)
            edge_key = (u_agent, v_agent)
            ge = GhostedEdge(v_agent, u_agent, self.graph.edges[edge_key])
            self.new_edges[edge_key] = ge
            self.edges_to_remove.pop(edge_key, None)
        else:
            self.graph.add_edge(u_agent, v_agent, **kwattr)

    def _edges(self, agent: Agent, data: bool = False):
        """Gets an iterator over the incoming and outgoing edges for the specifed agent.

        Args:
            agent: agent whose edges will be returned
            data: if true, the edge data dictionary will be returned, otherwise
            not

        Returns:
            An iterator over the incoming and outgoing edges for the specifed agent
        """
        return chain(self.graph.out_edges(agent, data=data), self.graph.in_edges(agent, data=data))

    def num_edges(self, agent: Agent) -> int:
        """Gets the number of edges that contain the specified agent.

        Returns:
            The number of edges that contain the specified agent
        """
        return len(self.graph.out_edges(agent)) + len(self.graph.in_edges(agent))


def _parse_graph_description(line: str):
    vals = line.split(' ')
    if len(vals) != 2:
        raise ValueError('Error reading graph description file. Invalid format on first line.')
    try:
        val = int(vals[1])
    except Exception:
        raise ValueError('Error reading graph description file. Invalid format on first line. Second value must be an integer')

    return (val[0], val != 0)


@dataclass
class GraphData:
    rank: int
    agents: Dict
    remote_agents: Dict
    requested_agents: List
    edges: List


def _parse_graph_desc(line: str):
    try:
        vals = line.split(' ')
        if len(vals) != 2:
            raise ValueError

        id = vals[0]
        is_directed = int(vals[1]) != 0

        return (id, is_directed)
    except Exception:
        raise ValueError('Error reading graph description file on line 1. Expected graph description with '
                         'format: "id, [0|1]", where 0 indicates an undirected graph and 1 directed')


_LINE_P = re.compile('\{[^}]+\}|\S+')
_EMPTY_DICT = {}


def _parse_node(line: str, line_num: int, graph_data: GraphData, ctx, create_agent: Callable):
    try:
        vals = _LINE_P.findall(line)
        n_vals = len(vals)
        if n_vals < 3 or n_vals > 4:
            raise ValueError
        n_id = int(vals[0])
        agent_type = int(vals[1])
        rank = int(vals[2])

        if rank == graph_data.rank or graph_data.rank == -1:
            attribs = _EMPTY_DICT
            if n_vals == 4:
                attribs = json.loads(vals[3])

            agent = create_agent(n_id, agent_type, rank, **attribs)
            ctx.add(agent)
            graph_data.agents[n_id] = agent
        else:
            graph_data.remote_agents[n_id] = (n_id, agent_type, rank)

    except Exception:
        raise ValueError(f'Error reading graph description file on line {line_num}: "{line}". Expected node description with format: '
                         '"node_id agent_type rank" following by an optional json agent attribute dictionary')


def _request_remote_agents(graph_data: GraphData, ctx, create_agent: Callable):
    requested_agents = ctx.request_agents(graph_data.requested_agents, create_agent)
    for agent in requested_agents:
        graph_data.agents[agent.uid[0]] = agent


def _parse_edge(line: str, line_num: int, graph, graph_data: GraphData):
    requested = set()
    try:
        vals = _LINE_P.findall(line)
        n_vals = len(vals)
        if n_vals < 2 or n_vals > 3:
            raise ValueError

        u_id = int(vals[0])
        v_id = int(vals[1])

        u_in = u_id in graph_data.agents
        v_in = v_id in graph_data.agents

        if u_in and v_in:
            u = graph_data.agents[u_id]
            v = graph_data.agents[v_id]
            if n_vals == 3:
                graph.add_edge(u, v, **(json.loads(vals[2])))
            else:
                graph.add_edge(u, v)
        elif u_in and not v_in:
            v_uid = graph_data.remote_agents[v_id]
            if v_id not in requested:
                graph_data.requested_agents.append((v_uid, v_uid[2]))
                requested.add(v_id)

            u_uid = graph_data.agents[u_id].uid
            if n_vals == 3:
                graph_data.edges.append((u_uid, v_uid, vals[2]))
            else:
                graph_data.edges.append((u_uid, v_uid))

        elif not u_in and v_in:
            u_uid = graph_data.remote_agents[u_id]
            if u_id not in requested:
                graph_data.requested_agents.append((u_uid, u_uid[2]))
                requested.add(u_id)
            v_uid = graph_data.agents[v_id].uid
            if n_vals == 3:
                graph_data.edges.append((u_uid, v_uid, vals[2]))
            else:
                graph_data.edges.append((u_uid, v_uid))

    except KeyError:
        raise ValueError(f'Error reading graph description file on line {line_num}. Agent with {u_id} or {v_id} cannot be found')

    except Exception:
        raise ValueError(f'Error reading graph description file on line {line_num}. Expected edge description with format: '
                         '"u_id v_id" following by an optional json edge attribute dictionary')


def _create_edges(graph, graph_data: GraphData):
    for edge in graph_data.edges:
        u = graph_data.agents[edge[0][0]]
        v = graph_data.agents[edge[1][0]]

        if len(edge) == 3:
            graph.add_edge(u, v, **(json.loads(edge[2])))
        else:
            graph.add_edge(u, v)


def read_network(fpath: str, ctx, create_agent: Callable, restore_agent: Callable):
    """Creates and initializes the network described in the specified file.

    The network will be created and added to the specified context as a projection. The nodes
    defined in the file will be created as agents. Those agents with a local rank will be added
    to the specified context, and those remote agents that participate in an edge with a local
    agent with be added as ghost agents.

    The format of the file is as follows. The first line consists of a network id (name)
    followed by a space followed by 0 or 1, where
    0 indicates that the network is undirected and 1 that it is directed. This first line is followed the node descriptions.
    Each node description line defines a node and consists at least 3 space separated elements.
    These 3 are the node id, the agent type,
    and the rank on which the agent should be created. These 3 are optionally followed by a json dictionary string
    containing any attributes of that agent. The node descriptions should be followed by "EDGES" to indicate
    that the edge descriptions follow in the remaining lines. Each edge description line defines an edge and consists of at least 2
    space separated elements. These two are the node ids of the u and v components of the edge. An optional 3rd
    element, a json dictionary, defines any attributes (e.g., weight) to associate with that edge. For example::

        friend_network 0
        1 0 1 {"age": 23}
        2 0 1 {"age" : 24}
        3 0 0 {"age" : 30}
        EDGES
        1 2 {"weight": 0.5}
        3 1 {"weight": 0.75}

    Without node or edge attributes:::

        friend_network 0
        1 0 1
        2 0 1
        3 0 0
        EDGES
        1 2
        3 1

    The node id, the type and the rank elements are passed to a user specified Callable to create the agents.
    In that Callable, these 3 elements must be used to create the agents' unique ids

    Args:
        fpath: the path to the network description file.
        ctx (repast4py.context.SharedContext): the context to add the agents to
        create_agent: a Callable used to create an agent from a node description.
            The Callable must have the following signature :samp:`(node_id: int, agent_type: int, rank: int, **agent_attributes)`
            and return an agent.
        restore_agent: a Callable used to deserialize agents from agent data sent from
            another rank, that is, one that takes the tuple returned by an agent's save()
            method and returns an agent.
    """
    with open(fpath) as f_in:
        gid, is_directed = _parse_graph_desc(f_in.readline())
        g = DirectedSharedNetwork(gid, ctx.comm) if is_directed else UndirectedSharedNetwork(gid, ctx.comm)
        ctx.add_projection(g)
        # -1 flags single proc run
        gd_rank = ctx.rank if ctx.comm.size > 1 else -1
        graph_data = GraphData(rank=gd_rank, agents={}, remote_agents={}, requested_agents=[],
                               edges=[])
        parse_nodes = True
        for i, line in enumerate(f_in):
            line = line.strip()
            if parse_nodes:
                if line == 'EDGES':
                    parse_nodes = False

                else:
                    _parse_node(line, i + 2, graph_data, ctx, create_agent)
            else:
                _parse_edge(line, i + 2, g, graph_data)

    _request_remote_agents(graph_data, ctx, restore_agent)
    ctx.comm.Barrier()
    _create_edges(g, graph_data)
    ctx.synchronize(restore_agent)


def _write_node(f_out, node_data: Tuple, rank: int):
    n = node_data[0]
    attrib = node_data[1]
    agent_type = 0
    if 'agent_type' in attrib:
        try:
            agent_type = int(attrib['agent_type'])
            if agent_type < 0:
                raise ValueError('Error generating graph description file. Node attribute "agent_type" must be a non-negative integer.')
            attrib = attrib.copy()
            del attrib['agent_type']
        except Exception:
            raise ValueError('Error generating graph description file. Node attribute "agent_type" must be a non-negative integer.')

    line = f'{n} {agent_type} {rank}'
    if (len(attrib) > 0):
        line = f'{line} {json.dumps(attrib)}'

    f_out.write(line)
    f_out.write('\n')


def _write_edges(f_out, graph: nx.Graph):
    f_out.write('EDGES\n')
    for u, v, attrib in graph.edges(data=True):
        line = f'{u} {v}'
        if (len(attrib) > 0):
            line = f'{line} {json.dumps(attrib)}'

        f_out.write(line)
        f_out.write('\n')


def _random_partition(graph: nx.Graph, network_name: str, fpath: str, n_ranks: int, **partitioning_args):
    num_nodes = graph.number_of_nodes()
    nodes_per_rank = num_nodes / n_ranks
    a = np.arange(n_ranks)
    assigned_ranks = np.tile(a, math.ceil(nodes_per_rank))
    # assigned_ranks = array.array('i', (i for i in range(n_ranks))) * math.ceil(nodes_per_rank)
    if 'rng' in partitioning_args:
        rng_val = partitioning_args['rng']
        if rng_val == 'default':
            rng = random.default_rng
        else:
            rng = rng_val
    else:
        rng = random.default_rng
    rng.shuffle(assigned_ranks)

    if len(assigned_ranks) > num_nodes:
        assigned_ranks = assigned_ranks[: num_nodes]

    with open(fpath, 'w') as f_out:
        is_d = 1 if nx.is_directed(graph) else 0
        f_out.write(f'{network_name} {is_d}\n')
        for i, nd in enumerate(graph.nodes(data=True)):
            _write_node(f_out, nd, assigned_ranks[i])

        _write_edges(f_out, graph)


def _metis_partition(graph: nx.Graph, network_name: str, fpath: str, n_ranks: int, **partitioning_args):
    # import here so that users without nxmetis can
    # use the other network code.
    import nxmetis

    _, partitions = nxmetis.partition(graph, n_ranks, **partitioning_args)

    with open(fpath, 'w') as f_out:
        is_d = 1 if nx.is_directed(graph) else 0
        f_out.write(f'{network_name} {is_d}\n')
        for i, partition in enumerate(partitions):
            for nid in partition:
                _write_node(f_out, (nid, graph.nodes[nid]), i)

        _write_edges(f_out, graph)


def write_network(graph: nx.Graph, network_name: str, fpath: str, n_ranks: int, **partition_args):
    """
    Partitions a specified network over the specified process ranks and writes that network to a file
    in a format that can be read by the :func:`repast4py.network.read_network` function. The intention is
    that `write_network` is used to create a distributed partitioned graph file outside of a running
    simulation, and once created that file can then be read during simulation initialization to create
    a repast4py network.

    The format of the file is as follows. The first line consists of a network id (name)
    followed by a space followed by 0 or 1, where
    0 indicates that the network is undirected and 1 that it is directed. This first line is followed the node descriptions.
    Each node description line defines a node and consists at least 3 space separated elements.
    These 3 are the node id, the agent type,
    and the rank on which the agent should be created. These 3 are optionally followed by a json dictionary string
    containing any attributes of that agent. The node descriptions should be followed by "EDGES" to indicate
    that the edge descriptions follow in the remaining lines. Each edge description line defines an edge and consists of at least 2
    space separated elements. These two are the node ids of the u and v components of the edge. An optional 3rd
    element, a json dictionary, defines any attributes (e.g., weight) to associate with that edge. If a node in the graph
    contains an 'agent_type' attribute then that will written as the agent type id for that node, otherwise the type is 0.

    The network is partitioned across process ranks by assigning each node to a rank. A `partition_method`
    can be specified in the `partition_args` and can be one of 'random' or 'metis'. If no `partition_method`
    is defined, then the method defaults to random where the nodes will be uniformly distributed among the process
    ranks without any load balancing. A partition method of `metis` will use the metis
    load balancer to allocate nodes to ranks. Metis is not included with repast4py and must be installed
    separately.  See the `networkx metis docs <https://networkx-metis.readthedocs.io/en/latest/index.html>`__ for
    more information on installation and nxmetis. Note that installing from source or github may be necessary.

    Args:
        graph: the graph to partition and write out.
        network_name: the id / name  to assign to the partitioned network
        fpath: the path of the file to write the partitioned network to
        n_ranks: the number of ranks to partition the network over
        partition_args: keyword arguments that determine how the graph will be partitioned.

            * The `partition_method` keyword is used to determine the partition method ('random' or 'metis').

            If `partition_method` is 'random':
                * `rng` specifies the random number generator to use in the random partitioning. Valid values are any numpy.random.Generator instance or 'default' to use the repast4py.random.default_rng. If no 'rng' is specified then by default the repast4py.random.default_rng is used.

            If `partition_method` is 'metis':
                * The partition_args are forwared to the nxmetis' partition function. The various arguments to that can be found in the `networkx metis reference <https://networkx-metis.readthedocs.io/en/latest/reference/generated/nxmetis.partition.html>`__.

    Examples:
        Random Patitioning

        >>> import networkx as nx
        >>> g = nx.generators.dual_barabasi_albert_graph(60, 2, 1, 0.25)
        >>> fname = './test_data/gen_net_test.txt'
        >>> write_network(g, "test", fname, 3, partition_method='random', rng='default')

        Metis Partitioning

        >>> g = nx.complete_graph(60)
        >>> options = nxmetis.types.MetisOptions(seed=1)
        >>> write_network(g, "metis_test", fname, 3, partition_method='metis', options=options)
    """
    if 'partition_method' in partition_args:
        partition_method = partition_args['partition_method']
        if partition_method == 'random':
            _random_partition(graph, network_name, fpath, n_ranks, **partition_args)
        elif partition_method == 'metis':
            del partition_args['partition_method']
            _metis_partition(graph, network_name, fpath, n_ranks, **partition_args)
        else:
            raise ValueError('Invalid partition method, must be one of "random" or "metis"')
    else:
        _random_partition(graph, network_name, fpath, n_ranks, **partition_args)
