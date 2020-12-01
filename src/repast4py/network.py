from mpi4py import MPI
from networkx import OrderedGraph, OrderedDiGraph

from typing import List, Iterable

from ._core import Agent


class SharedNetwork:

    def __init__(self, name: str, comm: MPI.Comm, graph):
        """Creates a SharedNetwork with the specified name, communicator
        and type (directed or not).

        Args:
            name: the name of the SharedNetwork
            comm: the communicator over which this SharedNetwork is distributed
            graph (networkx Graph): the networkx graph object responsible for
            network related operations
        """
        self.name = name
        self.comm = comm
        self.graph = graph

    def __iter__(self) -> Iterable[Agent]:
        """Gets an Iterable over all the nodes (agents) in 
        the network.

        Returns:
            An Iterable over all the nodes (agents) in the network.
        """
        return self.graph.__iter__()

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
        """Adds each agent in specified list of agents as
        nodes in the network.
        """
        self.graph.add_nodes_from(agents)

    def add_edge(self, u_agent: Agent, v_agent: Agent, **kwattr):
        """Adds an edge betwwn u_agent and v_agent. If the
        u and v agents are not existing nodes in the network, they
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

        self.graph.add_edge(u_agent, v_agent, **kwattr)

    def remove(self, agent: Agent):
        """Removes the specified agent from this SharedNetwork.

        Args:
            agent: the agent to remove

        Raises:
            NetworkXError if the agent is not a node in this
            SharedNetwork.
        """
        self.graph.remove_node(agent)

    def remove_edge(self, u_agent: Agent, v_agent: Agent):
        """Removes the edge between u_agent and v_agent from
        this SharedNetwork.

        Args:
            u_agent: the u agent.
            v_agent: the v agent.
        Raises:
            NetworkXError if there is no edge between u_agent and v_agent.
        """
        self.graph.remove_edge(u_agent, v_agent)

    def clear_edges(self):
        """Removes all the edges.
        """
        self.graph.clear_edges()

    def contains_edge(self, u_agent: Agent, v_agent: Agent) -> bool:
        """Gets whether or not an edge exists between u_agent and v_agent.

        Returns:
            True if an edge exists, otherwise False.
        """
        return self.graph.has_edge(u_agent, v_agent)


class UndirectedSharedNetwork(SharedNetwork):
    """Encapsulates an undirected network shared over multiple processes.

    This wraps a networkx Graph object and delegates all the network
    related operations to it. That Graph is exposed as the `graph`
    attribute. See the networkx Graph documentation for more information
    on the network functionality that it provides. The network structure
    should _NOT_ be changed using the networkx functions and methods
    (adding and removing nodes, for example). Use this class' parent class
    methods for manipulating the network structure.

    Attributes:
        graph (networkx.OrderedGraph): a network graph object responsible for
        network operations.
        comm (MPI.Comm): the communicator over which the network is shared.
        names (str): the name of this network.
    """

    def __init__(self, name: str, comm: MPI.Comm):
        """Creates a SharedNetwork with the specified name, communicator
        and type (directed or not).

        Args:
            name: the name of the SharedNetwork
            comm: the communicator over which this SharedNetwork is distributed
            directed: specifies whether this SharedNetwork is directed or not
        """
        super().__init__(name, comm, OrderedGraph())

    @property
    def is_directed(self) -> bool:
        """Returns False
        """
        return False


class DirectedSharedNetwork(SharedNetwork):
    """Encapsulates a directed network shared over multiple processes.

    This wraps a networkx DiGraph object and delegates all the network
    related operations to it. That Graph is exposed as the `graph`
    attribute. See the networkx Graph documentation for more information
    on the network functionality that it provides. The network structure
    should _NOT_ be changed using the networkx functions and methods
    (adding and removing nodes, for example). Use this class' parent class
    methods for manipulating the network structure.

    Attributes:
        graph (networkx.OrderedDiGraph): a network graph object responsible for
        network operations.
        comm (MPI.Comm): the communicator over which the network is shared.
        names (str): the name of this network.
    """

    def __init__(self, name: str, comm: MPI.Comm):
        """Creates a SharedNetwork with the specified name, communicator
        and type (directed or not).

        Args:
            name: the name of the SharedNetwork
            comm: the communicator over which this DirectedSharedNetwork is distributed
            directed: specifies whether this DirectedSharedNetwork is directed or not
        """
        super().__init__(name, comm, OrderedDiGraph())

    @property
    def is_directed(self) -> bool:
        """Returns True
        """
        return True
