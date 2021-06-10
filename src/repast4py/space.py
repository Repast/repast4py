from typing import Callable, List
import mpi4py

from ._space import DiscretePoint, ContinuousPoint
from ._space import SharedGrid as _SharedGrid
from ._space import SharedContinuousSpace as _SharedContinuousSpace
# not used in this file but imported into space namespace from _.
# for space API
from ._space import GridStickyBorders, GridPeriodicBorders
from ._space import CartesianTopology, Grid, ContinuousSpace

from .geometry import BoundingBox

from .core import AgentManager, Agent

from collections import namedtuple


class BorderType:
    """An enum defining the border types that can be used with a space or grid.

    The border type determines an agents location when the location is beyond the grid's or space's
    bounds. For example, during agent movement, and that movement carries the agent beyond the
    borders of a grid or space. Valid values are "Sticky" and "Periodic":
    * Sticky: clips any point coordinates to the maximum or minimum value when the coordinates are less than or
      greater than the grid or spaces maximum or minimum value. For example, if the minimum grid x location is 0, and
      an agent moves to an x of -1, then movment in the x dimension is stopped at 0.
    * Periodic: wraps any point coordinates when they coordinates coordinates are less than or
      greater than the grid or spaces maximum or minimum value. For example, if the minimum grid x location is 0,
      the maximum is 20, and an agent moves to an x of -2, then the new x coordinate is 19.

    """
    Sticky = 0
    Periodic = 1


class OccupancyType:
    """An enum defining the occupancy types of a location in a space or grid.

    Currently only "Multiple" is supported.
    * Multiple: allows any number of agents to exist in the same location. 
    """
    Multiple = 0


class SharedGrid(_SharedGrid):
    """An N-dimensional cartesian discrete space shared across ranks, where agents can occupy locations defined by
    a discretete integer coordinate.

    The grid is shared over all the ranks in the specified communicator by sub-dividing the global bounds into
    some number of smaller grids, one for each rank. For example, given a global grid size of (100 x 25) and
    2 ranks, the global grid will be split along the x dimension such that the SharedGrid in the first MPI rank
    covers (0-50 x 0-25) and the second rank (50-100 x 0-25). Each rank's SharedGrid contains buffers of a specified
    size that duplicate or "ghosts" an adjacent
    area of the neighboring rank's SharedGrid. In the above example, the rank 1 grid buffers the area from
    (50-52 x 0-25) in rank 2, and rank 2 buffers (48-50 x 0-25) in rank 1. Be sure to specify a buffer size appropriate
    to any agent behavior. For example, if an agent can "see" 3 units away and take some action based on what it
    perceives, then the buffer size should be at least 3, insuring that an agent can properly see beyond the borders of
    its own local SharedGrid. When an agent moves beyond the borders of its current SharedGrid, it will be transferred
    from its current rank, and into that containing the section of the global grid that it has moved into.

    Args:
        name: the name of the grid.
        bounds: the global dimensions of the grid.
        borders: the border semantics: BorderType.Sticky or BorderType.Periodic
        occupancy: the type of occupancy in each cell: OccupancyType.Multiple.
        buffersize: the size of this SharedGrid buffered area. This single value is used for all dimensions.
        comm: the communicator containing all the ranks over which this SharedGrid is shared.
    """

    def __init__(self, name: str, bounds: BoundingBox, borders: BorderType, occupancy: OccupancyType, buffersize: int,
                 comm: mpi4py.MPI.Intracomm):
        super().__init__(name, bounds, borders, occupancy, buffersize, comm)
        self.buffered_agents = []
        self.rank = comm.Get_rank()

        self.gather = self._gather_1d
        if bounds.yextent > 0:
            self.gather = self._gather_2d
        if bounds.zextent > 0:
            self.gather = self._gather_3d

    def __contains__(self, agent):
        return self.contains(agent)

    def _fill_send_data(self):
        send_data = [[] for i in range(self._cart_comm.size)]
        # bd: (rank, (ranges)) e.g., (1, (8, 10, 0, 0, 0, 0))
        for bd in self._get_buffer_data():
            data_list = send_data[bd[0]]
            ranges = bd[1]

            self.gather(data_list, ranges)

        return send_data

    def _process_recv_data(self, recv_data, agent_manager: AgentManager, create_agent):
        pt = DiscretePoint(0, 0, 0)
        for sending_rank, data_list in enumerate(recv_data):
            for data in data_list:
                agent_data = data[0]
                pt_data = data[1]
                # get will increment ref count if ghost exists
                agent = agent_manager.get_ghost(agent_data[0])
                if agent is None:
                    agent = create_agent(agent_data)
                    agent_manager.add_ghost(sending_rank, agent)
                self.buffered_agents.append(agent)
                self.add(agent)
                pt._reset(pt_data)
                self.move(agent, pt)

    def _pre_synch_ghosts(self, agent_manager: AgentManager):
        """Called prior to synchronizing ghosts and before any cross-rank movement
        synchronization.

        This removes the currently buffered agents.

        Args:
            agent_manager: this rank's AgentManager
        """
        for agent in self.buffered_agents:
            self.remove(agent)
            agent_manager.remove_ghost(agent)
        self.buffered_agents.clear()

    def _synch_ghosts(self, agent_manager: AgentManager, create_agent):
        """Synchronizes the ghosted part of this projection

        Args:
            agent_manager: this rank's AgentManager
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        send_data = self._fill_send_data()
        recv_data = self._cart_comm.alltoall(send_data)
        self._process_recv_data(recv_data, agent_manager, create_agent)

    def _gather_1d(self, data_list, ranges):
        pt = DiscretePoint(0, 0, 0)
        for x in range(ranges[0], ranges[1]):
            pt._reset1D(x)
            agents = self.get_agents(pt)
            for a in agents:
                data_list.append((a.save(), (pt.x, pt.y, pt.z)))

    def _gather_2d(self, data_list, ranges):
        pt = DiscretePoint(0, 0, 0)
        for x in range(ranges[0], ranges[1]):
            for y in range(ranges[2], ranges[3]):
                pt._reset2D(x, y)
                agents = self.get_agents(pt)
                for a in agents:
                    data_list.append((a.save(), (pt.x, pt.y, pt.z)))

    def _gather_3d(self, data_list, ranges):
        pt = DiscretePoint(0, 0, 0)
        for x in range(ranges[0], ranges[1]):
            for y in range(ranges[2], ranges[3]):
                for z in range(ranges[4], ranges[5]):
                    pt._reset3D(x, y, z)
                    agents = self.get_agents(pt)
                    for a in agents:
                        data_list.append((a.save(), (pt.x, pt.y, pt.z)))

    def _agent_moving_rank(self, moving_agent: Agent, dest_rank: int, moved_agent_data: List,
                           agent_manager: AgentManager):
        self.remove(moving_agent)

    def _agents_moved_rank(self, moved_agents: List, agent_manager: AgentManager):
        pass

    def _post_agents_moved_rank(self, agent_manager: AgentManager, create_agent: Callable):
        pass


class SharedCSpace(_SharedContinuousSpace):
    """An N-dimensional cartesian discrete space where agents can occupy locations defined by
    a continuous floating point coordinate.
    The space is shared over all the ranks in the specified communicator by sub-dividing the global bounds into
    some number of smaller spaces, one for each rank. For example, given a global spaces size of (100 x 25) and
    2 ranks, the global space will be split along the x dimension such that the SharedContinuousSpace in the first
    MPI rank covers (0-50 x 0-25) and the second rank (50-100 x 0-25). 
    Each rank's SharedContinuousSpace contains a buffer of the specified size that duplicates or "ghosts" an adjacent
    area of the neighboring rank's SharedContinuousSpace. In the above example, the rank 1 space buffers the area from
    (50-52 x 0-25) in rank 2, and rank 2 buffers (48-50 x 0-25) in rank 1. Be sure to specify a buffer size appropriate
    to any agent behavior. For example, if an agent can "see" 3 units away and take some action based on what it
    perceives, then the buffer size should be at least 3, insuring that an agent can properly see beyond the borders of
    its own local SharedContinuousSpace. When an agent moves beyond the borders of its current SharedContinuousSpace,
    it will be transferred
    from its current rank, and into that containing the section of the global grid that it has moved into.
    The SharedContinuousSpace uses a `tree <https://en.wikipedia.org/wiki/Quadtree>`_ (quad or oct depending on the number of 
    dimensions) to speed up spatial queries. The tree can be tuned using the tree threshold parameter.

    Args:
       name: the name of the grid.
       bounds: the global dimensions of the grid.
       borders: the border semantics: BorderType.Sticky or BorderType.Periodic
       occupancy: the type of occupancy in each cell: OccupancyType.Multiple.
       buffersize: the size of this SharedContinuousSpace's buffered area. This single value is used for all dimensions.
       comm: the communicator containing all the ranks over which this SharedGrid is shared.
       tree_threshold: the space's tree cell maximum capacity. When this capacity is reached, the cell splits.

    """

    def __init__(self, name: str, bounds: BoundingBox, borders: BorderType, occupancy: OccupancyType, 
                 buffersize: int, comm: mpi4py.MPI.Intracomm, tree_threshold: int):
        super().__init__(name, bounds, borders, occupancy, buffersize, comm, tree_threshold)
        self.buffered_agents = []

    def __contains__(self, agent):
        return self.contains(agent)

    def _pre_synch_ghosts(self, agent_manager):
        """Called prior to synchronizing ghosts and before any cross-rank movement
        synchronization.

        This removes the currently buffered agents.

        Args:
            agent_manager: this rank's AgentManager
        """
        for agent in self.buffered_agents:
            self.remove(agent)
            agent_manager.remove_ghost(agent)

        self.buffered_agents.clear()

    def _fill_send_data(self):
        send_data = [[] for i in range(self._cart_comm.size)]
        # bd: (rank, (ranges)) e.g., (1, (8, 10, 0, 0, 0, 0))

        for bd in self._get_buffer_data():
            data_list = send_data[bd[0]]
            box = bd[1]
            bounds = BoundingBox(box[0], box[1] - box[0], box[2], box[3] - box[2],
                                 box[4], box[5] - box[4])
            for a in self.get_agents_within(bounds):
                pt = self.get_location(a)
                data_list.append((a.save(), (pt.x, pt.y, pt.z)))

        return send_data

    def _process_recv_data(self, recv_data, agent_manager: AgentManager, create_agent):
        pt = ContinuousPoint(0, 0, 0)
        for sending_rank, data_list in enumerate(recv_data):
            for agent_data, pt_data in data_list:
                agent = agent_manager.get_ghost(agent_data[0])
                if agent is None:
                    agent = create_agent(agent_data)
                    agent_manager.add_ghost(sending_rank, agent)
                self.buffered_agents.append(agent)
                self.add(agent)
                pt._reset(pt_data)
                self.move(agent, pt)

    def _synch_ghosts(self, agent_manager: AgentManager, create_agent):
        send_data = self._fill_send_data()
        recv_data = self._cart_comm.alltoall(send_data)
        self._process_recv_data(recv_data, agent_manager, create_agent)

    def _agent_moving_rank(self, moving_agent: Agent, dest_rank: int, moved_agent_data: List,
                           agent_manager: AgentManager):
        self.remove(moving_agent)

    def _agents_moved_rank(self, moved_agents: List, agent_manager: AgentManager):
        pass

    def _post_agents_moved_rank(self, agent_manager: AgentManager, create_agent: Callable):
        pass
