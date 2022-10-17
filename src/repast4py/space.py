# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

from typing import Callable, List, Tuple
import mpi4py
import numpy as np

from ._core import Agent
from .core import AgentManager

from ._space import DiscretePoint, ContinuousPoint
from ._space import SharedGrid as _SharedGrid
from ._space import SharedContinuousSpace as _SharedContinuousSpace
# not used in this file but imported into space namespace from _.
# for space API
from ._space import GridStickyBorders, GridPeriodicBorders
from ._space import CartesianTopology, Grid, ContinuousSpace

from .geometry import BoundingBox, get_num_dims


class BorderType:
    """An enum defining the border types that can be used with a space or grid.

    The border type determines an agent's new location when the assigned location is beyond
    a grid or space's bounds. For example, during agent movement, when that movement carries the agent beyond the
    borders of a grid or space. Valid values are :attr:`Sticky`, and :attr:`Periodic`.
    """
    # NOTE: IF THESE CHANGE THE C++ GRID INIT AND SPACE INIT CODE NEEDS TO CHANGE TOO
    Sticky = 0
    """
    Clips any point coordinate to the maximum or minimum value when the coordinate is less than or
    greater than the grid or spaces maximum or minimum value. For example, if the minimum grid x location is 0, and
    an agent moves to an x of -1, then the new coordinate is 0.
    """
    Periodic = 1
    """
    Wraps point coordinates when the point coordinate is less than or
    greater than the grid or spaces maximum or minimum value. For example, if the minimum grid x location is 0,
    the maximum is 20, and an agent moves to an x of -2, then the new x coordinate is 19.
    """


class OccupancyType:
    """An enum defining the occupancy type of a location in a space or grid. The
    occupancy type determines how many agents are allowed at a single location.

    Valid values are: :attr:`Multiple`, :attr:`Single`.
    """
    # NOTE: IF THESE CHANGE THE C++ GRID INIT AND SPACE INIT CODE NEEDS TO CHANGE TOO
    Multiple = 0
    """Any number of agents can inhabit inhabit a location."""
    Single = 1
    """Only a single agent can inhabit a location."""


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
        buffer_size: the size of this SharedGrid buffered area. This single value is used for all dimensions.
        comm: the communicator containing all the ranks over which this SharedGrid is shared.
    """

    def __init__(self, name: str, bounds: BoundingBox, borders: BorderType, occupancy: OccupancyType, buffer_size: int,
                 comm: mpi4py.MPI.Intracomm):
        super().__init__(name, bounds, borders, occupancy, buffer_size, comm)
        self.buffered_agents = []
        self.rank = comm.Get_rank()

        self.gather = self._gather_1d
        if bounds.yextent > 0:
            self.gather = self._gather_2d
        if bounds.zextent > 0:
            self.gather = self._gather_3d

        local_bounds = self.get_local_bounds()
        self.random_pt = self._create_random_pt_func(local_bounds)

    def _create_random_pt_func(self, local_bounds):
        nd = get_num_dims(local_bounds)
        if nd == 1:
            def f(rng):
                x = rng.integers(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                return DiscretePoint(x, 0, 0)
            return f
        elif nd == 2:
            def f(rng):
                x = rng.integers(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                y = rng.integers(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
                return DiscretePoint(x, y, 0)
            return f
        else:
            def f(rng):
                x = rng.integers(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                y = rng.integers(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
                z = rng.integers(local_bounds.zmin, local_bounds.zmin + local_bounds.zextent)
                return DiscretePoint(x, y, z)
            return f

    def __contains__(self, agent):
        return self.contains(agent)

    def get_random_local_pt(self, rng: np.random.Generator) -> DiscretePoint:
        """Gets a random location within the local bounds of this SharedGrid.

        Args:
            rng: the random number generator to use to select the point.

        Returns:
            DiscretePoint: the random point
        """
        return self.random_pt(rng)

    def _fill_send_data(self):
        """Retrieves agents and locations from this SharedGrid for placement
        in a neighboring SharedGrid's buffer.

        This creates and returns a list of lists where the index of the nested list is the
        rank to send to. Each nested list is a list of tuples to send to the index rank.
        Each tuple consists of an agent and its location in this SharedGrid: (agent_data, (pt.x, pt.y, pt.z)

        Returns:
            List: a list of lists containing agent and location data.
        """
        send_data = [[] for i in range(self._cart_comm.size)]
        # bd: (rank, (ranges)) e.g., (1, (8, 10, 0, 0, 0, 0))
        for bd in self._get_buffer_data():
            data_list = send_data[bd[0]]
            ranges = bd[1]

            self.gather(data_list, ranges)

        return send_data

    def _process_recv_data(self, recv_data: List, agent_manager: AgentManager, create_agent: Callable):
        """Processes received agent data into this SharedGrid's buffer area.

        This iterates over the specified recv_data agent data and location list, and creates
        agents and places them in this SharedGrid using that data.

        Args:
            recv_data: a list of lists where the nested list contains tuples of agent and location data:
            (agent_data, (pt.x, pt.y, pt.z)
            agent_manager: the AgentManager used by this model to coordinate ghost agents.
            create_agent: a callable that takes the agent tuple and returns an Agent
        """
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

    def _synch_ghosts(self, agent_manager: AgentManager, create_agent: Callable):
        """Synchronizes the buffers across the ranks of this SharedGrid.

        Args:
            agent_manager: this rank's AgentManager
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        send_data = self._fill_send_data()
        recv_data = self._cart_comm.alltoall(send_data)
        self._process_recv_data(recv_data, agent_manager, create_agent)

    def _gather_1d(self, data_list: List, ranges: Tuple):
        """Gathers the serialized agent and location data for agents in the specified 1D range. The
        data is placed into the specified data_list.

        Args:
            data_list: a list into which the agent and location tuple (agent_data, (pt.x, pt.y, pt.z) is put.
            ranges: the ranges in which to get the gather the agents
        """
        pt = DiscretePoint(0, 0, 0)
        for x in range(ranges[0], ranges[1]):
            pt._reset1D(x)
            agents = self.get_agents(pt)
            for a in agents:
                data_list.append((a.save(), (pt.x, pt.y, pt.z)))

    def _gather_2d(self, data_list, ranges):
        """Gathers the serialized agent and location data for agents in the specified 2D range. The
        data is placed into the specified data_list.

        Args:
            data_list: a list into which the agent and location tuple (agent_data, (pt.x, pt.y, pt.z) is put.
            ranges: the ranges in which to get the gather the agents
        """
        pt = DiscretePoint(0, 0, 0)
        for x in range(ranges[0], ranges[1]):
            for y in range(ranges[2], ranges[3]):
                pt._reset2D(x, y)
                agents = self.get_agents(pt)
                for a in agents:
                    data_list.append((a.save(), (pt.x, pt.y, pt.z)))

    def _gather_3d(self, data_list, ranges):
        """Gathers the serialized agent and location data for agents in the specified 3D range. The
        data is placed into the specified data_list.

        Args:
            data_list: a list into which the agent and location tuple (agent_data, (pt.x, pt.y, pt.z) is put.
            ranges: the ranges in which to get the gather the agents
        """
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
    """An N-dimensional cartesian space where agents can occupy locations defined by
    a continuous floating point coordinate.

    The space is shared over all the ranks in the specified communicator by sub-dividing the global bounds into
    some number of smaller spaces, one for each rank. For example, given a global 2D space size of 100 x 25 and
    2 ranks, the global space will be split along the x dimension such that the SharedCSpace in the first
    MPI rank covers 0-50 x 0-25 and the second rank 50-100 x 0-25.

    Each rank's SharedCSpace contains a buffer of a specified size that duplicates or "ghosts" an adjacent
    area of the neighboring rank's SharedCSpace. In the above example, the rank 1 space buffers the area from
    50-52 x 0-25 in rank 2, and rank 2 buffers 48-50 x 0-25 in rank 1. **Be sure to specify a buffer size appropriate
    to any agent behavior**. For example, if an agent can "see" 3 units away and take some action based on what it
    perceives, then the buffer size should be at least 3, insuring that an agent can properly see beyond the borders of
    its own local SharedCSpace. When an agent moves beyond the borders of its current SharedCSpace,
    it will be transferred
    from its current rank, and into that containing the section of the global space that it has moved into.
    The SharedCSpace uses a `tree <https://en.wikipedia.org/wiki/Quadtree>`_ (quad or oct depending on the number of
    dimensions) to optimize spatial queries. The tree can be tuned using the tree threshold parameter.

    Args:
       name: the name of the space.
       bounds: the global dimensions of the space.
       borders: the border semantics - :attr:`BorderType.Sticky` or :attr:`BorderType.Periodic`.
       occupancy: the type of occupancy in each cell - :attr:`OccupancyType.Single` or :attr:`OccupancyType.Multiple`.
       buffer_size: the size of this SharedCSpace's buffered area. This single value is used for all dimensions.
       comm: the communicator containing all the ranks over which this SharedCSpace is shared.
       tree_threshold: the space's tree cell maximum capacity. When this capacity is reached, the cell splits.

    """

    def __init__(self, name: str, bounds: BoundingBox, borders: BorderType, occupancy: OccupancyType,
                 buffer_size: int, comm: mpi4py.MPI.Intracomm, tree_threshold: int):
        super().__init__(name, bounds, borders, occupancy, buffer_size, comm, tree_threshold)
        self.buffered_agents = []

        local_bounds = self.get_local_bounds()
        self.random_pt = self._create_random_pt_func(local_bounds)

    def _create_random_pt_func(self, local_bounds):
        nd = get_num_dims(local_bounds)
        if nd == 1:
            def f(rng):
                x = rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                return ContinuousPoint(x, 0, 0)
            return f
        elif nd == 2:
            def f(rng):
                x = rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                y = rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
                return ContinuousPoint(x, y, 0)
            return f
        else:
            def f(rng):
                x = rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                y = rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
                z = rng.uniform(local_bounds.zmin, local_bounds.zmin + local_bounds.zextent)
                return ContinuousPoint(x, y, z)
            return f

    def __contains__(self, agent):
        return self.contains(agent)

    def get_random_local_pt(self, rng: np.random.Generator) -> ContinuousPoint:
        """Gets a random location within the local bounds of this SharedCSpace.

        Args:
            rng: the random number generator to use to select the point.

        Returns:
            ContinuousPoint: the random point
        """
        return self.random_pt(rng)

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
        """Retrieves agents and locations from this SharedGrid for placement
        in a neighboring SharedCSpace's buffer.

        This creates and returns a list of lists where the index of the nested list is the
        rank to send to. Each nested list is a list of tuples to send to the index rank.
        Each tuple consists of an agent and its location in this SharedCSpace: (agent_data, (pt.x, pt.y, pt.z)

        Returns:
            List: a list of lists containing agent and location data.
        """
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
        """Processes received agent data into this SharedCSpace's buffer area.

        This iterates over the specified recv_data agent data and location list, and creates
        agents and places them in this SharedCSpace using that data.

        Args:
            recv_data: a list of lists where the nested list contains tuples of agent and location data:
            (agent_data, (pt.x, pt.y, pt.z)
            agent_manager: the AgentManager used by this model to coordinate ghost agents.
            create_agent: a callable that takes the agent tuple and returns an Agent
        """
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
        """Synchronizes the buffers across the ranks of this SharedCSpace.

        Args:
            agent_manager: this rank's AgentManager
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
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
