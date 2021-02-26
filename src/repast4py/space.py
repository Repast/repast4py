from typing import Callable, List

from ._space import DiscretePoint, ContinuousPoint
from ._space import SharedGrid as _SharedGrid
from ._space import SharedContinuousSpace as _SharedContinuousSpace
# not used in this file but imported into space namespace from _.
# for space API
from ._space import GridStickyBorders, GridPeriodicBorders
from ._space import CartesianTopology, Grid, ContinuousSpace

from .core import AgentManager, Agent

from collections import namedtuple
import sys


class BorderType:
    Sticky = 0
    Periodic = 1


class OccupancyType:
    Multiple = 0


if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    BoundingBox = namedtuple('BoundingBox', ['xmin', 'xextent', 'ymin', 'yextent', 'zmin', 'zextent'],
                             defaults=[0, 0])
else:
    BoundingBox = namedtuple('BoundingBox', ['xmin', 'xextent', 'ymin', 'yextent', 'zmin', 'zextent'])


class SharedGrid(_SharedGrid):

    def __init__(self, name, bounds, borders, occupancy, buffersize, comm):
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

    def __init__(self, name, bounds, borders, occupancy, buffersize, comm, tree_threshold):
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
