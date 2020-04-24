from mpi4py import MPI

from ._space import Grid, DiscretePoint, ContinuousPoint, ContinuousSpace
from ._space import SharedGrid as _SharedGrid
from ._space import SharedContinuousSpace as _SharedContinuousSpace

from enum import Enum
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
        
        self.gather = self._gather_1d
        if bounds.yextent > 0:
            self.gather = self._gather_2d
        if bounds.zextent > 0:
            self.gather = self._gather_3d

    def _fill_send_data(self):
        send_data = [[] for i in range(self._cart_comm.size)]
        # bd: (rank, (ranges)) e.g., (1, (8, 10, 0, 0, 0, 0))
        for bd in self._get_buffer_data():
            data_list = send_data[bd[0]]
            ranges = bd[1]

            self.gather(data_list, ranges)

        return send_data

    def _process_recv_data(self, recv_data, create_agent):
        pt = DiscretePoint(0, 0, 0)
        for data_list in recv_data:
            for data in data_list:
                agent_data = data[0]
                pt_data = data[1]
                agent = create_agent(agent_data)
                self.buffered_agents.append(agent)
                self.add(agent)
                pt._reset(pt_data)
                self.move(agent, pt)


    def _clear_buffer(self):
        for agent in self.buffered_agents:
            self.remove(agent)
        self.buffered_agents.clear()

    def synchronize_buffer(self, create_agent):
        self._clear_buffer()
        send_data = self._fill_send_data()
        recv_data = self._cart_comm.alltoall(send_data)
        self._process_recv_data(recv_data, create_agent)
    
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
                        data_list.append((o.save(), (pt.x, pt.y, pt.z)))


class SharedCSpace(_SharedContinuousSpace):

    def __init__(self, name, bounds, borders, occupancy, buffersize, comm):
        super().__init__(name, bounds, borders, occupancy, buffersize, comm)
    
    def synchronize_buffer(self, create_agent):
        pass


