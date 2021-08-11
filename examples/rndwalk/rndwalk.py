from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py.util import create_args_parser, parse_params
from repast4py import core, space, schedule, logging
from repast4py import context as ctx
from repast4py.random import default_rng as rng
from repast4py.space import DiscretePoint as dpt


@dataclass
class MeetLog:
    total_meets: int = 0
    min_meets: int = 0
    max_meets: int = 0


class Walker(core.Agent):

    ID = 0
    OFFSETS = np.array([-1, 0, 1])

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=Walker.ID, rank=rank)
        self.pt = pt
        self.meet_count = 0

    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.

        Returns:
            The saved state of this Walker.
        """
        return (self.uid, self.meet_count, self.pt.coordinates)

    def walk(self, grid, meet_log: MeetLog):
        # choose two elements from the OFFSET array
        # to select the direction to walk in the
        # x and y dimensions
        xy_dirs = rng.choice(Walker.OFFSETS, size=2)
        self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0))
        # subtract self
        num_here = grid.get_num_agents(self.pt) - 1
        meet_log.total_meets += num_here
        if num_here < meet_log.min_meets:
            meet_log.min_meets = num_here
        if num_here > meet_log.max_meets:
            meet_log.max_meets = num_here
        self.meet_count += num_here


walker_cache = {}


def restore_walker(walker_data: Tuple):
    """
    Args:
        walker_data: tuple containing the data returned by Walker.save.
    """
    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
    uid = walker_data[0]
    pt_array = walker_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)

    if uid in walker_cache:
        walker = walker_cache[uid]
    else:
        walker = Walker(uid[0], uid[2], pt)
        walker_cache[uid] = walker

    walker.meet_count = walker_data[1]
    pt = walker_data[2]
    walker.pt = dpt(pt[0], pt[1], 0)
    return walker


class Model:
    """
    The Model class encapsulates the simulation, and is 
    responsible for intialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        # create the schedule
        self.runner = schedule.SharedScheduleRunner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_repeating_event(1.1, 1, self.log_agents)
        self.runner.schedule_stop(float(params['stop.at']))
        self.runner.schedule_end_event(self.at_end)

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky, 
                                     occupancy=space.OccupancyType.Multiple, buffersize=2, comm=comm)
        self.context.add_projection(self.grid)

        rank = comm.Get_rank()
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            pt = self.grid.get_random_local_pt(rng)
            # create and add the walker to the context
            walker = Walker(i, rank, pt)
            self.context.add(walker)
            self.grid.move(walker, pt)

        # initialize the logging
        self.meet_log = MeetLog()
        loggers = logging.create_loggers(self.meet_log, op=MPI.SUM, names={'total_meets': 'total'}, rank=rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.MIN, names={'min_meets': 'min'}, rank=rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.MAX, names={'max_meets': 'max'}, rank=rank)
        self.data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD, params['meet_log_file'])

        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'], ['tick', 'agent_id', 'meet_count'])

    def step(self):
        for walker in self.context.agents():
            walker.walk(self.grid, self.meet_log)
        self.context.synchronize(restore_walker)

        tick = self.runner.schedule.tick
        self.data_set.log(tick)
        # clear the meet log counts for the next tick
        self.meet_log.max_meets = self.meet_log.min_meets = self.meet_log.total_meets = 0

    def log_agents(self):
        tick = self.runner.schedule.tick
        for walker in self.context.agents():
            id_str = f'{walker.id}_{walker.uid_rank}'
            self.agent_logger.log_row(tick, id_str, walker.meet_count)

        self.agent_logger.write()

    def at_end(self):
        self.data_set.close()
        self.agent_logger.close()

    def run(self):
        self.runner.execute()


def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = parse_params(args.parameters_file, args.parameters)
    run(params)
