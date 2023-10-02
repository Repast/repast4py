from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
import os
from time import time
import csv

from repast4py import core, random, space, schedule, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import ContinuousPoint as cpt


class Walker(core.Agent):

    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: cpt):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)
        self.pt = pt

    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.

        Returns:
            The saved state of this Walker.
        """
        return (self.uid, self.pt.coordinates)

    def walk(self, space):
        # choose two elements from the OFFSET array
        # to select the direction to walk in the
        # x and y dimensions
        xy_dirs = random.default_rng.choice(Walker.OFFSETS, size=2)
        self.pt = space.move(self, cpt(self.pt.x + xy_dirs[0], self.pt.y + xy_dirs[1], 0))

    # def count_colocations(self, grid, meet_log: MeetLog):
    #     # subtract self
    #     num_here = grid.get_num_agents(self.pt) - 1
    #     meet_log.total_meets += num_here
    #     if num_here < meet_log.min_meets:
    #         meet_log.min_meets = num_here
    #     if num_here > meet_log.max_meets:
    #         meet_log.max_meets = num_here
    #     self.meet_count += num_here


walker_cache = {}


def restore_walker(walker_data: Tuple):
    """
    Args:
        walker_data: tuple containing the data returned by Walker.save.
    """
    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
    uid = walker_data[0]
    pt_array = walker_data[1]
    pt = cpt(pt_array[0], pt_array[1], 0)

    if uid in walker_cache:
        walker = walker_cache[uid]
    else:
        walker = Walker(uid[0], uid[2], pt)
        walker_cache[uid] = walker

    walker.meet_count = walker_data[1]
    walker.pt = pt
    return walker


class Model:
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_repeating_event(1.1, 10, self.log_agents)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.cspace = space.SharedCSpace(name='space', bounds=box,
                                         borders=space.BorderType.Sticky,
                                         occupancy=space.OccupancyType.Multiple,
                                         tree_threshold=50,
                                         buffer_size=2, comm=comm)
        self.context.add_projection(self.cspace)

        self.rank = comm.Get_rank()
        rng = repast4py.random.default_rng
        walker_count = params['walker.count']
        for i in range(walker_count):
            # get a random x,y location in the space
            pt = self.cspace.get_random_local_pt(rng)
            # create and add the walker to the context
            walker = Walker(i, self.rank, pt)
            self.context.add(walker)
            self.cspace.move(walker, pt)

        self.pts = [self.cspace.get_random_local_pt(rng) for _ in range(120)]

        self.runtimes = [['tick', 'rank', 'n_agents', 'step_time']]
        ts = time() if self.rank == 0 else None
        self.ts = comm.bcast(ts, root=0)
        if self.rank == 0:
            print(self.ts, flush=True)
            os.mkdir(f'./output/{self.ts}')

    def step(self):
        time_start = time()
        max = len(self.pts)
        for walker in self.context.agents():
            idx = random.default_rng.integers(0, max)
            pt = self.pts[idx]
            self.cspace.move(walker, pt)

        time_end = time()
        time_duration = time_end - time_start
        tick = self.runner.schedule.tick
        self.runtimes.append([tick, self.rank, self.context.size()[-1], time_duration])

        self.context.synchronize(restore_walker)

    def log_agents(self):
        pass

    def at_end(self):
        with open(f'./output/{self.ts}/runtimes_{self.rank}.csv', 'w') as fout:
            writer = csv.writer(fout)
            writer.writerows(self.runtimes)

    def start(self):
        self.runner.execute()


def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
