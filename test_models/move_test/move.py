from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt


import time


@dataclass
class MeetLog:
    total_meets: int = 0
    min_meets: int = 0
    max_meets: int = 0


class Walker(core.Agent):
    TYPE = 0

    def __init__(self, local_id: int, rank: int):
        super().__init__(id=local_id, type=Walker.TYPE, rank=rank)

    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.

        Returns:
            The saved state of this Walker.
        """
        return (self.uid,)


walker_cache = {}


def restore_walker(walker_data: Tuple):
    """
    Args:
        walker_data: tuple containing the data returned by Walker.save.
    """
    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
    uid = walker_data[0]
    if uid in walker_cache:
        walker = walker_cache[uid]
    else:
        walker = Walker(uid[0], uid[2])
        walker_cache[uid] = walker

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
        self.runner.schedule_stop(params['stop.at'])
        self.comm = comm

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        # create a bounding box equal to the size of the entire global world grid
        self.box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedCSpace(name='grid', bounds=self.box, borders=space.BorderType.Periodic,
                                       occupancy=space.OccupancyType.Multiple, buffer_size=2, tree_threshold=50,
                                       comm=comm)
        self.context.add_projection(self.grid)

        rank = comm.Get_rank()
        self.local_bounds = self.grid.get_local_bounds()
        self.rank = comm.Get_rank()

        for i in range(params['walker.count']):
            walker = Walker(i, rank)
            self.context.add(walker)
            pt = self.grid.get_random_local_pt(random.default_rng)
            self.grid.move(walker, pt)

    def step(self):
        for walker in self.context.agents():
            for _ in range(10):
                pt = self.get_random_global_pt()
                self.grid.move(walker, pt)

        self.context.synchronize(restore_walker)

        counts = self.comm.gather(self.context.size()[-1], root=0)
        if self.rank == 0:
            sum = 0
            tick = self.runner.tick()
            print(f'tick: {tick}', flush=True)
            for rank, count in enumerate(counts):
                print(f'\trank: {rank}, count: {count}', flush=True)
                sum += count
            print(f'sum: {sum}\n', flush=True)

    def contains(self, pt):
        return self.local_bounds.xmin <= pt.x and self.local_bounds.xmin + self.local_bounds.xextent > pt.y and \
            self.local_bounds.ymin <= pt.y and self.local_bounds.ymin + self.local_bounds.yextent > pt.y

    def get_random_global_pt(self):
        x = random.default_rng.uniform(self.box.xmin, self.box.xmin + self.box.xextent)
        y = random.default_rng.uniform(self.box.ymin, self.box.ymin + self.box.yextent)
        return space.ContinuousPoint(x, y, 0)

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
