import sys
import math
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

model = None


@numba.jit(nopython=True)
def find_min_zombies(nghs, grid):
    """Given """
    minimum = [[], sys.maxsize]
    at = dpt(0, 0, 0)
    for ngh in nghs:
        at._reset_from_array(ngh)
        count = 0
        for obj in grid.get_agents(at):
            if obj.id[2] == Zombie.ID:
                count += 1
        if count < minimum[1]:
            minimum[0] = [ngh]
            minimum[1] = count
        elif count == minimum[1]:
            minimum[0].append(ngh)

    return minimum[0][random.default_rng.integers(0, len(minimum[0]))]


@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]


spec = [
    ('m', int32[:]),
    ('n', int32[:]),
    ('mo', int32[:]),
    ('no', int32[:]),
    ('xmin', int32),
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]


@jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.m = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        self.n = np.array([1, 1, 1, 0, 0, -1, -1, -1], dtype=np.int32)
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        # self.zs = np.zeros(9, dtype=np.int32)

    def find(self, x, y):  # include_origin=False):
        # if include_origin:
        xs = self.mo + x
        ys = self.no + y
        # else:
        # xs = self.m + x
        # ys = self.n + y

        xd = (xs >= self.xmin) & (xs < self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys < self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)


class Human(core.Agent):
    """The Human Agent

    Args:
        a_id: a integer that uniquely identifies this Human on its starting rank
        rank: the starting MPI rank of this Human.
    """

    ID = 0

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Human.ID, rank=rank)
        self.infected = False
        self.infected_duration = 0

    def save(self) -> Tuple:
        """Saves the state of this Human as a Tuple.

        Used to move this Human from one MPI rank to another.

        Returns:
            The saved state of this Human.
        """
        return (self.uid, self.infected, self.infected_duration)

    def infect(self):
        self.infected = True

    # @profile
    def step(self):
        space_pt = model.space.get_location(self)
        alive = True
        if self.infected:
            self.infected_duration += 1
            alive = self.infected_duration < 10

        if alive:
            grid = model.grid
            pt = grid.get_location(self)
            nghs = model.ngh_finder.find(pt.x, pt.y)  # include_origin=True)
            # timer.stop_timer('ngh_finder')

            # timer.start_timer('zombie_finder')
            minimum = [[], sys.maxsize]
            at = dpt(0, 0, 0)
            for ngh in nghs:
                at._reset_from_array(ngh)
                count = 0
                for obj in grid.get_agents(at):
                    if obj.uid[1] == Zombie.ID:
                        count += 1
                if count < minimum[1]:
                    minimum[0] = [ngh]
                    minimum[1] = count
                elif count == minimum[1]:
                    minimum[0].append(ngh)

            min_ngh = minimum[0][random.default_rng.integers(0, len(minimum[0]))]
            # timer.stop_timer('zombie_finder')

            # if not np.all(min_ngh == pt.coordinates):
            # if min_ngh[0] != pt.coordinates[0] or min_ngh[1] != pt.coordinates[1]:
            # if not np.array_equal(min_ngh, pt.coordinates):
            if not is_equal(min_ngh, pt.coordinates):
                direction = (min_ngh - pt.coordinates) * 0.5
                model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])

        return (not alive, space_pt)


class Zombie(core.Agent):

    ID = 1

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Zombie.ID, rank=rank)

    def save(self):
        return (self.uid,)

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)  # include_origin=True)

        at = dpt(0, 0)
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == Human.ID:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)

        if len(maximum[0]) > 0:
            max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

            if not np.all(max_ngh == pt.coordinates):
                direction = (max_ngh - pt.coordinates[0:3]) * 0.25
                pt = model.space.get_location(self)
                # timer.start_timer('zombie_move')
                model.move(self, pt.x + direction[0], pt.y + direction[1])
                # timer.stop_timer('zombie_move')

        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == Human.ID:
                obj.infect()
                break


agent_cache = {}


def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank
    if uid[1] == Human.ID:
        if uid in agent_cache:
            h = agent_cache[uid]
        else:
            h = Human(uid[0], uid[2])
            agent_cache[uid] = h

        # restore the agent state from the agent_data tuple
        h.infected = agent_data[1]
        h.infected_duration = agent_data[2]
        return h
    else:
        # note that the zombie has no internal state
        # so there's nothing to restore other than
        # the Zombie itself
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            z = Zombie(uid[0], uid[2])
            agent_cache[uid] = z
            return z


@dataclass
class Counts:
    """Dataclass used by repast4py aggregate logging to record
    the number of Humans and Zombies after each tick.
    """
    humans: int = 0
    zombies: int = 0


class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)
        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)
        self.context.add_projection(self.space)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.counts = Counts()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD, params['counts_file'])

        local_bounds = self.space.get_local_bounds()
        world_size = comm.Get_size()

        total_human_count = params['human.count']
        pp_human_count = int(total_human_count / world_size)
        if self.rank < total_human_count % world_size:
            pp_human_count += 1

        for i in range(pp_human_count):
            h = Human(i, self.rank)
            self.context.add(h)
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(h, x, y)

        total_zombie_count = params['zombie.count']
        pp_zombie_count = int(total_zombie_count / world_size)
        if self.rank < total_zombie_count % world_size:
            pp_zombie_count += 1

        for i in range(pp_zombie_count):
            zo = Zombie(i, self.rank)
            self.context.add(zo)
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(zo, x, y)

        self.zombie_id = pp_zombie_count

    def at_end(self):
        self.data_set.close()

    def move(self, agent, x, y):
        # timer.start_timer('space_move')
        self.space.move(agent, cpt(x, y))
        # timer.stop_timer('space_move')
        # timer.start_timer('grid_move')
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))
        # timer.stop_timer('grid_move')

    def step(self):
        # print("{}: {}".format(self.rank, len(self.context.local_agents)))
        tick = self.runner.schedule.tick
        self.log_counts(tick)
        self.context.synchronize(restore_agent)

        # timer.start_timer('z_step')
        for z in self.context.agents(Zombie.ID):
            z.step()
        # timer.stop_timer('z_step')

        # timer.start_timer('h_step')
        dead_humans = []
        for h in self.context.agents(Human.ID):
            dead, pt = h.step()
            if dead:
                dead_humans.append((h, pt))

        for h, pt in dead_humans:
            model.remove_agent(h)
            model.add_zombie(pt)

        # timer.stop_timer('h_step')

    def run(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_zombie(self, pt):
        z = Zombie(self.zombie_id, self.rank)
        self.zombie_id += 1
        self.context.add(z)
        self.move(z, pt.x, pt.y)
        # print("Adding zombie at {}".format(pt))

    def log_counts(self, tick):
        # Get the current number of zombies and humans and log
        counts = self.context.size([Human.ID, Zombie.ID])
        self.counts.humans = counts[Human.ID]
        self.counts.zombies = counts[Zombie.ID]
        self.data_set.log(tick)

        # Do the cross-rank reduction manually and print the result
        if tick % 10 == 0:
            human_count = np.zeros(1, dtype='int64')
            zombie_count = np.zeros(1, dtype='int64')
            self.comm.Reduce(np.array([self.counts.humans], dtype='int64'), human_count, op=MPI.SUM, root=0)
            self.comm.Reduce(np.array([self.counts.zombies], dtype='int64'), zombie_count, op=MPI.SUM, root=0)
            if (self.rank == 0):
                print("Tick: {}, Human Count: {}, Zombie Count: {}".format(tick, human_count[0], zombie_count[0]),
                      flush=True)


def run(params: Dict):
    """Creates and runs the Zombies Model.

    Args:
        params: the model input parameters
    """
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
