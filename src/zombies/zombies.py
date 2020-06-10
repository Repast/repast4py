from mpi4py import MPI
import sys
import configparser
import random, math
import numpy as np

import numba
from numba import int32

from repast4py import core, space, util, schedule

from repast4py.space import ContinuousPoint as CPt
from repast4py.space import DiscretePoint as DPt
from repast4py.space import BorderType, OccupancyType

timer = util.Timer()
model = None

def printf(msg):
    print(msg)
    sys.stdout.flush()


@numba.jit
def find_min_zombies(nghs, grid):
    minimum = [[], sys.maxsize]
    at = Dpt(0, 0, 0)
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

    return minimum[0][random.randint(0, len(minimum[0]) - 1)]

spec = [
    ('m', int32[:]),
    ('n', int32[:]),
    ('mo', int32[:]),
    ('no', int32[:]),
    ('xmin', int32),
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32),
]

@numba.jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.m = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        self.n = np.array([1, 1, 1, 0, 0, -1, -1, -1], dtype=np.int32)
        self.mo= np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y): # include_origin=False):
        #if include_origin:
        xs = self.mo + x
        ys = self.no + y
        # else:
        #     xs = self.m + x
        #     ys = self.n + y

        xd = (xs >= self.xmin) & (xs < self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys < self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys), axis=-1)


class Human(core.Agent):

    ID = 0

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Human.ID, rank=rank)
        self.infected = False
        self.infected_duration = 0

    def save(self):
        return (self.uid, self.infected, self.infected_duration)

    def restore(self, data):
        self.infected = data[1]
        self.infected_duration = data[2]

    def infect(self):
        self.infected = True

    def step(self):
        space_pt = model.space.get_location(self)
        alive = True
        if self.infected:
            self.infected_duration += 1
            if self.infected_duration == 10:
                model.remove_agent(self)
                model.add_zombie(space_pt)
                alive = False

        if alive:
            grid = model.grid
            pt = grid.get_location(self)
            timer.start_timer('ngh_finder')
            nghs = model.ngh_finder.find(pt.x, pt.y) # include_origin=True)
            timer.stop_timer('ngh_finder')

            timer.start_timer('zombie_finder')
            minimum = [[], sys.maxsize]
            at = DPt(0, 0, 0)
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

            min_ngh = minimum[0][random.randint(0, len(minimum[0]) - 1)]
            timer.stop_timer('zombie_finder')


            if not np.all(min_ngh == pt.coordinates):
                direction = (min_ngh - pt.coordinates[:2]) * 0.5
                timer.start_timer('human_move')
                model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])
                timer.stop_timer('human_move')


class Zombie(core.Agent):

    ID = 1

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Zombie.ID, rank=rank)

    def save(self):
        return (self.uid,)

    def restore(self):
        pass

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y) # include_origin=True)

        at = DPt(0, 0)
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

        max_ngh = maximum[0][random.randint(0, len(maximum[0]) - 1)]

        if not np.all(max_ngh == pt.coordinates):
            direction = (max_ngh - pt.coordinates[0:2]) * 0.25
            pt = model.space.get_location(self)
            timer.start_timer('zombie_move')
            model.move(self, pt.x + direction[0], pt.y + direction[1])
            timer.stop_timer('zombie_move')

        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == Human.ID:
                obj.infect()
                break

def create_agent(agent_data):
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank
    if uid[1] == Human.ID:
        h = Human(uid[0], uid[2])
        h.infected = agent_data[1]
        h.infected_duration = agent_data[2]
        return h
    else:
        return Zombie(uid[0], uid[2])


class Model:

    def __init__(self, comm, props):
        self.comm = comm
        self.context = core.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.SharedScheduleRunner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(float(props['stop.at']))

        box = space.BoundingBox(0, int(props['world.width']), 0, int(props['world.height']), 0, 0)

        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
            buffersize=2, comm=comm)
        self.context.add_projection(self.grid)

        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
            buffersize=2, comm=comm, tree_threshold=100)
        self.context.add_projection(self.space)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        local_bounds = self.space.get_local_bounds()

        human_count = int(props['human.count'])
        for i in range(human_count):
            h = Human(i, self.rank)
            self.context.add(h)
            x = random.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(h, x, y)

        zombie_count = int(props['zombie.count'])
        for i in range(zombie_count):
            zo = Zombie(i, self.rank)
            self.context.add(zo)
            x = random.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(zo, x, y)

        self.zombie_id = zombie_count

        self.calc_counts()


    def move(self, agent, x, y):
        timer.start_timer('space_move')
        self.space.move(agent, CPt(x, y))
        timer.stop_timer('space_move')
        timer.start_timer('grid_move')
        self.grid.move(agent, DPt(int(math.floor(x)), int(math.floor(y))))
        timer.stop_timer('grid_move')

    def step(self):
        # print("{}: {}".format(self.rank, len(self.context.local_agents)))
        tick =  self.runner.schedule.tick
        if tick % 10 == 0:
            hc, zc = self.calc_counts()
            if (self.rank == 0):
                printf("Tick: {}, Human Count: {}, Zombie Count: {}".format(tick, hc, zc))

        # self.grid.clear_buffer()
        # self.space.clear_buffer()
        self.context.synchronize(create_agent)
        # self.grid.synchronize_buffer(create_agent)
        # self.space.synchronize_buffer(create_agent)

        timer.start_timer('z_step')
        for z in self.context.agents(Zombie.ID):
            z.step()
        timer.stop_timer('z_step')

        timer.start_timer('h_step')
        for h in self.context.agents(Human.ID):
            h.step()
        timer.stop_timer('h_step')

    def run(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_zombie(self, pt):
        z = Zombie(self.zombie_id, self.rank)
        self.zombie_id += 1
        self.context.add(z)
        self.move(z, pt.x, pt.y)
        #print("Adding zombie at {}".format(pt))

    def calc_counts(self):
        human_count = np.zeros(1, dtype='int64')
        zombie_count = np.zeros(1, dtype='int64')
        counts = {Human.ID : 0, Zombie.ID : 0}
        self.context.get_counts(counts)
        self.comm.Reduce(np.array([counts[Human.ID]], dtype='int64'), human_count, op=MPI.SUM, root=0)
        self.comm.Reduce(np.array([counts[Zombie.ID]], dtype='int64'), zombie_count, op=MPI.SUM, root=0)

        return (human_count[0], zombie_count[0])


def run(config_file):

    with open(config_file, 'r') as f_in:
        lines = f_in.readlines()
        config_string = '[DEFAULT]\n{}'.format('\n'.join(lines))

    config = configparser.ConfigParser()
    config.read_string(config_string)
    props = config['DEFAULT']

    if 'random.seed' in props:
        random.seed(int(props['random.seed']))

    global model
    model = Model(MPI.COMM_WORLD, props)
    timer.start_timer('all')
    model.run()
    timer.stop_timer('all')
    timer.print_times()


if __name__ == "__main__":
    run(sys.argv[1])
