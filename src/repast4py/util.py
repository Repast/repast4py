import time
from mpi4py import MPI
import numpy as np

from typing import List

class Timer:

    def __init__(self):
        self.times = {}

    def start_timer(self, name):
        if not name in self.times:
            self.times[name] = [time.time(), 0, []]
        else:
            self.times[name][0] = time.time()

    def stop_timer(self, name):
        t = time.time()
        data = self.times[name]
        data[1] = t
        data[2].append(data[1] - data[0])

    def print_times(self):
        comm = MPI.COMM_WORLD
        all_timings = comm.gather(self.times)
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print('{:<6s}{:<16s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.
                format('rank', 'timer_name', 'sum','min', 'max', 'mean', 'std'))

            for r, timings in enumerate(all_timings):
                for k, v in timings.items():
                    mean = np.mean(v[2])
                    sm = np.sum(v[2])
                    mn = np.min(v[2])
                    mx = np.max(v[2])
                    std = np.std(v[2])
                    print('{:<6d}{:<16s}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}'.
                        format(r, k,
                        sm, mn, mx, mean, std))


def is_empty(lst: List) -> bool:
    for nl in lst:
        if len(nl) > 0:
            return False
    return True
