import time, sys

import torch

from mpi4py import MPI
from scipy import stats
import numpy as np

from repast4py import value_layer, space
from diffusion import Diffuser


def run(use_cuda):
    box = space.BoundingBox(0, 10016, 0, 10016, 0, 0)
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    vl = value_layer.ReadWriteValueLayer('vl', bounds=box, borders=space.BorderType.Sticky, 
            buffer_size=2, comm=MPI.COMM_WORLD, init_value=0, device=device)
    diffuser = Diffuser(vl)
    vl.read_layer.impl.grid[2, 2] = 1
    #print(vl.read_layer.impl.grid)
    #print(vl.write_layer.impl.grid)
    for _ in range(50):
        vl.apply(diffuser)
        # vl.synchronize_buffer()
        vl.swap_layers()

    #print(vl.read_layer.impl.grid)
    #print(vl.write_layer.impl.grid)

if __name__ == "__main__":
    for _ in range(21):
        start_time = time.time()
        run(sys.argv[1] == 'on')
        end_time = time.time()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(end_time - start_time)
