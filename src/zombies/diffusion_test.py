import time

from mpi4py import MPI

from repast4py import value_layer, space
from diffusion import Diffuser


def run():
    box = space.BoundingBox(0, 5, 0, 5, 0, 0)
    vl = value_layer.ReadWriteValueLayer(MPI.COMM_WORLD,  bounds=box, borders=space.BorderType.Sticky, 
            buffer_size=2, init_value=0)
    diffuser = Diffuser(vl)
    vl.read_layer.impl.grid[2, 2] = 1
    print(vl.read_layer.impl.grid)
    print(vl.write_layer.impl.grid)
    for _ in range(1):
        vl.apply(diffuser)
        # vl.synchronize_buffer()
        vl.swap_layers()

    print(vl.read_layer.impl.grid)
    print(vl.write_layer.impl.grid)

if __name__ == "__main__":
    start_time = time.time()
    run()
    end_time = time.time()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Runtime: {}'.format(end_time - start_time))